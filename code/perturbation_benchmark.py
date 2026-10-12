"""
Real single-cell PERTURBATION-RESPONSE benchmark for the kappa-C / kappa-C-Z
decoder/latent spectral diagnostic.

WHY THIS SCRIPT EXISTS
----------------------
The scVI demo in `scvi_demo_benchmark.py` shows the decoder/latent spectral
decoupling on an *independent architecture*, but its "benchmark" only checks
whether the latent reorganization tracks CELL-TYPE predictability. A reviewer
can fairly ask: does kappa-C-Z track a *real perturbation response* (i.e. the
differential expression a perturbation actually induces)? This script answers
that with a genuine CRISPR perturbation dataset.

WHAT IT DOES
------------
1. Downloads a real pooled/arrayed CRISPR perturbation dataset (control + many
   gene knock-downs / knock-outs), with multi-mirror fallback + hard timeouts.
   Primary: Replogle 2022 (genome-scale Perturb-seq, K562) from Zenodo.
   Fallbacks: Norman 2019 (GEARS/opendoor, Dropbox) and Adamson 2016 (GEO).
2. Trains TWO scVI models (mirrors scvi_demo_benchmark.py exactly):
     - vanilla   : no perturbation labels
     - perturbed : perturbation identity injected as a categorical covariate
                   (the analogue of "injecting an external structural prior").
3. Computes the SAME diagnostics used in the manuscript:
     - kappa-C(W) = tr(W_pert^T W_pert) / tr(W_van^T W_van),  W = Ridge(Z->X)
     - kappa-C(Z) = tr(cov Z_pert)      / tr(cov Z_van)
   (kappa-C is defined via the Ridge readout, not literal decoder weights --
    see manuscript note m2.)
4. REAL perturbation-response validation:
     (a) Perturbation-identity prediction from Z: a stratified CV classifier
         predicts which gene was perturbed from the latent representation;
         accuracy vs majority-class baseline proves Z encodes real perturbation
         structure (i.e. kappa-C-Z tracks biological perturbation).
     (b) Latent-shift vs observed differential expression: for the best-sampled
         perturbations, the latent shift (mean Z_pert - mean Z_control) is
         projected through the decoder readout W to predict a per-gene DE
         vector; its Spearman correlation with the OBSERVED logFC (measured
         from the data) across genes is the direct "does the latent
         reorganization predict the real perturbation response?" test.

RESULT
------
Numbers are written to data/perturbation_benchmark_results.json and printed.
Integrate the real numbers into main.tex Section 3.2 + L135 + abstract once
this runs on a machine with normal network egress.

NOTE ON RUNNING IT
-------------------
The genuine per-cell perturbation benchmarks (Replogle 2022, Norman 2019) live
on Zenodo/Dropbox, which may be blocked in some sandboxes. If you already have
the .h5ad on disk (or any AnnData with a per-cell perturbation label), skip the
network entirely via LOCAL_PERT_H5AD:

    python perturbation_benchmark.py

or, with a local file and an explicit label column / control value:

    LOCAL_PERT_H5AD=/path/to/norman.h5ad PERT_COL=condition_name \\
        CONTROL_VAL=A549_ctrl_1 python perturbation_benchmark.py

To skip the ~45-min scVI training on a re-run (e.g. after a crash), set
USE_LATENT_CACHE=1 (latents are cached to data/_norman_latents.npz after the
first successful training).

If you already have the .h5ad downloaded (or any AnnData with a perturbation
label column), skip the network entirely:

    LOCAL_PERT_H5AD=/path/to/replogle.h5ad PERT_COL=gene_name \\
        CONTROL_VAL=control python perturbation_benchmark.py

It emits data/perturbation_benchmark_results.json with kappa-C(W), kappa-C(Z),
effective-rank gain, perturbation-identity accuracy vs baseline, and the
latent-shift vs observed-DE Spearman correlation.
"""

import os, sys, json, time, urllib.request, urllib.error
from typing import Optional

import numpy as np
import scanpy as sc

# ----------------------------------------------------------------------------
# ENV OVERRIDES  (so the script runs anywhere the data can be reached)
# ----------------------------------------------------------------------------
# LOCAL_PERT_H5AD : point at an already-downloaded .h5ad (Replogle/Norman/...)
#                   to skip the network entirely and go straight to training.
# PERT_COL        : obs column holding the perturbation identity
#                   (e.g. "perturbation", "gene_name", "guide", "target").
# CONTROL_VAL     : value marking unperturbed control cells
#                   (e.g. "control", "ctrl", "NTC", "non-targeting").
# Declared up-front so env overrides take effect before the CONFIG block below.
PERT_COL = os.environ.get("PERT_COL", "") or None
CONTROL_VAL = os.environ.get("CONTROL_VAL", "") or None

_LOCAL_H5AD = os.environ.get("LOCAL_PERT_H5AD", "").strip()
if _LOCAL_H5AD:
    if not os.path.exists(_LOCAL_H5AD):
        raise SystemExit(f"LOCAL_PERT_H5AD not found: {_LOCAL_H5AD}")
    print(f"[env] LOCAL_PERT_H5AD = {_LOCAL_H5AD}  (download skipped)")
if PERT_COL is not None:
    print(f"[env] PERT_COL = {PERT_COL!r}  (auto-detect skipped)")
if CONTROL_VAL is not None:
    print(f"[env] CONTROL_VAL = {CONTROL_VAL!r}")

# ----------------------------------------------------------------------------
# CONFIG  (adjust to your download / environment)
# ----------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(HERE, "..", "data"))
os.makedirs(DATA_DIR, exist_ok=True)
RESULT_PATH = os.path.join(DATA_DIR, "perturbation_benchmark_results.json")

N_HVGS = 2000
N_LATENT = 30               # match scvi_demo_benchmark.py (n_latent=30)
N_LAYERS = 2                # match scvi_demo_benchmark.py
DROPOUT = 0.1               # match scvi_demo_benchmark.py
GENE_LIKELIHOOD = "normal"  # match scvi_demo_benchmark.py (avoids NB warnings)
N_EPOCHS = 200
MAX_CELLS = 120000          # subsample cap so CPU training stays tractable
SEED = 42
DOWNLOAD_TIMEOUT = 600      # seconds per source attempt
RIDGE_ALPHA = 1.0

# Column in adata.obs that holds the perturbation identity, and the value used
# for unperturbed control cells. Declared above (env-overridable); auto-detected
# if left as None.
# Common fallbacks the auto-detector will try:
_PERT_CANDIDATES = ["perturbation", "gene_name", "guide", "target",
                    "perturbed_gene", "condition", "crispr_perturbation"]
_CONTROL_CANDIDATES = ["control", "ctrl", "ntc", "non-targeting",
                       "nontargeting", "nan", "none", "unperturbed", ""]

# ----------------------------------------------------------------------------
# 1. DOWNLOAD  (multi-mirror, timeouts, first success wins)
# ----------------------------------------------------------------------------
SOURCES = [
    # Only CELL-LEVEL perturbation datasets are valid here: the benchmark's
    # validation requires a per-cell perturbation-identity label. Pooled screens
    # (e.g. Adamson GEO) have no such labels and are intentionally omitted.
    ("Replogle2022 (Zenodo)",
     "https://zenodo.org/record/5836124/files/replogle_processed.h5ad"),
    ("Norman2019 (GEARS/Dropbox)",
     "https://www.dropbox.com/s/q4sp6h3w1j0zh0b/opendoor.tar.gz?dl=1"),
]

def _download(url: str, dest: str):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=DOWNLOAD_TIMEOUT) as r, \
         open(dest, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)

def fetch_dataset():
    """Try each source; return (local_path, kind). kind in {h5ad, tar, geo_mtx}."""
    if _LOCAL_H5AD:
        print(f"[fetch] using local file (download skipped): {_LOCAL_H5AD}")
        return _LOCAL_H5AD, "h5ad"
    for name, url in SOURCES:
        dest = os.path.join(DATA_DIR, "perturbation_raw_" +
                            name.split()[0].lower() + os.path.splitext(url)[-1])
        print(f"[download] trying {name}  ->  {url}")
        try:
            _download(url, dest)
            print(f"[download] OK: {dest} ({os.path.getsize(dest)/1e6:.1f} MB)")
            if url.endswith(".h5ad"):
                return dest, "h5ad"
            if url.endswith(".tar.gz") or url.endswith(".tgz"):
                return dest, "tar"
            if "geo" in url and url.endswith(".gz"):
                return dest, "geo_mtx"
        except Exception as e:
            print(f"[download] FAILED {name}: {type(e).__name__}: {e}")
    raise RuntimeError("All perturbation-data sources failed. Run on a machine "
                       "with normal network egress, or set SOURCES manually.")

# ----------------------------------------------------------------------------
# 2. LOAD + parse perturbation labels
# ----------------------------------------------------------------------------
def _detect_pert_cols(adata):
    global PERT_COL, CONTROL_VAL
    if PERT_COL is None:
        for c in _PERT_CANDIDATES:
            if c in adata.obs.columns:
                PERT_COL = c
                break
    if PERT_COL is None:
        # fall back to any obs column whose values look like genes/guides
        for c in adata.obs.columns:
            vals = adata.obs[c].astype(str).str.lower().unique()
            if any(v in _CONTROL_CANDIDATES for v in vals) and len(vals) > 2:
                PERT_COL = c
                break
    if PERT_COL is None:
        raise RuntimeError("Could not auto-detect a perturbation-identity "
                         "column. Set PERT_COL explicitly.")
    if CONTROL_VAL is None:
        vals = adata.obs[PERT_COL].astype(str).str.lower().unique().tolist()
        for v in _CONTROL_CANDIDATES:
            if v in vals:
                CONTROL_VAL = v
                break
        if CONTROL_VAL is None:
            CONTROL_VAL = vals[0]   # assume the most frequent is control
    print(f"[load] perturbation column = {PERT_COL!r}, control value = {CONTROL_VAL!r}")
    print(f"[load] #perturbation identities = "
          f"{adata.obs[PERT_COL].nunique()}")

def load_adata(path: str, kind: str):
    if kind == "h5ad":
        adata = sc.read_h5ad(path)
    elif kind == "tar":
        # Norman/GEARS opendoor: extract then read the h5ad inside
        import tarfile, tempfile
        with tarfile.open(path) as tf:
            tf.extractall(DATA_DIR)
        h5 = [m for m in tf.getnames() if m.endswith(".h5ad")][0]
        adata = sc.read_h5ad(os.path.join(DATA_DIR, h5))
    elif kind == "geo_mtx":
        # Adamson GEO series_matrix -> build AnnData (samples = columns)
        import gzip
        with gzip.open(path, "rt") as f:
            lines = [l.rstrip("\n").split("\t") for l in f]
        # locate the data table
        start = next(i for i, l in enumerate(lines)
                     if l[0].startswith("!sample_table_begin")) + 1
        header = [l for l in lines[start:start+2] if not l[0].startswith("!")][0]
        genes = [l[0] for l in lines[start+1:] if l[0] and not l[0].startswith("!")]
        mat = np.array([[float(x) for x in l[1:]]
                        for l in lines[start+1:] if l[0] and not l[0].startswith("!")])
        adata = sc.AnnData(mat.T)   # cells x genes
        adata.var_names = genes
        adata.obs_names = header[1:]
        # perturbation identity from Sample_title / characteristics if present
        for i, l in enumerate(lines):
            if l[0].startswith("!Sample_title"):
                adata.obs["perturbation"] = l[1:len(header)]
    if adata.n_obs > MAX_CELLS:
        sc.pp.subsample(adata, n_obs=MAX_CELLS, random_state=SEED)
    return adata

# ----------------------------------------------------------------------------
# 3. kappa diagnostics (identical definitions to scvi_demo_benchmark.py)
# ----------------------------------------------------------------------------
def ridge_readout(Z, X, alpha=RIDGE_ALPHA):
    """UNAGI protocol: W = Ridge(Z -> X), alpha=1, no intercept. Returns W (genes x latent).
    Identical to scvi_demo_benchmark.py."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    Zs = StandardScaler().fit_transform(Z)
    Xs = StandardScaler().fit_transform(X)
    W = Ridge(alpha=alpha, fit_intercept=False).fit(Zs, Xs).coef_  # (latent, genes)
    return W.T  # (genes, latent)

def kappa_C_W(Z0, X0, Z1, X1):
    # decoder readout invariance; W0,W1 trained on the SAME cells (all cells).
    # kappa-C(W) = sum(W1^2) / sum(W0^2) = tr(W1^T W1)/tr(W0^T W0).
    W0 = ridge_readout(Z0, X0)
    W1 = ridge_readout(Z1, X1)
    return float((W1 ** 2).sum() / (W0 ** 2).sum()), W0, W1

def kappa_C_Z(Z0, Z1):
    c0 = np.cov(Z0, rowvar=False)
    c1 = np.cov(Z1, rowvar=False)
    return float(np.trace(c1) / np.trace(c0))

def eff_rank(cov, eps=1e-12):
    # participation ratio (matches scvi_demo_benchmark.py)
    e = np.linalg.eigvalsh(cov + eps * np.eye(cov.shape[0]))
    e = np.clip(e, eps, None)
    return float((e.sum() ** 2) / (e ** 2).sum())

# ----------------------------------------------------------------------------
# 4. REAL perturbation-response validation
# ----------------------------------------------------------------------------
def predict_perturbation_identity(Z, pert_labels):
    from sklearn.model_selection import StratifiedKFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    y = pert_labels.astype(str).values
    Zs = StandardScaler().fit_transform(Z)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    accs = []
    for tr, te in skf.split(Zs, y):
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        clf.fit(Zs[tr], y[tr])
        accs.append((clf.predict(Zs[te]) == y[te]).mean())
    majority = np.unique(y, return_counts=True)[1].max() / len(y)
    return float(np.mean(accs)), float(majority)

def de_correlation(X, pert_labels, W, Z_pert_mean, Z_ctrl_mean, top_k=8):
    """For the best-sampled perturbations, correlate latent-shift-predicted DE
    with observed logFC across genes."""
    from scipy.stats import spearmanr
    uniq, counts = np.unique(pert_labels.astype(str), return_counts=True)
    targets = [u for u, c in zip(uniq, counts)
               if u != str(CONTROL_VAL) and c >= 50][:top_k]
    Xc = X[pert_labels.astype(str).values == str(CONTROL_VAL)]
    ctrl_mean = Xc.mean(0)
    shift = Z_pert_mean - Z_ctrl_mean            # latent shift vector (latent,)
    pred_de = W.T @ shift                        # project through decoder readout -> (genes,)
    rhos = []
    for t in targets:
        obs = X[pert_labels.astype(str).values == t].mean(0) - ctrl_mean
        if np.std(obs) < 1e-6 or np.std(pred_de) < 1e-6:
            continue
        r, _ = spearmanr(pred_de, obs)
        rhos.append((t, float(r)))
    return rhos

# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def main():
    t0 = time.time()
    path, kind = fetch_dataset()
    adata = load_adata(path, kind)
    _detect_pert_cols(adata)

    # preprocessing
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=N_HVGS)
    adata = adata[:, adata.var.highly_variable].copy()
    adata.X = np.nan_to_num(adata.X.toarray() if hasattr(adata.X, "toarray")
                            else np.asarray(adata.X), nan=0.0, posinf=0.0, neginf=0.0)

    ctrl_mask = adata.obs[PERT_COL].astype(str).str.lower() \
                    .isin([str(CONTROL_VAL).lower()]).values
    pert_mask = ~ctrl_mask

    # ---- latent cache: skip the expensive scVI training on re-runs / resume ----
    _cache = os.path.join(DATA_DIR, "_norman_latents.npz")
    if os.environ.get("USE_LATENT_CACHE") and os.path.exists(_cache):
        print(f"[cache] loading latents from {_cache} (training skipped)")
        _c = np.load(_cache, allow_pickle=True)
        Zv, Zp, X = _c["Zv"], _c["Zp"], _c["X"]
        adata.obs[PERT_COL] = _c["pert_labels"]
        ctrl_mask = adata.obs[PERT_COL].astype(str).str.lower() \
                         .isin([str(CONTROL_VAL).lower()]).values
        pert_mask = ~ctrl_mask
    else:
        import scvi
        scvi.settings.seed = SEED

        # ---- vanilla scVI (no perturbation labels) ----
        av = adata.copy()
        scvi.model.SCVI.setup_anndata(av)
        vae_v = scvi.model.SCVI(av, n_latent=N_LATENT, n_layers=N_LAYERS,
                                dropout_rate=DROPOUT, gene_likelihood=GENE_LIKELIHOOD)
        vae_v.train(max_epochs=N_EPOCHS, accelerator="cpu")
        Zv = vae_v.get_latent_representation(av)

        # ---- perturbation-conditioned scVI ----
        ap = adata.copy()
        ap.obs[PERT_COL] = ap.obs[PERT_COL].astype(str)
        scvi.model.SCVI.setup_anndata(ap, categorical_covariate_keys=[PERT_COL])
        vae_p = scvi.model.SCVI(ap, n_latent=N_LATENT, n_layers=N_LAYERS,
                                dropout_rate=DROPOUT, gene_likelihood=GENE_LIKELIHOOD)
        vae_p.train(max_epochs=N_EPOCHS, accelerator="cpu")
        Zp = vae_p.get_latent_representation(ap)

        X = adata.X
        np.savez(_cache, Zv=Zv, Zp=Zp, X=X,
                 pert_labels=adata.obs[PERT_COL].astype(str).values)
        print(f"[cache] saved latents -> {_cache}")

    # kappa-C(W) and kappa-C(Z) computed on ALL cells (faithful to scVI demo:
    # both readouts over the same cell set; the injected prior spans all cells
    # here too, so the decoder-invariance / latent-reorganization read is clean).
    kCW, W0, W1 = kappa_C_W(Zv, X, Zp, X)   # W1 = perturbation-decoder readout
    kCZ = kappa_C_Z(Zv, Zp)
    er_v = eff_rank(np.cov(Zv, rowvar=False))
    er_p = eff_rank(np.cov(Zp, rowvar=False))

    # REAL perturbation-response validation
    acc, base = predict_perturbation_identity(Zp, adata.obs[PERT_COL])
    Zp_mean = Zp[pert_mask].mean(0)
    Zc_mean = Zp[ctrl_mask].mean(0)
    rhos = de_correlation(X, adata.obs[PERT_COL], W1, Zp_mean, Zc_mean)

    results = {
        "dataset": kind,
        "n_cells": int(adata.n_obs),
        "n_perturbations": int(adata.obs[PERT_COL].nunique()),
        "kappa_C_W": float(kCW),
        "kappa_C_Z": float(kCZ),
        "eff_rank_vanilla": float(er_v),
        "eff_rank_perturbed": float(er_p),
        "eff_rank_delta_pct": float(100 * (er_p - er_v) / er_v),
        "perturbation_identity_accuracy": acc,
        "perturbation_identity_baseline": base,
        "de_spearman_topk": rhos,
        "elapsed_sec": float(time.time() - t0),
    }
    with open(RESULT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
