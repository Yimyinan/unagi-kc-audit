"""GRQ virtual-perturbation CLOSURE on a real CRISPR screen (Norman 2019 GEARS).

WHY THIS EXISTS
---------------
The manuscript's stated limitation is that the GRQ readout was "demonstrated only
on vanilla-equivalent models", so it remained unclear whether a model that the
kappa-C audit calls *perturbation-ready* actually predicts real perturbation
biology. This script closes that loop.

It asks, for each well-sampled real CRISPR perturbation p:
    If we compute the GRQ-optimal latent direction v*_p on the
    PERTURBATION-CONDITIONED model, and decode along it, does the predicted
    expression response match what the real cells actually did?

GRQ definition (identical to the manuscript, Methods):
    v* = argmax_v  (v' J_T' J_T v) / (v' (J'J + eps I) v)
where J is the decoder Jacobian (G x D) and T is the target-gene index set.
Here J is the UNAGI-protocol linear readout J = W.T with W = Ridge(Z -> X),
the same estimator the manuscript cross-validates against the analytical
Jacobian (Methods, Limitations).

THREE-WAY COMPARISON (the point of the experiment)
    rho_grq_pert   : GRQ direction on the perturbation-conditioned model
    rho_grq_vanilla: GRQ direction on the vanilla model        (model control)
    rho_shift      : naive mean latent-shift direction          (direction control)
If GRQ on the conditioned model beats both controls, the loop is closed:
virtual perturbation on an audited model predicts real perturbation response.

Runs ENTIRELY from cached latents (data/_norman_latents.npz) -- no retraining.
Gene names are recovered by re-running the identical preprocessing pipeline via
perturbation_benchmark.load_adata (guarantees the same 2000 HVGs).

USAGE
-----
    LOCAL_PERT_H5AD=<norman.h5ad> PERT_COL=condition_name \
    CONTROL_VAL=A549_ctrl_1 python grq_perturbation_closure.py
"""

import os
import sys
import json
import time
import re

import numpy as np

# Reuse the exact, already-validated machinery (preprocessing, Ridge readout,
# perturbation-label handling) so the GRQ numbers stay on the same footing as
# the published kappa-C(W)/kappa-C(Z) figures.
import perturbation_benchmark as pb

from scipy.stats import spearmanr
from scipy.linalg import eigh

OUT_JSON = os.path.join(os.path.dirname(__file__), "..", "data",
                        "grq_closure_results.json")

MIN_CELLS = 50          # per-perturbation cell count to be "well sampled"
EPS_REL = 1e-6          # GRQ ridge epsilon, relative to trace(J'J)/D


# ---------------------------------------------------------------------------
def parse_target_genes(label, control_val):
    """Extract the perturbed gene symbol(s) from a Norman condition label.

    Labels look like 'A549_AHR+ctrl_1+1' or 'A549_GENE1+GENE2+1+1'.
    Strategy: drop the leading cell-line prefix, split on '+', and keep tokens
    that are neither 'ctrl' nor a pure replicate index.
    """
    name = str(label)
    if "_" in name:
        name = name.split("_", 1)[1]          # drop 'A549_'
    tokens = re.split(r"[+]", name)
    genes = []
    for t in tokens:
        t = t.strip()
        if not t:
            continue
        if t.lower().startswith("ctrl"):
            continue
        if re.fullmatch(r"\d+", t):           # replicate index like '1'
            continue
        if re.fullmatch(r"[\d.]+", t):
            continue
        genes.append(t)
    return genes


def grq_direction(J, target_idx, eps_rel=EPS_REL):
    """GRQ-optimal latent direction.

    J : (G, D) decoder readout (Jacobian proxy)
    target_idx : array of gene indices defining the target set T
    returns v* (D,) unit-norm, maximising (v'J_T'J_T v)/(v'(J'J+eps I)v)
    """
    JT = J[target_idx, :]                      # (|T|, D)
    A = JT.T @ JT                              # (D, D)
    B = J.T @ J                                # (D, D)
    eps = eps_rel * float(np.trace(B)) / B.shape[0]
    B = B + eps * np.eye(B.shape[0])
    # generalized eigenproblem A v = lambda B v ; take largest eigenvalue
    w, V = eigh(A, B)
    v = V[:, np.argmax(w)]
    n = np.linalg.norm(v)
    return v / (n if n > 0 else 1.0)


def target_specificity(dx, target_idx):
    """Fold-enrichment of the predicted response on target vs all genes
    (manuscript's 'erythroid specificity' analogue)."""
    a = np.abs(dx)
    if a.mean() <= 0:
        return float("nan")
    return float(a[target_idx].mean() / a.mean())


def main():
    t0 = time.time()

    # ---- 1. cached latents -------------------------------------------------
    cache = pb._cache if hasattr(pb, "_cache") else os.path.join(
        os.path.dirname(__file__), "..", "data", "_norman_latents.npz")
    if not os.path.exists(cache):
        raise SystemExit("latent cache not found: %s (run perturbation_benchmark.py first)" % cache)
    d = np.load(cache, allow_pickle=True)
    Zv, Zp, X, labels = d["Zv"], d["Zp"], d["X"], d["pert_labels"]
    print("[cache] Zv%s Zp%s X%s labels%s" % (Zv.shape, Zp.shape, X.shape, labels.shape))

    # ---- 2. gene names via the IDENTICAL preprocessing pipeline ------------
    # NOTE: perturbation_benchmark.main() applies the HVG contraction AFTER
    # load_adata(), so we must replicate that exact step or the gene names will
    # correspond to the pre-HVG 5045-gene matrix instead of the cached 2000.
    import scanpy as sc
    path, kind = pb.fetch_dataset()
    adata = pb.load_adata(path, kind)
    sc.pp.highly_variable_genes(adata, n_top_genes=pb.N_HVGS)
    adata = adata[:, adata.var.highly_variable].copy()
    genes = np.array([str(g) for g in adata.var_names])
    print("[adata] %s  genes=%d" % (adata.shape, len(genes)))
    print("[genes] sample:", list(genes[:8]))
    if adata.shape[0] != X.shape[0] or adata.shape[1] != X.shape[1]:
        print("[WARN] adata shape %s != cached X %s -- gene mapping may be misaligned"
              % (adata.shape, X.shape), file=sys.stderr)
    # The pipeline sets var_names to Ensembl IDs, but the SYMBOL column
    # ('feature_name') survives in adata.var -- perturbation labels name genes by
    # symbol, so the target lookup must use symbols, not var_names.
    sym = None
    for c in ("feature_name", "gene_symbols", "gene_name", "symbol", "gene_short_name"):
        if c in adata.var.columns:
            sym = np.array([str(x) for x in adata.var[c]])
            print("[genes] symbol column='%s' sample=%s" % (c, list(sym[:6])))
            break
    if sym is None:
        sym = genes
        print("[genes] WARN no symbol column found; falling back to var_names")
    gene_pos = {}
    for i, s in enumerate(sym):
        gene_pos.setdefault(s.upper(), i)
    print("[genes] 'AHR' ->", gene_pos.get("AHR"))

    # sanity: show how labels parse before the loop
    _lab_demo = [str(x) for x in np.unique(np.array([str(v) for v in labels]))[:4]]
    print("[parse] demo:", [(l, parse_target_genes(l, str(pb.CONTROL_VAL))) for l in _lab_demo])

    # ---- 3. decoder readouts ----------------------------------------------
    W0 = pb.ridge_readout(Zv, X)      # vanilla            (latent, genes)
    W1 = pb.ridge_readout(Zp, X)      # perturbation-conditioned
    J0, J1 = W0.T, W1.T               # (genes, latent) Jacobian proxies
    print("[readout] J0%s J1%s" % (J0.shape, J1.shape))

    kCW = float((W1 ** 2).sum() / (W0 ** 2).sum())
    kCZ = pb.kappa_C_Z(Zv, Zp)
    print("[kappa] kappa-C(W)=%.3f  kappa-C(Z)=%.3f" % (kCW, kCZ))

    # ---- 4. per-perturbation GRQ closure ----------------------------------
    labels_s = np.array([str(x) for x in labels])
    ctrl_val = str(pb.CONTROL_VAL)
    ctrl_mask = labels_s == ctrl_val
    ctrl_mean = X[ctrl_mask].mean(0)
    # naive baseline direction: mean latent shift (perturbation-conditioned model)
    shift = Zp[~ctrl_mask].mean(0) - Zp[ctrl_mask].mean(0)
    sn = np.linalg.norm(shift)
    shift_u = shift / (sn if sn > 0 else 1.0)
    # same direction computed on the VANILLA model -- the control that decides
    # whether conditioning is what makes the shift predictive
    shift_v = Zv[~ctrl_mask].mean(0) - Zv[ctrl_mask].mean(0)
    svn = np.linalg.norm(shift_v)
    shift_v_u = shift_v / (svn if svn > 0 else 1.0)

    uniq, counts = np.unique(labels_s, return_counts=True)
    cands = [(u, int(c)) for u, c in zip(uniq, counts)
             if u != ctrl_val and c >= MIN_CELLS]
    cands.sort(key=lambda t: -t[1])
    print("[pert] %d candidate perturbations (n>=%d cells)"
          % (len(cands), MIN_CELLS))

    rows = []
    for lab, ncell in cands:
        genes_t = parse_target_genes(lab, ctrl_val)
        tidx = np.array(sorted({gene_pos[g.upper()] for g in genes_t
                                if g.upper() in gene_pos}), dtype=int)
        if tidx.size == 0:
            continue                                  # target gene not in HVGs
        mask = labels_s == lab
        obs = X[mask].mean(0) - ctrl_mean             # observed logFC
        if np.std(obs) < 1e-9:
            continue

        v_pert = grq_direction(J1, tidx)
        v_vani = grq_direction(J0, tidx)

        pred_pert = J1 @ v_pert
        pred_vani = J0 @ v_vani
        pred_shift = J1 @ shift_u
        pred_shift_van = J0 @ shift_v_u

        r_pert, _ = spearmanr(pred_pert, obs)
        r_vani, _ = spearmanr(pred_vani, obs)
        r_shift, _ = spearmanr(pred_shift, obs)
        r_shift_van, _ = spearmanr(pred_shift_van, obs)

        rows.append(dict(
            perturbation=lab,
            target_genes=",".join(genes_t),
            n_cells=ncell,
            rho_grq_pert=float(r_pert),
            rho_grq_vanilla=float(r_vani),
            rho_shift_baseline=float(r_shift),
            rho_shift_vanilla=float(r_shift_van),
            spec_grq_pert=target_specificity(pred_pert, tidx),
            spec_grq_vanilla=target_specificity(pred_vani, tidx),
            spec_shift=target_specificity(pred_shift, tidx),
            spec_observed=target_specificity(obs, tidx),
        ))
        print("  %-28s n=%-5d  rho: GRQ_pert=%+.3f  GRQ_van=%+.3f  shift=%+.3f   spec=%.2fx"
              % (lab[:28], ncell, r_pert, r_vani, r_shift,
                 target_specificity(pred_pert, tidx)))

    if not rows:
        raise SystemExit("no perturbation had its target gene among the 2000 HVGs")

    def agg(key):
        v = [r[key] for r in rows if np.isfinite(r[key])]
        return float(np.mean(v)), float(np.std(v)), len(v)

    out = dict(
        dataset="Norman 2019 GEARS (per-cell CRISPR perturbation screen)",
        n_cells=int(X.shape[0]),
        n_genes=int(X.shape[1]),
        n_latent=int(Zp.shape[1]),
        control_label=ctrl_val,
        kappa_C_W=kCW,
        kappa_C_Z=kCZ,
        n_perturbations_evaluated=len(rows),
        min_cells=MIN_CELLS,
        estimator="GRQ on UNAGI-protocol linear readout J = W.T, W = Ridge(Z->X, alpha=1, no intercept)",
        summary={k: {"mean": agg(k)[0], "sd": agg(k)[1], "n": agg(k)[2]}
                 for k in ["rho_grq_pert", "rho_grq_vanilla", "rho_shift_baseline",
                           "rho_shift_vanilla",
                           "spec_grq_pert", "spec_grq_vanilla", "spec_shift",
                           "spec_observed"]},
        per_perturbation=rows,
        elapsed_sec=float(time.time() - t0),
    )

    # headline test: does GRQ on the conditioned model beat the two controls?
    mp = out["summary"]["rho_grq_pert"]["mean"]
    mv = out["summary"]["rho_grq_vanilla"]["mean"]
    ms = out["summary"]["rho_shift_baseline"]["mean"]
    ms_van = out["summary"]["rho_shift_vanilla"]["mean"]
    out["headline"] = dict(
        mean_rho_grq_perturbation_conditioned=mp,
        mean_rho_grq_vanilla=mv,
        mean_rho_shift_baseline=ms,
        mean_rho_shift_vanilla=ms_van,
        grq_beats_vanilla=bool(mp > mv),
        grq_beats_shift_baseline=bool(mp > ms),
        conditioning_improves_shift_prediction=bool(ms > ms_van),
    )

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("\n[saved] %s" % os.path.abspath(OUT_JSON))
    print("[headline] GRQ(pert)=%.4f  GRQ(vanilla)=%.4f  shift=%.4f  -> beats_vanilla=%s beats_shift=%s"
          % (mp, mv, ms, mp > mv, mp > ms))
    print("[done] %.1f s" % (time.time() - t0))


if __name__ == "__main__":
    main()
