# -*- coding: utf-8 -*-
"""S1l -- re-run the pbmc3k scVI cross-architecture audit in the corrected statistic.

Why
---
Section 2.9 reports the scVI/pbmc3k pair as decoder-invariant (kappa-C(W) = 0.87,
"comfortably inside the vanilla-equivalence band").  That value is in the released
trace form and the pbmc3k latents were never retained, so it is the last
kappa-C(W) in the manuscript whose corrected counterpart is unknown.  The training
protocol is fully preserved in ``code/scvi_demo_benchmark.py``, so it is re-run here.

Environment note
----------------
``scvi-tools 1.1.6.post2`` is installed in the unagi-py39 environment but its import
chain needs ``scipy.linalg.tril`` / ``triu``, which scipy >= 1.13 removed (the runtime
has scipy 1.13.1).  Two runtime aliases to the numpy equivalents are applied below
instead of touching the environment.  The manuscript quotes scVI 1.5.0; the version
actually used is recorded in the output.

Protocol (unchanged from scvi_demo_benchmark.py)
------------------------------------------------
pbmc3k -> min_genes 200, min_cells 3, pct_counts_mt < 5 -> normalize_total 1e4 ->
log1p -> 2000 HVGs -> PCA(arpack) -> KMeans(k=8) on X_pca as the injected
"cell-type" prior -> two SCVI(n_latent=30, n_layers=2, dropout 0.1,
gene_likelihood=LIK), seed 0, max_epochs 200; the second is label-conditioned.

Reads  : (downloads pbmc3k via scanpy on first use)
Writes : outputs/positive_result/s1l_pbmc3k_scvi.json
Usage  : python code/s1l_pbmc3k_scvi_recheck.py
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")


def log(*a):
    print(*a, flush=True)

# ---- scipy shim (must precede the scvi import) -----------------------------
import scipy.linalg  # noqa: E402

for _n in ("tril", "triu"):
    if not hasattr(scipy.linalg, _n):
        setattr(scipy.linalg, _n, getattr(np, _n))

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER1 = os.path.abspath(os.path.join(HERE, ".."))
ROOT = os.path.abspath(os.path.join(PAPER1, "..", ".."))
OUT = os.path.join(ROOT, "outputs", "positive_result")
OUTNAME = "s1l_pbmc3k_scvi.json"
sys.path.insert(0, os.path.join(PAPER1, "kappa_c_audit"))

import pandas as pd  # noqa: E402
import scanpy as sc  # noqa: E402
import scvi  # noqa: E402
import importlib.metadata as md  # noqa: E402
from kappa_c_audit.core import (eff_rank, predictability_total,  # noqa: E402
                                ridge_readout)


def corr_cond(Z):
    Z = np.asarray(Z, float)
    Z = Z - Z.mean(0, keepdims=True)
    sd = Z.std(0, ddof=0, keepdims=True)
    sd[sd < 1e-12] = 1.0
    Zs = Z / sd
    C = Zs.T @ Zs / Zs.shape[0]
    ev = np.linalg.eigvalsh(C)
    return float(ev.max() / max(ev.min(), 1e-300)), int((ev < 0.01).sum())


def _v(pkg):
    try:
        return md.version(pkg)
    except Exception:
        return None


def phenotype_predictability(Z, labels, n_splits=5, seed=0):
    """Copied verbatim from code/scvi_demo_benchmark.py so results are comparable."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    accs = []
    for tr, te in skf.split(Z, labels):
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=2000,
                                               class_weight="balanced"))
        clf.fit(Z[tr], labels[tr])
        accs.append(balanced_accuracy_score(labels[te], clf.predict(Z[te])))
    return float(np.mean(accs)), float(np.std(accs))


def main():
    global ACCEL, EPOCHS, LIK, OUTNAME
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--accelerator", default="gpu")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--likelihood", default="zinb")
    ap.add_argument("--out", default=OUTNAME)
    a = ap.parse_args()
    ACCEL, EPOCHS, LIK, OUTNAME = a.accelerator, a.epochs, a.likelihood, a.out
    out = {"architecture_scvi_tools": md.version("scvi-tools"),
           "accelerator": ACCEL, "max_epochs": EPOCHS,
           "gene_likelihood": LIK,
           "versions": {p: _v(p) for p in ("scvi-tools", "torch", "scanpy",
                                           "anndata", "numpy", "scikit-learn")},
           "model_kwargs": dict(n_latent=30, n_layers=2, dropout_rate=0.1),
           "dataset": "pbmc3k (scanpy)",
           "note": "re-run of code/scvi_demo_benchmark.py to obtain the corrected statistic"}

    log("=== loading pbmc3k ===")
    adata = sc.datasets.pbmc3k()
    _ds = os.path.join(str(sc.settings.datasetdir), "pbmc3k_raw.h5ad")
    _dsh = None
    if os.path.exists(_ds):
        with open(_ds, "rb") as _fh:
            _dsh = hashlib.sha256(_fh.read()).hexdigest()
    log(f"dataset cache: {_ds}  sha256={str(_dsh)[:16]}")
    out["dataset_sha256"] = _dsh
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)
    adata = adata[adata.obs.pct_counts_mt < 5].copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    adata = adata[:, adata.var.highly_variable].copy()
    _X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    adata.X = np.nan_to_num(_X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    sc.tl.pca(adata, svd_solver="arpack")
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=8, n_init=10, random_state=0).fit(adata.obsm["X_pca"])
    adata.obs["celltype"] = pd.Categorical([f"C{int(c)}" for c in km.labels_])
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
    log(f"cells={adata.n_obs}  genes={adata.n_vars}  "
        f"clusters={adata.obs['celltype'].value_counts().to_dict()}")
    out["n_cells"] = int(adata.n_obs)
    out["n_genes"] = int(adata.n_vars)

    log("=== training vanilla scVI ===")
    scvi.settings.seed = 0
    scvi.model.SCVI.setup_anndata(adata)
    m0 = scvi.model.SCVI(adata, n_latent=30, n_layers=2, dropout_rate=0.1,
                         gene_likelihood=LIK)
    m0.train(max_epochs=EPOCHS, accelerator=ACCEL)
    Z0 = m0.get_latent_representation(adata).astype(np.float64)

    log("=== training label-conditioned scVI ===")
    adata_lab = adata.copy()
    adata_lab.obs["celltype"] = adata_lab.obs["celltype"].astype("category")
    scvi.model.SCVI.setup_anndata(adata_lab, labels_key="celltype")
    m1 = scvi.model.SCVI(adata_lab, n_latent=30, n_layers=2, dropout_rate=0.1,
                         gene_likelihood=LIK)
    m1.train(max_epochs=EPOCHS, accelerator=ACCEL)
    Z1 = m1.get_latent_representation(adata_lab).astype(np.float64)

    W0 = ridge_readout(Z0, X).astype(np.float64)
    W1 = ridge_readout(Z1, X).astype(np.float64)
    tr0, tr1 = float((W0 ** 2).sum()), float((W1 ** 2).sum())
    s0, s1 = predictability_total(Z0, X), predictability_total(Z1, X)
    c0 = eff_rank(np.cov(Z0, rowvar=False))
    c1 = eff_rank(np.cov(Z1, rowvar=False))
    kz = float(np.trace(np.cov(Z1, rowvar=False)) / np.trace(np.cov(Z0, rowvar=False)))
    cond0, n0 = corr_cond(Z0)
    cond1, n1 = corr_cond(Z1)

    out.update(
        kappa_C_W_trace=tr1 / tr0,
        kappa_C_W_corrected=s1 / s0,
        C_W_trace_pct=(1 - tr1 / tr0) * 100.0,
        C_W_corrected_pct=(1 - s1 / s0) * 100.0,
        sum_r2_vanilla=s0, sum_r2_conditioned=s1,
        trace_vanilla=tr0, trace_conditioned=tr1,
        kappa_C_Z=kz,
        eff_rank_vanilla=c0, eff_rank_conditioned=c1,
        eff_rank_delta_pct=100 * (c1 - c0) / c0,
        corr_cond_vanilla=cond0, corr_cond_conditioned=cond1,
        n_eig_below_0p01_vanilla=n0, n_eig_below_0p01_conditioned=n1,
    )
    _labels = adata.obs["celltype"].values
    acc0, sd0 = phenotype_predictability(Z0, _labels)
    acc1, sd1 = phenotype_predictability(Z1, _labels)
    accX, sdX = phenotype_predictability(X, _labels)
    out["benchmark"] = dict(acc_from_Z_vanilla=acc0, sd_Z_vanilla=sd0,
                            acc_from_Z_labeled=acc1, sd_Z_labeled=sd1,
                            acc_from_X_raw=accX, sd_X_raw=sdX)
    out["benchmark_published"] = dict(acc_from_Z_vanilla=0.895180179070892,
                                      acc_from_Z_labeled=0.901030074232701,
                                      acc_from_X_raw=0.9268170347626252)

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, OUTNAME), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)

    log()
    log(f"  published trace-form kappa-C(W) = 0.873  (C = 12.7%)")
    log(f"  re-run  trace-form kappa-C(W)   = {out['kappa_C_W_trace']:.4f}"
          f"  (C = {out['C_W_trace_pct']:+.1f}%)")
    log(f"  re-run  CORRECTED  kappa-C(W)   = {out['kappa_C_W_corrected']:.4f}"
        f"  (C = {out['C_W_corrected_pct']:+.1f}%)")
    log(f"  kappa-C(Z) = {kz:.4f}   eff_rank {c0:.2f} -> {c1:.2f} "
          f"({out['eff_rank_delta_pct']:+.1f}%)")
    log(f"  cond Corr(Z): vanilla {cond0:.0f} ({n0} eig<0.01)  "
          f"conditioned {cond1:.0f} ({n1})")
    log(f"  benchmark acc: Z_van(pub 0.8952) {acc0:.4f}  "
        f"Z_lab(pub 0.9010) {acc1:.4f}  X(pub 0.9268) {accX:.4f}")
    log(f"\nwrote {os.path.join(OUT, OUTNAME)}")


if __name__ == "__main__":
    main()
