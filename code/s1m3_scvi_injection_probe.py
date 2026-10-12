# -*- coding: utf-8 -*-
"""S1m3 -- does anything actually inject the labels into scVI?

Section 2.9 claims eight k-means cell-type clusters were "injected as a categorical
structural prior via scVI's label-conditioning mechanism".  This probe tests that
claim mechanically, at a fixed seed, by comparing the vanilla fit against every
route scvi-tools offers for getting a categorical covariate into the model:

  A. vanilla                      -- ``setup_anndata(adata)``
  B. ``labels_key``               -- what the published script does
  C. ``categorical_covariate_keys`` -- decoder-side covariate (if accepted)
  D. ``labels_key`` + ``dispersion="gene-label"``
                                     -- the only path by which labels reach
                                        scVI's generative model (px_r is allocated
                                        with shape (n_input, n_labels) only here)

The random seed is reset before every arm, so any difference between arms is
attributable to the injection route, not to the random stream.  A route whose
latent is bit-identical to the vanilla latent did not inject anything.

Reads  : pbmc3k (same preprocessing as code/scvi_demo_benchmark.py)
Writes : outputs/positive_result/s1m3_scvi_injection_probe.json
Usage  : python code/s1m3_scvi_injection_probe.py --seed 0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
import warnings

import numpy as np

warnings.filterwarnings("ignore")


def log(*a):
    print(*a, flush=True)


HERE = os.path.dirname(os.path.abspath(__file__))
PAPER1 = os.path.abspath(os.path.join(HERE, ".."))
ROOT = os.path.abspath(os.path.join(PAPER1, "..", ".."))
OUT = os.path.join(ROOT, "outputs", "positive_result")
sys.path.insert(0, os.path.join(PAPER1, "kappa_c_audit"))

import pandas as pd                                       # noqa: E402
import scanpy as sc                                       # noqa: E402
import scvi                                               # noqa: E402
import torch                                              # noqa: E402
import importlib.metadata as md                           # noqa: E402
from kappa_c_audit.core import (eff_rank, predictability_total,  # noqa: E402
                                ridge_readout)

ROUTES = [
    ("vanilla", {}),
    ("labels_key", dict(labels_key="celltype")),
    ("categorical_covariate_keys", dict(categorical_covariate_keys=["celltype"])),
    ("labels_key + dispersion=gene-label", dict(labels_key="celltype"),
     dict(dispersion="gene-label")),
]


def load_data():
    adata = sc.datasets.pbmc3k()
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
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.array(adata.X)
    return adata, X


def metrics(Z, X):
    W = ridge_readout(Z, X).astype(np.float64)
    return dict(trace=float((W ** 2).sum()),
                sum_r2=float(predictability_total(Z, X)),
                eff_rank=float(eff_rank(np.cov(Z, rowvar=False))),
                trace_Z=float(np.trace(np.cov(Z, rowvar=False))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--accelerator", default="cpu")
    ap.add_argument("--likelihood", default="normal")
    ap.add_argument("--out", default="s1m3_scvi_injection_probe.json")
    a = ap.parse_args()

    out = {"scvi_tools": md.version("scvi-tools"), "torch": md.version("torch"),
           "seed": a.seed, "max_epochs": a.epochs,
           "gene_likelihood": a.likelihood, "accelerator": a.accelerator,
           "design": ("seed reset before every arm, so an arm that differs from "
                      "vanilla differs because the route injected something")}

    adata, X = load_data()
    out["n_cells"], out["n_genes"] = int(adata.n_obs), int(adata.n_vars)
    log(f"cells={adata.n_obs} genes={adata.n_vars}")

    base = None
    arms = []
    for spec in ROUTES:
        name, setup_kw = spec[0], spec[1]
        model_kw = spec[2] if len(spec) > 2 else {}
        rec = {"route": name, "setup_kwargs": setup_kw, "model_kwargs": model_kw}
        try:
            scvi.settings.seed = a.seed
            torch.manual_seed(a.seed)
            ad = adata.copy()
            ad.obs["celltype"] = ad.obs["celltype"].astype("category")
            scvi.model.SCVI.setup_anndata(ad, **setup_kw)
            m = scvi.model.SCVI(ad, n_latent=30, n_layers=2, dropout_rate=0.1,
                                gene_likelihood=a.likelihood, **model_kw)
            m.train(max_epochs=a.epochs, accelerator=a.accelerator)
            Z = m.get_latent_representation(ad).astype(np.float64)
            rec.update(metrics(Z, X))
            rec["n_labels_in_model"] = int(getattr(m.module, "n_labels", -1))
            rec["n_parameters"] = int(sum(p.numel() for p in m.module.parameters()))
            rec["dispersion"] = str(getattr(m.module, "dispersion", "?"))
            if base is None:
                base = Z
                rec.update(max_abs_dev_from_vanilla=0.0,
                           bitwise_identical_to_vanilla=True)
            else:
                rec.update(max_abs_dev_from_vanilla=float(np.abs(Z - base).max()),
                           bitwise_identical_to_vanilla=bool(np.array_equal(Z, base)))
            rec["status"] = "ok"
            log(f"{name:<38} status=ok  max|dZ|="
                f"{rec['max_abs_dev_from_vanilla']:.3e}  "
                f"identical={rec['bitwise_identical_to_vanilla']}  "
                f"eff_rank={rec['eff_rank']:.3f}  n_params={rec['n_parameters']}")
        except Exception as e:                       # noqa: BLE001
            rec["status"] = f"FAILED: {type(e).__name__}: {e}"
            rec["traceback"] = traceback.format_exc()[-800:]
            log(f"{name:<38} status=FAILED  {type(e).__name__}: {str(e)[:120]}")
        arms.append(rec)

    out["arms"] = arms
    ok = {r["route"]: r for r in arms if r["status"] == "ok"}
    van = ok.get("vanilla")
    out["verdict"] = {
        r["route"]: ("INERT (bit-identical to vanilla)"
                     if r.get("bitwise_identical_to_vanilla") else "injected")
        for r in arms if r["status"] == "ok" and r["route"] != "vanilla"}
    if van:
        out["verdict"]["vanilla"] = "reference"
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, a.out), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)

    log()
    log("route -> " + json.dumps(out["verdict"], ensure_ascii=False))
    log(f"wrote {os.path.join(OUT, a.out)}")


if __name__ == "__main__":
    main()
