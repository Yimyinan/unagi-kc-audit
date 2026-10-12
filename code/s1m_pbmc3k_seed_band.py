# -*- coding: utf-8 -*-
"""S1m -- family-matched seed band for the scVI/pbmc3k cross-architecture pair.

Motivation
----------
Section 2.9's pbmc3k pair (kappa-C(W) = 0.873, kappa-C(Z) = 1.078, eff_rank
12.57 -> 15.66, +24.6%) does not reproduce under the manuscript's own declared
configuration.  Re-running ``code/scvi_demo_benchmark.py`` verbatim in an
isolated ``scvi-tools 1.5.0`` environment (see ``code/s1l_pbmc3k_scvi_recheck.py
--likelihood normal``) reproduces the *vanilla* arm (eff_rank 12.88 vs 12.57, and
the raw-expression label benchmark matches the published value to 16 digits) but
produces a label-conditioned arm that moves the other way (-3.6%).

Why the "conditioned" arm is not conditioned
--------------------------------------------
In scvi-tools, the ``labels_key`` passed to ``setup_anndata`` does not by itself
make ``SCVI`` condition on labels.  Labels reach the generative model only through
``dispersion="gene-label"`` (``scvi/module/_vae.py`` L205-215, where ``px_r`` is
allocated with shape ``(n_input, n_labels)``); the default is ``dispersion="gene"``.
``scvi_demo_benchmark.py`` never sets ``dispersion``.  The two arms are therefore
two independent fits on the same data, differing only in their random stream.

What this script measures
-------------------------
It repeats both arms over several seeds and reports the distribution of the paired
ratio, together with the cross-seed null of two *vanilla* fits.  If the label arm
is inert, the two spreads coincide and the published 0.873 is a draw from that
distribution -- which is the honest statement of what section 2.9's number is.

Reads  : pbmc3k raw h5ad from ``sc.settings.datasetdir`` (downloads on first use)
Writes : outputs/positive_result/s1m_pbmc3k_seed_band.json
Usage  : python code/s1m_pbmc3k_seed_band.py --seeds 6
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import sys
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

import pandas as pd                                        # noqa: E402
import scanpy as sc                                        # noqa: E402
import scvi                                                # noqa: E402
import torch                                               # noqa: E402
import importlib.metadata as md                            # noqa: E402
from kappa_c_audit.core import (eff_rank, predictability_total,   # noqa: E402
                                ridge_readout)

PUBLISHED = dict(kappa_C_W_trace=0.872875208973927,
                 kappa_C_Z=1.0778976280359904,
                 eff_rank_vanilla=12.567126033416036,
                 eff_rank_conditioned=15.660359435356279,
                 eff_rank_delta_pct=24.613689667115004)


def corr_cond(Z):
    Z = np.asarray(Z, float)
    Z = Z - Z.mean(0, keepdims=True)
    sd = Z.std(0, ddof=0, keepdims=True)
    sd[sd < 1e-12] = 1.0
    Zs = Z / sd
    C = Zs.T @ Zs / Zs.shape[0]
    ev = np.linalg.eigvalsh(C)
    return float(ev.max() / max(ev.min(), 1e-300)), int((ev < 0.01).sum())


def load_data():
    """Preprocessing copied verbatim from code/scvi_demo_benchmark.py."""
    adata = sc.datasets.pbmc3k()
    ds = os.path.join(str(sc.settings.datasetdir), "pbmc3k_raw.h5ad")
    dsh = None
    if os.path.exists(ds):
        with open(ds, "rb") as fh:
            dsh = hashlib.sha256(fh.read()).hexdigest()
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
    return adata, X, dsh


def fit(adata, seed, injection, epochs, accel, lik):
    """One SCVI fit.  ``injection`` selects how (or whether) labels enter.

    "none"                   -- vanilla
    "labels_key"             -- what the published script did: inert, because labels
                                reach scVI's generative model only through
                                dispersion="gene-label" and the default is "gene"
    "categorical_covariate"  -- registered as a decoder-side covariate; this is the
                                route the manuscript's Norman experiment uses and
                                the only one that actually conditions the decoder
    """
    scvi.settings.seed = seed
    torch.manual_seed(seed)
    a = adata.copy()
    a.obs["celltype"] = a.obs["celltype"].astype("category")
    if injection == "labels_key":
        scvi.model.SCVI.setup_anndata(a, labels_key="celltype")
    elif injection == "categorical_covariate":
        scvi.model.SCVI.setup_anndata(a, categorical_covariate_keys=["celltype"])
    else:
        scvi.model.SCVI.setup_anndata(a)
    m = scvi.model.SCVI(a, n_latent=30, n_layers=2, dropout_rate=0.1,
                        gene_likelihood=lik)
    m.train(max_epochs=epochs, accelerator=accel)
    return m.get_latent_representation(a).astype(np.float64)


def describe(v):
    v = np.asarray(v, float)
    sd = float(v.std(ddof=1)) if v.size > 1 else float("nan")
    return dict(n=int(v.size), mean=float(v.mean()), sd=sd,
                lo2s=float(v.mean() - 2 * sd), hi2s=float(v.mean() + 2 * sd),
                min=float(v.min()), max=float(v.max()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--accelerator", default="cpu")
    ap.add_argument("--likelihood", default="normal")
    ap.add_argument("--injection", default="labels_key",
                    choices=["labels_key", "categorical_covariate", "none"])
    ap.add_argument("--out", default="s1m_pbmc3k_seed_band.json")
    a = ap.parse_args()

    out = {"architecture_scvi_tools": md.version("scvi-tools"),
           "torch": md.version("torch"),
           "scanpy": md.version("scanpy"),
           "gene_likelihood": a.likelihood, "max_epochs": a.epochs,
           "accelerator": a.accelerator,
           "model_kwargs": dict(n_latent=30, n_layers=2, dropout_rate=0.1,
                                dispersion="gene (default)"),
           "injection_route": a.injection,
           "note": ("family-matched seed band for the scVI/pbmc3k pair")}

    log("=== loading + preprocessing pbmc3k (verbatim from scvi_demo_benchmark) ===")
    adata, X, dsh = load_data()
    out["dataset_sha256"] = dsh
    out["n_cells"], out["n_genes"] = int(adata.n_obs), int(adata.n_vars)
    log(f"cells={adata.n_obs} genes={adata.n_vars} dataset_sha256={str(dsh)[:16]}")

    rows = []
    if a.merge and os.path.exists(os.path.join(OUT, a.out)):
        with open(os.path.join(OUT, a.out), encoding="utf-8") as f:
            rows = list(json.load(f).get("per_seed", []))
        log(f"merging {len(rows)} existing seed rows")
    for s in range(a.seed_start, a.seed_start + a.seeds):
        if any(r["seed"] == s for r in rows):
            log(f"seed {s} already present, skipping")
            continue
        Zv = fit(adata, s, "none", a.epochs, a.accelerator, a.likelihood)
        Zl = fit(adata, s, a.injection, a.epochs, a.accelerator, a.likelihood)
        Wv, Wl = ridge_readout(Zv, X).astype(np.float64), ridge_readout(Zl, X).astype(np.float64)
        trv, trl = float((Wv ** 2).sum()), float((Wl ** 2).sum())
        sv, sl = predictability_total(Zv, X), predictability_total(Zl, X)
        cv = np.cov(Zv, rowvar=False)
        cl = np.cov(Zl, rowvar=False)
        ev, el = eff_rank(cv), eff_rank(cl)
        c0, n0 = corr_cond(Zv)
        row = dict(seed=s, tr_vanilla=trv, tr_labeled=trl, sum_r2_vanilla=sv,
                   sum_r2_labeled=sl, eff_rank_vanilla=ev, eff_rank_labeled=el,
                   eff_rank_delta_pct=100 * (el - ev) / ev,
                   kappa_trace=trl / trv, kappa_corrected=sl / sv,
                   kappa_C_Z=float(np.trace(cl) / np.trace(cv)),
                   trace_Z_vanilla=float(np.trace(cv)),
                   trace_Z_labeled=float(np.trace(cl)),
                   eff_rank_ratio=el / ev, corr_cond_vanilla=c0,
                   n_eig_below_0p01_vanilla=n0)
        rows.append(row)
        log(f"seed {s}: kappa_tr={row['kappa_trace']:.4f}  "
            f"kappa_r2={row['kappa_corrected']:.4f}  "
            f"kappa_Z={row['kappa_C_Z']:.4f}  "
            f"eff_rank {ev:.2f}->{el:.2f} ({row['eff_rank_delta_pct']:+.1f}%)  "
            f"cond={c0:.0f}")

    rows.sort(key=lambda r: r["seed"])
    out["seeds"] = [r["seed"] for r in rows]
    out["per_seed"] = rows
    out["paired_label_vs_vanilla"] = {
        k: describe([r[k] for r in rows])
        for k in ("kappa_trace", "kappa_corrected", "kappa_C_Z", "eff_rank_ratio")}

    # cross-seed null: two VANILLA fits, all unordered pairs -> same seeds, no labels
    null_rows = []
    for i, j in itertools.combinations(range(len(rows)), 2):
        ri, rj = rows[i], rows[j]
        null_rows.append(dict(pair=f"{i}/{j}",
                              kappa_trace=ri["tr_vanilla"] / rj["tr_vanilla"],
                              kappa_corrected=ri["sum_r2_vanilla"] / rj["sum_r2_vanilla"],
                              kappa_C_Z=ri["trace_Z_vanilla"] / rj["trace_Z_vanilla"],
                              eff_rank_ratio=ri["eff_rank_vanilla"] / rj["eff_rank_vanilla"]))
    out["cross_seed_vanilla_null"] = {
        k: describe([r[k] for r in null_rows])
        for k in ("kappa_trace", "kappa_corrected", "kappa_C_Z", "eff_rank_ratio")}
    out["cross_seed_vanilla_null_pairs"] = null_rows
    out["published"] = PUBLISHED

    # where does the published value sit in the paired band?
    band = out["paired_label_vs_vanilla"]["kappa_trace"]
    z = ((PUBLISHED["kappa_C_W_trace"] - band["mean"]) / band["sd"]
         if band["sd"] else float("nan"))
    out["published_position"] = dict(
        band_mean=band["mean"], band_sd=band["sd"], band_lo2s=band["lo2s"],
        band_hi2s=band["hi2s"], published=PUBLISHED["kappa_C_W_trace"],
        z=z, inside_2sigma=bool(band["lo2s"] <= PUBLISHED["kappa_C_W_trace"] <= band["hi2s"]))

    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, a.out)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)

    log()
    log("================= family-matched seed band (scVI / pbmc3k) =================")
    log(f"{'statistic':<26}{'mean':>9}{'sd':>9}{'2sigma band':>22}")
    for k, v in out["paired_label_vs_vanilla"].items():
        log(f"{k:<26}{v['mean']:>9.4f}{v['sd']:>9.4f}"
            f"{'[' + format(v['lo2s'], '.3f') + ', ' + format(v['hi2s'], '.3f') + ']':>22}")
    log()
    log("cross-seed null (two VANILLA fits, no labels):")
    for k, v in out["cross_seed_vanilla_null"].items():
        log(f"  {k:<24}{v['mean']:>9.4f}{v['sd']:>9.4f}  "
            f"[{v['lo2s']:.3f}, {v['hi2s']:.3f}]  n={v['n']}")
    log()
    log(f"published kappa-C(W) trace = {PUBLISHED['kappa_C_W_trace']:.4f}  "
        f"-> z = {z:+.2f} vs the paired band; inside 2 sigma: "
        f"{out['published_position']['inside_2sigma']}")
    log(f"published eff_rank ratio  = "
        f"{PUBLISHED['eff_rank_conditioned'] / PUBLISHED['eff_rank_vanilla']:.4f}")
    log()
    log(f"wrote {path}")


if __name__ == "__main__":
    main()
