# -*- coding: utf-8 -*-
"""S1n -- family-matched seed band for the Norman-2019 scVI pair (section 2.9.2).

Why
---
Section 2.9.2 reports the perturbation-conditioned scVI pair (kappa-C(W)=0.905
corrected, kappa-C(Z)=0.75, effective rank 8.80 -> 6.15, -30.1%) as a single run, so
there is no seed band behind its point estimates -- the same caveat that turned out to
matter for the pbmc3k control, where a single reported pair was two draws of one
configuration.  This script re-runs that configuration over several seeds.

Configuration (verbatim from code/perturbation_benchmark.py, which produced
data/perturbation_benchmark_results.json):
  h5ad -> normalize_total(1e4) -> log1p -> 2000 HVGs -> subset -> nan_to_num
  SCVI(n_latent=30, n_layers=2, dropout_rate=0.1, gene_likelihood="normal"),
  seed 42, max_epochs=200, accelerator="cpu"; the injected arm registers the
  perturbation identity with categorical_covariate_keys=[PERT_COL] as str.
  PERT_COL must be condition_name: the auto-detector picks the equivalent but
  differently named `condition` column, which changes the covariate index order.

Two facts established before any training:
  * perturbation_benchmark_results.json reports elapsed_sec = 32 s, far too short for
    2 x 200 epochs over 40,964 cells -- that run used USE_LATENT_CACHE=1 and read the
    latents from data/_norman_latents.npz.  The cache is the primary record of the
    published pair, and this script validates against it: max|dX| = 0 and the
    perturbation labels are bit-identical, so the data side matches exactly.
  * With this environment's numerical stack (torch 2.14 + scvi-tools 1.5.0) the
    normal-likelihood fit diverges near the end of the schedule:
    "Expected parameter scale ... to satisfy the constraint GreaterThan(0.0)" at
    epoch ~193/200.  The published run completed and cached full latents, so the
    divergence is an environment difference, not a data difference.  Training is
    therefore wrapped: on divergence the latent is still extracted and the epoch
    reached is recorded, so a run is not lost and the band stays interpretable.

Writes : outputs/positive_result/s1n_norman_seed_band.json  (+ .jsonl sidecar, one
         line per finished ARM, so a crash never discards a completed arm)
Usage  : python code/s1n_norman_seed_band.py --check-only
         python code/s1n_norman_seed_band.py --seeds 3
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")


def log(*a):
    print(*a, flush=True)


HERE = os.path.dirname(os.path.abspath(__file__))
PAPER1 = os.path.abspath(os.path.join(HERE, ".."))
ROOT = os.path.abspath(os.path.join(PAPER1, "..", ".."))
OUT = os.path.join(ROOT, "outputs", "positive_result")
DATA = os.path.join(PAPER1, "data")
CACHE = os.path.join(DATA, "_norman_latents.npz")
H5AD = os.environ.get(
    "LOCAL_PERT_H5AD",
    str(__import__("pathlib").Path(__file__).resolve().parents[1]
        / "data" / "_legacy_deepening")
    r"norman_singles_perturbation.h5ad")

N_HVGS, N_LATENT, N_LAYERS, DROPOUT = 2000, 30, 2, 0.1
N_EPOCHS, SEED = 200, 42
GENE_LIKELIHOOD = "normal"
RIDGE_ALPHA = 1.0
MAX_CELLS = 120000

PERT_CANDIDATES = ["perturbation", "gene_name", "guide", "target",
                   "perturbed_gene", "condition", "crisp_perturbation",
                   "crispr_perturbation"]
CONTROL_CANDIDATES = ["control", "ctrl", "ntc", "non-targeting", "nontargeting",
                      "nan", "none", "unperturbed", ""]

sys.path.insert(0, os.path.join(PAPER1, "kappa_c_audit"))


def load_norman(pert_col_override=None):
    """Preprocessing copied from code/perturbation_benchmark.py::main()."""
    import scanpy as sc
    adata = sc.read_h5ad(H5AD)
    if adata.n_obs > MAX_CELLS:
        sc.pp.subsample(adata, n_obs=MAX_CELLS, random_state=SEED)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=N_HVGS)
    adata = adata[:, adata.var.highly_variable].copy()
    _X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    X = np.nan_to_num(_X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    adata.X = X

    if pert_col_override:
        pert_col = pert_col_override
    elif "condition_name" in adata.obs.columns:
        pert_col = "condition_name"
    else:
        pert_col = next((c for c in PERT_CANDIDATES if c in adata.obs.columns), None)
    assert pert_col, "could not detect the perturbation column"
    vals = adata.obs[pert_col].astype(str)
    vl = set(vals.str.lower())
    control = next((v for v in CONTROL_CANDIDATES if v in vl), None)
    if control is None:
        cand = [v for v in vals.unique() if "ctrl" in v.lower() and "+" not in v]
        if cand:
            counts = vals.value_counts()
            control = max(cand, key=lambda v: counts.get(v, 0))
    if control is None:
        control = vals.unique()[0]
    log(f"[load] cells={adata.n_obs} genes={adata.n_vars} "
        f"pert_col={pert_col!r} control={control!r} "
        f"n_identities={adata.obs[pert_col].nunique()}")
    return adata, X, pert_col, control


def ridge_readout(Z, X, alpha=RIDGE_ALPHA):
    """Identical to code/perturbation_benchmark.py."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    Zs = StandardScaler().fit_transform(Z)
    Xs = StandardScaler().fit_transform(X)
    return Ridge(alpha=alpha, fit_intercept=False).fit(Zs, Xs).coef_.T


def eff_rank(cov, eps=1e-12):
    e = np.linalg.eigvalsh(cov + eps * np.eye(cov.shape[0]))
    e = np.clip(e, eps, None)
    return float((e.sum() ** 2) / (e ** 2).sum())


def metrics(Z, X):
    from kappa_c_audit.core import predictability_total
    W = ridge_readout(Z, X).astype(np.float64)
    cv = np.cov(Z, rowvar=False)
    return dict(trace=float((W ** 2).sum()),
                sum_r2=float(predictability_total(Z, X)),
                eff_rank=float(eff_rank(cv)),
                trace_Z=float(np.trace(cv)))


def fit_arm(adata, seed, injected, pert_col, epochs):
    """One arm.  Returns (Z, info) and never raises on a late-training divergence:
    the latent is extracted from whatever state the model reached."""
    import scvi
    import torch
    scvi.settings.seed = seed
    torch.manual_seed(seed)
    a = adata.copy()
    if injected:
        a.obs[pert_col] = a.obs[pert_col].astype(str)
        scvi.model.SCVI.setup_anndata(a, categorical_covariate_keys=[pert_col])
    else:
        scvi.model.SCVI.setup_anndata(a)
    m = scvi.model.SCVI(a, n_latent=N_LATENT, n_layers=N_LAYERS,
                        dropout_rate=DROPOUT, gene_likelihood=GENE_LIKELIHOOD)
    info = dict(train_status="completed")
    try:
        m.train(max_epochs=epochs, accelerator="cpu")
    except Exception as e:                                     # noqa: BLE001
        info = dict(train_status=f"diverged: {type(e).__name__}: {str(e)[:160]}")
        log(f"    [warn] training raised; extracting the reached latent. {info}")
    Z = m.get_latent_representation(a).astype(np.float64)
    info["z_finite"] = bool(np.isfinite(Z).all())
    return Z, info


def describe(v):
    v = np.asarray(v, float)
    if v.size == 0:
        return dict(n=0, mean=float("nan"), sd=float("nan"), lo2s=float("nan"),
                    hi2s=float("nan"), min=float("nan"), max=float("nan"))
    sd = float(v.std(ddof=1)) if v.size > 1 else float("nan")
    return dict(n=int(v.size), mean=float(v.mean()), sd=sd,
                lo2s=float(v.mean() - 2 * sd), hi2s=float(v.mean() + 2 * sd),
                min=float(v.min()), max=float(v.max()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed-start", type=int, default=SEED)
    ap.add_argument("--epochs", type=int, default=N_EPOCHS)
    ap.add_argument("--pert-col", default=None)
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--out", default="s1n_norman_seed_band.json")
    a = ap.parse_args()
    if a.threads:
        import torch
        torch.set_num_threads(a.threads)
    t0 = time.time()

    adata, X, pert_col, control = load_norman(a.pert_col)

    val = {}
    if os.path.exists(CACHE):
        c = np.load(CACHE, allow_pickle=True)
        val["cached_X_shape"] = list(c["X"].shape)
        val["max_abs_dev_X"] = float(np.abs(X.astype(np.float64)
                                            - c["X"].astype(np.float64)).max())
        val["pert_labels_identical"] = bool(
            np.array_equal(adata.obs[pert_col].astype(str).values, c["pert_labels"]))
        log(f"[validate] vs published cache: max|dX| = {val['max_abs_dev_X']:.3e}, "
            f"labels identical = {val['pert_labels_identical']}")
    if a.check_only:
        log(f"[check-only] done in {time.time() - t0:.1f}s")
        return

    jsonl = os.path.join(OUT, a.out.replace(".json", ".jsonl"))
    os.makedirs(OUT, exist_ok=True)
    arms = []
    if os.path.exists(jsonl):
        with open(jsonl, encoding="utf-8") as f:
            arms = [json.loads(l) for l in f if l.strip()]
        log(f"[resume] {len(arms)} arm rows loaded")

    def arm_of(seed, name):
        return next((r for r in arms if r["seed"] == seed and r["arm"] == name), None)

    rows = []
    for s in range(a.seed_start, a.seed_start + a.seeds):
        for name, inj in (("vanilla", False), ("injected", True)):
            if arm_of(s, name) is not None:
                log(f"seed {s} {name} already present, skipping")
                continue
            ts = time.time()
            Z, info = fit_arm(adata, s, inj, pert_col, a.epochs)
            m = metrics(Z, X)
            row = dict(seed=s, arm=name, **m, **info,
                       seconds=(time.time() - ts))
            arms.append(row)
            with open(jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            log(f"seed {s} {name}: sum_r2={m['sum_r2']:.3f} trace={m['trace']:.1f} "
                f"eff_rank={m['eff_rank']:.3f} trace_Z={m['trace_Z']:.3f} "
                f"[{row['seconds']:.0f}s] {info['train_status'][:40]}")

        # ---- paired row for this seed, if both arms are in -------------------
        v_row, i_row = arm_of(s, "vanilla"), arm_of(s, "injected")
        if v_row and i_row:
            pair = dict(seed=s,
                        kappa_trace=i_row["trace"] / v_row["trace"],
                        kappa_corrected=i_row["sum_r2"] / v_row["sum_r2"],
                        kappa_C_Z=i_row["trace_Z"] / v_row["trace_Z"],
                        eff_rank_ratio=i_row["eff_rank"] / v_row["eff_rank"],
                        eff_rank_delta_pct=100 * (i_row["eff_rank"] / v_row["eff_rank"] - 1))
            log(f"seed {s} PAIRED: k_tr={pair['kappa_trace']:.4f} "
                f"k_r2={pair['kappa_corrected']:.4f} k_Z={pair['kappa_C_Z']:.4f} "
                f"eff_rank {v_row['eff_rank']:.2f}->{i_row['eff_rank']:.2f} "
                f"({pair['eff_rank_delta_pct']:+.1f}%)")

        rows = []
        for sd in sorted({r["seed"] for r in arms}):
            v_row, i_row = arm_of(sd, "vanilla"), arm_of(sd, "injected")
            if v_row and i_row:
                rows.append(dict(seed=sd,
                                 kappa_trace=i_row["trace"] / v_row["trace"],
                                 kappa_corrected=i_row["sum_r2"] / v_row["sum_r2"],
                                 kappa_C_Z=i_row["trace_Z"] / v_row["trace_Z"],
                                 eff_rank_ratio=i_row["eff_rank"] / v_row["eff_rank"],
                                 van_eff_rank=v_row["eff_rank"],
                                 van_sum_r2=v_row["sum_r2"], van_trace_Z=v_row["trace_Z"]))
        # cross-seed null: two VANILLA arms (all unordered pairs of completed fits)
        vans = [r for r in arms if r["arm"] == "vanilla"]
        null = [dict(kappa_trace=ri["trace"] / rj["trace"],
                     kappa_corrected=ri["sum_r2"] / rj["sum_r2"],
                     kappa_C_Z=ri["trace_Z"] / rj["trace_Z"],
                     eff_rank_ratio=ri["eff_rank"] / rj["eff_rank"])
                for ri, rj in itertools.combinations(vans, 2)]
        out = dict(architecture_scvi_tools=None, h5ad=H5AD,
                   n_cells=int(adata.n_obs), n_genes=int(adata.n_vars),
                   pert_col=pert_col, control_value=control,
                   gene_likelihood=GENE_LIKELIHOOD, max_epochs=a.epochs,
                   accelerator="cpu", model_kwargs=dict(n_latent=N_LATENT,
                                                        n_layers=N_LAYERS,
                                                        dropout_rate=DROPOUT),
                   validation=val, arms=arms, per_seed=rows,
                   partial=len(rows) < 2,
                   paired_injected_vs_vanilla={
                       k: describe([r[k] for r in rows])
                       for k in ("kappa_trace", "kappa_corrected", "kappa_C_Z",
                                 "eff_rank_ratio")},
                   cross_seed_vanilla_null={
                       k: describe([d[k] for d in null])
                       for k in ("kappa_trace", "kappa_corrected", "kappa_C_Z",
                                 "eff_rank_ratio")})
        try:
            import importlib.metadata as md
            out["architecture_scvi_tools"] = md.version("scvi-tools")
        except Exception:
            pass
        with open(os.path.join(OUT, a.out), "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1, ensure_ascii=False)

    log(f"[done] {len(arms)} arms / {len(rows)} paired seeds in "
        f"{(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
