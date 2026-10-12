# -*- coding: utf-8 -*-
"""S1n -- family-matched null for the Norman-2019 scVI pair (section 2.9.2).

Why
---
Section 2.9.2 reports the perturbation-conditioned scVI pair (kappa-C(W)=0.905
corrected, kappa-C(Z)=0.746, effective rank 8.80 -> 6.15, -30.1%) as a single run.
The pbmc3k control showed why that is not enough: a reported pair there turned out to
be two draws of one configuration, so the relevant question is whether the published
values exceed the scatter of two independent fits of the same configuration.

Design (and why the injected arm is not retrained here)
------------------------------------------------------
The published run's own latents survive in ``data/_norman_latents.npz`` (X, Zv, Zp,
labels), so the published pair's statistics can be recomputed exactly; what is missing
is the *null*, and the null only needs independent BASELINE fits -- unordered pairs of
vanilla latents, exactly as in the pbmc3k analysis.  Retraining the injected arm is
therefore unnecessary, which matters because it is unstable here: with this stack
(torch 2.14 + scvi-tools 1.5.0) the ``gene_likelihood="normal"`` fit raises
"Expected parameter scale ... GreaterThan(0.0)" before completing 200 epochs, in either
arm, depending on the seed.  The published run completed and cached full latents, so
that divergence is an environment difference -- the data side is identical
(max|dX| = 0 and bit-identical perturbation labels versus the cache).

Two nulls are reported:
  null_mine          unordered pairs of THIS environment's vanilla fits (seed scatter)
  null_with_published the same, with the published cached vanilla as one more member
                     (seed scatter + environment difference)

Cross-check: the metrics function applied to the cached latents must reproduce the
published trace kappa-C(W) = 1.2192, which validates the protocol end to end.

Writes : outputs/positive_result/s1n_norman_null.json (+ .jsonl, one line per arm)
Usage  : python code/s1n_norman_null.py --seeds 3        # trains missing vanilla arms
         python code/s1n_norman_null.py --analyse-only
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


def fit_vanilla(adata, seed, epochs, chunked=False):
    """Baseline arm.

    ``chunked=False`` reproduces the published protocol (one 200-epoch call).  Chunked
    training exists only as a fallback for the divergence case -- scvi refuses to serve
    a latent whose trainer never completed a fit -- and it is NOT equivalent: at seed 43
    the two protocols differ by 10% in readout predictability, 19% in effective rank and
    142% in the latent trace, so chunked arms are excluded from the null."""
    import scvi
    import torch
    scvi.settings.seed = seed
    torch.manual_seed(seed)
    a = adata.copy()
    scvi.model.SCVI.setup_anndata(a)
    m = scvi.model.SCVI(a, n_latent=N_LATENT, n_layers=N_LAYERS,
                        dropout_rate=DROPOUT, gene_likelihood=GENE_LIKELIHOOD)
    info = dict(train_status="completed", epochs_done=0)
    CH = 50
    info["protocol"] = "chunked" if chunked else "single"
    try:
        if chunked:
            for _ in range(max(1, epochs // CH)):
                m.train(max_epochs=CH, accelerator="cpu")
                info["epochs_done"] += CH
                if info["epochs_done"] >= epochs:
                    break
        else:
            m.train(max_epochs=epochs, accelerator="cpu")
            info["epochs_done"] = epochs
    except Exception as e:                                     # noqa: BLE001
        info["train_status"] = (f"diverged after {info['epochs_done']} epochs: "
                                f"{type(e).__name__}: {str(e)[:140]}")
        log(f"    [warn] {info['train_status']}")
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


KEYS = ("kappa_trace", "kappa_corrected", "kappa_C_Z", "eff_rank_ratio")


def pair_stats(mi, mj):
    return dict(kappa_trace=mi["trace"] / mj["trace"],
                kappa_corrected=mi["sum_r2"] / mj["sum_r2"],
                kappa_C_Z=mi["trace_Z"] / mj["trace_Z"],
                eff_rank_ratio=mi["eff_rank"] / mj["eff_rank"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed-start", type=int, default=SEED)
    ap.add_argument("--epochs", type=int, default=N_EPOCHS)
    ap.add_argument("--pert-col", default=None)
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--analyse-only", action="store_true")
    ap.add_argument("--protocol", default="single", choices=["single", "chunked"],
                    help="single = the published 200-epoch call (default); chunked is "
                         "a divergence fallback and is excluded from the null")
    ap.add_argument("--out", default="s1n_norman_null.json")
    a = ap.parse_args()
    t0 = time.time()
    jsonl = os.path.join(OUT, a.out.replace(".json", ".jsonl"))
    os.makedirs(OUT, exist_ok=True)

    adata = X = pert_col = control = None
    val = {}
    if not a.analyse_only:
        adata, X, pert_col, control = load_norman(a.pert_col)
        if os.path.exists(CACHE):
            c = np.load(CACHE, allow_pickle=True)
            val = dict(cached_X_shape=list(c["X"].shape),
                       max_abs_dev_X=float(np.abs(X.astype(np.float64)
                                                  - c["X"].astype(np.float64)).max()),
                       pert_labels_identical=bool(np.array_equal(
                           adata.obs[pert_col].astype(str).values, c["pert_labels"])))
            log(f"[validate] vs published cache: max|dX| = {val['max_abs_dev_X']:.3e}, "
                f"labels identical = {val['pert_labels_identical']}")
        if a.check_only:
            log(f"[check-only] done in {time.time() - t0:.1f}s")
            return

    arms = []
    if os.path.exists(jsonl):
        with open(jsonl, encoding="utf-8") as f:
            arms = [json.loads(l) for l in f if l.strip()]
        log(f"[resume] {len(arms)} arm rows loaded")

    if not a.analyse_only:
        for s in range(a.seed_start, a.seed_start + a.seeds):
            if any(r["seed"] == s and r["arm"] == "vanilla"
                   and r.get("protocol", "single") == a.protocol for r in arms):
                log(f"seed {s} vanilla already present, skipping")
                continue
            ts = time.time()
            Z, info = fit_vanilla(adata, s, a.epochs, chunked=(a.protocol == "chunked"))
            m = metrics(Z, X)
            row = dict(seed=s, arm="vanilla", **m, **info, seconds=time.time() - ts)
            arms.append(row)
            with open(jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            log(f"seed {s} vanilla: sum_r2={m['sum_r2']:.3f} trace={m['trace']:.1f} "
                f"eff_rank={m['eff_rank']:.3f} trace_Z={m['trace_Z']:.3f} "
                f"[{row['seconds']:.0f}s] {info['train_status'][:50]}")

    # ---- published pair, recomputed from its own cached latents ------------
    published = None
    if os.path.exists(CACHE):
        c = np.load(CACHE, allow_pickle=True)
        Xc = c["X"].astype(np.float64)
        mv = metrics(c["Zv"].astype(np.float64), Xc)
        mp = metrics(c["Zp"].astype(np.float64), Xc)
        published = dict(source="data/_norman_latents.npz (published run)",
                         van=mv, inj=mp,
                         eff_rank_vanilla=mv["eff_rank"],
                         eff_rank_injected=mp["eff_rank"],
                         **pair_stats(mp, mv))
        log(f"[published] recomputed from the cache: trace kappa-C(W)="
            f"{published['kappa_trace']:.4f} (published json says 1.2192), "
            f"corrected={published['kappa_corrected']:.4f}, "
            f"kappa-C(Z)={published['kappa_C_Z']:.4f}, "
            f"eff_rank {published['eff_rank_vanilla']:.2f}->"
            f"{published['eff_rank_injected']:.2f}")

    vans = [r for r in arms if r["arm"] == "vanilla"
            and r.get("protocol", "single") == a.protocol]
    null_mine = [pair_stats(ri, rj) for ri, rj in itertools.combinations(vans, 2)]
    members = list(vans) + ([published["van"]] if published else [])
    null_with_pub = [pair_stats(ri, rj) for ri, rj in itertools.combinations(members, 2)]

    out = dict(design=("null = unordered pairs of independent BASELINE fits; the "
                       "published pair's own latents are recomputed from the cache, so "
                       "the injected arm is not retrained (it diverges in this stack)"),
               h5ad=H5AD, gene_likelihood=GENE_LIKELIHOOD, max_epochs=a.epochs,
               accelerator="cpu", model_kwargs=dict(n_latent=N_LATENT,
                                                   n_layers=N_LAYERS,
                                                   dropout_rate=DROPOUT),
               validation=val, arms=arms, protocol=a.protocol,
               my_vanilla_eff_rank=[r["eff_rank"] for r in vans],
               published=published,
               null_mine={k: describe([d[k] for d in null_mine]) for k in KEYS},
               null_with_published={k: describe([d[k] for d in null_with_pub])
                                    for k in KEYS},
               published_z={})
    if published:
        for src, name in ((null_mine, "vs_null_mine"),
                          (null_with_pub, "vs_null_with_published")):
            band = {k: describe([d[k] for d in src]) for k in KEYS}
            out["published_z"][name] = {
                k: dict(published=published[k], null_mean=band[k]["mean"],
                        null_sd=band[k]["sd"], band=[band[k]["lo2s"], band[k]["hi2s"]],
                        z=(published[k] - band[k]["mean"]) / band[k]["sd"],
                        verdict=("inside" if band[k]["lo2s"] <= published[k]
                                 <= band[k]["hi2s"] else "OUTSIDE"))
                for k in KEYS}
    with open(os.path.join(OUT, a.out), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)

    log()
    log(f"[done] {len(vans)} own vanilla fits, "
        f"{len(null_mine)} own pairs, {len(null_with_pub)} pairs incl. published")
    for name, table in out["published_z"].items():
        log(f"--- published values {name} (n_pairs={out['null_' + ('mine' if 'mine' in name else 'with_published')]['kappa_trace']['n']})")
        for k, v in table.items():
            log(f"    {k:<16} published {v['published']:.4f} vs null "
                f"{v['null_mean']:.4f} +- {v['null_sd']:.4f} -> z {v['z']:+.2f}  {v['verdict']}")


if __name__ == "__main__":
    main()
