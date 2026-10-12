"""Re-analyse the S1b grid with the *correct* statistic.

s1c/s1d established that the released ``kappa_C_W`` returns tr(W^T W) of the
Ridge loadings, which equals the manuscript's named quantity
``sum_g R^2_g(prior) / sum_g R^2_g(vanilla)`` only when Corr(Z) ~ I.  For real
trained latents Corr(Z) is ill-conditioned (cond ~2.4e3), so the tr form is
noise-dominated -- which is why S1b's ``d_readout`` was ~1.33 for *every* pair,
including the cross-seed null.

So S1b's null result is not evidence about the audit's sensitivity: it was
measured with the wrong statistic.  This script recomputes the same trained
models with

    tr form      (Wp**2).sum() / (Wv**2).sum()          <- as released
    R2 form      predictability_ratio(Zv, X, Zp, X)     <- as documented

and asks the real question: does either statistic show a dose-ordered response
to a prior that is verifiably wired into the reconstruction path?

Usage: python s1e_s1b_reanalysis.py
"""
from __future__ import annotations

import itertools
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER1 = os.path.dirname(HERE)
OUT = r"F:/UNAGI_Project/outputs/positive_result/s1_live_path"
sys.path.insert(0, os.path.join(PAPER1, "kappa_c_audit"))
from kappa_c_audit import eff_rank, predictability_ratio, ridge_readout  # noqa: E402

SEEDS = [42, 123, 456]
ARMS = ["wp_b0", "wp_b2", "wp_b10", "wp_perm2"]


def reconstruct_X():
    import anndata as ad
    from scipy.sparse import issparse
    a = ad.read_h5ad(r"F:/UNAGI_Project/UNAGI/result/R14/R14_hemato_vanilla/"
                     r"iter_4/stagedata/dataset.h5ad", backed="r")
    idx = np.random.default_rng(0).choice(a.n_obs, size=15000, replace=False)
    idx.sort()
    Xall = a.X[:]
    Xs = Xall[idx]
    X = (Xs.toarray() if issparse(Xs) else np.asarray(Xs)).astype(np.float32)
    del a, Xall, Xs
    return np.log1p(X).astype(np.float64)


def main():
    runs = {}
    for arm in ARMS:
        for s in SEEDS:
            p = os.path.join(OUT, f"{arm}_s{s}.npz")
            if os.path.exists(p):
                runs[(arm, s)] = np.load(p)["Z"].astype(np.float64)
    print(f"loaded {len(runs)} runs\n")
    X = reconstruct_X()

    # one Ridge readout per run (for the tr form)
    read = {k: ridge_readout(v, X) for k, v in runs.items()}
    print("fitted one Ridge readout per run\n")

    def stats(base_key, prior_key):
        Zv, Zp = runs[base_key], runs[prior_key]
        tr = float((read[prior_key] ** 2).sum() / (read[base_key] ** 2).sum())
        r2 = float(predictability_ratio(Zv, X, Zp, X))
        er_v = eff_rank(np.cov(Zv, rowvar=False))
        er_p = eff_rank(np.cov(Zp, rowvar=False))
        return tr, r2, 100 * (er_p / er_v - 1)

    rows = []
    for a, b in itertools.combinations(SEEDS, 2):
        tr, r2, der = stats(("wp_b0", a), ("wp_b0", b))
        rows.append(dict(group="null (b0 vs b0, cross-seed)", seed=f"{a}/{b}",
                         tr=tr, r2=r2, d_eff_rank=der))
    for arm in ["wp_b2", "wp_b10", "wp_perm2"]:
        for s in SEEDS:
            if (arm, s) not in runs:
                continue
            tr, r2, der = stats(("wp_b0", s), (arm, s))
            rows.append(dict(group=f"{arm} vs b0 (same seed)", seed=str(s),
                             tr=tr, r2=r2, d_eff_rank=der))

    hdr = f"{'group':<34}{'seed':<9}{'tr-form':>10}{'R2-form':>10}{'dEffR%':>9}"
    print(hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r['group']:<34}{r['seed']:<9}{r['tr']:>10.4f}{r['r2']:>10.4f}"
              f"{r['d_eff_rank']:>9.2f}")

    print("\n=== grouped (mean +- sd) ===")
    g = {}
    for r in rows:
        g.setdefault(r["group"], []).append(r)
    summary = {}
    for k, rs in g.items():
        s = {}
        msg = f"\n{k}  (n={len(rs)})"
        for m in ("tr", "r2", "d_eff_rank"):
            v = np.array([x[m] for x in rs], float)
            sd = float(v.std(ddof=1)) if len(v) > 1 else None
            s[m] = dict(mean=float(v.mean()), sd=sd,
                        vals=[round(float(x), 4) for x in v])
            msg += (f"\n   {m:<12} {v.mean():>9.4f}"
                    + (f" +- {sd:.4f}" if sd is not None else "        ")
                    + f"  {s[m]['vals']}")
        summary[k] = dict(n=len(rs), **s)
        print(msg)

    # dose ordering test on the R2 form
    order = ["wp_b2", "wp_b10"]
    print("\n=== dose ordering (R2 form), per seed ===")
    for s in SEEDS:
        row = {}
        for arm in order:
            if (arm, s) in runs:
                row[arm] = float(predictability_ratio(
                    runs[("wp_b0", s)], X, runs[(arm, s)], X))
        if len(row) == 2:
            print(f"   seed {s}: b2 {row['wp_b2']:.4f} -> b10 {row['wp_b10']:.4f}"
                  f"   monotone {'YES' if row['wp_b10'] < row['wp_b2'] else 'no'}")

    with open(os.path.join(OUT, "s1e_reanalysis.json"), "w") as f:
        json.dump(dict(rows=rows, summary=summary), f, indent=1)
    print("\nsaved ->", os.path.join(OUT, "s1e_reanalysis.json"))


if __name__ == "__main__":
    main()
