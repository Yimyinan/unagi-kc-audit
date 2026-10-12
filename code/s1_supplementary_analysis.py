"""S1 supplementary analysis -- **grid A only, DEPRECATED**.

Grid A (contrastvae, P_on/P_off/P_wrong) turned out to be trained on a fully
collapsed latent (eff_rank 2.57/64, KL = 0), so its audit rows are uninformative
about sensitivity.  See S1_LIVE_PATH_RESULTS.md sections 1 and 2.  The live
analysis is now:

    s1c_trace_vs_r2_diagnostic.py   tr-form vs R2-form divergence + condition number
    s1d_whitened_check.py           whitening restores agreement (causal confirmation)
    s1e_s1b_reanalysis.py           S1b grid re-scored with the correct R2 form

Kept for the record: it still computes the dose metrics (d_readout, d_W, d_cov)
that established the prior in grid A changed the GRN matrix by 30x while leaving
the latent and the readout untouched.

Metrics per pair (a, b)
    d_readout  ||Wp - Wv||_F / max(...)   between the fitted Ridge readouts
    d_W        ||Wa - Wb||_F / max(...)   between the GRN matrices
    d_cov      1 - cos(vec cov(Za), vec cov(Zb))
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
from kappa_c_audit import (audit_from_readout, eff_rank,  # noqa: E402
                           kappa_C_Z, ridge_readout)

ARMS = ["P_on", "P_off", "P_wrong"]
SEEDS = [42, 123, 456]
N_CELLS = 15000


def load_runs():
    runs = {}
    for arm in ARMS:
        for s in SEEDS:
            p = os.path.join(OUT, f"{arm}_s{s}.npz")
            if os.path.exists(p):
                d = np.load(p)
                runs[(arm, s)] = dict(Z=d["Z"].astype(np.float64),
                                      W=d["W"].astype(np.float64))
    return runs


def d_W(Wa, Wb):
    den = max(np.linalg.norm(Wa), np.linalg.norm(Wb))
    return float(np.linalg.norm(Wa - Wb) / den) if den else float("nan")


def d_cov(Za, Zb):
    Ca, Cb = np.cov(Za, rowvar=False).ravel(), np.cov(Zb, rowvar=False).ravel()
    n = np.linalg.norm(Ca) * np.linalg.norm(Cb)
    return float(1.0 - (Ca @ Cb) / n) if n else float("nan")


def reconstruct_X():
    import anndata as ad
    from scipy.sparse import issparse
    a = ad.read_h5ad(r"F:/UNAGI_Project/UNAGI/result/R14/R14_hemato_vanilla/"
                     r"iter_4/stagedata/dataset.h5ad", backed="r")
    idx = np.random.default_rng(0).choice(a.n_obs, size=N_CELLS, replace=False)
    idx.sort()
    Xall = a.X[:]
    Xs = Xall[idx]
    X = (Xs.toarray() if issparse(Xs) else np.asarray(Xs)).astype(np.float32)
    del a, Xall, Xs
    return np.log1p(X).astype(np.float64)


def classify_group(aa, sa, bb, sb):
    if aa == "P_off" and bb == "P_off":
        return "null (P_off vs P_off, cross-seed)"
    if sa == sb and "P_off" in (aa, bb):
        prior = bb if aa == "P_off" else aa
        return f"{prior} vs P_off (same seed)"
    return "prior vs P_off (cross-seed)"


def main():
    runs = load_runs()
    keys = sorted(runs)
    if not keys:
        print("no runs found in", OUT)
        return
    print(f"loaded {len(keys)} runs: {keys}\n")
    X = reconstruct_X()
    print(f"X reconstructed: {X.shape}\n")

    readouts = {k: ridge_readout(v["Z"], X) for k, v in runs.items()}
    print("fitted one Ridge readout per run (alpha=1, per-column standardised)\n")

    rows = []
    for (aa, sa), (bb, sb) in itertools.combinations(keys, 2):
        base_key, prior_key = ((aa, sa), (bb, sb)) if aa == "P_off" else ((bb, sb), (aa, sa))
        Wb, Wp = readouts[base_key], readouts[prior_key]
        rb = audit_from_readout(Wb, Wp)
        runs_base, runs_prior = runs[base_key], runs[prior_key]
        rows.append(dict(
            group=classify_group(aa, sa, bb, sb),
            pair=f"{aa}_s{sa}/{bb}_s{sb}",
            d_readout=float(np.linalg.norm(Wp - Wb)
                            / max(np.linalg.norm(Wb), np.linalg.norm(Wp))),
            d_W=d_W(runs[(aa, sa)]["W"], runs[(bb, sb)]["W"]),
            d_cov=d_cov(runs[(aa, sa)]["Z"], runs[(bb, sb)]["Z"]),
            kappa_C_W=rb["kappa_C_W"],
            verdict=rb["verdict"],
            kappa_C_Z=float(kappa_C_Z(runs_base["Z"], runs_prior["Z"])),
            eff_rank_base=float(eff_rank(np.cov(runs_base["Z"], rowvar=False))),
            eff_rank_prior=float(eff_rank(np.cov(runs_prior["Z"], rowvar=False))),
        ))
        r = rows[-1]
        r["d_eff_rank_pct"] = 100 * (r["eff_rank_prior"] / r["eff_rank_base"] - 1)

    hdr = (f"{'group':<34}{'pair':<24}{'d_read':>8}{'d_W':>8}{'d_cov':>9}"
           f"{'kCW':>8}{'dEffR%':>9}")
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(rows, key=lambda x: (x["group"], x["pair"])):
        print(f"{r['group']:<34}{r['pair']:<24}{r['d_readout']:>8.3f}"
              f"{r['d_W']:>8.3f}{r['d_cov']:>9.4f}"
              f"{r['kappa_C_W']:>8.3f}{r['d_eff_rank_pct']:>9.2f}")

    print("\n=== grouped summary ===")
    groups = {}
    for r in rows:
        groups.setdefault(r["group"], []).append(r)
    summary = {}
    for g, rs in sorted(groups.items()):
        s = {}
        print(f"\n{g}   (n={len(rs)})")
        for m in ("d_W", "d_cov", "kappa_C_W", "d_eff_rank_pct"):
            v = np.array([x[m] for x in rs], float)
            sd = float(v.std(ddof=1)) if len(v) > 1 else None
            s[m] = dict(mean=float(v.mean()), sd=sd,
                        vals=[round(float(x), 4) for x in v])
            print(f"   {m:<14} {v.mean():>10.4f}"
                  + (f" +- {sd:.4f}" if sd is not None else "        ")
                  + f"   {s[m]['vals']}")
        summary[g] = dict(n=len(rs), **s)

    with open(os.path.join(OUT, "s1_supplementary.json"), "w") as f:
        json.dump(dict(rows=rows, summary=summary), f, indent=1)
    print("\nsaved ->", os.path.join(OUT, "s1_supplementary.json"))


if __name__ == "__main__":
    main()
