#!/usr/bin/env python3
"""
Build the IDREM iteration trajectory table: kappa(t) and C(t) for every
provenance-verified configuration, plus the per-dimension geometry needed for
the "latent-space resolution" figure.

Provenance-verified pairs (terminal iteration reproduces kappa_complete_audit.json)
----------------------------------------------------------------------------------
  neural : vanilla R0_seed42   vs prior R1_seed42        (5 iters, 11,932 genes)
  lung   : vanilla R10         vs prior R9               (5 iters, 41,995 genes)
  hemo   : vanilla R14         vs priors R12_s123, R12_s456, R13_s123, R13_s456
                                      (5 iters, 20,466 genes)
                                      R84_s123, R111_s42 (4 iters)

The vanilla denominator is time-matched: kappa(t) = tr_P(t) / tr_V(t) using the
vanilla readout at the SAME IDREM iteration, not the terminal vanilla value.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

OUTDIR = Path(r"F:/UNAGI_Project/outputs/kc_trajectory")
DEC = Path(r"F:/UNAGI_Project/UNAGI/decoder_out")

VANILLA = {"neural": "R0_seed42", "lung": "R10", "hemo": "R14"}

PAIRS = [
    # (prior config, tissue, label, prior edges)
    ("R1_seed42",    "neural", "R1 full PG-UNAGI (C+M+T)", None),
    ("R9",           "lung",   "R9 IPF disease prior",     None),
    ("R13_seed123",  "hemo",   "R13 full blood prior",     325),
    ("R13_seed456",  "hemo",   "R13 full blood prior",     325),
    ("R12_seed123",  "hemo",   "R12 GATA2+GFI1B+PAX5",      55),
    ("R12_seed456",  "hemo",   "R12 GATA2+GFI1B+PAX5",      55),
    ("R84_seed123",  "hemo",   "R84 erythroid",             84),
    ("R111_seed42",  "hemo",   "R111 erythroid+temporal",  111),
]

# extra vanilla seeds (no matched prior trajectory, but show denominator drift)
EXTRA_VANILLA = ["R0_seed123", "R0_seed456", "PCA_baseline_hema"]


def load(cfg: str) -> pd.DataFrame:
    d = pd.read_csv(OUTDIR / "trajectory.csv")
    return d[d.config == cfg].sort_values("iteration")


def main():
    traj = pd.read_csv(OUTDIR / "trajectory.csv")

    def get(cfg, it, col="tr"):
        s = traj[(traj.config == cfg) & (traj.iteration == it)]
        return float(s[col].iloc[0]) if len(s) else np.nan

    rows = []
    for prior, tissue, label, edges in PAIRS:
        van = VANILLA[tissue]
        its = sorted(traj[traj.config == prior].iteration.dropna().astype(int))
        for it in its:
            trp, trv = get(prior, it), get(van, it)
            if not np.isfinite(trp) or not np.isfinite(trv):
                continue
            kappa = trp / trv
            rows.append(dict(
                tissue=tissue, prior=prior, vanilla=van, label=label,
                prior_edges=edges, iteration=int(it),
                tr_P=trp, tr_V=trv, kappa=kappa,
                C=(trv - trp) / trv * 100.0,
                eff_rank_P=get(prior, it, "eff_rank"),
                eff_rank_V=get(van, it, "eff_rank"),
                top1_P=get(prior, it, "top1_share"),
                top1_V=get(van, it, "top1_share"),
                ent_P=get(prior, it, "entropy"),
                ent_V=get(van, it, "entropy"),
                dim55_P=get(prior, it, "dim55_pct"),
                dim3_P=get(prior, it, "dim3_pct"),
                dim1_P=get(prior, it, "dim1_pct"),
                dim55_V=get(van, it, "dim55_pct"),
                dim3_V=get(van, it, "dim3_pct"),
            ))
    df = pd.DataFrame(rows)
    df.to_csv(OUTDIR / "kappa_trajectory.csv", index=False)

    # ---------------- headline statistics ----------------
    lines = ["# IDREM iteration trajectory: kappa(t) and C(t)", "",
             "Vanilla denominator is **time-matched** at the same IDREM iteration.",
             "Only provenance-verified configurations are included",
             "(see `provenance.md`).", ""]

    lines.append("## Per-configuration trajectory")
    lines.append("")
    lines.append("| tissue | prior | edges | iter | tr_P | tr_V | kappa | C (%) | eff_rank_P | eff_rank_V |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for _, r in df.iterrows():
        e = "-" if pd.isna(r.prior_edges) else int(r.prior_edges)
        lines.append(f"| {r.tissue} | {r['prior']} | {e} | {int(r.iteration)} | "
                     f"{r.tr_P:,.1f} | {r.tr_V:,.1f} | {r.kappa:.3f} | {r.C:.1f} | "
                     f"{r.eff_rank_P:.2f} | {r.eff_rank_V:.2f} |")

    # ---------------- the 0/N extension ----------------
    n = len(df)
    n50 = int((df.C > 50).sum())
    cmax = df.loc[df.C.idxmax()]
    lines += ["", "## Does any configuration reach C > 50% at ANY iteration?", "",
              f"- Trajectory observations (config x iteration), provenance-verified: **{n}**",
              f"- Observations with C > 50%: **{n50}**",
              f"- Maximum C observed: **{cmax.C:.1f}%** "
              f"({cmax['prior']}, {cmax.tissue}, iteration {int(cmax.iteration)})", ""]

    # ---------------- non-stationarity ----------------
    lines += ["", "## Non-stationarity of the terminal snapshot", "",
              "For each configuration, the spread of C across IDREM iterations on the",
              "**same model with the same prior**:", "",
              "| tissue | prior | C at iter0..4 | mean | SD | range | terminal |",
              "|---|---|---|---|---|---|---|"]
    for (tissue, prior), g in df.groupby(["tissue", "prior"], sort=False):
        g = g.sort_values("iteration")
        cs = g.C.values
        lines.append(f"| {tissue} | {prior} | " +
                     ", ".join(f"{c:.1f}" for c in cs) +
                     f" | {cs.mean():.1f} | {cs.std(ddof=1):.1f} | "
                     f"{cs.max()-cs.min():.1f} | {cs[-1]:.1f} |")

    lines += ["", "## Vanilla denominator drift", "",
              "tr_V(t) is not constant, so a terminal-only kappa mixes prior effect",
              "with vanilla drift:", "",
              "| tissue | vanilla | tr_V at iter0..4 | max/min |", "|---|---|---|---|"]
    for tissue, van in VANILLA.items():
        v = df[df.vanilla == van].drop_duplicates("iteration").sort_values("iteration")
        if not len(v):
            continue
        vals = v.tr_V.values
        lines.append(f"| {tissue} | {van} | " + ", ".join(f"{x:,.0f}" for x in vals) +
                     f" | {vals.max()/vals.min():.2f}x |")
    for van in EXTRA_VANILLA:
        s = traj[traj.config == van].sort_values("iteration")
        if not len(s):
            continue
        vals = s.tr.values
        lines.append(f"| (extra) | {van} | " + ", ".join(f"{x:,.0f}" for x in vals) +
                     f" | {vals.max()/vals.min():.2f}x |")

    (OUTDIR / "kappa_trajectory.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:8]))
    print("\n".join(lines[-24:]))


if __name__ == "__main__":
    main()
