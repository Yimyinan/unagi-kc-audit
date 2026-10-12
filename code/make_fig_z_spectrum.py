# -*- coding: utf-8 -*-
"""
Figure: the GRN prior DOES compress the latent spectrum -- kappa-C was reading
the wrong space.

Panel (a) effective spectral rank of Z vs IDREM iteration
Panel (b) leading-axis share of Z vs IDREM iteration
Panel (c) normalised spectral entropy of Z vs IDREM iteration
Panel (d) effect size: prior - vanilla, per seed, vs IDREM iteration

Style is inherited from paper1_kc_diagnostic/figures (Arial, 7.5 pt base,
6.7 in canvas, 300 dpi).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import sys as _sys, os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
for _p in (_HERE, _os.path.join(_HERE, "..", "code"), _os.path.join(_HERE, "code")):
    if _os.path.isdir(_p) and _p not in _sys.path:
        _sys.path.insert(0, _p)
import figure_palette as fp
fp.apply_paper_style()
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(r"F:/UNAGI_Project")
OUT = ROOT / "outputs" / "positive_result"
CSV = OUT / "z_spectrum.csv"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7.5,
    "axes.labelsize": 8.0,
    "axes.titlesize": 8.5,
    "axes.titleweight": "bold",
    "axes.linewidth": 0.6,
    "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.0,
    "legend.fontsize": 6.2,
    "xtick.major.size": 2.4,
    "ytick.major.size": 2.4,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "lines.linewidth": 1.2,
    "lines.markersize": 3.4,
    "figure.dpi": 300,
    "savefig.dpi": 300,
})

GREY = "#888780"
SEED_COLOR = {123: "#0F6E56", 42: "#1D9E75", 456: "#5DCAA5"}
SEED_MARK = {123: "o", 42: "s", 456: "^"}
RED = "#C44E52"


def style_ax(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction="out", length=2.4, width=0.6)
    return ax


def main():
    df = pd.read_csv(CSV)
    van = df[df.role == "vanilla"].sort_values("iteration")
    pri = df[df.role == "prior"]

    fig, axes = plt.subplots(2, 2, figsize=(6.7, 5.4))
    fig.subplots_adjust(left=0.085, right=0.975, top=0.915, bottom=0.095,
                        wspace=0.30, hspace=0.42)

    panels = [
        ("eff_rank", "effective spectral rank of Z", "lower = more compressed", True),
        ("top1", "leading-axis share of Z (%)", "higher = more compressed", False),
        ("entropy", "spectral entropy of Z (normalised)", "lower = more compressed", True),
    ]

    for idx, (metric, ylab, hint, lower_better) in enumerate(panels):
        ax = axes.flat[idx]
        ax.plot(van.iteration, van[metric], ls="--", color=GREY, lw=1.2,
                marker="D", ms=3.2, label="vanilla (R14)", zorder=2)
        for seed, g in pri.groupby("seed", sort=False):
            g = g.sort_values("iteration")
            ax.plot(g.iteration, g[metric], ls="-", lw=1.3,
                    color=SEED_COLOR[int(seed)], marker=SEED_MARK[int(seed)],
                    label=f"prior, seed {int(seed)}", zorder=3)
        ax.set_xlabel("IDREM iteration")
        ax.set_ylabel(ylab)
        ax.set_xticks(range(5))
        ax.set_xticklabels([f"iter {i}" for i in range(5)])
        ax.set_title(f"{'abcd'[idx]}  {hint}", fontsize=7.6, loc="left")
        style_ax(ax)
        if idx == 0:
            ax.legend(frameon=False, fontsize=6.0, handlelength=1.3,
                      loc="upper right")
        # slope annotation
        rhos = [stats.spearmanr(g.sort_values("iteration").iteration,
                                g.sort_values("iteration")[metric]).correlation
                for _, g in pri.groupby("seed")]
        rv = stats.spearmanr(van.iteration, van[metric]).correlation
        ax.text(0.985, 0.035,
                f"prior slope $\\rho$ = {rhos[0]:+.2f}, {rhos[1]:+.2f}, {rhos[2]:+.2f}"
                f"\nvanilla $\\rho$ = {rv:+.2f}",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=5.9, color="#444441",
                bbox=dict(boxstyle="round,pad=0.28", fc="#FFFFFF",
                          ec="#D3D1C7", lw=0.5))

    # ---- panel (d): effect size per seed ----
    ax = axes.flat[3]
    vmap = van.set_index("iteration")["eff_rank"]
    for seed, g in pri.groupby("seed", sort=False):
        g = g.sort_values("iteration")
        delta = g.eff_rank.values - vmap.loc[g.iteration].values
        ax.plot(g.iteration, delta, ls="-", lw=1.3,
                color=SEED_COLOR[int(seed)], marker=SEED_MARK[int(seed)],
                label=f"seed {int(seed)}")
        ax.annotate(f"{delta[-1]:+.2f}",
                    xy=(g.iteration.values[-1], delta[-1]),
                    xytext=(4.5, delta[-1]), fontsize=5.9,
                    color=SEED_COLOR[int(seed)], va="center")
    ax.axhline(0, color=GREY, lw=0.7, ls=":")
    ax.set_xlabel("IDREM iteration")
    ax.set_ylabel(r"$\Delta$ effective rank (prior $-$ vanilla)")
    ax.set_xticks(range(5))
    ax.set_xticklabels([f"iter {i}" for i in range(5)])
    ax.set_xlim(-0.25, 5.9)
    ax.set_title("d  effect accumulates with prior injection", fontsize=7.6,
                 loc="left")
    style_ax(ax)
    ax.text(0.5, 0.93, "3/3 seeds below zero at iter 4",
            transform=ax.transAxes, ha="center", va="top", fontsize=5.9,
            color="#0F6E56")

    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_z_spectrum_compression.{ext}",
                    bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_z_spectrum_compression.{png,pdf} ->", OUT)


if __name__ == "__main__":
    main()
