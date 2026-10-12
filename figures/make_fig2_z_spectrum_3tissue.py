# -*- coding: utf-8 -*-
r"""make_fig2_z_spectrum_3tissue.py -- rebuild the tissue-dependent latent-spectrum
figure from the recovered per-iteration data.

WHY THIS SCRIPT EXISTS
----------------------
figures/fig2_z_spectrum_3tissue.pdf was for a long time the one main figure with no
producing script in this tree: the generator (code/scan_z_spectrum_3tissue.py) and
its output both lived in the pre-migration working copy at
    F:/UNAGI_Project/outputs/positive_result/
It was recovered from there, so the figure can now be regenerated in the current
house style (Okabe-Ito palette, one stroke/font ladder, uppercase panel letters,
canvas = printed width).

DATA
----
data/z_spectrum_all.csv -- one row per (run, IDREM iteration), with the effective
rank and participation ratio of the encoder output Z, plus tissue / role / seed.
183 runs, iterations 0-4.
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "code"))
import figure_palette as fp  # noqa: E402

DATA = os.path.join(ROOT, "data", "z_spectrum_all.csv")
OUT = os.path.join(HERE, "fig2_z_spectrum_3tissue.pdf")
OUT_PNG = os.path.join(HERE, "fig2_z_spectrum_3tissue.png")

TISSUES = [
    ("hemo", "Hematopoietic", fp.C_HEMA, "$n=23$; 29,985 cells"),
    ("neural", "Neural crest", fp.C_NEURAL, "$n=11$; 21,078 cells"),
    ("lung", "Lung IPF", fp.C_LUNG, "$n=2$; 19,999 cells"),
]


def main():
    df = pd.read_csv(DATA)
    df = df[df["eff_rank"].notna()]
    fp.apply_paper_style()
    # Canvas width = printed width (\textwidth = 6.6929 in).
    fig, axes = plt.subplots(1, 3, figsize=(6.6929, 2.62))

    for ax, (key, label, colour, note), letter in zip(axes, TISSUES, "ABC"):
        sub = df[df["tissue"] == key]
        is_van = sub["role"].astype(str).str.lower().str.contains("vanilla")
        # vanilla controls: thick neutral lines
        for _, g in sub[is_van].groupby("run"):
            g = g.sort_values("iteration")
            ax.plot(g["iteration"], g["eff_rank"], "-", color=fp.ARM_VANILLA,
                    lw=1.5, alpha=0.9, zorder=4, solid_capstyle="round")
        # prior / ablation arms: thin coloured lines
        for _, g in sub[~is_van].groupby("run"):
            g = g.sort_values("iteration")
            ax.plot(g["iteration"], g["eff_rank"], "-", color=colour,
                    lw=0.75, alpha=0.55, zorder=3)
        ax.set_xlabel("IDREM iteration")
        if ax is axes[0]:
            ax.set_ylabel("effective rank of $Z$")
        ax.set_xticks([0, 1, 2, 3, 4])
        ax.set_title("(%s) %s" % (letter, label), loc="left")
        ax.text(0.03, 0.06, note, transform=ax.transAxes, fontsize=6.0,
                color=fp.MUTED, ha="left", va="bottom")

    # one shared y-range so the three tissues are comparable
    lo = float(df["eff_rank"].min()) - 1.0
    hi = float(df["eff_rank"].max()) + 1.0
    for ax in axes:
        ax.set_ylim(lo, hi)

    # congruence gradient (Jaccard 0.75 -> 0.54 -> 0.33), stated in the caption too
    for ax, j in zip(axes, [0.75, 0.54, 0.33]):
        ax.text(0.97, 0.94, "congruence $J=%.2f$" % j, transform=ax.transAxes,
                fontsize=6.0, color=fp.MUTED, ha="right", va="top")

    axes[0].legend(handles=[
        plt.Line2D([], [], color=fp.ARM_VANILLA, lw=1.5, label="vanilla control"),
        plt.Line2D([], [], color=fp.C_HEMA, lw=0.75, label="prior-injected arm"),
    ], loc="upper left", frameon=False, fontsize=6.0, handlelength=1.3)

    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("[saved]", OUT)
    print("[saved]", OUT_PNG)
    for key, label, _c, _n in TISSUES:
        s = df[df["tissue"] == key]
        print("  %-16s runs=%-3d mean eff_rank iter0=%.2f iter4=%.2f"
              % (label, s["run"].nunique(),
                 s[s.iteration == 0]["eff_rank"].mean(),
                 s[s.iteration == s.iteration.max()]["eff_rank"].mean()))


if __name__ == "__main__":
    main()
