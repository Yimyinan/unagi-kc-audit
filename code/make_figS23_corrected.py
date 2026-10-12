# -*- coding: utf-8 -*-
"""Supplementary Figure S23 (regenerated 2026-09-17): the matched-scale purity
contrast dissolves under the corrected readout statistic.

The original S23 asserted a "purity paradox": R111 (5 pure erythroid TFs) at
C = -71.4% against R107 (5 mixed-lineage TFs) at C = +0.7%, a 72-percentage-point
gap said to implicate prior content over lineage purity.  That gap was produced by
the trace form of the readout.  In the corrected predictability form

    R111_s42  kappa = 1.714 -> 0.9910   (C = -71.4% -> +0.90%)
    R107_s42  kappa = 0.993 -> 0.9924   (C =  +0.7% -> +0.76%)

i.e. the contrast collapses from 72 pp to 0.14 pp.  The figure therefore now
documents the withdrawal rather than the paradox.

Panel (a) matched-scale pair, both statistics.
Panel (b) all 33 configurations, trace vs corrected kappa, against the
          band-independent lock threshold kappa < 0.5.

Reads  : outputs/positive_result/s1f_0of33_recheck.json
Writes : figures/figS23_purity_corrected.pdf / .png
Usage  : python code/make_figS23_corrected.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import figure_palette as fp  # noqa: E402

S1F = os.path.abspath(os.path.join(HERE, "..", "..", "..", "outputs",
                                   "positive_result", "s1f_0of33_recheck.json"))
FIG = os.path.join(HERE, "..", "figures", "figS23_purity_corrected")

fp.apply_paper_style()
rows = [r for r in json.load(open(S1F, encoding="utf-8-sig"))["rows"]
        if r.get("status") == "ok"]
by = {r["config"]: r for r in rows}
TCOL = {"hemo": fp.C_HEMA, "neural": fp.C_NEURAL, "lung": fp.C_LUNG}
TNAME = {"hemo": "hematopoietic", "neural": "neural crest", "lung": "lung fibrosis"}

PAIR = [("R111_s42", "R111\n5 pure erythroid TFs\n111 edges"),
        ("R107_s42", "R107\n5 mixed-lineage TFs\n107 edges")]

fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.2))

# ---- (a) matched-scale pair, both statistics -------------------------------
ax = axes[0]
x = np.arange(len(PAIR))
w = 0.34
ctr = [(1.0 - by[c]["kappa_tr"]) * 100.0 for c, _ in PAIR]
cr2 = [(1.0 - by[c]["kappa_r2"]) * 100.0 for c, _ in PAIR]
ax.bar(x - w / 2, ctr, w, color=fp.RED, edgecolor="white", linewidth=0.8,
       label="trace form (as released)")
ax.bar(x + w / 2, cr2, w, color=fp.C_HEMA, edgecolor="white", linewidth=0.8,
       label="corrected predictability form")
ax.axhline(50, ls="--", lw=1.2, color=fp.GRAY)
ax.text(-0.42, 52, r"deep-compression threshold  $C=50\%$", fontsize=7, color=fp.GRAY)
for xi, (a, b) in enumerate(zip(ctr, cr2)):
    ax.text(xi - w / 2, a + (2 if a >= 0 else -6), "%+.1f%%" % a, ha="center",
            fontsize=7.5, color=fp.RED)
    ax.text(xi + w / 2, b + 2, "%+.2f%%" % b, ha="center", fontsize=7.5,
            color=fp.C_HEMA)
ax.annotate("", xy=(0.5, ctr[1]), xytext=(0.5, ctr[0]),
            arrowprops=dict(arrowstyle="<->", color=fp.GRAY, lw=1.0))
ax.text(0.56, (ctr[0] + ctr[1]) / 2, "72 pp\ncontrast", fontsize=7, color=fp.GRAY)
ax.text(0.56, 6, "0.14 pp\nafter correction", fontsize=7, color=fp.C_HEMA)
ax.set_xticks(x)
ax.set_xticklabels([lab for _, lab in PAIR], fontsize=7.5)
ax.set_ylabel(r"$C=(1-\kappa)\times100\%$", fontsize=9)
ax.set_title("(a) matched-scale purity contrast", fontsize=9.5)
ax.set_ylim(-95, 70)
fp.safe_legend(ax, loc="lower left", fontsize=7)

# ---- (b) all 33 configurations, both statistics ----------------------------
ax = axes[1]
bnd = [0.5, 6.2]
ax.axhspan(bnd[0], bnd[1], color=fp.RED, alpha=0.07, zorder=0)
ax.axhline(0.5, ls="--", lw=1.3, color=fp.RED, zorder=1)
ax.text(0.02, 0.55, r"lock threshold  $\kappa=0.5$", fontsize=7, color=fp.RED)
ax.axhline(1.0, lw=1.0, color=fp.GRAY, zorder=1)
for r in rows:
    c = TCOL[r["tissue"]]
    ax.plot([0, 1], [r["kappa_tr"], r["kappa_r2"]], color=c, lw=1.0,
            alpha=0.75, marker="o", markersize=3.4, zorder=3)
for t in ("hemo", "neural", "lung"):
    ax.plot([], [], color=TCOL[t], lw=1.6, marker="o", markersize=4,
            label=TNAME[t])
ax.set_xticks([0, 1])
ax.set_xticklabels(["trace form\n(as released)", "corrected\npredictability form"],
                   fontsize=8)
ax.set_ylabel(r"$\kappa$-$C(W)$", fontsize=9)
ax.set_xlim(-0.28, 1.28)
ax.set_ylim(*bnd)
ax.set_title("(b) all 33 configurations", fontsize=9.5)
ax.text(0.03, 5.6, "trace: 0.518-5.78\ncorrected: 0.873-1.003\n$0/33$ below the threshold in both",
        fontsize=7, va="top", color=fp.GRAY,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=fp.GRAY, lw=0.6,
                  alpha=0.92))
fp.safe_legend(ax, loc="lower right", fontsize=7)

fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(FIG + "." + ext, dpi=300, bbox_inches="tight")
print("[saved]", FIG + ".pdf / .png")
print("(a) trace C%%: %s   corrected C%%: %s"
      % (["%+.2f" % v for v in ctr], ["%+.2f" % v for v in cr2]))
print("    contrast: %.1f pp -> %.2f pp" % (abs(ctr[0] - ctr[1]),
                                            abs(cr2[0] - cr2[1])))
ktr = np.array([r["kappa_tr"] for r in rows])
kr2 = np.array([r["kappa_r2"] for r in rows])
print("(b) trace %.3f-%.3f   corrected %.3f-%.3f   n_below_0.5 tr=%d r2=%d"
      % (ktr.min(), ktr.max(), kr2.min(), kr2.max(),
         int((ktr < 0.5).sum()), int((kr2 < 0.5).sum())))
