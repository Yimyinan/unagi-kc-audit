# -*- coding: utf-8 -*-
"""Supplementary Figure S26: live-path positive control for the corrected statistic.

Purpose
-------
The synthetic controls (Supplementary Figs S5, S24) were run at a latent covariance
for which the trace and predictability forms of the readout coincide, so they could
not have caught the readout-form defect.  This control wires the prior into a real
trained model's reconstruction path as a per-gene loss weight and shows that the
corrected statistic resolves the resulting effect while the released trace form
does not.

Arms (3 seeds each, 15 k cells, beta = 0.002, 150 epochs):
    b0     no loss weighting (baseline)
    b2     per-gene weight boost 2x on 204 prior-target genes
    b10    boost 10x on the same 204 genes
    perm2  boost 2x on a permuted (wrong) gene set of the same size

Reads  : outputs/positive_result/s1_live_path/s1e_reanalysis.json
Writes : figures/figS26_live_path_control.pdf / .png
Usage  : python code/make_figS26_live_path.py
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

SRC = os.path.abspath(os.path.join(HERE, "..", "..", "..", "outputs",
                                   "positive_result", "s1_live_path",
                                   "s1e_reanalysis.json"))
FIG = os.path.join(HERE, "..", "figures", "figS26_live_path_control")

fp.apply_paper_style()
d = json.load(open(SRC, encoding="utf-8-sig"))
rows = d["rows"]

GROUPS = ["null (b0 vs b0, cross-seed)", "wp_b2 vs b0 (same seed)",
          "wp_b10 vs b0 (same seed)", "wp_perm2 vs b0 (same seed)"]
LABELS = ["null\n(b0 vs b0)", "true prior\nboost 2x", "true prior\nboost 10x",
          "wrong prior\nboost 2x"]
COLORS = [fp.GRAY, fp.C_HEMA, fp.RED, fp.PURPLE if hasattr(fp, "PURPLE") else fp.GRAY]


def vals(group, key):
    return np.array([r[key] for r in rows if r["group"] == group], float)


def panel(ax, key, title, ylabel, fmt):
    null = vals(GROUPS[0], key)
    mu, sd = null.mean(), null.std(ddof=1)
    lo, hi = mu - 2 * sd, mu + 2 * sd
    ax.axhspan(lo, hi, color=fp.BAND, alpha=0.9, zorder=0)
    ax.axhline(mu, color=fp.GRAY, ls=":", lw=0.8, zorder=1)
    for i, g in enumerate(GROUPS):
        v = vals(g, key)
        jitter = np.linspace(-0.10, 0.10, len(v))
        ax.scatter(np.full(len(v), i) + jitter, v, s=30, color=COLORS[i],
                   edgecolor="white", linewidth=0.5, zorder=3)
        ax.plot([i - 0.22, i + 0.22], [v.mean()] * 2, color=COLORS[i], lw=1.8,
                zorder=4)
    ax.set_xticks(range(len(GROUPS)))
    ax.set_xticklabels(LABELS, fontsize=6.8)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=9, loc="left")
    ax.text(0.985, 0.04, "shaded: cross-seed null $\\pm2$ sd\n(sd = %s)" % fmt(sd),
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.3,
            color=fp.GRAY)
    return mu, sd


fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))

mu_a, sd_a = panel(axes[0], "r2",
                   "(a) corrected statistic", r"$\kappa$-$C(W)$  ($\sum_g R^2_g$)",
                   lambda s: "%.4f" % s)
a_lo = mu_a - 2 * sd_a
axes[0].text(1.0, 0.9955, "3/3 seeds below\nthe null band", fontsize=6.5,
             color=fp.RED, ha="left", va="top")
axes[0].set_ylim(0.977, 1.006)

mu_t, sd_t = panel(axes[1], "tr",
                   "(b) released trace form", r"$\kappa$-$C(W)$  (trace)",
                   lambda s: "%.4f" % s)
axes[1].text(1.0, 1.12, "signal sits inside\nthe null band", fontsize=6.5,
             color=fp.GRAY, ha="left", va="top")
axes[1].set_ylim(0.78, 1.20)

fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(FIG + "." + ext, dpi=300, bbox_inches="tight")
print("[saved]", FIG + ".pdf / .png")

for key, nm in (("r2", "corrected"), ("tr", "trace")):
    n0, n10 = vals(GROUPS[0], key), vals(GROUPS[2], key)
    t = ((n0.mean() - n10.mean())
         / np.sqrt(n0.var(ddof=1) / n0.size + n10.var(ddof=1) / n10.size))
    print(f"{nm:<10} null sd {n0.std(ddof=1):.4f}  b10 mean {n10.mean():.4f}  "
          f"delta {n0.mean() - n10.mean():+.4f}  Welch t {t:+.2f}  "
          f"separated={bool(n10.max() < n0.min())}")
