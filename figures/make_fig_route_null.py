# -*- coding: utf-8 -*-
r"""make_fig_route_null.py -- the protocol's discriminant, both directions.

Panel A (the route x null x result text table) has been REMOVED.  It was a text table
rendered as a figure, which belongs in the prose, not in a main display item.  Its content
now lives in the text.  What remains is the one panel that is real data: the protocol's
discriminant firing in both directions on the published synthetic control.

Data (no new runs): data/three_check_positive_control.json.
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "code"))
import figure_palette as fp  # noqa: E402

OUT = os.path.join(HERE, "fig_route_null.pdf")
OUT_PNG = os.path.join(HERE, "fig_route_null.png")

ctl = json.load(open(os.path.join(ROOT, "data",
                                  "three_check_positive_control.json"), encoding="utf-8"))
a, i2 = ctl["active"]["summary"], ctl["inert"]["summary"]
pairs = [("prior active\n(synthetic positive)", a["wiring_prior_mean"],
          a["wiring_shuffled_mean"], fp.PASS),
         ("prior inert\n(synthetic negative)", i2["wiring_prior_mean"],
          i2["wiring_shuffled_mean"], fp.FAIL)]

fig, ax = plt.subplots(figsize=(6.6929, 2.45))
x = np.arange(2)
wdt = 0.30
pr = [p[1] for p in pairs]
sh = [p[2] for p in pairs]
ax.bar(x - wdt / 2, pr, wdt, color=[p[3] for p in pairs], edgecolor=fp.INK,
       linewidth=0.6, zorder=3, label="prior arm")
ax.bar(x + wdt / 2, sh, wdt, color=fp.MUTED, edgecolor=fp.INK, linewidth=0.6,
       zorder=3, alpha=0.5, label="form-matched null arm")
for xi, (p, s) in enumerate(zip(pr, sh)):
    ax.text(xi - wdt / 2, p + 0.15, "%.2f" % p, ha="center", fontsize=7.0, color=fp.INK)
    ax.text(xi + wdt / 2, s + 0.15, "%.2f" % s, ha="center", fontsize=7.0, color=fp.INK)
    ax.text(xi, max(p, s) + 0.85, "$%.2f\\times$" % (p / s), ha="center", fontsize=8.0,
            color=pairs[xi][3], weight="bold")
ax.set_xticks(x)
ax.set_xticklabels([p[0] for p in pairs], fontsize=7.0)
ax.set_ylabel("wiring statistic", fontsize=7.4)
ax.set_ylim(0, 7.6)
ax.axhline(1.0, color=fp.GREY, ls="--", lw=0.9, zorder=2)
ax.legend(loc="upper right", frameon=False, fontsize=6.4)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(labelsize=6.6)
ax.grid(axis="y", lw=0.35, color=fp.BAND, zorder=0)
fig.tight_layout(pad=0.35)
fig.savefig(OUT)
fig.savefig(OUT_PNG, dpi=300)
print("[saved]", OUT)
print("[data] active %.3f/%.3f = %.2fx ; inert %.3f/%.3f = %.2fx"
      % (pr[0], sh[0], pr[0] / sh[0], pr[1], sh[1], pr[1] / sh[1]))
