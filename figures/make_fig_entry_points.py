# -*- coding: utf-8 -*-
r"""make_fig_entry_points.py -- where a prior enters, as a ROUTE DIAGRAM.

The previous version was a table of styled text boxes: a table, not a figure. This draws
the actual topology instead -- the model as a pipeline, the three boundaries a prior can
cross drawn as gates on that pipeline, and each gate marked with what the audit found. The
reader gets one picture of the mechanism instead of three paragraphs inside boxes.

Width = \textwidth = 6.6929 in. Palette = figure_palette.
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "code"))
import figure_palette as fp  # noqa: E402

OUT = os.path.join(HERE, "fig_entry_points.pdf")
OUT_PNG = os.path.join(HERE, "fig_entry_points.png")

fig, ax = plt.subplots(figsize=(6.6929, 2.30))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")

py_, ph = 30, 17
ax.add_patch(FancyBboxPatch((4, py_), 20, ph, boxstyle="round,pad=0.9",
                            facecolor=fp.BAND, edgecolor=fp.EDGE, lw=0.9))
ax.text(14, py_ + ph / 2, "data\n(genes x cells)", ha="center", va="center",
        fontsize=6.4, color=fp.INK)
ax.add_patch(FancyBboxPatch((38, py_), 20, ph, boxstyle="round,pad=0.9",
                            facecolor=fp.BAND, edgecolor=fp.EDGE, lw=0.9))
ax.text(48, py_ + ph / 2, "encoder\ndecoder", ha="center", va="center",
        fontsize=6.4, color=fp.INK)
ax.add_patch(FancyBboxPatch((72, py_), 24, ph, boxstyle="round,pad=0.9",
                            facecolor="white", edgecolor=fp.EDGE, lw=0.9))
ax.text(84, py_ + ph / 2, "objective\n$L_{\\rm data}+\\lambda L_{\\rm sig}$",
        ha="center", va="center", fontsize=6.2, color=fp.INK)
for x0, x1 in ((24.5, 37.5), (58.5, 71.5)):
    ax.add_patch(FancyArrowPatch((x0, py_ + ph / 2), (x1, py_ + ph / 2),
                                 arrowstyle="-|>", color=fp.EDGE, lw=1.0,
                                 mutation_scale=8))

GATES = [
    (14, "3", "the data", "NOT AUDITED",
     "adjacency used to\nbuild features;\nout of scope here", fp.MUTED),
    (48, "2", "architecture", "DELIVERED",
     "mask / conditioning\ninto the decoder; gain\nreproduced by a null", fp.CAUTION),
    (84, "1", "the loss", "NOT DELIVERED",
     "per-gene weights; vector\nall-zero, branch never\ninstantiated", fp.FAIL),
]
for x, num, name, verdict, note, col in GATES:
    ax.add_patch(FancyArrowPatch((x, py_ + ph + 1.0), (x, py_ + ph + 6.5),
                                 arrowstyle="-|>", color=col, lw=1.5,
                                 mutation_scale=9))
    ax.add_patch(Rectangle((x - 9, py_ + ph + 7.0), 18, 2.4, facecolor=col,
                           edgecolor="none"))
    ax.text(x, py_ + ph + 11.2, "%s  (%s)" % (name, num), ha="center", va="bottom",
            fontsize=6.8, weight="bold", color=fp.INK)
    ax.text(x, py_ + ph + 10.4, "", ha="center")
    ax.text(x, py_ + ph + 15.4, verdict, ha="center", va="bottom", fontsize=6.0,
            color=col, weight="bold")
    ax.text(x, py_ - 1.5, note, ha="center", va="top", fontsize=5.4, color=fp.INK,
            linespacing=1.45)

ax.text(50, 99, "A prior can enter at one of three points; this audit covers two",
        ha="center", va="top", fontsize=7.2, weight="bold", color=fp.INK)
ax.text(84, 88, "the entry the title names", ha="center", va="top", fontsize=6.0,
        color=fp.FAIL)

fig.tight_layout(pad=0.2)
fig.savefig(OUT)
fig.savefig(OUT_PNG, dpi=300)
print("[saved]", OUT)
print("[saved]", OUT_PNG)
for x, num, name, verdict, note, col in GATES:
    print("[gate] %s  %-14s %s" % (num, name, verdict))
