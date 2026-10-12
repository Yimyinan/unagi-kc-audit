#!/usr/bin/env python3
"""Graphical abstract for paper1 -- "three gates on the delivery path".

Implements submission/GRAPHICAL_ABSTRACT_SPEC.md.  Every number printed is from
submission/GRAPHICAL_ABSTRACT_SPEC.md section 8 (the verified list); the items in its
section 9 ("[DO NOT PRINT]") are deliberately absent -- in particular there is no
`max |delta|` annotation and the words "prior-driven" appear nowhere.

Layout is driven by explicit bands so text blocks cannot collide:
    0.93-1.00  headline
    0.83-0.92  spine title
    0.68-0.82  three gates
    0.63-0.67  gate-1 note
    0.44-0.62  callout box (left) + its annotation (right of the box)
    0.31-0.42  funnel title / evidence title  (side by side, non-overlapping x-ranges)
    0.15-0.30  funnel bars (left) / evidence strip (right)
    0.03-0.14  outcome band + takeaway

Run:  python code/make_graphical_abstract.py
Out:  figures/graphical_abstract.{pdf,png}
"""
from __future__ import annotations

import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import figure_palette as fp

OUT = os.path.join(HERE, "..", "figures", "graphical_abstract")

W, H = 7.2, 5.4
INK, MUTED, GREY = fp.INK, fp.MUTED, fp.GREY
FAIL, PASS, BAND = fp.FAIL, fp.PASS, fp.BAND
FAIL_FILL, PASS_FILL = fp.FAIL_FILL, fp.PASS_FILL

# column geometry.  The right column is indented past the left column's labels so that the two
# blocks can never collide horizontally.
L0, LW = 0.028, 0.232          # left column
L1 = L0 + LW                   # left column right edge
R0, RW = 0.290, 0.652          # right column (indented)


def box(ax, x, y, w, h, fc="white", ec=INK, lw=0.9, r=0.010):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0,rounding_size=%f" % r,
                                fc=fc, ec=ec, lw=lw, zorder=2, mutation_aspect=1))


def arrow(ax, x1, y1, x2, y2, color=INK, lw=1.0, ms=7):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=ms,
                                 color=color, lw=lw, zorder=3, shrinkA=0, shrinkB=0))


def cross(ax, cx, cy, r, color=FAIL, lw=1.6):
    """Arial has no U+2717, so draw the X as two strokes."""
    ax.plot([cx - r, cx + r], [cy - r, cy + r], color=color, lw=lw, zorder=4,
            solid_capstyle="round")
    ax.plot([cx - r, cx + r], [cy + r, cy - r], color=color, lw=lw, zorder=4,
            solid_capstyle="round")


def main():
    fp.apply_paper_style()
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # ------------------------------------------------------------------ headline
    ax.text(L0, 0.972,
            "A gene-regulatory-network prior is built, assigned to a single-cell VAE,",
            fontsize=11.0, fontweight="bold", color=INK, va="top", ha="left")
    ax.text(L0, 0.936,
            "and never read. The decoder readout is unchanged — not because the prior was "
            "rejected, but because it never arrived.",
            fontsize=8.2, color=GREY, va="top", ha="left")

    # --------------------------------------------------------------- spine title
    ax.text(L0, 0.900, "THREE GATES ON THE INJECTION PATH",
            fontsize=7.4, fontweight="bold", color=MUTED, va="top", ha="left")

    # --------------------------------------------------------------- three gates
    gy, gh, gw = 0.722, 0.132, 0.192
    x2, x3 = 0.262, 0.496
    for x, title, sub, failed in (
            (L0, "1  Delivered?", "per-gene weight vector\nreaches the loss", True),
            (x2, "2  Prior-active?", "gradient check, $\\lambda$-scan,\nstructural bound", False),
            (x3, "3  Effective?", "decoder readout moves\nvs the prior-free null", False)):
        if failed:
            box(ax, x, gy, gw, gh, fc=FAIL_FILL, ec=FAIL, lw=1.4)
            cross(ax, x + gw / 2, gy + gh - 0.026, 0.010)
        else:
            box(ax, x, gy, gw, gh, fc="#F4F4F4", ec=GREY, lw=0.8)
            ax.plot([x + gw / 2 - 0.011, x + gw / 2 + 0.011], [gy + gh - 0.026] * 2,
                    color=GREY, lw=1.3, zorder=4)
        ax.text(x + gw / 2, gy + 0.083, title, fontsize=8.0, fontweight="bold",
                color=INK if failed else GREY, ha="center", va="center")
        ax.text(x + gw / 2, gy + 0.040, sub, fontsize=6.1, color=GREY,
                ha="center", va="center", linespacing=1.32)

    arrow(ax, L0 + gw + 0.006, gy + gh / 2, x2 - 0.006, gy + gh / 2, color=GREY, lw=0.9)
    arrow(ax, x2 + gw + 0.006, gy + gh / 2, x3 - 0.006, gy + gh / 2, color=GREY, lw=0.9)

    ax.text(L0, 0.706,
            "Gate 1 fails, so gates 2 and 3 are never opened — a null at gate 3 cannot\n"
            "distinguish “prior inert” from “prior absent”.",
            fontsize=6.7, style="italic", color=GREY, va="top", ha="left", linespacing=1.38)

    # --------------------------------------------- callout: where the path stops
    cy, ch = 0.452, 0.208
    box(ax, L0, cy, LW, ch, fc="white", ec=INK, lw=1.0)
    ax.text(L0 + 0.011, cy + ch - 0.018, "the call graph terminates",
            fontsize=6.9, fontweight="bold", color=INK, va="top", ha="left")
    ax.text(L0 + 0.011, cy + ch - 0.039, "at the assignment",
            fontsize=6.9, fontweight="bold", color=INK, va="top", ha="left")
    steps = ["prior database (TRRUST v2)",
             "per-gene weight vector (203 scalars)",
             "_tf_prior_weights  —  assigned"]
    sy = cy + ch - 0.070
    for i, label in enumerate(steps):
        y = sy - i * 0.026
        ax.text(L0 + 0.013, y, label, fontsize=5.6, color=INK, va="center", ha="left")
        if i < len(steps) - 1:
            arrow(ax, L0 + 0.017, y - 0.007, L0 + 0.017, y - 0.018, color=GREY, lw=0.7, ms=5)
        else:
            cross(ax, L0 + LW - 0.016, y, 0.0075, lw=1.5)
    ax.text(L0 + 0.013, cy + 0.012,
            "no code path reads it;\nthe likelihood-weighting branch\nhas 0 instantiation sites",
            fontsize=5.7, color=FAIL, va="bottom", ha="left", linespacing=1.38)

    # ------------------------------------------ right of callout: what that means
    ax.text(L1 + 0.012, cy + ch - 0.022,
            "Every stage of the pipeline is present and executable.\n"
            "The prior is converted, assigned, and then never read,\n"
            "so no spectral change can be attributed to its content.",
            fontsize=6.5, color=INK, va="top", ha="left", linespacing=1.55)

    # -------------------------------------------------------- two lower columns
    TY = 0.402                                  # band-title baseline
    ax.text(L0, TY, "CONTENT THAT SURVIVES EACH STAGE",
            fontsize=6.7, fontweight="bold", color=MUTED, va="top", ha="left")
    ax.text(R0, TY, "THE CHANNEL IS EMPTY IN EVERY RETAINED RUN",
            fontsize=6.7, fontweight="bold", color=MUTED, va="top", ha="left")

    # funnel (left): four labelled rungs down to zero.
    # rung i occupies label y+0.010 .. value y; the next rung sits 0.034 below, so the label
    # of rung i+1 (y+0.010) never reaches the value text of rung i (y).
    for i, (lab, val, nn) in enumerate(
            [("prior source", "9,396 edges / 795 TFs", 9396),
             ("tissue subset", "325 edges / 11 TFs (3.5%)", 325),
             ("per-gene scalars", "203 values", 203),
             ("weight layer in artefacts", "0 non-zero entries", 0)]):
        y = 0.322 - i * 0.034
        frac = 0.0 if nn == 0 else math.log10(nn) / math.log10(9396)
        barw = 0.007 if nn == 0 else max(0.010, 0.115 * frac)
        col = FAIL if nn == 0 else fp.C_HEMA
        ax.add_patch(Rectangle((L0, y - 0.005), barw, 0.012, fc=col, ec="none", zorder=3))
        ax.text(L0, y + 0.008, lab, fontsize=5.4, color=MUTED, va="bottom", ha="left")
        ax.text(L0 + barw + 0.007, y, val, fontsize=5.9, color=INK, va="center", ha="left")

    # evidence (right): the zero-weight measure
    ey, eh = 0.196, 0.142
    box(ax, R0, ey, RW, eh, fc="white", ec="#CCCCCC", lw=0.7)
    ax.text(R0 + 0.013, ey + eh - 0.013, "non-zero entries in the per-gene weight layer",
            fontsize=5.4, color=MUTED, va="top", ha="left")
    ax.text(R0 + 0.013, ey + eh - 0.037, "796 / 796   nnz = 0", fontsize=8.0,
            fontweight="bold", color=FAIL, va="top", ha="left")
    strip_y = ey + 0.052
    marks = 96
    for k in range(marks):
        ax.add_patch(Rectangle((R0 + 0.013 + k * 0.0046, strip_y - 0.004),
                               0.0025, 0.008, fc=FAIL, ec="none", zorder=3))
    ax.add_patch(Rectangle((R0 + 0.013 + marks * 0.0046 + 0.022, strip_y - 0.006),
                           0.010, 0.012, fc=fp.C_LUNG, ec="white", lw=0.6, zorder=4))
    ax.text(R0 + 0.013 + marks * 0.0046 + 0.038, strip_y, "positive\ncontrol",
            fontsize=5.2, color=fp.C_LUNG, va="center", ha="left", linespacing=1.12)
    ax.text(R0 + 0.013, ey + 0.014,
            "All 796 stage files (29 run–iterations: vanilla controls, curated-, mismatched-\n"
            "prior and ablation arms). The positive control is non-zero.",
            fontsize=5.3, color=MUTED, va="bottom", ha="left", linespacing=1.34)

    ax.text(0.5, 0.166,
            "Audit delivery before attributing any spectral change to a prior’s content.",
            fontsize=7.4, style="italic", fontweight="bold", color=INK,
            va="bottom", ha="center")

    # ------------------------------------------------------------- outcome band
    oy, oh = 0.040, 0.112
    ax.add_patch(Rectangle((L0, oy), 0.944, oh, fc=BAND, ec="#CCCCCC", lw=0.7, zorder=1))
    cells = [
        ("THE NULL", "0 / 33 configurations reach deep\ndecoder compression ($\\kappa<0.5$).\n"
                     "Band-independent; 33 / 33 agree\nunder the analytical Jacobian."),
        ("THE ONE REAL EFFECT", "The latent spectrum reorganizes\n"
                                "(mean PR ratio 0.77; 29 / 33 below\n"
                                "unity) while the decoder readout does\n"
                                "not (0.95; 21 / 33) — a prior-agnostic\n"
                                "training dynamic, not prior content."),
        ("THE TEST TRANSFERS", "On VEGA a degree-preserving shuffled\n"
                               "mask reproduces the gain; in scVI an\n"
                               "API-registered prior never conditioned\nthe model."),
    ]
    for i, (head, body) in enumerate(cells):
        cx = L0 + i * (0.944 / 3)
        if i:
            ax.plot([cx, cx], [oy + 0.007, oy + oh - 0.007], color="#BBBBBB", lw=0.7, zorder=2)
        ax.text(cx + 0.013, oy + oh - 0.011, head, fontsize=6.3, fontweight="bold",
                color=FAIL if i == 0 else INK, va="top", ha="left")
        ax.text(cx + 0.013, oy + oh - 0.031, body, fontsize=5.4, color=INK,
                va="top", ha="left", linespacing=1.40)

    for ext in ("pdf", "png"):
        fig.savefig("%s.%s" % (OUT, ext), dpi=300)
    plt.close(fig)
    print("[saved] %s.pdf / .png" % os.path.abspath(OUT))


if __name__ == "__main__":
    main()
