"""Draw the four-check prior-activity protocol as a decision flowchart.

WHY
---
The protocol is the deliverable of this methods paper, so it deserves a one-glance
decision diagram: any failing check routes to "not prior-driven"; all four passing
is the only route to "prior-active".  The fourth check (injection-path) is this
work's own addition.  The controlled positive control (Fig.~\ref{fig:three_check})
is referenced as the demonstration that the "prior-active" leaf is reachable.

USAGE:  python figures/make_fig_three_check_flow.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "code"))

import figure_palette as fp  # noqa: E402
from figure_palette import (  # noqa: E402
    apply_paper_style, BAND, BAND_NULL, INK, MUTED, EDGE,
    FAIL, FAIL_FILL, PASS, PASS_FILL, ARM_PRIOR,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch  # noqa: E402

OUT = os.path.join(HERE, "fig0_three_check_flowchart.pdf")

SPINE = EDGE        # neutral main flow
FAILC = FAIL        # fail route only


def box(ax, x, y, w, h, text, fc, fs=8.2, bold=False,
        ec=EDGE, lw=1.0, tc=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.05",
                                linewidth=lw, edgecolor=ec, facecolor=fc, zorder=3))
    weight = "bold" if bold else "normal"
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, fontweight=weight, color=tc, zorder=4, linespacing=1.35)


def arrow(ax, x1, y1, x2, y2, label=None, color=SPINE, ls="-", lw=1.25):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=12, linewidth=lw,
                                 color=color, linestyle=ls, zorder=1))
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.13, label,
                fontsize=7, color=color, ha="center", va="bottom",
                style="italic", zorder=4)


def spine(ax, x, y1, y2):
    ax.plot([x, x], [y1, y2], color=SPINE, linewidth=2.0, zorder=1)


def main():
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(4.819, 5.542))
    ax.set_xlim(0, 11); ax.set_ylim(-2.7, 12.6); ax.axis("off")

    W, H = 4.8, 1.3
    CX = 4.5                       # centre column x-left anchor
    BX = CX - W / 2                # box left edge
    REX = CX + W / 2               # box right edge

    # geometry (bottom y, centre y) for each gate
    start = (11.0, 12.1)
    c1 = (8.7, 9.35)
    c2 = (6.1, 6.75)
    c3 = (3.5, 4.15)
    c4 = (0.9, 1.55)
    leaf = (-1.3, -0.75)

    # ---- start
    box(ax, BX, start[0], W, start[1] - start[0],
        "Observed spectral effect\n(e.g. large $C$, shifted $\\kappa$-$C$)",
        BAND, fs=8.4, bold=True)

    # ---- continuous vertical spine (branches only on FAIL)
    spine(ax, CX, start[0], c1[1] + H / 2)          # start -> C1
    spine(ax, CX, c1[0], c2[1] + H / 2)             # C1 -> C2
    spine(ax, CX, c2[0], c3[1] + H / 2)             # C2 -> C3
    spine(ax, CX, c3[0], c4[1] + H / 2)             # C3 -> C4
    spine(ax, CX, c4[0], leaf[0] + 1.1)   # C4 -> leaf (top of leaf box)

    # ---- the four checks
    box(ax, BX, c1[0], W, H,
        "Check 1 — gradient / wiring\nDo prior edges carry gradient?",
        BAND_NULL, fs=8.3, bold=True)
    box(ax, BX, c2[0], W, H,
        "Check 2 — $\\lambda$ strength scan\nDoes the response track prior weight?",
        BAND_NULL, fs=8.3, bold=True)
    box(ax, BX, c3[0], W, H,
        "Check 3 — structural bound\nIs the collapse outside the data-inherent bound?",
        BAND_NULL, fs=8.3, bold=True)
    # Check 4 — this work's addition: distinct ARM_PRIOR outline + marker
    box(ax, BX, c4[0], W, H,
        "Check 4 — injection path\nIs there a functional route into the loss?",
        BAND_NULL, fs=8.3, bold=True, ec=ARM_PRIOR, lw=2)
    # marker sits above the box's right corner so it clears the central spine
    ax.text(REX, c4[1] + H / 2 + 0.30, "this work",
            fontsize=7, color=ARM_PRIOR, ha="right", va="center",
            fontweight="bold", style="italic", zorder=4)

    # ---- single right-hand "not prior-driven" outcome column
    col_x, col_w = 7.85, 2.75
    col_top, col_bot = 10.15, 0.75
    ax.add_patch(FancyBboxPatch((col_x, col_bot), col_w, col_top - col_bot,
                               boxstyle="round,pad=0.02,rounding_size=0.05",
                               linewidth=1.25, edgecolor=FAIL, facecolor=FAIL_FILL, zorder=2))
    ax.text(col_x + col_w / 2, col_top - 0.32, "not prior-driven",
            fontsize=8.0, fontweight="bold", color=FAIL, ha="center", va="center", zorder=4)
    reasons = [
        (c1[1], "data-inherent /\nartefact"),
        (c2[1], "flat $\\lambda$\nresponse"),
        (c3[1], "collapse inside\nthe data-inherent bound"),
        (c4[1], "no functional route\ninto the loss"),
    ]
    for ry, rtext in reasons:
        ax.text(col_x + col_w / 2, ry, rtext, fontsize=7.5, color=INK,
                ha="center", va="center", zorder=4, linespacing=1.3)

    # ---- FAIL branches: each check feeds the reason that belongs to it
    for cy, ry in [(c1[1], c1[1]), (c2[1], c2[1]), (c3[1], c3[1]), (c4[1], c4[1])]:
        arrow(ax, REX, cy, col_x - 0.03, ry, label="FAIL", color=FAILC)

    # ---- terminal PRIOR-ACTIVE leaf (the only PASS in the figure)
    box(ax, BX, leaf[0], W, 1.1,
        "PRIOR-ACTIVE\n(all four checks pass)",
        PASS_FILL, fs=9.0, bold=True, ec=PASS, lw=1.5, tc=PASS)
    ax.text(CX, leaf[0] - 0.45,
            "reachable — demonstrated by the controlled positive control",
            fontsize=7.0, color=MUTED, ha="center", va="center", style="italic", zorder=4)

    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    fig.savefig(OUT.replace(".pdf", ".png"), dpi=170, bbox_inches="tight")
    print("[saved]", os.path.abspath(OUT))


if __name__ == "__main__":
    main()
