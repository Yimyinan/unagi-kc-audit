# -*- coding: utf-8 -*-
"""Fig. 0 (schematic): workflow of the kappa-C audit. SINGLE-PANEL version
(NCS main Fig. 1). The compact phase diagram that used to live in panel (b) is
now the dedicated, detailed Fig. 2 (fig2_kc_phase_diagram.pdf), so this figure
is purely the audit workflow / conceptual pipeline.

Redesigned on a strict grid: no two boxes intersect and no arrow crosses a box
it does not belong to. All colours are drawn from figure_palette (semantic
contract enforced by code/check_figure_palette.py). The protocol box names all
four checks of the four-check prior-activity protocol.
"""
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

BASE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(BASE)
sys.path.insert(0, os.path.join(PKG, "code"))
import figure_palette as fp
fp.apply_paper_style()

WHITE = "#FFFFFF"  # only structural pure-white literal permitted by checker

def add_box(ax, x, y, w, h, fc=WHITE, ec=fp.EDGE, lw=1):
    """Draw a rounded box on the grid (no text). Returns the patch."""
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                       fc=fc, ec=ec, lw=lw, zorder=1)
    ax.add_patch(p)
    return p

def label(ax, x, y, w, h, text, fs=7.2, tc=fp.INK, bold=False, ha="center"):
    """Centre a (possibly multi-line) string inside a box."""
    ax.text(x + w/2, y + h/2, text, ha=ha, va="center", fontsize=fs,
            color=tc, fontweight="bold" if bold else "normal", linespacing=1.35,
            zorder=3)

def arrow(ax, x1, y1, x2, y2, color=fp.EDGE):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=9,
                                 color=color, lw=1, shrinkA=0, shrinkB=0, zorder=2))

fig = plt.figure(figsize=(6.7, 3.35))
ax = fig.add_axes([0.01, 0.02, 0.98, 0.96]); ax.axis("off")
ax.set_xlim(0, 10); ax.set_ylim(0, 10)

# ============================================================ TOP ROW
# inputs -> models (vanilla VAE / GRN-prior VAE) -> decoder readout
add_box(ax, 0.15, 8.60, 2.50, 1.40)                       # scRNA-seq inputs
label(ax, 0.15, 8.60, 2.50, 1.40,
      "scRNA-seq\nGSE150774 / GSE136831 /\nGSE153162\n(N = 20k-30k cells)", fs=6.8)

add_box(ax, 3.30, 9.05, 3.00, 0.75, fc=fp.BAND_NULL)      # vanilla VAE (neutral band)
label(ax, 3.30, 9.05, 3.00, 0.75, "vanilla VAE\n(no prior)", fs=7.0)

add_box(ax, 3.30, 8.15, 3.00, 0.75, fc=fp.BAND_CROSS)     # GRN-prior VAE (cross band)
label(ax, 3.30, 8.15, 3.00, 0.75, "GRN-prior VAE\n(injected prior, 33 configs)", fs=6.6)

add_box(ax, 6.80, 8.60, 3.00, 1.40)                       # decoder readout
label(ax, 6.80, 8.60, 3.00, 1.40,
      "decoder readout\nRidge proxy / analytical\nJacobian  (H = W^T W)", fs=6.8)

arrow(ax, 2.65, 9.20, 3.28, 9.42)     # inputs -> vanilla VAE
arrow(ax, 2.65, 8.60, 3.28, 8.52)     # inputs -> GRN-prior VAE
arrow(ax, 6.32, 9.42, 6.78, 9.30)     # vanilla VAE -> readout
arrow(ax, 6.32, 8.52, 6.78, 8.95)     # GRN-prior VAE -> readout

# ============================================================ MIDDLE ROW
# the two spectra being audited
add_box(ax, 0.15, 6.35, 2.50, 1.60)                       # trace ratio
label(ax, 0.15, 6.35, 2.50, 1.60,
      "trace ratio\nk = tr(H_P) / tr(H_V)\nC = (1-k) x 100%", fs=7.0)

add_box(ax, 3.30, 6.35, 3.00, 1.60)                       # latent spectrum
label(ax, 3.30, 6.35, 3.00, 1.60,
      "latent spectrum\nZ = encoder output\n(tr Z^T Z: effective rank,\nleading-axis share)", fs=6.8)

arrow(ax, 2.65, 7.15, 3.28, 7.15)     # trace ratio -> latent spectrum
ax.text(0.15, 8.02, "two spectra, one audit", fontsize=7,
        fontweight="bold", color=fp.MUTED, zorder=3)

# spectra -> classification band
arrow(ax, 1.40, 6.35, 2.60, 5.72)     # trace ratio -> classification
arrow(ax, 4.80, 6.35, 5.00, 5.72)     # latent spectrum -> classification

# ============================================================ CLASSIFICATION BAND
add_box(ax, 1.00, 3.90, 8.00, 1.80)                       # classification (decoder readout k-C)
ax.text(1.15, 5.55, "classification (decoder readout k-C)", fontsize=7.5,
        fontweight="bold", color=fp.INK, va="top", zorder=3)
ax.text(1.15, 5.12, "k < 0.5  (C > 50%): deep compression / locking",
        fontsize=7, color=fp.FAIL, va="top", zorder=3)
ax.text(1.15, 4.74, "k in [0.937, 1.064]: vanilla-equivalent  (+-2sigma null)",
        fontsize=7, color=fp.MUTED, va="top", zorder=3)
ax.text(1.15, 4.36, "k > 1.064: spectral expansion (amplified Hessian trace)",
        fontsize=7, color=fp.MUTED, va="top", zorder=3)

arrow(ax, 2.60, 3.90, 2.60, 3.57)     # classification -> protocol
arrow(ax, 6.50, 3.90, 7.00, 3.57)     # classification -> result

# ============================================================ BOTTOM ROW
# four-check prior-activity protocol  +  result (audit)
add_box(ax, 0.15, 1.45, 4.85, 2.10)                       # protocol box
ax.text(0.30, 3.38, "four-check prior-activity protocol", fontsize=7.5,
        fontweight="bold", color=fp.INK, va="top", zorder=3)
ax.text(0.30, 3.00, "1.  gradient check", fontsize=7, color=fp.INK, va="top", zorder=3)
ax.text(0.30, 2.64, "2.  prior-strength (lambda) scan", fontsize=7, color=fp.INK, va="top", zorder=3)
ax.text(0.30, 2.28, "3.  structural bound  (gap <= (1-C) trH/gap_V)",
        fontsize=7, color=fp.INK, va="top", zorder=3)
ax.text(0.30, 1.92, "4.  injection-path check   [this work]",
        fontsize=7, color=fp.INK, va="top", zorder=3)

add_box(ax, 5.40, 1.45, 4.40, 2.10, fc=fp.FAIL_FILL, ec=fp.FAIL)   # result (audit)
ax.text(5.55, 3.38, "result (audit)", fontsize=7.5,
        fontweight="bold", color=fp.FAIL, va="top", zorder=3)
ax.text(5.55, 2.92, "0/33 prior-injection configs reach C > 50%",
        fontsize=7, color=fp.FAIL, va="top", zorder=3)
ax.text(5.55, 2.48, "33/33 Jacobian re-audit agree (0 locked)",
        fontsize=7, color=fp.INK, va="top", zorder=3)

arrow(ax, 5.02, 2.50, 5.38, 2.50)     # protocol -> result

# ============================================================ VERDICT BANNER
add_box(ax, 0.15, 0.15, 9.65, 0.95, fc=fp.PASS_FILL, ec=fp.EDGE)
label(ax, 0.15, 0.15, 9.65, 0.95,
      "verdict: perturbation readiness is a measured property, not a design "
      "guarantee - audit per model, per seed, per prior", fs=7.2, bold=True)

arrow(ax, 7.00, 1.45, 7.00, 1.12)     # result -> verdict

fig.savefig(os.path.join(BASE, "fig0_workflow.pdf"))
fig.savefig(os.path.join(BASE, "fig0_workflow.png"))
print("saved fig0_workflow.pdf/png (single-panel)")
