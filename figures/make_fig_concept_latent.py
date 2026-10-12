# -*- coding: utf-8 -*-
"""
Fig. CONCEPT -- "Conceptual model of tissue-dependent latent-space
reorganization under GRN-prior injection."

A synthesis figure (NCS-style) that makes the paper's core, non-obvious claim
visible at a glance:
  * The audit reads TWO spectra -- the decoder readout W (kappa-C) and the
    latent space Z -- and they move independently.
  * Prior injection reorganizes the LATENT manifold in a tissue-specific way
    (hematopoietic: strong low-rank collapse; neural: null; lung: suggestive)
    while the DECODER readout stays in the vanilla-equivalent region (0/33).
  * The effect is graded by prior-tissue congruence (specificity ladder).

Panel A : two-spectra framing (mechanism schematic)
Panel B : hematopoietic latent manifold -- vanilla cloud -> prior low-rank collapse
Panel C : specificity ladder (real seed-42 effective-rank values)
Panel D : decoder-vs-latent dissociation (the three tissues)

Real numbers are taken from the manuscript audit (kappa_complete_audit.json,
the z-spectrum 3-tissue analysis, and the seed-42 specificity ladder).
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyBboxPatch, FancyArrowPatch
import matplotlib.gridspec as gridspec

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE, "..", "code"))

import figure_palette as fp
fp.apply_paper_style()

# single-source-of-truth colours (kept identical to legacy hex so appearance is unchanged)
C_HEMA, C_NEURAL, C_LUNG = fp.C_HEMA, fp.C_NEURAL, fp.C_LUNG
GRAY, RED, GREEN = fp.GRAY, fp.RED, fp.GREEN

rng = np.random.default_rng(20260905)


def panel_label(ax, txt, x=-0.02, y=1.04):
    ax.text(x, y, txt, transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="top")


def make_cloud(n=220, lowrank=False, mild=False):
    """Schematic 2D latent embedding. Returns (x, y)."""
    z = rng.multivariate_normal([0, 0], [[1.0, 0.0], [0.0, 1.0]], size=n)
    if lowrank:
        z = z @ np.array([[0.98, 0.18], [-0.18, 0.98]])
        z[:, 1] *= 0.12
    elif mild:
        z[:, 1] *= 0.6
    return z[:, 0], z[:, 1]


def cov_ellipse(ax, pts, fc, ec, alpha=0.18):
    xs, ys = pts
    cx, cy = xs.mean(), ys.mean()
    cov = np.cov(xs, ys)
    vals, vecs = np.linalg.eigh(cov)
    ang = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    for k, nstd in enumerate((2.0, 1.0)):
        w, h = 2 * nstd * np.sqrt(vals + 1e-9)
        e = Ellipse((cx, cy), w, h, angle=ang, fc=fc, ec=ec,
                    lw=0.75, alpha=alpha if k == 0 else 0.0)
        ax.add_patch(e)


# ============================================================
fig = plt.figure(figsize=(6.7, 5.7))
gs = gridspec.GridSpec(2, 2, figure=fig,
                       left=0.07, right=0.97, top=0.94, bottom=0.06,
                       wspace=0.32, hspace=0.42)

# ---------------- Panel A : two-spectra framing ----------------
gsA = gs[0, 0].subgridspec(3, 1, height_ratios=[1, 0.45, 1], hspace=0.12)
axA_top = fig.add_subplot(gsA[0, 0])
axA_mid = fig.add_subplot(gsA[1, 0]); axA_mid.axis("off")
axA_bot = fig.add_subplot(gsA[2, 0])

def style_min(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(direction="out", length=2.0, width=0.45)
    return ax

# ---- top: latent Z spectrum reorganizes ----
xspec = np.arange(1, 65)
van_z = np.exp(-0.10 * (xspec - 1)) + 0.05
pri_z = np.exp(-0.60 * (xspec - 1)) + 0.04
axA_top.plot(xspec, van_z / van_z.max(), color=GRAY, ls="--", lw=1.25, label="vanilla")
axA_top.plot(xspec, pri_z / pri_z.max(), color=C_HEMA, lw=1.25, label="prior (hema.)")
axA_top.fill_between(xspec, pri_z / pri_z.max(), 0, color=C_HEMA, alpha=0.08)
axA_top.set_ylabel("norm. eigenvalue", fontsize=6)
axA_top.set_xlim(0, 65); axA_top.set_ylim(-0.05, 1.1)
axA_top.set_title("latent Z spectrum reorganizes", fontsize=7,
                  loc="left", color=C_HEMA, fontweight="bold", y=1.0)
axA_top.legend(frameon=False, fontsize=6, loc="upper right",
               handlelength=1.0)
style_min(axA_top)
axA_top.set_xticklabels([])
panel_label(axA_top, "a")

# ---- middle: prior injection note with arrows ----
axA_mid.text(0.5, 0.55, "GRN prior injected at encoder\n(reorganizes Z, not W)",
             ha="center", va="center", fontsize=6, color="#8B5A2B",
             fontweight="bold", transform=axA_mid.transAxes,
             bbox=dict(boxstyle="round,pad=0.22", fc="#FDF3E7",
                       ec="#DD8452", lw=0.75))
axA_mid.annotate("", xy=(0.5, 0.95), xytext=(0.5, 0.72),
                 arrowprops=dict(arrowstyle="-|>", color=C_HEMA, lw=1),
                 transform=axA_mid.transAxes)
axA_mid.annotate("", xy=(0.5, 0.05), xytext=(0.5, 0.28),
                 arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=1),
                 transform=axA_mid.transAxes)

# ---- bottom: decoder W spectrum unchanged ----
van_w = np.exp(-0.06 * (xspec - 1)) + 0.05
pri_w = van_w * 1.02
axA_bot.plot(xspec, van_w / van_w.max(), color=GRAY, ls="--", lw=1.25, label="vanilla")
axA_bot.plot(xspec, pri_w / van_w.max(), color=GRAY, lw=1.25, label="prior")
axA_bot.set_xlabel("spectrum index", fontsize=6)
axA_bot.set_ylabel("norm. eigenvalue", fontsize=6)
axA_bot.set_xlim(0, 65); axA_bot.set_ylim(-0.05, 1.1)
axA_bot.set_title("decoder W spectrum unchanged ($\\kappa$-$C$ null, 0/33)",
                   fontsize=7, loc="left", color=GRAY, fontweight="bold", y=1.0)
axA_bot.legend(frameon=False, fontsize=6, loc="upper right",
               handlelength=1.0)
style_min(axA_bot)

# ---------------- Panel B : hematopoietic latent manifold ----------------
axB = fig.add_subplot(gs[0, 1])
van = make_cloud(lowrank=False)
pri = make_cloud(lowrank=True)
axB.scatter(van[0], van[1], s=5, color=C_HEMA, alpha=0.45, label="vanilla (R14)")
cov_ellipse(axB, van, C_HEMA, C_HEMA)
axB.scatter(pri[0], pri[1], s=5, color=RED, alpha=0.55, label="prior (R13)")
cov_ellipse(axB, pri, RED, RED)
axB.set_xticks([]); axB.set_yticks([])
for s in ("top", "right", "left", "bottom"):
    axB.spines[s].set_visible(False)
axB.set_xlabel("latent axis 1", fontsize=7); axB.set_ylabel("latent axis 2", fontsize=7)
axB.set_title("hematopoietic latent manifold", fontsize=8, loc="left", fontweight="bold")
panel_label(axB, "b")
axB.text(0.5, 0.96, "collapses to low-rank manifold", transform=axB.transAxes,
         ha="center", va="top", fontsize=6, color=RED)
axB.text(0.98, 0.04, r"$\Delta\log$ eff. rank $= -1.09$",
         transform=axB.transAxes, ha="right", va="bottom", fontsize=6.0, color="#222")
axB.text(0.98, 0.16, r"$p = 1.5\times10^{-4}$ (17/18 vs vanilla)",
         transform=axB.transAxes, ha="right", va="bottom", fontsize=6, color="#222")
axB.legend(frameon=False, fontsize=6, loc="lower left")

# ---------------- Panel C : specificity ladder ----------------
axC = fig.add_subplot(gs[1, 0])
ladder = [("vanilla\n(R14)", 3.10, GRAY),
              ("empty-vector\nvehicle", 5.19, "#BBBBBB"),
              ("wrong-tissue\nneural prior", 2.30, C_NEURAL),
              ("curated hema.\nprior (R13)", 1.75, C_HEMA)]
xs = np.arange(len(ladder))
ys = [v for _, v, _ in ladder]
cols = [c for _, _, c in ladder]
bars = axC.bar(xs, ys, color=cols, width=0.55, edgecolor="#333", linewidth=0.75)
for b, (lab, v, _) in zip(bars, ladder):
    axC.text(b.get_x() + b.get_width()/2, v + 0.12, f"{v:.2f}",
             ha="center", va="bottom", fontsize=6)
axC.set_xticks(xs)
axC.set_xticklabels([lab for lab, _, _ in ladder], fontsize=6)
axC.set_ylabel("effective rank of Z  (participation ratio; lower = more compressed)")
axC.set_ylim(0, 6.5)
axC.set_title("specificity ladder (seed 42)", fontsize=8, loc="left",
              fontweight="bold", y=1.0)
panel_label(axC, "c")
axC.text(0.5, 0.98, "compression graded by prior-tissue congruence",
         transform=axC.transAxes, ha="center", va="top", fontsize=6.0, color=GREEN)
for s in ("top", "right"):
    axC.spines[s].set_visible(False)
axC.tick_params(length=2.2, width=0.5)

# ---------------- Panel D : decoder-vs-latent dissociation ----------------
axD = fig.add_subplot(gs[1, 1])
# vanilla-equivalent band on decoder C axis: kappa in [0.937, 1.064] -> C in [-6.4%, 6.3%]
axD.axvspan(-6.4, 6.3, color="#EEF2F7", alpha=0.9, zorder=0)
axD.axvline(0, color=GRAY, lw=0.75, ls=":")
# three tissues: (decoder C %, latent Dlog eff_rank)
pts = [("hematopoietic", 0.0, -1.09, C_HEMA),
       ("neural crest", -1.5, 0.0, C_NEURAL),
       ("lung fibrosis", 2.0, -1.1, C_LUNG)]
for name, x, y, c in pts:
    axD.scatter([x], [y], s=60, color=c, edgecolors="white", linewidths=0.75, zorder=3)
    axD.annotate(name, (x, y), xytext=(x + 6, y - 0.55), fontsize=6.0, color=c,
                 va="center")
axD.set_xlabel(r"decoder readout change  $\Delta C$  (%)  [$\kappa$-$C$]")
axD.set_ylabel(r"latent reorganization  $\Delta\log$ eff. rank")
axD.set_xlim(-45, 48); axD.set_ylim(-6.2, 1.6)
axD.set_title("the dissociation", fontsize=8, loc="left", fontweight="bold")
panel_label(axD, "d")
axD.text(0.5, 0.95, "all tissues sit in the decoder null band,\nbut only hematopoietic reorganizes Z",
         transform=axD.transAxes, ha="center", va="top", fontsize=6, color=RED)
for s in ("top", "right"):
    axD.spines[s].set_visible(False)
axD.tick_params(length=2.2, width=0.5)

fig.savefig(os.path.join(BASE, "fig_concept_latent.pdf"))
fig.savefig(os.path.join(BASE, "fig_concept_latent.png"))
print("saved fig_concept_latent.pdf/png")
