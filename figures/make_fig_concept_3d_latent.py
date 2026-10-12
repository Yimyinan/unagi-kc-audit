# -*- coding: utf-8 -*-
"""
Fig. CONCEPT_3D (v2) -- "Dual-spectrum view of GRN-prior injection."

A publication-quality 3-D coordinate-grid concept figure tailored to
paper1's central claim:
  * The latent manifold Z is reorganized by GRN-prior injection
    (red surface) relative to the vanilla VAE (blue surface).
  * Despite the Z reorganization, the decoder readout W is unchanged;
    the same decoder spectrum is recovered from both latent
    configurations.
  * kappa-C is a decoder-only (W-only) diagnostic; therefore it
    reports 0/33 null, even though a Z-spectrum diagnostic would
    detect tissue-specific reorganization.

This figure is intended as a Graphical Abstract / TOC image, or as a
concept panel to anchor the Results section.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(BASE), "code"))
import figure_palette as fp

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 0.6,
    "axes.unicode_minus": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

rng = np.random.default_rng(20260905)

# ---- latent axis labels as shorthand ----
z1_label = r"latent axis  $z_1$"
z2_label = r"latent axis  $z_2$"
z3_label = r"latent axis  $z_3$"

# ---- vanilla manifold: a smooth curved sheet in R^3 ----
def vanilla_manifold(u, v):
    """Parametric curved sheet."""
    x = u
    y = v
    z = 0.45 * np.sin(2.0 * u) + 0.22 * np.cos(2.0 * v)
    return x, y, z

# ---- GRN-prior manifold: same sheet, reorganized (rotated + sheared + compressed) ----
def prior_manifold(u, v):
    """Reorganized version of the same sheet."""
    x, y, z = vanilla_manifold(u, v)
    theta = np.radians(35)
    # rotation around z3 plus a shear that compresses z2 and tilts z3
    xr = x * np.cos(theta) - y * np.sin(theta)
    yr = 0.55 * (x * np.sin(theta) + y * np.cos(theta))
    zr = z + 0.35 * y
    # global shift so the two surfaces are visually separated
    xr += 0.35
    yr += 0.25
    zr += 0.15
    return xr, yr, zr

# ---- dense grid for surfaces ----
U = np.linspace(-1.5, 1.5, 35)
V = np.linspace(-1.5, 1.5, 35)
U, V = np.meshgrid(U, V)
Xv, Yv, Zv = vanilla_manifold(U, V)
Xp, Yp, Zp = prior_manifold(U, V)

# ---- sampled cell clouds ----
N = 500
u = rng.uniform(-1.4, 1.4, size=N)
v = rng.uniform(-1.4, 1.4, size=N)

xv, yv, zv = vanilla_manifold(u, v)
xv += rng.normal(0, 0.08, N)
yv += rng.normal(0, 0.08, N)
zv += rng.normal(0, 0.08, N)

xp, yp, zp = prior_manifold(u, v)
xp += rng.normal(0, 0.08, N)
yp += rng.normal(0, 0.08, N)
zp += rng.normal(0, 0.08, N)

# ---- base cell: same (u,v) on both manifolds ----
base_u, base_v = 0.0, 0.0
bx, by, bz = vanilla_manifold(base_u, base_v)
px, py, pz = prior_manifold(base_u, base_v)

# ---- figure ----
# Explicit two-column geometry (left: 3-D scene, right: the 2-D proof + the verdict), and NO
# tight-bbox on save -- with a 3-D axes, tight-bbox recomputes the extent and slid the inset
# partly off the canvas in the previous attempt.
FIG_W, FIG_H = 6.6929, 3.66   # = \textwidth; keeps font sizes at print size
fig = plt.figure(figsize=(FIG_W, FIG_H))
ax = fig.add_axes([0.005, 0.055, 0.665, 0.885], projection="3d")

# Palette contract: IDENTITY colours carry arm identity, VERDICT colours carry the verdict.
# The previous version painted the GRN-prior arm in RED -- the same red that Fig. 1 uses for a
# FAILED check -- so a reader who learned "red = failure" met "red = prior" here. The prior arm
# is now ARM_PRIOR (purple); red appears exactly once, on the 0/33 null verdict.
C_VAN, C_PRI, C_W = fp.ARM_VANILLA, fp.ARM_PRIOR, fp.C_DEC

# clean background
ax.set_facecolor("white")
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
    axis.pane.set_edgecolor(fp.EDGE)
    axis.pane.set_alpha(0.25)
ax.grid(True, color=fp.GRAY, alpha=0.35, linestyle="-", linewidth=0.75)

# ---- vanilla surface (faint mesh) ----
ax.plot_surface(Xv, Yv, Zv, rstride=4, cstride=4, color=C_VAN,
                alpha=0.10, edgecolor=C_VAN, linewidth=0.75, shade=False, zorder=1)

# ---- GRN-prior surface (faint mesh, same hue as its cloud) ----
ax.plot_surface(Xp, Yp, Zp, rstride=4, cstride=4, color=C_PRI,
                alpha=0.11, edgecolor=C_PRI, linewidth=0.75, shade=False, zorder=1)

# ---- sampled cell clouds ----
ax.scatter(xv, yv, zv, s=11, c=C_VAN, alpha=0.42,
           label="vanilla latent cloud", depthshade=True, zorder=2)
ax.scatter(xp, yp, zp, s=11, c=C_PRI, alpha=0.48,
           label="GRN-prior latent cloud", depthshade=True, zorder=2)

# ---- base cell and its reorganized counterpart ----
ax.scatter([bx], [by], [bz], s=150, c=C_VAN, marker="*",
           edgecolors="white", linewidths=1, label="base cell", zorder=6)
ax.scatter([px], [py], [pz], s=110, c=C_PRI, marker="*",
           edgecolors="white", linewidths=1, label="reorganized cell", zorder=6)

# reorganization arrow: the prior arm's movement
ax.quiver(bx, by, bz, px - bx, py - by, pz - bz,
          color=C_PRI, arrow_length_ratio=0.22, linewidths=2,
          length=1.0, normalize=False, zorder=5)
ax.text(bx - 0.05, by + 0.55, bz + 0.42,
        "prior-driven\nreorganization",
        color=C_PRI, fontsize=7.5, fontweight="bold", zorder=6,
        ha="right", va="bottom")

# ---- fixed decoder-readout plane W ----
z_w = -1.55
w_xx, w_yy = np.meshgrid(np.linspace(-2.0, 2.0, 12), np.linspace(-2.0, 2.0, 12))
w_zz = np.full_like(w_xx, z_w)
ax.plot_surface(w_xx, w_yy, w_zz, color=C_W, alpha=0.10,
                edgecolor=C_W, linewidth=0.75, shade=False, zorder=0)

# ---- decoder spectra on W: identical for vanilla and GRN-prior ----
s = np.linspace(-1.8, 1.8, 200)
spectrum = 0.55 * np.exp(-0.5 * ((s + 0.4) / 0.35) ** 2) + \
           0.35 * np.exp(-0.5 * ((s - 0.6) / 0.30) ** 2) + 0.08
delta = 0.02 * np.sin(4 * s)

ax.plot(s, np.full_like(s, 1.55), z_w + spectrum, color=C_VAN,
        linewidth=2, alpha=0.95, zorder=3, label="vanilla decoder spectrum")
ax.plot(s, np.full_like(s, 1.55), z_w + spectrum + delta,
        color=C_PRI, linewidth=2.0, linestyle="--", alpha=0.95,
        zorder=3, label="GRN-prior decoder spectrum")

# Label the W plane at its FRONT-LEFT corner, just above the surface: the centre of the plane
# is covered by the cell clouds, which is what hid this label in the previous version.
ax.text(-1.95, -1.95, z_w + 0.30, "decoder readout $W$  (identical spectrum)",
        color=fp.PASS, fontsize=8, fontweight="bold", zorder=4, ha="left", va="bottom")

# ---- inset: the claim that could not be verified in a 3-D perspective ----------------
# "the two spectra are identical" is the whole point, and in this projection the two curves
# nearly coincide along the view direction. A flat inset makes the claim checkable.
axin = fig.add_axes([0.725, 0.395, 0.255, 0.395])
axin.plot(s, spectrum, color=C_VAN, linewidth=1.5, label="vanilla")
axin.plot(s, spectrum + delta, color=C_PRI, linewidth=1.5, linestyle="--", label="GRN-prior")
axin.set_xlim(-1.9, 1.9)
axin.set_ylim(-0.03, 1.12)
axin.set_yticks([])
axin.set_xticks([])
axin.tick_params(length=0)
for side in ("top", "right"):
    axin.spines[side].set_visible(False)
for side in ("left", "bottom"):
    axin.spines[side].set_color(fp.EDGE)
    axin.spines[side].set_linewidth(0.6)
axin.set_title("decoder spectrum $W$", fontsize=7.5, color=fp.INK, pad=3)
axin.text(0.5, -0.13, r"max $|\Delta|$ = %.3f  (indistinguishable)" % np.abs(delta).max(),
          transform=axin.transAxes, ha="center", va="top", fontsize=7, color=fp.MUTED)
axin.legend(loc="upper right", fontsize=6, frameon=False,
            handlelength=1.2, borderaxespad=0.1)
axin.set_facecolor("white")

# ---- axes and limits ----
ax.set_xlabel(z1_label, labelpad=5, fontsize=9)
ax.set_ylabel(z2_label, labelpad=5, fontsize=9)
ax.set_zlabel(z3_label, labelpad=5, fontsize=9)
ax.set_xlim(-2.0, 2.0)
ax.set_ylim(-2.0, 2.0)
ax.set_zlim(-1.7, 1.25)

# ---- title ----
ax.set_title("Dual-spectrum view of GRN-prior injection\n"
             r"$Z$ reorganizes; the $W$ readout stays in the vanilla band",
             fontsize=11.5, fontweight="bold", pad=14)

# ---- key-claim text box: pinned to the reserved bottom band, clear of the inset ----
# Red appears exactly once in this figure: as the outline of the verdict box. Painting the
# GRN-prior arm red would collide with Fig. 1, where red means a FAILED check.
claim_text = (r"$\kappa$-$C$ reads $W$ only: 0/33 null" + "\n"
              r"$Z$ spectrum: tissue-specific reorganization")
fig.text(0.855, 0.115, claim_text, fontsize=8.0, color=fp.INK,
         ha="center", va="top", linespacing=1.6,
         bbox=dict(boxstyle="round,pad=0.36", fc="white", ec=fp.FAIL, lw=1, alpha=0.97))

# ---- legend: below the 3-D scene, two columns, so it cannot occlude the clouds ----
legend = ax.legend(loc="upper center", bbox_to_anchor=(0.50, -0.005), ncol=2,
                   frameon=False, fontsize=7, handlelength=1.6, columnspacing=1.4)

ax.view_init(elev=20, azim=-58)

# ---- save ----
# bbox_inches=None: the geometry above is already exact, and tight-bbox with a 3-D axes
# recomputes the extent and shifts the inset panel.
fig.savefig(os.path.join(BASE, "fig_concept_3d_latent.pdf"), bbox_inches=None)
fig.savefig(os.path.join(BASE, "fig_concept_3d_latent.png"), dpi=300, bbox_inches=None)
fig.savefig(os.path.join(BASE, "graphical_abstract.png"), dpi=300, bbox_inches=None)
print("saved fig_concept_3d_latent.pdf/png + graphical_abstract.png (v4)")

