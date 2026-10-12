# -*- coding: utf-8 -*-
"""
Fig. CONCEPT_3D -- "Latent manifold under GRN-prior injection: a 3-D
spectral-audit view."

A conceptual 3-D coordinate-grid figure in the style of the paper2
manifold perturbation demo, but tailored to paper1's dual-spectrum story:
  * A 3-D latent coordinate box (z1, z2, z3) with a sampled cell cloud
    lying on a curved manifold.
  * Vanilla cloud (blue) vs. GRN-prior-injected cloud (magenta): the
    prior collapses / reorganizes the latent manifold while a fixed
    decoder-readout plane (W, green grid) stays unchanged.
  * A base cell (black star), a random latent perturbation (blue),
    and the prior-driven reorganization direction (magenta arrow).

This is intended as a Graphical-Abstract / TOC figure, or as a
conceptual panel to anchor Fig. 1 / the S22 concept figure.
"""
import os
import numpy as np
import matplotlib
import sys as _sys, os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
for _p in (_HERE, _os.path.join(_HERE, "..", "code"), _os.path.join(_HERE, "code")):
    if _os.path.isdir(_p) and _p not in _sys.path:
        _sys.path.insert(0, _p)
import figure_palette as fp
fp.apply_paper_style()
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

BASE = os.path.dirname(os.path.abspath(__file__))

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

# ---- manifold parametrization: a curved sheet in 3-D ----
def manifold(u, v):
    """Curved 2-D sheet embedded in R^3."""
    x = u
    y = v
    z = 0.55 * np.sin(2.2 * u) + 0.25 * v
    return x, y, z

# ---- sample clouds on the manifold ----
N = 400
u = rng.uniform(-1.2, 1.2, size=N)
v = rng.uniform(-1.2, 1.2, size=N)

# vanilla cloud: isotropic jitter around the manifold
jitter = 0.12
xv, yv, zv = manifold(u, v)
xv += rng.normal(0, jitter, N)
yv += rng.normal(0, jitter, N)
zv += rng.normal(0, jitter, N)

# prior cloud: collapse / reorganize along one latent direction
#             and bend the sheet differently
collapse = 0.18  # squeeze v direction
u_p = u
v_p = v * collapse
xp, yp, zp = manifold(u_p, v_p)
# additional prior-driven shift: collapse toward a ridge along z1
shift = 0.45 * (u_p ** 2) * np.sign(u_p)
zp += shift
xp += rng.normal(0, 0.08, N)
yp += rng.normal(0, 0.04, N)
zp += rng.normal(0, 0.10, N)

# ---- base point and perturbation vectors ----
base_u, base_v = 0.0, 0.0
bx, by, bz = manifold(base_u, base_v)
bx += 0.02
by += 0.02
bz += 0.02

# Random latent perturbation vector (off-manifold-ish direction)
rand_vec = np.array([0.55, 0.25, 0.65])
rand_vec /= np.linalg.norm(rand_vec)
rand_vec *= 0.85
rx, ry, rz = bx + rand_vec[0], by + rand_vec[1], bz + rand_vec[2]

# Prior-driven reorganization direction (along the latent collapse axis)
prior_vec = np.array([0.25, -0.05, -0.75])
prior_vec /= np.linalg.norm(prior_vec)
prior_vec *= 1.10
px, py, pz = bx + prior_vec[0], by + prior_vec[1], bz + prior_vec[2]

# ---- figure ----
fig = plt.figure(figsize=(7.0, 5.8))
ax = fig.add_subplot(111, projection="3d")

# light background pane (like reference)
ax.set_facecolor("#F8F8F8")
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor("#CCCCCC")
ax.yaxis.pane.set_edgecolor("#CCCCCC")
ax.zaxis.pane.set_edgecolor("#CCCCCC")
ax.xaxis.pane.set_alpha(0.3)
ax.yaxis.pane.set_alpha(0.3)
ax.zaxis.pane.set_alpha(0.3)
ax.grid(True, color="#CCCCCC", alpha=0.5, linestyle="-", linewidth=0.75)

# plot manifold surface wireframe (faint)
U = np.linspace(-1.4, 1.4, 30)
V = np.linspace(-1.4, 1.4, 30)
U, V = np.meshgrid(U, V)
Xs, Ys, Zs = manifold(U, V)
ax.plot_surface(Xs, Ys, Zs, rstride=4, cstride=4, color="#E8E8E8",
                alpha=0.35, edgecolor="#BBBBBB", linewidth=0.75,
                shade=False, zorder=1)

# sampled clouds
ax.scatter(xv, yv, zv, s=12, c="#4C72B0", alpha=0.58,
           label="vanilla latent cloud", depthshade=True, zorder=2)
ax.scatter(xp, yp, zp, s=12, c="#C44E52", alpha=0.58,
           label="GRN-prior latent cloud", depthshade=True, zorder=2)

# base point
ax.scatter([bx], [by], [bz], s=140, c="k", marker="*",
           edgecolors="white", linewidths=0.75, label="base cell", zorder=5)

# random perturbation arrow (blue, dashed shaft, solid head)
ax.quiver(bx, by, bz, rand_vec[0], rand_vec[1], rand_vec[2],
          color="#4C72B0", arrow_length_ratio=0.25, linewidths=2,
          linestyles="dashed", length=1.0, normalize=False, zorder=4)
ax.scatter([rx], [ry], [rz], s=75, c="#4C72B0", marker="o",
           edgecolors="white", linewidths=0.75, zorder=5)
ax.text(rx + 0.08, ry + 0.12, rz + 0.05, "random\nperturb.",
        color="#4C72B0", fontsize=6, fontweight="bold", zorder=6)

# prior-driven reorganization arrow (magenta, solid thick)
ax.quiver(bx, by, bz, prior_vec[0], prior_vec[1], prior_vec[2],
          color="#C44E52", arrow_length_ratio=0.25, linewidths=2,
          length=1.0, normalize=False, zorder=4)
ax.scatter([px], [py], [pz], s=85, c="#C44E52", marker="^",
           edgecolors="white", linewidths=0.75, zorder=5)
ax.text(px - 0.05, py - 0.18, pz + 0.12, "prior-driven\nreorganization",
        color="#C44E52", fontsize=6, fontweight="bold", zorder=6,
        ha="right")

# fixed decoder-readout reference plane (W) -- a translucent green square
# behind the manifold, symbolizing that kappa-C reads the decoder spectrum
w_xx, w_yy = np.meshgrid(np.linspace(-1.6, 1.6, 10), np.linspace(-1.6, 1.6, 10))
w_zz = np.full_like(w_xx, -1.45)
ax.plot_surface(w_xx, w_yy, w_zz, color="#55A868", alpha=0.12,
                edgecolor="#55A868", linewidth=0.75, shade=False, zorder=0)
ax.text(-1.45, -1.35, -1.45, "decoder readout  $W$  (fixed)",
        color="#2F6B3A", fontsize=8, fontweight="bold",
        va="bottom", ha="left", zorder=3)

# axis labels and limits
ax.set_xlabel("latent axis  $z_1$", labelpad=4, fontsize=9)
ax.set_ylabel("latent axis  $z_2$", labelpad=4, fontsize=9)
ax.set_zlabel("latent axis  $z_3$", labelpad=4, fontsize=9)
ax.set_xlim(-1.8, 1.8)
ax.set_ylim(-1.8, 1.8)
ax.set_zlim(-1.6, 1.2)

# title / annotation
ax.set_title("GRN-prior injection reorganizes the latent manifold $Z$\n"
             "while the decoder readout $W$ stays in the vanilla band",
             fontsize=10.5, fontweight="bold", pad=10)

# key claim box in 3-D coordinates (upper-right region)
ax.text2D(0.98, 0.02,
          r"$\kappa$-$C$ reads $W$: 0/33 null" + "\n"
          r"$Z$ spectrum: tissue-specific reorganization",
          transform=ax.transAxes, fontsize=8, color="#333333",
          ha="right", va="bottom",
          bbox=dict(boxstyle="round,pad=0.25", fc="#FFFFFF",
                    ec="#888888", lw=0.75, alpha=0.95))

# legend (place in upper-left of axes)
legend = ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.88),
                   frameon=True, fontsize=8,
                   edgecolor="#888888", facecolor="white",
                   framealpha=0.92)
legend.get_frame().set_linewidth(0.7)

# viewing angle
ax.view_init(elev=22, azim=-58)

fig.savefig(os.path.join(BASE, "fig_concept_3d_latent.pdf"))
fig.savefig(os.path.join(BASE, "fig_concept_3d_latent.png"))
print("saved fig_concept_3d_latent.pdf/png")
