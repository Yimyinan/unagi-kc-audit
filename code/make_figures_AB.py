#!/usr/bin/env python3
"""
Figure A (discovery figure) + Figure B (latent-geometry figure) for the
re-framed Paper 1 narrative.

Figure A -- "kappa-C as an observing instrument":
    Row 1: tr(H) across IDREM iterations, prior-injected (solid) vs vanilla (dashed),
           one panel per tissue (neural / lung / hematopoietic).
    Row 2: C(t) = 100 * (tr_V - tr_P) / tr_V, with the vanilla-equivalence band
           (kappa in [0.937, 1.064]  ->  C in [-6.4%, +6.3%]) and the C > 50% line.

Figure B -- "geometry at latent-dimension resolution":
    (a) per-dimension redistribution of spectral mass, share_P - share_V (64 x 5)
    (b) effective spectral rank eff_rank(t)
    (c) leading-axis concentration top1_share(t)
    (d) the neural dim_55 developmental axis under vanilla vs PG-UNAGI

Style matches the existing paper figures (Arial/Helvetica/DejaVu, 7.5 pt base,
6.7 in canvas, tissue palette #4C72B0 / #DD8452 / #55A868, red #C44E52, 300 dpi).
"""
from __future__ import annotations

import json
from pathlib import Path

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
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

OUTDIR = Path(r"F:/UNAGI_Project/outputs/kc_trajectory")
FIGDIR = OUTDIR

DEC_ROOT = Path(r"F:/UNAGI_Project/UNAGI/decoder_out")

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7.5, "axes.titlesize": 9, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.8,
    "axes.linewidth": 0.6, "xtick.major.width": 0.4, "ytick.major.width": 0.4,
    "xtick.direction": "in", "ytick.direction": "in", "axes.unicode_minus": False,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

C_HEMA, C_NEURAL, C_LUNG = "#4C72B0", "#DD8452", "#55A868"
RED, GRAY = "#C44E52", "#888888"
TISSUE_COLOR = {"neural": C_NEURAL, "lung": C_LUNG, "hemo": C_HEMA}
TISSUE_NAME = {"neural": "Neural development", "lung": "Lung fibrosis (IPF)",
               "hemo": "Hematopoietic"}

# kappa in [0.937, 1.064]  ->  C = (1 - kappa) * 100
VAN_EQ_LO, VAN_EQ_HI = (1 - 1.064) * 100, (1 - 0.937) * 100   # -6.4 .. +6.3


def style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main():
    traj = pd.read_csv(OUTDIR / "kappa_trajectory.csv")
    raw = json.loads((OUTDIR / "trajectory.json").read_text(encoding="utf-8"))
    share = {(r["config"], r["iteration"]): np.array(r["share"]) for r in raw if r.get("share")}

    # ============================ FIGURE A ============================
    fig, axes = plt.subplots(2, 3, figsize=(6.7, 4.5))
    order = ["neural", "lung", "hemo"]

    for j, tissue in enumerate(order):
        g = traj[traj.tissue == tissue]
        col = TISSUE_COLOR[tissue]

        # ---- row 1: tr(H) ----
        ax = axes[0, j]
        van = g.drop_duplicates("vanilla").vanilla.iloc[0]
        vg = g.drop_duplicates("iteration").sort_values("iteration")
        ax.plot(vg.iteration, vg.tr_V, "--", color=GRAY, lw=1.1, marker="s", ms=3.2,
                label=f"vanilla ({van})")
        for prior, gp in g.groupby("prior", sort=False):
            gp = gp.sort_values("iteration")
            n_cfg = g.prior.nunique()
            ax.plot(gp.iteration, gp.tr_P, "-", color=col, lw=1.3 if n_cfg == 1 else 1.0,
                    marker="o", ms=3.2,
                    alpha=1.0 if n_cfg == 1 else 0.85,
                    label=f"prior: {prior}" if n_cfg > 1 else f"prior ({prior})")
        ax.set_title(f"{TISSUE_NAME[tissue]}", fontweight="bold", fontsize=8.5)
        if j == 0:
            ax.set_ylabel(r"$\mathrm{tr}(\mathbf{H})=\|\mathbf{W}\|_F^2$")
        ax.set_xticks(range(5))
        ax.set_xlabel("IDREM iteration")
        ax.legend(frameon=False, loc="best", handlelength=1.4)
        style_ax(ax)

        # ---- row 2: C(t) ----
        ax = axes[1, j]
        ax.axhspan(VAN_EQ_LO, VAN_EQ_HI, color="#EEF2F7", zorder=0)
        ax.axhline(0, color=GRAY, ls=":", lw=0.7, zorder=1)
        ax.axhline(50, color=RED, ls="--", lw=0.9, zorder=1)
        for prior, gp in g.groupby("prior", sort=False):
            gp = gp.sort_values("iteration")
            n_cfg = g.prior.nunique()
            ax.plot(gp.iteration, gp.C, "-", color=col, lw=1.3 if n_cfg == 1 else 1.0,
                    marker="o", ms=3.2, alpha=1.0 if n_cfg == 1 else 0.85,
                    label=prior if n_cfg > 1 else None)
        ax.set_xticks(range(5))
        ax.set_xlabel("IDREM iteration")
        if j == 0:
            ax.set_ylabel(r"compression $C$ (\%)")
        if n_cfg > 1:
            ax.legend(frameon=False, fontsize=5.8, ncol=2, handlelength=1.1)
        style_ax(ax)

    # panel labels + the 0/38 statement
    for i, lab in enumerate("abcdef"):
        axes.flat[i].text(-0.16, 1.10, lab, transform=axes.flat[i].transAxes,
                          fontsize=11, fontweight="bold", va="top")
    axes[1, 1].text(4.05, 32, "no observation\nreaches $C>50\\%$", fontsize=6.0,
                    color=RED, ha="right", va="top",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.85,
                              pad=1.5))
    axes[0, 2].set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(FIGDIR / "figA_idrem_trajectory.png")
    fig.savefig(FIGDIR / "figA_idrem_trajectory.pdf")
    plt.close(fig)
    print("wrote figA_idrem_trajectory.{png,pdf}")

    # ============================ FIGURE B ============================
    fig, axes = plt.subplots(2, 2, figsize=(6.7, 4.6),
                             gridspec_kw={"width_ratios": [1.25, 1]})

    # (a) lung R9 per-dimension L2 norm across iterations -- absolute scale,
    #     reveals concentration on the (dim 55, dim 3) polarity.
    ax = axes[0, 0]
    its = [0, 1, 2, 3, 4]
    M = np.zeros((64, len(its)))
    for k, it in enumerate(its):
        A = np.load(DEC_ROOT / f"R9/iter_{it}/loading_matrix.npy")
        # A has shape (D, G); per-dimension L2 norm = ||A[d, :]||_2
        M[:, k] = np.linalg.norm(A, axis=1)
    # sort dims by mean L2 to reveal the rank ordering at a glance
    dim_order = np.argsort(M.mean(axis=1))[::-1]
    M = M[dim_order, :]
    im = ax.imshow(M, aspect="auto", cmap="magma", origin="lower",
                   extent=[-0.5, len(its) - 0.5, -0.5, 63.5])
    ax.set_xticks(range(len(its)))
    ax.set_xticklabels([f"iter {i}" for i in its])
    ax.set_xlabel("IDREM iteration")
    ax.set_ylabel("latent dimension (sorted by mean L2)")
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label(r"$\Vert \mathbf{W}_{d,:}\Vert_2$ (a.u.)", fontsize=6.5)
    cb.ax.tick_params(labelsize=6)
    # locate dim 55 and dim 3 in the sorted y-axis
    pos_55 = int(np.where(dim_order == 55)[0][0])
    pos_3 = int(np.where(dim_order == 3)[0][0])
    for d, pos in ((55, pos_55), (3, pos_3)):
        ax.axhline(pos, color="white", ls=":", lw=0.8, alpha=0.9)
        ax.text(len(its) - 0.4, pos, f" dim {d}", fontsize=6.5, va="center",
                ha="left", color="white", fontweight="bold")
    ax.set_title("Lung IPF: dim 55 / dim 3 lead from iter 0", fontweight="bold",
                 fontsize=8.5)

    # (b) effective spectral rank: tissue means
    ax = axes[0, 1]
    for tissue in order:
        g = traj[traj.tissue == tissue]
        col = TISSUE_COLOR[tissue]
        vg = g.drop_duplicates("iteration").sort_values("iteration")
        ax.plot(vg.iteration, vg.eff_rank_V, "--", color=col, lw=1.1, alpha=0.7)
        # tissue-mean of prior
        pm = g.groupby("iteration").eff_rank_P.mean().reindex(vg.iteration)
        ax.plot(vg.iteration, pm, "-", color=col, lw=1.4, marker="o", ms=3.0)
    # manual entries for legend
    handles = [plt.Line2D([0], [0], color=col, lw=1.4, label=TISSUE_NAME[tissue],
                          linestyle="-") for tissue, col in
               [("neural", C_NEURAL), ("lung", C_LUNG), ("hemo", C_HEMA)]]
    handles.append(plt.Line2D([0], [0], color=GRAY, lw=1.1,
                              label="vanilla (dashed, per tissue)", linestyle="--"))
    ax.set_xlabel("IDREM iteration")
    ax.set_ylabel("effective spectral rank")
    ax.set_xticks(range(5))
    ax.set_title("Spectrum de-concentrates across iterations", fontweight="bold", fontsize=8.5)
    ax.legend(handles=handles, frameon=False, fontsize=5.8, handlelength=1.2,
              loc="upper left")
    style_ax(ax)

    # (c) leading-axis concentration: tissue means
    ax = axes[1, 0]
    for tissue in order:
        g = traj[traj.tissue == tissue]
        col = TISSUE_COLOR[tissue]
        vg = g.drop_duplicates("iteration").sort_values("iteration")
        ax.plot(vg.iteration, vg.top1_V, "--", color=col, lw=1.1, alpha=0.7)
        pm = g.groupby("iteration").top1_P.mean().reindex(vg.iteration)
        ax.plot(vg.iteration, pm, "-", color=col, lw=1.4, marker="o", ms=3.0)
    ax.set_xlabel("IDREM iteration")
    ax.set_ylabel(r"leading-axis share $\lambda_1/\mathrm{tr}(\mathbf{H})$")
    ax.set_xticks(range(5))
    ax.set_title("Leading axis weakens with iterations", fontweight="bold", fontsize=8.5)
    style_ax(ax)

    # (d) neural dim_55 developmental axis: vanilla collapses it,
    #     PG-UNAGI arrests the collapse at the last iteration.
    ax = axes[1, 1]
    g = traj[(traj.tissue == "neural")].sort_values("iteration")
    ax.plot(g.iteration, g.dim55_V, "--", color=GRAY, lw=1.1, marker="s", ms=3.2,
            label="vanilla (R0_seed42)")
    ax.plot(g.iteration, g.dim55_P, "-", color=C_NEURAL, lw=1.3, marker="o", ms=3.2,
            label="PG-UNAGI (R1_seed42)")
    ax.set_xlabel("IDREM iteration")
    ax.set_ylabel("dim-55 loading share (%)")
    ax.set_xticks(range(5))
    ax.set_title("Developmental axis (dim 55)", fontweight="bold", fontsize=8.5)
    ax.legend(frameon=False, fontsize=5.8, handlelength=1.2, loc="upper right")
    style_ax(ax)
    # clean annotation: the ratio of PG-UNAGI / vanilla at iterations 3 and 4
    r = g.drop_duplicates("iteration").set_index("iteration")
    if 3 in r.index and 4 in r.index:
        r3 = r.loc[3, "dim55_P"] / r.loc[3, "dim55_V"]
        r4 = r.loc[4, "dim55_P"] / r.loc[4, "dim55_V"]
        ax.text(0.02, 0.06,
                f"PG-UNAGI / vanilla ratio\n"
                f"iter 3: {r3:.2f}$\\times$\n"
                f"iter 4: {r4:.2f}$\\times$",
                transform=ax.transAxes, fontsize=6.0, color=C_NEURAL,
                va="bottom", ha="left",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1.5))

    for i, lab in enumerate("abcd"):
        axes.flat[i].text(-0.16, 1.10, lab, transform=axes.flat[i].transAxes,
                          fontsize=11, fontweight="bold", va="top")
    fig.tight_layout()
    fig.savefig(FIGDIR / "figB_latent_geometry.png")
    fig.savefig(FIGDIR / "figB_latent_geometry.pdf")
    plt.close(fig)
    print("wrote figB_latent_geometry.{png,pdf}")


if __name__ == "__main__":
    main()
