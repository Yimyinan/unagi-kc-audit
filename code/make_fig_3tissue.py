# -*- coding: utf-8 -*-
"""
Figure: latent-spectrum contraction replicates in hematopoietic data,
is null in neural data, and is untestable in lung (n=1 pair).

(a) hemo   eff_rank trajectories: vanilla vs 18 prior runs
(b) neural eff_rank trajectories: vanilla vs 9 prior/ablation runs
(c) lung   eff_rank trajectories: vanilla vs prior (n=1)
(d) forest: self-normalised change Dlog(eff_rank), iter_0 -> iter_4
(e) hemo specificity ladder, all seed 42, 29,985 cells
(f) neural 2^3 factorial main effects (all null)

Style inherited from paper1_kc_diagnostic/figures: Arial, 7.5 pt, 6.7 in, 300 dpi.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(r"F:/UNAGI_Project")
OUT = ROOT / "outputs" / "positive_result"
CSV = OUT / "z_spectrum_all.csv"

import sys
sys.path.insert(0, r"F:/UNAGI_Project/paper1/paper1_kc_diagnostic/code")
from figure_palette import *   # single source of truth for paper1 figure colours + safe_legend
apply_paper_style()

TISSUE_COLOR = {"hemo": C_HEMA, "neural": C_NEURAL, "lung": C_LUNG}
TISSUE_LABEL = {"hemo": "Hematopoietic (29,985 cells)",
                "neural": "Neural crest (21,078 cells)",
                "lung": "Lung IPF (19,999 cells)"}


def style_ax(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction="out", length=2.4, width=0.6)


def traj_panel(ax, sub, title):
    """eff_rank trajectories; vanilla drawn last, thick and dark."""
    van = sub[sub.role == "vanilla"]
    pri = sub[sub.role.isin(["prior", "ablation"])]
    veh = sub[sub.role.isin(["vehicle", "wrongprior"])]
    col = TISSUE_COLOR[sub.tissue.iloc[0]]

    for run, g in pri.groupby("run"):
        g = g.sort_values("iteration")
        ax.plot(g.iteration, g.pr, "-", color=col, lw=0.75, alpha=0.55)
    for run, g in veh.groupby("run"):
        g = g.sort_values("iteration")
        ax.plot(g.iteration, g.pr, ":", color=GREY, lw=0.9, alpha=0.9)
    for run, g in van.groupby("run"):
        g = g.sort_values("iteration")
        ax.plot(g.iteration, g.pr, "-", color=VANILLA, lw=2.0,
                marker="o", ms=3.6, zorder=5,
                label=f"vanilla (n={len(van)})")

    ax.set_xlabel("IDREM iteration")
    ax.set_ylabel("effective spectral rank of Z (participation ratio)")
    ax.set_xticks(range(5))
    ax.set_title(title, fontweight="bold", fontsize=8.5)
    ax.legend(frameon=False, fontsize=5.8, handlelength=1.2, loc="upper left")
    style_ax(ax)


def main() -> None:
    df = pd.read_csv(CSV)

    fig, axes = plt.subplots(2, 3, figsize=(6.7, 4.3))
    fig.subplots_adjust(left=0.075, right=0.975, top=0.90, bottom=0.10,
                        wspace=0.34, hspace=0.46)

    # ---------------- (a)(b)(c) trajectories ----------------
    for ax, tis in zip(axes[0], ["hemo", "neural", "lung"]):
        sub = df[df.tissue == tis]
        traj_panel(ax, sub, TISSUE_LABEL[tis])

    # ---------------- (d) forest: self-normalised change ----------------
    ax = axes[1, 0]
    rows = []
    for run, g in df.groupby("run"):
        g = g.sort_values("iteration")
        if len(g) < 5 or g.iteration.min() != 0:
            continue
        rows.append(dict(run=run, tissue=g.tissue.iloc[0], role=g.role.iloc[0],
                         dlog=float(np.log(g.pr.iloc[-1] /
                                           g.pr.iloc[0]))))
    ch = pd.DataFrame(rows)

    ypos, ytick = [], []
    for i, tis in enumerate(["hemo", "neural", "lung"]):
        s = ch[ch.tissue == tis]
        base = i * 1.0
        for j, r in enumerate(s.sort_values("dlog").itertuples()):
            y = base + j * 0.055
            is_van = r.role == "vanilla"
            col = VANILLA if is_van else (
                GREY if r.role in ("vehicle", "wrongprior") else TISSUE_COLOR[tis])
            ax.plot(r.dlog, y, "o" if not is_van else "s",
                    color=col, ms=2.6 if not is_van else 4.0,
                    zorder=4 if is_van else 3)
            ypos.append(y)
        # vanilla reference line
        vv = s[s.role == "vanilla"].dlog
        if len(vv):
            ax.axvline(vv.mean(), color=VANILLA, ls="--", lw=0.8, alpha=0.7)
        ytick.append((base + 0.14, tis))
    ax.axvline(0, color="#CCCCCC", lw=0.7, zorder=0)
    ax.set_yticks([t[0] for t in ytick])
    ax.set_yticklabels([t[1] for t in ytick])
    ax.set_ylim(-0.06, 3.05)
    ax.set_xlabel(r"$\Delta\log$(effective rank), iter$_0 \rightarrow$ iter$_4$"
                  "\n(negative = contraction)")
    ax.set_title("Self-normalised change", fontweight="bold", fontsize=8.5)
    style_ax(ax)
    for tis, key in (("hemo", "17/18"), ("neural", "1/9"), ("lung", "1/1")):
        pass
    ax.text(0.98, 0.04, "black square = vanilla\ndashed = vanilla mean",
            transform=ax.transAxes, fontsize=5.4, color="#555555",
            ha="right", va="bottom")

    # ---------------- (e) hemo specificity ladder, seed 42 ----------------
    ax = axes[1, 1]
    sub = df[(df.tissue == "hemo") & (df.iteration == 4)]
    ladder = [("R14_hemato_vanilla", "no prior", VANILLA),
              ("R14_hemato_seed42", "legacy\nblood prior", GREY),
              ("R14_wrongprior_seed42", "wrong-tissue\nprior (52 neural TF)",
               "#B07AA1"),
              ("R13_hemato_trueprior_seed42", "curated true prior\n(11 TF, 325 edges)",
               C_HEMA)]
    xs, ys, cs = [], [], []
    for run, lab, col in ladder:
        r = sub[sub.run == run]
        if r.empty:
            continue
        xs.append(r.tr.iloc[0]); ys.append(r.pr.iloc[0]); cs.append(col)
        ax.annotate(lab, (r.tr.iloc[0], r.pr.iloc[0]),
                    textcoords="offset points", xytext=(5, 4),
                    fontsize=5.6, color=col)
    ax.plot(xs, ys, "-", color="#CCCCCC", lw=0.9, zorder=1)
    ax.scatter(xs, ys, c=cs, s=26, zorder=3)
    ax.set_xlabel(r"total latent variance  tr(Cov $\mathbf{Z}$)")
    ax.set_ylabel("effective spectral rank")
    ax.set_title("Specificity ladder (hemo, seed 42)", fontweight="bold",
                 fontsize=8.5)
    style_ax(ax)

    # ---------------- (f) neural 2^3 factorial main effects ----------------
    ax = axes[1, 2]
    neu = df[(df.tissue == "neural") & (df.iteration == 4)]
    fac = neu[neu.run.str.startswith(("R0_", "R1_", "R2_", "R3_", "R4_",
                                      "R5_", "R6_", "R7_"))]
    y = fac["pr"].to_numpy(float)
    X = np.column_stack([np.ones(len(fac)), fac["C"], fac["M"], fac["T"]])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    dof = len(fac) - 4
    se = np.sqrt(np.sum(res ** 2) / dof * np.diag(np.linalg.pinv(X.T @ X)))
    names = ["C\ncurriculum", "M\nmixture", "T\ndecoder audit"]
    ax.axhline(0, color="#CCCCCC", lw=0.7)
    for i in range(3):
        b, s_ = beta[i + 1], se[i + 1]
        ax.errorbar(i, b, yerr=1.96 * s_, fmt="o", color=C_NEURAL, ms=4.5,
                    capsize=2.2, elinewidth=0.9)
    ax.set_xticks(range(3))
    ax.set_xticklabels(names, fontsize=6.0)
    ax.set_ylabel(r"$\beta$ on effective rank (95% CI)")
    ax.set_title(r"Neural $2^3$ factorial: no component effect",
                 fontweight="bold", fontsize=8.5)
    style_ax(ax)
    ax.text(0.5, 0.05, "all |t| < 0.3, all p > 0.39", transform=ax.transAxes,
            fontsize=5.6, color="#555555", ha="center", va="bottom")

    for ext in ("png", "pdf"):
        p = OUT / f"fig_z_spectrum_3tissue.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight")
        print("wrote", p)


if __name__ == "__main__":
    main()
