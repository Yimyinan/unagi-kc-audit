# -*- coding: utf-8 -*-
r"""make_fig5_injection_path.py —— 新建 Fig 5：注入路径的实证图（方案 A）。

素材**全部来自已审计的表** `tab:info_decay`（supp_addendum.tex）+ 正文 L117/L119/L121，
**零新增实验、零新增数据**。

设计（4 面板，2x2）：
  (A) 损耗漏斗：9,396 → 9,369 → 325 → 203 → **0**（逐阶段保留率）
  (B) 调用图终止：先验库 → 逐基因向量 → `_tf_prior_weights` 赋值 → **无读取者**
  (C) 零权重层实测：全部 796 个 stage 文件（29 run-iteration，覆盖 vanilla/curated-prior/
      mismatched-prior/erythroid-ablation 各臂）nnz = 0（对照非零）
  (D) 覆盖缺口：ENCODE 参考库缺 6/12 造血因子，KLF1 无任何边

排版约束：画布宽 = \textwidth = 6.6929 in（否则 LaTeX 缩图会把字号/线宽一起压小）。
配色 = figure_palette（Okabe–Ito）。柱宽 0.55。
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "code"))
import figure_palette as fp  # noqa: E402

OUT = os.path.join(HERE, "fig5_injection_path.pdf")
OUT_PNG = os.path.join(HERE, "fig5_injection_path.png")

# ---- 已审计数据（tab:info_decay + 正文 L117/L119/L121）----
STAGES = [
    ("TRRUST v2 human\n(source)", 9396, "9{,}396 edges / 795 TFs\n+direction +topology"),
    ("$\\mathtt{build\\_tf\\_database}$\nhaematopoietic subset", 325, "325 edges / 11 TFs\ntargets only (3.5\\%)"),
    ("$\\mathtt{build\\_prior\\_matrix}$\nper-gene scalars", 203, "203 values\nno direction, no topology"),
    ("$\\mathtt{geneWeight}$\ntraining artefact", 0, "all-zero ($\\mathrm{nnz}=0$, 796/796 files)"),
]
N_FILES = 796
ENCODE_TFS, HEMA_TFS = 126, 12
MISSING_TFS = 6


def panel_a(ax):
    """损耗漏斗：逐阶段保留的内容（边缘数，对数轴）。"""
    labels = [s[0] for s in STAGES]
    vals = [max(s[1], 1) for s in STAGES]          # 0 画在轴底
    y = range(len(STAGES))
    colors = [fp.C_HEMA, fp.C_HEMA, fp.CAUTION, fp.FAIL]
    ax.barh(list(y), vals, height=0.55, color=colors, edgecolor=fp.INK, linewidth=0.75, zorder=3)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=6)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlim(0.7, 3e4)
    ax.set_xlabel("content retained (edges, log scale)")
    for i, (lab, v, note) in enumerate(STAGES):
        txt = "0" if v == 0 else "%,d".replace(",d", "d") % v
        txt = "0" if v == 0 else f"{v:,}"
        ax.text(max(v, 1) * 1.35, i, txt, va="center", ha="left", fontsize=7,
                color=fp.FAIL if v == 0 else fp.INK,
                fontweight="bold" if v == 0 else "normal")
    # 末级：醒目标出"零"
    ax.annotate("no reader reaches the loss", xy=(1.05, 3.0),
                xytext=(0.42, 0.05), textcoords="axes fraction",
                fontsize=6, color=fp.FAIL, ha="left",
                arrowprops=dict(arrowstyle="->", color=fp.FAIL, lw=1,
                                connectionstyle="arc3,rad=-0.3"))
    fp.panel_label(ax, "A")


def panel_b(ax):
    """调用图终止：赋值存在，读取者不存在。"""
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    boxes = [
        (0.2, 7.4, "prior database\n(TRRUST v2)", fp.C_HEMA),
        (0.2, 5.0, "per-gene weight vector\n(203 scalars)", fp.C_HEMA),
        (0.2, 2.6, "$\\mathtt{\\_tf\\_prior\\_weights}$\n(assigned, never read)", fp.C_HEMA),
    ]
    for x, y, txt, c in boxes:
        ax.add_patch(FancyBboxPatch((x, y), 4.6, 1.5, boxstyle="round,pad=0.12",
                                    fc="white", ec=c, lw=1))
        ax.text(x + 2.3, y + 0.75, txt, ha="center", va="center", fontsize=6)
    for y0 in (7.4, 5.0):
        ax.add_patch(FancyArrowPatch((2.5, y0), (2.5, y0 - 0.9), arrowstyle="->",
                                     color=fp.INK, lw=1.0, mutation_scale=9))
    # 断头：向右的箭头被红叉截断
    ax.add_patch(FancyArrowPatch((4.8, 3.35), (6.6, 3.35), arrowstyle="->",
                                 color=fp.FAIL, lw=1.25, ls=(0, (3, 2)), mutation_scale=9))
    ax.plot([5.6], [3.35], marker="x", ms=9, mew=1.8, color=fp.FAIL, zorder=5)
    ax.text(6.8, 3.35, "no code path reads it", fontsize=6, color=fp.FAIL,
            va="center", ha="left")
    ax.text(6.8, 2.35, "likelihood-weighting branch:\n0 instantiation sites",
            fontsize=6.0, color=fp.MUTED, va="top", ha="left")
    ax.text(0.2, 0.7, "call graph terminates at the assignment", fontsize=6,
            color=fp.INK, fontweight="bold", ha="left")
    fp.panel_label(ax, "B")


def panel_c(ax):
    """零权重层实测：全部 796 个 stage 文件 nnz = 0（对照非零）。"""
    xs = list(range(1, N_FILES + 1))
    ax.scatter(xs, [0] * N_FILES, s=2.2, color=fp.FAIL, zorder=3,
               edgecolors="none")
    ax.scatter([N_FILES + 40], [0], s=26, color=fp.C_HEMA, zorder=3, marker="s",
               edgecolors="white", linewidths=0.75)
    ax.set_ylim(-1.2, 1.2)
    ax.set_xlim(0, N_FILES + 80)
    ax.set_yticks([0]); ax.set_yticklabels(["0"])
    ax.set_xlabel("retained stage files (all runs, all arms)", fontsize=7)
    ax.set_ylabel("non-zero entries", fontsize=7)
    ax.annotate("796 / 796\n$\\mathrm{nnz}=0$", xy=(N_FILES / 2, 0),
                xytext=(N_FILES / 2, 0.62),
                fontsize=7, color=fp.FAIL, ha="center",
                arrowprops=dict(arrowstyle="->", color=fp.FAIL, lw=0.75))
    ax.text(N_FILES + 40, 0.42, "positive\ncontrol", fontsize=6.0, color=fp.C_HEMA,
            ha="center", va="bottom")
    fp.panel_label(ax, "C")


def panel_d(ax):
    """覆盖缺口：运行时参考库缺 6/12 造血因子。"""
    present = HEMA_TFS - MISSING_TFS
    ax.bar([0], [present], width=0.55, color=fp.C_LUNG, edgecolor=fp.INK,
           linewidth=0.75, label="present in ENCODE reference")
    ax.bar([0], [MISSING_TFS], width=0.55, bottom=[present], color=fp.FAIL,
           edgecolor=fp.INK, linewidth=0.75, label="absent (incl. KLF1)")
    ax.set_xticks([0]); ax.set_xticklabels(["haematopoietic TFs\ncopied into the runtime library"],
                                          fontsize=6)
    ax.set_ylabel("transcription factors", fontsize=7)
    ax.set_ylim(0, 16)
    ax.set_xlim(-1.15, 2.0)
    ax.text(0, present + MISSING_TFS + 0.45, "$6/12$ absent", ha="center",
            fontsize=7, color=fp.FAIL, fontweight="bold")
    ax.text(-1.10, 15.4, "ENCODE reference: 126 TFs, no direction column",
            fontsize=6.0, color=fp.MUTED, ha="left", va="top")
    ax.legend(frameon=False, fontsize=6, loc="center right",
              bbox_to_anchor=(1.02, 0.55))
    fp.panel_label(ax, "D")


def main():
    fp.apply_paper_style()
    fig, axes = plt.subplots(2, 2, figsize=(6.6929, 4.90))
    panel_a(axes[0][0]); panel_b(axes[0][1])
    panel_c(axes[1][0]); panel_d(axes[1][1])
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  [saved]", OUT)
    print("  [saved]", OUT_PNG)


if __name__ == "__main__":
    main()
