"""make_fig_latent_directionality.py — 分析 A 的图件

三个面板：
  (a) S1 的分布：观测（TRRUST 方向）vs 表达标签伪造 vs 检出率匹配零模型
      —— 说明"表达量能造出比真实方向标注更强的假信号"
  (b) 逐 TF 的平均 S1，按符号分类，标注符号一致性
      —— 说明符号异质性：17 个 TF 一致为负
  (c) 12 个造血 TF 的 S1 vs 检出率匹配后的 S1
      —— 说明真正"用到的"先验在表达量控制后基本消失
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.join(_HERE, "..", "code"), os.path.join(_HERE, "code")):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import figure_palette as fp
fp.apply_paper_style()

OUT = r"F:/UNAGI_Project/outputs/positive_result"
FIG = _HERE

J = json.load(open(os.path.join(OUT, "latent_directionality_final.json"), encoding="utf-8"))
per_tf = J["per_tf"]
gs = J["global_stats"]

fig, axes = plt.subplots(1, 3, figsize=(15.0, 3.5))

# ------------------------------------------------------------------ (a)
ax = axes[0]
obs_m = gs["obs"]["mean"]
dl_m = gs["c6a_detection_label"]["mean"]
dm_m = gs["c6b_detection_matched"]["mean"]
labels = ["TRRUST\ndirection\n(observed)", "Detection\nlabel\n(control)",
          "Detection\nmatched\n(null)"]
vals = [obs_m, dl_m, dm_m]
cols = [fp.RED, fp.GRAY, fp.GRAY]
bars = ax.bar(range(3), vals, color=cols, edgecolor="none", width=0.55)
ax.axhline(0, color=fp.VANILLA, lw=0.75)
for i, (b, v) in enumerate(zip(bars, vals)):
    off = 0.004 if v >= 0 else -0.008
    ax.text(i, v + off, f"{v:+.4f}", ha="center",
            va="bottom" if v >= 0 else "top", fontsize=6)
ax.set_xticks(range(3)); ax.set_xticklabels(labels, fontsize=6)
ax.set_ylim(-0.022, 0.215)
ax.set_ylabel(r"mean $S_1$  (alignment with TRRUST sign)", fontsize=7)
ax.set_title("(A) Real direction labels are weaker\nthan an expression-based fake",
             fontsize=7.5, loc="left")
ax.annotate("", xy=(2, dm_m + 0.006), xytext=(0, obs_m + 0.006),
            arrowprops=dict(arrowstyle="->", color=fp.C_HEMA, lw=1.0,
                            connectionstyle="arc3,rad=0.30"))
ax.text(-0.42, 0.212,
        r"observed $>$ matched null ($p=2.7\times10^{-5}$)" "\n"
        r"but $7\times$ weaker than the label fake",
        fontsize=6, ha="left", va="top", color=fp.VANILLA)
fp.style_ax(ax)

# ------------------------------------------------------------------ (b)
ax = axes[1]
srt = sorted(per_tf, key=lambda s: s["mean_s1"])
y = np.arange(len(srt))
vals_b = np.array([s["mean_s1"] for s in srt])
cons = np.array([bool(s["sign_consistent"]) for s in srt])
c = [fp.RED if (v > 0 and k) else (fp.C_HEMA if (v < 0 and k) else fp.GREY)
     for v, k in zip(vals_b, cons)]
ax.barh(y, vals_b, color=c, height=0.55, edgecolor="none")
ax.axvline(0, color=fp.VANILLA, lw=0.75)
ax.set_yticks([])
ax.set_xlabel(r"mean $S_1$ across 23 configurations", fontsize=7)
ax.set_ylabel("93 TFs (sorted)", fontsize=7)
n_pos = int(((vals_b > 0) & cons).sum())
n_neg = int(((vals_b < 0) & cons).sum())
ax.set_title(f"(B) Sign is reproducible but not uniform:\n"
             f"{n_pos} TFs consistently $+$, {n_neg} consistently $-$",
             fontsize=7.5, loc="left")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(facecolor=fp.RED, label=f"$S_1>0$, sign-locked (n={n_pos})"),
                   Patch(facecolor=fp.C_HEMA, label=f"$S_1<0$, sign-locked (n={n_neg})"),
                   Patch(facecolor=fp.GREY, label="sign not locked")],
          loc="lower right", fontsize=6, frameon=False, handlelength=1.0)
fp.style_ax(ax)

# ------------------------------------------------------------------ (c)
ax = axes[2]
HEM = ["GATA1", "GATA2", "LMO2", "RUNX1", "SPI1", "PAX5"]
d = {s["tf"]: s for s in per_tf}
x = np.arange(len(HEM))
w = 0.36
o = [d[t]["mean_s1"] for t in HEM]
mm = [d[t]["mean_s1_detmatched"] for t in HEM]
ax.bar(x - w / 2, o, w, color=fp.RED, edgecolor="none", label="observed $S_1$")
ax.bar(x + w / 2, mm, w, color=fp.GRAY, edgecolor="none",
       label="after detection matching")
ax.axhline(0, color=fp.VANILLA, lw=0.75)
ax.set_xticks(x); ax.set_xticklabels(HEM, fontsize=6, rotation=30, ha="right")
ax.set_ylabel(r"$S_1$", fontsize=7)
ax.set_ylim(-0.20, 0.34)
ax.set_title("(C) For the haematopoietic TFs actually used,\nthe effect does not survive",
             fontsize=7.5, loc="left")
ax.legend(loc="upper right", fontsize=6.0, frameon=False, handlelength=1.0)
ax.text(0.02, 0.02,
        "only 6 of 12 analysable; the rest have\n"
        r"$<$3 targets per direction",
        transform=ax.transAxes, fontsize=6, va="bottom", color=fp.VANILLA)
fp.style_ax(ax)

fig.tight_layout()
os.makedirs(FIG, exist_ok=True)
fp.save_paper(fig, os.path.join(FIG, "fig_latent_directionality.pdf"),
              os.path.join(FIG, "fig_latent_directionality.png"))
print("saved fig_latent_directionality.pdf/.png")
