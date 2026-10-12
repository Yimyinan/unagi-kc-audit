# -*- coding: utf-8 -*-
"""
make_fig4_prior_activity.py
===========================
Generate fig4_prior_activity.pdf/.png.

Panels
------
(a) lambda-scan: C flat          (IPF, inert prior)
(b) lambda-scan: kappa_gap       (IPF, subcritical)
(c) structural ceiling           (IPF, infeasible target)
(d) CellOracle-style priors      (cross-architecture, subcritical)

Panel (d) is the cross-architecture external validation: eight single-TF
priors built by one-shot Ridge regression, all subcritical. The reference
band is drawn as the critical threshold kappa = 0.5; the safety margin
(the factor by which the closest prior misses it) is annotated because the
original audit's vanilla reference could not be reproduced exactly and the
claim is therefore reported as bounded rather than as a point estimate.
"""
import json
import os
import sys as _sys
import os as _os

import matplotlib
import numpy as np

_HERE = _os.path.dirname(_os.path.abspath(__file__))
for _p in (_HERE, _os.path.join(_HERE, "..", "code"), _os.path.join(_HERE, "code")):
    if _os.path.isdir(_p) and _p not in _sys.path:
        _sys.path.insert(0, _p)

import figure_palette as fp
fp.apply_paper_style()
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = _HERE
OUT_FIG = os.path.join(HERE, "fig4_prior_activity.pdf")
OUT_PNG = os.path.join(HERE, "fig4_prior_activity.png")

SRC_SCAN = (r"F:\UNAGI_Project\next_phase\results\ptvae_human_ipf_ln_v2"
            r"\optionB_lambda_scan.json")
SRC_CO = (r"F:\UNAGI_Project\outputs"
          r"\positive_result\celloracle_sensitivity.json")

# fall-back to the audited source tree if the results path moved
if not os.path.exists(SRC_CO):
    SRC_CO = (r"F:\UNAGI_Project\UNAGI\celloracle_audit\output"
              r"\kappa_alpha_batch.json")

with open(SRC_SCAN, "r", encoding="utf-8") as f:
    scan = json.load(f)

lams = [0.1, 1.0, 10.0]
bn_C = [scan[f"lam{v}"]["bn_prior_kc_s42"]["C"] for v in ["0.1", "1", "10"]]
ln_C = [scan[f"lam{v}"]["ln_prior_kc_s42"]["C"] for v in ["0.1", "1", "10"]]
bn_k = [scan[f"lam{v}"]["bn_prior_kc_s42"]["kappa"] for v in ["0.1", "1", "10"]]
ln_k = [scan[f"lam{v}"]["ln_prior_kc_s42"]["kappa"] for v in ["0.1", "1", "10"]]

# Structural bound: kappa_gap <= (1-C)*trH_V / gap_V
trH_V, gap_V = 10590.0, 9663.0
Cgrid = np.linspace(0, 100, 201)
ub = (1 - Cgrid / 100.0) * trH_V / gap_V

# ---- panel (d) data -------------------------------------------------------
raw = json.load(open(SRC_CO, encoding="utf-8-sig"))
if "rows" in raw:                      # sensitivity output
    co_rows = sorted(raw["rows"], key=lambda r: r["kappa_original"])
    co_kappa = [r["kappa_original"] for r in co_rows]
    co_names = [r["tf"] for r in co_rows]
    co_safety = raw["safety_factor"]
else:                                   # raw audit fallback
    co_rows = sorted(raw["tf_results"], key=lambda r: r["kappa"])
    co_kappa = [r["kappa"] for r in co_rows]
    co_names = [r["tf"] for r in co_rows]
    co_safety = None

KAPPA_CRIT = 0.5

# ★ 画布 = 最终排版宽；1x4 在 6.69 in 下每格仅 1.67 in 太窄，故改 2x2。
fig, axes = plt.subplots(2, 2, figsize=(6.6929, 4.60))
axes = axes.flat   # 兼容下面 axes[0..3] 的写法

# (a) lambda-scan C
ax = axes[0]
ax.plot(lams, bn_C, "o-", color=fp.RED, label="BN decoder")
ax.plot(lams, ln_C, "s-", color=fp.C_LUNG, label="LN decoder")
ax.set_xscale("log"); ax.set_xticks(lams); ax.set_xticklabels(["0.1", "1", "10"])
ax.set_xlabel(r"Prior strength $\lambda$")
ax.set_ylabel("Global $C$ (%)")
ax.set_ylim(94.5, 97.5)
ax.set_title("(A) IPF: $C$ flat")
ax.legend(frameon=False, fontsize=8)
ax.text(0.04, 0.06, "var < 0.5 pp", transform=ax.transAxes, fontsize=8,
        color=fp.VANILLA)

# (b) lambda-scan kappa_gap
ax = axes[1]
ax.plot(lams, bn_k, "o-", color=fp.RED, label="BN decoder")
ax.plot(lams, ln_k, "s-", color=fp.C_LUNG, label="LN decoder")
ax.axhline(2.0, ls="--", color=fp.GRAY, lw=1.0)
ax.text(0.06, 0.9, r"target $\kappa_{\mathrm{gap}}=2$", transform=ax.transAxes,
        fontsize=8, color=fp.GRAY)
ax.set_xscale("log"); ax.set_xticks(lams); ax.set_xticklabels(["0.1", "1", "10"])
ax.set_xlabel(r"Prior strength $\lambda$")
ax.set_ylabel(r"$\kappa_{\mathrm{gap}}$")
ax.set_ylim(0, 0.05)
ax.set_title("(B) IPF: subcritical")
# (c) structural bound
ax = axes[2]
ax.plot(Cgrid, ub, "-", color=fp.C_HEMA, lw=2,
        label=r"$\kappa_{\mathrm{gap}}\leq(1{-}C)\,\mathrm{tr}H_V/\mathrm{gap}_V$")
ax.axhline(2.0, ls="--", color=fp.GRAY, lw=1.0)
ax.annotate("infeasible\ntarget (2, 90)", xy=(90, 2.0), xytext=(55, 1.55),
            arrowprops=dict(arrowstyle="->", lw=0.75, color=fp.VANILLA),
            fontsize=8, color=fp.VANILLA)
ax.plot([90], [0.11], "o", color=fp.RED, ms=5)
ax.annotate(r"$\kappa_{\mathrm{gap}}\leq 0.11$ at $C=90\%$", xy=(90, 0.11),
            xytext=(38, 0.6), arrowprops=dict(arrowstyle="->", lw=0.75,
                                              color=fp.VANILLA), fontsize=8)
ax.set_xlabel("Global $C$ (%)")
ax.set_ylabel(r"$\kappa_{\mathrm{gap}}$ upper bound")
ax.set_ylim(0, 2.2); ax.set_xlim(0, 100)
ax.set_title("(C) IPF: structural ceiling")
ax.text(0.03, 0.06, "vanilla top axis = 91.2% of trH", transform=ax.transAxes,
        fontsize=8, color=fp.VANILLA)
ax.legend(frameon=False, fontsize=7.5, loc="upper right")

# (d) cross-architecture: CellOracle-style single-TF priors
ax = axes[3]
ypos = np.arange(len(co_names))
ax.barh(ypos, co_kappa, color=fp.C_NEURAL, height=0.55)
ax.axvline(KAPPA_CRIT, ls="--", color=fp.RED, lw=1.25)
ax.set_yticks(ypos)
ax.set_yticklabels(co_names, fontsize=8)
ax.set_xlabel(r"$\kappa$")
ax.set_xlim(0, 0.60)
ax.set_title(r"(D) CellOracle-style priors")
ax.text(KAPPA_CRIT + 0.015, len(co_names) - 0.6,
        r"critical $\kappa=0.5$" + "\n" + "8/8 subcritical",
        fontsize=8, color=fp.RED, va="top")
if co_safety:
    ax.text(0.97, 0.06, f"safety factor {co_safety:.0f}x",
            transform=ax.transAxes, fontsize=8, color=fp.VANILLA, ha="right")

fig.tight_layout()
fig.savefig(OUT_FIG)
fig.savefig(OUT_PNG, dpi=200)
print("saved", OUT_FIG)
print("saved", OUT_PNG)
