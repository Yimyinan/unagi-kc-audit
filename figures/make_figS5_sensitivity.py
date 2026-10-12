# -*- coding: utf-8 -*-
"""Supplementary Fig. S5: diagnostic sensitivity control.
Known k-dimensional latent collapse (from a real vanilla encoder) is detected as locked;
none of the 33 configurations is. Data: diagnostic_sensitivity_control.json + kappa_complete_audit.json."""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PKG = r"F:/UNAGI_Project/paper1/paper1_kc_diagnostic"
BASE = os.path.join(PKG, "figures")
NEX = os.path.dirname(os.path.dirname(PKG))
sys.path.insert(0, os.path.join(PKG, "code"))
import figure_palette as fp
fp.apply_paper_style()

C_HEMA, C_NEURAL, C_LUNG = fp.C_HEMA, fp.C_NEURAL, fp.C_LUNG
RED, GRAY = fp.RED, fp.GRAY

sens = json.load(open(os.path.join(PKG, "data", "diagnostic_sensitivity_control.json"), encoding="utf-8"))
aud = json.load(open(os.path.join(PKG, "data", "kappa_complete_audit.json"), encoding="utf-8"))

# Panel (b) is drawn in the CORRECTED predictability statistic: the released trace
# form is not invariant to the latent basis and its scatter swamps the bands.
S1G = os.path.join(NEX, "outputs", "positive_result", "s1g_corrected_summaries.json")
_corr = {r["config"]: r["kappa_r2"]
         for r in json.load(open(S1G, encoding="utf-8-sig"))["rows"]}
_missing = [r["config"] for r in aud if r["config"] not in _corr]
if _missing:
    raise SystemExit("missing corrected kappa for: %s" % _missing)

BAND_GVAE = (0.937, 1.064)   # corrected statistic
BAND_HEMO = (0.971, 1.029)

fig, (axa, axb) = plt.subplots(1, 2, figsize=(6.7, 2.6), gridspec_kw={"width_ratios": [1, 1.15]})
ks = [int(k) for k in sens["k"]]
kappas = [sens["k"][str(k)]["kappa"] for k in ks]
axa.plot(ks, kappas, "o-", color=RED, lw=1.25, ms=4, label="known collapse")
axa.axhline(0.5, color=RED, ls="--", lw=1)
axa.axhspan(0.937, 1.064, color=fp.BAND, alpha=0.8, zorder=0)
axa.axhline(1.0, color=GRAY, ls=":", lw=0.75)
axa.set_xlabel("effective latent dims $k$ (of 64)")
axa.set_ylabel(r"trace estimator $\kappa_k$")
axa.set_xlim(0, 33); axa.set_ylim(0, 1.4)
axa.set_xticks([1, 2, 3, 5, 8, 16, 32])
axa.text(32.4, 0.5, "locking threshold", ha="right", va="center", fontsize=6, color=RED)
axa.text(32.4, 1.0, "vanilla-equiv.", ha="right", va="center", fontsize=6, color="#4C72B0")
axa.set_title("Sensitivity: rank collapse is detected", fontweight="bold")
axa.legend(loc="lower right", frameon=False)

axb.axhspan(*BAND_GVAE, color="0.92", zorder=0)
axb.axhspan(*BAND_HEMO, color=fp.BAND, alpha=0.85, zorder=1)
axb.axhline(0.5, color=RED, ls="--", lw=1)
for r in aud:
    c = {"hemo": C_HEMA, "neural": C_NEURAL, "lung": C_LUNG}[r["tissue"]]
    axb.scatter(_corr[r["config"]], 0.02 * np.random.randn() + 0.5, s=26, color=c,
                edgecolors="white", linewidths=0.75, zorder=3, alpha=0.9)
axb.set_xscale("log")
axb.set_xticks([0.5, 0.7, 0.9, 1.0, 1.1, 1.3])
axb.set_xticklabels(["0.5", "0.7", "0.9", "1.0", "1.1", "1.3"])
axb.set_xlabel(r"corrected audit ratio $\kappa$ (log)")
axb.set_yticks([])
axb.set_xlim(0.45, 1.4); axb.set_ylim(-0.1, 1.1)
axb.set_title("33 configurations: 0 locked", fontweight="bold")
for lab, cc in [("hemato (22)", C_HEMA), ("neural (10)", C_NEURAL), ("lung (1)", C_LUNG)]:
    axb.scatter([], [], s=20, color=cc, label=lab)
axb.legend(loc="upper left", frameon=False, fontsize=6)
axa.text(-0.12, 1.06, "a", transform=axa.transAxes, fontsize=11, fontweight="bold", va="top")
axb.text(-0.12, 1.06, "b", transform=axb.transAxes, fontsize=11, fontweight="bold", va="top")

fig.savefig(os.path.join(BASE, "figS5_diagnostic_sensitivity.pdf"))
fig.savefig(os.path.join(BASE, "figS5_diagnostic_sensitivity.png"))
print("saved figS5_diagnostic_sensitivity")
