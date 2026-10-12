# -*- coding: utf-8 -*-
"""Regenerate supplementary figure S3 (encoder per-layer divergence) from audited data.

Style matches generate_main_figures.py (same fonts, unicode_minus fix, print-safe sizes).
Old figS3_encoder_layers.pdf used the stale 3-layer naming (Layer0/Layer1) and an unbacked
"decoder layers r>0.90" claim; this version uses only audited values:
  - results/encoder_layer_correlations_real.json (encoder per-layer off-diagonal r, 4 seeds)
  - manuscripts/figures_v1/fig1_coadaptation_data.json (decoder per-gene sensitivity r)
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(BASE)
NEX = os.environ.get("UNAGI_NEX_DIR", r"F:/UNAGI_Project/next_phase")
sys.path.insert(0, os.path.join(PKG, "code"))
import figure_palette as fp
fp.apply_paper_style()

C_ENC, C_DEC = fp.C_HEMA, fp.C_LUNG

enc = json.load(open(os.path.join(NEX, "results", "encoder_layer_correlations_real.json"), encoding="utf-8"))
d1 = json.load(open(os.path.join(NEX, "manuscripts", "figures_v1", "fig1_coadaptation_data.json"), encoding="utf-8"))

layers = [("Layer 1 (G→512)", enc["Layer1_G512"]),
          ("Layer 2 (512→512)", enc["Layer2_512"]),
          ("mu head", enc["mu_head"])]
dec_mean = d1["dec_mean_offdiag"]
dec_off = np.array(d1["dec_corr"])[np.triu_indices(4, 1)]
dec_lo, dec_hi = float(dec_off.min()), float(dec_off.max())

fig = plt.figure(figsize=(4.6, 2.35))

# ---- (a) encoder per-layer ----
axa = fig.add_axes([0.11, 0.17, 0.42, 0.76])
names = [n for n, _ in layers]
means = [float(v["mean"]) for _, v in layers]
sds = [float(v["sd"]) for _, v in layers]
x = np.arange(len(layers))
bars = axa.bar(x, means, width=0.55, color=C_ENC, edgecolor="#333333", linewidth=0.75)
axa.errorbar(x, means, yerr=sds, fmt="none", ecolor="black", elinewidth=0.75, capsize=2.5)
for xi, m, sd in zip(x, means, sds):
    axa.text(xi, m + sd + 0.002, f"{m:+.3f}", ha="center", fontsize=6, color="#222222")
axa.axhline(0.02, color="#8B0000", ls="--", lw=0.75)
axa.text(0.99, 0.021, "all $r<0.02$", ha="right", fontsize=6, color="#8B0000")
axa.set_xticks(x); axa.set_xticklabels([n.split(" (")[0] for n in names])
axa.set_ylim(-0.01, 0.035)
axa.set_ylabel("Pearson $r$ (off-diagonal)")
axa.set_title("Encoder weights: divergent across seeds", fontsize=8)
axa.text(0.5, -0.38, "2-layer MLP, hidden 512 (4 seeds, recomputed from checkpoints)",
         transform=axa.transAxes, ha="center", fontsize=6, color="#555555")
axa.text(-0.10, 1.06, "a", transform=axa.transAxes, fontsize=11, fontweight="bold", va="top")

# ---- (b) decoder sensitivity ----
axb = fig.add_axes([0.62, 0.17, 0.34, 0.76])
axb.bar([0], [dec_mean], width=0.55, color=C_DEC, edgecolor="#333333", linewidth=0.75)
axb.errorbar([0], [dec_mean], yerr=[[dec_mean - dec_lo], [dec_hi - dec_mean]], fmt="none",
             ecolor="black", elinewidth=0.75, capsize=2.5)
axb.text(0, dec_mean + (dec_hi - dec_mean) + 0.004, f"$r={dec_mean:.3f}$\n[{dec_lo:.2f}, {dec_hi:.2f}]",
         ha="center", fontsize=6, color="#222222")
axb.axhline(0.9, color="#888888", ls=":", lw=0.75)
axb.set_xticks([0]); axb.set_xticklabels(["per-gene\nsensitivity"])
axb.set_ylim(0.88, 1.03)
axb.set_ylabel("Pearson $r$")
axb.set_title("Decoder: convergent across seeds", fontsize=8)
axb.text(-0.22, 1.06, "b", transform=axb.transAxes, fontsize=11, fontweight="bold", va="top")

fig.savefig(os.path.join(BASE, "figS3_encoder_layers.pdf"))
fig.savefig(os.path.join(BASE, "figS3_encoder_layers.png"))
plt.close(fig)
print("saved figS3_encoder_layers (from audited data)")
