# -*- coding: utf-8 -*-
"""Regenerate all real-data supplementary figures for paper1_kc_diagnostic.

figS1  ridge validation: (a) cos(theta) vs Ridge alpha per seed (note-literal z-space metric);
                         (b) C_ridge vs C_jacobian across 6 G-VAE configs + Spearman rho.
figS2  diagnostic robustness: (a) kappa(alpha) for 22 hema configs; (b) scaler effect (std vs raw).
figS4  GRQ direction v* per seed (4 panels, real v_grq).
supp_figS8  CTCP real vs label-shuffled (48 paired cluster observations).
supp_figS9  specificity-manifold Pareto: linear GRQ / geodesic / DDP.
Style matches generate_main_figures.py.
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel, spearmanr

BASE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(BASE)
NEX = os.path.abspath(os.path.join(PKG, "..", ".."))
RES = os.path.join(PKG, "data")
if not os.path.isdir(RES):
    RES = os.path.join(NEX, "results")
SUPP = os.path.join(BASE, "..", "supplementary")
sys.path.insert(0, os.path.join(PKG, "code"))
import figure_palette as fp
fp.apply_paper_style()

C1, C2, C3, C4 = fp.C1, fp.C2, fp.C3, fp.C4
GRAY = fp.GRAY

def save(fig, name, folder=BASE):
    p = os.path.join(folder, name)
    fig.savefig(p + ".pdf"); fig.savefig(p + ".png"); plt.close(fig)
    print("saved", p)

def panel_label(ax, lab):
    ax.text(-0.10, 1.06, lab, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")

# ============================================================ data
v2 = json.load(open(os.path.join(RES, "supp_s1s2_recompute_v2.json"), encoding="utf-8"))
s1a = v2["s1a"]; s1b = v2["s1b"]; s2 = v2["s2"]
alphas = v2["alphas"]
cp = json.load(open(os.path.join(RES, "ctcp_pure_validation.json"), encoding="utf-8"))
geo = json.load(open(os.path.join(RES, "geodesic_perturb_results.json"), encoding="utf-8"))
oa = json.load(open(os.path.join(RES, "optionA_grq_comparison.json"), encoding="utf-8"))["models"]

# ============================================================ figS1
fig = plt.figure(figsize=(6.7, 2.6))
axa = fig.add_axes([0.09, 0.20, 0.44, 0.72])
seeds = ["s9", "s10", "s11", "s12"]
cols = [C1, C2, C3, C4]
for s, c in zip(seeds, cols):
    ys = [s1a[s][str(a)] for a in alphas]
    axa.plot(alphas, ys, "o-", color=c, lw=1.0, ms=3.5, label=s)
vals1 = [s1a[s]["1.0"] for s in seeds]
mean1, sd1 = float(np.mean(vals1)), float(np.std(vals1, ddof=1))
axa.set_ylim(0, 0.9)
axa.axhline(0.78, color=GRAY, ls=":", lw=0.75)
axa.text(0.985, 0.875, "reported 0.78", transform=axa.transAxes, ha="right", va="bottom",
         fontsize=6, color=GRAY)
axa.text(0.02, 0.05, f"$\\alpha$=1: {mean1:.2f}$\\pm${sd1:.2f} ({min(vals1):.2f}--{max(vals1):.2f})",
         transform=axa.transAxes, ha="left", fontsize=6, color="#222222")
axa.set_xscale("log"); axa.set_xticks([0.01, 0.1, 1, 10, 100])
axa.set_xticklabels(["0.01", "0.1", "1", "10", "100"])
axa.set_xlabel("Ridge $\\alpha$"); axa.set_ylabel("$\\cos\\theta$ (top mode)")
axa.set_title("Ridge readout vs analytical Jacobian", fontsize=8)
axa.legend(loc="lower right", frameon=False, fontsize=6, ncol=2)
panel_label(axa, "a")

axb = fig.add_axes([0.62, 0.20, 0.33, 0.72])
crs = np.array([c["C_ridge"] for c in s1b["configs"]])
cjs = np.array([c["C_jacobian"] for c in s1b["configs"]])
rho = float(s1b["spearman_rho"])
axb.scatter(crs, cjs, s=22, color=C1, edgecolor="#222222", linewidths=0.75, zorder=3)
for c in s1b["configs"]:
    lab = c["config"].replace("gvae_", "").replace("_", "-")
    axb.annotate(lab, (c["C_ridge"], c["C_jacobian"]), textcoords="offset points",
                 xytext=(4, 3), fontsize=6, color="#333333")
axb.axhline(0, color=GRAY, ls=":", lw=0.75); axb.axvline(0, color=GRAY, ls=":", lw=0.75)
axb.set_xlabel("$C$ (Ridge readout, %)"); axb.set_ylabel("$C$ (Jacobian, %)")
axb.set_title("Rank agreement across 6 configs", fontsize=8)
axb.text(0.04, 0.95, f"Spearman $\\rho$ = {rho:.2f} (n=6)", transform=axb.transAxes,
         fontsize=6, color="#8B0000", va="top")
axb.text(-0.18, 1.06, "b", transform=axb.transAxes, fontsize=11, fontweight="bold", va="top")
save(fig, "figS1_ridge_validation")

# ============================================================ figS2
fig = plt.figure(figsize=(6.7, 2.6))
axa = fig.add_axes([0.09, 0.20, 0.44, 0.72])
devs = {c["config"]: c["max_abs_pct_dev_from_alpha1"] for c in s2["configs"]}
worst = max(devs, key=devs.get)
for c in s2["configs"]:
    ks = [c["kappa_alpha"][str(a)]["kappa"] for a in alphas]
    if c["config"] == worst:
        axa.plot(alphas, ks, "-", color=C3, lw=1.25, label=f"{worst} (max dev {devs[worst]:.0f}%)")
    else:
        axa.plot(alphas, ks, "-", color="#cccccc", lw=0.75)
axa.axhspan(0.937, 1.064, color="#55A868", alpha=0.12)
axa.text(0.02, 1.245, "vanilla-equivalent", fontsize=6, color="#2d6b35", va="bottom")
axa.set_xscale("log"); axa.set_xticks([0.01, 0.1, 1, 10, 100])
axa.set_xticklabels(["0.01", "0.1", "1", "10", "100"])
axa.set_xlabel("Ridge $\\alpha$"); axa.set_ylabel("$\\kappa$")
axa.set_title("$\\kappa$ vs Ridge $\\alpha$ (22 hema configs)", fontsize=8)
axa.legend(loc="upper right", frameon=False, fontsize=6)
axa.text(0.04, 0.02, "max deviation from $\\alpha$=1: 35.6% (R111_s123); median 5.7%",
         transform=axa.transAxes, fontsize=6, color="#333333")
panel_label(axa, "a")

axb = fig.add_axes([0.62, 0.20, 0.33, 0.72])
ks = np.array([c["kappa_at_alpha1"] for c in s2["configs"]])
kr = np.array([c["kappa_no_scaler"] for c in s2["configs"]])
axb.scatter(ks, kr, s=22, color=C2, edgecolor="#222222", linewidths=0.75, zorder=3)
lim = (0.5, 6.2)
axb.plot([0.5, 6.2], [0.5, 6.2], "--", color=GRAY, lw=0.75)
axb.fill_between([0.937, 1.064], [0.937, 1.064], [6.2, 6.2], color="#55A868", alpha=0.10)
axb.set_xlim(lim); axb.set_ylim(lim)
axb.set_xlabel("$\\kappa$ (StandardScaler)"); axb.set_ylabel("$\\kappa$ (no scaler)")
axb.set_title("StandardScaler effect (22 configs)", fontsize=8)
mx = float(np.max(kr / ks))
axb.text(0.04, 0.97, f"max $\\kappa$ ratio: {mx:.2f}$\\times$ (no inflation)",
         transform=axb.transAxes, fontsize=6, color="#333333", va="top")
panel_label(axb, "b")
save(fig, "figS2_diagnostic_robustness")

# ============================================================ figS4 (GRQ direction v*)
fig = plt.figure(figsize=(6.7, 4.6))
for idx, s in enumerate(seeds):
    v = np.array(oa[f"gvae_scan_{s}"]["decoder_W"]["v_grq"])
    spec = oa[f"gvae_scan_{s}"]["decoder_W"]["grq_spec_std"]
    ax = fig.add_axes([0.09 + (idx % 2) * 0.46, 0.06 + (1 - idx // 2) * 0.50, 0.42, 0.36])
    x = np.arange(1, 65)
    ax.bar(x, np.abs(v), color="#cccccc", edgecolor="none", width=0.55)
    top5 = np.argsort(-np.abs(v))[:5] + 1
    ax.bar(top5, np.abs(v)[top5], color=C3, edgecolor="none", width=0.55)
    topstr = ", ".join([f"d{d}" for d in top5])
    ax.text(0.5, 0.965, f"top dims: {topstr}", transform=ax.transAxes, ha="center", va="top",
            fontsize=6, color="#8B0000")
    ax.set_xticks([1, 17, 33, 49, 64]); ax.set_xticklabels(["1", "17", "33", "49", "64"])
    ax.set_ylim(0, 0.5)
    ax.set_xlabel("Latent dimension"); ax.set_ylabel("$|v^*|$")
    ax.set_title(f"{s}: GRQ direction (readout spec {spec:.1f}$\\times$)", fontsize=7.5)
    ax.text(-0.12, 1.05, chr(97 + idx), transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")
save(fig, "figS4_grq_direction")

# ============================================================ supp_figS8
real, shuf, glob = [], [], []
for sd in cp.values():
    for c in sd["clusters"].values():
        real.append(c["spec_ctcp"]); shuf.append(c["spec_shuffled"]); glob.append(c["spec_global"])
real = np.array(real); shuf = np.array(shuf); glob = np.array(glob)
t, p = ttest_rel(real, shuf)

fig = plt.figure(figsize=(4.2, 2.9))
ax = fig.add_axes([0.13, 0.15, 0.83, 0.78])
vp = ax.violinplot([glob, real, shuf], showmeans=True, showextrema=False, widths=0.7)
for i, (vd, color) in enumerate(zip(vp["bodies"], [GRAY, C1, C2])):
    vd.set_facecolor(color); vd.set_alpha(0.55); vd.set_edgecolor("#333333"); vd.set_linewidth(0.4)
for i, (data, color) in enumerate(zip([glob, real, shuf], [GRAY, C1, C2])):
    xj = np.random.default_rng(100 + i).uniform(-0.10, 0.10, len(data))
    ax.scatter(i + 1 + xj, data, s=9, color=color, alpha=0.6, edgecolor="none", zorder=3)
ax.set_xticks([1, 2, 3]); ax.set_xticklabels(["Global", "Real CTCP", "Label-shuffled"])
ax.set_ylabel("Perturbation specificity ($\\times$)")
ax.text(0.03, 0.97, f"real {real.mean():.1f}$\\times$ vs shuffled {shuf.mean():.1f}$\\times$\npaired $p$={p:.2f} (n={len(real)})",
        transform=ax.transAxes, fontsize=7, va="top", color="#8B0000")
ax.set_title("CTCP gains are indistinguishable from label shuffling", fontsize=8)
save(fig, "supp_figS8_ctcp_polarity", folder=SUPP)

# ============================================================ supp_figS9
fig = plt.figure(figsize=(4.2, 2.9))
ax = fig.add_axes([0.13, 0.15, 0.83, 0.78])
lin = geo["linear"]; gd = geo["geodesic"]; ddp = geo["ddp"]
ax.plot([x["dz_norm"] for x in lin], [x["specificity"] for x in lin], "o-", color=C1,
        lw=1.0, ms=3.5, label="GRQ (linear)")
ax.plot([x["dz_norm"] for x in gd], [x["specificity"] for x in gd], "s-", color=C2,
        lw=1.0, ms=3.5, label="GRQ (geodesic)")
ddp_pos = [x for x in ddp if x["specificity"] > 0]
ddp_neg = [x for x in ddp if x["specificity"] <= 0]
ax.scatter([x["dz_norm"] for x in ddp_pos], [x["specificity"] for x in ddp_pos],
           color=C3, marker="^", s=34, edgecolors="#222222", linewidths=0.75, zorder=5, label="DDP (off-manifold)")
ax.scatter([x["dz_norm"] for x in ddp_neg], [x["specificity"] for x in ddp_neg],
           color="#AA0000", marker="v", s=34, edgecolors="#222222", linewidths=0.75, zorder=5)
ax.axhline(0, color=GRAY, ls=":", lw=0.75)
# (mean inter-cell distance line removed: no reproducible source for the value 5.0)
ax.set_xlabel("$\\|\\Delta z\\|$ (manifold distance)"); ax.set_ylabel("Erythroid specificity ($\\times$)")
ax.set_title("Specificity–manifold Pareto frontier", fontsize=8)
ax.legend(loc="upper left", frameon=False, fontsize=6)
save(fig, "supp_figS9_pareto", folder=SUPP)

print("done")
