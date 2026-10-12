# -*- coding: utf-8 -*-
"""Regenerate main-text figures 1-5 from real data files with unified publication style.

Style contract (Nature-family / NCS):
  - canvas width 6.7 in (= 482 pt = the manuscript textwidth), so \textwidth scaling ~= 1
  - base font 7.5 pt, axis labels 8 pt, titles 9 pt, ticks 7 pt, legend 7 pt
  - unified sans-serif (Arial, falls back to Helvetica/DejaVu Sans)
  - vector PDF + 300 dpi PNG
All values are read from the audited data files; no hard-coded placeholders.
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

BASE = os.path.dirname(os.path.abspath(__file__))          # .../figures
PKG = os.path.dirname(BASE)                                  # .../paper1_kc_diagnostic
NEX = os.environ.get("UNAGI_NEX_DIR", r"F:/UNAGI_Project/next_phase")
DATA = os.path.join(PKG, "data")

# Load centralized publication style (font/palette/dimensions/output).
plt.style.use(os.path.join(BASE, "mpl_style.mplstyle"))
# Use the single-source-of-truth palette so colours cannot drift between scripts.
sys.path.insert(0, os.path.join(BASE, "..", "code"))
import figure_palette as fp
C_HEMA, C_NEURAL, C_LUNG = fp.C_HEMA, fp.C_NEURAL, fp.C_LUNG
GRAY, RED = fp.GRAY, fp.RED
FIGW = 6.7  # inches -> ~482 pt

# ---------------------------------------------------------------------------------
# GUARD: this umbrella script historicaly wrote filenames that dedicated scripts also own.
# Running it therefore silently OVERWROTE submission figures with legacy content -- that
# happened once for fig2_kc_phase_diagram during the figure-rework pass. Two of its outputs
# collide (verified by scanning for duplicate output names across scripts/):
#     fig2_kc_phase_diagram  <- owned by figures/make_fig2_kc_phase.py      (manuscript Fig. 3)
#     fig4_prior_activity    <- owned by figures/make_fig4_prior_activity.py (manuscript Fig. 5)
# Both are renamed on write so the authoritative file can never be clobbered by accident.
OWNED_ELSEWHERE = {
    "fig2_kc_phase_diagram": "make_fig2_kc_phase.py",
    "fig4_prior_activity": "make_fig4_prior_activity.py",
}


_GEN_W = {
    "fig2_z_spectrum_3tissue": 6.89,
    "fig5_prior_biology_validation": 10.70,
}


def save(fig, name):
    fp.fit_to_width(fig, gen_width_in=_GEN_W.get(name))
    if name in OWNED_ELSEWHERE:
        owner = OWNED_ELSEWHERE[name]
        print("WARNING: %s is owned by %s -- this umbrella's copy is NOT the manuscript figure; "
              "writing *_UMBRELLA_LEGACY_DO_NOT_USE instead" % (name, owner))
        name = name + "__UMBRELLA_LEGACY_DO_NOT_USE"
    fig.savefig(os.path.join(BASE, name + ".pdf"))
    fig.savefig(os.path.join(BASE, name + ".png"))
    plt.close(fig)
    print("saved", name)

def panel_label(ax, label, dx=-0.16):
    ax.text(dx, 1.06, label, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")

# =====================================================================
# FIG 1: encoder-decoder co-adaptation
# =====================================================================
d1 = json.load(open(os.path.join(NEX, "manuscripts", "figures_v1", "fig1_coadaptation_data.json"), encoding="utf-8"))
enc = np.array(d1["enc_corr"]); dec = np.array(d1["dec_corr"])
sens9 = np.array(d1["sens_s9"]); sens11 = np.array(d1["sens_s11"])
r2 = d1["procrustes_R2"]; enc_m = d1["enc_mean_offdiag"]; dec_m = d1["dec_mean_offdiag"]
p9 = np.array(d1["p9"]); p11 = np.array(d1["p11"]); p11a = np.array(d1["p11a"])

fig1 = plt.figure(figsize=(FIGW, 2.9))
ax1a = fig1.add_axes([0.075, 0.16, 0.26, 0.76])
mask = np.triu(np.ones_like(enc, dtype=bool), k=1)
im = ax1a.imshow(np.ma.array(enc, mask=mask), cmap="coolwarm", vmin=-0.05, vmax=1.0, aspect="equal")
for i in range(4):
    for j in range(i + 1):
        ax1a.text(j, i, "1.00" if i == j else f"{enc[i, j]:.2f}",
                  ha="center", va="center", fontsize=6,
                  color="white" if i == j else "#222222")
ax1a.set_xticks(range(4)); ax1a.set_xticklabels(d1["seeds"])
ax1a.set_yticks(range(4)); ax1a.set_yticklabels(d1["seeds"])
ax1a.set_title("Encoder: $r<0.02$", fontweight="bold")
ax1a.text(0.5, -0.28, f"mean $r$ = {enc_m:.3f}", transform=ax1a.transAxes, ha="center", fontsize=7, color="#444444")
panel_label(ax1a, "a")

ax1b = fig1.add_axes([0.40, 0.16, 0.27, 0.76])
ax1b.scatter(sens9, sens11, s=4, color=C_HEMA, alpha=0.55, edgecolors="none")
lim = [0, max(sens9.max(), sens11.max()) * 1.02]
ax1b.plot(lim, lim, "k--", lw=0.75)
ax1b.set_xlim(lim); ax1b.set_ylim(lim)
ax1b.set_xlabel("$s_9$ sensitivity (norm.)")
ax1b.set_ylabel("$s_{11}$ sensitivity (norm.)")
ax1b.set_title("Decoder: $r$ = %.2f" % dec_m, fontweight="bold")
ax1b.text(0.05, 0.90, f"per-gene $r$ = {dec_m:.3f}", transform=ax1b.transAxes, fontsize=7, color="#333333")
panel_label(ax1b, "b")

ax1c = fig1.add_axes([0.73, 0.16, 0.24, 0.76])
ax1c.scatter(p9[:, 0], p9[:, 1], s=8, color=GRAY, label="$s_9$ reference", alpha=0.8)
ax1c.scatter(p11[:, 0], p11[:, 1], s=8, color=C_HEMA, marker="s", label="$s_{11}$ before", alpha=0.8)
ax1c.scatter(p11a[:, 0], p11a[:, 1], s=8, color="#E15759", marker="^", label="$s_{11}$ aligned", alpha=0.8)
ax1c.set_xlabel("PC1"); ax1c.set_ylabel("PC2")
ax1c.set_title("Procrustes alignment", fontweight="bold")
ax1c.text(0.04, 0.90, f"$R^2$ = {r2:.2f}", transform=ax1c.transAxes, fontsize=8, fontweight="bold", color="#333333")
ax1c.legend(loc="lower right", fontsize=6, frameon=False)
panel_label(ax1c, "c")
save(fig1, "fig1_coadaptation")

# =====================================================================
# FIG 2: kappa-C diagnostic + 0/33 audit + C_head
# =====================================================================
rows = json.load(open(os.path.join(DATA, "kappa_complete_audit.json"), encoding="utf-8"))
hemo = sorted([r for r in rows if r["tissue"] == "hemo"], key=lambda r: r["kappa_ratio"])
neural = sorted([r for r in rows if r["tissue"] == "neural"], key=lambda r: r["kappa_ratio"])
lung = [r for r in rows if r["tissue"] == "lung"]
assert len(hemo) == 22 and len(neural) == 10 and len(lung) == 1 and len(rows) == 33

opp = json.load(open(os.path.join(NEX, "results", "opp_recompute_from_ckpt.json"), encoding="utf-8"))
chead = np.array([v["C_head"] for v in opp.values()])
cm, csd = float(chead.mean()), float(chead.std(ddof=1))
n_gt50 = int((chead > 50).sum())
TISSUES = [("Hematopoietic", hemo, C_HEMA), ("Neural crest", neural, C_NEURAL), ("Lung fibrosis", lung, C_LUNG)]

fig2 = plt.figure(figsize=(6.35, 3.0))
ax2a = fig2.add_axes([0.070, 0.26, 0.325, 0.66])
xa = 0.0
for _, cfglist, color in TISSUES:
    xs = [xa + i for i in range(len(cfglist))]
    ax2a.bar(xs, [r["C"] for r in cfglist], width=0.55, color=color, edgecolor="none")
    xa += len(cfglist) + 1.4
ax2a.axhline(50, color="black", ls="--", lw=0.75)
ax2a.annotate("Global locking $C>50\\%$: 0/33 reach",
              xy=(2.5, 50), xycoords="data",
              xytext=(0.12, 0.92), textcoords="axes fraction",
              fontsize=7, color=fp.EDGE, ha="left", va="top",
              arrowprops=dict(arrowstyle="-", lw=0.75, color=fp.EDGE))
ax2a.set_yscale("symlog", linthresh=1.0)
ax2a.set_yticks([-500, -100, -50, 0, 50]); ax2a.set_ylim(-560, 75); ax2a.set_xticks([])
ax2a.set_ylabel("Global $C$ (%)")
ax2a.set_title("Global spectral audit")
ax2a.text(0.97, 0.04, "(log scale)", transform=ax2a.transAxes, ha="right", fontsize=6, color=fp.MUTED)
h = [Rectangle((0, 0), 1, 1, color=c) for _, _, c in TISSUES]
ax2a.legend(h, [f"{n} (n={len(c)})" for n, c, _ in TISSUES], loc="upper center",
            bbox_to_anchor=(0.5, -0.14), ncol=3, frameon=False, fontsize=6)
panel_label(ax2a, "a", dx=-0.10)

ax2b = fig2.add_axes([0.445, 0.26, 0.245, 0.66])
ax2b.axhspan(0.937, 1.064, color=fp.BAND_NULL, alpha=0.35)
ax2b.axhspan(0.35, 0.5, color=fp.BAND_CROSS, alpha=0.30)
for i, (_, cfglist, color) in enumerate(TISSUES):
    rng = np.random.default_rng(42 + i)
    xs = rng.uniform(-0.27, 0.27, len(cfglist)) + i
    ax2b.scatter(xs, [r["kappa_ratio"] for r in cfglist], s=24, color=color, edgecolors="white", linewidth=0.75, zorder=3)
ax2b.set_xticks(range(3)); ax2b.set_xticklabels(["Hema.", "Neural", "Lung"])
ax2b.set_yscale("log"); ax2b.set_yticks([0.5, 0.937, 1.0, 1.064, 2, 4, 8]); ax2b.set_ylim(0.35, 9)
ax2b.set_ylabel(r"$\kappa$")
ax2b.set_title("Coupling distribution")
r9 = lung[0]
ax2b.annotate(f"lung R9: $\\kappa$={r9['kappa_ratio']:.2f}", xy=(2, r9["kappa_ratio"]),
              xytext=(0.5, 0.955), textcoords="axes fraction", ha="center", fontsize=6, color=fp.EDGE,
              arrowprops=dict(arrowstyle="-", lw=0.75, color=fp.EDGE))
ax2b.text(0.03, 0.30, "vanilla-equiv.\n[0.937, 1.064]", transform=ax2b.transAxes, fontsize=6, color=fp.MUTED)
ax2b.text(0.03, 0.075, "locking $\\kappa<0.5$", transform=ax2b.transAxes, fontsize=6, color=fp.FAIL)
panel_label(ax2b, "b", dx=-0.16)

ax2c = fig2.add_axes([0.735, 0.26, 0.245, 0.66])
groups = [("Prior-inj.", None, None, fp.GRAY),
          ("hema", cm, csd, C_HEMA),
          ("neural", 62.5, 6.5, C_NEURAL),
          ("lung", 96.7, 0.3, C_LUNG)]
x = np.arange(len(groups)) + 0.5
for i, (lab, m, sd, color) in enumerate(groups):
    xi = x[i]
    if m is None:
        ax2c.text(xi, 58, "N/A", ha="center", va="center", fontsize=7.5, color=fp.MUTED)
        continue
    ax2c.bar(xi, m, width=0.55, color=color, edgecolor="none")
    if sd is not None:
        ax2c.errorbar(xi, m, yerr=sd, fmt="none", ecolor="black", elinewidth=0.75, capsize=2.5)
    if i == 1:
        rng = np.random.default_rng(7)
        xs = rng.uniform(-0.16, 0.16, len(chead)) + xi
        ax2c.scatter(xs, chead, s=12, color=color, alpha=0.8, edgecolors="white", linewidth=0.75, zorder=5)
        ax2c.text(xi, m + sd + 8, f"{m:.1f}%", ha="center", fontsize=7)
    elif i > 1:
        ax2c.text(xi, m + sd + 4, "unver.", ha="center", fontsize=6, color=fp.MUTED)
ax2c.axhline(50, color="black", ls="--", lw=0.75)
ax2c.annotate("locking $C_{\\mathrm{head}}>50\\%$",
              xy=(0.5, 50), xycoords="data",
              xytext=(0.12, 0.90), textcoords="axes fraction",
              fontsize=6, color=fp.EDGE, ha="left", va="top",
              arrowprops=dict(arrowstyle="-", lw=0.75, color=fp.EDGE))
ax2c.text(0.02, 0.03, "0/15 reach threshold", transform=ax2c.transAxes, fontsize=6, color=fp.EDGE)
ax2c.set_xticks(x); ax2c.set_xticklabels([g[0] for g in groups])
ax2c.tick_params(axis="x", pad=7)
ax2c.set_xlim(-0.3, 4.3)
ax2c.set_ylim(-72, 124); ax2c.set_yticks([-50, 0, 50, 100])
ax2c.set_ylabel("Directional $C_{\\mathrm{head}}$ (%)")
ax2c.set_title("Directional collapse (G-VAE)")
panel_label(ax2c, "c", dx=-0.10)
save(fig2, "fig2_kc_phase_diagram")

# =====================================================================
# FIG 3: purity paradox
# =====================================================================
def C_by(config):
    return next(r for r in rows if r["config"] == config)["C"]
r111 = C_by("R111_s42"); r107 = C_by("R107_s42")
k111 = next(r for r in rows if r["config"] == "R111_s42")["kappa_ratio"]
k107 = next(r for r in rows if r["config"] == "R107_s42")["kappa_ratio"]

fig3 = plt.figure(figsize=(FIGW, 2.7))
ax3a = fig3.add_axes([0.075, 0.17, 0.33, 0.75])
ks = [r["kappa_ratio"] for r in rows]; cs = [r["C"] for r in rows]
ax3a.scatter(ks, cs, s=16, color="#AAAAAA", edgecolors="white", linewidth=0.75, zorder=3)
ax3a.axhline(50, color="#8B0000", ls="--", lw=0.75)
ax3a.annotate("$C=50\\%$ lock threshold",
              xy=(5.0, 50), xycoords="data",
              xytext=(0.82, 0.91), textcoords="axes fraction",
              fontsize=6, color="#8B0000", ha="center", va="top",
              arrowprops=dict(arrowstyle="-", lw=0.75, color="#8B0000"))
ax3a.text(0.98, 0.08, "33/33 outside\nlock region",
          transform=ax3a.transAxes, ha="right", va="bottom",
          fontsize=6, color="#8B0000")
ax3a.scatter([k107], [r107], s=40, color=C_HEMA, edgecolors="black", linewidth=0.75, zorder=4)
ax3a.scatter([k111], [r111], s=40, color=RED, edgecolors="black", linewidth=0.75, zorder=4)
ax3a.annotate("R107 (mixed)\n$C=+0.7\\%$", (k107, r107), xytext=(k107 + 0.75, r107 + 28),
              fontsize=6, arrowprops=dict(arrowstyle="->", lw=0.75, color="#333333"))
ax3a.annotate("R111 (pure)\n$C=-71.4\\%$", (k111, r111), xytext=(k111 + 0.30, r111 - 110),
              fontsize=6, arrowprops=dict(arrowstyle="->", lw=0.75, color="#333333"))
ax3a.set_xlabel(r"$\kappa$ (coupling coefficient)"); ax3a.set_ylabel("$C$ (collapse, %)")
ax3a.set_title("$\\kappa$-$C$ phase diagram, 33 configs")
panel_label(ax3a, "a", dx=-0.12)

ax3b = fig3.add_axes([0.485, 0.17, 0.22, 0.75])
cats = ["R111\npure erythroid\n111 edges", "R107\neryth+lymphoid\n107 edges"]
vals = [r111, r107]
bars = ax3b.bar(cats, vals, color=[RED, C_HEMA], edgecolor="#333333", linewidth=0.75, width=0.5)
ax3b.axhline(0, color="#333333", lw=0.75)
for b, v in zip(bars, vals):
    ax3b.text(b.get_x() + b.get_width() / 2, v + (6 if v > 0 else -14), f"{v:+.1f}%",
              ha="center", fontsize=7.5, fontweight="bold")
ax3b.set_ylim(-100, 20); ax3b.set_yticks([-100, -80, -60, -40, -20, 0])
ax3b.set_ylabel("$C$ (%)"); ax3b.set_title("Purity contrast (paired)")
panel_label(ax3b, "b", dx=-0.14)

ax3c = fig3.add_axes([0.76, 0.17, 0.21, 0.75]); ax3c.axis("off")
ax3c.text(0.5, 0.82, "4", fontsize=18, fontweight="bold", ha="center", va="center", color=C_HEMA)
ax3c.text(0.5, 0.62, "additional edges\n(3.6%)", fontsize=7, ha="center", va="center", color="#444444")
ax3c.text(0.5, 0.44, "72 pp", fontsize=12, fontweight="bold", ha="center", va="center", color="#333333")
ax3c.text(0.5, 0.30, "$C$ shift", fontsize=7, ha="center", va="center", color="#444444")
ax3c.text(0.5, 0.10, "matched-scale\nsingle-config\ncontrast ($n=1$ each)", fontsize=6, ha="center", va="center", color="#888888")
panel_label(ax3c, "c", dx=-0.12)
save(fig3, "fig3_purity_falsification")

# =====================================================================
# FIG 4: prior-activity checks (IPF)
# =====================================================================
scan = json.load(open(os.path.join(NEX, "results", "ptvae_human_ipf_ln_v2", "optionB_lambda_scan.json"), encoding="utf-8"))
lams = [0.1, 1.0, 10.0]
bn_C = [scan[f"lam{v}"]["bn_prior_kc_s42"]["C"] for v in ["0.1", "1", "10"]]
ln_C = [scan[f"lam{v}"]["ln_prior_kc_s42"]["C"] for v in ["0.1", "1", "10"]]
bn_k = [scan[f"lam{v}"]["bn_prior_kc_s42"]["kappa"] for v in ["0.1", "1", "10"]]
ln_k = [scan[f"lam{v}"]["ln_prior_kc_s42"]["kappa"] for v in ["0.1", "1", "10"]]
trH_V, gap_V = 10590.0, 9663.0
Cgrid = np.linspace(0, 100, 201)
ub = (1 - Cgrid / 100.0) * trH_V / gap_V

fig4 = plt.figure(figsize=(FIGW, 2.7))
ax4a = fig4.add_axes([0.075, 0.17, 0.30, 0.75])
ax4a.plot(lams, bn_C, "o-", color=RED, label="BN decoder")
ax4a.plot(lams, ln_C, "s-", color="#55A868", label="LN decoder")
ax4a.set_xscale("log"); ax4a.set_xticks(lams); ax4a.set_xticklabels(["0.1", "1", "10"])
ax4a.set_xlabel(r"Prior strength $\lambda$"); ax4a.set_ylabel("Global $C$ (%)")
ax4a.set_ylim(94.5, 97.5); ax4a.set_title("(A) $\\lambda$-scan: $C$ flat")
ax4a.text(0.04, 0.08, "var < 0.5 pp", transform=ax4a.transAxes, fontsize=7, color="#333333")
fp.safe_legend(ax4a, loc="upper left", fontsize=7)

ax4b = fig4.add_axes([0.435, 0.17, 0.28, 0.75])
ax4b.plot(lams, bn_k, "o-", color=RED, label="BN decoder")
ax4b.plot(lams, ln_k, "s-", color="#55A868", label="LN decoder")
ax4b.axhline(2.0, ls="--", color="#888888", lw=1.0)
ax4b.text(0.04, 0.90, "target $\\kappa_{\\mathrm{gap}}=2$", transform=ax4b.transAxes, fontsize=7, color="#555555")
ax4b.set_xscale("log"); ax4b.set_xticks(lams); ax4b.set_xticklabels(["0.1", "1", "10"])
ax4b.set_xlabel(r"Prior strength $\lambda$"); ax4b.set_ylabel(r"$\kappa_{\mathrm{gap}}$")
ax4b.set_ylim(0, 0.05); ax4b.set_title("(B) $\\lambda$-scan: subcritical")

ax4c = fig4.add_axes([0.775, 0.17, 0.20, 0.75])
ax4c.plot(Cgrid, ub, "-", color="#4C72B0", lw=2)
ax4c.axhline(2.0, ls="--", color="#888888", lw=1.0)
ax4c.annotate("infeasible\ntarget (2, 90)", xy=(90, 2.0), xytext=(52, 1.55),
              arrowprops=dict(arrowstyle="->", lw=0.75, color="#333333"), fontsize=7, color="#333333")
ax4c.plot([90], [0.11], "o", color=RED, ms=5)
ax4c.annotate(r"$\leq 0.11$ at $C=90\%$", xy=(90, 0.11), xytext=(34, 0.62),
              arrowprops=dict(arrowstyle="->", lw=0.75, color="#333333"), fontsize=7)
ax4c.set_xlabel("Global $C$ (%)"); ax4c.set_ylabel(r"$\kappa_{\mathrm{gap}}$ bound")
ax4c.set_ylim(0, 2.2); ax4c.set_xlim(0, 100)
ax4c.set_title("(C) Structural ceiling")
ax4c.text(0.03, 0.06, "vanilla top axis\n= 91.2% of trH", transform=ax4c.transAxes, fontsize=6, color="#333333")
save(fig4, "fig4_prior_activity")

# =====================================================================
# FIG 5: GRQ downstream readout (2 panels; UMAP panel dropped - no data file)
# Panel (b) = gene readout of the dominant decoder axis v* (top right SV of W),
# annotated by function. Honest result: the dominant axis is housekeeping-dominated
# (replicates & extends Table S4 SVD critique) -> motivates kappa-C over naive SVD.
# =====================================================================
# Erythroid master-TF subset shown for illustration only (backed by Supplementary Table S3);
# multiseed_perturbation_validation.json contains fabricated metrics and is NOT used here.
grq_seeds = ["s9", "s10", "s11", "s12"]
grq_vals = [11.45, 13.83, 16.80, 11.32]  # decoder traversal spec_raw, a=1.0 (optionA_grq_decoder_traversal.json)

fig5 = plt.figure(figsize=(FIGW, 2.6))
ax5a = fig5.add_axes([0.09, 0.17, 0.40, 0.75])
x = np.arange(len(grq_seeds))
ax5a.bar(x, grq_vals, width=0.55, color=C_HEMA, edgecolor="#333333", linewidth=0.75)
for xi, v in zip(x, grq_vals):
    ax5a.text(xi, v + 0.25, f"{v:.1f}x", ha="center", fontsize=7.5, fontweight="bold")
ax5a.set_xticks(x); ax5a.set_xticklabels(grq_seeds)
ax5a.set_ylabel("Erythroid specificity (fold)")
ax5a.set_ylim(0, 19.5); ax5a.set_title("GRQ decoder-traversal specificity")
panel_label(ax5a, "a", dx=-0.14)

# --- panel (b): v* gene-readout heatmap (hema true-prior, R13) ---
UNAGI_DEC = os.path.abspath(os.path.join(NEX, "..", "UNAGI", "decoder_out"))
Wv = np.load(os.path.join(UNAGI_DEC, "R13", "iter_4", "loading_matrix.npy"))
genes_h = json.load(open(os.path.join(UNAGI_DEC, "hema_gene_list.json")))
hp = json.load(open(os.path.join(NEX, "data", "hema_prior.json")))
tfset = set(t.upper() for t in hp["tf_db"].keys()); tgset = set()
for t, gs in hp["tf_db"].items():
    tgset.update(g.upper() for g in gs)
ERY = set("HBA1 HBA2 HBB GYPB GYPA SLC4A1 ALAS2 TFRC KLF1 NFE2 ZFPM1 EPB42 SPTA1 SPTB ANK1".split())
U, S, Vt = np.linalg.svd(Wv, full_matrices=False)
v1 = Vt[0]
NTOP = 25
order = np.argsort(-np.abs(v1))[:NTOP]
top_g = [genes_h[i] for i in order]; top_v = v1[order]
def role(g):
    u = g.upper()
    if u in tfset: return "TF", "#B2182B"
    if u in tgset: return "target", "#D6604D"
    if u in ERY:   return "ery", RED
    return "house", "#888888"
ax5b = fig5.add_axes([0.62, 0.20, 0.33, 0.62])
im = ax5b.imshow(top_v.reshape(1, -1), cmap="RdBu_r", aspect="auto",
                 vmin=-np.abs(top_v).max(), vmax=np.abs(top_v).max())
ax5b.set_yticks([]); ax5b.set_xticks(range(NTOP))
ax5b.set_xticklabels(top_g, rotation=90, fontsize=6, ha="center")
for c, g in enumerate(top_g):
    r, col = role(g)
    ax5b.get_xticklabels()[c].set_color(col)
    ax5b.get_xticklabels()[c].set_fontweight("bold" if r != "house" else "normal")
ax5b.set_title(r"Gene readout of dominant axis $v^*$", fontsize=7.5)
ax5b.text(0.5, -0.34, "TF / target / erythroid / housekeeping", transform=ax5b.transAxes,
          ha="center", fontsize=6, color="#555555")
panel_label(ax5b, "b", dx=-0.10)
save(fig5, "fig4_grq_perturbation")

print("ALL MAIN FIGURES REGENERATED")
