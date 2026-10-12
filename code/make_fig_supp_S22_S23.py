"""Generate Supplementary Fig. S22 (three-way kappa-C dual-spectrum on the
Norman CRISPR benchmark) and Supplementary Fig. S23 (GRQ virtual-perturbation
closure). Reads all numbers directly from the JSON artifacts so nothing is
transcribed by hand.

Style mirrors the existing make_*.py scripts: sans-serif 7.5 pt, 6.7 in canvas,
300 dpi, vanilla-equivalence band [0.937, 1.064].

Panel B note. It originally plotted the *absolute* effective ranks of the two
published runs and the caption read the within-architecture change off them (-30% for
scVI, -2.8% for family 4). Both are single draws, and refitting both arms at matched
seeds shows the scVI ratio averages ~1.0 (no resolved change) while the family-4 ratio
is ~0.87 rather than 0.97. So the panel now shows the per-seed *paired* ratio, which is
the quantity that actually replicates, with the published single run marked for
comparison.
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = "F:/UNAGI_Project/paper1/paper1_kc_diagnostic"
FIGDIR = f"{ROOT}/figures"

# ---- style: single source of truth ---------------------------------------
import sys as _sys
_sys.path.insert(0, ROOT + "/code")
import figure_palette as fp
fp.apply_paper_style()

BLUE = fp.C_HEMA       # decoder spectrum  kappa-C(W)
ORANGE = fp.C_NEURAL   # latent spectrum    kappa-C(Z)
GREEN = fp.C_LUNG      # family-4 model
GREY = fp.GRAY         # vanilla reference
BAND = "#EEF2F7"       # vanilla-equivalence band
VAN_LO, VAN_HI = 0.937, 1.064


def pr_effrank(cov):
    """Participation ratio of a covariance spectrum."""
    e = np.linalg.eigvalsh(cov + 1e-10 * np.eye(cov.shape[0]))
    e = np.clip(e, 0, None)
    return float((e.sum() ** 2) / (e ** 2).sum())


# ---- load numbers --------------------------------------------------------
grq = json.load(open(f"{ROOT}/data/grq_closure_results.json"))
kap = json.load(open(f"{ROOT}/data/kappac_realdata_verification.json"))
f4 = json.load(open(f"{ROOT}/data/vae_family4_results.json"))
sel = f4["selected"]

# verify scVI effective ranks from raw latents (should reproduce 8.80 / 6.15)
Z = np.load(f"{ROOT}/data/_norman_latents.npz", allow_pickle=True)
er_v = pr_effrank(np.cov(Z["Zv"], rowvar=False))
er_p = pr_effrank(np.cov(Z["Zp"], rowvar=False))
print(f"[check] scVI eff_rank vanilla={er_v:.2f} conditioned={er_p:.2f} "
      f"(JSON says {kap['eff_rank_vanilla']:.2f} / {kap['eff_rank_model']:.2f})")

# three models on the SAME Norman data
models = ["Vanilla\nscVI", "Perturbation-\nconditioned scVI",
          "Decoder-only-\nconditioned VAE\n(family 4)"]
kc_W = [1.00, kap["kappa_C_W"], sel["kappa_C_W"]]
kc_Z = [1.00, kap["kappa_C_Z"], sel["kappa_C_Z"]]
er_van = [er_v, er_v, sel["eff_rank_vanilla"]]
er_mod = [er_p, er_p, sel["eff_rank_conditioned"]]

# =========================================================================
# Fig. S22 -- three-way kappa-C dual-spectrum
# =========================================================================
fig, (axA, axB) = plt.subplots(1, 2, figsize=(6.7, 2.7))

# --- Panel A: grouped kappa-C(W) / kappa-C(Z) ---
x = np.arange(len(models))
w = 0.36
axA.axhspan(VAN_LO, VAN_HI, color=BAND, zorder=0)
axA.axhline(1.0, color=GREY, ls="--", lw=0.9, zorder=1)
b1 = axA.bar(x - w / 2, kc_W, w, color=BLUE, label=r"$\kappa$-$C(W)$ (decoder)",
             edgecolor="white", linewidth=0.5, zorder=3)
b2 = axA.bar(x + w / 2, kc_Z, w, color=ORANGE, label=r"$\kappa$-$C(Z)$ (latent)",
             edgecolor="white", linewidth=0.5, zorder=3)
for xi, (vw, vz) in enumerate(zip(kc_W, kc_Z)):
    axA.text(xi - w / 2, vw + 0.015, f"{vw:.2f}", ha="center", va="bottom",
             fontsize=6.2, color=BLUE)
    axA.text(xi + w / 2, vz + 0.015, f"{vz:.2f}", ha="center", va="bottom",
             fontsize=6.2, color=ORANGE)
axA.set_xticks(x)
axA.set_xticklabels(models)
axA.set_ylabel(r"$\kappa$-$C$ ratio (model / vanilla)")
axA.set_ylim(0.55, 1.45)
axA.set_title("A.  Dual-spectrum ratios", loc="left")
axA.legend(loc="upper right", frameon=False, fontsize=6.6)
axA.text(0.01, 0.97, "grey band = vanilla-equivalence [0.937, 1.064]",
         transform=axA.transAxes, fontsize=5.8, color="#555555", va="top")

# --- Panel B: latent effective-rank ratio, paired design with seed scatter ---
# NB: ROOT here is the paper1 directory, while the analysis artifacts live at the
# workspace root under outputs/positive_result.
OUTDIR = os.path.abspath(os.path.join(ROOT, "..", "..", "outputs", "positive_result"))


def _load_json(name):
    p = os.path.join(OUTDIR, name)
    return json.load(open(p)) if os.path.exists(p) else None


groups = []
_nm = _load_json("s1n3_norman_injected_paired.json")
if _nm and _nm.get("per_seed"):
    groups.append(("Norman\nscVI",
                   [r["eff_rank_ratio"] for r in _nm["per_seed"]],
                   _nm["published_single_pair"]["eff_rank_ratio"], BLUE))
_f4 = _load_json("s1o_family4_seed_band.json")
if _f4 and _f4.get("per_seed"):
    _f4pub = _f4["published"]
    _f4_ratio = _f4pub.get("eff_rank_ratio")
    if _f4_ratio is None:      # older artifact stores the two ranks separately
        _f4_ratio = _f4pub["eff_rank_conditioned"] / _f4pub["eff_rank_vanilla"]
    groups.append(("family-4\nMLP VAE",
                   [r["eff_rank_ratio"] for r in _f4["per_seed"]],
                   _f4_ratio, GREEN))

axB.axhspan(VAN_LO, VAN_HI, color=BAND, zorder=0)
axB.axhline(1.0, color=GREY, ls="--", lw=0.9, zorder=1)
_rng = np.random.default_rng(0)
for _i, (_lab, _vals, _pub, _col) in enumerate(groups):
    _vals = np.asarray(_vals, float)
    axB.scatter(np.full(_vals.size, _i) + _rng.uniform(-0.075, 0.075, _vals.size),
                _vals, s=20, color=_col, edgecolors="white", linewidths=0.4, zorder=4)
    _m = float(_vals.mean())
    _sd = float(_vals.std(ddof=1)) if _vals.size > 1 else 0.0
    axB.errorbar([_i], [_m], yerr=[_sd], fmt="o", ms=4.5, color=_col, capsize=3,
                 lw=1.1, zorder=5)
    axB.scatter([_i], [_pub], marker="D", facecolors="none", edgecolors="0.35",
                s=24, linewidths=0.9, zorder=6)
    _t = (_m - 1) / (_sd / np.sqrt(_vals.size)) if _vals.size > 1 and _sd > 0 else np.nan
    # annotation must stay INSIDE the axes: text placed outside the data limits is still
    # laid out by tight_layout and silently balloons the canvas with white space.
    axB.text(_i + 0.19, 0.03, f"$n$={_vals.size}\n$t$={_t:+.1f}", ha="center",
             va="bottom", fontsize=6.0, color=_col,
             transform=axB.get_xaxis_transform())
axB.set_xticks(np.arange(len(groups)))
axB.set_xticklabels([g[0] for g in groups])
axB.set_xlim(-0.55, len(groups) - 0.45)
_lo = min(min(g[1]) for g in groups)
_hi = max(max(g[1]) for g in groups)
_pad = 0.18 * (_hi - _lo)
axB.set_ylim(_lo - 2.1 * _pad, _hi + 0.6 * _pad)
axB.set_ylabel("Latent effective-rank ratio\n(conditioned / vanilla)")
axB.set_title("B.  Latent rank: paired, with seed scatter", loc="left")
axB.text(0.02, 0.98, "dots = seeds, bar = mean$\\pm$s.d.,\nopen diamond = published single run",
         transform=axB.transAxes, fontsize=5.8, color="#555555", va="top")

fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(f"{FIGDIR}/figS22_norman_threeway.{ext}")
print("wrote figS22_norman_threeway.pdf/.png")
plt.close(fig)

# =========================================================================
# Fig. S23 -- GRQ virtual-perturbation closure
# =========================================================================
d = grq["summary"]
n = d["rho_grq_pert"]["n"]

rho_labels = ["GRQ virtual\nperturbation", "GRQ vanilla\n(readout)",
              "Mean-shift\nbaseline", "Mean-shift\n(vanilla latent)"]
rho_means = [d["rho_grq_pert"]["mean"],
             d["rho_grq_vanilla"]["mean"],
             d["rho_shift_baseline"]["mean"],
             d["rho_shift_vanilla"]["mean"]]
rho_sd = [d["rho_grq_pert"]["sd"],
          d["rho_grq_vanilla"]["sd"],
          d["rho_shift_baseline"]["sd"],
          d["rho_shift_vanilla"]["sd"]]
rho_cols = [BLUE, BLUE, ORANGE, ORANGE]

spec_labels = ["GRQ virtual\nperturbation", "GRQ vanilla\n(readout)",
               "Mean-shift\nbaseline", "Observed\n(perturbed cells)"]
spec_means = [d["spec_grq_pert"]["mean"], d["spec_grq_vanilla"]["mean"],
              d["spec_shift"]["mean"], d["spec_observed"]["mean"]]
spec_sd = [d["spec_grq_pert"]["sd"], d["spec_grq_vanilla"]["sd"],
           d["spec_shift"]["sd"], d["spec_observed"]["sd"]]
spec_cols = [BLUE, BLUE, ORANGE, GREY]

ci = 1.96 * np.array(rho_sd) / np.sqrt(n)
sci = 1.96 * np.array(spec_sd) / np.sqrt(n)

fig, (axA, axB) = plt.subplots(1, 2, figsize=(6.7, 2.7))

xr = np.arange(len(rho_labels))
axA.axhline(0, color=GREY, ls=":", lw=0.8, zorder=1)
bars = axA.bar(xr, rho_means, 0.6, color=rho_cols, edgecolor="white",
               linewidth=0.5, zorder=3, yerr=ci, capsize=3, ecolor="#444444")
for xi, v in zip(xr, rho_means):
    axA.text(xi, v + (0.012 if v >= 0 else -0.028), f"{v:+.2f}",
             ha="center", va="bottom" if v >= 0 else "top", fontsize=6.2)
axA.set_xticks(xr)
axA.set_xticklabels(rho_labels)
axA.set_ylabel(r"Spearman $\rho$  (predicted vs observed DE direction)")
axA.set_ylim(-0.18, 0.42)
axA.set_title("A.  Response-direction prediction", loc="left")
axA.text(0.01, 0.02, f"n = {n} perturbations, 95% CI",
         transform=axA.transAxes, fontsize=5.8, color="#555555")

xs = np.arange(len(spec_labels))
bars = axB.bar(xs, spec_means, 0.6, color=spec_cols, edgecolor="white",
               linewidth=0.5, zorder=3, yerr=sci, capsize=3, ecolor="#444444")
for xi, v in zip(xs, spec_means):
    axB.text(xi, v + sci[xi] + 0.15, f"{v:.1f}$\\times$",
             ha="center", va="bottom", fontsize=6.2)
axB.set_xticks(xs)
axB.set_xticklabels(spec_labels)
axB.set_ylabel("Target specificity (fold enrichment)")
axB.set_ylim(0, max(np.array(spec_means) + np.array(sci)) * 1.18)
axB.set_title("B.  Target specificity", loc="left")

fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(f"{FIGDIR}/figS23_grq_closure.{ext}")
print("wrote figS23_grq_closure.pdf/.png")
plt.close(fig)
print("DONE")
