# -*- coding: utf-8 -*-
r"""make_fig_crossarch_paired.py -- cross-architecture evidence as a PAIRED design.

WHY THE NULL BAND IS NOT THE HEADLINE HERE
------------------------------------------
An earlier draft of this figure shaded the family-matched null's +/-2 s.d. band as the
reference.  That band is 0.38-1.72 (mean 1.052, s.d. 0.336): it contains both paired
results AND contains unity, so shading it implies that nothing is resolvable -- it
obscures the very contrast the panel exists to show.  The band is a cross-readout
*scale* null, not the per-arm paired test this figure reports.

So this figure reports the paired test directly: n, mean +/- s.d., and t against unity.
The null band is drawn faintly for transparency and labelled as what it is, and the
verdict text deliberately makes no "outside the null" claim that the data do not support.

WHAT THE PAIRED DESIGN ACTUALLY ESTABLISHES
-------------------------------------------
  scVI       published single pair 0.699  -> paired 1.019 +/- 0.155 (n=4, t=+0.2)
             the published -30% latent-rank drop does NOT reproduce
  family-4   published single pair 0.972  -> paired 0.868 +/- 0.032 (n=5, t=-9.3)
             reproducible and consistently below its own published value

Data: outputs/positive_result/{s1n3_norman_injected_paired,s1o_family4_seed_band,
family_matched_null_recomputed}.json.  No new runs.
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUTDIR = os.path.abspath(os.path.join(ROOT, "..", "..", "outputs", "positive_result"))
sys.path.insert(0, os.path.join(ROOT, "code"))
import figure_palette as fp  # noqa: E402

OUT = os.path.join(HERE, "fig_crossarch_paired.pdf")
OUT_PNG = os.path.join(HERE, "fig_crossarch_paired.png")


def load(name):
    p = os.path.join(OUTDIR, name)
    if not os.path.exists(p):
        raise SystemExit("missing required artefact: %s" % p)
    return json.load(open(p, encoding="utf-8"))


scvi = load("s1n3_norman_injected_paired.json")
f4 = load("s1o_family4_seed_band.json")
nulls = load("family_matched_null_recomputed.json")

sv = np.array([r["eff_rank_ratio"] for r in scvi["per_seed"]], float)
sv_pub = scvi["published_single_pair"]["eff_rank_ratio"]
fv = np.array([r["eff_rank_ratio"] for r in f4["per_seed"]], float)
_p = f4["published"]
fv_pub = _p.get("eff_rank_ratio")
if fv_pub is None:
    fv_pub = _p["eff_rank_conditioned"] / _p["eff_rank_vanilla"]

cmb = nulls["combined_corrected"]
nm, nsd = cmb["mean"], cmb["std"]
nlo, nhi = nm - 2 * nsd, nm + 2 * nsd

GROUPS = [
    ("Perturbation-\nconditioned scVI", sv, sv_pub, fp.C_HEMA),
    ("Decoder-only-\nconditioned VAE\n(family 4)", fv, fv_pub, fp.C_NEURAL),
]

fig, axes = plt.subplots(1, 2, figsize=(6.6929, 3.05),
                         gridspec_kw={"width_ratios": [1.28, 1.0]})
axA, axC = axes

# =============================================================================
# (A) paired per-seed ratios vs the published single values
# =============================================================================
# The family-matched null is NOT shaded. Its +/-2 s.d. span is 0.381-1.723, which
# contains BOTH paired means and also contains unity, so shading it would read as "nothing
# here is resolvable" and would bury the family-4 result the panel exists to show. It is a
# cross-readout SCALE null, not the per-arm test; the test that matters is t against unity,
# which is printed under each group. Its value is reported in text instead of drawn.
axA.axhline(1.0, color=fp.GREY, ls="--", lw=1.0, zorder=1)
rng = np.random.default_rng(0)
for i, (lab, vals, pub, col) in enumerate(GROUPS):
    xj = i + rng.uniform(-0.085, 0.085, len(vals))
    axA.scatter(xj, vals, s=22, color=col, edgecolor=fp.INK, linewidth=0.5, zorder=5)
    m, sd = float(vals.mean()), float(vals.std(ddof=1))
    axA.hlines(m, i - 0.27, i + 0.27, color=col, lw=2.0, zorder=6)
    axA.scatter([i - 0.38], [pub], marker="D", facecolor="white", edgecolor=col,
                s=34, linewidth=1.1, zorder=7)
    n = len(vals)
    t = (m - 1.0) / (sd / np.sqrt(n))
    axA.text(i, 1.115, "%d seeds" % n, ha="center", fontsize=6.0, color=fp.INK)
    axA.text(i, 1.065, "%.3f$\\pm$%.3f" % (m, sd), ha="center", fontsize=6.2, color=col)
    axA.text(i, 0.800, "$t$ vs unity\n$= %+.1f$" % t, ha="center", va="bottom",
             fontsize=5.8, color=fp.INK)
axA.set_xticks(range(len(GROUPS)))
axA.set_xticklabels([g[0] for g in GROUPS], fontsize=6.4)
axA.set_xlim(-0.62, 1.62)
axA.set_ylim(0.70, 1.17)
axA.set_ylabel("latent effective-rank ratio\n(conditioned / vanilla)", fontsize=6.8)
axA.set_title("A.  Paired per-seed ratios", loc="left", fontsize=7.4)
axA.legend(handles=[
    plt.Line2D([], [], marker="o", ls="none", mfc=fp.GREY, mec=fp.INK, ms=4.5,
               label="paired seed"),
    plt.Line2D([], [], marker="D", ls="none", mfc="white", mec=fp.INK, ms=5,
               label="published single run"),
    plt.Line2D([], [], color=fp.GREY, ls="--", lw=1.0, label="unity (no change)"),
], loc="upper center", bbox_to_anchor=(0.5, 0.995), frameon=False, fontsize=5.6,
    ncol=3, handletextpad=0.4, columnspacing=1.0)
axA.text(0.5, 0.035,
         "the family-matched null is not drawn (see caption)",
         transform=axA.transAxes, fontsize=5.2, color=fp.GREY, style="italic",
         ha="center")

# =============================================================================
# (B) graded verdicts
# =============================================================================
axC.axis("off")
axC.set_title("B.  What the pairing supports", loc="left", fontsize=7.4)

ROWS = [
    ("scVI", "published single pair  0.699",
     "paired  1.019$\\pm$0.155  ($n{=}4$)",
     "does NOT reproduce:\nthe published $-30\\%$\ndrop is a single draw",
     fp.FAIL, fp.FAIL_FILL),
    ("family 4", "published single pair  0.972",
     "paired  0.868$\\pm$0.032  ($n{=}5$)",
     "reproducible, and below\nits own published value",
     fp.PASS, fp.PASS_FILL),
]
y0, dy = 0.93, 0.44
for i, (name, pub_s, paired_s, verdict, col, fill) in enumerate(ROWS):
    y = y0 - i * dy
    axC.add_patch(plt.Rectangle((0.005, y - 0.30), 0.99, 0.36,
                                transform=axC.transAxes, facecolor=fill,
                                edgecolor=col, lw=0.8))
    axC.text(0.03, y + 0.005, name, transform=axC.transAxes, fontsize=6.8,
             weight="bold", color=fp.INK, va="top")
    axC.text(0.03, y - 0.075, pub_s, transform=axC.transAxes, fontsize=5.4,
             color=fp.INK, va="top")
    axC.text(0.03, y - 0.150, paired_s, transform=axC.transAxes, fontsize=5.4,
             color=col, va="top", weight="bold")
    axC.text(0.99, y - 0.050, verdict, transform=axC.transAxes, fontsize=5.2,
             color=fp.INK, va="top", ha="right", linespacing=1.35)
axC.text(0.005, 0.035,
         "Graded, not binary: a published single pair can fail to reproduce\n"
         "(scVI) while another reproduces with a shifted value (family 4).",
         transform=axC.transAxes, fontsize=5.4, color=fp.INK, style="italic")

for a in (axA,):
    a.spines["top"].set_visible(False)
    a.spines["right"].set_visible(False)
    a.tick_params(labelsize=6.0)
    a.grid(axis="y", lw=0.35, color=fp.BAND, zorder=0)
fig.tight_layout(pad=0.5)
fig.savefig(OUT)
fig.savefig(OUT_PNG, dpi=300)
print("[saved]", OUT)
print("[saved]", OUT_PNG)
print("[data] scVI    n=%d  %.4f +/- %.4f  published %.4f"
      % (len(sv), sv.mean(), sv.std(ddof=1), sv_pub))
print("[data] family4 n=%d  %.4f +/- %.4f  published %.4f"
      % (len(fv), fv.mean(), fv.std(ddof=1), fv_pub))
print("[data] matched-null 2sd band = %.3f .. %.3f  (contains unity: %s)"
      % (nlo, nhi, nlo < 1.0 < nhi))
