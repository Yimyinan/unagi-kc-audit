"""Rebuild Supplementary Figure S8 as a vanilla-equivalence BAND CALIBRATION figure.

WHY
---
The S8 slot previously held a near-duplicate of the main Figure 2 (both were
"decoder-centric kappa-C audit of the 33 configurations").  Now that main Fig. 2
shows the configurations against *both* bands, S8 is redundant as a repeat and is
re-purposed to show *where the two bands come from* --- i.e. the per-tissue
vanilla-vs-vanilla nulls that define them --- which is exactly the evidence behind
the "band-independent deep-compression verdict" claim and the answer to the
cross-family-band objection.

Band values are the COMPUTED null of record, `outputs/positive_result/s1k_gvae_band.json`,
which evaluates both statistics on the same 15 G-VAE hematopoietic vanilla seeds over 210
ordered seed-pair comparisons:

    trace form      tr(W^T W)     mean 1.0062 +/- 0.1123  -> +/-2sigma = [0.781, 1.231]
    corrected form  sum_g R^2_g   mean 1.0005 +/- 0.0319  -> +/-2sigma = [0.9367, 1.0644]

The CORRECTED form is the one the manuscript reports (1.001 +/- 0.032 -> [0.937, 1.064]); the
trace form's null sd is inflated ~13x (`s1m2_band_summary.json: trace_form_sd_inflation`), which
is why the two bands differ so much. The family-matched UNAGI recalibration (R14 and R0 families)
is now reported in the CORRECTED predictability form (star_methods.tex): hematopoietic
(empty-vector vehicle) kappa = 1.000 +/- 0.015 -> [0.971, 1.029], neural-crest
kappa = 1.005 +/- 0.146 -> [0.713, 1.298], combined kappa = 1.001 +/- 0.043 -> [0.915, 1.086];
lung has a single vanilla run and is excluded. The earlier trace-form family bands
([0.65, 1.38] / [0.22, 1.96] / [0.38, 1.72]) are superseded and are NOT used for classification.

USAGE:  python figures/make_figS8_band_calibration.py
"""
import os
import sys
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "code"))

from figure_palette import apply_paper_style, C_HEMA, C_NEURAL, C_LUNG, RED, GRAY, C4  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

JSON = os.path.join(ROOT, "data", "kappa_c_pr_comparison.json")
OUT = os.path.join(HERE, "..", "supplementary", "supp_figS8_kc_phase_diagram.pdf")

# (label, kappa mean, kappa sd) per calibration, per tissue, CORRECTED statistic
CAL = {
    "hemo":   {"gv": (1.001, 0.032), "un": (1.000, 0.015)},
    "neural": {"gv": (1.001, 0.032), "un": (1.005, 0.146)},
    "lung":   {"gv": (1.001, 0.032), "un": None},        # single vanilla run
}
TISSUE = {"hemo": ("Hematopoietic", C_HEMA),
          "neural": ("Neural crest", C_NEURAL),
          "lung": ("Lung fibrosis", C_LUNG)}
LOCK = 0.5


def band(m, s):
    return (m - 2 * s, m + 2 * s)


def main():
    d = json.load(open(JSON, encoding="utf-8"))
    rows = d["rows"]

    # scatter in the CORRECTED predictability statistic (the released trace form is
    # not invariant to the latent basis; see S1F_0OF33_RECHECK.md)
    _s1g = os.path.join(HERE, "..", "..", "..", "outputs", "positive_result",
                        "s1g_corrected_summaries.json")
    corr = {r["config"]: r["kappa_r2"]
            for r in json.load(open(os.path.abspath(_s1g), encoding="utf-8-sig"))["rows"]}
    miss = [r["config"] for r in rows if r["config"] not in corr]
    if miss:
        raise SystemExit("missing corrected kappa for: %s" % miss)

    apply_paper_style()
    fig, ax = plt.subplots(figsize=(7.4, 4.6))

    for i, key in enumerate(["hemo", "neural", "lung"]):
        name, col = TISSUE[key]
        lo_gv, hi_gv = band(*CAL[key]["gv"])
        ax.plot([lo_gv, hi_gv], [i + 0.16, i + 0.16], lw=7, solid_capstyle="butt",
                color="0.62", zorder=1)
        if CAL[key]["un"] is not None:
            lo_u, hi_u = band(*CAL[key]["un"])
            ax.plot([lo_u, hi_u], [i - 0.16, i - 0.16], lw=7, solid_capstyle="butt",
                    color="#C3CCE0", zorder=1)
        else:
            ax.text(1.02, i - 0.16, "n/a (single vanilla run)", fontsize=7,
                    color="0.45", va="center")
        ks = [corr[r["config"]] for r in rows if r["tissue"] == key]
        ax.scatter(ks, [i + np.random.default_rng(3).uniform(-0.07, 0.07) for _ in ks],
                   c=col, s=22, alpha=0.9, edgecolor="white", linewidth=0.75, zorder=3,
                   label=name if i == 0 else None)

    ax.axvline(LOCK, ls="--", lw=1.25, color=RED, zorder=4)
    ax.text(LOCK * 0.93, 2.55, "deep compression\n$\\kappa<0.5$", fontsize=7.5,
            color=RED, ha="right")
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels([TISSUE[k][0] for k in ["hemo", "neural", "lung"]])
    ax.set_xscale("log")
    ax.set_xlabel(r"$\kappa$ (log scale)")
    ax.set_title("Vanilla-equivalence band calibration: cross-family vs family-matched",
                 loc="left", fontsize=9.5)
    ax.legend(handles=[
        plt.Line2D([], [], lw=7, color="0.62", label="G-VAE cross-family band ($\\pm2\\sigma$)"),
        plt.Line2D([], [], lw=7, color="#C3CCE0", label="UNAGI family-matched band ($\\pm2\\sigma$)"),
        plt.Line2D([], [], marker="o", ls="", color="0.25", markersize=5,
                   label="audited configurations"),
    ], loc="lower right", frameon=False, fontsize=7)

    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    fig.savefig(OUT.replace(".pdf", ".png"), dpi=170, bbox_inches="tight")
    print("[saved]", os.path.abspath(OUT))


if __name__ == "__main__":
    main()
