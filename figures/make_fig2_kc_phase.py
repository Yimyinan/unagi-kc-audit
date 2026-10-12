"""Regenerate main Figure 2 (the kappa-C verdict) from the AUTHORITATIVE JSON.

WHY THIS SCRIPT EXISTS
----------------------
The original ``figures/fig2_kc_phase_diagram.pdf`` (dated Sep 5) predates the
authoritative recomputation in ``data/kappa_c_pr_comparison.json`` (Sep 12) and is
emitted by ``figures/generate_main_figures.py`` --- the "umbrella" generator whose
inputs are missing in this environment, so the figure cannot be rebuilt there.
This script bypasses the umbrella and redraws the panel set from the single
source of truth.

PANEL SET (revised 2026-09-29, r35)
-----------------------------------
Panel (C), the historical "directional collapse" bar chart, is REMOVED: its own
legend read "excluded from evidence base" and the caption called it historical,
so it should not occupy a main-figure panel.

Panel (A) is made POSITIONAL: kappa is drawn on a log axis that spans the
deep-compression boundary (kappa < 0.5) as a shaded region, together with both
vanilla-equivalence bands.  The 0/33 verdict is then a position a reader can see
rather than a count they must perform.

NOTE ON "PHASE DIAGRAM".  A genuine two-dimensional kappa-vs-C phase plane is
degenerate here: C is defined as (1 - kappa) x 100%, so every configuration lies
exactly on that line and the plane carries one degree of freedom, not two.  The
verdict is therefore drawn as a one-dimensional phase line with the forbidden
region shaded.

STYLING
-------
All colours come from figure_palette (Okabe-Ito, colour-blind safe).  No data
value, threshold or axis range was changed.

USAGE:  python figures/make_fig2_kc_phase.py
"""
import os
import sys
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "code"))   # figure_palette lives in code/

import figure_palette as fp  # noqa: E402
from figure_palette import (apply_paper_style, C_HEMA, C_NEURAL, C_LUNG,  # noqa: E402
                            FAIL, FAIL_FILL, INK, BAND_NULL, BAND_CROSS, MUTED)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

JSON = os.path.join(ROOT, "data", "kappa_c_pr_comparison.json")
OUT = os.path.join(HERE, "fig2_kc_phase_diagram.pdf")

TISSUE = {
    "hemo":   ("Hematopoietic (n=22)", C_HEMA),
    "neural": ("Neural crest (n=10)",  C_NEURAL),
    "lung":   ("Lung fibrosis (n=1)",  C_LUNG),
}
BAND_GVAE = (0.937, 1.064)   # cross-family reference (G-VAE, 15 seeds, corrected)
BAND_UNAGI = (0.915, 1.086)  # family-matched UNAGI, corrected predictability form
LOCK = 0.5
ORDER = ["hemo", "neural", "lung"]


def main():
    d = json.load(open(JSON, encoding="utf-8"))
    rows = d["rows"]
    assert len(rows) == d["n"] == 33, (len(rows), d["n"])

    # Both panels are drawn in the CORRECTED predictability statistic; the released
    # trace form is not invariant to the latent basis and is retained only in the
    # reproduction record (S1F_0OF33_RECHECK.md).
    _s1g = os.path.abspath(os.path.join(HERE, "..", "..", "..", "outputs",
                                        "positive_result",
                                        "s1g_corrected_summaries.json"))
    corr = {r["config"]: r["kappa_r2"]
            for r in json.load(open(_s1g, encoding="utf-8-sig"))["rows"]}
    miss = [r["config"] for r in rows if r["config"] not in corr]
    if miss:
        raise SystemExit("missing corrected kappa for: %s" % miss)

    def KC(r):
        return corr[r["config"]]

    apply_paper_style()
    # Canvas width = the final printed width (\textwidth = 6.6929 in): generating
    # wider makes LaTeX shrink the whole figure, taking font sizes and stroke
    # weights down with it (a 13.6 in canvas shrinks to x0.49, 7.5 pt -> 3.7 pt).
    fig, ax = plt.subplots(1, 2, figsize=(6.6929, 3.05))

    # ------------------------------------------------- (A) the phase line
    # Forbidden region: kappa < 0.5 (equivalently C > 50%), shaded across the panel.
    ax[0].axvspan(0.30, LOCK, color=FAIL_FILL, zorder=0)
    ax[0].axvline(LOCK, ls="--", lw=1.25, color=FAIL, zorder=5)
    ax[0].axvspan(*BAND_UNAGI, color=BAND_NULL, zorder=1)
    ax[0].axvspan(*BAND_GVAE, color=BAND_CROSS, zorder=2)
    for i, key in enumerate(ORDER):
        ks = [KC(r) for r in rows if r["tissue"] == key]
        rng = np.random.default_rng(1)
        ax[0].scatter(ks, [i + rng.uniform(-0.16, 0.16) for _ in ks],
                      s=30, alpha=0.9, color=TISSUE[key][1],
                      edgecolor="white", linewidth=0.75, zorder=6)
    ax[0].set_xscale("log")
    ax[0].set_xlim(0.30, 1.30)
    ax[0].set_ylim(-0.7, 2.7)
    ax[0].set_xticks([0.3, 0.5, 0.7, 1.0, 1.3])
    ax[0].set_xticklabels(["0.3", "0.5", "0.7", "1.0", "1.3"])
    ax[0].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax[0].set_yticks(list(range(3)))
    ax[0].set_yticklabels([TISSUE[k][0].split(" (")[0] for k in ORDER])
    ax[0].set_xlabel(r"$\kappa$ (log scale)")
    ax[0].set_title(r"(A) No configuration reaches $\kappa<0.5$", loc="left")
    ax[0].text(0.32, 2.52, "deep compression\n$\\kappa<0.5$", fontsize=7, color=FAIL,
               ha="left", va="top", fontweight="bold")
    ax[0].text(0.55, 0.06, "$0/33$", fontsize=9, color=INK, ha="left", va="bottom",
               fontweight="bold")
    ax[0].legend(handles=[plt.Line2D([], [], marker="s", ls="none", ms=6,
                                     mfc=BAND_NULL, mec="none",
                                     label="UNAGI family-matched null"),
                          plt.Line2D([], [], marker="s", ls="none", ms=6,
                                     mfc=BAND_CROSS, mec="none",
                                     label="G-VAE cross-family band")],
                 loc="upper right", frameon=False, fontsize=6)

    # ------------------------------------------------- (B) how far from the threshold
    x, y, c = [], [], []
    for i, key in enumerate(ORDER):
        for k in [KC(r) for r in rows if r["tissue"] == key]:
            x.append(i + np.random.default_rng(7).uniform(-0.18, 0.18))
            y.append((1.0 - k) * 100.0)
            c.append(TISSUE[key][1])
    ax[1].scatter(x, y, c=c, s=34, alpha=0.88, edgecolor="white", linewidth=0.75, zorder=3)
    ax[1].axhline(0, lw=0.75, color=MUTED, zorder=1)
    ax[1].set_xticks([0, 1, 2])
    ax[1].set_xticklabels([TISSUE[k][0].split(" (")[0] for k in ORDER])
    ax[1].set_ylim(-16, 26)
    ax[1].set_xlim(-0.45, 2.55)
    ax[1].set_ylabel(r"Global $C$ (%)")
    ax[1].set_title(r"(B) Deepest configuration: $48.2\%$", loc="left")
    ax[1].annotate("deep compression  $C>50\\%$  (off scale)",
                   xy=(0.05, 25.2), xytext=(0.05, 18.5), xycoords="data",
                   fontsize=7, color=FAIL, ha="left", va="top",
                   arrowprops=dict(arrowstyle="->", lw=1, color=FAIL))
    ax[1].text(2.42, 12.6, "lung R9\n$48.2\\%$", fontsize=7, va="top", ha="right",
               color=MUTED)

    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    fig.savefig(OUT.replace(".pdf", ".png"), dpi=170, bbox_inches="tight")
    print("[saved]", OUT)

    ks_hemo = [KC(r) for r in rows if r["tissue"] == "hemo"]
    ks_neu = [KC(r) for r in rows if r["tissue"] == "neural"]
    ks_lung = [KC(r) for r in rows if r["tissue"] == "lung"]
    print("n =", len(rows), "| deep compression (kappa<0.5):", sum(1 for r in rows if KC(r) < LOCK))
    print("hemo   kappa range: %.3f -- %.3f" % (min(ks_hemo), max(ks_hemo)))
    print("neural kappa range: %.3f -- %.3f" % (min(ks_neu), max(ks_neu)))
    print("lung   kappa      : %.3f" % ks_lung[0])
    print("global kappa range: %.3f -- %.3f" % (min(KC(r) for r in rows), max(KC(r) for r in rows)))


if __name__ == "__main__":
    main()
