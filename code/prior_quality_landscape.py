"""
prior_quality_landscape.py
==========================
Link prior "quality / transferability" to the tissue-dependent Z-spectrum effect.

The Z-spectrum audit shows GRN prior injection compresses effective rank in
haematopoietic but not neural-crest data. Why? One hypothesis is prior size
(more edges -> stronger effect). Another is tissue match (correct TFs / gene
space -> stronger effect). This script tests both on the same per-run data.

Inputs
------
- kappa_trajectory.csv (workspace) : per-iteration Z-spectrum metrics
- known prior edge counts from PRIOR_CONFIGS.md / prior_specificity_controls

Outputs
-------
- ../data/prior_quality_landscape.json
- ../figures/fig_prior_quality_landscape.pdf / .png
"""
import os, sys, json
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import figure_palette as fp
TRAJ = "F:/UNAGI_Project/paper1/kc_trajectory/kappa_trajectory.csv"
if not os.path.exists(TRAJ):
    TRAJ = "F:/UNAGI_Project/outputs/kc_trajectory/kappa_trajectory.csv"
OUT_DATA = os.path.join(HERE, "..", "data", "prior_quality_landscape.json")
OUT_PDF = os.path.join(HERE, "..", "figures", "fig_prior_quality_landscape.pdf")
OUT_PNG = os.path.join(HERE, "..", "figures", "fig_prior_quality_landscape.png")

# Edge counts recovered in PRIOR_CONFIGS.md (hema) and prior_specificity_controls (neural)
EDGE_COUNT = {
    "R11": 21, "R11_seed42": 21, "R11_KLF1_TAL1_seed123": 21, "R11_KLF1_TAL1_seed456": 21,
    "R12": 55, "R12_seed42": 55, "R12_seed123": 55, "R12_seed456": 55,
    "R13": 325, "R13_seed42": 325, "R13_seed123": 325, "R13_seed456": 325,
    "R84": 84, "R84_seed42": 84, "R84_seed123": 84,
    "R107": 107, "R107_seed42": 107,
    "R111": 111, "R111_seed42": 111, "R111_seed123": 111, "R111_seed456": 111,
    "R165": 165, "R165_seed42": 165,
    "R15": 165, "R15_seed42": 165, "R15_seed123": 165, "R15_seed456": 165,
    "R1": 50, "R1_seed42": 50, "R1_seed123": 50, "R1_seed456": 50,
    "R9": 0,  # IPF disease prior is not a TF GRN prior
}

# Cross-tissue edge survival from prior_specificity_controls.py (R16 = neural prior in blood)
CROSS_SURVIVAL = {
    "R13_hema": dict(native=292, foreign=29),   # hema prior in hema vs cortex space
    "R1_cortex": dict(native=50, foreign=0),    # neural prior in cortex vs hema space (approx)
    "R16_blood": dict(native=29, foreign=292),  # wrong-tissue: cortex TFs injected into marrow
}


def main():
    df = pd.read_csv(TRAJ)
    # keep iter_4 only, non-vanilla priors
    df4 = df[df["iteration"] == 4].copy()
    # map config name to edge count
    def get_edges(prior):
        if pd.notna(prior) and prior in EDGE_COUNT:
            return EDGE_COUNT[prior]
        # try stripping seed suffix? keep explicit
        return np.nan
    df4["edges"] = df4["prior"].apply(get_edges)
    df4 = df4[df4["edges"].notna()].copy()
    df4["delta_eff_rank"] = df4["eff_rank_P"] - df4["eff_rank_V"]
    # standardise tissue labels
    df4["tissue_label"] = df4["tissue"].map({"hemo": "haematopoietic",
                                              "neural": "neural crest",
                                              "lung": "lung IPF"})

    # regression / correlation
    from scipy import stats
    hema = df4[df4["tissue"] == "hemo"]
    if len(hema) >= 3:
        rho, p = stats.spearmanr(np.log10(hema["edges"] + 1), hema["delta_eff_rank"])
    else:
        rho, p = np.nan, np.nan

    records = df4[["tissue", "prior", "vanilla", "label", "edges",
                   "delta_eff_rank", "C"]].to_dict("records")
    out = dict(
        n=len(records),
        spearman_edges_vs_delta_eff_rank=dict(rho=float(rho), p=float(p)),
        cross_tissue_survival=CROSS_SURVIVAL,
        records=records,
    )
    json.dump(out, open(OUT_DATA, "w"), indent=2)
    print(f"wrote {OUT_DATA}; {len(records)} configs; Spearman rho={rho:.3f} p={p:.3g}")
    make_figure(df4, rho, p)


def make_figure(df, rho, p):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fp.apply_paper_style()  # unified paper style (single source of truth)

    palette = {"haematopoietic": fp.C_HEMA, "neural crest": fp.C_NEURAL, "lung IPF": fp.C_LUNG}
    fig, ax = plt.subplots(figsize=(6.7, 3.2))

    for lab, grp in df.groupby("tissue_label"):
        ax.scatter(np.log10(grp["edges"] + 1), grp["delta_eff_rank"],
                   c=palette.get(lab, fp.GRAY), label=lab, s=35, alpha=0.85, edgecolors="k", linewidth=0.3)
        # annotate a few key points
        for _, r in grp.iterrows():
            if any(x in str(r["prior"]) for x in ["R13", "R1_seed", "R11", "R12", "R9"]):
                offset = (3, 3) if r["delta_eff_rank"] < 0 else (3, -8)
                ax.annotate(str(r["prior"]).replace("_seed", "_s"),
                            (np.log10(r["edges"] + 1), r["delta_eff_rank"]),
                            fontsize=5.5, xytext=offset, textcoords="offset points")

    ax.axhline(0, color="#888888", linestyle="--", linewidth=0.6)
    ax.set_xlabel("log10(prior edges + 1)")
    ax.set_ylabel("eff_rank(prior) - eff_rank(vanilla) at iter_4\n(<0 = compression)")
    ax.set_title("Prior size and tissue match predict Z-spectrum compression")
    ax.legend(loc="upper right", frameon=False, fontsize=7)

    # annotate Spearman
    ax.text(0.02, 0.02, f"Spearman(hema) rho={rho:.2f}, p={p:.2g}",
            transform=ax.transAxes, fontsize=6.5, va="bottom")
    # annotate cross-tissue survival
    note = ("cross-tissue survival: R13 hema prior 292 edges in hema space, "
            "29 in cortex space; R1 neural prior 50 edges in cortex space")
    ax.text(0.02, 0.98, note, transform=ax.transAxes, fontsize=5.8, va="top")

    fig.tight_layout()
    fig.savefig(OUT_PDF, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    print(f"wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
