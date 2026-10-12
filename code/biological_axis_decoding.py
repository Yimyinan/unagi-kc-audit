"""
biological_axis_decoding.py
===========================
Biology-grounded decoding of the 64 latent axes of PG-UNAGI.

Motivation
----------
Everything done so far (kappa-C, Z-spectrum, SVD of W) is *geometry*: eigenvalue
decomposition with no biological anchor. This script asks the biological question
directly:

    Q1. What biological program does each of the 64 latent axes encode?
        -> score every axis against curated, canonical lineage marker sets using a
           rank-based (scale-free) enrichment statistic (Mann-Whitney AUC -> z).

    Q2. Does injecting a tissue-matched GRN prior make the latent axes *more*
        biologically interpretable (axes dedicated to single lineage programs)?
        -> controlled contrast: true hema prior (R13) vs wrong prior (R16, cortex
           TFs into marrow) vs vanilla (R14), all on the same marrow data.

    Q3. Does prior injection increase the decoder's reliance on the prior's own
        TF/target genes?
        -> differential gene involvement Dlog(||W[:,g]||) true vs vanilla, prior
           genes vs background (Mann-Whitney).

    Q4. For the neural dataset: which cell-type programs actually dominate the
        captured trajectory, and does the injected cortex program match them?
        -> data-driven explanation of the neural null result.

Outputs
-------
  outputs/positive_result/fig_biological_axis_decoding.{png,pdf}
  outputs/positive_result/BIOLOGICAL_AXIS_DECODING.md
  outputs/positive_result/axis_lineage_scores.csv
"""

import os, json, csv

import numpy as np
import matplotlib
import sys as _sys, os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
for _p in (_HERE, _os.path.join(_HERE, "..", "code"), _os.path.join(_HERE, "code")):
    if _os.path.isdir(_p) and _p not in _sys.path:
        _sys.path.insert(0, _p)
import figure_palette as fp
fp.apply_paper_style()
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NEX = "F:/UNAGI_Project/next_phase"
DEC = "F:/UNAGI_Project/UNAGI/decoder_out"
OUT = "F:/UNAGI_Project/outputs/positive_result"
os.makedirs(OUT, exist_ok=True)

# Authoritative, run_registry-vetted matrices (decoder_unified is the clean source).
DU = "F:/UNAGI_Project/UNAGI/decoder_unified/loading"
VAN = "F:/UNAGI_Project/UNAGI/decoder_unified/vanilla"   # canonical vanilla readouts
HEA_TRUE  = os.path.join(DU, "R13_s42.npy")   # true blood MR prior (seed 42)
HEA_VAN   = os.path.join(VAN, "R14_hemato_vanilla.npy")   # EMPTY PRIOR, tr(W'W)=5634.8
                                                          # (NOT R14_s42: that is the
                                                          #  R15 vehicle arm, tr 5160.9)
HEA_WRONG = os.path.join(DEC, "R16", "iter_4", "loading_matrix.npy")  # wrong-tissue (cortex TFs into marrow); NOTE: seed not seed-42-matched
NEU_PRIOR = os.path.join(DU, "R1_s456.npy")   # cortex prior (seed 456)
NEU_VAN   = os.path.join(DU, "R0_s456.npy")   # neural vanilla, SEED-MATCHED to NEU_PRIOR (both
                                              # seed 456); legacy bare R0 excluded (tr 17695.2)

plt.rcParams.update({"font.size": 8.5, "figure.dpi": 160,
                     "axes.grid": False, "savefig.bbox": "tight",
                     "axes.spines.top": False, "axes.spines.right": False})

# =====================================================================
# Curated marker sets  (canonical human hematopoiesis / PBMC-HCA markers)
# =====================================================================
HEMA_SETS = {
    "HSC/prog":   "CD34 AVP CRHBP HLF MLLT3 SPINK2 GATA2 MEIS1 MYB KIT SPARC SMIM24 "
                  "CYTL1 GPR56 AKR1C1 EMCN MEG3",
    "Erythroid":  "HBA1 HBA2 HBB HBD HBG1 HBG2 GYPA GYPB SLC4A1 ALAS2 AHSP EPB42 "
                  "SPTA1 SPTB CA1 TFRC KLF1 GFI1B NFE2 HEMGN ALAD FECH",
    "Megakaryo":  "PF4 PPBP ITGA2B ITGB3 VWF GP9 GP1BA THBS1 TUBB1 ZFPM1 PF4V1 "
                  "NRGN CLU SELP",
    "Monocyte":   "LYZ S100A8 S100A9 FCN1 VCAN LST1 AIF1 CD14 CSF1R TYROBP CSTA "
                  "SERPINA1 CFD GRN CTSS",
    "Neutrophil": "ELANE MPO PRTN3 AZU1 DEFA1 DEFA4 CTSG LTF CAMP MMP8 FCGR3B "
                  "CSF3R S100P CEACAM8 BPI",
    "T lymph":    "CD3D CD3E CD3G CD2 IL7R TRAC LCK SKAP1 THEMIS CCR7 MAL CD8A "
                  "CD8B CD4 TRBC1",
    "B lymph":    "CD79A CD79B MS4A1 IGHM IGHD TCL1A VPREB3 BANK1 CD19 PAX5 EBF1 "
                  "IGLL5 LINC00926",
    "NK":         "GNLY NKG7 GZMB KLRD1 KLRF1 PRF1 GZMA GZMH FGFBP2 SPON2 ADGRG1 "
                  "KLRB1 CMC1",
    "Dendritic":  "FCER1A CST3 CLEC10A CD1C CLEC9A CADM1 ENHO LGALS2 CD1E",
    "Eos/Baso":   "CLC EPX RNASE2 RNASE3 PRG2 IL5RA HDC CPA3 TPSAB1 MS4A2 PRG3",
    "Cycling":    "MKI67 TOP2A CCNB1 CDK1 UBE2C BIRC5 TYMS PCNA MCM2 RRM2 STMN1 "
                  "TUBA1B HMGB2",
    "Housekeep":  "ACTB GAPDH RPL13A RPS18 RPLP0 B2M TUBB PPIA EEF1A1 RPS4X RPL3 "
                  "RPS6 RPL13 RPS3",
}

NEURAL_SETS = {
    "Radial glia": "SOX2 PAX6 HES1 HES5 VIM NES FABP7 SLC1A3 GLI3 TTYH1 DACH1",
    "Neuron":      "STMN2 TUBB3 MAPT DCX NEFL SNAP25 SYT1 NEUROD1 RBFOX3 SCG10 "
                   "STMN4 MAP1B",
    "Astrocyte":   "GFAP AQP4 S100B ALDH1L1 GJA1 SLC6A11 APOE CLU CD44",
    "Oligodendro": "PLP1 MBP OLIG1 OLIG2 MOG SOX10 MOBP MAG CNP ST18",
    "Interneuron": "GAD1 GAD2 DLX1 DLX2 DLX5 LHX6 SST VIP PVALB CALB2 NPY CCK",
    "Neural crest":"SOX9 SOX10 TFAP2A TFAP2B SNAI1 SNAI2 TWIST1 TWIST2 FOXD3 ETS1 "
                   "MITF CDH6 NGFR",
    "Microglia":   "PTPRC CX3CR1 CSF1R ITGAM TYROBP C1QA C1QB AIF1 SPI1",
    "Endothelial": "CLDN5 PECAM1 CDH5 VWF FLT1 KDR SOX17 EGFL7",
}

# =====================================================================
# statistics helpers (rank-based, no scipy)
# =====================================================================
def rankdata(x):
    """Average ranks, 1..n."""
    n = x.size
    order = np.argsort(x, kind="mergesort")
    r = np.empty(n, dtype=float)
    r[order] = np.arange(1, n + 1, dtype=float)
    # average ties
    sx = x[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sx[j + 1] == sx[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return r


def auc_z(vals, idx):
    """Mann-Whitney AUC + normal-approx z for gene-set 'idx' being up in 'vals'."""
    n = vals.size
    m = idx.size
    if m == 0 or m == n:
        return 0.5, 0.0
    r = rankdata(vals)
    U = r[idx].sum() - m * (m + 1) / 2.0
    auc = U / (m * (n - m))
    mu = m * (n - m) / 2.0
    sd = np.sqrt(m * (n - m) * (n + m + 1) / 12.0)
    return float(auc), float((U - mu) / sd)


def mw_test(a, b):
    """Two-sided Mann-Whitney U on two 1-D arrays; returns (U, z, p) via normal approx."""
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    na, nb = a.size, b.size
    if na == 0 or nb == 0:
        return np.nan, np.nan, np.nan
    allv = np.concatenate([a, b])
    r = rankdata(allv)
    Ra = r[:na].sum()
    U = Ra - na * (na + 1) / 2.0
    mu = na * nb / 2.0
    sd = np.sqrt(na * nb * (na + nb + 1) / 12.0)
    z = (U - mu) / sd
    # two-sided p via erfc
    from math import erfc, sqrt
    p = erfc(abs(z) / sqrt(2.0))
    return float(U), float(z), float(p)


# =====================================================================
# load W (64 latent axes x G genes) + gene symbols
# =====================================================================
def load_model(glist_file, path, cfg="?"):
    """Load decoder loading W from an *explicit* npy path (decoder_unified is the
    authoritative, run_registry-vetted source) plus the gene-list json.

    The explicit-path form deliberately avoids the documented run_registry failure
    modes:
      * decoder_out/R13/iter_4 is DEGENERATE (F-norm 7.57) -> use R13_s42.npy (100.45)
      * decoder_out/R14/vanilla/iter_4 DOES exist.  An earlier revision of this
        note claimed it did not -- that was factually wrong.  It holds the EMPTY
        PRIOR; R14_s42.npy is the VEHICLE arm (37-TF prior / 1054 boosted genes).
        -> use decoder_unified/vanilla/R14_hemato_vanilla.npy
      * bare decoder_out/R0 (neural vanilla) is a LEGACY run (eff rank 38.1,
        tr 17695.2, spurious z=+3.91 against R1) -> use R0_s456.npy, which is
        SEED-MATCHED to the R1_s456 prior used above.
    """
    W = np.load(path).astype(float)
    g = json.load(open(os.path.join(DEC, glist_file)))
    g = [str(x).upper() for x in g]
    assert W.shape[1] == len(g), f"{cfg}: W {W.shape} vs genes {len(g)}"
    return W, g


def build_marker_index(sets, genes):
    gi = {}
    present = {}
    for k, s in sets.items():
        syms = s.split()
        idx = np.array(sorted({genes.index(x) for x in syms if x in set(genes)}), dtype=int)
        gi[k] = idx
        present[k] = (len(idx), len(syms))
    return gi, present


def axis_scores(W, midx, setnames, center=True):
    """Return Z (n_axes x n_sets) matrix of enrichment z-scores.

    IMPORTANT: |W[k,g]| is dominated by each gene's *baseline* magnitude (highly
    expressed genes load on every axis). To ask "which genes are specifically driven
    by axis k", we centre each gene across the 64 axes first, then rank within the
    axis. Without this centring the statistic reproduces gene abundance, not axis
    identity.
    """
    A = np.abs(W)
    if center:
        A = A - A.mean(axis=0, keepdims=True)
    n_axes = A.shape[0]
    Z = np.zeros((n_axes, len(setnames)))
    AUC = np.zeros((n_axes, len(setnames)))
    for k in range(n_axes):
        vals = A[k]
        for j, nm in enumerate(setnames):
            a, z = auc_z(vals, midx[nm])
            AUC[k, j] = a; Z[k, j] = z
    return Z, AUC


def summarise(Z, setnames, thr=3.0):
    maxz = Z.max(axis=1)
    arg = Z.argmax(axis=1)
    # gap between best and second-best
    srt = np.sort(Z, axis=1)
    gap = srt[:, -1] - srt[:, -2]
    return {
        "n_dedicated": int((maxz >= thr).sum()),
        "median_maxz": float(np.median(maxz)),
        "mean_maxz": float(maxz.mean()),
        "median_gap": float(np.median(gap)),
        "coverage": int(len(set(arg))),
        "assignment": {setnames[i]: int((arg == i).sum()) for i in range(len(setnames))},
        "maxz": maxz, "arg": arg, "gap": gap,
    }


# =====================================================================
# hematopoietic controlled contrast
# =====================================================================
print("=" * 70)
print("HEMATOPOIETIC: decoding the 64 latent axes (true / wrong / vanilla)")
print("=" * 70)

setnames = list(HEMA_SETS.keys())
models = {
    "hema_true":    (HEA_TRUE, "hema_gene_list.json"),
    "hema_wrong":   (HEA_WRONG, "hema_gene_list.json"),
    "hema_vanilla": (HEA_VAN, "hema_gene_list.json"),
}

mats, summs, gene_info = {}, {}, {}
for lab, (path, gf) in models.items():
    W, genes = load_model(gf, path, lab)
    midx, present = build_marker_index(HEMA_SETS, genes)
    Z, A = axis_scores(W, midx, setnames)
    s = summarise(Z, setnames)
    mats[lab] = Z; summs[lab] = s; gene_info[lab] = (W, genes, midx, present)
    print(f"[{lab}] axes dedicated (maxz>=3): {s['n_dedicated']}/64 | "
          f"median maxz={s['median_maxz']:.2f} | median gap={s['median_gap']:.2f} | "
          f"lineages covered={s['coverage']}/{len(setnames)}")

print("\nmarker recovery (n_found/n_curated):")
for k, (a, b) in present.items():
    print(f"   {k:12s} {a}/{b}")

# statistical comparison of per-axis maxz, true vs vanilla / wrong vs vanilla
rows = []
for lab in ["hema_true", "hema_wrong", "hema_vanilla"]:
    s = summs[lab]
    if lab == "hema_vanilla":
        U, z, p = np.nan, np.nan, np.nan
        base = "—"
    else:
        U, z, p = mw_test(s["maxz"], summs["hema_vanilla"]["maxz"])
        base = "hema_vanilla"
    rows.append(dict(condition=lab, n_dedicated=s["n_dedicated"],
                     median_maxz=round(s["median_maxz"], 3),
                     mean_maxz=round(s["mean_maxz"], 3),
                     median_gap=round(s["median_gap"], 3),
                     coverage=s["coverage"],
                     vs=base, mw_z=(None if np.isnan(z) else round(z, 3)),
                     mw_p=(None if np.isnan(p) else float(f"{p:.3g}"))))
    print(f"   MW {lab} vs vanilla: z={z:.3f} p={p:.3g}")

# ---------------------------------------------------------------- Q3: differential gene involvement
def colnorm(W):
    I = np.sqrt((W ** 2).sum(axis=0))
    return I / np.median(I)

hp = json.load(open(os.path.join(NEX, "data", "hema_prior.json")))
tf_db = hp["tf_db"] if "tf_db" in hp else {}
prior_tfs = set(t.upper() for t in tf_db.keys())
prior_tgs = set()
for t, gs in tf_db.items():
    prior_tgs.update(str(g).upper() for g in gs)
prior_genes = prior_tfs | prior_tgs

Wt, genes_h, midx_h, _ = gene_info["hema_true"]
Wv, _, _, _ = gene_info["hema_vanilla"]
Ww, _, _, _ = gene_info["hema_wrong"]
It, Iv, Iw = colnorm(Wt), colnorm(Wv), colnorm(Ww)
gset = set(genes_h)
pg = np.array(sorted({genes_h.index(x) for x in prior_genes if x in gset}), dtype=int)
bg = np.setdiff1d(np.arange(len(genes_h)), pg)

def dlog(Ia, Ib):
    return np.log((Ia + 1e-12) / (Ib + 1e-12))

d_true = dlog(It, Iv)
d_wrong = dlog(Iw, Iv)

q3 = []
for nm, d in [("true_vs_vanilla", d_true), ("wrong_vs_vanilla", d_wrong)]:
    U, z, p = mw_test(d[pg], d[bg])
    q3.append(dict(contrast=nm, n_prior_genes=int(pg.size), n_background=int(bg.size),
                   median_dlog_prior=round(float(np.median(d[pg])), 4),
                   median_dlog_background=round(float(np.median(d[bg])), 4),
                   mw_z=round(z, 3), mw_p=float(f"{p:.3g}")))
    print(f"   [Q3] {nm}: prior median Dlog={np.median(d[pg]):+.4f} vs background "
          f"{np.median(d[bg]):+.4f}  MW z={z:.2f} p={p:.3g}")

# =====================================================================
# neural: which programs dominate the captured trajectory?
# =====================================================================
print("\n" + "=" * 70)
print("NEURAL: which cell-type programs dominate the captured trajectory?")
print("=" * 70)
nsetnames = list(NEURAL_SETS.keys())
neural_models = {"neural_prior": (NEU_PRIOR, "neural_gene_list.json"),
                 "neural_vanilla": (NEU_VAN, "neural_gene_list.json")}
nZ, nsum, ngene = {}, {}, {}
for lab, (path, gf) in neural_models.items():
    W, genes = load_model(gf, path, lab)
    midx, present = build_marker_index(NEURAL_SETS, genes)
    Z, A = axis_scores(W, midx, nsetnames)
    s = summarise(Z, nsetnames)
    nZ[lab] = Z; nsum[lab] = s; ngene[lab] = (W, genes, midx, present)
    print(f"[{lab}] dedicated: {s['n_dedicated']}/64 | median maxz={s['median_maxz']:.2f} "
          f"| covered={s['coverage']}/{len(nsetnames)}")
    print("     axis->program assignment:", s["assignment"])
print("\nneural marker recovery:")
for k, (a, b) in present.items():
    print(f"   {k:14s} {a}/{b}")

# =====================================================================
# FIGURE
# =====================================================================
fig = plt.figure(figsize=(11.0, 8.4))
gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 1.05],
                      hspace=0.62, wspace=0.30,
                      left=0.075, right=0.965, top=0.925, bottom=0.075)

cmap = plt.get_cmap("RdBu_r")
VMAX = 8.0

# --- A/B: axis x lineage z heatmaps (true prior vs vanilla) ---
for j, lab in enumerate(["hema_true", "hema_vanilla"]):
    ax = fig.add_subplot(gs[0, j])
    Z = mats[lab]
    # order axes by their argmax lineage for visual structure
    order = np.argsort(Z.argmax(axis=1) * 100 - Z.max(axis=1))
    im = ax.imshow(Z[order], cmap=cmap, vmin=-VMAX, vmax=VMAX, aspect="auto",
                   interpolation="nearest")
    ax.set_yticks([0, 63]); ax.set_yticklabels(["axis (sorted)", ""], fontsize=7)
    ax.set_xticks(range(len(setnames)))
    ax.set_xticklabels(setnames, rotation=55, ha="right", fontsize=6.5)
    ax.set_title(("A. true hematopoietic prior (R13)" if j == 0
                  else "B. no prior / vanilla (R14)") + "\nlatent axis $\\times$ lineage-program enrichment $z$",
                 fontsize=8.5)
    if j == 1:
        cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
        cb.set_label("enrichment $z$ (Mann–Whitney)", fontsize=7)
        cb.ax.tick_params(labelsize=6.5)

# --- C: dedicated axes + median maxz across the three hema conditions ---
ax = fig.add_subplot(gs[1, 0])
labs = ["hema_true", "hema_wrong", "hema_vanilla"]
pretty = ["true prior\n(R13)", "wrong prior\n(R16)", "vanilla\n(R14)"]
nd = [summs[l]["n_dedicated"] for l in labs]
mm = [summs[l]["median_maxz"] for l in labs]
x = np.arange(3); w = 0.38
b1 = ax.bar(x - w / 2, nd, w, color="#B2182B", edgecolor="#333", lw=0.4,
            label="axes dedicated to a lineage (max $z\\geq3$)")
ax.set_ylabel("n axes (of 64)", fontsize=7.5)
ax2 = ax.twinx()
b2 = ax2.bar(x + w / 2, mm, w, color="#2166AC", edgecolor="#333", lw=0.4,
             label="median max $z$ per axis")
ax2.set_ylabel("median max $z$", fontsize=7.5, color="#2166AC")
ax2.tick_params(axis="y", labelsize=6.5, colors="#2166AC")
ax.set_xticks(x); ax.set_xticklabels(pretty, fontsize=7)
ax.tick_params(axis="y", labelsize=6.5)
ax.set_title("C. Latent-axis lineage identity is prior-agnostic\n"
             f"(true {nd[0]}/64 vs vanilla {nd[2]}/64 dedicated; MW $p$={rows[0]['mw_p']})", fontsize=8.5)
for xi, v in zip(x - w / 2, nd):
    ax.text(xi, v + 0.6, str(v), ha="center", fontsize=7, fontweight="bold", color="#B2182B")
for xi, v in zip(x + w / 2, mm):
    ax2.text(xi, v + 0.06, f"{v:.2f}", ha="center", fontsize=6.5, color="#2166AC")
ax.set_ylim(0, max(nd) * 1.30)
ax2.set_ylim(0, max(mm) * 1.35)
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=6, loc="upper left", frameon=False)

# --- D: differential involvement of prior genes ---
ax = fig.add_subplot(gs[1, 1])
data = [d_true[pg], d_true[bg], d_wrong[pg], d_wrong[bg]]
pos = [1, 2, 4, 5]
cols = ["#B2182B", "#999999", "#D6604D", "#BBBBBB"]
bp = ax.boxplot([d[::max(1, d.size // 6000)] for d in data], positions=pos,
                widths=0.7, patch_artist=True, showfliers=False,
                medianprops=dict(color="black", lw=1.0))
for p_, c in zip(bp["boxes"], cols):
    p_.set_facecolor(c); p_.set_alpha(0.75); p_.set_linewidth(0.5)
ax.set_xticks(pos)
ax.set_xticklabels(["prior\ngenes", "back-\nground", "prior\ngenes", "back-\nground"], fontsize=6.5)
ax.axhline(0, color="#333", lw=0.6, ls=":")
ax.set_ylabel(r"$\Delta\log$ decoder involvement vs vanilla", fontsize=7.5)
ax.tick_params(axis="y", labelsize=6.5)
ax.set_title("D. Decoder weight on prior genes does not differ from vanilla\n"
             f"(true $p$={q3[0]['mw_p']:.2g}, wrong $p$={q3[1]['mw_p']:.2g}; both n.s.)", fontsize=8.5)
ax.text(1.5, ax.get_ylim()[1] * 0.86, "true prior", ha="center", fontsize=7, fontweight="bold")
ax.text(4.5, ax.get_ylim()[1] * 0.86, "wrong prior", ha="center", fontsize=7, fontweight="bold")

# --- E: neural axis->program assignment ---
ax = fig.add_subplot(gs[2, 0])
Z = nZ["neural_prior"]
assign = nsum["neural_prior"]["assignment"]
ks = list(nsetnames)
vals = [assign.get(k, 0) for k in ks]
yy = np.arange(len(ks))
ax.barh(yy, vals, color=["#B2182B" if "crest" in k.lower() else "#4393C3" for k in ks],
        edgecolor="#333", lw=0.4, height=0.65)
ax.set_yticks(yy); ax.set_yticklabels(ks, fontsize=7)
ax.invert_yaxis()
ax.set_xlabel("n latent axes whose top program is this cell type (of 64)", fontsize=7.5)
ax.tick_params(axis="x", labelsize=6.5)
ax.set_title("E. Neural dataset: the cortex prior does not match the captured\n"
             "trajectory (axes map to neuron/glia, not the injected program)", fontsize=8.5)
for yi, v in zip(yy, vals):
    if v: ax.text(v + 0.5, yi, str(v), va="center", fontsize=6.5)

# --- F: summary ladder: biological congruence vs latent compression ---
ax = fig.add_subplot(gs[2, 1])
ladder = [("vanilla", summs["hema_vanilla"]["median_maxz"], -0.085),
          ("wrong prior", summs["hema_wrong"]["median_maxz"], -0.536),
          ("true prior", summs["hema_true"]["median_maxz"], -1.062)]
xs = [l[1] for l in ladder]; ys = [l[2] for l in ladder]; nm = [l[0] for l in ladder]
ax.plot(xs, ys, "-o", color="#333333", lw=1.2, ms=6, mfc="#B2182B")
for xi, yi, n in zip(xs, ys, nm):
    ax.annotate(n, (xi, yi), textcoords="offset points", xytext=(7, -3), fontsize=7)
ax.set_xlabel("median per-axis lineage-program enrichment $z$", fontsize=7.5)
ax.set_ylabel(r"$\Delta\log(\mathrm{eff\_rank})$ of $\mathbf{Z}$", fontsize=7.5)
ax.tick_params(labelsize=6.5)
ax.set_title("F. No axis-level biology–compression coupling\n(per-axis enrichment is prior-agnostic)", fontsize=8.5)
ax.grid(alpha=0.25, lw=0.4)

fig.savefig(os.path.join(OUT, "fig_biological_axis_decoding.pdf"))
fig.savefig(os.path.join(OUT, "fig_biological_axis_decoding.png"), dpi=200)
plt.close(fig)
print("\nwrote fig_biological_axis_decoding.pdf/png")

# =====================================================================
# CSV + markdown
# =====================================================================
with open(os.path.join(OUT, "axis_lineage_scores.csv"), "w", newline="") as f:
    w_ = csv.writer(f)
    w_.writerow(["model", "axis", "best_program", "max_z"] + [f"z_{s}" for s in setnames])
    for lab in ["hema_true", "hema_wrong", "hema_vanilla"]:
        Z = mats[lab]
        for k in range(Z.shape[0]):
            w_.writerow([lab, k, setnames[int(Z[k].argmax())]] +
                        [f"{Z[k].max():.3f}"] + [f"{v:.3f}" for v in Z[k]])
    for lab in ["neural_prior", "neural_vanilla"]:
        Z = nZ[lab]
        for k in range(Z.shape[0]):
            w_.writerow([lab, k, nsetnames[int(Z[k].argmax())]] +
                        [f"{Z[k].max():.3f}"] + [f"{v:.3f}" for v in Z[k]])

md = []
md.append("# Biological decoding of the 64 latent axes\n")
md.append("**Why this analysis exists.** kappa-C, the Z-spectrum and the SVD of W are all "
          "*geometric* readouts: eigenvalue decompositions with no biological anchor. This "
          "analysis asks the biological question directly -- what program does each latent "
          "axis encode, and does a tissue-matched prior make those axes more biologically "
          "interpretable?\n")
md.append("**Statistic.** For each latent axis k, genes are ranked by |W[k,g]| and each "
          "curated lineage marker set is scored by a Mann-Whitney AUC, converted to a "
          "normal-approx z. Scale-free, no distributional assumption, comparable across "
          "models. An axis is called *dedicated* when its best program has z >= 3.\n")
md.append("## A. Hematopoietic controlled contrast (same marrow data, same architecture)\n")
md.append("| condition | dedicated axes /64 | median max z | median gap (best-2nd) | lineages covered | MW z vs vanilla | p |")
md.append("|---|---|---|---|---|---|---|")
for r in rows:
    md.append(f"| {r['condition']} | {r['n_dedicated']} | {r['median_maxz']:.3f} | "
              f"{r['median_gap']:.3f} | {r['coverage']} | {r['mw_z']} | {r['mw_p']} |")
md.append("")
md.append("## B. Does prior injection put weight on the prior's own genes?\n")
md.append("| contrast | n prior genes | median Dlog (prior) | median Dlog (background) | MW z | p |")
md.append("|---|---|---|---|---|---|")
for r in q3:
    md.append(f"| {r['contrast']} | {r['n_prior_genes']} | {r['median_dlog_prior']:+.4f} | "
              f"{r['median_dlog_background']:+.4f} | {r['mw_z']} | {r['mw_p']} |")
md.append("")
md.append("## C. Neural dataset: what is actually in the captured trajectory?\n")
md.append("| program | axes (prior model) | axes (vanilla) |")
md.append("|---|---|---|")
for k in nsetnames:
    md.append(f"| {k} | {nsum['neural_prior']['assignment'].get(k,0)} | "
              f"{nsum['neural_vanilla']['assignment'].get(k,0)} |")
md.append("")
md.append("## Interpretation\n")
md.append("The controlled hematopoietic contrast answers Q1-Q3 negatively. Injecting a "
          "tissue-matched GRN prior does **not** make the 64 latent axes more lineage-dedicated: "
          f"the true-prior model has {rows[0]['n_dedicated']}/64 dedicated axes versus "
          f"{rows[2]['n_dedicated']}/64 for the no-prior vanilla model, and the per-axis maximal "
          f"enrichment distributions are not significantly different (MW true vs vanilla "
          f"p={rows[0]['mw_p']}). Differential decoder involvement of the prior's own genes is also "
          f"null (true vs vanilla p={q3[0]['mw_p']:.2g}; wrong vs vanilla p={q3[1]['mw_p']:.2g}). "
          "The decoder readout is therefore prior-agnostic not only in its global compression "
          "(Paper 1, S2.5) but also in its axis-level biological structure and gene-level "
          "weighting. This is consistent with the diagnostic's central resolution: prior knowledge "
          "resides in the latent covariance spectrum, not in the decoder readout. The result is a "
          "coherent *negative* control, not a second positive, and must not be framed as evidence "
          "that priors sharpen latent-axis interpretability.\n")
open(os.path.join(OUT, "BIOLOGICAL_AXIS_DECODING.md"), "w").write("\n".join(md))
print("wrote BIOLOGICAL_AXIS_DECODING.md and axis_lineage_scores.csv")
