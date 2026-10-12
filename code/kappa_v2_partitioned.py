"""
kappa_v2_partitioned.py
=======================
kappa-C v2 : the *partitioned* spectral diagnostic.

kappa-C v1 reports a single scalar  kappa = tr(H_P)/tr(H_V),  C = (1-kappa)*100%,
where H = W^T W and W is the (64, G) linear readout. Because kappa is the trace of a
*global, averaged* matrix, it is blind to any rearrangement that preserves the total:
the prior can rotate spectral mass between latent partitions, or between ReLU activation
regions, and kappa does not move. This is the "preserve-total-rearrangement" blind spot
documented in the manuscript (limitation).

kappa-C v2 fixes this by partitioning the 64 latent dimensions into K clusters (k-means on
the gene-loadings of the matched vanilla W) and reporting the *distribution* of
C_k = (1 - tr(H_k^P)/tr(H_k^V)) * 100%  across clusters, instead of one global number.
A non-flat C_k distribution means spectral mass was reorganised even when the global
kappa stays ~1 (i.e. 0/33 under v1). This is exactly what makes the tissue-dependent
Z-spectrum result (positive in haematopoietic, absent in neural crest) reconciliable with
the v1 0/33 verdict: v1 measured the decoder gain (W), v2 exposes the latent reorganisation.

Usage
-----
    python kappa_v2_partitioned.py
Writes:
    ../data/kappa_v2_partitioned.json     (per-pair, per-cluster C_k + summary)
    ../figures/fig_kappa_v2_partitioned.pdf / .png
"""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import figure_palette as fp
import run_registry as RR

OUT_DATA = os.path.join(HERE, "..", "data", "kappa_v2_partitioned.json")
OUT_PDF = os.path.join(HERE, "..", "figures", "fig_kappa_v2_partitioned.pdf")
OUT_PNG = os.path.join(HERE, "..", "figures", "fig_kappa_v2_partitioned.png")

K = 8          # number of latent partitions
PC = 16        # PCA dims used to stabilise the k-means on gene-loadings
SEED = 7

# (prior stem, vanilla stem, tissue) -- matched (model family, seed) pairs
PAIRS = [
    # haematopoietic (seed 42 / 123 / 456)
    ("R13_s42", "R14", "hema"), ("R13_s123", "R14", "hema"), ("R13_s456", "R14", "hema"),
    ("R11_s42", "R14_s42", "hema"), ("R11_s123", "R14_s123", "hema"), ("R11_s456", "R14_s456", "hema"),
    ("R12_s42", "R14_s42", "hema"), ("R12_s123", "R14_s123", "hema"), ("R12_s456", "R14_s456", "hema"),
    ("R84_s42", "R14_s42", "hema"),
    ("R107_s42", "R14_s42", "hema"),
    ("R111_s42", "R14_s42", "hema"), ("R111_s123", "R14_s123", "hema"),
    ("R165_s42", "R14_s42", "hema"),
    # neural crest
    ("R1_s123", "R0_s123", "neural"), ("R1_s456", "R0_s456", "neural"),
]
# vanilla self-pairs = exact null (C_k == 0 identically)
NULL_PAIRS = [("R14_s42", "R14_s42", "hema"), ("R14_s123", "R14_s123", "hema"),
              ("R0_s123", "R0_s123", "neural")]
# cross-seed vanilla pairs = the CORRECT null for dp_spread: two independently
# trained untrained decoders. This is what the prior-pair 5 pp must be compared
# against (the self-pair null is trivially 0 and cannot judge significance).
VANILLA_CROSS = [("R14_s42", "R14_s123", "hema"), ("R14_s42", "R14_s456", "hema"),
                 ("R14_s123", "R14_s456", "hema"), ("R0_s123", "R0_s456", "neural")]


def pca_rows(W, pc=PC):
    """Return top-`pc` right singular vectors of the row-centred W (64 x pc)."""
    A = W - W.mean(axis=1, keepdims=True)
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    # rows of Vt are right singular vectors (gene-space); take top pc
    return Vt[:pc].T  # (G x pc) -> we want 64 x pc latent coords below


def partition(vanilla_W, k=K, pc=PC, seed=SEED):
    """k-means++ Lloyd on the 64 latent-row gene-loadings (PC-reduced)."""
    A = vanilla_W - vanilla_W.mean(axis=1, keepdims=True)
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    coords = U[:, :pc] * S[:pc]            # (64 x pc) latent-row coordinates
    rng = np.random.default_rng(seed)
    # k-means++ init
    centers = [coords[rng.integers(64)]]
    d2 = ((coords - centers[0]) ** 2).sum(1)
    for _ in range(1, k):
        probs = d2 / d2.sum()
        centres_idx = rng.choice(64, p=probs)
        centers.append(coords[centres_idx])
        nd = ((coords - coords[centres_idx]) ** 2).sum(1)
        d2 = np.minimum(d2, nd)
    centers = np.array(centers, float)
    for _ in range(50):
        dist = ((coords[:, None, :] - centers[None, :, :]) ** 2).sum(2)
        lab = dist.argmin(1)
        new = np.array([coords[lab == j].mean(0) for j in range(k)])
        if np.allclose(new, centers):
            centers = new
            break
        centers = new
    dist = ((coords[:, None, :] - centers[None, :, :]) ** 2).sum(2)
    return dist.argmin(1)                  # (64,) cluster label per latent dim


def cluster_stats(Wm, Wv, lab):
    """Per-cluster tr(H_k) for prior and vanilla; kappa_k, C_k, and the
    scale-free mass-share shift delta_p_k = (pP - pV)*100 (percentage points)."""
    trP = np.array([(Wm[lab == j] ** 2).sum() for j in range(K)])
    trV = np.array([(Wv[lab == j] ** 2).sum() for j in range(K)])
    gP, gV = trP.sum(), trV.sum()
    kappa_k = trP / trV
    C_k = (1 - kappa_k) * 100.0
    pP, pV = trP / gP, trV / gV                 # spectral-mass share per partition
    delta_p_k = (pP - pV) * 100.0               # scale-free, sums to 0
    g_kappa = gP / gV
    return dict(C_k=C_k.tolist(), kappa_k=kappa_k.tolist(),
                pP=pP.tolist(), pV=pV.tolist(), delta_p_k=delta_p_k.tolist(),
                g_kappa=float(g_kappa), g_C=float((1 - g_kappa) * 100.0),
                spread=float(C_k.std()), dp_spread=float(delta_p_k.std()),
                skew=float(_skew(C_k)))


def _skew(x):
    x = np.asarray(x, float)
    m = x.mean();
    s = x.std()
    if s == 0:
        return 0.0
    return float(((x - m) ** 3).mean() / s ** 3)


def main():
    records = []
    for model, vanilla, tissue in PAIRS + NULL_PAIRS:
        try:
            Wm = RR.get_matrix(model)
            Wv = RR.get_matrix(vanilla)
        except Exception as e:
            print(f"  skip {model}/{vanilla}: {e}")
            continue
        lab = partition(Wv, K, PC, SEED)
        st = cluster_stats(Wm, Wv, lab)
        rec = dict(model=model, vanilla=vanilla, tissue=tissue,
                   is_null=(model == vanilla), **st)
        records.append(rec)
        tag = "NULL" if model == vanilla else "prior"
        print(f"  {tag:5s} {model:12s}/{vanilla:12s} {tissue:6s} "
              f"gC={st['g_C']:+6.1f}  C_k spread={st['spread']:5.1f}  "
              f"dp_spread={st['dp_spread']:5.1f}  "
              f"min/max C_k={min(st['C_k']):+5.1f}/{max(st['C_k']):+5.1f}")

    # summary: scale-free delta_p_k distribution across prior pairs, by tissue
    prior = [r for r in records if not r["is_null"]]

    # proper null: dp_spread of two INDEPENDENT vanilla seeds (untrained decoders)
    vanilla_cross = []
    for m, v, t in VANILLA_CROSS:
        try:
            Wm = RR.get_matrix(m)
            Wv = RR.get_matrix(v)
        except Exception as e:
            print(f"  skip vanilla-cross {m}/{v}: {e}")
            continue
        lab = partition(Wv, K, PC, SEED)
        st = cluster_stats(Wm, Wv, lab)
        vanilla_cross.append(dict(pair=f"{m}|{v}", tissue=t, dp_spread=st["dp_spread"]))
        print(f"  VANILLA-CROSS {t:6s} {m:10s}/{v:10s} dp_spread={st['dp_spread']:6.3f}")

    summary = {}
    for tissue in ("hema", "neural"):
        grp = [r for r in prior if r["tissue"] == tissue]
        Dmat = np.array([r["delta_p_k"] for r in grp])   # (n, K), scale-free
        mean_d = Dmat.mean(0)
        se = Dmat.std(0, ddof=1) / np.sqrt(len(Dmat))
        t = mean_d / np.where(se == 0, 1e-9, se)
        p = 2 * (1 - _student_cdf(np.abs(t), len(Dmat) - 1))
        summary[tissue] = dict(
            n=len(grp),
            dp_spread_mean=float(Dmat.std(1).mean()),
            dp_spread_std=float(Dmat.std(1).std()),
            cluster_mean_dp=mean_d.tolist(),
            cluster_t=t.tolist(),
            cluster_p=p.tolist(),
            n_sig_clusters=int((p < 0.05).sum()),
        )
    out = dict(K=K, PC=PC, seed=SEED, records=records,
               vanilla_cross=vanilla_cross, summary=summary)
    json.dump(out, open(OUT_DATA, "w"), indent=2)
    print(f"\nwrote {OUT_DATA}")
    print("haematopoietic:", summary["hema"]["n"], "pairs,",
          summary["hema"]["n_sig_clusters"], "partitions with delta_p_k != 0 (p<0.05),",
          "mean dp_spread=%.1f" % summary["hema"]["dp_spread_mean"])
    print("neural crest:  ", summary["neural"]["n"], "pairs,",
          summary["neural"]["n_sig_clusters"], "partitions with delta_p_k != 0 (p<0.05),",
          "mean dp_spread=%.1f" % summary["neural"]["dp_spread_mean"])
    vc = [r["dp_spread"] for r in vanilla_cross]
    print("vanilla cross-seed null:",
          "  ".join(f"{r['pair']}={r['dp_spread']:.2f}" for r in vanilla_cross),
          f"  mean={np.mean(vc):.2f}")
    make_figure(records, summary, vanilla_cross)


def _student_cdf(t, df):
    """Two-sided p via scipy if present, else rough normal approx."""
    try:
        from scipy import stats
        return stats.t.cdf(t, df)
    except Exception:
        from math import erf, sqrt
        return 0.5 * (1 + erf(t / sqrt(2 * (df / (df - 1)))))


def make_figure(records, summary, vanilla_cross):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7.5,
        "axes.titlesize": 9, "axes.labelsize": 8, "xtick.labelsize": 7,
        "ytick.labelsize": 7, "axes.linewidth": 0.5, "axes.unicode_minus": False,
    })
    prior = [r for r in records if not r["is_null"]]
    hema = [r for r in prior if r["tissue"] == "hema"]
    neural = [r for r in prior if r["tissue"] == "neural"]
    null = [r for r in records if r["is_null"]]

    fig, ax = plt.subplots(1, 2, figsize=(6.7, 3.0))

    # ---- Panel A: per-config delta_p_k heatmap (scale-free mass-share shift) ----
    def heat(ax, recs, title):
        if not recs:
            ax.set_title(title); return
        M = np.array([r["delta_p_k"] for r in recs])   # (n, K)
        vmax = np.abs(M).max()
        im = ax.imshow(M, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(K)); ax.set_xticklabels([f"k{j+1}" for j in range(K)])
        ax.set_yticks(range(len(recs)))
        ax.set_yticklabels([r["model"] for r in recs], fontsize=6)
        ax.set_xlabel("latent partition")
        ax.set_title(title)
        return im
    im = heat(ax[0], hema + neural,
              "scale-free spectral-mass shift per partition (prior - vanilla, pp)")
    if im is not None:
        fig.colorbar(im, ax=ax[0], fraction=0.046, pad=0.04, label="delta_p_k (pp)")

    # ---- Panel B: dp_spread distribution by group (scale-free) ----
    # The proper null is two INDEPENDENT vanilla seeds (untrained decoders), NOT
    # the trivial self-pair (which is identically 0 and cannot judge significance).
    vc_vals = [r["dp_spread"] for r in vanilla_cross]
    groups = [("haematopoietic", [r["dp_spread"] for r in hema]),
              ("neural crest", [r["dp_spread"] for r in neural]),
              ("vanilla cross-seed (null)", vc_vals)]
    GROUP_COLS = [fp.C_HEMA, fp.C_NEURAL, fp.GRAY]   # haematopoietic, neural crest, vanilla-null
    for i, (lab, vals) in enumerate(groups):
        xs = np.random.default_rng(0).uniform(i - 0.18, i + 0.18, len(vals))
        ax[1].plot(xs, vals, "o", ms=3, color=GROUP_COLS[i], alpha=0.7)
        ax[1].boxplot([vals], positions=[i], widths=0.3, showfliers=False,
                      patch_artist=True,
                      boxprops=dict(facecolor=GROUP_COLS[i], alpha=0.3, linewidth=0.5),
                      medianprops=dict(color="k", linewidth=0.8))
    ax[1].set_xticks(range(3))
    ax[1].set_xticklabels([g[0] for g in groups], rotation=15, fontsize=7)
    ax[1].set_ylabel("std(delta_p_k) across partitions")
    ax[1].set_title("no prior-specific reorganisation (kappa-C v2)")
    h_spread = np.mean([r["dp_spread"] for r in hema])
    n_spread = np.mean([r["dp_spread"] for r in neural])
    vc_spread = np.mean(vc_vals)
    ax[1].text(0.02, 0.96,
               f"hema prior={h_spread:.1f}  neural prior={n_spread:.1f}\n"
               f"vanilla cross-seed null={vc_spread:.1f}  (prior BELOW null)",
               transform=ax[1].transAxes, fontsize=6.5, va="top")

    fig.tight_layout()
    fig.savefig(OUT_PDF, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    print(f"wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
