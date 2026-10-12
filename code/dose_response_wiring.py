"""
dose_response_wiring.py
=======================
Does the decoder internalise the prior *in proportion to how much prior is
injected*? And is the seed-456 failure a property of the seed or of the dose?

Design
------
The hematopoietic arm is not a 2-group comparison — it is a graded design.
Recovered prior sizes (edges):

    R14  0    (TRUE VANILLA, no prior machinery -- shared by all seeds)
    R11  21   (KLF1 + TAL1)
    R12  55   (GATA2 + GFI1B + PAX5)
    R84  84   (GATA1 + GFI1B + KLF1 + TAL1)
    R107 107  (erythroid + lymphoid)
    R13  325  (pan-hematopoietic, 11 TFs)

each available for seeds 42 / 123 / 456 (R107 only seed 42), every one paired
against the TRUE VANILLA (the vanilla readout exists only at seed 42, so
all seeds share it -- spec "B"); see code/wiring_baseline_compare.py.

The trap
--------
A per-edge Mann-Whitney z grows like sqrt(n_edges). Comparing z across doses of
21 vs 325 edges is therefore circular: a bigger prior automatically buys a
bigger z even if the per-edge effect is identical. We therefore report BOTH:

  * delta-coupling  mean prior-edge cosine (model) - mean (vanilla), which is
    an effect size independent of the number of edges, and
  * z against a SIZE-MATCHED random-edge null: for each dose we draw 2000
    random edge sets of the SAME size (matched on TF degree where possible)
    and recompute z. The reported quantity is
        z_null_corrected = (z_obs - mean(z_null)) / sd(z_null).
    This is what makes doses comparable.

Outputs
-------
  outputs/positive_result/fig_dose_response_wiring.{png,pdf}
  outputs/positive_result/DOSE_RESPONSE_WIRING.md
  outputs/positive_result/dose_response_wiring.csv
"""
import os, sys, json, csv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import run_registry as rr
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

DEC = "F:/UNAGI_Project/UNAGI/decoder_out"
OUT = "F:/UNAGI_Project/outputs/positive_result"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({"font.size": 8.5, "figure.dpi": 160,
                     "axes.spines.top": False, "axes.spines.right": False})

RNG = np.random.default_rng(20260904)
N_NULL = 2000


# --------------------------------------------------------------- helpers
# Matrix loading, gene indexing and cross-axis centring all come from
# run_registry, which enforces a quality gate. A naive loader resolves the bare
# name "R13" to decoder_out/R13/iter_4, which is a DEGENERATE matrix
# (||W||=7.6, column-norm CV 1.96); the correct seed-42 full-prior matrix is
# decoder_unified/loading/R13_s42.npy. That bug inflated the R13 seed-42
# wiring z, so the registry is not optional.
def centred_unit(W):
    return rr.centred_unit(W)


def load(cfg):
    return rr.get_matrix(cfg)


def genes_for(G):
    return rr.get_gene_index(G)


def coupling(Ac, pairs):
    p = np.asarray(pairs, dtype=int)
    return np.einsum("ij,ij->i", Ac[:, p[:, 0]].T, Ac[:, p[:, 1]].T)


def mw_z(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    allv = np.concatenate([a, b])
    order = np.argsort(allv, kind="mergesort")
    r = np.empty(allv.size); r[order] = np.arange(1, allv.size + 1).astype(float)
    U = r[:a.size].sum() - a.size * (a.size + 1) / 2.0
    mu = a.size * b.size / 2.0
    sd = np.sqrt(a.size * b.size * (a.size + b.size + 1) / 12.0)
    return (U - mu) / sd


def stouffer(zs):
    zs = np.asarray(zs, float)
    return zs.sum() / np.sqrt(len(zs))


def stouffer_p(z):
    from math import erfc, sqrt
    return erfc(abs(z) / sqrt(2.0))


def spearman(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx = rx - rx.mean(); ry = ry - ry.mean()
    den = np.linalg.norm(rx) * np.linalg.norm(ry)
    rho = float(rx @ ry / den) if den else 0.0
    n = len(x)
    # t approximation
    if abs(rho) >= 1.0:
        return rho, 0.0
    t = rho * np.sqrt((n - 2) / (1 - rho ** 2))
    from math import erfc, sqrt
    p = erfc(abs(t) / sqrt(2.0))
    return rho, p


# --------------------------------------------------------------- design
CONF = json.load(open(os.path.join(OUT, "prior_configs.json")))

# (label, config, vanilla config, seed)
HEMA = [
    ("R11",  "R11_seed42",            "R14",  42,  21),
    ("R11",  "R11_KLF1_TAL1_seed123", "R14", 123, 21),
    ("R11",  "R11_KLF1_TAL1_seed456", "R14", 456, 21),
    ("R12",  "R12_seed42",            "R14",  42,  55),
    ("R12",  "R12_seed123",           "R14", 123, 55),
    ("R12",  "R12_seed456",           "R14", 456, 55),
    ("R84",  "R84_seed42",            "R14",  42,  84),
    ("R84",  "R84_seed123",           "R14", 123, 84),
    ("R107", "R107_seed42",           "R14",  42,  107),
    ("R13",  "R13",                   "R14",  42,  325),
    ("R13",  "R13_seed123",           "R14", 123, 325),
    ("R13",  "R13_seed456",           "R14", 456, 325),
]

cache = {}
def get(cfg):
    if cfg not in cache:
        W = load(cfg)
        A = centred_unit(W)
        idx = genes_for(W.shape[1])
        cache[cfg] = (A, idx, W.shape)
    return cache[cfg]


print("=" * 80)
print("DOSE-RESPONSE WIRING: hematopoietic prior (0 -> 325 edges), 3 seeds")
print("=" * 80)

rows = []
for label, cfg, van, seed, nedges in HEMA:
    edges = [(t.upper(), g.upper()) for t, g in CONF[cfg]["edges"]]
    Am, idxm, shapem = get(cfg)
    Av, idxv, shapev = get(van)
    assert shapem == shapev, f"shape mismatch {cfg} {shapem} vs {van} {shapev}"

    pairs = []
    for a, b in edges:
        if a in idxm and b in idxm and a != b:
            pairs.append((idxm[a], idxm[b]))
    pairs = np.array(sorted(set(pairs)), dtype=int)
    if len(pairs) == 0:
        print(f"  {cfg}: no usable edges"); continue
    genes = np.array(sorted(idxm.values()))

    cm = coupling(Am, pairs); cv = coupling(Av, pairs)
    d = float(cm.mean() - cv.mean())
    z = float(mw_z(cm, cv))

    # ---- matched random-edge nulls.
    # Three nulls of increasing strictness, because a uniform random gene pair
    # is not a fair comparison for a TF->target edge: the TF end is a hub gene.
    #   A tf_fixed       : keep the TF end, randomise the target over all genes
    #   B tf_nontarget   : keep the TF end, randomise the target over genes that
    #                      are NOT a target of that TF (strictest - it controls
    #                      for both TF hubness and target-gene identity)
    tf_idx = np.array([p[0] for p in pairs])
    tg_idx = np.array([p[1] for p in pairs])
    nontarget_pool = genes[~np.isin(genes, np.unique(tg_idx))]
    nulls = {"tf_fixed": None, "tf_nontarget": None}
    res = {}
    for mode in nulls:
        zs = np.empty(N_NULL); ds = np.empty(N_NULL)
        for i in range(N_NULL):
            pool = genes if mode == "tf_fixed" else nontarget_pool
            b = RNG.choice(pool, len(pairs))
            rp = np.stack([tf_idx, b], 1)
            rp = rp[rp[:, 0] != rp[:, 1]]
            c1 = coupling(Am, rp); c2 = coupling(Av, rp)
            zs[i] = mw_z(c1, c2); ds[i] = c1.mean() - c2.mean()
        res[mode] = (zs, ds)
    zc = float((z - res["tf_nontarget"][0].mean()) / (res["tf_nontarget"][0].std() + 1e-12))
    dc = float((d - res["tf_nontarget"][1].mean()) / (res["tf_nontarget"][1].std() + 1e-12))
    zcA = float((z - res["tf_fixed"][0].mean()) / (res["tf_fixed"][0].std() + 1e-12))

    rows.append(dict(label=label, config=cfg, vanilla=van, seed=seed,
                     n_edges=len(pairs), delta=d, z=z,
                     z_nullcorr=zc, delta_nullcorr=dc, z_nullcorr_tf=zcA,
                     mean_model=float(cm.mean()), mean_vanilla=float(cv.mean())))
    print(f"  {label:5s} s{seed:3d} n={len(pairs):4d}  "
          f"delta={d:+.4f} (null {dc:+.2f})   z={z:+.2f} "
          f"(null_tf {zcA:+.2f}, null_strict {zc:+.2f})")

# --------------------------------------------------------------- dose trend
print("\n--- dose trend (effect size, pooled over seeds) ---")
doses = sorted({r["n_edges"] for r in rows})
for ned in doses:
    sub = [r for r in rows if r["n_edges"] == ned]
    per = ", ".join(f"{r['delta']:+.4f}" for r in sub)
    print(f"  {ned:4d} edges: delta mean={np.mean([r['delta'] for r in sub]):+.4f} "
          f"[{per}]   z_nullcorr mean={np.mean([r['z_nullcorr'] for r in sub]):+.2f}")

rho_d, p_d = spearman([r["n_edges"] for r in rows], [r["delta"] for r in rows])
rho_z, p_z = spearman([r["n_edges"] for r in rows], [r["z_nullcorr"] for r in rows])
print(f"\n  Spearman(dose, delta coupling)      rho={rho_d:+.3f} p={p_d:.3g} (n={len(rows)})")
print(f"  Spearman(dose, null-corrected z)    rho={rho_z:+.3f} p={p_z:.3g} (n={len(rows)})")

# raw z also reported, to show it is confounded by edge count
rho_raw, p_raw = spearman([r["n_edges"] for r in rows], [r["z"] for r in rows])
print(f"  Spearman(dose, RAW z)               rho={rho_raw:+.3f} p={p_raw:.3g} "
      f"<- confounded by sqrt(n_edges)")

# --------------------------------------------------------------- seed anatomy
print("\n--- seed anatomy: is seed 456 globally weak, or only at the top dose? ---")
for seed in [42, 123, 456]:
    sub = [r for r in rows if r["seed"] == seed]
    sub.sort(key=lambda r: r["n_edges"])
    print(f"  seed {seed}: " + "  ".join(f"{r['n_edges']:>3d}e:{r['delta']:+.4f}"
                                        for r in sub))

print("\n--- within-seed dose slope (delta vs log dose) ---")
slopes = []
for seed in [42, 123, 456]:
    sub = sorted([r for r in rows if r["seed"] == seed], key=lambda r: r["n_edges"])
    x = np.log(np.array([r["n_edges"] for r in sub], float))
    y = np.array([r["delta"] for r in sub], float)
    if len(sub) >= 3:
        b = np.polyfit(x, y, 1)[0]
        slopes.append(b)
        print(f"  seed {seed}: slope={b:+.4f} per log-edge  (n doses={len(sub)})")
if slopes:
    from math import erfc, sqrt
    m = np.mean(slopes); se = np.std(slopes, ddof=1) / np.sqrt(len(slopes))
    t = m / se if se > 0 else 0.0
    print(f"  MEAN slope={m:+.4f} +- {se:.4f} (SE), t={t:+.2f}, "
          f"p={erfc(abs(t)/sqrt(2.0)):.3g} across {len(slopes)} seeds")

# --------------------------------------------------------------- figure
fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.9))
COL = {42: "#B2182B", 123: "#2166AC", 456: "#4D9221"}

ax = axes[0]
for seed in [42, 123, 456]:
    sub = sorted([r for r in rows if r["seed"] == seed], key=lambda r: r["n_edges"])
    ax.plot([r["n_edges"] for r in sub], [r["delta"] for r in sub], "-o",
            color=COL[seed], ms=4, lw=1.2, label=f"seed {seed}")
ax.axhline(0, color="#333", lw=0.6)
ax.set_xscale("log")
ax.set_xticks([21, 55, 84, 107, 325])
ax.set_xticklabels(["21", "55", "84", "107", "325"])
ax.set_xlabel("prior edges injected (log)", fontsize=8)
ax.set_ylabel(r"$\Delta$prior-edge cosine (model $-$ vanilla)", fontsize=8)
ax.set_title(f"A. Dose-response of prior internalisation\n"
             f"Spearman rho={rho_d:+.2f}, p={p_d:.2g}, n={len(rows)}", fontsize=8.5)
ax.legend(fontsize=7, frameon=False)
ax.grid(alpha=0.25, lw=0.4)

ax = axes[1]
for seed in [42, 123, 456]:
    sub = sorted([r for r in rows if r["seed"] == seed], key=lambda r: r["n_edges"])
    ax.plot([r["n_edges"] for r in sub], [r["z_nullcorr"] for r in sub], "-o",
            color=COL[seed], ms=4, lw=1.2, label=f"seed {seed}")
ax.axhline(0, color="#333", lw=0.6)
ax.axhline(1.96, color="#888", lw=0.6, ls="--")
ax.set_xscale("log")
ax.set_xticks([21, 55, 84, 107, 325])
ax.set_xticklabels(["21", "55", "84", "107", "325"])
ax.set_xlabel("prior edges injected (log)", fontsize=8)
ax.set_ylabel("size-matched null-corrected $z$", fontsize=8)
ax.set_title(f"B. Significance after removing the $\\sqrt{{n}}$ advantage\n"
             f"dashed = 1.96", fontsize=8.5)
ax.grid(alpha=0.25, lw=0.4)

ax = axes[2]
labels = [f"{r['label']}\n{r['n_edges']}e" for r in rows]
ax.bar(range(len(rows)), [r["delta"] for r in rows],
       color=[COL[r["seed"]] for r in rows], edgecolor="#333", lw=0.4)
ax.axhline(0, color="#333", lw=0.6)
ax.set_xticks(range(len(rows)))
ax.set_xticklabels([f"{r['label']}\ns{r['seed']}" for r in rows], fontsize=6.2)
ax.set_ylabel(r"$\Delta$prior-edge cosine", fontsize=8)
ax.set_title("C. Every dose x seed pair\ncolour = seed", fontsize=8.5)
ax.grid(axis="y", alpha=0.25, lw=0.4)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_dose_response_wiring.pdf"))
fig.savefig(os.path.join(OUT, "fig_dose_response_wiring.png"), dpi=200)
plt.close(fig)
print("\nwrote fig_dose_response_wiring.pdf/png")

with open(os.path.join(OUT, "dose_response_wiring.csv"), "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    wr.writeheader(); wr.writerows(rows)

md = ["# Dose-response of prior internalisation", "",
      "Per-edge cosine coupling of injected prior edges, model vs the SAME seed's "
      "vanilla (R14) model. `delta` is an effect size independent of edge count; "
      "`z_nullcorr` is the Mann-Whitney z standardised against 2000 size-matched "
      "random edge sets, which removes the sqrt(n_edges) advantage of larger priors.",
      "",
      "| config | seed | edges | delta cosine | null-corr delta | z | null-corr z |",
      "|---|---|---|---|---|---|---|"]
for r in rows:
    md.append(f"| {r['label']} ({r['config']}) | {r['seed']} | {r['n_edges']} | "
              f"{r['delta']:+.4f} | {r['delta_nullcorr']:+.2f} | {r['z']:+.2f} | "
              f"{r['z_nullcorr']:+.2f} |")
md += ["", f"Spearman(dose, delta) = {rho_d:+.3f} (p={p_d:.3g}); "
           f"Spearman(dose, null-corrected z) = {rho_z:+.3f} (p={p_z:.3g}); "
           f"Spearman(dose, RAW z) = {rho_raw:+.3f} (p={p_raw:.3g}, confounded)."]
open(os.path.join(OUT, "DOSE_RESPONSE_WIRING.md"), "w").write("\n".join(md))
print("wrote DOSE_RESPONSE_WIRING.md and dose_response_wiring.csv")
