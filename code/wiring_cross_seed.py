"""
wiring_cross_seed.py
====================
Cross-seed reproducibility of the prior-wiring test.

For each (tissue, condition) we have multiple training seeds. We pair each
prior-injected seed with the SAME seed's vanilla model (matched seed, matched
data, matched architecture; only the prior differs) and compute the per-edge
Mann-Whitney z of prior-edge coupling vs vanilla. We then aggregate across seeds
with Stouffer's z (combines per-seed zs assuming independence; effective for a
small fixed-seed design) and a sign test (non-parametric).

Outputs
-------
  outputs/positive_result/fig_wiring_cross_seed.{png,pdf}
  outputs/positive_result/WIRING_CROSS_SEED.md
  outputs/positive_result/wiring_cross_seed.csv
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
import run_registry as rr

NEX = "F:/UNAGI_Project/next_phase"
DEC = "F:/UNAGI_Project/UNAGI/decoder_out"
OUT = "F:/UNAGI_Project/outputs/positive_result"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({"font.size": 8.5, "figure.dpi": 160,
                     "axes.spines.top": False, "axes.spines.right": False})


# ---------------------------------------------------------- helpers (from wiring test)
def centred_unit(W):
    A = W - W.mean(axis=0, keepdims=True)
    n = np.linalg.norm(A, axis=0, keepdims=True)
    n[n == 0] = 1.0
    return A / n


def load_lm(cfg, it="iter_4"):
    """Load via run_registry so the quality gate applies.

    Two silent failures are avoided: (i) the bare name R13 used to resolve to
    decoder_out/R13/iter_4, which is a degenerate matrix (||W||=7.6, column-norm
    CV 1.96); the correct seed-42 full-prior matrix is unified/R13_s42.npy;
    (ii) the seed-42 neural vanilla must be R0_seed42, not the legacy bare R0
    run (effective rank 38.1 vs 24.0-27.0 for the seed-matched vanillas), which
    previously produced a spurious z=+3.91 instead of z=+0.19.
    """
    W = rr.get_matrix(cfg)
    G = W.shape[1]
    idx = rr.get_gene_index(G)
    g = [None] * G
    for name, i in idx.items():
        g[i] = name
    return W, g, cfg


def coupling(Ac, pairs):
    if len(pairs) == 0:
        return np.array([])
    p = np.asarray(pairs, dtype=int)
    return np.einsum("ij,ij->i", Ac[:, p[:, 0]].T, Ac[:, p[:, 1]].T)


def mw_z_per_edge(a, b):
    """Two-sided Mann-Whitney normal-approx z of a > b (per-edge test)."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    allv = np.concatenate([a, b])
    order = np.argsort(allv, kind="mergesort")
    r = np.empty(allv.size); r[order] = np.arange(1, allv.size + 1).astype(float)
    Ra = r[:a.size].sum()
    U = Ra - a.size * (a.size + 1) / 2.0
    mu = a.size * b.size / 2.0
    sd = np.sqrt(a.size * b.size * (a.size + b.size + 1) / 12.0)
    return (U - mu) / sd


def stouffer(zs):
    zs = np.asarray(zs, float)
    return zs.sum() / np.sqrt(len(zs))


def stouffer_p(z):
    from math import erfc, sqrt
    return erfc(abs(z) / sqrt(2.0))


# ---------------------------------------------------------- TRRUST and prior edges
TRRUST = "F:/UNAGI_Project/unagi_custom_data/trrust_raw_human.txt"
trrust = {}
with open(TRRUST, encoding="utf-8", errors="ignore") as f:
    for line in f:
        pp = line.rstrip("\n").split("\t")
        if len(pp) < 3:
            continue
        trrust.setdefault(pp[0].upper(), set()).add(pp[1].upper())

hp = json.load(open(os.path.join(NEX, "data", "hema_prior.json")))
hema_edges = [(str(t).upper(), str(g).upper()) for t, gs in hp.get("tf_db", {}).items() for g in gs]

NEURO_TFS = """SOX2 PAX6 HES1 HES5 VIM PROM1 SOX9 EOMES NEUROG1 NEUROG2 ASCL1
MKI67 CCND1 TBR1 BCL11B FEZF2 SOX5 CPLX3 CPLX4 SATB2 CUX1 CUX2 POU3F2 POU3F3
MEF2C DLX1 DLX2 DLX5 DLX6 GAD1 GAD2 LHX6 NKX2-1 SST PVALB VIP CALB2 GFAP S100B
AQP4 OLIG1 OLIG2 PLP1 MBP ALDH1L1""".split()
neuro_edges = [(t, g) for t in NEURO_TFS for g in trrust.get(t.upper(), ())]


# ---------------------------------------------------------- paired configs
HEMA_SEEDS = [42, 123, 456]
HEMA = {
    "true":    ["R13"] + [f"R13_seed{s}" for s in HEMA_SEEDS[1:]],
    "wrong":   ["R16"],   # R16 has a labelled "WrongPrior" summary; R165 is undocumented, exclude
    # spec "B": one shared TRUE VANILLA for every seed (the vanilla readout
    # exists only at seed 42).  NOT [f"R14_seed{s}"] -- those are the VEHICLE.
    "vanilla": ["R14"] * len(HEMA_SEEDS),
}
HEMA_PAIRS = {("true", s): ("true", s) for s in HEMA_SEEDS}
HEMA_PAIRS[("wrong", 42)] = ("wrong", 42)  # R16 default = seed 42; R165_seed42

NEURAL_SEEDS = [42, 123, 456]
NEURAL = {
    "prior":   ["R1"] + [f"R1_seed{s}" for s in NEURAL_SEEDS[1:]],
    "vanilla": ["R0_seed42"] + [f"R0_seed{s}" for s in NEURAL_SEEDS[1:]],
}
# NEURAL_SEEDS[0] = 42, [1] = 123, [2] = 456 — index by position
NEURAL_CFG_BY_SEED = {42: ("R1", "R0_seed42"), 123: ("R1_seed123", "R0_seed123"),
                      456: ("R1_seed456", "R0_seed456")}


def load_genes(glist):
    g = [str(x).upper() for x in json.load(open(os.path.join(DEC, glist)))]
    return {x: i for i, x in enumerate(g)}


# ---------------------------------------------------------- run hema cross-seed
def run_hema_one(model_cfg, van_cfg, edges, glist="hema_gene_list.json", seed=None):
    Wm, gm, pm = load_lm(model_cfg)
    Wv, gv, pv = load_lm(van_cfg)
    assert Wm.shape == Wv.shape, f"shape mismatch {model_cfg} vs {van_cfg}"
    Acm = centred_unit(Wm); Acv = centred_unit(Wv)
    idxm = {g: i for i, g in enumerate(gm)}; idxv = {g: i for i, g in enumerate(gv)}
    pairs = []
    for a, b in edges:
        if a in idxm and b in idxm and a in idxv and b in idxv and idxm[a] != idxm[b]:
            pairs.append((idxm[a], idxm[b]))
    if not pairs:
        return None
    p = np.asarray(pairs, dtype=int)
    ce_m = coupling(Acm, p); ce_v = coupling(Acv, p)
    z = mw_z_per_edge(ce_m, ce_v)
    return dict(seed=seed, model=model_cfg, vanilla=van_cfg, n_edges=len(pairs),
                mean_model=ce_m.mean(), mean_vanilla=ce_v.mean(),
                z=float(z), p=float(stouffer_p(z)))


print("=" * 70)
print("HEMATOPOIETIC cross-seed (true prior; hema edges)")
print("=" * 70)
rows_true = []
# BASELINE = the TRUE VANILLA -- spec "B" (user decision, 2026-09-16).
# The vanilla readout exists ONLY at seed 42, so all three seeds share it; B is
# the only spec in which every seed has a genuine no-prior control.
# Alternatives, and their aggregates, are tabulated by
# code/wiring_baseline_compare.py -> outputs/positive_result/wiring_baseline_compare.json:
#   A  seed42 vanilla, 123/456 vehicle   Stouffer z=+1.589  p=0.112   (mixed spec)
#   B  all seeds vanilla                 Stouffer z=+2.039  p=0.042   <-- USED HERE
#   C  all seeds vehicle (R14_seed*)     Stouffer z=+1.396  p=0.163   (control is a prior arm)
# Do NOT revert to per-seed "R14_seed{s}": that is the VEHICLE arm (37 TF /
# 1054 genes), not an empty prior, and it inflated the seed-456 "flip" roughly
# threefold (-2.454 vs the correct -0.877).
for s in HEMA_SEEDS:
    m = "R13" if s == 42 else f"R13_seed{s}"
    v = "R14"            # TRUE VANILLA for every seed (spec B)
    r = run_hema_one(m, v, hema_edges, seed=s)
    if r: rows_true.append(r)
    print(f"  seed {s:3d}: {m:>14s} vs {v:<14s}  z={r['z']:+.2f} p={r['p']:.2g}  "
          f"mean {r['mean_model']:+.4f} vs {r['mean_vanilla']:+.4f}  n={r['n_edges']}")

z_agg = stouffer([r["z"] for r in rows_true])
p_agg = stouffer_p(z_agg)
signs = [int(r["z"] > 0) for r in rows_true]
sign_test = ("all positive" if all(signs)
             else "majority positive" if sum(signs) >= 2
             else "mixed")
print(f"  STOUFFER: z={z_agg:+.2f}, p={p_agg:.2g}  (signs: {signs}, {sign_test})")

print("\n--- wrong prior (R16 vs R14; hema edges — control: should NOT be wired) ---")
rows_wrong = []
wrong_pairs = [(42, "R16", "R14")]
for s, m, v in wrong_pairs:
    r = run_hema_one(m, v, hema_edges, seed=s)
    if r: rows_wrong.append(r)
    print(f"  {m:>14s} vs {v:<14s}  z={r['z']:+.2f} p={r['p']:.2g}")
zw = stouffer([r["z"] for r in rows_wrong])
print(f"  STOUFFER (wrong): z={zw:+.2f}, p={stouffer_p(zw):.2g}  "
      f"(signs: {[int(r['z']>0) for r in rows_wrong]})")


# ---------------------------------------------------------- run neural cross-seed
def run_neural_one(model_cfg, van_cfg, edges, seed):
    Wm, gm, pm = load_lm(model_cfg)
    Wv, gv, pv = load_lm(van_cfg)
    assert Wm.shape == Wv.shape
    Acm = centred_unit(Wm); Acv = centred_unit(Wv)
    idxm = {g: i for i, g in enumerate(gm)}; idxv = {g: i for i, g in enumerate(gv)}
    pairs = []
    for a, b in edges:
        if a in idxm and b in idxm and a in idxv and b in idxv and idxm[a] != idxm[b]:
            pairs.append((idxm[a], idxm[b]))
    if not pairs:
        return None
    p = np.asarray(pairs, dtype=int)
    ce_m = coupling(Acm, p); ce_v = coupling(Acv, p)
    z = mw_z_per_edge(ce_m, ce_v)
    return dict(seed=seed, model=model_cfg, vanilla=van_cfg, n_edges=len(pairs),
                mean_model=ce_m.mean(), mean_vanilla=ce_v.mean(),
                z=float(z), p=float(stouffer_p(z)))


print("\n" + "=" * 70)
print("NEURAL cross-seed (cortex prior; TRRUST-derived cortex edges)")
print("=" * 70)
rows_neu = []
for s in NEURAL_SEEDS:
    m, v = NEURAL_CFG_BY_SEED[s]
    r = run_neural_one(m, v, neuro_edges, seed=s)
    if r: rows_neu.append(r)
    print(f"  seed {s:3d}: {m:>14s} vs {v:<14s}  z={r['z']:+.2f} p={r['p']:.2g}  "
          f"mean {r['mean_model']:+.4f} vs {r['mean_vanilla']:+.4f}  n={r['n_edges']}")
zn = stouffer([r["z"] for r in rows_neu])
pn = stouffer_p(zn)
signs = [int(r["z"] > 0) for r in rows_neu]
print(f"  STOUFFER: z={zn:+.2f}, p={pn:.2g}  (signs: {signs})")

# ---------------------------------------------------------- figure
fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8))

# Panel A: hema true vs vanilla — per-seed + stouffer
ax = axes[0]
seeds = [r["seed"] for r in rows_true]
zs = [r["z"] for r in rows_true]
ax.bar(range(len(seeds)), zs, color="#B2182B", edgecolor="#333", lw=0.5)
ax.axhline(0, color="#333", lw=0.6)
ax.set_xticks(range(len(seeds))); ax.set_xticklabels([f"s{s}" for s in seeds], fontsize=8)
ax.set_ylabel("per-seed Mann–Whitney z (true vs vanilla)", fontsize=8)
ax.set_title(f"A. Hematopoietic — true prior wires hema edges\n"
             f"Stouffer z={z_agg:+.2f}, p={p_agg:.1e}, {len(seeds)} seeds", fontsize=8.5)
for i, v in enumerate(zs):
    ax.text(i, v + 0.25, f"{v:+.2f}", ha="center", fontsize=7.5, fontweight="bold",
            color="#B2182B")
ax.set_ylim(min(min(zs), 0) * 1.4 - 0.5, max(zs) * 1.3 + 0.5)
ax.grid(axis="y", alpha=0.25, lw=0.4)

# Panel B: neural — per-seed + stouffer
ax = axes[1]
seeds = [r["seed"] for r in rows_neu]
zs = [r["z"] for r in rows_neu]
ax.bar(range(len(seeds)), zs, color="#2166AC", edgecolor="#333", lw=0.5)
ax.axhline(0, color="#333", lw=0.6)
ax.set_xticks(range(len(seeds))); ax.set_xticklabels([f"s{s}" for s in seeds], fontsize=8)
ax.set_ylabel("per-seed Mann–Whitney z (prior vs vanilla)", fontsize=8)
ax.set_title(f"B. Neural — cortex prior wires its edges\n"
             f"Stouffer z={zn:+.2f}, p={pn:.1e}, {len(seeds)} seeds", fontsize=8.5)
for i, v in enumerate(zs):
    ax.text(i, v + 0.25, f"{v:+.2f}", ha="center", fontsize=7.5, fontweight="bold",
            color="#2166AC")
ax.set_ylim(min(min(zs), 0) * 1.4 - 0.5, max(zs) * 1.3 + 0.5)
ax.grid(axis="y", alpha=0.25, lw=0.4)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_wiring_cross_seed.pdf"))
fig.savefig(os.path.join(OUT, "fig_wiring_cross_seed.png"), dpi=200)
plt.close(fig)
print("\nwrote fig_wiring_cross_seed.pdf/png")

# ---------------------------------------------------------- CSV + MD
out_csv = os.path.join(OUT, "wiring_cross_seed.csv")
with open(out_csv, "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=list(rows_true[0].keys()))
    wr.writeheader()
    for r in rows_true + rows_neu: wr.writerow(r)

md = ["# Prior-wiring cross-seed reproducibility\n",
      "## Hema (true prior; hema edges)\n",
      "| seed | model | z | p | n edges | mean model | mean vanilla |",
      "|---|---|---|---|---|---|---|"]
for r in rows_true:
    md.append(f"| {r['seed']} | {r['model']} | {r['z']:+.2f} | {r['p']:.2g} | {r['n_edges']} | "
              f"{r['mean_model']:+.4f} | {r['mean_vanilla']:+.4f} |")
md += [f"| **Stouffer** | — | **{z_agg:+.2f}** | **{p_agg:.2g}** | "
       f"signs={[int(r['z']>0) for r in rows_true]} | — | — |",
       "",
       "## Hema (wrong prior; hema edges — should NOT be wired)\n",
       "| seed | model | z | p |",
       "|---|---|---|---|"]
for r in rows_wrong:
    md.append(f"| {r['seed']} | {r['model']} | {r['z']:+.2f} | {r['p']:.2g} |")
md += [f"| **Stouffer** | — | **{zw:+.2f}** | **{stouffer_p(zw):.2g}** |",
       "",
       "## Neural (cortex prior; TRRUST-derived cortex edges)\n",
       "| seed | model | z | p | n edges | mean model | mean vanilla |",
       "|---|---|---|---|---|---|---|"]
for r in rows_neu:
    md.append(f"| {r['seed']} | {r['model']} | {r['z']:+.2f} | {r['p']:.2g} | {r['n_edges']} | "
              f"{r['mean_model']:+.4f} | {r['mean_vanilla']:+.4f} |")
md += [f"| **Stouffer** | — | **{zn:+.2f}** | **{pn:.2g}** | "
       f"signs={[int(r['z']>0) for r in rows_neu]} | — | — |"]
open(os.path.join(OUT, "WIRING_CROSS_SEED.md"), "w").write("\n".join(md))
print("wrote WIRING_CROSS_SEED.md and wiring_cross_seed.csv")