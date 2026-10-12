"""
prior_wiring_test.py
====================
Is the injected GRN prior actually written into the decoder?

Idea
----
Under the linear decoder approximation  x_g ~ z . W[:,g],  gene g's expression is
driven by its 64-dim loading vector W[:,g]. If the prior wires a TF to its target,
then after training the TF and its target should be driven by *similar* latent
directions, i.e. their loading profiles should be correlated.

CRITICAL correction over the earlier attempt: the raw columns W[:,g] are all
dominated by a shared across-axis mean component (every gene loads on the "grand"
axis simply because the decoder must reconstruct mean expression). A raw cosine is
therefore ~1 for every pair and carries no information. We centre each gene ACROSS
the 64 axes first, then take the cosine. This is the same fix that was required for
the axis-level marker enrichment statistic.

Test
----
  For prior edges (tf -> target) present in the gene list, compute
      c_edge = cos( centred W[:,tf], centred W[:,target] )
  Compare against a null of random gene pairs (same genes, shuffled pairing).
  Report mean edge coupling, z vs null, and the fraction of edges in the top 5%.

Controlled contrast (all on the same marrow data, same architecture):
  R13 = true hematopoietic prior (12 TFs)
  R16 = wrong prior (cortex TFs injected into marrow)
  R14 = no prior (the TRUE VANILLA; R14_seed* = vehicle)

Cross-tissue replication:
  R1 = neural (cortex) prior on the neural dataset, R0 = neural vanilla.
  Edges for the cortex TFs are taken from TRRUST v2 (curated human TF->target).
"""

import os, json, csv, gzip
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

rng = np.random.default_rng(0)


# ---------------------------------------------------------------- helpers
def centred_unit(W):
    """Centre each gene across the 64 latent axes, then L2-normalise."""
    A = W - W.mean(axis=0, keepdims=True)
    n = np.linalg.norm(A, axis=0, keepdims=True)
    n[n == 0] = 1.0
    return A / n


def load(cfg, glist, it="iter_4"):
    """Load via run_registry so the degenerate-matrix gate applies.

    Bare decoder_out/R13/iter_4 is a degenerate matrix (||W||=7.6, column-norm
    CV 1.96); the correct seed-42 full-prior matrix is unified/R13_s42.npy
    (||W||=100.4). The registry also refuses the legacy bare 'R0' neural vanilla
    (which previously produced a spurious z) in favour of R0_seed42.
    """
    W = rr.get_matrix(cfg)
    G = W.shape[1]
    idx = rr.get_gene_index(G)
    g = [None] * G
    for name, i in idx.items():
        g[i] = name
    return W, g


def coupling(Ac, pairs):
    """cosine between centred loading profiles for a list of (i, j) gene-index pairs."""
    if len(pairs) == 0:
        return np.array([])
    p = np.asarray(pairs, dtype=int)
    return np.einsum("ij,ij->i", Ac[:, p[:, 0]].T, Ac[:, p[:, 1]].T)


def null_pairs(n_genes, n, exclude=None, seed=1):
    r = np.random.default_rng(seed)
    a = r.integers(0, n_genes, size=n)
    b = r.integers(0, n_genes, size=n)
    ok = a != b
    return np.stack([a[ok], b[ok]], axis=1)


# ---------------------------------------------------------------- TRRUST
TRRUST = "F:/UNAGI_Project/unagi_custom_data/trrust_raw_human.txt"
trrust = {}
with open(TRRUST, encoding="utf-8", errors="ignore") as f:
    for line in f:
        pp = line.rstrip("\n").split("\t")
        if len(pp) < 3:
            continue
        trrust.setdefault(pp[0].upper(), set()).add(pp[1].upper())
print(f"[TRRUST] {len(trrust)} TFs, "
      f"{sum(len(v) for v in trrust.values())} curated edges")

# ---------------------------------------------------------------- hema edges
hp = json.load(open(os.path.join(NEX, "data", "hema_prior.json")))
tf_db = hp.get("tf_db", {})
hema_pairs_sym = [(str(t).upper(), str(g).upper()) for t, gs in tf_db.items() for g in gs]
print(f"[prior] hematopoietic: {len(tf_db)} TFs, {len(hema_pairs_sym)} edges")

# ---------------------------------------------------------------- neuro edges (TRRUST-derived)
NEURO_TFS = """SOX2 PAX6 HES1 HES5 VIM PROM1 SOX9 EOMES NEUROG1 NEUROG2 ASCL1
MKI67 CCND1 TBR1 BCL11B FEZF2 SOX5 CPLX3 CPLX4 SATB2 CUX1 CUX2 POU3F2 POU3F3
MEF2C DLX1 DLX2 DLX5 DLX6 GAD1 GAD2 LHX6 NKX2-1 SST PVALB VIP CALB2 GFAP S100B
AQP4 OLIG1 OLIG2 PLP1 MBP ALDH1L1""".split()
neuro_pairs_sym = [(t, g) for t in NEURO_TFS for g in trrust.get(t.upper(), ())]
print(f"[prior] neural cortex: {len(NEURO_TFS)} TFs, {len(neuro_pairs_sym)} TRRUST edges")


# ---------------------------------------------------------------- run one contrast
def run(tag, cfg, glist, pair_syms, n_null=40000, seed=1):
    W, genes = load(cfg, glist)
    idx = {g: i for i, g in enumerate(genes)}
    Ac = centred_unit(W)
    G = len(genes)
    pairs = np.array([(idx[a], idx[b]) for a, b in pair_syms
                      if a in idx and b in idx and idx[a] != idx[b]], dtype=int)
    if pairs.size == 0:
        return None
    ce = coupling(Ac, pairs)
    cb = coupling(Ac, null_pairs(G, n_null, seed=seed))
    z = (ce.mean() - cb.mean()) / (cb.std() / np.sqrt(ce.size))
    p95 = np.percentile(cb, 95)
    res = dict(label=tag, cfg=cfg, n_edges=int(ce.size),
               mean_edge_coupling=float(ce.mean()),
               mean_null_coupling=float(cb.mean()),
               null_sd=float(cb.std()),
               z=float(z),
               frac_edges_above_null95=float((ce > p95).mean()),
               fold=float(ce.mean() / cb.mean()) if cb.mean() != 0 else float("nan"))
    print(f"  [{tag}] n={res['n_edges']:5d}  edge={ce.mean():+.5f}  "
          f"null={cb.mean():+.5f}+-{cb.std():.5f}  z={z:+.2f}  "
          f"top5%={res['frac_edges_above_null95']*100:.1f}%")
    return res, ce, cb


# ================================================================ hematopoietic
print("\n=== HEMATOPOIETIC (marrow; hema prior edges) ===")
hema_rows, hema_d = [], {}
# spec "B": baseline = the TRUE VANILLA (not R14_seed42 = the VEHICLE arm).
for tag, cfg in [("hema_true", "R13"), ("hema_wrong", "R16"), ("hema_vanilla", "R14")]:
    r, ce, cb = run(tag, cfg, "hema_gene_list.json", hema_pairs_sym)
    hema_rows.append(r); hema_d[tag] = (ce, cb)

# per-edge paired comparison: true vs vanilla, wrong vs vanilla
def mw(a, b):
    from math import erfc, sqrt
    a = np.asarray(a, float); b = np.asarray(b, float)
    allv = np.concatenate([a, b])
    order = np.argsort(allv, kind="mergesort")
    r = np.empty(allv.size); r[order] = np.arange(1, allv.size + 1).astype(float)
    Ra = r[:a.size].sum()
    U = Ra - a.size * (a.size + 1) / 2
    mu = a.size * b.size / 2
    sd = np.sqrt(a.size * b.size * (a.size + b.size + 1) / 12)
    z = (U - mu) / sd
    return z, erfc(abs(z) / sqrt(2))

print("\nper-edge Mann-Whitney (same edge set, different models):")
for tag in ["hema_true", "hema_wrong"]:
    z, p = mw(hema_d[tag][0], hema_d["hema_vanilla"][0])
    print(f"   {tag} vs hema_vanilla: z={z:+.2f} p={p:.3g}")

# ---------------------------------------------------------------- 2x3 decisive control
# Does each model wire BOTH edge sets? For every (model, edge-set) cell we compare the
# per-edge coupling against the *same* edge set measured in the vanilla model.
#   - if a model wires only its OWN injected edges  -> internalisation is specific
#   - if it wires both                              -> internalisation is generic
print("\n=== 2x3 control: which edge set does each marrow model wire? ===")
marrow_models = [("hema_true(R13)", "R13"), ("hema_wrong(R16)", "R16"),
                 ("vanilla(R14)", "R14/vanilla")]
edge_sets = {"hema GRN edges": hema_pairs_sym, "cortex GRN edges": neuro_pairs_sym}
cell = {}
Wv, gv = load("R14/vanilla", "hema_gene_list.json")
Acv = centred_unit(Wv); idv = {g: i for i, g in enumerate(gv)}
for ename, eset in edge_sets.items():
    pidx = np.array([(idv[a], idv[b]) for a, b in eset
                     if a in idv and b in idv and idv[a] != idv[b]], dtype=int)
    base = coupling(Acv, pidx)
    for mname, cfg in marrow_models:
        Wm, gm = load(cfg, "hema_gene_list.json")
        if gm is gv and cfg == "R14/vanilla":
            ce = base
        else:
            Acm = centred_unit(Wm)
            ce = coupling(Acm, pidx)
        z, p = mw(ce, base)
        cell[(mname, ename)] = (float(ce.mean()), float(z), float(p))
        print(f"   {mname:16s} x {ename:18s} mean={ce.mean():+.4f} "
              f"z_vs_vanilla={z:+.2f} p={p:.3g}")

# ---------------------------------------------------------------- expression gating
# Hypothesis for why the mismatched prior fails to wire even its OWN edges in marrow:
# its TFs are not expressed / not variable there, so there is no signal to couple.
print("\n=== expression gating: decoder involvement of each prior's TFs ===")


def tf_percentile(cfg, glist, tf_syms):
    W, g = load(cfg, glist)
    I = np.sqrt((W ** 2).sum(axis=0))
    pct = np.argsort(np.argsort(I)) / (len(g) - 1) * 100.0
    found = [t for t in tf_syms if t in set(g)]
    idx = {x: i for i, x in enumerate(g)}
    v = np.array([pct[idx[t]] for t in found])
    return v, len(found), len(tf_syms)


hema_tfs = sorted(set(t.upper() for t in tf_db.keys()))
cort_tfs = sorted(set(NEURO_TFS))
for cfg, glist in [("R13", "hema_gene_list.json"), ("R14_seed42", "hema_gene_list.json"),
                   ("R1", "neural_gene_list.json"), ("R0_seed42", "neural_gene_list.json")]:
    vh, nh, th = tf_percentile(cfg, glist, hema_tfs)
    vc, nc, tc = tf_percentile(cfg, glist, cort_tfs)
    print(f"  [{cfg:14s}] hema TFs: n={nh}/{th} median pct={np.median(vh):5.1f} | "
          f"cortex TFs: n={nc}/{tc} median pct={np.median(vc):5.1f}")

# ================================================================ neural
print("\n=== NEURAL (cortex prior edges, TRRUST-derived) ===")
neu_rows, neu_d = [], {}
for tag, cfg in [("neural_prior", "R1"), ("neural_vanilla", "R0_seed42")]:
    r, ce, cb = run(tag, cfg, "neural_gene_list.json", neuro_pairs_sym, seed=2)
    neu_rows.append(r); neu_d[tag] = (ce, cb)
z, p = mw(neu_d["neural_prior"][0], neu_d["neural_vanilla"][0])
print(f"   neural_prior vs neural_vanilla: z={z:+.2f} p={p:.3g}")

# ================================================================ figure
fig = plt.figure(figsize=(11.0, 7.6))
gs = fig.add_gridspec(2, 2, height_ratios=[0.95, 1.0], hspace=0.52, wspace=0.30,
                      left=0.085, right=0.965, top=0.905, bottom=0.085)

# ---- Panel A: 2x3 decisive control (models x edge sets), z vs vanilla ----
ax = fig.add_subplot(gs[0, :])
cell_keys = [m for m, _ in marrow_models]
models3_disp = ["true\nprior (R13)", "wrong\nprior (R16)", "vanilla\n(R14)"]
esets = list(edge_sets.keys())
M = np.array([[cell[(m, e)][1] for e in esets] for m in cell_keys])
vmax = 4.5
im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
ax.set_xticks(range(len(esets)))
ax.set_xticklabels([e.replace(" GRN edges", "") for e in esets], fontsize=8.5)
ax.set_yticks(range(len(models3_disp)))
ax.set_yticklabels(models3_disp, fontsize=8.5)
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        z, p = cell[(cell_keys[i], esets[j])][1], cell[(cell_keys[i], esets[j])][2]
        star = "***" if p < 1e-3 else ("**" if p < 1e-2 else ("*" if p < 0.05 else "n.s."))
        ax.text(j, i, f"{z:+.2f}\n{star}", ha="center", va="center",
                fontsize=9, fontweight="bold",
                color="white" if abs(z) > vmax * 0.55 else "#222222")
ax.set_title("A. The prior is wired into the decoder only when it is tissue-matched "
             "(per-edge Mann–Whitney z vs vanilla, same edge set)", fontsize=9)
cb = fig.colorbar(im, ax=ax, fraction=0.012, pad=0.012)
cb.set_label("z vs vanilla", fontsize=8)
cb.ax.tick_params(labelsize=7)

# ---- Panel B: per-edge coupling distributions, hema edges, three marrow models ----
ax = fig.add_subplot(gs[1, 0])
bins = np.linspace(-0.6, 0.9, 46)
for tag, c, ls in [("hema_true", "#B2182B", "-"),
                   ("hema_wrong", "#D6604D", "--"),
                   ("hema_vanilla", "#777777", ":")]:
    ce, cb_ = hema_d[tag]
    ax.hist(ce, bins=bins, histtype="step", color=c, lw=1.5, ls=ls,
            density=True, label=tag.replace("hema_", ""))
ax.axvline(0, color="#333", lw=0.6)
ax.set_xlabel("coupling of centred loading profiles", fontsize=8)
ax.set_ylabel("density (per edge)", fontsize=8)
ax.set_title("B. Hematopoietic GRN edges, three marrow models", fontsize=9)
ax.legend(fontsize=7.5, frameon=False)
ax.grid(alpha=0.2, lw=0.4)

# ---- Panel C: cross-tissue replication (neural) ----
ax = fig.add_subplot(gs[1, 1])
zw, pw = mw(neu_d["neural_prior"][0], neu_d["neural_vanilla"][0])
for tag, c, ls in [("neural_prior", "#2166AC", "-"),
                   ("neural_vanilla", "#777777", ":")]:
    ce, cb_ = neu_d[tag]
    ax.hist(ce, bins=np.linspace(-0.7, 0.9, 40), histtype="step", color=c, lw=1.5,
            ls=ls, density=True, label=tag.replace("neural_", ""))
ax.axvline(0, color="#333", lw=0.6)
ax.set_xlabel("coupling of centred loading profiles", fontsize=8)
ax.set_ylabel("density (per edge)", fontsize=8)
ax.set_title(f"C. Neural cortex GRN edges (cross-tissue replication)\n"
             f"prior vs vanilla: z={zw:+.2f}, p={pw:.1e}", fontsize=9)
ax.legend(fontsize=7.5, frameon=False)
ax.grid(alpha=0.2, lw=0.4)

fig.savefig(os.path.join(OUT, "fig_prior_wiring_test.pdf"))
fig.savefig(os.path.join(OUT, "fig_prior_wiring_test.png"), dpi=200)
plt.close(fig)
print("\nwrote fig_prior_wiring_test.pdf/png")

# ================================================================ outputs
with open(os.path.join(OUT, "prior_wiring_test.csv"), "w", newline="") as f:
    keys = list(hema_rows[0].keys())
    wr = csv.DictWriter(f, fieldnames=keys); wr.writeheader()
    for r in hema_rows + neu_rows:
        wr.writerow(r)

md = ["# Is the injected GRN prior written into the decoder?\n",
      "## Statistic\n",
      "For each prior edge (TF -> target) we take the two genes' 64-dimensional",
      "decoder loading profiles `W[:,g]`, centre them **across the 64 latent axes**,",
      "L2-normalise, and take the cosine. Centring is essential: the raw columns share",
      "a dominant across-axis mean component (the decoder must reconstruct mean",
      "expression), so a raw cosine is near 1 for every pair and is uninformative.",
      "Null: random gene pairs from the same gene universe.\n",
      "## Hematopoietic (marrow; hema prior edges, same data & architecture)\n",
      "| condition | n edges | mean edge coupling | null mean +- sd | z | fold | % edges > null p95 |",
      "|---|---|---|---|---|---|---|"]
for r in hema_rows:
    md.append(f"| {r['label']} | {r['n_edges']} | {r['mean_edge_coupling']:+.5f} | "
              f"{r['mean_null_coupling']:+.5f} +- {r['null_sd']:.5f} | {r['z']:+.2f} | "
              f"{r['fold']:.3f} | {r['frac_edges_above_null95']*100:.1f}% |")
md += ["", "## Neural (cortex prior, TRRUST-derived edges)\n",
       "| condition | n edges | mean edge coupling | null mean +- sd | z | fold | % edges > null p95 |",
       "|---|---|---|---|---|---|---|"]
for r in neu_rows:
    md.append(f"| {r['label']} | {r['n_edges']} | {r['mean_edge_coupling']:+.5f} | "
              f"{r['mean_null_coupling']:+.5f} +- {r['null_sd']:.5f} | {r['z']:+.2f} | "
              f"{r['fold']:.3f} | {r['frac_edges_above_null95']*100:.1f}% |")
md += ["\n## Interpretation placeholder\n"]
open(os.path.join(OUT, "PRIOR_WIRING_TEST.md"), "w").write("\n".join(md))
print("wrote PRIOR_WIRING_TEST.md and prior_wiring_test.csv")
