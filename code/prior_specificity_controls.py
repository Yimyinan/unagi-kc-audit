"""
prior_specificity_controls.py
=============================
Three independent controls that decide whether the "prior wiring" signal is
actually about the prior's CONTENT, plus a causal test.

Why this script exists
----------------------
`dose_response_wiring.py` showed two uncomfortable facts:

  1. no dose-response (Spearman rho = -0.20, p = 0.51), and
  2. once the null keeps the TF end fixed and randomises only the target, the
     prior-edge specificity of R13/seed42 collapses from z = +4.16 (raw) to
     +0.53 (null-corrected).

A raw z of +4.16 therefore partly reflects that prior TFs are hub genes, not
that the model wired the specific TF->target pairs it was told about. To settle
what the signal IS, we run four controls:

  A. MISMATCH PRIOR (seed-matched): R16 injects the *neural* prior into
     haematopoietic data. Tested on BOTH edge sets. If the signal is about
     prior content, R16 should score low on haematopoietic edges even though it
     is otherwise an identically-trained model at the same seed.

  B. CROSS-TISSUE: the neural arm (R1 prior vs R0 vanilla, 3 seeds) evaluated
     on TRRUST-derived cortex edges. Independent tissue, independent prior.

  C. SEED PERMUTATION: is the complete sign separation by seed (42/123
     positive, 456 negative) larger than expected under seed-label permutation?

  D. CAUSAL PERTURBATION: R13_KO_KLF1 and R13_OE_KLF1 are in-silico knock-out
     and over-expression of KLF1 applied to the SAME R13 baseline. If the
     decoder genuinely encodes KLF1->target wiring, KLF1-edge coupling should
     move DOWN under KO and UP under OE. This is the only control that is
     causal rather than correlational.

Outputs
-------
  outputs/positive_result/fig_prior_specificity.{png,pdf}
  outputs/positive_result/PRIOR_SPECIFICITY.md
  outputs/positive_result/prior_specificity.csv
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
TRRUST = "F:/UNAGI_Project/unagi_custom_data/trrust_raw_human.txt"
OUT = "F:/UNAGI_Project/outputs/positive_result"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({"font.size": 8.5, "figure.dpi": 160,
                     "axes.spines.top": False, "axes.spines.right": False})
RNG = np.random.default_rng(20260904)
N_NULL = 2000

NEURO_TFS = """SOX2 PAX6 HES1 HES5 VIM PROM1 SOX9 EOMES NEUROG1 NEUROG2 ASCL1
MKI67 CCND1 TBR1 BCL11B FEZF2 SOX5 CPLX3 CPLX4 SATB2 CUX1 CUX2 POU3F2 POU3F3
MEF2C DLX1 DLX2 DLX5 DLX6 GAD1 GAD2 LHX6 NKX2-1 SST PVALB VIP CALB2 GFAP S100B
AQP4 OLIG1 OLIG2 PLP1 MBP ALDH1L1""".split()


# --------------------------------------------------------------- helpers
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


def zc_score(Am, Av, pairs, genes, n_null=N_NULL):
    """delta, raw z, and null-corrected z (TF end fixed, target randomised
    over non-target genes)."""
    cm = coupling(Am, pairs); cv = coupling(Av, pairs)
    d = float(cm.mean() - cv.mean()); z = float(mw_z(cm, cv))
    tf_idx = pairs[:, 0]
    pool = genes[~np.isin(genes, np.unique(pairs[:, 1]))]
    zs = np.empty(n_null); ds = np.empty(n_null)
    for i in range(n_null):
        b = RNG.choice(pool, len(pairs))
        rp = np.stack([tf_idx, b], 1); rp = rp[rp[:, 0] != rp[:, 1]]
        c1 = coupling(Am, rp); c2 = coupling(Av, rp)
        zs[i] = mw_z(c1, c2); ds[i] = c1.mean() - c2.mean()
    return d, z, float((z - zs.mean()) / (zs.std() + 1e-12)), \
        float((d - ds.mean()) / (ds.std() + 1e-12))


def pairify(edges, idx):
    out = set()
    for a, b in edges:
        a, b = a.upper(), b.upper()
        if a in idx and b in idx and a != b:
            out.add((idx[a], idx[b]))
    return np.array(sorted(out), dtype=int)


# --------------------------------------------------------------- edge sets
trrust = {}
with open(TRRUST, encoding="utf-8", errors="ignore") as f:
    for line in f:
        pp = line.rstrip("\n").split("\t")
        if len(pp) >= 3:
            trrust.setdefault(pp[0].upper(), set()).add(pp[1].upper())

CONF = json.load(open(os.path.join(OUT, "prior_configs.json")))
HEMA_EDGES = [(t.upper(), g.upper()) for t, g in CONF["R13"]["edges"]]
NEURAL_EDGES = [(t, g) for t in NEURO_TFS for g in trrust.get(t.upper(), ())]

print("=" * 78)
print("A. MISMATCH PRIOR CONTROL  (R16 = neural prior injected into blood)")
print("=" * 78)
mismatch = []
# spec "B": baseline = the TRUE VANILLA (not R14_seed42 = the VEHICLE arm).
for cfg, van, seed in [("R16", "R14", 42), ("R13", "R14", 42)]:
    Wm = load(cfg); Wv = load(van)
    if Wm.shape != Wv.shape:
        print(f"  skip {cfg}: shape {Wm.shape} vs {Wv.shape}"); continue
    idx = genes_for(Wm.shape[1]); genes = np.array(sorted(idx.values()))
    Am, Av = centred_unit(Wm), centred_unit(Wv)
    for ename, edges in [("hema", HEMA_EDGES), ("cortex", NEURAL_EDGES)]:
        p = pairify(edges, idx)
        if len(p) < 10:
            print(f"  {cfg} {ename}: only {len(p)} edges, skipped"); continue
        d, z, zc, dc = zc_score(Am, Av, p, genes)
        mismatch.append(dict(config=cfg, seed=seed, edgeset=ename,
                             n=len(p), delta=d, z=z, z_nullcorr=zc, delta_nullcorr=dc))
        print(f"  {cfg:5s} s{seed} {ename:7s} n={len(p):4d}  delta={d:+.4f} "
              f"z={z:+.2f}  null-corr z={zc:+.2f} (delta {dc:+.2f})")

print("\n" + "=" * 78)
print("B. CROSS-TISSUE (neural arm: R1 cortex prior vs R0 vanilla)")
print("=" * 78)
neural_rows = []
for cfg, van, seed in [("R1", "R0_seed42", 42),
                       ("R1_seed123", "R0_seed123", 123),
                       ("R1_seed456", "R0_seed456", 456)]:
    try:
        Wm = load(cfg); Wv = load(van)
    except FileNotFoundError as e:
        print(f"  skip {cfg}: {e}"); continue
    if Wm.shape != Wv.shape:
        print(f"  skip {cfg}: shape mismatch"); continue
    idx = genes_for(Wm.shape[1]); genes = np.array(sorted(idx.values()))
    Am, Av = centred_unit(Wm), centred_unit(Wv)
    p = pairify(NEURAL_EDGES, idx)
    d, z, zc, dc = zc_score(Am, Av, p, genes)
    neural_rows.append(dict(config=cfg, seed=seed, n=len(p),
                            delta=d, z=z, z_nullcorr=zc, delta_nullcorr=dc))
    print(f"  {cfg:12s} s{seed} n={len(p):4d}  delta={d:+.4f} z={z:+.2f} "
          f"null-corr z={zc:+.2f} (delta {dc:+.2f})")

print("\n" + "=" * 78)
print("C. SEED PERMUTATION TEST on the dose-response deltas")
print("=" * 78)
dr = list(csv.DictReader(open(os.path.join(OUT, "dose_response_wiring.csv"))))
delta = np.array([float(r["delta"]) for r in dr])
seedlab = np.array([int(r["seed"]) for r in dr])
np.random.seed(0)


def between_seed_var(y, lab):
    tot = y.var() * len(y)
    wit = sum(((y[lab == s] - y[lab == s].mean()) ** 2).sum() for s in set(lab))
    return (tot - wit) / tot


obs = between_seed_var(delta, seedlab)
null = np.array([between_seed_var(delta, np.random.permutation(seedlab))
                 for _ in range(20000)])
p_seed = float((null >= obs).mean())
print(f"  fraction of variance explained by seed = {obs:.3f}")
print(f"  permutation p = {p_seed:.4f}  (20000 permutations)")
print(f"  seed means: " + ", ".join(
    f"s{s}={delta[seedlab == s].mean():+.4f}" for s in [42, 123, 456]))

print("\n" + "=" * 78)
print("D. CAUSAL TEST: in-silico KLF1 knock-out / over-expression on R13")
print("=" * 78)
Wbase = load("R13")
idx = genes_for(Wbase.shape[1]); genes = np.array(sorted(idx.values()))
Abase = centred_unit(Wbase)
KLF1_EDGES = [(t.upper(), g.upper()) for t, g in CONF["R13"]["edges"]
              if t.upper() == "KLF1"]
p_klf1 = pairify(KLF1_EDGES, idx)
p_other = pairify([(t, g) for t, g in CONF["R13"]["edges"] if t.upper() != "KLF1"], idx)
print(f"  KLF1 edges n={len(p_klf1)}; other-prior edges n={len(p_other)}")

causal = {}
base_k = float(coupling(Abase, p_klf1).mean())
base_o = float(coupling(Abase, p_other).mean())
print(f"  baseline R13: KLF1-edge cosine={base_k:+.4f}  other={base_o:+.4f}")
for cfg in ["R13_KO_KLF1", "R13_OE_KLF1"]:
    W = load(cfg)
    if W.shape != Wbase.shape:
        print(f"  skip {cfg}: shape mismatch"); continue
    A = centred_unit(W)
    ck = float(coupling(A, p_klf1).mean())
    co = float(coupling(A, p_other).mean())
    causal[cfg] = dict(klf1=ck, other=co,
                       d_klf1=ck - base_k, d_other=co - base_o)
    print(f"  {cfg:12s}: KLF1-edge={ck:+.4f} (d={ck-base_k:+.4f})   "
          f"other={co:+.4f} (d={co-base_o:+.4f})")

# is the KLF1-edge shift bigger than the rest of the prior's shift?
if "R13_KO_KLF1" in causal and "R13_OE_KLF1" in causal:
    dk_ko = causal["R13_KO_KLF1"]["d_klf1"]; do_ko = causal["R13_KO_KLF1"]["d_other"]
    dk_oe = causal["R13_OE_KLF1"]["d_klf1"]; do_oe = causal["R13_OE_KLF1"]["d_other"]
    print(f"\n  KO: KLF1 shift {dk_ko:+.4f} vs rest-of-prior shift {do_ko:+.4f} "
          f"-> difference {dk_ko-do_ko:+.4f}")
    print(f"  OE: KLF1 shift {dk_oe:+.4f} vs rest-of-prior shift {do_oe:+.4f} "
          f"-> difference {dk_oe-do_oe:+.4f}")
    directional = (dk_ko - do_ko) < 0 < (dk_oe - do_oe)
    print(f"  KO moves KLF1 edges DOWN and OE moves them UP, relative to the "
          f"rest of the prior: {directional}")

# --------------------------------------------------------------- figure
fig, axes = plt.subplots(1, 4, figsize=(15.6, 3.7))

ax = axes[0]
mm = {(r["config"], r["edgeset"]): r for r in mismatch}
labels = ["R13\nhema", "R13\ncortex", "R16\nhema", "R16\ncortex"]
vals, cols = [], []
for cfg, es, c in [("R13", "hema", "#B2182B"), ("R13", "cortex", "#F0A0A0"),
                   ("R16", "hema", "#2166AC"), ("R16", "cortex", "#A8C8E8")]:
    r = mm.get((cfg, es))
    vals.append(r["delta"] if r else 0.0); cols.append(c)
ax.bar(range(4), vals, color=cols, edgecolor="#333", lw=0.5)
ax.axhline(0, color="#333", lw=0.6)
ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=7)
ax.set_ylabel(r"$\Delta$edge cosine vs vanilla", fontsize=8)
ax.set_title("A. Matched-seed mismatch control\nR16 = neural prior in blood", fontsize=8)
ax.grid(axis="y", alpha=0.25, lw=0.4)

ax = axes[1]
if neural_rows:
    ax.bar(range(len(neural_rows)), [r["delta"] for r in neural_rows],
           color="#2166AC", edgecolor="#333", lw=0.5)
    ax.axhline(0, color="#333", lw=0.6)
    ax.set_xticks(range(len(neural_rows)))
    ax.set_xticklabels([f"s{r['seed']}" for r in neural_rows], fontsize=7.5)
ax.set_ylabel(r"$\Delta$edge cosine vs vanilla", fontsize=8)
ax.set_title("B. Neural arm, cortex edges\nindependent tissue & prior", fontsize=8)
ax.grid(axis="y", alpha=0.25, lw=0.4)

ax = axes[2]
for s, c in [(42, "#B2182B"), (123, "#2166AC"), (456, "#4D9221")]:
    m = seedlab == s
    ax.scatter(np.full(m.sum(), s), delta[m], color=c, s=32, zorder=3)
ax.axhline(0, color="#333", lw=0.6)
ax.set_xticks([42, 123, 456]); ax.set_xticklabels(["42", "123", "456"])
ax.set_xlabel("seed", fontsize=8)
ax.set_ylabel(r"$\Delta$edge cosine", fontsize=8)
ax.set_title(f"C. Seed explains {obs*100:.0f}% of variance\n"
             f"permutation p={p_seed:.3f}", fontsize=8)
ax.grid(alpha=0.25, lw=0.4)

ax = axes[3]
if causal:
    ks = ["R13_KO_KLF1", "R13_OE_KLF1"]
    x = np.arange(2); w = 0.38
    ax.bar(x - w/2, [causal[k]["d_klf1"] for k in ks], w,
           color="#B2182B", label="KLF1 edges", edgecolor="#333", lw=0.5)
    ax.bar(x + w/2, [causal[k]["d_other"] for k in ks], w,
           color="#999999", label="rest of prior", edgecolor="#333", lw=0.5)
    ax.axhline(0, color="#333", lw=0.6)
    ax.set_xticks(x); ax.set_xticklabels(["KO KLF1", "OE KLF1"], fontsize=8)
    ax.legend(fontsize=7, frameon=False)
ax.set_ylabel(r"$\Delta$cosine vs R13 baseline", fontsize=8)
ax.set_title("D. Causal perturbation of one TF", fontsize=8)
ax.grid(axis="y", alpha=0.25, lw=0.4)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_prior_specificity.pdf"))
fig.savefig(os.path.join(OUT, "fig_prior_specificity.png"), dpi=200)
plt.close(fig)
print("\nwrote fig_prior_specificity.pdf/png")

with open(os.path.join(OUT, "prior_specificity.csv"), "w", newline="") as f:
    fields = ["kind", "config", "seed", "edgeset", "n", "delta", "z",
              "z_nullcorr", "delta_nullcorr"]
    wr = csv.DictWriter(f, fieldnames=fields); wr.writeheader()
    for r in mismatch:
        wr.writerow(dict(kind="mismatch", edgeset=r["edgeset"], **{k: r[k] for k in
                    ["config", "seed", "n", "delta", "z", "z_nullcorr", "delta_nullcorr"]}))
    for r in neural_rows:
        wr.writerow(dict(kind="neural", edgeset="cortex", **{k: r[k] for k in
                    ["config", "seed", "n", "delta", "z", "z_nullcorr", "delta_nullcorr"]}))
    for k, v in causal.items():
        wr.writerow(dict(kind="causal", config=k, edgeset="KLF1", n=len(p_klf1),
                         delta=v["d_klf1"], z="", z_nullcorr="", delta_nullcorr=""))
        wr.writerow(dict(kind="causal", config=k, edgeset="other_prior", n=len(p_other),
                         delta=v["d_other"], z="", z_nullcorr="", delta_nullcorr=""))

lines = ["# Prior-specificity controls", "",
         "## A. Mismatch (R16: neural prior injected into blood, seed 42)", "",
         "| model | edge set | n | delta | z | null-corr z |", "|---|---|---|---|---|---|"]
for r in mismatch:
    lines.append(f"| {r['config']} | {r['edgeset']} | {r['n']} | {r['delta']:+.4f} | "
                 f"{r['z']:+.2f} | {r['z_nullcorr']:+.2f} |")
lines += ["", "## B. Neural arm (cortex edges)", "",
          "| model | seed | n | delta | z | null-corr z |", "|---|---|---|---|---|---|"]
for r in neural_rows:
    lines.append(f"| {r['config']} | {r['seed']} | {r['n']} | {r['delta']:+.4f} | "
                 f"{r['z']:+.2f} | {r['z_nullcorr']:+.2f} |")
lines += ["", "## C. Seed permutation on the dose-response deltas", "",
          f"Seed explains **{obs*100:.1f}%** of the variance in delta-cosine; "
          f"permutation p = **{p_seed:.4f}** (20000 permutations).", "",
          "## D. Causal perturbation of KLF1 (in-silico, on the R13 baseline)", "",
          "| perturbation | KLF1-edge shift | rest-of-prior shift | difference |",
          "|---|---|---|---|"]
for k, v in causal.items():
    lines.append(f"| {k} | {v['d_klf1']:+.4f} | {v['d_other']:+.4f} | "
                 f"{v['d_klf1']-v['d_other']:+.4f} |")
open(os.path.join(OUT, "PRIOR_SPECIFICITY.md"), "w").write("\n".join(lines))
print("wrote PRIOR_SPECIFICITY.md and prior_specificity.csv")
