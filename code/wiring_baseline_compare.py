# -*- coding: utf-8 -*-
"""Wiring-z baseline comparison after the vanilla-identity correction (2026-09-16).

The hema wiring test compares prior arm (R13 family) against a baseline.
`wiring_cross_seed.py` currently uses "R14" for seed 42 and "R14_seed{123,456}"
for the others:
    * before the run_registry "fix": bare "R14" fell through to
      decoder_out/R14  = the TRUE VANILLA  (tr 5634.8)
    * after  the run_registry "fix": bare "R14" -> R14_s42 = the VEHICLE (tr 5160.9)
    * after  OUR fix (2026-09-16) : bare "R14" -> the TRUE VANILLA again

So the registry silently converted  scenario A (seed-42 vanilla) into
scenario C (all-vehicle). This script quantifies both, plus the two
consistent alternatives (all-vanilla / all-vehicle).

Metric: exactly as wiring_cross_seed.py -- cross-axis column centring + unit
normalisation, per-edge dot product, two-sided Mann-Whitney z of model vs
baseline couplings over the same 347 hema edges; Stouffer across seeds.
"""
import os, sys, json
import numpy as np

CODE = r"F:/UNAGI_Project/paper1/paper1_kc_diagnostic/code"
sys.path.insert(0, CODE)
import run_registry as rr   # noqa: E402


def centred_unit(W):
    A = W - W.mean(axis=0, keepdims=True)
    n = np.linalg.norm(A, axis=0, keepdims=True)
    n[n == 0] = 1.0
    return A / n


def coupling(Ac, pairs):
    p = np.asarray(pairs, dtype=int)
    return np.einsum("ij,ij->i", Ac[:, p[:, 0]].T, Ac[:, p[:, 1]].T)


def mw_z_per_edge(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    allv = np.concatenate([a, b])
    order = np.argsort(allv, kind="mergesort")
    r = np.empty(allv.size); r[order] = np.arange(1, allv.size + 1).astype(float)
    Ra = r[:a.size].sum()
    U = Ra - a.size * (a.size + 1) / 2.0
    mu = a.size * b.size / 2.0
    sd = np.sqrt(a.size * b.size * (a.size + b.size + 1) / 12.0)
    return (U - mu) / sd


def stouffer(zz):
    zz = np.asarray(zz, float)
    return zz.sum() / np.sqrt(len(zz))


def stouffer_p(z):
    from math import erfc, sqrt
    return erfc(abs(z) / sqrt(2.0))


# ---- the 347 hema edges, exactly as wiring_cross_seed.py builds them ----
hp_path = None
for c in [r"F:/UNAGI_Project/next_phase/data/hema_prior.json",
          r"F:/UNAGI_Project/UNAGI/next_phase/data/hema_prior.json"]:
    if os.path.exists(c):
        hp_path = c
        break
assert hp_path, "hema_prior.json not found"
hp = json.load(open(hp_path))
edges = [(str(t).upper(), str(g).upper())
         for t, gs in hp.get("tf_db", {}).items() for g in gs]
print(f"hema_prior.json: {hp_path}")
print(f"edges: {len(edges)}  TFs: {len(hp.get('tf_db', {}))}")


def run(model_cfg, van_cfg, seed=None):
    Wm = rr.get_matrix(model_cfg); Wv = rr.get_matrix(van_cfg)
    assert Wm.shape == Wv.shape, f"{model_cfg} {Wm.shape} vs {van_cfg} {Wv.shape}"
    gm = rr.get_gene_index(Wm.shape[1])
    Acm = centred_unit(Wm); Acv = centred_unit(Wv)
    pairs = [(gm[a], gm[b]) for a, b in edges
             if a in gm and b in gm and gm[a] != gm[b]]
    p = np.asarray(pairs, dtype=int)
    ce_m = coupling(Acm, p); ce_v = coupling(Acv, p)
    return dict(seed=seed, model=model_cfg, vanilla=van_cfg, n=len(pairs),
                mean_m=float(ce_m.mean()), mean_v=float(ce_v.mean()),
                z=float(mw_z_per_edge(ce_m, ce_v)))


MODEL = {42: "R13", 123: "R13_seed123", 456: "R13_seed456"}
VAN_TRUE = "R14"                 # -> decoder_unified/vanilla/R14_hemato_vanilla (5634.8)
VAN_VEH = "R14_seed{seed}"       # -> R14_s{seed} = vehicle (5160.9 / 9109.9 / 14272.9)

SCEN = [
    ("A  seed42=vanilla, 123/456=vehicle  [pre-registry-fix AND post-our-fix]",
     lambda s: VAN_TRUE if s == 42 else VAN_VEH.format(seed=s)),
    ("B  ALL vanilla (loses seed-matching)",
     lambda s: VAN_TRUE),
    ("C  ALL vehicle  [what the registry 'fix' actually did]",
     lambda s: VAN_VEH.format(seed=s)),
]

out = {}
for label, pick in SCEN:
    rows = [run(MODEL[s], pick(s), s) for s in (42, 123, 456)]
    zs = [r["z"] for r in rows]
    zs_agg = stouffer(zs)
    out[label] = dict(rows=rows, stouffer_z=float(zs_agg),
                      stouffer_p=float(stouffer_p(zs_agg)),
                      signs=[int(z > 0) for z in zs])
    print()
    print("=" * 100)
    print(label)
    print("=" * 100)
    for r in rows:
        print(f"  seed {r['seed']:3d}: {r['model']:>12s} vs {r['vanilla']:<16s} "
              f"z={r['z']:+.3f} p={stouffer_p(r['z']):.3g}  "
              f"mean {r['mean_m']:+.4f} vs {r['mean_v']:+.4f}  n={r['n']}")
    print(f"  STOUFFER: z={zs_agg:+.3f} p={stouffer_p(zs_agg):.3g}  "
          f"signs={[int(z>0) for z in zs]}")

OUT = r"F:/UNAGI_Project/outputs/positive_result"
json.dump(out, open(os.path.join(OUT, "wiring_baseline_compare.json"), "w"),
          indent=1)
print(f"\nwrote {OUT}/wiring_baseline_compare.json")
