# -*- coding: utf-8 -*-
"""Merge the injected-arm shards and form the paired design against the baseline fits.

The published Norman pair is one run per arm. `s1n_norman_null.py` gave the baseline
(null) side; `s1n3_norman_injected_band.py` gives the injected side at the matching
seeds. This script puts them together:

  paired ratio(s) = metric_injected(s) / metric_vanilla(s)

which cancels the baseline scatter entirely and yields the standard error of the
*effect*, not of a single fit. Both the paired mean and the published single pair are
then reported, so the manuscript can say whether the published run represents the
configuration -- the same question that was asked of the family-4 pair, where the
answer turned out to be "no" for the effective-rank ratio.

Reads  : outputs/positive_result/s1n3_*.jsonl, s1n_norman_null.jsonl
Writes : outputs/positive_result/s1n3_norman_injected_paired.json
Usage  : python code/s1n4_injected_paired_summary.py
"""
from __future__ import annotations

import glob
import io
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER1 = os.path.abspath(os.path.join(HERE, ".."))
ROOT = os.path.abspath(os.path.join(PAPER1, "..", ".."))
OUT = os.path.join(ROOT, "outputs", "positive_result")

PAIRS = [("kappa_C_W_corrected", "sum_r2", "sum_r2"),
         ("kappa_C_W_trace", "trace", "trace"),
         ("kappa_C_Z", "trace_Z", "trace_Z"),
         ("eff_rank_ratio", "eff_rank", "eff_rank")]


def load_jsonl(p):
    if not os.path.exists(p):
        return []
    with io.open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def desc(v):
    v = np.asarray(v, float)
    if v.size == 0:
        return dict(n=0)
    if v.size == 1:
        return dict(n=1, mean=float(v[0]), sd=None)
    return dict(n=int(v.size), mean=float(v.mean()), sd=float(v.std(ddof=1)),
                min=float(v.min()), max=float(v.max()),
                t_vs_1=float((v.mean() - 1.0) / (v.std(ddof=1) / np.sqrt(v.size))))


def main():
    inj = []
    for p in sorted(glob.glob(os.path.join(OUT, "s1n3_*.jsonl"))):
        rows = [r for r in load_jsonl(p) if r.get("arm") == "injected"]
        print(f"  shard {os.path.basename(p)}: {len(rows)} injected arms "
              f"{[r['seed'] for r in rows]}")
        inj += rows

    van = {r["seed"]: r for r in load_jsonl(os.path.join(OUT, "s1n_norman_null.jsonl"))
           if r.get("arm") == "vanilla" and r.get("protocol", "single") == "single"}
    print(f"  vanilla baselines (single protocol): {sorted(van)}")

    per_seed = []
    for r in sorted(inj, key=lambda x: x["seed"]):
        v = van.get(r["seed"])
        if v is None:
            print(f"    seed {r['seed']}: NO matched vanilla -- excluded")
            continue
        if not r.get("z_finite"):
            print(f"    seed {r['seed']}: latent not finite -- excluded")
            continue
        rec = dict(seed=r["seed"], train_status=r["train_status"], seconds=r["seconds"])
        for name, ik, vk in PAIRS:
            rec[name] = r[ik] / v[vk]
        rec["D"] = rec["kappa_C_Z"] - rec["kappa_C_W_corrected"]
        per_seed.append(rec)

    completed = [r for r in per_seed if r["train_status"].startswith("completed")]

    out = dict(n_injected_arms=len(inj), n_paired=len(per_seed),
               n_paired_completed=len(completed), per_seed=per_seed,
               dropped=[dict(seed=r["seed"], status=r["train_status"])
                        for r in inj if r["seed"] not in {p["seed"] for p in per_seed}])
    for tag, per in (("all", per_seed), ("completed", completed)):
        out[f"paired_{tag}"] = {name: desc([r[name] for r in per])
                                for name, _, _ in PAIRS}
        D = np.array([r["D"] for r in per], float)
        out[f"paired_{tag}"]["D"] = dict(
            n=int(D.size), mean=float(D.mean()) if D.size else float("nan"),
            sd=float(D.std(ddof=1)) if D.size > 1 else None,
            t_vs_0=float(D.mean() / (D.std(ddof=1) / np.sqrt(D.size)))
            if D.size > 1 and D.std(ddof=1) > 0 else None)

    # published single pair, recomputed from the cache
    import sys
    sys.path.insert(0, os.path.join(PAPER1, "kappa_c_audit"))
    from kappa_c_audit import predictability_total as pt
    from kappa_c_audit import ridge_readout
    c = np.load(os.path.join(PAPER1, "data", "_norman_latents.npz"), allow_pickle=True)
    X = c["X"].astype(np.float64)
    Zv, Zp = c["Zv"].astype(np.float64), c["Zp"].astype(np.float64)
    cov = lambda Z: np.cov(Z, rowvar=False)
    er = lambda Z: float(np.trace(cov(Z)) ** 2 / (cov(Z) ** 2).sum())
    tr = lambda Z: float((ridge_readout(Z, X) ** 2).sum())
    pub = dict(kappa_C_W_corrected=float(pt(Zp, X) / pt(Zv, X)),
               kappa_C_W_trace=float(tr(Zp) / tr(Zv)),
               kappa_C_Z=float(np.trace(cov(Zp)) / np.trace(cov(Zv))),
               eff_rank_ratio=float(er(Zp) / er(Zv)))
    pub["D"] = pub["kappa_C_Z"] - pub["kappa_C_W_corrected"]
    out["published_single_pair"] = pub

    print("\n=== paired design: same seed, injected / vanilla ===")
    print(f"{'metric':<24}{'paired mean':>14}{'sd':>9}{'n':>4}{'t vs 1':>9}"
          f"{'published':>12}{'z vs paired':>13}")
    for name, _, _ in PAIRS + [("D", "", "")]:
        d = out["paired_completed"].get(name) or out["paired_all"].get(name)
        p = pub.get(name)
        if not d or d.get("n", 0) == 0:
            continue
        if d.get("sd") and p is not None:
            z = (p - d["mean"]) / d["sd"]
            zs = f"{z:+13.2f}"
        else:
            zs = f"{'-':>13}"
        tt = (f"{d.get('t_vs_1', d.get('t_vs_0')):+9.2f}"
              if (d.get("t_vs_1") or d.get("t_vs_0")) is not None else f"{'-':>9}")
        sds = f"{d['sd']:9.4f}" if d.get("sd") else f"{'-':>9}"
        pub_s = f"{p:12.4f}" if p is not None else f"{'-':>12}"
        print(f"{name:<24}{d['mean']:14.4f}{sds}{d['n']:4d}{tt}{pub_s}{zs}")

    print("\n  per seed:")
    for r in per_seed:
        print(f"    seed {r['seed']}  kappa_W(corr) {r['kappa_C_W_corrected']:.4f}  "
              f"kappa_Z {r['kappa_C_Z']:.4f}  eff_rank ratio {r['eff_rank_ratio']:.4f}  "
              f"D {r['D']:+.4f}  [{r['train_status'][:34]}]")
    if out["dropped"]:
        print("\n  dropped:")
        for d in out["dropped"]:
            print(f"    seed {d['seed']}: {d['status'][:90]}")

    with io.open(os.path.join(OUT, "s1n3_norman_injected_paired.json"), "w",
                 encoding="utf-8", newline="\n") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"\nwrote {os.path.join(OUT, 's1n3_norman_injected_paired.json')}")


if __name__ == "__main__":
    main()
