"""
kappa_v2_vanilla_null.py
=========================
The published kappa-C v2 (S8) reports a mean spectral-mass reorganisation of
~5 percentage points (dp_spread = std of delta_p_k across K=8 partitions) for
prior--vanilla pairs, and uses the TRIVIAL self-pair null (W_model == W_vanilla
-> delta_p_k === 0) as its baseline. A trivial self-null cannot tell us whether
5 pp is distinguishable from ordinary seed-to-seed variation of an untrained
decoder.

This script builds the CORRECT null: pairs of TWO INDEPENDENTLY TRAINED vanilla
seeds of the same model family (hematopoietic R14 / neural R0, seeds 42/123/456)
and computes the same dp_spread. If vanilla cross-seed dp_spread is comparable to
the ~5 pp prior-pair value, then the "local reorganisation" is within seed noise
and kappa-C v2 does NOT isolate a prior-specific effect -- the honest conclusion
becomes "no prior effect at any level". If vanilla cross-seed dp_spread is
substantially smaller, the v2 local-reorganisation claim stands.

Reuses partition()/cluster_stats() from kappa_v2_partitioned.py (SEED=7 to match
the published S8 figure). No GPU / retraining.

Usage:
    python kappa_v2_vanilla_null.py
Writes:
    ../data/kappa_v2_vanilla_null.json
"""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_registry as RR
import kappa_v2_partitioned as K2

OUT = os.path.join(HERE, "..", "data", "kappa_v2_vanilla_null.json")

# Independently trained vanilla seeds of the same family.
# NOTE (2026-09-16): R14_s42 / R14_s123 / R14_s456 are the VEHICLE arm
# (37-TF prior / 1054 genes), NOT empty priors.  They are retained HERE as a
# seed-replicate null (same pipeline, seed-only variation) -- which is what
# this script needs, and the vanilla has only one seed so it cannot form a
# cross pair.  The spacing is deliberately the same as kappa-v2 PAIRS.
# Neural vanilla:            R0_s123 / R0_s456               (R0_s42 legacy, unmapped)
VANILLA_CROSS = {
    "hema": [("R14_s42", "R14_s123"), ("R14_s42", "R14_s456"), ("R14_s123", "R14_s456")],
    "neural": [("R0_s123", "R0_s456")],
}


def dp_spread(model, vanilla, pseed=K2.SEED):
    Wm = RR.get_matrix(model)
    Wv = RR.get_matrix(vanilla)
    lab = K2.partition(Wv, K2.K, K2.PC, pseed)
    st = K2.cluster_stats(Wm, Wv, lab)
    return st["dp_spread"], st["delta_p_k"]


def main():
    out = {"K": K2.K, "PC": K2.PC, "pseed": K2.SEED,
           "vanilla_cross": {}, "prior_pair": {}, "self_null": {}}

    for tissue, pairs in VANILLA_CROSS.items():
        vals, per_pair = [], []
        for m, v in pairs:
            try:
                d, dk = dp_spread(m, v)
            except Exception as e:
                print(f"  skip {m}/{v}: {e}")
                continue
            vals.append(d)
            per_pair.append(dict(pair=f"{m}|{v}", dp_spread=round(d, 3),
                                 delta_p_k=[round(x, 2) for x in dk]))
            print(f"  VANILLA {tissue:6s} {m:10s}/{v:10s}  dp_spread={d:6.3f}")
        out["vanilla_cross"][tissue] = dict(
            n=len(vals),
            dp_spread=vals,
            mean=round(float(np.mean(vals)), 3) if vals else None,
            std=round(float(np.std(vals)), 3) if len(vals) > 1 else None,
            per_pair=per_pair)

    # Recompute prior-pair dp_spread with the SAME partition seed (SEED=7) for
    # an apples-to-apples comparison against the vanilla cross-seed null.
    for tissue in ("hema", "neural"):
        grp = [p for p in K2.PAIRS if p[2] == tissue]
        vals, per_pair = [], []
        for m, v, _ in grp:
            try:
                d, dk = dp_spread(m, v)
            except Exception as e:
                print(f"  skip prior {m}/{v}: {e}")
                continue
            vals.append(d)
            per_pair.append(dict(pair=f"{m}|{v}", dp_spread=round(d, 3)))
        out["prior_pair"][tissue] = dict(
            n=len(vals),
            dp_spread=vals,
            mean=round(float(np.mean(vals)), 3) if vals else None,
            std=round(float(np.std(vals)), 3) if len(vals) > 1 else None,
            per_pair=per_pair)
        print(f"  PRIOR   {tissue:6s}  mean dp_spread={np.mean(vals):.3f}  (n={len(vals)})")

    # Verdict
    hv = out["vanilla_cross"]["hema"]["mean"]
    hp = out["prior_pair"]["hema"]["mean"]
    nv = out["vanilla_cross"]["neural"]["mean"]
    np_ = out["prior_pair"]["neural"]["mean"]
    out["verdict"] = dict(
        hema_vanilla_mean=hv, hema_prior_mean=hp,
        hema_ratio=(hp / hv) if hv else None,
        neural_vanilla_mean=nv, neural_prior_mean=np_,
        note=("If vanilla_cross_mean is within ~2x of prior_mean, the kappa-C v2 "
              "'local reorganisation' is indistinguishable from seed noise and the "
              "honest conclusion is NO prior effect at any level."))
    json.dump(out, open(OUT, "w"), indent=2)
    print(f"\nwrote {OUT}")
    print(f"  hema:   vanilla_cross={hv}  prior={hp}  ratio={hp/hv if hv else 'NA'}")
    print(f"  neural: vanilla_cross={nv}  prior={np_}")


if __name__ == "__main__":
    main()
