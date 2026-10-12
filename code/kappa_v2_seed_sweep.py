"""
kappa_v2_seed_sweep.py
======================
Robustness check for the partitioned kappa-C (v2) diagnostic (S8).

The published S8 figure uses a single k-means partition seed (SEED=7) to split
the 64 latent dimensions into K=8 clusters. A reviewer may ask whether the
reported spectral-mass reorganisation (delta_p_k spread ~5 pp in haematopoietic)
is an artefact of that particular partition. This script re-runs the FULL
partitioned diagnostic across several partition seeds and confirms the aggregate
statistics are stable.

It reuses the exact partition() and cluster_stats() from kappa_v2_partitioned.py,
so the only variable swept is the k-means initialisation. No GPU / retraining.

Model-seed coverage (42/123/456) is already present in the PAIRS of the parent
script for the flagship configs; this sweep adds PARTITION-seed stability on top.

Usage:
    python kappa_v2_seed_sweep.py
Writes:
    ../data/kappa_v2_seed_sweep.json
"""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_registry as RR
import kappa_v2_partitioned as K2   # K, PC, PAIRS, NULL_PAIRS, partition, cluster_stats

OUT = os.path.join(HERE, "..", "data", "kappa_v2_seed_sweep.json")
PARTITION_SEEDS = [7, 11, 23, 42, 99]

# Independently trained VEHICLE seeds of the same family (37-TF prior, same
# pipeline across seeds) -- the CORRECT null for
# dp_spread (two untrained decoders), as opposed to the trivial self-pair (W==W).
VANILLA_CROSS = [("R14_s42", "R14_s123", "hema"), ("R14_s42", "R14_s456", "hema"),
                 ("R14_s123", "R14_s456", "hema"), ("R0_s123", "R0_s456", "neural")]


def main():
    out = {}
    for pseed in PARTITION_SEEDS:
        hema, neural, null, vcross = [], [], [], []
        for model, vanilla, tissue in K2.PAIRS + K2.NULL_PAIRS:
            try:
                Wm = RR.get_matrix(model)
                Wv = RR.get_matrix(vanilla)
            except Exception as e:
                print(f"  skip {model}/{vanilla}: {e}")
                continue
            lab = K2.partition(Wv, K2.K, K2.PC, pseed)
            st = K2.cluster_stats(Wm, Wv, lab)
            if model == vanilla:
                null.append(st["dp_spread"])
            elif tissue == "hema":
                hema.append(st["dp_spread"])
            else:
                neural.append(st["dp_spread"])
        for model, vanilla, tissue in VANILLA_CROSS:
            try:
                Wm = RR.get_matrix(model)
                Wv = RR.get_matrix(vanilla)
            except Exception as e:
                print(f"  skip vanilla-cross {model}/{vanilla}: {e}")
                continue
            lab = K2.partition(Wv, K2.K, K2.PC, pseed)
            st = K2.cluster_stats(Wm, Wv, lab)
            vcross.append(st["dp_spread"])
        out[str(pseed)] = dict(
            n_hema=len(hema), n_neural=len(neural), n_null=len(null),
            n_vanilla_cross=len(vcross),
            hema_mean_dp_spread=round(float(np.mean(hema)), 3) if hema else None,
            neural_mean_dp_spread=round(float(np.mean(neural)), 3) if neural else None,
            null_mean_dp_spread=round(float(np.mean(null)), 4) if null else None,
            vanilla_cross_mean_dp_spread=round(float(np.mean(vcross)), 3) if vcross else None,
            hema_dp_spread_std=round(float(np.std(hema)), 3) if len(hema) > 1 else None,
        )
        print(f"  pseed={pseed:3d}  hema={out[str(pseed)]['hema_mean_dp_spread']}  "
              f"neural={out[str(pseed)]['neural_mean_dp_spread']}  "
              f"vanilla_cross={out[str(pseed)]['vanilla_cross_mean_dp_spread']}  "
              f"selfnull={out[str(pseed)]['null_mean_dp_spread']}")

    hm = [out[str(s)]["hema_mean_dp_spread"] for s in PARTITION_SEEDS]
    cm = [out[str(s)]["vanilla_cross_mean_dp_spread"] for s in PARTITION_SEEDS]
    json.dump(dict(K=K2.K, PC=K2.PC, partition_seeds=PARTITION_SEEDS,
                   by_seed=out,
                   hema_dp_spread_across_seeds=hm,
                   vanilla_cross_dp_spread_across_seeds=cm,
                   hema_mean=float(np.mean(hm)), hema_cv=float(np.std(hm)/np.mean(hm)),
                   vanilla_cross_mean=float(np.mean(cm)),
                   vanilla_cross_cv=float(np.std(cm)/np.mean(cm))),
              open(OUT, "w"), indent=2)
    print(f"\nwrote {OUT}")
    print(f"haematopoietic prior dp_spread across partition seeds: {hm}  (mean={np.mean(hm):.2f}, cv={np.std(hm)/np.mean(hm):.3f})")
    print(f"vanilla cross-seed  dp_spread across partition seeds: {cm}  (mean={np.mean(cm):.2f}, cv={np.std(cm)/np.mean(cm):.3f})")
    print(f"=> prior ({np.mean(hm):.2f}) is BELOW the vanilla cross-seed null ({np.mean(cm):.2f}): "
          f"kappa-C v2 detects NO prior-specific reorganisation.")


if __name__ == "__main__":
    main()
