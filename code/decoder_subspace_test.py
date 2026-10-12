"""A2: does prior injection change the decoder readout GEOMETRY (its subspace)?

WHY THIS TEST
-------------
The audit's headline null (0/33) is a statement about the trace RATIO, i.e. about
scale.  The review's open objection is that the paper never directly tested whether
the decoder readout *subspace* (its geometry) differs between prior-injected and
vanilla models beyond seed variation.  kappa-C(W) is scale-normalised and therefore
blind to a pure re-orientation of that subspace.

WHAT IT DOES
------------
For each audit readout W (D x G, rows = latent directions in gene space) we
orthonormalise the rows and compare subspaces via principal angles.  Overlap is
    overlap(W1, W2) = ||Q1^T Q2||_F^2 / D  in [0, 1]      (1 = identical subspace)
which is invariant to per-row scaling -- i.e. purely geometric.

Two distributions are then compared per tissue:
  * control-vs-control  (null: seed variation of the SAME construction)
  * prior-vs-control    (does the prior move the subspace?)
A rank test on the two distributions says whether prior injection perturbs the
readout subspace more than re-training alone.

WHAT THE "CONTROL" READOUTS ACTUALLY ARE (corrected 2026-09-16 -- the previous
revision called them "vanilla", which is wrong for the hematopoietic tissue):
    R0_s123 / R0_s456   = neural vanilla readouts (genuinely no prior machinery)
    R14_s* / R14_seed*  = the hematopoietic VEHICLE arm -- a 37-TF prior DB with
                          1054 boosted genes, i.e. NOT an empty prior.
The true hematopoietic vanilla (decoder_unified/vanilla/R14_hemato_vanilla,
tr 5634.8) is NOT in this directory, and it has only one seed, so it cannot form
a cross-seed control pair at all.  The hematopoietic control distribution here is
therefore a *seed-replicate* null within a single prior arm -- which is the right
object for "how much does re-training alone move the subspace", but it must not be
described as vanilla-vs-vanilla in the manuscript.
See outputs/positive_result/VANILLA_BASED_AUDIT.md.

INPUT : F:/UNAGI_Project/UNAGI/decoder_unified/loading/*.npy   (30 readouts)
OUTPUT: data/decoder_subspace_test.json
USAGE : python code/decoder_subspace_test.py
"""
import os
import re
import json
import glob
import numpy as np

LOAD_DIR = r"F:/UNAGI_Project/UNAGI/decoder_unified/loading"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "data", "decoder_subspace_test.json")

# Seed-replicate CONTROL readouts (NOT "vanillas" for the hematopoietic tissue --
# R14_s* / R14_seed* are the VEHICLE arm; see the module docstring).
CONTROL_BASES = ("R0", "R14")


def load_all():
    files = sorted(glob.glob(os.path.join(LOAD_DIR, "*.npy")))
    out = {}
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        W = np.load(f).astype(np.float64)
        Q, _ = np.linalg.qr(W.T)          # (G, D) orthonormal columns
        out[name] = dict(W=W, Q=Q, D=W.shape[0], G=W.shape[1],
                         tr=float((W ** 2).sum()))
    return out


def overlap(Q1, Q2):
    if Q1.shape != Q2.shape:
        raise ValueError(f"shape mismatch {Q1.shape} vs {Q2.shape}")
    s = np.linalg.svd(Q1.T @ Q2, compute_uv=False)
    return float((s ** 2).sum() / Q1.shape[1])


def is_control(name):
    return name.split("_")[0] in CONTROL_BASES


def dedupe(names, R, thr=0.999):
    """Drop readouts that are byte-equivalent runs under a second name.

    R14_s42 and R14_seed42 are the SAME run spelled two ways (overlap == 1.000);
    keeping both would inflate the control-vs-control null and bias the test.
    """
    kept, dropped = [], []
    for n in sorted(names):
        dup = None
        for k in kept:
            if overlap(R[n]["Q"], R[k]["Q"]) >= thr:
                dup = k
                break
        if dup is None:
            kept.append(n)
        else:
            dropped.append((n, dup))
    return kept, dropped


def main():
    R = load_all()
    print(f"loaded {len(R)} readouts")

    # group by gene count (tissue proxy)
    by_tissue = {}
    for n, d in R.items():
        by_tissue.setdefault(d["G"], []).append(n)

    result = {}
    for G, names in sorted(by_tissue.items()):
        van_all = sorted([n for n in names if is_control(n)])
        pri_all = sorted([n for n in names if not is_control(n)])
        van, drop_v = dedupe(van_all, R)
        pri, drop_p = dedupe(pri_all, R)
        if drop_v or drop_p:
            print(f"  deduped: {drop_v} {drop_p}")
        vv, pv = [], []
        for i in range(len(van)):
            for j in range(i + 1, len(van)):
                vv.append(overlap(R[van[i]]["Q"], R[van[j]]["Q"]))
        for v in van:
            for p in pri:
                pv.append(overlap(R[v]["Q"], R[p]["Q"]))
        vv, pv = np.array(vv), np.array(pv)
        # one-sided rank test: prior overlap should be LOWER (subspace moved)
        allv = np.concatenate([vv, pv])
        ranks = allv.argsort().argsort()
        U = ranks[len(vv):].sum()
        n1, n2 = len(vv), len(pv)
        Umin = n1 * n2 / 2
        sd = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
        z = (U - Umin) / sd if sd > 0 else 0.0
        result[str(G)] = dict(
            n_control=len(van), n_prior=len(pri),
            control_names=van, prior_names=pri,
            dropped_duplicates=[list(x) for x in (drop_v + drop_p)],
            control_vs_control=dict(mean=float(vv.mean()), sd=float(vv.std(ddof=1)) if n1 > 1 else None,
                                    values=[round(x, 4) for x in vv]),
            prior_vs_control=dict(mean=float(pv.mean()), sd=float(pv.std(ddof=1)) if n2 > 1 else None,
                                  values=[round(x, 4) for x in pv]),
            # legacy keys kept for readers of the old schema (same numbers)
            n_vanilla=len(van), vanilla_names=van,
            vanilla_vs_vanilla=dict(mean=float(vv.mean()),
                                    sd=float(vv.std(ddof=1)) if n1 > 1 else None),
            prior_vs_vanilla=dict(mean=float(pv.mean()),
                                  sd=float(pv.std(ddof=1)) if n2 > 1 else None),
            mannwhitney_U=float(U), z=float(z),
            note=("overlap in [0,1]; lower = prior subspace further from the control "
                  "than seed-replicates of the control are from each other. "
                  "CONTROL CAVEAT: for the hematopoietic tissue the control readouts "
                  "are the VEHICLE arm (R14_s* = 37-TF prior), not an empty prior; "
                  "the true vanilla has a single seed and cannot form a control pair."),
        )
        print(f"G={G}: {len(van)} control, {len(pri)} prior | "
              f"C-C overlap {vv.mean():.3f} vs P-C {pv.mean():.3f} | z={z:+.2f}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(result, open(OUT, "w"), indent=2)
    print("[saved]", os.path.abspath(OUT))


if __name__ == "__main__":
    main()
