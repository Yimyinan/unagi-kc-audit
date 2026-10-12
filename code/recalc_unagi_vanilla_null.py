"""
recalc_unagi_vanilla_null.py
=============================
Build the UNAGI-family vanilla null band for kappa-C(W) using the SAME Ridge
protocol as the published audit (Z = obsm['z'] latent, X = expression;
W = Ridge(alpha=1.0, fit_intercept=True) fit Z->X; kappa = tr(W^T W) of the
prior readout / vanilla readout).

Source of truth: F:/UNAGI_Project/UNAGI/result/<run>/iter_4/stagedata/*.h5ad
(audit "stagedata_unified"). Real UNAGI vanilla seeds discovered in stagedata:
    hema  : R14_hemato_seed42 / seed123 / seed456  (+ R14_hemato_vanilla)
    neural: R0_vanilla_seed123 / seed456
(lung has a single vanilla, R10, no stagedata-vanilla pair found -> excluded)

Validation: Ridge trH of R14_hemato_vanilla must == audit hema tr_V = 5634.8.

Outputs: ../data/unaug_vanilla_null_ridge.json
"""
import os, sys, glob, json
import numpy as np
import h5py

RESULT = r"F:/UNAGI_Project/UNAGI/result"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                   "data", "unaug_vanilla_null_ridge.json")
ALPHA = 1.0
LAT = "obsm/z"
ITER = "iter_4"


def trH_from_root(root):
    stages = sorted(glob.glob(os.path.join(root, ITER, "stagedata", "*.h5ad")))
    stages = [s for s in stages if not s.endswith("dataset.h5ad")]
    if not stages:
        raise FileNotFoundError(f"no stagedata at {root}/{ITER}")
    A = None
    B = None
    G = None
    N = 0
    for s in stages:
        with h5py.File(s, "r") as h:
            Z = np.asarray(h[LAT][:], dtype=np.float64)        # (Ns,64)
            X = np.asarray(h["X"][:], dtype=np.float64)        # (Ns,G)
            if G is None:
                G = X.shape[1]
                B = np.zeros((Z.shape[1], G))
                A = np.zeros((Z.shape[1], Z.shape[1]))
            A += Z.T @ Z
            B += Z.T @ X
            N += Z.shape[0]
    A += ALPHA * np.eye(Z.shape[1])
    W = np.linalg.solve(A, B)                                   # (64, G)
    return float((W ** 2).sum()), N


# run-root -> stagedata parent  (glob to be robust to nesting)
def root_for(name):
    hits = glob.glob(os.path.join(RESULT, "**", name, ITER, "stagedata"),
                     recursive=True)
    hits = [h for h in hits if not h.endswith("dataset.h5ad")]
    return hits[0][:-(len(f"{ITER}/stagedata"))] if hits else None


VANILLA = {
    "hemo": ["R14_hemato_seed42", "R14_hemato_seed123", "R14_hemato_seed456",
             "R14_hemato_vanilla"],
    "neural": ["R0_vanilla_seed123", "R0_vanilla_seed456"],
}


def main():
    out = {"protocol": "Ridge(alpha=1) Z->X on iter_4 stagedata; kappa=trH_i/trH_j",
           "validation": {}, "vanilla_trH": {}, "null": {}}
    # ---- validation: R14_hemato_vanilla trH should be ~5634.8 ----
    rv = root_for("R14_hemato_vanilla")
    tr, n = trH_from_root(rv)
    out["validation"] = dict(root=rv, trH=round(tr, 2), n_cells=n,
                             audit_hema_trV=5634.8,
                             match=abs(tr - 5634.8) < 50)
    # ---- vanilla trH per seed ----
    for tissue, seeds in VANILLA.items():
        out["vanilla_trH"][tissue] = {}
        for s in seeds:
            r = root_for(s)
            if not r:
                out["vanilla_trH"][tissue][s] = "NO_STAGEDATA"
                continue
            tr, n = trH_from_root(r)
            out["vanilla_trH"][tissue][s] = dict(trH=round(tr, 2), n_cells=n)
    # ---- build cross-seed null (exclude the extra 'R14_hemato_vanilla' from
    #      the seeded 3-seed set to keep pairs comparable; include all seeds) ----
    all_pairs = []
    for tissue, seeds in VANILLA.items():
        trs = [out["vanilla_trH"][tissue][s].get("trH") for s in seeds
               if isinstance(out["vanilla_trH"][tissue][s], dict)]
        trs = [t for t in trs if t]
        pairs = []
        for i in range(len(trs)):
            for j in range(len(trs)):
                if i == j:
                    continue
                pairs.append(round(trs[i] / trs[j], 4))
                all_pairs.append(trs[i] / trs[j])
        arr = np.array(pairs)
        out["null"][tissue] = dict(n_seeds=len(trs), n_pairs=len(pairs),
            mean=round(float(arr.mean()), 4), std=round(float(arr.std()), 4),
            lo2s=round(float(arr.mean() - 2 * arr.std()), 4),
            hi2s=round(float(arr.mean() + 2 * arr.std()), 4),
            p05=round(float(np.percentile(arr, 5)), 4),
            p95=round(float(np.percentile(arr, 95)), 4),
            min=round(float(arr.min()), 4), max=round(float(arr.max()), 4))
    arr = np.array(all_pairs)
    out["null"]["UNAGI_combined"] = dict(n_pairs=len(all_pairs),
        mean=round(float(arr.mean()), 4), std=round(float(arr.std()), 4),
        lo2s=round(float(arr.mean() - 2 * arr.std()), 4),
        hi2s=round(float(arr.mean() + 2 * arr.std()), 4),
        p05=round(float(np.percentile(arr, 5)), 4),
        p95=round(float(np.percentile(arr, 95)), 4))
    out["comparison_to_GVAE"] = dict(
        GVAE_band=[0.78, 1.23], UNAGI_combined_band=[out["null"]["UNAGI_combined"]["lo2s"],
        out["null"]["UNAGI_combined"]["hi2s"]])
    json.dump(out, open(OUT, "w"), indent=2)
    print("== validation ==")
    print(f"  R14_hemato_vanilla trH={tr:.2f}  (audit 5634.8)  match={out['validation']['match']}")
    print("== vanilla trH ==")
    for t, d in out["vanilla_trH"].items():
        print(f"  {t}: {d}")
    print("== UNAGI vanilla null bands (kappa=trH_i/trH_j) ==")
    for t, d in out["null"].items():
        print(f"  {t:16s}: {d}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
