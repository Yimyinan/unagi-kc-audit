"""S1 sensitivity calibration -- the model-free positive control.

Grid A (s1_live_path_positive_control.py) asks whether a *real* prior injection
moves kappa_C_W.  That is a necessary test but it is not sufficient, because a
null result is ambiguous: it can mean "the tool is blind" OR "the prior did not
change the latent->expression map in this platform".

This script removes the ambiguity by measuring the tool's **response curve**
directly: take one trained model's own (Z, X) and degrade the latent->expression
map by a controlled, monotone amount, then read out kappa_C_W.

Families
    noise       Z' = Z + sigma * noise        -> theory: kappa = 1/(1+sigma^2)
    subspace    Z' = Z projected on top-k PCs -> monotone information loss
    invariance  Z' = QZ (orthogonal) and Z' = diag(s)Z  -> must stay 1.000
                X' = per-gene rescaling of X            -> must stay 1.000

Outputs a detection threshold: the smallest degradation that separates from the
1.0 baseline, expressed in sigma-equivalent units.  Any prior effect measured in
grid A can then be placed on this axis and judged as above or below threshold.

Usage: python s1_sensitivity_calibration.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER1 = os.path.dirname(HERE)
OUT = r"F:/UNAGI_Project/outputs/positive_result/s1_live_path"
sys.path.insert(0, os.path.join(PAPER1, "kappa_c_audit"))
from kappa_c_audit import eff_rank, ridge_readout  # noqa: E402

N_CELLS = 15000
SIGMAS = [0.0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6]
KEEP_DIMS = [64, 48, 32, 24, 16, 8, 4]


def reconstruct_X():
    import anndata as ad
    from scipy.sparse import issparse
    a = ad.read_h5ad(r"F:/UNAGI_Project/UNAGI/result/R14/R14_hemato_vanilla/"
                     r"iter_4/stagedata/dataset.h5ad", backed="r")
    idx = np.random.default_rng(0).choice(a.n_obs, size=N_CELLS, replace=False)
    idx.sort()
    Xall = a.X[:]
    Xs = Xall[idx]
    X = (Xs.toarray() if issparse(Xs) else np.asarray(Xs)).astype(np.float32)
    del a, Xall, Xs
    return np.log1p(X).astype(np.float64)


def kcw(Z0, X0, Z1, X1):
    W0 = ridge_readout(Z0, X0)
    W1 = ridge_readout(Z1, X1)
    return float((W1 ** 2).sum() / (W0 ** 2).sum()), W0, W1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="wp_b0_s42.npz",
                    help="reference run inside the s1_live_path output dir. "
                         "Use wp_b0_s42.npz (healthy, eff_rank>16) -- the grid-A "
                         "P_off_s42 is COLLAPSED (eff_rank 2.57) and invalid here.")
    args = ap.parse_args()

    z = np.load(os.path.join(OUT, args.ref))
    Z0 = z["Z"].astype(np.float64)
    X = reconstruct_X()
    colsd = Z0.std(0)
    print(f"reference {args.ref}: Z{Z0.shape}  X{X.shape}")
    print(f"  latent column sd: median {np.median(colsd):.4f}  "
          f"eff_rank {eff_rank(np.cov(Z0, rowvar=False)):.2f}\n")

    res = {"noise": [], "subspace": [], "invariance": [], "reference": args.ref}

    print("--- noise injection  Z' = Z + sigma*sd_col*eps   "
          "(theory: kappa = 1/(1+sigma^2)) ---")
    print(f"{'sigma':>7}{'kappa_C_W':>12}{'theory':>10}{'ratio':>9}")
    rng = np.random.default_rng(7)
    for s in SIGMAS:
        eps = rng.standard_normal(Z0.shape)
        Z1 = Z0 + s * colsd * eps
        k, _, _ = kcw(Z0, X, Z1, X)
        th = 1.0 / (1.0 + s * s)
        res["noise"].append(dict(sigma=s, kappa_C_W=k, theory=th))
        print(f"{s:>7.2f}{k:>12.4f}{th:>10.4f}{k/th:>9.3f}")

    print("\n--- subspace ablation  keep top-k latent PCs ---")
    print(f"{'k':>7}{'kappa_C_W':>12}")
    U, S, Vt = np.linalg.svd(Z0 - Z0.mean(0), full_matrices=False)
    for k in KEEP_DIMS:
        Zk = (U[:, :k] * S[:k]) @ Vt[:k] + Z0.mean(0)
        v, _, _ = kcw(Z0, X, Zk, X)
        res["subspace"].append(dict(k=k, kappa_C_W=v))
        print(f"{k:>7d}{v:>12.4f}")

    print("\n--- invariance checks (must be 1.000) ---")
    Q, _ = np.linalg.qr(rng.standard_normal((Z0.shape[1], Z0.shape[1])))
    checks = {
        "orthogonal_rotation_QZ": (Z0 @ Q, X),
        "per_dim_rescale_2Z": (2.0 * Z0, X),
        "per_dim_random_scale": (Z0 * rng.uniform(0.2, 5.0, Z0.shape[1]), X),
        "per_gene_rescale_X": (Z0, X * rng.uniform(0.2, 5.0, X.shape[1])),
        "identity": (Z0, X),
    }
    print(f"{'check':<26}{'kappa_C_W':>12}")
    for name, (Za, Xa) in checks.items():
        v, _, _ = kcw(Z0, X, Za, Xa)
        res["invariance"].append(dict(check=name, kappa_C_W=v))
        print(f"{name:<26}{v:>12.6f}")

    # detection threshold: smallest sigma whose kappa is below the identity value
    ident = [c for c in res["invariance"] if c["check"] == "identity"][0]["kappa_C_W"]
    thr = None
    for row in res["noise"]:
        if row["sigma"] > 0 and row["kappa_C_W"] < ident - 0.01:
            thr = row["sigma"]
            break
    print(f"\nidentity kappa = {ident:.6f}")
    print(f"smallest sigma detectable at the 1% level: {thr}")
    res["identity_kappa"] = ident
    res["detection_threshold_sigma"] = thr

    with open(os.path.join(OUT, "s1_sensitivity_calibration.json"), "w") as f:
        json.dump(res, f, indent=1)
    print("\nsaved ->", os.path.join(OUT, "s1_sensitivity_calibration.json"))


if __name__ == "__main__":
    main()
