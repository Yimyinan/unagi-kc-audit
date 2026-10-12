"""Confirm the mechanism behind the trace-vs-R2 divergence.

Diagnosis (s1c): the released ``kappa_C_W`` returns tr(W^T W) of the Ridge
loadings, while the manuscript names ``sum_g R^2_g``.  With standardised Z and X

    sum_g R^2_g = sum_g W[:,g]' Corr(Z) W[:,g]      (Corr has unit diagonal)
    tr(W^T W)   = sum_g W[:,g]' W[:,g]

so the two coincide **iff Corr(Z) ~ I**.  For wp_b0_s42 the condition number of
Corr(Z) is 2.4e3 and 11/64 eigenvalues are < 0.01 -- precondition violated, and
the two statistics diverge (rotation: 1.247 vs 1.000; sigma=0.05: 0.597 vs 0.986).

If that is the whole story, then **whitening the latent** (Corr -> I exactly)
must make the two statistics agree, restore rotation invariance, and restore the
analytic 1/(1+sigma^2) noise law.  That is what this script tests.

Usage: python s1d_whitened_check.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER1 = os.path.dirname(HERE)
OUT = r"F:/UNAGI_Project/outputs/positive_result/s1_live_path"
sys.path.insert(0, os.path.join(PAPER1, "kappa_c_audit"))
from kappa_c_audit import predictability_ratio, ridge_readout  # noqa: E402

N_CELLS = 15000


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


def both(Z0, X, Z1):
    W0, W1 = ridge_readout(Z0, X), ridge_readout(Z1, X)
    return float((W1 ** 2).sum() / (W0 ** 2).sum()), float(
        predictability_ratio(Z0, X, Z1, X))


def report(tag, Z0, X, rng):
    Zs = (Z0 - Z0.mean(0)) / Z0.std(0)
    C = np.corrcoef(Zs, rowvar=False)
    ev = np.sort(np.linalg.eigvalsh(C))[::-1]
    off = np.abs(C - np.eye(len(C))).mean()
    print(f"\n--- {tag} ---")
    print(f"  Corr(Z): cond {ev[0]/max(ev[-1],1e-300):.3e}   "
          f"eigs<0.01 {int((ev<0.01).sum())}/64   mean|offdiag| {off:.4f}")

    Q, _ = np.linalg.qr(rng.standard_normal((Z0.shape[1], Z0.shape[1])))
    colsd = Z0.std(0)
    cases = {"identity": Z0, "rotation QZ": Z0 @ Q}
    for s in (0.02, 0.05, 0.2):
        cases[f"noise sigma={s}"] = Z0 + s * colsd * rng.standard_normal(Z0.shape)

    print(f"  {'case':<20}{'tr-ratio':>10}{'pred-ratio':>12}{'theory':>9}"
          f"{'agree?':>8}")
    rows = []
    for name, Z1 in cases.items():
        tr, pr = both(Z0, X, Z1)
        th = 1.0
        if "sigma=" in name:
            s = float(name.split("=")[1])
            th = 1.0 / (1 + s * s)
        rows.append(dict(case=name, trace_ratio=tr, predictability_ratio=pr,
                         theory=th))
        print(f"  {name:<20}{tr:>10.4f}{pr:>12.4f}{th:>9.4f}"
              f"{('yes' if abs(tr-pr)<0.02 else 'NO'):>8}")
    return rows


def main():
    z = np.load(os.path.join(OUT, "wp_b0_s42.npz"))
    Z0 = z["Z"].astype(np.float64)
    X = reconstruct_X()
    rng = np.random.default_rng(7)

    res = {}
    res["raw"] = report("raw VAE latent (wp_b0_s42)", Z0, X, rng)

    # whiten: Z_w = Z_s C^{-1/2}  =>  Corr(Z_w) = I exactly
    Zs = (Z0 - Z0.mean(0)) / Z0.std(0)
    C = np.corrcoef(Zs, rowvar=False)
    w, V = np.linalg.eigh(C)
    Chalf = V @ np.diag(w ** -0.5) @ V.T
    Zw = Zs @ Chalf
    res["whitened"] = report("whitened latent (Corr = I by construction)", Zw, X, rng)

    with open(os.path.join(OUT, "s1d_whitened_check.json"), "w") as f:
        json.dump(res, f, indent=1)
    print("\nsaved ->", os.path.join(OUT, "s1d_whitened_check.json"))

    print("\nVERDICT: if the whitened block shows tr-ratio == pred-ratio == theory")
    print("         and rotation == 1.000, then the divergence is fully explained")
    print("         by Corr(Z) != I, i.e. by latent ill-conditioning.")


if __name__ == "__main__":
    main()
