"""Diagnose why the sensitivity calibration deviates from theory.

Observed on the healthy model (wp_b0_s42, eff_rank 16.75):
    sigma=0.02 noise  -> kappa 0.894   (theory 0.9996)
    orthogonal rotation -> kappa 1.231 (expected 1.000)
    subspace k=48       -> kappa 0.303

Hypothesis: the released ``kappa_C_W`` is computed as ``tr(W^T W)`` of the Ridge
loadings, whereas the quantity the manuscript *names* is
``sum_g R^2_g(prior) / sum_g R^2_g(vanilla)``.  For standardised Z and X,

    sum_g R^2_g = sum_g  W[:,g]' Corr(Z) W[:,g]        (Corr = unit diagonal)
    tr(W^T W)   = sum_g  W[:,g]' W[:,g]

These coincide **iff Corr(Z) ~ I**.  When the latent is compressed
(eff_rank << n_latent) Corr is ill-conditioned, (Z'Z)^-1 amplifies the
small-eigenvalue directions, and the two quantities diverge.

This script measures, on one healthy trained model:
    1. the eigenvalue spectrum of Corr(Z)  (is the precondition met?)
    2. kappa_C_W vs predictability_ratio for identity / rotation / noise
    3. what each of them reports for a *controlled* degradation (does either
       follow the analytic 1/(1+sigma^2) law?)

Usage: python s1c_trace_vs_r2_diagnostic.py
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


def trace_ratio(W0, W1):
    return float((W1 ** 2).sum() / (W0 ** 2).sum())


def main():
    z = np.load(os.path.join(OUT, "wp_b0_s42.npz"))
    Z0 = z["Z"].astype(np.float64)
    X = reconstruct_X()
    rng = np.random.default_rng(7)

    # --- 1. is the precondition Corr(Z) ~ I met? ---
    Zs = (Z0 - Z0.mean(0)) / Z0.std(0)
    C = np.corrcoef(Zs, rowvar=False)
    ev = np.sort(np.linalg.eigvalsh(C))[::-1]
    off = C - np.eye(len(C))
    print("=" * 74)
    print("1) conditioning of the latent correlation matrix (n_latent = 64)")
    print(f"   eigenvalues: max {ev[0]:.3f}  min {ev[-1]:.3e}  "
          f"median {np.median(ev):.3e}")
    print(f"   condition number {ev[0]/max(ev[-1],1e-300):.3e}")
    print(f"   |off-diagonal| mean {np.abs(off).mean():.4f}  "
          f"max {np.abs(off).max():.4f}")
    print(f"   eigenvalues < 0.01: {int((ev < 0.01).sum())} / {len(ev)}")
    print("   -> Corr(Z) ~ I ?  " +
          ("YES" if np.abs(off).mean() < 0.05 else "NO (precondition violated)"))

    # --- 2. three controlled perturbations, both statistics ---
    Q, _ = np.linalg.qr(rng.standard_normal((Z0.shape[1], Z0.shape[1])))
    colsd = Z0.std(0)
    cases = {
        "identity": Z0,
        "orthogonal rotation QZ": Z0 @ Q,
        "noise sigma=0.02": Z0 + 0.02 * colsd * rng.standard_normal(Z0.shape),
        "noise sigma=0.05": Z0 + 0.05 * colsd * rng.standard_normal(Z0.shape),
        "noise sigma=0.20": Z0 + 0.20 * colsd * rng.standard_normal(Z0.shape),
    }
    W0 = ridge_readout(Z0, X)
    print("\n" + "=" * 74)
    print("2) tr-ratio (as implemented)  vs  predictability_ratio (as named)")
    print(f"{'case':<26}{'tr(WtW) ratio':>15}{'pred-ratio':>13}{'agree?':>9}")
    rows = []
    for name, Z1 in cases.items():
        W1 = ridge_readout(Z1, X)
        tr = trace_ratio(W0, W1)
        pr = float(predictability_ratio(Z0, X, Z1, X))
        rows.append(dict(case=name, trace_ratio=tr, predictability_ratio=pr))
        agree = "yes" if abs(tr - pr) < 0.02 else "NO"
        print(f"{name:<26}{tr:>15.4f}{pr:>13.4f}{agree:>9}")

    # --- 3. which one tracks the analytic law for noise? ---
    print("\n" + "=" * 74)
    print("3) noise dose-response: which statistic follows 1/(1+sigma^2)?")
    print(f"{'sigma':>7}{'theory':>10}{'tr-ratio':>11}{'pred-ratio':>12}"
          f"{'tr/theory':>11}{'pr/theory':>11}")
    dose = []
    for s in [0.0, 0.02, 0.05, 0.1, 0.2, 0.4]:
        Z1 = Z0 + s * colsd * rng.standard_normal(Z0.shape)
        W1 = ridge_readout(Z1, X)
        tr = trace_ratio(W0, W1)
        pr = float(predictability_ratio(Z0, X, Z1, X))
        th = 1.0 / (1.0 + s * s)
        dose.append(dict(sigma=s, theory=th, trace_ratio=tr, predictability_ratio=pr))
        print(f"{s:>7.2f}{th:>10.4f}{tr:>11.4f}{pr:>12.4f}"
              f"{tr/th:>11.3f}{pr/th:>11.3f}")

    res = dict(corr_eigen=dict(max=float(ev[0]), min=float(ev[-1]),
                               median=float(np.median(ev)),
                               n_below_0p01=int((ev < 0.01).sum()),
                               max_offdiag=float(np.abs(off).max()),
                               mean_offdiag=float(np.abs(off).mean())),
               cases=rows, dose=dose)
    with open(os.path.join(OUT, "s1c_trace_vs_r2.json"), "w") as f:
        json.dump(res, f, indent=1)
    print("\nsaved ->", os.path.join(OUT, "s1c_trace_vs_r2.json"))


if __name__ == "__main__":
    main()
