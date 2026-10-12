# -*- coding: utf-8 -*-
"""S1i -- recheck the cross-architecture kappa-C(W) claims in the corrected statistic.

The manuscript's cross-architecture section (main.tex, "The decoder/latent
decoupling reproduces on an independent VAE (scVI)") reports

    kappa-C(W) = 1.22   (decoder adapts)  vs  kappa-C(Z) = 0.75

for the Norman 2019 GEARS conditioning experiment, and the same for a family-4
MLP VAE.  Those were computed with the trace form of the readout.  Because the
cached latents are on disk (Zv = vanilla, Zp = perturbation-conditioned, shared X)
the claim is recomputable at no training cost.

Both forms, for a readout W = Ridge(Z -> X) fitted on per-column-standardised Z
and X (alpha = 1, no intercept):

    trace form        tr(W W^T)
    corrected form    tr(Corr(Z) W W^T)        (= sum_g R^2_g, since SS_tot = N
                                                for a standardised X)

so the only extra ingredient the corrected form needs is the latent correlation
matrix -- not the expression matrix.

Reads  : data/_norman_latents.npz        (Zv, Zp, X, pert_labels)
         data/_norman_vae4_latents.npz   (Zv, Zp; X shared with the above)
Writes : outputs/positive_result/s1i_crossarch_recheck.json

Usage: python code/s1i_crossarch_recheck.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "kappa_c_audit"))
PAPER1 = os.path.abspath(os.path.join(HERE, ".."))
ROOT = os.path.abspath(os.path.join(PAPER1, "..", ".."))
OUT = os.path.join(ROOT, "outputs", "positive_result")

from kappa_c_audit.core import ridge_readout, eff_rank  # noqa: E402


def corr_of(Z):
    Z = np.asarray(Z, dtype=np.float64)
    Z = Z - Z.mean(0, keepdims=True)
    sd = Z.std(0, ddof=0, keepdims=True)
    sd[sd < 1e-12] = 1.0
    Zs = Z / sd
    return Zs.T @ Zs / Zs.shape[0]


def forms(Z, X):
    """(trace form, corrected form, corr condition number) for one model."""
    W = ridge_readout(Z, X).astype(np.float64)
    WWt = W @ W.T
    C = corr_of(Z)
    tr_form = float(np.trace(WWt))
    r2_form = float(np.trace(C @ WWt))
    ev = np.linalg.eigvalsh(C)
    cond = float(ev.max() / max(ev.min(), 1e-300))
    return dict(tr=tr_form, sum_r2=r2_form,
                corr_cond=cond, n_eig_below_0p01=int((ev < 0.01).sum()))


def main():
    n1 = np.load(os.path.join(PAPER1, "data", "_norman_latents.npz"),
                 allow_pickle=True)
    n4 = np.load(os.path.join(PAPER1, "data", "_norman_vae4_latents.npz"),
                 allow_pickle=True)
    X = n1["X"]

    out = {"protocol": ("W = Ridge(Z->X) on per-column-standardised Z, X "
                        "(alpha=1, no intercept); trace = tr(WW'), corrected = "
                        "tr(Corr(Z) WW') = sum_g R2_g"),
           "architectures": {}}

    cases = {
        "scVI_Norman": (n1["Zv"], n1["Zp"]),
        "family4_MLP_Norman": (n4["Zv"], n4["Zp"]),
    }
    print(f"Norman 2019 GEARS: X {X.shape}, shared across architectures\n")
    hdr = (f"{'architecture':<22}{'model':<12}{'tr':>14}{'sum_r2':>12}"
           f"{'corrCond':>11}{'effRank':>9}")
    print(hdr)
    print("-" * len(hdr))
    for name, (Zv, Zp) in cases.items():
        rec = {}
        for tag, Z in (("vanilla", Zv), ("conditioned", Zp)):
            f = forms(Z, X)
            f["eff_rank"] = float(eff_rank(corr_of(Z)))
            rec[tag] = f
            print(f"{name:<22}{tag:<12}{f['tr']:>14.1f}{f['sum_r2']:>12.2f}"
                  f"{f['corr_cond']:>11.1f}{f['eff_rank']:>9.2f}")
        v, p = rec["vanilla"], rec["conditioned"]
        rec["kappa_tr"] = p["tr"] / v["tr"]
        rec["kappa_r2"] = p["sum_r2"] / v["sum_r2"]
        rec["kappa_C_Z_effrank"] = p["eff_rank"] / v["eff_rank"]
        out["architectures"][name] = rec
        print(f"{'':<22}{'kappa_W':<12}{rec['kappa_tr']:>14.4f}"
              f"{rec['kappa_r2']:>12.4f}   (trace vs corrected)")

    with open(os.path.join(OUT, "s1i_crossarch_recheck.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)

    print()
    print(f"{'architecture':<22}{'k_tr':>9}{'k_r2':>9}   published / reading")
    for name, rec in out["architectures"].items():
        print(f"{name:<22}{rec['kappa_tr']:>9.4f}{rec['kappa_r2']:>9.4f}")
    print(f"\nwrote {os.path.join(OUT, 's1i_crossarch_recheck.json')}")


if __name__ == "__main__":
    main()
