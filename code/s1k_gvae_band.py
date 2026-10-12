# -*- coding: utf-8 -*-
"""Recompute the G-VAE cross-family vanilla band in the corrected statistic.

Why
---
The manuscript's [0.78, 1.23] band is the G-VAE hematopoietic vanilla null, quoted
as a cross-family reference.  It was calibrated on the released trace form and its
generator is no longer in the repository, but every ingredient survives on disk:

    checkpoints   F:/UNAGI_Project/next_phase/results/gvae_hema_scan/vanilla_s1..12.pt
                  F:/UNAGI_Project/next_phase/results/gvae_hema_v13/vanilla_s42/123/456.pt
    model         ptvae.gvae_model.GVAE(n_genes, n_latent=64, n_hidden=512, n_layers=2)
    data          UNAGI/result/R14/R14_hemato_vanilla/iter_4/stagedata/dataset.h5ad
                  (the shared hematopoietic expression matrix, as in the original)

The published per-seed trace norms are stored in data/vanilla_null_v2.json, so this
script first tries to REPRODUCE them (validation), then recomputes the same pool in
the corrected predictability form sum_g R2_g.

Protocol, both forms, per seed:
    Zs = StandardScaler(Z);  Xs = StandardScaler(X)          (per column/gene)
    A  = Ridge(alpha=1, fit_intercept=False).fit(Zs, Xs).coef_     (G x d)
    trace form         tr(A A^T)
    corrected form     G - ||Xs - Zs A^T||_F^2 / N             (== sum_g R2_g)

Reads  : data/vanilla_null_v2.json
Writes : outputs/positive_result/s1k_gvae_band.json
Usage  : python code/s1k_gvae_band.py [--cells 2000]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER1 = os.path.abspath(os.path.join(HERE, ".."))
ROOT = os.path.abspath(os.path.join(PAPER1, "..", ".."))
OUT = os.path.join(ROOT, "outputs", "positive_result")

NEXT_PHASE = r"F:/UNAGI_Project/next_phase"
SCAN = os.path.join(NEXT_PHASE, "results", "gvae_hema_scan")
V13 = os.path.join(NEXT_PHASE, "results", "gvae_hema_v13")
H5AD = (r"F:/UNAGI_Project/UNAGI/result/R14/R14_hemato_vanilla/iter_4"
        r"/stagedata/dataset.h5ad")

SEEDS = [str(s) for s in range(1, 13)] + ["42", "123", "456"]


def ckpt_path(seed):
    if seed in ("42", "123", "456"):
        return os.path.join(V13, f"vanilla_s{seed}.pt")
    return os.path.join(SCAN, f"vanilla_s{seed}.pt")


def load_X(n_cells):
    import h5py
    import scipy.sparse as sp
    with h5py.File(H5AD, "r") as f:
        X = f["X"]
        if isinstance(X, h5py.Group):                      # sparse CSR/CSC
            data, indices, indptr = X["data"][:], X["indices"][:], X["indptr"][:]
            shape = tuple(X.attrs["shape"])
            M = sp.csr_matrix((data, indices, indptr), shape=shape)
        else:
            M = np.asarray(X)
    M = M[:n_cells]
    return np.asarray(M.todense() if sp.issparse(M) else M, dtype=np.float32)


def one(seed, X, device):
    import torch
    sys.path.insert(0, NEXT_PHASE)
    from ptvae.gvae_model import GVAE
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    model = GVAE(n_genes=X.shape[1], n_latent=64, n_hidden=512, n_layers=2).to(device)
    ck = torch.load(ckpt_path(seed), map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()
    with torch.no_grad():
        Z = model.get_latent(torch.tensor(X, dtype=torch.float32).to(device))
    del model
    Z = np.asarray(Z.detach().cpu().numpy(), dtype=np.float64)

    Zs = StandardScaler().fit_transform(Z)
    Xs = StandardScaler().fit_transform(X.astype(np.float64))
    A = Ridge(alpha=1.0, fit_intercept=False).fit(Zs, Xs).coef_     # (G, d)
    tr_form = float((A ** 2).sum())
    resid = Xs - Zs @ A.T
    r2_form = float(Xs.shape[1] - (resid ** 2).sum() / Xs.shape[0])
    return tr_form, r2_form


def band(vals):
    v = np.asarray(vals, float)
    n = v.size
    r = np.array([v[i] / v[j] for i in range(n) for j in range(n) if i != j])
    sd = r.std(ddof=1)
    return dict(n_seeds=n, n_pairs=int(r.size), mean=float(r.mean()), sd=float(sd),
                lo2s=float(r.mean() - 2 * sd), hi2s=float(r.mean() + 2 * sd),
                min=float(r.min()), max=float(r.max()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", type=int, default=2000)
    args = ap.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  cells={args.cells}")
    X = load_X(args.cells)
    print(f"X {X.shape}")

    pub = json.load(open(os.path.join(PAPER1, "data", "vanilla_null_v2.json"),
                         encoding="utf-8-sig"))
    pub_tr = pub["trH"]

    tr, r2 = {}, {}
    print(f"\n{'seed':<6}{'computed tr':>14}{'published tr':>14}{'rel diff':>11}")
    for s in SEEDS:
        t, rr = one(s, X, device)
        tr[s], r2[s] = t, rr
        p = pub_tr.get(str(s)) if str(s) in pub_tr else pub_tr.get(s)
        rel = f"{abs(t - p) / p * 100:.1f}%" if p else "n/a"
        print(f"{s:<6}{t:>14.1f}{(p if p else float('nan')):>14.1f}{rel:>11}")

    out = dict(cells=int(args.cells), protocol=(
        "Zs=StandardScaler(Z); Xs=StandardScaler(X); Ridge(alpha=1, no intercept); "
        "trace = tr(AA^T); corrected = G - ||Xs - Zs A^T||^2/N"),
        trace_form=dict(trH=tr, band=band(list(tr.values()))),
        corrected_form=dict(sum_r2=r2, band=band(list(r2.values()))),
        published=dict(band={"mean": pub["mean"], "sd": pub["std"],
                             "lo2s": pub["lo2s"], "hi2s": pub["hi2s"]},
                       n_seeds=pub["n_seeds"], n_pairs=pub["n_pairs"]))
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "s1k_gvae_band.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)

    print("\nband: mean +/- 2sd of the 210 ordered cross-seed ratios")
    for k, lbl in (("trace_form", "trace (released)"),
                   ("corrected_form", "corrected")):
        b = out[k]["band"]
        print(f"  {lbl:<18} {b['mean']:.4f} +/- {b['sd']:.4f}  "
              f"[{b['lo2s']:.3f}, {b['hi2s']:.3f}]  min {b['min']:.3f} max {b['max']:.3f}")
    pb = pub
    print(f"  {'published (trace)':<18} {pb['mean']:.4f} +/- {pb['std']:.4f}  "
          f"[{pb['lo2s']:.3f}, {pb['hi2s']:.3f}]")
    print(f"\nwrote {os.path.join(OUT, 's1k_gvae_band.json')}")


if __name__ == "__main__":
    main()
