"""S1b -- decisive sensitivity test on a *healthy* VAE whose prior is wired into
the reconstruction path.

Why grid A had to be discarded
------------------------------
Grid A (contrastvae, 3 arms x 3 seeds) gave kappa_C_W = 1.000 for every paired
prior-vs-off comparison.  Diagnosis showed those models are **fully collapsed**:

    KL -> 0.0000 from epoch ~50;  Z column sd ~ 1e-3;  eff_rank(cov Z) = 2.57/64
    P_on_s42 and P_off_s42 latents are bit-identical

So grid A's exact 1.000 is a property of a degenerate solution, not evidence
about the audit.  It remains a valid *structural* observation -- contrastvae's
prior only touches ``GRNModule``, which ``forward()`` never uses -- but it says
nothing about sensitivity.

This script therefore builds a healthy platform:

  * per-gene z-scoring of log1p expression  (so constant prediction is not the
    optimum, which is what drives the collapse on 97%-zero data)
  * a small fixed KL weight (no anneal to 1.0)
  * an explicit health gate on eff_rank(cov Z) before any audit is trusted

Prior wiring (the path PG-UNAGI's dead ``geneWeight`` branch was meant to be):

    w_g = (1 + boost * mask_g) / mean(1 + boost * mask_g)
    recon = mean_g w_g (x_g - x_hat_g)^2

A gene the prior claims is optimised harder, so the model must trade R^2 across
genes -- exactly what kappa_C_W = sum_g R^2_g(prior) / sum_g R^2_g(vanilla)
aggregates.  ``boost`` is swept, so the answer is a dose-response.

Arms (paired: identical seed => identical init, batch order, dropout masks)
    wp_b0     boost = 0                    <- prior off, clean scalar switch
    wp_b2/10  boost swept                  <- prior on, graded strength
    wp_perm2  boost = 2, permuted skeleton <- strength-matched wrong prior

Usage
    python s1b_weighted_prior_platform.py --health        # 1 run, gate check
    python s1b_weighted_prior_platform.py                 # full sweep
    python s1b_weighted_prior_platform.py --analyse-only
"""
from __future__ import annotations

import argparse
import gc
import itertools
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER1 = os.path.dirname(HERE)
OUT = r"F:/UNAGI_Project/outputs/positive_result/s1_live_path"
CONTRASTVAE = r"F:/UNAGI_Project/UNAGI/contrastvae"
sys.path.insert(0, CONTRASTVAE)
sys.path.insert(0, r"F:/UNAGI_Project/UNAGI/pg_unagi")
sys.path.insert(0, os.path.join(PAPER1, "kappa_c_audit"))

import torch  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from kappa_c_audit import eff_rank, kappa_C_Z, ridge_readout  # noqa: E402

HEMA_TFS = ["GATA1", "KLF1", "TAL1", "GFI1B", "GATA2", "LMO2",
            "NFE2", "SPI1", "RUNX1", "CEBPA", "PAX5"]
BOOSTS = [0.0, 2.0, 10.0]
PERM_BOOST = 2.0
SEEDS = [42, 123, 456]
N_CELLS = 8000
BETA = 0.05          # fixed and small -- the collapse in grid A came from beta -> 1
EPOCHS = 40
HEALTH_MIN_EFF_RANK = 8.0


# ----------------------------------------------------------------- data
def load_data(n_cells=N_CELLS):
    """log1p, then per-gene z-score.  z-scoring is essential: on 97%-zero data a
    constant prediction is the MSE optimum, which collapses the KL term.  It also
    matches the audit's own per-gene standardisation."""
    import anndata as ad
    from scipy.sparse import issparse
    a = ad.read_h5ad(r"F:/UNAGI_Project/UNAGI/result/R14/R14_hemato_vanilla/"
                     r"iter_4/stagedata/dataset.h5ad", backed="r")
    genes = list(a.var_names)
    idx = np.random.default_rng(0).choice(a.n_obs, size=n_cells, replace=False)
    idx.sort()
    Xall = a.X[:]
    Xs = Xall[idx]
    X = (Xs.toarray() if issparse(Xs) else np.asarray(Xs)).astype(np.float32)
    del a, Xall, Xs
    gc.collect()
    X = np.log1p(X)
    mu, sd = X.mean(0), X.std(0)
    sd[sd < 1e-8] = 1.0
    X = (X - mu) / sd
    print(f"  data: {X.shape}  z-scored per gene  "
          f"(median sd before: {np.median(X.std(0)):.3f})")
    return X.astype(np.float32), genes


def build_mask(genes, kind="real"):
    from tf_database import build_tf_database
    up = [g.upper() for g in genes]
    names = [tf for tf in HEMA_TFS if tf.upper() in up]
    db = build_tf_database(names, force_rebuild=False)
    m = np.zeros(len(genes), dtype=np.float32)
    for tf in names:
        for tgt in db.get(tf, []):
            tu = tgt.upper()
            if tu in up:
                m[up.index(tu)] = 1.0
    if kind == "perm":
        rng = np.random.default_rng(1)
        p = np.zeros_like(m)
        p[rng.choice(len(m), size=int(m.sum()), replace=False)] = 1.0
        m = p
    return m, names


# ----------------------------------------------------------------- train
def run_one(tag, boost, mask_kind, seed, X, genes, epochs=EPOCHS, beta=BETA, device="cuda"):
    from model import Decoder, Encoder

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    mask, names = build_mask(genes, mask_kind)
    w = 1.0 + boost * mask
    w = w / w.mean()
    wt = torch.tensor(w, dtype=torch.float32, device=device)

    n_genes = X.shape[1]
    enc = Encoder(n_genes, 64, 256, 2).to(device)
    dec = Decoder(64, n_genes, 256, 2).to(device)
    params = list(enc.parameters()) + list(dec.parameters())
    opt = torch.optim.Adam(params, lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)

    Xt = torch.tensor(X, dtype=torch.float32)
    loader = DataLoader(TensorDataset(Xt), batch_size=512, shuffle=True,
                        num_workers=0, pin_memory=(device == "cuda"),
                        drop_last=True)

    t0 = time.time()
    hist = []
    for ep in range(epochs):
        enc.train(); dec.train()
        tot = rec = klv = 0.0; nb = 0
        for (xb,) in loader:
            xb = xb.to(device)
            opt.zero_grad()
            mu, logvar = enc(xb)
            std = torch.exp(0.5 * logvar)
            z = mu + torch.randn_like(std) * std
            xr = dec(z)
            recon = (wt * (xr - xb) ** 2).mean()
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon + beta * kl
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
            tot += loss.item(); rec += recon.item(); klv += kl.item(); nb += 1
        sched.step()
        hist.append(dict(epoch=ep, loss=tot / nb, recon=rec / nb, kl=klv / nb))
        if ep % max(1, epochs // 4) == 0 or ep == epochs - 1:
            print(f"   ep{ep:3d} loss={tot/nb:.4f} recon={rec/nb:.4f} kl={klv/nb:.4f}")
    dt = (time.time() - t0) / 60

    enc.eval()
    mus = []
    with torch.no_grad():
        for i in range(0, X.shape[0], 2048):
            m, _ = enc(Xt[i:i + 2048].to(device))
            mus.append(m.cpu().numpy())
    Z = np.concatenate(mus, 0).astype(np.float32)

    er = float(eff_rank(np.cov(Z.astype(np.float64), rowvar=False)))
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(os.path.join(OUT, f"{tag}.npz"),
                        Z=Z, mask=mask, w=w.astype(np.float32))
    diag = dict(tag=tag, boost=boost, mask=mask_kind, seed=seed,
                minutes=dt, epochs=epochs, mask_genes=int(mask.sum()),
                w_max=float(w.max()), w_spread=float(w.max() / w.min()),
                eff_rank_Z=er, Z_sd=float(Z.std()),
                final_kl=hist[-1]["kl"], final_recon=hist[-1]["recon"],
                healthy=bool(er >= HEALTH_MIN_EFF_RANK))
    del Xt, loader, enc, dec
    torch.cuda.empty_cache(); gc.collect()
    return diag


# ----------------------------------------------------------------- analyse
def analyse(X):
    runs = {}
    for b in BOOSTS:
        for s in SEEDS:
            p = os.path.join(OUT, f"wp_b{b:g}_s{s}.npz")
            if os.path.exists(p):
                runs[(f"b{b:g}", s)] = np.load(p)["Z"].astype(np.float64)
    for s in SEEDS:
        p = os.path.join(OUT, f"wp_perm{PERM_BOOST:g}_s{s}.npz")
        if os.path.exists(p):
            runs[(f"perm{PERM_BOOST:g}", s)] = np.load(p)["Z"].astype(np.float64)
    if not runs:
        print("no runs"); return []

    read = {k: ridge_readout(v, X) for k, v in runs.items()}
    rows = []
    for a, b in itertools.combinations(SEEDS, 2):
        if ("b0", a) not in runs or ("b0", b) not in runs:
            continue
        Wa, Wb = read[("b0", a)], read[("b0", b)]
        rows.append(dict(kind="null (b0 vs b0, cross-seed)", seed=f"{a}/{b}",
                         kappa=float((Wb ** 2).sum() / (Wa ** 2).sum()),
                         d_readout=float(np.linalg.norm(Wb - Wa)
                                         / max(np.linalg.norm(Wa), np.linalg.norm(Wb))),
                         d_eff_rank=100 * (eff_rank(np.cov(runs[("b0", b)], rowvar=False))
                                           / eff_rank(np.cov(runs[("b0", a)], rowvar=False)) - 1)))
    for arm in [f"b{b:g}" for b in BOOSTS[1:]] + [f"perm{PERM_BOOST:g}"]:
        for s in SEEDS:
            if (arm, s) not in runs or ("b0", s) not in runs:
                continue
            Wp, Wv = read[(arm, s)], read[("b0", s)]
            rows.append(dict(
                kind=f"{arm} vs b0 (same seed)", seed=str(s),
                kappa=float((Wp ** 2).sum() / (Wv ** 2).sum()),
                d_readout=float(np.linalg.norm(Wp - Wv)
                                / max(np.linalg.norm(Wv), np.linalg.norm(Wp))),
                d_eff_rank=100 * (eff_rank(np.cov(runs[(arm, s)], rowvar=False))
                                  / eff_rank(np.cov(runs[("b0", s)], rowvar=False)) - 1),
                kappa_C_Z=float(kappa_C_Z(runs[("b0", s)], runs[(arm, s)]))))

    hdr = (f"{'comparison':<34}{'seed':<9}{'kappa':>9}{'d_read':>9}"
           f"{'dEffR%':>9}{'kCZ':>8}")
    print(hdr); print("-" * len(hdr))
    for r in rows:
        kz = f"{r['kappa_C_Z']:>8.4f}" if "kappa_C_Z" in r else f"{'-':>8}"
        print(f"{r['kind']:<34}{r['seed']:<9}{r['kappa']:>9.4f}"
              f"{r['d_readout']:>9.4f}{r['d_eff_rank']:>9.2f}{kz}")
    with open(os.path.join(OUT, "s1b_rows.json"), "w") as f:
        json.dump(rows, f, indent=1)
    print("\nsaved ->", os.path.join(OUT, "s1b_rows.json"))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--analyse-only", action="store_true")
    ap.add_argument("--health", action="store_true")
    ap.add_argument("--beta", type=float, default=BETA)
    ap.add_argument("--cells", type=int, default=N_CELLS)
    args = ap.parse_args()
    seeds = args.seeds or SEEDS
    if args.health:
        seeds = [42]
        BOOSTS[:] = [0.0]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  epochs={args.epochs}  seeds={seeds}  beta={args.beta} fixed")
    X, genes = load_data(args.cells)

    diags = []
    if not args.analyse_only:
        jobs = [(f"wp_b{b:g}", b, "real") for b in BOOSTS]
        if not args.health:
            jobs.append((f"wp_perm{PERM_BOOST:g}", PERM_BOOST, "perm"))
        for tag, boost, kind in jobs:
            for s in seeds:
                print(f"\n=== {tag} seed={s} (boost={boost}, {kind}) ===")
                d = run_one(f"{tag}_s{s}", boost, kind, s, X, genes,
                            epochs=args.epochs, beta=args.beta, device=device)
                flag = "" if d["healthy"] else "   <-- UNHEALTHY (collapsed)"
                print(f"   {d['minutes']:.2f} min | mask {d['mask_genes']} genes | "
                      f"w_spread {d['w_spread']:.2f}x | eff_rank(Z)={d['eff_rank_Z']:.2f}"
                      f" | Z_sd={d['Z_sd']:.3f} | final KL={d['final_kl']:.4f}{flag}")
                diags.append(d)
                with open(os.path.join(OUT, "s1b_diagnostics.json"), "w") as f:
                    json.dump(diags, f, indent=1)

    if args.health:
        ok = all(d["healthy"] for d in diags)
        print(f"\nHEALTH GATE: {'PASS' if ok else 'FAIL'}")
        return
    analyse(X)


if __name__ == "__main__":
    main()
