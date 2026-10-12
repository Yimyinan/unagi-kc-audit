"""S1 -- Live-path positive control for the kappa-C audit.

Question
--------
When a GRN prior *demonstrably* enters the training loss, does ``kappa_C_W``
detect it?  This is the decisive test that separates

    "the audit is insensitive"   (tool limitation)      -> 0/33 is a readout artefact
    "the prior never arrived"    (implementation fact)  -> 0/33 is about the pipeline

Platform
--------
``F:/UNAGI_Project/UNAGI/contrastvae`` -- a plain VAE whose GRN prior is a real
loss term (static, verifiable, no patcher, no iDREM):

    model.py:74-94   GRNModule.prior_penalty() = ||W * (1 - prior_mask)||_1
    model.py:180     graph_loss = self.grn.prior_penalty()
    model.py:196     total = recon + beta*kl + lambda_graph*graph_loss + cdsl + ...

Arms (paired: identical seed => identical init, batch order and dropout masks)
    P_on     real skeleton,   lambda_graph = 0.01
    P_off    real skeleton,   lambda_graph = 0.0    <- clean single-variable switch-off
    P_wrong  permuted skeleton (same per-TF edge counts), lambda_graph = 0.01

Trap avoided: ``prior_skeleton=None`` is NOT "prior off" -- the mask becomes all
zero and prior_penalty() degenerates to a plain L1 on *every* edge.  The switch
must be ``lambda_graph = 0``.

Null
----
P_off vs P_off across seeds (the "vanilla-vs-vanilla" analogue).  The prior arms
must separate from P_off by more than P_off separates from itself.

Usage
-----
    python s1_live_path_positive_control.py --probe           # 1 epoch timing test
    python s1_live_path_positive_control.py                   # full 3 arms x 3 seeds
    python s1_live_path_positive_control.py --analyse-only    # re-analysis
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER1 = os.path.dirname(HERE)
OUT = r"F:/UNAGI_Project/outputs/positive_result/s1_live_path"

CONTRASTVAE = r"F:/UNAGI_Project/UNAGI/contrastvae"
PG_UNAGI = r"F:/UNAGI_Project/UNAGI/pg_unagi"
KCA = os.path.join(PAPER1, "kappa_c_audit")
DATA = (r"F:/UNAGI_Project/UNAGI/result/R14/R14_hemato_vanilla/"
        r"iter_4/stagedata/dataset.h5ad")

sys.path.insert(0, CONTRASTVAE)
sys.path.insert(0, PG_UNAGI)
sys.path.insert(0, KCA)

import torch  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from kappa_c_audit import audit  # noqa: E402

HEMA_TFS = ["GATA1", "KLF1", "TAL1", "GFI1B", "GATA2", "LMO2",
            "NFE2", "SPI1", "RUNX1", "CEBPA", "PAX5"]

ARMS = {
    "P_on":    ("real", 0.01),
    "P_off":   ("real", 0.0),
    "P_wrong": ("perm", 0.01),
}
SEEDS = [42, 123, 456]


# ----------------------------------------------------------------------
# data / prior
# ----------------------------------------------------------------------
def load_data(n_cells: int):
    """Deterministic cell subsample, identical for every arm (paired design)."""
    import anndata as ad
    from scipy.sparse import issparse

    a = ad.read_h5ad(DATA, backed="r")
    n_total = a.n_obs
    genes = list(a.var_names)
    idx = np.random.default_rng(0).choice(n_total, size=min(n_cells, n_total),
                                          replace=False)
    idx.sort()
    # NOTE: a backed SparseDataset does NOT support fancy row indexing
    # (ValueError: operands could not be broadcast).  Read the whole sparse
    # matrix, subset rows on the scipy object, only then densify.
    Xall = a.X[:]
    X = Xall[idx]
    X = X.toarray() if issparse(X) else np.asarray(X)
    X = np.log1p(X.astype(np.float32))
    del a, Xall
    gc.collect()
    print(f"  data: {X.shape[0]} / {n_total} cells x {X.shape[1]} genes  "
          f"(subsample rng=0, log1p)")
    return X, genes


def build_skeletons(genes, kind: str):
    from tf_database import build_tf_database

    tf_idx, tf_names = [], []
    up = [g.upper() for g in genes]
    for tf in HEMA_TFS:
        if tf.upper() in up:
            tf_idx.append(up.index(tf.upper()))
            tf_names.append(tf)
    print(f"  TFs matched: {len(tf_names)}/{len(HEMA_TFS)} -> {tf_names}")

    db = build_tf_database(tf_names, force_rebuild=False)
    skel = np.zeros((len(tf_idx), len(genes)), dtype=np.float32)
    for i, tf in enumerate(tf_names):
        for tgt in db.get(tf, []):
            tu = tgt.upper()
            if tu in up:
                skel[i, up.index(tu)] = 1.0
    print(f"  real skeleton: {int(skel.sum())} edges "
          f"({skel.sum() / skel.size * 100:.3f}% of matrix)")

    if kind == "real":
        return skel, tf_idx, tf_names, db

    # permuted skeleton: preserve each TF's edge count, relocate the targets
    rng = np.random.default_rng(1)
    perm = np.zeros_like(skel)
    for i in range(skel.shape[0]):
        n = int(skel[i].sum())
        if n:
            perm[i, rng.choice(skel.shape[1], size=n, replace=False)] = 1.0
    print(f"  permuted skeleton: {int(perm.sum())} edges (matched per-TF counts)")
    return perm, tf_idx, tf_names, db


# ----------------------------------------------------------------------
# train one arm / seed
# ----------------------------------------------------------------------
def run_one(arm, seed, X, genes, tf_idx, db, epochs, batch=512, device="cuda"):
    from model import ContrastVAE, train_contrastvae

    kind, lam = ARMS[arm]
    skel, tf_idx, tf_names, _ = build_skeletons(genes, kind)

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    Xt = torch.tensor(X, dtype=torch.float32)
    loader = DataLoader(TensorDataset(Xt), batch_size=batch, shuffle=True,
                        num_workers=0, pin_memory=(device == "cuda"),
                        drop_last=True)

    model = ContrastVAE(n_genes=X.shape[1], n_tf=len(tf_idx), n_latent=64,
                        n_hidden=256, n_layers=2, prior_skeleton=skel,
                        tf_indices=tf_idx, vanilla_gap=478.0,
                        lambda_kappa=0.1, lambda_alpha=0.1, lambda_sparse=0.01,
                        lambda_graph=lam, target_dim=46)
    model = model.to(device)

    t0 = time.time()
    hist = train_contrastvae(model, loader, n_epochs=epochs, lr=1e-3,
                             beta_start=0.1, beta_end=1.0,
                             log_interval=max(1, epochs // 4),
                             save_dir=None, device=device)
    dt = time.time() - t0

    # ---- extract latent (encoder mean) and GRN W ----
    model.eval()
    mus = []
    with torch.no_grad():
        for i in range(0, X.shape[0], 2048):
            xb = Xt[i:i + 2048].to(device)
            mu, _ = model.encoder(xb)
            mus.append(mu.cpu().numpy())
    Z = np.concatenate(mus, 0)
    W = model.grn.get_W().cpu().numpy()

    # ---- intervention-effectiveness diagnostics (did the manipulation bite?) ----
    off = float(np.abs(W * (1 - skel)).sum())
    on = float(np.abs(W * skel).sum())
    diag = dict(
        arm=arm, seed=seed, skeleton=kind, lambda_graph=lam,
        minutes=dt / 60.0, epochs=epochs,
        off_skeleton_l1=off, on_skeleton_l1=on,
        off_fraction=off / (off + on + 1e-12),
        W_fro=float(np.linalg.norm(W)),
        model_kappa=float(hist["kappa"][-1]), model_alpha=float(hist["alpha"][-1]),
        final_recon=float(hist["recon"][-1]), final_kl=float(hist["kl"][-1]),
        Z_shape=list(Z.shape),
    )

    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(os.path.join(OUT, f"{arm}_s{seed}.npz"),
                        Z=Z.astype(np.float32), W=W, skel=skel)
    del Xt, loader, model
    torch.cuda.empty_cache()
    gc.collect()
    return diag


# ----------------------------------------------------------------------
# analysis
# ----------------------------------------------------------------------
def analyse(X):
    runs = {}
    for arm in ARMS:
        for s in SEEDS:
            p = os.path.join(OUT, f"{arm}_s{s}.npz")
            if os.path.exists(p):
                d = np.load(p)
                runs[(arm, s)] = dict(Z=d["Z"], W=d["W"])

    def pair(a, b):
        za = runs[a]["Z"].astype(np.float64)
        zb = runs[b]["Z"].astype(np.float64)
        r = audit(za, X, zb, X)
        return r

    rows = []
    # null: P_off vs P_off, cross-seed
    offs = [s for s in SEEDS if ("P_off", s) in runs]
    for i in range(len(offs)):
        for j in range(i + 1, len(offs)):
            r = pair(("P_off", offs[i]), ("P_off", offs[j]))
            rows.append(dict(kind="null (P_off vs P_off)", a=f"s{offs[i]}",
                             b=f"s{offs[j]}", kappa_C_W=r["kappa_C_W"],
                             C_percent=r["C_percent"], verdict=r["verdict"],
                             kappa_C_Z=r["kappa_C_Z"],
                             d_eff_rank=r["eff_rank_change_percent"]))
    # prior arms vs the SAME seed's P_off
    for arm in ("P_on", "P_wrong"):
        for s in SEEDS:
            if (arm, s) not in runs or ("P_off", s) not in runs:
                continue
            r = pair(("P_off", s), (arm, s))
            rows.append(dict(kind=f"{arm} vs P_off", a=f"s{s}", b=f"s{s}",
                             kappa_C_W=r["kappa_C_W"], C_percent=r["C_percent"],
                             verdict=r["verdict"], kappa_C_Z=r["kappa_C_Z"],
                             d_eff_rank=r["eff_rank_change_percent"]))
    return rows


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", type=int, default=15000)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--arms", nargs="*", default=None)
    ap.add_argument("--analyse-only", action="store_true")
    args = ap.parse_args()

    seeds = args.seeds or SEEDS
    arms = args.arms or list(ARMS)
    if args.probe:
        seeds, arms, args.epochs = [42], ["P_on"], 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(OUT, exist_ok=True)
    print(f"device={device}  arms={arms}  seeds={seeds}  epochs={args.epochs}")

    X, genes = load_data(args.cells)

    diags = []
    if not args.analyse_only:
        for arm in arms:
            for s in seeds:
                print(f"\n=== {arm} seed={s} ===")
                d = run_one(arm, s, X, genes, None, None, args.epochs, device=device)
                print(f"  {d['minutes']:.2f} min | off-skel |W|={d['off_skeleton_l1']:.3g} "
                      f"({d['off_fraction']*100:.1f}% of mass) | ||W||={d['W_fro']:.4g} | "
                      f"alpha={d['model_alpha']:.4f}")
                diags.append(d)
                with open(os.path.join(OUT, "run_diagnostics.json"), "w") as f:
                    json.dump(diags, f, indent=1)

    rows = analyse(X)
    print("\n" + "=" * 78)
    print(f"{'comparison':<26}{'kappa_C_W':>11}{'C%':>9}{'verdict':>22}{'dEffRank':>11}")
    print("-" * 78)
    for r in rows:
        print(f"{r['kind'] + ' ' + r['a'] + '/' + r['b']:<26}"
              f"{r['kappa_C_W']:>11.3f}{r['C_percent']:>9.1f}"
              f"{r['verdict']:>22}{r['d_eff_rank']:>11.1f}")
    with open(os.path.join(OUT, "s1_audit_rows.json"), "w") as f:
        json.dump(rows, f, indent=1)
    print("\nsaved ->", OUT)


if __name__ == "__main__":
    main()
