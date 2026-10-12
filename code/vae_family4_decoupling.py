"""Family-4 test: does the decoder/latent spectral decoupling reproduce in a
plain, self-contained PyTorch VAE?

WHY A HAND-WRITTEN VAE
----------------------
The named alternatives are unavailable or unsafe here:
  * trVAE 1.1.2 pins tensorflow==1.15.2 (impossible on Python 3.13)
  * scGen 2.1.0 imports `scvi._compat`, removed in the installed scvi-tools 1.4.3
  * scarches would move the scvi-tools version and risk breaking the environment
    that produced every published number, so it is deliberately not installed.
A minimal PyTorch VAE is therefore used as an unambiguous FURTHER architecture:
no library parameterisation, Gaussian MSE likelihood, plain MLP, no batch-norm
or dropout -- structurally distinct from both UNAGI/G-VAE (GRN-masked decoder)
and scVI (NB likelihood, library-specific encoder/decoder).

A DIFFERENT CONDITIONING MECHANISM (the interesting part)
---------------------------------------------------------
scVI in the companion benchmark conditions via a categorical covariate fed to
BOTH encoder and decoder. Here the perturbation identity is injected as a
learned condition embedding concatenated to the DECODER INPUT ONLY:
        decoder input = [z, c]        (encoder never sees c)
This is a strictly decoder-side structural prior, so it is a direct probe of the
manuscript's central claim: when structure is injected on the decoder side, does
the LATENT spectrum still reorganise?

BETA SWEEP + LATENT HEALTH GATE
-------------------------------
A plain VAE on 40k x 2000 expression is prone to two failure modes that both
produce uninterpretable kappa-C(Z):
  * posterior collapse  (beta too large -> latent carries no signal,
                          effective rank -> ~1 of 30, Ridge R^2 ~ 0)
  * latent explosion    (beta too small -> no KL pressure, latent variance
                          blows up, kappa-C(Z) -> astronomical)
So beta is swept and every run is gated on explicit health checks: the latent
must keep a VAE-like effective rank AND actually predict expression (Ridge R^2).
Only runs passing the gate are reported as scientific results.

USAGE
-----
    LOCAL_PERT_H5AD=<norman.h5ad> PERT_COL=condition_name \
    CONTROL_VAL=A549_ctrl_1 python vae_family4_decoupling.py
    # optional: VAE4_BETAS="1.0,0.5,0.1"  VAE4_EPOCHS=80
"""

import os
import sys
import json
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import Ridge

import perturbation_benchmark as pb

# ---------------------------------------------------------------------------
SEED = 42
LATENT = 30
HIDDEN = 512
COND_EMB = 16
EPOCHS = int(os.environ.get("VAE4_EPOCHS", "80"))
BATCH = 512
LR = 1e-3
BETAS = [float(b) for b in os.environ.get("VAE4_BETAS", "1.0,0.5,0.1").split(",")]
# KL warm-up: the canonical fix for posterior collapse. beta is ramped linearly
# from 0 to its target over WARMUP epochs, so the encoder first learns to use z
# before the KL penalty is applied at full strength.
WARMUP = int(os.environ.get("VAE4_WARMUP", "30"))
STANDARDIZE = True      # per-gene z-score so MSE scale is comparable to KL

OUT_JSON = os.path.join(os.path.dirname(__file__), "..", "data",
                        "vae_family4_results.json")

torch.manual_seed(SEED)
np.random.seed(SEED)


# ---------------------------------------------------------------------------
class VAE(nn.Module):
    """Plain MLP VAE. If n_cond > 0, a condition embedding is concatenated to
    the DECODER input only (encoder is condition-blind)."""

    def __init__(self, n_genes, n_latent=LATENT, hidden=HIDDEN,
                 n_cond=0, cond_emb=COND_EMB):
        super().__init__()
        self.n_cond = n_cond
        self.cond = nn.Embedding(n_cond, cond_emb) if n_cond > 0 else None
        self.enc = nn.Sequential(
            nn.Linear(n_genes, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
        )
        self.mu = nn.Linear(hidden // 2, n_latent)
        self.lv = nn.Linear(hidden // 2, n_latent)
        # Signal-to-noise fix. With the default init, lv ~ 0 => std ~ 1 while
        # mu has std ~0.2, so z is mostly noise; the decoder cannot extract any
        # signal, degenerates to predicting the mean (MSE = data variance), and
        # the encoder then receives no gradient -- a deadlock that looks exactly
        # like "posterior collapse". Starting with a small std (bias -4) makes
        # z ~ mu informative from step one, so the decoder can start learning.
        nn.init.zeros_(self.lv.weight)
        nn.init.constant_(self.lv.bias, -4.0)
        dec_in = n_latent + (cond_emb if n_cond > 0 else 0)
        self.dec = nn.Sequential(
            nn.Linear(dec_in, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, hidden), nn.ReLU(),
            nn.Linear(hidden, n_genes),
        )

    def encode(self, x):
        h = self.enc(x)
        return self.mu(h), self.lv(h)

    def reparam(self, mu, lv):
        std = torch.exp(0.5 * lv)
        return mu + torch.randn_like(std) * std

    def decode(self, z, c=None):
        if self.cond is not None and c is not None:
            z = torch.cat([z, self.cond(c)], dim=1)
        return self.dec(z)

    def forward(self, x, c=None):
        mu, lv = self.encode(x)
        z = self.reparam(mu, lv)
        return self.decode(z, c), mu, lv


def loss_fn(recon, x, mu, lv, beta):
    mse = nn.functional.mse_loss(recon, x, reduction="mean")
    kld = -0.5 * torch.mean(1 + lv - mu.pow(2) - lv.exp())
    return mse + beta * kld, mse, kld


@torch.no_grad()
def get_latent(model, X, device):
    model.eval()
    out = []
    for i in range(0, len(X), 4096):
        xb = torch.as_tensor(X[i:i + 4096], dtype=torch.float32, device=device)
        mu, _ = model.encode(xb)
        out.append(mu.cpu().numpy())
    return np.vstack(out)


def train_model(X, cond_idx, n_cond, device, tag, beta):
    model = VAE(X.shape[1], n_latent=LATENT, n_cond=n_cond).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    ds = TensorDataset(torch.as_tensor(X, dtype=torch.float32),
                       torch.as_tensor(cond_idx, dtype=torch.long))
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True, drop_last=True)
    model.train()
    for ep in range(EPOCHS):
        beta_t = beta * min(1.0, (ep + 1) / float(max(WARMUP, 1)))
        tot = mse_t = kld_t = 0.0
        for xb, cb in dl:
            xb, cb = xb.to(device), cb.to(device)
            recon, mu, lv = model(xb, cb)
            loss, mse, kld = loss_fn(recon, xb, mu, lv, beta_t)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item(); mse_t += mse.item(); kld_t += kld.item()
        if (ep + 1) % 20 == 0 or ep == 0:
            print("  [%s b=%.2f] epoch %3d/%d loss=%.4f (mse=%.4f kld=%.4f)"
                  % (tag, beta, ep + 1, EPOCHS, tot / len(dl),
                     mse_t / len(dl), kld_t / len(dl)), flush=True)
    return model


def evaluate(Zv, Zp, X):
    """kappa-C(W), kappa-C(Z), effective ranks + latent-health gate."""
    W0 = pb.ridge_readout(Zv, X)
    W1 = pb.ridge_readout(Zp, X)
    kCW = float((W1 ** 2).sum() / (W0 ** 2).sum())
    kCZ = pb.kappa_C_Z(Zv, Zp)
    er0 = pb.eff_rank(np.cov(Zv, rowvar=False))
    er1 = pb.eff_rank(np.cov(Zp, rowvar=False))
    r2v = float(Ridge(alpha=1.0).fit(Zv, X).score(Zv, X))
    r2p = float(Ridge(alpha=1.0).fit(Zp, X).score(Zp, X))
    sv, sp = float(Zv.std()), float(Zp.std())
    # Gate calibrated against the scVI reference on the SAME data: scVI achieves
    # Ridge R^2 = 0.033 (vanilla) / 0.030 (conditioned) with effective rank
    # 8.80 / 6.15. So an absolute R^2 > 0.05 threshold would reject scVI itself.
    # A plain VAE is called usable if it clears a quarter of scVI's R^2 AND keeps
    # a VAE-like effective rank AND has not let the latent variance explode.
    R2_MIN = 0.008
    healthy = bool(er0 > 3.0 and er1 > 3.0 and r2v > R2_MIN and r2p > R2_MIN
                   and sp < 10.0 * max(sv, 1e-8))
    return dict(kappa_C_W=kCW,
                C_percent=float((1.0 - kCW) * 100.0),
                kappa_C_Z=kCZ,
                eff_rank_vanilla=er0,
                eff_rank_conditioned=er1,
                eff_rank_change_percent=float((er1 / er0 - 1.0) * 100.0) if er0 else float("nan"),
                ridge_r2_vanilla=r2v,
                ridge_r2_conditioned=r2p,
                z_std_vanilla=sv,
                z_std_conditioned=sp,
                latent_usable=healthy)


def main():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[device] %s   betas=%s  epochs=%d" % (device, BETAS, EPOCHS))

    cache = pb._cache if hasattr(pb, "_cache") else os.path.join(
        os.path.dirname(__file__), "..", "data", "_norman_latents.npz")
    d = np.load(cache, allow_pickle=True)
    X = d["X"].astype(np.float32)
    labels = np.array([str(x) for x in d["pert_labels"]])
    if STANDARDIZE:
        mu = X.mean(0); sd = X.std(0); sd[sd < 1e-8] = 1.0
        X = ((X - mu) / sd).astype(np.float32)
        print("[data] per-gene z-scored")
    print("[data] X%s  n_cells=%d" % (X.shape, X.shape[0]))

    uniq = np.unique(labels)
    cond_of = {c: i for i, c in enumerate(uniq)}
    cond_idx = np.array([cond_of[l] for l in labels], dtype=np.int64)
    n_cond = len(uniq)
    print("[cond] %d conditions" % n_cond)

    runs = []
    for beta in BETAS:
        print("\n############ beta = %.2f ############" % beta)
        mv = train_model(X, cond_idx, 0, device, "vanilla", beta)
        Zv = get_latent(mv, X, device)
        mp = train_model(X, cond_idx, n_cond, device, "cond", beta)
        Zp = get_latent(mp, X, device)
        rec = evaluate(Zv, Zp, X)
        rec["beta"] = beta
        runs.append(rec)
        print("[beta=%.2f] kappa-C(W)=%.3f (C=%.1f%%)  kappa-C(Z)=%.3f  "
              "eff_rank %.2f->%.2f (%+.1f%%)  R2 %.4f/%.4f  usable=%s"
              % (beta, rec["kappa_C_W"], rec["C_percent"], rec["kappa_C_Z"],
                 rec["eff_rank_vanilla"], rec["eff_rank_conditioned"],
                 rec["eff_rank_change_percent"], rec["ridge_r2_vanilla"],
                 rec["ridge_r2_conditioned"], rec["latent_usable"]))

    usable = [r for r in runs if r["latent_usable"]]
    out = dict(
        architecture="self-contained PyTorch VAE (plain MLP, Gaussian MSE likelihood)",
        conditioning="perturbation-identity embedding concatenated to DECODER INPUT ONLY "
                     "(encoder condition-blind) -- distinct from scVI's categorical covariate",
        why_not_scgen="scGen 2.1.0 imports scvi._compat (removed in scvi-tools 1.4.3); "
                      "trVAE pins tensorflow==1.15.2 (Python 3.13 incompatible)",
        dataset="Norman 2019 GEARS",
        n_cells=int(X.shape[0]), n_genes=int(X.shape[1]),
        n_latent=LATENT, epochs=EPOCHS, n_conditions=int(n_cond),
        per_gene_standardized=bool(STANDARDIZE),
        health_gate=("latent usable iff eff_rank>3 for BOTH models AND Ridge R^2>0.008 "
                     "for both AND conditioned z_std < 10x vanilla z_std. The R^2 "
                     "floor is calibrated on the scVI reference, not chosen freely."),
        scvi_reference=dict(ridge_r2_vanilla=0.0333, ridge_r2_conditioned=0.0302,
                            eff_rank_vanilla=8.80, eff_rank_conditioned=6.15,
                            note="scVI on the same cells and genes; the plain VAE must "
                                 "be judged against these values, because an absolute "
                                 "R^2>0.05 threshold would reject scVI itself"),
        n_betas_tested=len(runs),
        n_betas_usable=len(usable),
        runs=runs,
        usable_runs=usable,
        elapsed_sec=float(time.time() - t0),
    )
    if usable:
        best = max(usable, key=lambda r: min(r["ridge_r2_vanilla"], r["ridge_r2_conditioned"]))
        out["selected"] = best
        print("\n===== FAMILY-4 RESULT (healthiest usable beta = %.2f) =====" % best["beta"])
        print("kappa-C(W) = %.3f  (C = %.1f%%)" % (best["kappa_C_W"], best["C_percent"]))
        print("kappa-C(Z) = %.3f" % best["kappa_C_Z"])
        print("effective rank %.2f -> %.2f (%+.1f%%)"
              % (best["eff_rank_vanilla"], best["eff_rank_conditioned"],
                 best["eff_rank_change_percent"]))
    else:
        out["selected"] = None
        print("\n===== FAMILY-4: NO beta produced a usable latent =====")
        print("kappa-C(Z) for a plain VAE on this data is NOT interpretable; "
              "see runs[] for the collapse/explosion evidence.")
        sys.stderr.write("[RESULT] no usable latent at any beta\n")

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("[saved] %s" % os.path.abspath(OUT_JSON))
    print("[done] %.1f s" % (time.time() - t0))


if __name__ == "__main__":
    main()
