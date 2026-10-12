"""P0-1  Synthetic positive control for the kappa-C diagnostic.

MOTIVATION
----------
Paper 1's diagnostic was validated only on NEGATIVE cases: the historical G-VAE
21/21 positive control was withdrawn as non-reproducible, so the tool has never
been shown to fire on a case where the decoder readout *genuinely* changes. A
reviewer can therefore ask: "does kappa-C ever say YES?"  This script answers
that on synthetic data where ground truth is known exactly.

ANALYTIC NOTE -- what kappa-C(W) actually measures
--------------------------------------------------
ridge_readout standardises Z and X per column/gene. For the exact linear model
X = W Z it follows that every row of the standardised readout has unit norm:
    sum_j What[g,j]^2 = sum_j W[g,j]^2 / sum_j' W[g,j']^2 = 1,
hence kappa-C(W) = (sum_gj What_1^2)/(sum_gj What_0^2) = 1 IDENTICALLY, for ANY
W and ANY latent covariance. With additive noise, sum_j What[g,j]^2 = R2_g, so
    kappa-C(W) = sum_g R2_g(perturbed) / sum_g R2_g(vanilla).
i.e. kappa-C(W) is the aggregate latent->expression PREDICTABILITY RATIO; it is
INVARIANT to per-gene rescaling of the decoder and to attenuation of latent->gene
directions. Two consequences drive the design below:
  (1) a valid sensitivity positive control must move PREDICTABILITY;
  (2) the invariance is itself a reportable property of the metric (invariance
      probe), and it WEAKENS the reading of the headline decoder null.

ARMS
----
  sensitivity  -- perturbed model's latent predicts its expression with relative
                  R^2 = level. kappa-C(W) must FALL and cross the lock line.
  specificity  -- only the LATENT covariance is reorganised (W and predictability
                  unchanged). kappa-C(W) must STAY in band; kappa-C(Z) moves.
  invariance   -- latent->gene directions attenuated (per-gene structure kept).
                  kappa-C(W) must STAY at 1 -- documents the analytic property.

USAGE:  python kappa_c_positive_control.py
"""
import os
import json
import numpy as np

import perturbation_benchmark as pb
from figure_palette import apply_paper_style, C_HEMA, C_LUNG, C4, RED, GRAY

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 20260912
G_GENES = 2000
K_LATENT = 30
N_CELLS = 8000
N_REP = 20
COMPRESS_LEVELS = [0.0, 0.25, 0.5, 0.75, 0.95]   # fraction of directions changed
R2_LEVELS = [1.0, 0.9, 0.75, 0.5, 0.3, 0.15]      # relative predictability
TAU = 0.05
VANILLA_BAND = (0.937, 1.064)
LOCKED = 0.5

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_JSON = os.path.join(HERE, "..", "data", "kappa_c_positive_control.json")
OUT_FIG = os.path.join(HERE, "..", "figures", "fig_positive_control.pdf")


def make_decoder(rng, k=K_LATENT, g=G_GENES):
    W = rng.standard_normal((g, k))
    W /= np.linalg.norm(W, axis=0, keepdims=True)
    return W


def draw_latent(rng, n, k, cov=None):
    if cov is None:
        return rng.standard_normal((n, k))
    return rng.standard_normal((n, k)) @ np.linalg.cholesky(cov).T


def stats(Z0, X0, Z1, X1):
    W0 = pb.ridge_readout(Z0, X0)
    W1 = pb.ridge_readout(Z1, X1)
    kCW = float((W1 ** 2).sum() / (W0 ** 2).sum())
    kCZ = pb.kappa_C_Z(Z0, Z1)
    er0 = pb.eff_rank(np.cov(Z0, rowvar=False))
    er1 = pb.eff_rank(np.cov(Z1, rowvar=False))
    return kCW, kCZ, er0, er1


def _summ(rows, **extra):
    r = np.array(rows)
    d = dict(kCW_mean=float(r[:, 0].mean()),
             kCW_lo=float(np.percentile(r[:, 0], 2.5)),
             kCW_hi=float(np.percentile(r[:, 0], 97.5)),
             kCZ_mean=float(r[:, 1].mean()),
             er_ratio_mean=float((r[:, 3] / r[:, 2]).mean()))
    d.update(extra)
    return d


def main():
    rng = np.random.default_rng(SEED)
    W0 = make_decoder(rng)

    # ------------------------------------------------------------------ null
    nkCW, nkCZ = [], []
    for _ in range(N_REP):
        Z0 = draw_latent(rng, N_CELLS, K_LATENT); X0 = Z0 @ W0.T
        Z1 = draw_latent(rng, N_CELLS, K_LATENT); X1 = Z1 @ W0.T   # same process
        a, b, _, _ = stats(Z0, X0, Z1, X1); nkCW.append(a); nkCZ.append(b)
    nkCW = np.array(nkCW); nkCZ = np.array(nkCZ)
    band = (float(np.percentile(nkCW, 2.5)), float(np.percentile(nkCW, 97.5)))

    # --------------------- sensitivity: perturbed latent less predictive of X
    pred = []
    for r2 in R2_LEVELS:
        rows = []
        for _ in range(N_REP):
            Z0 = draw_latent(rng, N_CELLS, K_LATENT); X0 = Z0 @ W0.T
            Z1 = draw_latent(rng, N_CELLS, K_LATENT)      # latent cov UNCHANGED
            sig = Z1 @ W0.T
            v = float((sig ** 2).mean())
            sd = float(np.sqrt(max(v * (1.0 / max(r2, 1e-6) - 1.0), 0.0)))
            X1 = sig + sd * rng.standard_normal(sig.shape)
            rows.append(stats(Z0, X0, Z1, X1))
        pred.append(_summ(rows, target_r2=r2))

    # ------------------------------------------------- specificity: latent only
    spec = []
    for c in COMPRESS_LEVELS:
        rows = []
        for _ in range(N_REP):
            n_cut = int(round(c * K_LATENT)); ev = np.ones(K_LATENT)
            if n_cut > 0:
                ev[-n_cut:] = TAU
            Z0 = draw_latent(rng, N_CELLS, K_LATENT); X0 = Z0 @ W0.T
            Z1 = draw_latent(rng, N_CELLS, K_LATENT, cov=np.diag(ev)); X1 = Z1 @ W0.T
            rows.append(stats(Z0, X0, Z1, X1))
        spec.append(_summ(rows, level=c))

    # -------------------------------------- invariance: attenuate latent->gene
    inv = []
    for c in COMPRESS_LEVELS:
        rows = []
        for _ in range(N_REP):
            n_cut = int(round(c * K_LATENT)); m = np.ones(K_LATENT)
            if n_cut > 0:
                m[-n_cut:] = TAU
            W1 = W0 * m[None, :]
            Z0 = draw_latent(rng, N_CELLS, K_LATENT); X0 = Z0 @ W0.T
            Z1 = draw_latent(rng, N_CELLS, K_LATENT); X1 = Z1 @ W1.T
            rows.append(stats(Z0, X0, Z1, X1))
        inv.append(_summ(rows, level=c))

    lock_level = next((p["target_r2"] for p in pred if p["kCW_mean"] < LOCKED), None)
    out = dict(
        seed=SEED, n_genes=G_GENES, k_latent=K_LATENT, n_cells=N_CELLS, n_rep=N_REP,
        protocol="perturbation_benchmark.ridge_readout / kappa_C_Z / eff_rank (unchanged)",
        vanilla_band=list(VANILLA_BAND), locked_threshold=LOCKED,
        metric_identity="kappa-C(W) = sum_g R2_g(pert) / sum_g R2_g(vanilla)  "
                        "(invariant to per-gene rescaling and latent-direction attenuation)",
        null=dict(kCW_mean=float(nkCW.mean()), kCW_std=float(nkCW.std()),
                  kCW_band=list(band), kCZ_mean=float(nkCZ.mean()),
                  kCZ_std=float(nkCZ.std())),
        sensitivity_predictability=pred,
        specificity_latent=spec,
        invariance_probe=inv,
        verdict=dict(
            sensitivity_passes=bool(lock_level is not None and
                                    pred[-1]["kCW_mean"] < pred[0]["kCW_mean"]),
            specificity_passes=bool(all(band[0] <= s["kCW_mean"] <= band[1] for s in spec)),
            invariance_confirmed=bool(all(band[0] <= s["kCW_mean"] <= band[1] for s in inv)),
            lock_crossed_at_r2=lock_level,
        ),
    )
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)

    # ------------------------------------------------------------- figure
    apply_paper_style()
    fig, ax = plt.subplots(1, 3, figsize=(13.2, 3.6))
    kw = dict(capsize=3, lw=1.4, markersize=5)

    ax[0].axhspan(*VANILLA_BAND, color="0.9", zorder=0)
    ax[0].axhline(LOCKED, ls="--", lw=1, color=RED)
    ax[0].errorbar([p["target_r2"] for p in pred], [p["kCW_mean"] for p in pred],
                   yerr=[[p["kCW_mean"] - p["kCW_lo"] for p in pred],
                         [p["kCW_hi"] - p["kCW_mean"] for p in pred]],
                   marker="o", color=C_HEMA, **kw)
    ax[0].set_xlabel("relative predictability $R^2$ of perturbed model")
    ax[0].set_ylabel(r"$\kappa$-$C(W)$")
    ax[0].set_title("Sensitivity: fires when predictability drops")

    ax[1].axhspan(*VANILLA_BAND, color="0.9", zorder=0)
    ax[1].errorbar([s["level"] for s in spec], [s["kCW_mean"] for s in spec],
                   yerr=[[s["kCW_mean"] - s["kCW_lo"] for s in spec],
                         [s["kCW_hi"] - s["kCW_mean"] for s in spec]],
                   marker="s", color=C_LUNG, label=r"$\kappa$-$C(W)$", **kw)
    ax[1].errorbar([s["level"] for s in spec], [s["kCZ_mean"] for s in spec],
                   marker="^", color=C4, label=r"$\kappa$-$C(Z)$", **kw)
    ax[1].set_xlabel("fraction of latent directions contracted")
    ax[1].set_ylabel(r"$\kappa$-$C$")
    ax[1].set_title("Specificity: latent-only change is invisible to W")
    ax[1].legend(frameon=False, fontsize=8)

    ax[2].axhspan(*VANILLA_BAND, color="0.9", zorder=0)
    ax[2].errorbar([s["level"] for s in inv], [s["kCW_mean"] for s in inv],
                   yerr=[[s["kCW_mean"] - s["kCW_lo"] for s in inv],
                         [s["kCW_hi"] - s["kCW_mean"] for s in inv]],
                   marker="D", color=GRAY, **kw)
    ax[2].set_xlabel("fraction of latent$\\to$gene directions attenuated")
    ax[2].set_ylabel(r"$\kappa$-$C(W)$")
    ax[2].set_title("Invariance: attenuation leaves $W$ at 1")

    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT_FIG), exist_ok=True)
    fig.savefig(OUT_FIG, bbox_inches="tight")
    fig.savefig(OUT_FIG.replace(".pdf", ".png"), dpi=160, bbox_inches="tight")

    print(json.dumps(out["null"], indent=2))
    print("VERDICT:", json.dumps(out["verdict"], indent=2))
    for p in pred:
        print("sens  relR2=%.2f  kCW=%.3f [%.3f,%.3f]  kCZ=%.3f"
              % (p["target_r2"], p["kCW_mean"], p["kCW_lo"], p["kCW_hi"], p["kCZ_mean"]))
    for s in spec:
        print("spec  lvl=%.2f  kCW=%.3f  kCZ=%.3f  er_ratio=%.3f"
              % (s["level"], s["kCW_mean"], s["kCZ_mean"], s["er_ratio_mean"]))
    for s in inv:
        print("inv   lvl=%.2f  kCW=%.3f [%.3f,%.3f]"
              % (s["level"], s["kCW_mean"], s["kCW_lo"], s["kCW_hi"]))
    print("[saved]", os.path.abspath(OUT_JSON))
    print("[saved]", os.path.abspath(OUT_FIG))


if __name__ == "__main__":
    main()
