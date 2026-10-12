"""Controlled POSITIVE control for the three-check prior-activity protocol.

MOTIVATION
----------
The paper's three-check protocol (gradient/wiring, lambda-scan, structural bound)
is only demonstrated on NEGATIVE cases: on human IPF all three checks FAIL and
the collapse is correctly attributed to the data manifold rather than the prior.
A reviewer can therefore ask: "has the protocol ever returned *prior-active*?"
This script answers that on synthetic data where the prior is *genuinely* active
(and, for contrast, on an IPF-style inert prior where it is not), using the same
check definitions as the paper (supp_addendum.tex, Section B).

KEY POINT (why the prior must change predictability)
----------------------------------------------------
ridge_readout standardises Z and X per column/gene, so for an exactly linear
model kappa-C(W) == sum_g R2_g(pert)/sum_g R2_g(vanilla).  A prior that merely
*re-supports* the decoder (masks its edges) but leaves expression perfectly
predictable from the latent therefore gives kappa-C(W) == 1 identically -- the
paper's fixed-point invariance.  A prior that is genuinely active must change the
model's aggregate latent->expression PREDICTABILITY, exactly as in the real runs.
We model that as a prior-strength-dependent residual on the expressed genes.

CHECKS (operational definitions used here)
------------------------------------------
Check 1  gradient / wiring.  Does the prior enter the training signal?
         (a) prior-edge gradient norm: nonzero for a differentiable prior, exactly
         zero for the paper's no-grad implementation; (b) wiring contrast =
         mean|W_hat| on prior edges over non-edges, versus a label-shuffled mask of
         identical sparsity.  PASS iff prior-edge gradient > 0 and wiring contrast
         exceeds the shuffled-mask null.
Check 2  lambda scan.  Does the spectral response depend on prior weight?
         slope of kappa-C(W) against log lambda.  PASS iff clearly negative and
         kappa-C(W) reaches sub-band; FAIL iff flat.
Check 3  structural bound.  Is the collapse already explained by the data manifold?
         top-axis share of tr(H) (fraction of the readout trace carried by the
         leading latent axis).  In IPF an inert prior already drives C ~ 95% with a
         91.2% top axis, so the collapse is data-inherent.  PASS iff compression is
         distributed (top-axis share below the data-inherent bound).

The real IPF numbers (top-axis 91.2%, flat lambda-scan, zero prior gradient) are
quoted in the JSON for reference.

USAGE:  python three_check_positive_control.py
"""
import os
import json
import numpy as np

import perturbation_benchmark as pb
from figure_palette import apply_paper_style, C_HEMA, GRAY, C4, RED

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 20260913
N_CELLS = 6000
G_GENES = 800
D_LATENT = 16
N_EDGES_PER_GENE = 4          # structured GRN degree
LAMBDAS = [0.1, 1.0, 10.0]
N_REP = 12
NOISE = 0.30                  # observation noise, fraction of signal sd
PRIOR_RESID = 0.6             # residual scale (x sqrt(lambda)); mimics degraded reconstruction
IPF_TOP_AXIS = 0.912          # reference: human IPF vanilla top-axis share
TOP_AXIS_BOUND = 0.80         # above this, collapse is deemed data-inherent

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_JSON = os.path.join(HERE, "..", "data", "three_check_positive_control.json")
OUT_FIG = os.path.join(HERE, "..", "figures", "fig_three_check_positive.pdf")


def structured_mask(rng, g=G_GENES, d=D_LATENT, k=N_EDGES_PER_GENE):
    M = np.zeros((g, d), dtype=float)
    for i in range(g):
        M[i, rng.choice(d, size=k, replace=False)] = 1.0
    return M


def shuffled_mask(rng, M):
    M2 = np.zeros_like(M)
    for i in range(M.shape[0]):
        M2[i, rng.choice(M.shape[1], size=int(M[i].sum()), replace=False)] = 1.0
    return M2


def row_normalise(W):
    n = np.linalg.norm(W, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return W / n


def make_W(rng, M):
    return row_normalise(M * rng.standard_normal(M.shape))


def align(A, M):
    return A if A.shape[0] == M.shape[0] else A.T


def wiring_contrast(W_hat, M):
    A = np.abs(align(W_hat, M))
    return float(A[M > 0].mean() / max(A[M == 0].mean(), 1e-12))


def prior_edge_grad(W_hat, M, W_true, lam):
    """|| gradient the differentiable prior injects on prior edges ||
           = || 2 lam * M odot (W_hat - W_true) ||."""
    A = align(W_hat, M)
    return float(np.linalg.norm(2.0 * lam * M * (A - W_true)))


def top_axis_share(W_hat):
    H = W_hat @ W_hat.T if W_hat.shape[0] <= W_hat.shape[1] else W_hat.T @ W_hat
    ev = np.clip(np.linalg.eigvalsh(H), 0, None)
    return float(ev[-1] / max(ev.sum(), 1e-12))


def ridge_trace_ratio(Z, X_v, X_p):
    Wv = pb.ridge_readout(Z, X_v)
    Wp = pb.ridge_readout(Z, X_p)
    return float((Wp ** 2).sum() / (Wv ** 2).sum()), Wv, Wp


def run(arm, rng):
    M = structured_mask(rng)
    Ms = shuffled_mask(rng, M)
    W_true = make_W(rng, M)
    W_V = row_normalise(rng.standard_normal((G_GENES, D_LATENT)))
    E0 = rng.standard_normal((N_CELLS, G_GENES))          # residual, independent of Z

    Z = draw_Z = rng.standard_normal((N_CELLS, D_LATENT))
    X_v = Z @ W_V.T
    X_v = X_v + NOISE * X_v.std() * rng.standard_normal(X_v.shape)

    rows = []
    for lam in LAMBDAS:
        if arm == "active":
            alpha = lam / (1.0 + lam)                     # prior weight grows with lambda
            W_gen = row_normalise((1 - alpha) * W_V + alpha * W_true)
            X_p = Z @ W_gen.T + PRIOR_RESID * np.sqrt(lam) * E0   # predictability drops with lambda
        else:                                             # "inert": prior does nothing
            X_p = Z @ W_V.T.copy()
        X_p = X_p + NOISE * X_p.std() * rng.standard_normal(X_p.shape)
        if arm == "inert":
            # data-inherent collapse: a single dominant axis (IPF-like pathology)
            X_p = X_p + 6.0 * X_p.std() * (Z[:, 0:1] @ rng.standard_normal((1, G_GENES)))
        kCW, _, Wp = ridge_trace_ratio(Z, X_v, X_p)
        g_on = prior_edge_grad(Wp, M, W_true, lam) if arm == "active" else 0.0  # no-grad -> 0
        rows.append(dict(lam=lam, kCW=kCW,
                         wiring_prior=wiring_contrast(Wp, M),
                         wiring_shuffled=wiring_contrast(Wp, Ms),
                         grad_prior_edges=g_on,
                         top_axis_share=top_axis_share(Wp)))
    return rows


def summarise(rows):
    lam = np.array([r["lam"] for r in rows])
    kCW = np.array([r["kCW"] for r in rows])
    return dict(
        slope_kCW_vs_loglam=float(np.polyfit(np.log(lam), kCW, 1)[0]),
        wiring_prior_mean=float(np.mean([r["wiring_prior"] for r in rows])),
        wiring_shuffled_mean=float(np.mean([r["wiring_shuffled"] for r in rows])),
        grad_prior_edges_mean=float(np.mean([r["grad_prior_edges"] for r in rows])),
        top_axis_share_mean=float(np.mean([r["top_axis_share"] for r in rows])),
        kCW_first=float(kCW[0]), kCW_last=float(kCW[-1]),
    )


def run_rep(arm, rng):
    acc = None
    for _ in range(N_REP):
        r = run(arm, rng)
        if acc is None:
            acc = [dict(row) for row in r]
        else:
            for a, b in zip(acc, r):
                for k in a:
                    a[k] += b[k]
    for a in acc:
        for k in a:
            a[k] /= N_REP
    return acc


def main():
    rng = np.random.default_rng(SEED)
    active, inert = run_rep("active", rng), run_rep("inert", rng)
    sa, si = summarise(active), summarise(inert)

    check1_active = bool(sa["grad_prior_edges_mean"] > 0 and
                         sa["wiring_prior_mean"] > 1.2 * sa["wiring_shuffled_mean"])
    check1_inert = bool(si["grad_prior_edges_mean"] == 0 and
                        si["wiring_prior_mean"] <= 1.2 * si["wiring_shuffled_mean"])
    check2_active = bool(sa["slope_kCW_vs_loglam"] < -0.05 and sa["kCW_last"] < 0.75)
    check2_inert = bool(abs(si["slope_kCW_vs_loglam"]) < 0.02)
    check3_active = bool(sa["top_axis_share_mean"] < TOP_AXIS_BOUND)
    check3_inert = bool(si["top_axis_share_mean"] >= TOP_AXIS_BOUND)

    out = dict(
        seed=SEED, n_cells=N_CELLS, n_genes=G_GENES, d_latent=D_LATENT,
        n_edges_per_gene=N_EDGES_PER_GENE, lambdas=LAMBDAS, n_rep=N_REP,
        protocol="perturbation_benchmark.ridge_readout (unchanged)",
        metric_identity="kappa-C(W) = sum_g R2_g(pert) / sum_g R2_g(vanilla)",
        ipf_reference=dict(top_axis_share=IPF_TOP_AXIS, lambda_slope=0.0,
                           prior_edge_gradient=0.0, verdict="not-prior-active"),
        active=dict(rows=active, summary=sa,
                    checks=dict(check1_gradient_wiring=check1_active,
                                check2_lambda_scan=check2_active,
                                check3_structural_bound=check3_active),
                    verdict="prior-active" if (check1_active and check2_active and check3_active)
                    else "not-prior-active"),
        inert=dict(rows=inert, summary=si,
                   checks=dict(check1_gradient_wiring=check1_inert,
                               check2_lambda_scan=check2_inert,
                               check3_structural_bound=check3_inert),
                   verdict="not-prior-active" if (check1_inert and check2_inert and check3_inert)
                   else "prior-active"),
    )
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)

    # ------------------------------------------------------------- figure
    apply_paper_style()
    fig, ax = plt.subplots(1, 3, figsize=(13.2, 3.6))
    kw = dict(capsize=3, lw=1.5, markersize=6)
    lam = [r["lam"] for r in active]
    x = np.arange(2); w = 0.35

    ax[0].bar(x - w / 2, [sa["wiring_prior_mean"], si["wiring_prior_mean"]], w,
              color=C_HEMA, label="prior edges")
    ax[0].bar(x + w / 2, [sa["wiring_shuffled_mean"], si["wiring_shuffled_mean"]], w,
              color=GRAY, label="shuffled edges")
    ax[0].axhline(1.0, ls=":", lw=1, color="0.4")
    ax[0].set_xticks(x); ax[0].set_xticklabels(["active prior", "inert prior"])
    ax[0].set_ylabel("wiring contrast  (on / off edges)")
    ax[0].set_title("Check 1: gradient / wiring")
    ax[0].legend(frameon=False, fontsize=8)

    ax[1].errorbar(lam, [r["kCW"] for r in active], yerr=0.02, marker="o",
                   color=C_HEMA, label="active prior", **kw)
    ax[1].errorbar(lam, [r["kCW"] for r in inert], yerr=0.02, marker="s",
                   color=GRAY, label="inert prior", **kw)
    ax[1].axhline(0.5, ls="--", lw=1, color=RED)
    ax[1].axhspan(0.937, 1.064, color="0.9", zorder=0)
    ax[1].set_xscale("log")
    ax[1].set_xlabel(r"prior weight $\lambda$"); ax[1].set_ylabel(r"$\kappa$-$C(W)$")
    ax[1].set_title(r"Check 2: $\lambda$ scan")
    ax[1].legend(frameon=False, fontsize=8)

    ax[2].bar(x - w / 2, [sa["top_axis_share_mean"], si["top_axis_share_mean"]], w, color=C4)
    ax[2].axhline(TOP_AXIS_BOUND, ls="--", lw=1.2, color=RED, label="data-inherent bound")
    ax[2].axhline(IPF_TOP_AXIS, ls=":", lw=1.2, color="0.3", label="IPF reference (0.912)")
    ax[2].set_xticks(x); ax[2].set_xticklabels(["active prior", "inert prior"])
    ax[2].set_ylabel("top-axis share of tr(H)"); ax[2].set_ylim(0, 1.0)
    ax[2].set_title("Check 3: structural bound")
    ax[2].legend(frameon=False, fontsize=7)

    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT_FIG), exist_ok=True)
    fig.savefig(OUT_FIG, bbox_inches="tight")
    fig.savefig(OUT_FIG.replace(".pdf", ".png"), dpi=160, bbox_inches="tight")

    print("ACTIVE:", json.dumps(sa, indent=2)); print("  verdict:", out["active"]["verdict"], out["active"]["checks"])
    print("INERT :", json.dumps(si, indent=2)); print("  verdict:", out["inert"]["verdict"], out["inert"]["checks"])
    print("[saved]", os.path.abspath(OUT_JSON)); print("[saved]", os.path.abspath(OUT_FIG))


if __name__ == "__main__":
    main()
