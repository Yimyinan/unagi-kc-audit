"""High-level audit API.

Usage
-----
>>> from kappa_c_audit import audit
>>> res = audit(Z_vanilla, X_vanilla, Z_prior, X_prior)
>>> res["verdict"], round(res["kappa_C_W"], 3)
('vanilla-equivalent', 0.99)

``Z`` is the latent code (cells x latent dims) and ``X`` the expression matrix
(cells x genes) of the same cells. Pass the prior-injected model's ``(Z, X)`` as
the second pair; pass neither for a baseline-only report.
"""

from __future__ import annotations

import numpy as np

from .core import (DEFAULT_BAND, LOCKED_THRESHOLD, classify, eff_rank, kappa_C_W,
                   kappa_C_Z, predictability_ratio, ridge_readout)

__all__ = ["audit", "audit_from_readout"]


def audit(Z_vanilla, X_vanilla, Z_prior=None, X_prior=None,
          band=DEFAULT_BAND, locked: float = LOCKED_THRESHOLD, alpha: float = 1.0):
    """Dual-spectrum audit of a prior-injected model against a vanilla baseline.

    Returns a dict with the decoder readout ratio ``kappa_C_W`` (the aggregate
    predictability ratio ``sum_g R2_g(prior) / sum_g R2_g(vanilla)``), the
    compression ``C_percent``, the three-way ``verdict``, the latent ratio
    ``kappa_C_Z`` and effective ranks, and ``predictability_ratio`` (identical to
    ``kappa_C_W`` -- both use :func:`predictability_total`).

    ``kappa_C_W_trace`` and ``readout_trace_*`` carry the legacy
    ``tr(W^T W)`` estimator for reproduction of earlier releases.  The two forms
    coincide only for a whitened latent code; see :func:`kappa_C_W` for the
    relation and its consequences.
    """
    Z_vanilla, X_vanilla = np.asarray(Z_vanilla), np.asarray(X_vanilla)
    out = dict(n_cells=int(Z_vanilla.shape[0]), n_latent=int(Z_vanilla.shape[1]),
               n_genes=int(X_vanilla.shape[1]) if X_vanilla.ndim > 1 else 1,
               band=list(band), locked_threshold=locked,
               estimator="sum_g R2_g ratio (predictability form)")

    if Z_prior is None or X_prior is None:
        W0 = ridge_readout(Z_vanilla, X_vanilla, alpha=alpha)
        out.update(kappa_C_W=1.0, kappa_C_W_trace=1.0, C_percent=0.0,
                   verdict="baseline-only",
                   eff_rank_vanilla=eff_rank(np.cov(Z_vanilla, rowvar=False)),
                   readout_trace_vanilla=float((W0 ** 2).sum()))
        return out

    kCW, W0, W1 = kappa_C_W(Z_vanilla, X_vanilla, Z_prior, X_prior, alpha=alpha)
    kCW_trace = float((W1 ** 2).sum() / (W0 ** 2).sum())
    er0 = eff_rank(np.cov(np.asarray(Z_vanilla), rowvar=False))
    er1 = eff_rank(np.cov(np.asarray(Z_prior), rowvar=False))
    out.update(
        kappa_C_W=kCW,
        kappa_C_W_trace=kCW_trace,
        C_percent=float((1.0 - kCW) * 100.0),
        verdict=classify(kCW, band=band, locked=locked),
        kappa_C_Z=kappa_C_Z(Z_vanilla, Z_prior),
        eff_rank_vanilla=er0,
        eff_rank_prior=er1,
        eff_rank_change_percent=float((er1 / er0 - 1.0) * 100.0) if er0 else float("nan"),
        predictability_ratio=predictability_ratio(Z_vanilla, X_vanilla, Z_prior, X_prior, alpha=alpha),
        readout_trace_vanilla=float((W0 ** 2).sum()),
        readout_trace_prior=float((W1 ** 2).sum()),
    )
    return out


def audit_from_readout(W_vanilla, W_prior, band=DEFAULT_BAND,
                       locked: float = LOCKED_THRESHOLD):
    """Audit from pre-computed readout matrices ``W`` (latent x genes).

    **Trace form only.**  ``sum_g R2_g`` cannot be recovered from ``W`` alone --
    it needs the latent correlation matrix (or the data).  The returned
    ``kappa_C_W`` is therefore the legacy ``tr(W1^T W1) / tr(W0^T W0)`` and is
    named ``kappa_C_W_trace`` as well, so callers cannot mistake it for the
    predictability form.  Prefer :func:`audit` whenever ``(Z, X)`` are available.
    """
    W0 = np.asarray(W_vanilla, float)
    W1 = np.asarray(W_prior, float)
    kCW = float((W1 ** 2).sum() / (W0 ** 2).sum())
    return dict(kappa_C_W=kCW, kappa_C_W_trace=kCW,
                C_percent=float((1.0 - kCW) * 100.0),
                verdict=classify(kCW, band=band, locked=locked),
                estimator="tr(W^T W) ratio (trace form; legacy)",
                readout_trace_vanilla=float((W0 ** 2).sum()),
                readout_trace_prior=float((W1 ** 2).sum()))
