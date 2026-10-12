"""Core kappa-C spectral quantities.

All definitions are IDENTICAL to the audit protocol reported in the manuscript
(``perturbation_benchmark.py`` / ``scvi_demo_benchmark.py``), so third parties
reproduce the published numbers rather than a look-alike.

Key identity (proved in the manuscript, verified numerically in
``examples/quickstart.py``)
---------------------------------------------------------------------------
``ridge_readout`` standardises Z and X per column/gene. Therefore, for an exactly
linear model X = W Z, every row of the standardised readout has unit norm and

    kappa_C_W = sum_g R2_g(pert) / sum_g R2_g(vanilla),

i.e. kappa-C(W) is an aggregate latent->expression *predictability ratio*. It is
invariant to per-gene rescaling of the decoder and to attenuation of latent->gene
directions. This is what makes the decoder audit a predictability test rather
than a decoder-geometry test.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "ridge_readout",
    "kappa_C_W",
    "kappa_C_Z",
    "eff_rank",
    "participation_ratio",
    "predictability_ratio",
    "classify",
    "DEFAULT_BAND",
    "LOCKED_THRESHOLD",
]

DEFAULT_BAND = (0.937, 1.064)  # vanilla-equivalence band (G-VAE hematopoietic null, corrected
                               # predictability form; 1.001 +/- 0.032 over 15 seeds / 210 ordered
                               # seed-pair comparisons). Supersedes the trace-form (0.78, 1.23).
LOCKED_THRESHOLD = 0.5        # deep-compression / "locked" boundary


def _as2d(a) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    return a.reshape(-1, 1) if a.ndim == 1 else a


def ridge_readout(Z, X, alpha: float = 1.0) -> np.ndarray:
    """UNAGI-protocol decoder readout W = Ridge(Z -> X), alpha=1, no intercept.

    Z and X are standardised per column/gene before the fit. Returns W with shape
    ``(n_latent, n_genes)`` -- i.e. one row per latent dimension (note: the
    upstream script's inline comment labels this the other way round; the metric
    is orientation-agnostic, but the array shape is ``(n_latent, n_genes)``).
    """
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    Zs = StandardScaler().fit_transform(_as2d(Z))
    Xs = StandardScaler().fit_transform(_as2d(X))
    W = Ridge(alpha=alpha, fit_intercept=False).fit(Zs, Xs).coef_  # (n_genes, n_latent)
    return W.T                                                     # (n_latent, n_genes)


def predictability_total(Z, X, alpha: float = 1.0) -> float:
    """``sum_g R2_g`` for a single model, computed exactly from Ridge residuals.

    ``Z`` and ``X`` are standardised per column/gene -- the identical
    preprocessing :func:`ridge_readout` applies.  Because each standardised gene
    has ``SS_tot = N``,

        ``sum_g R2_g = G - ||X_s - Z_s W||_F^2 / N``,

    which is exact for any ``alpha`` and does not rely on a correlation-matrix
    identity.  Individual ``R2_g`` may be negative; only the sum is reported.
    """
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    Zs = StandardScaler().fit_transform(_as2d(Z))
    Xs = StandardScaler().fit_transform(_as2d(X))
    fit = Ridge(alpha=alpha, fit_intercept=False).fit(Zs, Xs)
    resid = Xs - fit.predict(Zs)
    return float(Xs.shape[1] - (resid ** 2).sum() / Xs.shape[0])


def kappa_C_W(Z0, X0, Z1, X1, alpha: float = 1.0, form: str = "predictability"):
    """Decoder-readout ratio between a baseline (0) and a perturbed (1) model.

    ``form="predictability"`` (default) returns ``sum_g R2_g(1) / sum_g R2_g(0)``,
    the aggregate latent->expression predictability ratio.  ``form="trace"``
    returns the legacy ``tr(W1^T W1) / tr(W0^T W0)``.

    **The two forms agree only when the latent code is whitened.**  For
    standardised ``Z``, ``X``, ``sum_g R2_g = sum_g W[:,g]' Corr(Z) W[:,g]``
    whereas ``tr(W^T W) = sum_g W[:,g]' W[:,g]``.  Trained latent codes are
    strongly correlated (in the published UNAGI audit the condition number of
    ``Corr(Z)`` had median 3.3e3), so the trace form is not invariant to the
    arbitrary latent basis: rotating ``Z`` leaves every ``R2_g`` unchanged yet
    moves the trace ratio, and the resulting scatter inflation swamped a real
    effect by roughly 40x.  Prefer the default; use ``form="trace"`` only to
    reproduce an earlier release.  Returns ``(kappa, W0, W1)``.
    """
    W0 = ridge_readout(Z0, X0, alpha=alpha)
    W1 = ridge_readout(Z1, X1, alpha=alpha)
    if form == "trace":
        kappa = float((W1 ** 2).sum() / (W0 ** 2).sum())
    elif form == "predictability":
        kappa = float(predictability_total(Z1, X1, alpha=alpha)
                      / predictability_total(Z0, X0, alpha=alpha))
    else:
        raise ValueError("form must be 'predictability' or 'trace'")
    return kappa, W0, W1


def kappa_C_Z(Z0, Z1) -> float:
    """Latent-covariance spectral ratio ``tr(cov(Z1)) / tr(cov(Z0))``."""
    c0 = np.cov(_as2d(Z0), rowvar=False)
    c1 = np.cov(_as2d(Z1), rowvar=False)
    return float(np.trace(c1) / np.trace(c0))


def eff_rank(cov, eps: float = 1e-12) -> float:
    """Effective rank as the participation ratio ``(sum l)^2 / sum l^2``."""
    cov = _as2d(cov)
    e = np.linalg.eigvalsh(cov + eps * np.eye(cov.shape[0]))
    e = np.clip(e, eps, None)
    return float((e.sum() ** 2) / (e ** 2).sum())


def participation_ratio(cov, eps: float = 1e-12) -> float:
    """Alias of :func:`eff_rank` (the single reported effective-rank metric)."""
    return eff_rank(cov, eps=eps)


def predictability_ratio(Z0, X0, Z1, X1, alpha: float = 1.0) -> float:
    """``sum_g R2_g(model 1) / sum_g R2_g(model 0)``.

    Identical to :func:`kappa_C_W` with ``form="predictability"``: both call
    :func:`predictability_total`, so they agree by construction.  (An earlier
    release computed this from ``sklearn``'s per-output ``score``, which returns
    the *average* R^2 rather than the sum, and used ``fit_intercept=True`` on
    unstandardised data -- so it did **not** match :func:`kappa_C_W`.  Both
    issues are fixed here.)
    """
    return float(predictability_total(Z1, X1, alpha=alpha)
                 / predictability_total(Z0, X0, alpha=alpha))


def classify(kappa: float, band=DEFAULT_BAND, locked: float = LOCKED_THRESHOLD) -> str:
    """Map a kappa-C(W) value to an audit verdict.

    ``locked`` (``kappa < 0.5``), ``sub-vanilla`` (``0.5 <= kappa < band[0]``),
    ``vanilla-equivalent`` (inside the band) or ``expanded`` (above the band).
    The ``sub-vanilla`` region is a genuine gap between the locking threshold and
    the vanilla-equivalence band; it is reported rather than folded into a
    neighbouring class.
    """
    if kappa < locked:
        return "locked"
    if kappa < band[0]:
        return "sub-vanilla"
    if kappa <= band[1]:
        return "vanilla-equivalent"
    return "expanded"
