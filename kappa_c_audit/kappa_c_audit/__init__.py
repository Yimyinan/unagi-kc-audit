"""kappa-c-audit: a dual-spectrum audit for GRN-prior injection
in single-cell VAEs, plus the three-check prior-activity protocol.

Quick start
-----------
>>> from kappa_c_audit import audit
>>> audit(Z_vanilla, X_vanilla, Z_prior, X_prior)["verdict"]

The decoder audit ``kappa_C_W`` is an aggregate latent->expression predictability
ratio, ``sum_g R2_g(prior) / sum_g R2_g(vanilla)``: it fires when a model's latent
code predicts its expression less well, and is invariant to per-gene rescaling and
to attenuation of latent->gene directions (see ``README.md`` and
``examples/quickstart.py``).

The legacy estimator ``tr(W^T W)_prior / tr(W^T W)_vanilla`` is still available as
``form="trace"`` and is reported by :func:`audit` as ``kappa_C_W_trace``.  It equals
the predictability form only when the latent code is whitened, which trained latent
codes are not, so it is not invariant to the latent basis; prefer the default.
"""

from .audit import audit, audit_from_readout
from .checks import (gradient_check, injection_path_check, prior_strength_scan,
                     structural_bound)
from .core import (DEFAULT_BAND, LOCKED_THRESHOLD, classify, eff_rank, kappa_C_W,
                   kappa_C_Z, participation_ratio, predictability_ratio,
                   predictability_total, ridge_readout)

__version__ = "0.1.3"

__all__ = [
    "audit", "audit_from_readout",
    "gradient_check", "prior_strength_scan", "structural_bound",
    "injection_path_check",
    "ridge_readout", "kappa_C_W", "kappa_C_Z", "eff_rank",
    "participation_ratio", "predictability_ratio", "predictability_total",
    "classify", "DEFAULT_BAND", "LOCKED_THRESHOLD", "__version__",
]
