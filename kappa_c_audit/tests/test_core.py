"""Minimal correctness tests for kappa-c-audit (run: pytest -q)."""
import numpy as np

from kappa_c_audit import (audit, classify, kappa_C_W, prior_strength_scan,
                           predictability_ratio, ridge_readout, structural_bound)

RNG = np.random.default_rng(1)
G, K, N = 400, 15, 2000


def _W():
    W = RNG.standard_normal((G, K))
    return W / np.linalg.norm(W, axis=0, keepdims=True)


def _Z(n=N, cov=None):
    if cov is None:
        return RNG.standard_normal((n, K))
    return RNG.standard_normal((n, K)) @ np.linalg.cholesky(cov).T


def _ACOV():
    """Strongly anisotropic latent covariance -- the regime trained VAEs are in.

    Under an isotropic latent the two forms of kappa_C_W coincide and the defect
    is invisible; anisotropy is precisely what separates them.
    """
    A = RNG.standard_normal((K, K))
    C = A @ A.T / K + np.diag(np.geomspace(0.01, 3.0, K))
    return C


def test_identity_kappa_equals_predictability_ratio():
    W = _W()
    Z0 = _Z(); X0 = Z0 @ W.T
    Z1 = _Z(); sig = Z1 @ W.T
    v = float((sig ** 2).mean())
    X1 = sig + np.sqrt(v * (1 / 0.3 - 1)) * RNG.standard_normal(sig.shape)
    k = kappa_C_W(Z0, X0, Z1, X1)[0]
    assert abs(k - predictability_ratio(Z0, X0, Z1, X1)) < 0.02


def test_predictability_ratio_matches_kappa_exactly():
    """Both now route through predictability_total, so agreement is exact."""
    W = _W()
    Z0 = _Z(); X0 = Z0 @ W.T
    Z1 = _Z(); X1 = Z1 @ W.T + 0.5 * RNG.standard_normal((N, G))
    k = kappa_C_W(Z0, X0, Z1, X1)[0]
    assert abs(k - predictability_ratio(Z0, X0, Z1, X1)) < 1e-12


def test_pred_form_invariant_trace_form_is_not():
    """Rotating one model's latent basis leaves every R2_g unchanged.

    The predictability ratio must therefore be invariant, while the trace ratio
    moves -- this asymmetry is the defect the default fixes.
    """
    W = _W()
    C = _ACOV()
    Z0 = _Z(cov=C); X0 = Z0 @ W.T
    Z1 = _Z(cov=C); X1 = Z1 @ W.T + 0.4 * RNG.standard_normal((N, G))
    Q, _ = np.linalg.qr(RNG.standard_normal((K, K)))

    kp = kappa_C_W(Z0, X0, Z1, X1)[0]
    kp_q = kappa_C_W(Z0, X0, Z1 @ Q, X1)[0]
    kt = kappa_C_W(Z0, X0, Z1, X1, form="trace")[0]
    kt_q = kappa_C_W(Z0, X0, Z1 @ Q, X1, form="trace")[0]

    move_pred = abs(kp - kp_q)
    move_trace = abs(kt - kt_q)
    assert move_pred < 1e-6                          # invariant to solver tolerance
    assert move_trace > 1e3 * max(move_pred, 1e-12)  # >1000x more basis-dependent
    assert move_trace / max(abs(kt), 1e-12) > 1e-2   # >1% relative movement


def test_predictability_total_matches_corr_identity():
    """sum_g R2_g = tr(Corr(Z) W W^T) + (2 alpha / N) tr(W W^T)."""
    from kappa_c_audit import predictability_total
    alpha = 1.0
    W0 = _W()
    Z = _Z(cov=np.diag(np.linspace(0.2, 3.0, K)))
    X = Z @ W0.T + 0.3 * RNG.standard_normal((N, G))
    Zr = ridge_readout(Z, X, alpha=alpha).astype(float)
    Zs = (Z - Z.mean(0)) / Z.std(0)
    C = Zs.T @ Zs / N
    lhs = predictability_total(Z, X, alpha=alpha)
    rhs = float(np.trace(C @ (Zr @ Zr.T))
                + (2 * alpha / N) * float(np.trace(Zr @ Zr.T)))
    assert abs(lhs - rhs) < 1e-6 * max(1.0, abs(lhs))


def test_audit_reports_both_forms_and_agrees_with_core():
    W = _W()
    Z0 = _Z(); X0 = Z0 @ W.T
    Z1 = _Z(); X1 = Z1 @ W.T + 0.4 * RNG.standard_normal((N, G))
    res = audit(Z0, X0, Z1, X1)
    assert "kappa_C_W_trace" in res
    assert abs(res["kappa_C_W"] - kappa_C_W(Z0, X0, Z1, X1)[0]) < 1e-12
    assert abs(res["kappa_C_W_trace"]
               - kappa_C_W(Z0, X0, Z1, X1, form="trace")[0]) < 1e-12
    assert res["kappa_C_W"] != res["kappa_C_W_trace"]


def test_trace_form_rejects_unknown_form():
    W = _W()
    Z0 = _Z(); X0 = Z0 @ W.T
    try:
        kappa_C_W(Z0, X0, Z0, X0, form="nonsense")
    except ValueError:
        return
    raise AssertionError("expected ValueError for an unknown form")


def test_specificity_latent_only_stays_in_band():
    W = _W()
    Z0 = _Z(); X0 = Z0 @ W.T
    ev = np.ones(K); ev[-10:] = 0.05
    Z1 = _Z(cov=np.diag(ev)); X1 = Z1 @ W.T
    res = audit(Z0, X0, Z1, X1)
    assert 0.937 <= res["kappa_C_W"] <= 1.064
    assert res["verdict"] == "vanilla-equivalent"
    assert res["kappa_C_Z"] < 0.8          # the latent audit does fire


def test_sensitivity_predictability_drop_fires():
    W = _W()
    Z0 = _Z(); X0 = Z0 @ W.T
    Z1 = _Z(); sig = Z1 @ W.T
    v = float((sig ** 2).mean())
    X1 = sig + np.sqrt(v * (1 / 0.3 - 1)) * RNG.standard_normal(sig.shape)
    k = audit(Z0, X0, Z1, X1)["kappa_C_W"]
    assert k < 0.5 and classify(k) == "locked"


def test_invariance_under_direction_attenuation():
    W = _W()
    Z0 = _Z(); X0 = Z0 @ W.T
    m = np.ones(K); m[-10:] = 0.05
    Z1 = _Z(); X1 = Z1 @ (W * m[None, :]).T
    assert abs(audit(Z0, X0, Z1, X1)["kappa_C_W"] - 1.0) < 0.05


def test_readout_shape():
    Z = _Z()
    X = Z @ _W().T
    W = ridge_readout(Z, X)
    assert W.shape == (K, G)          # (n_latent, n_genes)


def test_prior_strength_scan_flat_vs_responding():
    assert not prior_strength_scan({0.1: 1.0, 1.0: 1.01, 10.0: 1.0})["responds"]
    assert prior_strength_scan({0.1: 1.0, 1.0: 0.6, 10.0: 0.25})["responds"]


def test_structural_bound_decreases_with_compression():
    b90 = structural_bound(90, 1000, 100)
    b50 = structural_bound(50, 1000, 100)
    assert b90 < b50
