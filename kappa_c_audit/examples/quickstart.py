"""Self-contained demonstration of kappa-c-audit on synthetic ground truth.

Run:  python examples/quickstart.py

It shows, with the audit API only:
  1. the audited quantity is an aggregate predictability ratio, and
     predictability_ratio() reproduces it exactly (both unchanged since 0.1.1);
  1b. the legacy trace form IS basis-dependent while the predictability form is
     not -- the defect release 0.1.1 fixed;
  2. sensitivity   -- the decoder audit fires when predictability drops;
  3. specificity   -- a latent-only reorganisation leaves the decoder audit null;
  4. invariance    -- attenuating latent->gene directions leaves it at 1;
  5. the three-check helpers.
"""
import numpy as np

from kappa_c_audit import (audit, classify, gradient_check, kappa_C_W,
                           predictability_ratio, prior_strength_scan,
                           structural_bound)

RNG = np.random.default_rng(0)
G, K, N = 600, 20, 3000


def make_decoder():
    W = RNG.standard_normal((G, K))
    return W / np.linalg.norm(W, axis=0, keepdims=True)


def latent(n=N, cov=None):
    if cov is None:
        return RNG.standard_normal((n, K))
    return RNG.standard_normal((n, K)) @ np.linalg.cholesky(cov).T


def main():
    W = make_decoder()
    Z0, X0 = latent(), None
    X0 = Z0 @ W.T

    # 1. identity -----------------------------------------------------------
    Zp = latent(); sig = Zp @ W.T
    v = float((sig ** 2).mean())
    Xp = sig + np.sqrt(v * (1 / 0.5 - 1)) * RNG.standard_normal(sig.shape)  # rel R2 = 0.5
    kCW = kappa_C_W(Z0, X0, Zp, Xp)[0]
    pr = predictability_ratio(Z0, X0, Zp, Xp)
    print("[1] identity      kappa-C(W)=%.4f  predictability_ratio=%.4f" % (kCW, pr))

    # 1b. basis dependence: the two forms part company on an anisotropic latent
    A = RNG.standard_normal((K, K))
    Cv = A @ A.T / K + np.diag(np.geomspace(0.01, 3.0, K))
    Zc = latent(cov=Cv)
    Xc = Zc @ W.T + 0.4 * RNG.standard_normal((N, G))
    Q, _ = np.linalg.qr(RNG.standard_normal((K, K)))
    kt = kappa_C_W(Z0, X0, Zc, Xc, form="trace")[0]
    kt_q = kappa_C_W(Z0, X0, Zc @ Q, Xc, form="trace")[0]
    kp = kappa_C_W(Z0, X0, Zc, Xc)[0]
    kp_q = kappa_C_W(Z0, X0, Zc @ Q, Xc)[0]
    print("[1b] basis        rotating one model's latent by Q:")
    print("                  trace form      %.4f -> %.4f   (moves %.1e)"
          % (kt, kt_q, abs(kt - kt_q)))
    print("                  predictability  %.4f -> %.4f   (moves %.1e)"
          % (kp, kp_q, abs(kp - kp_q)))

    # 2. sensitivity --------------------------------------------------------
    for r2 in (1.0, 0.75, 0.5, 0.3):
        Xp = sig + np.sqrt(v * (1 / max(r2, 1e-6) - 1)) * RNG.standard_normal(sig.shape)
        k = audit(Z0, X0, Zp, Xp)["kappa_C_W"]
        print("[2] sensitivity   target R2=%.2f -> kappa-C(W)=%.3f (%s)"
              % (r2, k, classify(k)))

    # 3. specificity: latent covariance reorganised, decoder fixed ----------
    ev = np.ones(K); ev[-int(0.75 * K):] = 0.05
    Zl = latent(cov=np.diag(ev)); Xl = Zl @ W.T
    res = audit(Z0, X0, Zl, Xl)
    print("[3] specificity   kappa-C(W)=%.3f (%s)  kappa-C(Z)=%.3f  eff-rank %+.1f%%"
          % (res["kappa_C_W"], res["verdict"], res["kappa_C_Z"],
             res["eff_rank_change_percent"]))

    # 4. invariance: attenuate latent->gene directions ----------------------
    m = np.ones(K); m[-int(0.75 * K):] = 0.05
    Za = latent(); Xa = Za @ (W * m[None, :]).T
    print("[4] invariance    kappa-C(W)=%.3f" % audit(Z0, X0, Za, Xa)["kappa_C_W"])

    # 5. three-check helpers ------------------------------------------------
    print("[5] structural_bound(C=90, trH=1000, gap=100) = %.3f"
          % structural_bound(90, 1000, 100))
    print("    prior_strength_scan(flat)     =", prior_strength_scan({0.1: 1.0, 1.0: 1.01, 10.0: 1.0}))
    print("    gradient_check(...)           =",
          {k: round(v, 3) if isinstance(v, float) else v
           for k, v in gradient_check([0.1, 0.5, 0.8, 0.95], [1, 2, 3, 4]).items()})


if __name__ == "__main__":
    main()
