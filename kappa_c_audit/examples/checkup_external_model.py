"""Third-party model check-up demo for kappa-c-audit.

WHY THIS FILE EXISTS
--------------------
The package README and ``quickstart.py`` show the audit on *synthetic* ground
truth, where the answer is known.  That is the right way to demonstrate the
metric, but it does not answer the question a foundation-model user actually
asks:  "I have a pretrained single-cell model and a prior-injected / fine-tuned
variant of it.  Can I tell, in ten lines, whether the prior actually changed
anything the decoder does?"

This script is that ten-line answer.  It audits ANY pair of
(vanilla, prior-injected) checkpoints through one uniform interface, with NO
assumptions about architecture, training objective, or prior type.

WHAT A "CHECKPOINT" IS HERE
---------------------------
A model is a directory containing two ``.npy`` arrays over the SAME cells:

    Z.npy   latent code      (n_cells, n_latent)
    X.npy   expression       (n_cells, n_genes)

Extract them however the model exposes them (``model.get_latent_representation()``
for scVI, ``model.encode()`` for a VAE, any projection for a foundation model)
and point this script at the two directories.  The audit itself is
architecture-agnostic: it compares the *fitted* latent->expression readout of
the two codes on the same cells.

USAGE
-----
    python -u examples/checkup_external_model.py --vanilla RUN_DIR \\\\
                                               --prior   RUN_DIR2

    # or run the built-in self-test, which fabricates two checkpoints that
    # reproduce the paper's two canonical outcomes (a null decoder audit plus a
    # real latent reorganisation, and a genuine decoder compression):
    python -u examples/checkup_external_model.py --self-test

WHAT IT PRINTS
--------------
A six-line report: the decoder readout ratio kappa-C(W), its compression C%,
the verdict, the latent companion kappa-C(Z), the effective-rank change, and a
one-line interpretation.  Exit code is 0 for any successfully computed report;
the *judgement* is in the verdict, not the exit status, so the script is safe to
wire into a CI gate.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

# allow running from a source checkout without installing
try:
    from kappa_c_audit import audit, classify
except ImportError:  # pragma: no cover
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    from kappa_c_audit import audit, classify


# --------------------------------------------------------------------------
# checkpoint I/O
# --------------------------------------------------------------------------
def load_checkpoint(d):
    """Load (Z, X) from a directory holding Z.npy and X.npy."""
    zp, xp = os.path.join(d, "Z.npy"), os.path.join(d, "X.npy")
    if not (os.path.isfile(zp) and os.path.isfile(xp)):
        raise FileNotFoundError(
            "checkpoint dir %r must contain Z.npy and X.npy" % d)
    return np.load(zp), np.load(xp)


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------
_INTERP = {
    "vanilla-equivalent":
        "decoder unchanged within seed noise -- the prior did NOT alter what the "
        "latent can predict; any benefit must come from the latent side.",
    "compressed":
        "decoder readout compressed beyond the vanilla band -- the latent now "
        "predicts expression less well; check whether this is a real constraint "
        "or a collapsed/inert prior (run the three checks).",
    "amplified":
        "decoder readout amplified beyond the vanilla band -- unusual; verify "
        "the two checkpoints share the same cells and preprocessing.",
    "locked":
        "extreme compression (C > 50%) -- the decoder has been effectively "
        "locked; the prior is dominating, not gently shaping, the geometry.",
    "sub-vanilla":
        "readout below the band but above the locked threshold -- partial "
        "compression; report the direction, not a binary verdict.",
    "baseline-only":
        "no prior checkpoint supplied; baseline statistics reported only.",
}


def report(title, res):
    band = res.get("band", [0.937, 1.064])
    print("=" * 68)
    print(title)
    print("-" * 68)
    print("  cells / latent / genes : %d / %d / %d"
          % (res["n_cells"], res["n_latent"], res["n_genes"]))
    print("  kappa-C(W) [decoder]   : %.3f    (band %.2f-%.2f)"
          % (res["kappa_C_W"], band[0], band[1]))
    print("  compression C          : %+.1f%%" % res.get("C_percent", 0.0))
    print("  verdict                : %s" % res["verdict"])
    if "kappa_C_Z" in res:
        print("  kappa-C(Z) [latent]    : %.3f" % res["kappa_C_Z"])
        print("  effective-rank change  : %+.1f%%" % res["eff_rank_change_percent"])
        if "predictability_ratio" in res:
            print("  predictability ratio   : %.4f  (= kappa-C(W) by identity)"
                  % res["predictability_ratio"])
    print("  interpretation         : %s" % _INTERP.get(res["verdict"], ""))
    print()


# --------------------------------------------------------------------------
# built-in self-test: fabricate the paper's two canonical outcomes
# --------------------------------------------------------------------------
def self_test():
    rng = np.random.default_rng(7)
    G, K, N = 800, 24, 4000
    W = rng.standard_normal((G, K))
    W /= np.linalg.norm(W, axis=0, keepdims=True)
    Z0 = rng.standard_normal((N, K))
    X0 = Z0 @ W.T

    # (a) decoder untouched, latent reorganised -> the paper's headline null
    ev = np.ones(K); ev[K // 4:] = 0.08
    Za = rng.standard_normal((N, K)) @ np.diag(np.sqrt(ev))
    Xa = Za @ W.T
    report("[self-test a] prior-injected: latent reorganised, decoder untouched",
           audit(Z0, X0, Za, Xa))

    # (b) decoder genuinely compressed -> the positive control must fire
    sig = rng.standard_normal((N, K)) @ W.T
    v = float((sig ** 2).mean())
    Xb = sig + np.sqrt(v * (1 / 0.4 - 1)) * rng.standard_normal(sig.shape)
    report("[self-test b] prior-injected: decoder genuinely compressed (rel R2=0.4)",
           audit(Z0, X0, rng.standard_normal((N, K)), Xb))

    print("Self-test complete: (a) must read vanilla-equivalent, (b) must fire.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vanilla", help="dir with Z.npy, X.npy for the vanilla model")
    ap.add_argument("--prior", help="dir with Z.npy, X.npy for the prior-injected model")
    ap.add_argument("--name", default="model", help="label for the report")
    ap.add_argument("--self-test", action="store_true",
                    help="run the built-in two-outcome demonstration")
    a = ap.parse_args()

    if a.self_test or not a.vanilla:
        self_test()
        return 0

    Z0, X0 = load_checkpoint(a.vanilla)
    if a.prior:
        Z1, X1 = load_checkpoint(a.prior)
        report("[%s] vanilla vs prior-injected" % a.name, audit(Z0, X0, Z1, X1))
    else:
        report("[%s] baseline only" % a.name, audit(Z0, X0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
