# -*- coding: utf-8 -*-
"""S1j -- re-run Holm test 1 (product != unity) in the corrected statistic.

The manuscript's pre-specified trio is:
  (1) the product kappa-C(W) * kappa-C(Z) differs from unity
  (2) the latent effective rank contracts
  (3) the latent contracts MORE than the decoder (paired)
Tests (2) and (3) are built on participation-ratio quantities, which are
basis-invariant, so they are unaffected by the readout-form correction.  Test (1)
uses kappa-C(W) and therefore must be recomputed.

Method matches the original: percentile bootstrap over the 33 configurations,
B = 5000, seed 20260912, two-sided p from the bootstrap distribution of the
product (doubled one-tail mass), Holm-corrected inside the same 3-test family.

Reads  : outputs/positive_result/s1g_corrected_summaries.json
Writes : outputs/positive_result/s1j_holm_test1.json
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT = os.path.join(ROOT, "outputs", "positive_result")

B = 5000
SEED = 20260912


def boot(vals, rng):
    v = np.asarray(vals, float)
    v = v[np.isfinite(v)]
    idx = rng.integers(0, v.size, size=(B, v.size))
    return np.array([np.prod(v[i] ** (1.0 / v.size)) for i in idx])  # geometric mean


def two_sided_p(logp):
    """Two-sided bootstrap p for H0: mean log-product = 0."""
    return float(2.0 * min((logp <= 0).mean(), (logp >= 0).mean()))


def holm(raw):
    """Holm step-down over {name: raw p}."""
    items = sorted(raw.items(), key=lambda kv: kv[1])
    m = len(items)
    out, prev = {}, 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, max(prev, (m - i) * p))
        out[k] = adj
        prev = adj
    return out


def main():
    g = json.load(open(os.path.join(OUT, "s1g_corrected_summaries.json"),
                       encoding="utf-8-sig"))
    rows = g["rows"]
    prod_tr = [r["product_tr"] for r in rows if r.get("product_tr")]
    prod_r2 = [r["product_r2"] for r in rows if r.get("product_r2")]

    print(f"n = {len(prod_r2)} configurations\n")
    res = {}
    for tag, vals in (("trace (as released)", prod_tr), ("corrected", prod_r2)):
        rng = np.random.default_rng(SEED)
        logp = np.log(boot(vals, rng))
        prod = np.exp(logp)
        lo, hi = np.percentile(prod, [2.5, 97.5])
        p = two_sided_p(logp)
        res[tag] = dict(median_product=float(np.median(vals)),
                        median_log_product=float(np.median(np.log(vals))),
                        ci_product=[float(lo), float(hi)], p_raw=p)
        print(f"[{tag}]")
        print(f"   median product      {np.median(vals):.4f}")
        print(f"   median log-product  {np.median(np.log(vals)):+.4f}")
        print(f"   bootstrap 95% CI    [{lo:.3f}, {hi:.3f}]")
        print(f"   p (two-sided)       {p:.3e}")

    # Holm inside the pre-specified trio; tests 2-3 keep their published raw p
    # (latent pr-ratio p_raw = 2e-6, paired p_raw = 1e-4 -- both basis-invariant)
    raw = {"product_differs_from_unity (corrected)": res["corrected"]["p_raw"],
           "latent_effrank_contracts": 2e-6,
           "latent_contracts_more_paired": 1e-4}
    adj = holm(raw)
    print("\nHolm-corrected (same 3-test family):")
    for k, v in raw.items():
        print(f"   {k:<42} raw {v:.2e}   Holm {adj[k]:.2e}")
    res["holm_raw"] = raw
    res["holm_adjusted"] = adj

    with open(os.path.join(OUT, "s1j_holm_test1.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    print(f"\nwrote {os.path.join(OUT, 's1j_holm_test1.json')}")


if __name__ == "__main__":
    main()
