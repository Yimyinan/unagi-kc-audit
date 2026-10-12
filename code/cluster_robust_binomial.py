# -*- coding: utf-8 -*-
"""
cluster_robust_binomial.py
===========================

Cluster-robust one-sample binomial proportion for correlated Bernoulli
observations grouped into clusters (e.g. experimental units that share a
training seed, so their directional statistics are not statistically
independent).

WHY THIS EXISTS
---------------
A naive exact binomial / sign test treats every unit as independent. When units
share seeds and training data, that *overstates* precision. The reviewer
response therefore replaces the seed-clustered permutation (which with only
three seeds has 2**3 = 8 flip-configurations and maximum one-sided power 1/8)
with a cluster-robust binomial that (i) keeps the ordinary sample proportion as
the point estimate, (ii) widens its standard error to account for within-cluster
correlation via the Liang-Zeger (sandwich) variance, and (iii) reports an honest
confidence interval on the compressed proportion together with a one-sided test
of H0: p <= p0 (default p0 = 0.5, "no directional preference").

Because only a few seed-clusters are available, we report TWO intervals:
  * analytic  : Liang-Zeger robust SE + normal-approximation CI/p (closed form);
  * bootstrap : cluster (block) bootstrap percentile CI (B resamples of whole
                clusters), the honest small-K interval that does not rely on a
                normal or t approximation when K is tiny.

METHOD (intercept-only GEE, independence working correlation, logit link,
delta-method back to the proportion scale):
    p_hat   = sum_i y_i / N
    U_c     = sum_{i in c} (y_i - p_hat)            # cluster score sum
    V_robust(p_hat) = sum_c U_c**2 / N**2
    SE_robust = sqrt(V_robust)
The cluster (block) bootstrap resamples whole clusters with replacement and
recomputes the proportion on each resample; the percentile interval is the
reported CI.

Dependency-free (standard library only) -> trivially released as source.
"""

import math
import random
from collections import defaultdict
from typing import Sequence, Tuple, Dict, Any


# --------------------------------------------------------------------------- #
# normal-distribution helpers (pure stdlib)
# --------------------------------------------------------------------------- #
def _norm_cdf(z: float) -> float:
    """Standard normal CDF via erf."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _norm_ppf(p: float, lo: float = -8.0, hi: float = 8.0, tol: float = 1e-12) -> float:
    """Inverse standard normal CDF (quantile) by bisection on _norm_cdf."""
    if p <= 0.0:
        return lo
    if p >= 1.0:
        return hi
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if _norm_cdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# --------------------------------------------------------------------------- #
# core statistics
# --------------------------------------------------------------------------- #
def cluster_robust_proportion(
    y: Sequence[float],
    cluster: Sequence,
    p0: float = 0.5,
    conf: float = 0.95,
    alternative: str = "greater",
    seed: int = 20260913,
    n_boot: int = 10000,
) -> Dict[str, Any]:
    """Cluster-robust one-sample binomial proportion.

    Parameters
    ----------
    y          : binary indicators (1 = "compressed" / event of interest, else 0).
    cluster    : cluster label per observation (e.g. training seed).
    p0         : null proportion for the one-sided test (default 0.5).
    conf       : confidence level for the interval (default 0.95).
    alternative: "greater" (H1: p > p0), "less", or "two-sided".
    seed       : RNG seed for the cluster bootstrap.
    n_boot     : number of cluster-bootstrap resamples.

    Returns a dict with the point estimate, analytic robust SE + normal CI/p,
    the cluster-bootstrap percentile CI, and per-cluster sizes.
    """
    ys = [float(v) for v in y]
    N = len(ys)
    if N == 0:
        raise ValueError("no observations")
    k = sum(1 for v in ys if v > 0.5)
    p_hat = k / N

    # group observations by cluster
    grouped: Dict[Any, list] = defaultdict(list)
    for v, c in zip(ys, cluster):
        grouped[c].append(v)
    sizes = {c: len(vs) for c, vs in grouped.items()}
    K = len(grouped)

    # ---- Liang-Zeger robust variance of the sample proportion -------------- #
    U = [sum(vs) - p_hat * len(vs) for vs in grouped.values()]
    V = sum(u * u for u in U) / (N * N)
    se = math.sqrt(V) if V > 0 else 0.0

    # normal-approximation analytic CI on the proportion
    alpha = 1.0 - conf
    z = _norm_ppf(1.0 - alpha / 2.0)
    half = z * se
    ci_low_a = max(0.0, p_hat - half)
    ci_high_a = min(1.0, p_hat + half)

    # one-sided (and two-sided) test H0: p = p0, using the null robust SE
    U0 = [sum(vs) - p0 * len(vs) for vs in grouped.values()]
    V0 = sum(u * u for u in U0) / (N * N)
    se0 = math.sqrt(V0) if V0 > 0 else 0.0
    if se0 == 0:
        t_stat = float("inf") if p_hat > p0 else (-float("inf") if p_hat < p0 else 0.0)
        p_one = 0.0 if p_hat > p0 else (1.0 if p_hat < p0 else 0.5)
    else:
        t_stat = (p_hat - p0) / se0
        p_one = 1.0 - _norm_cdf(t_stat)
    if alternative == "two-sided":
        p_value = 2.0 * min(p_one, 1.0 - p_one)
    elif alternative == "less":
        p_value = _norm_cdf(t_stat)
    else:  # greater
        p_value = p_one

    # ---- cluster (block) bootstrap percentile CI --------------------------- #
    rng = random.Random(seed)
    cl_labels = list(grouped.keys())
    boots = []
    for _ in range(n_boot):
        # resample whole clusters with replacement (K clusters -> K draws)
        succ = 0
        tot = 0
        for _ in range(len(cl_labels)):
            c = rng.choice(cl_labels)
            for v in grouped[c]:
                succ += 1 if v > 0.5 else 0
                tot += 1
        if tot > 0:
            boots.append(succ / tot)
    boots.sort()
    if boots:
        lo_i = max(0, int(math.floor(alpha / 2.0 * len(boots))))
        hi_i = min(len(boots) - 1, int(math.ceil((1.0 - alpha / 2.0) * len(boots))) - 1)
        ci_low_b = boots[lo_i]
        ci_high_b = boots[hi_i]
        # bootstrap one-sided p (H1: p > p0): fraction of resamples <= p0
        p_boot = sum(1 for b in boots if b <= p0) / len(boots)
        if alternative == "two-sided":
            p_boot = 2.0 * min(p_boot, 1.0 - p_boot)
        elif alternative == "less":
            p_boot = 1.0 - p_boot
    else:
        ci_low_b = ci_high_b = p_hat
        p_boot = float("nan")

    return {
        "n": N,
        "k": k,
        "n_clusters": K,
        "p_hat": p_hat,
        "se_robust": se,
        "analytic_ci": [ci_low_a, ci_high_a],
        "bootstrap_ci": [ci_low_b, ci_high_b],
        "z_stat": t_stat,
        "p_value_normal": p_value,
        "p_value_bootstrap": p_boot,
        "null_p0": p0,
        "conf": conf,
        "clusters": {str(c): sizes[c] for c in sizes},
        "n_boot": n_boot,
    }


# --------------------------------------------------------------------------- #
# self-test (no external deps)
# --------------------------------------------------------------------------- #
def _selftest() -> None:
    # Case 1: 1 unit / cluster removes dependence -> robust SE equals binomial SE.
    y = [1, 1, 1, 0, 0]
    c = [1, 2, 3, 4, 5]
    r = cluster_robust_proportion(y, c, alternative="two-sided")
    binom_se = math.sqrt(0.6 * 0.4 / 5)
    assert abs(r["se_robust"] - binom_se) < 1e-9, (r["se_robust"], binom_se)
    print(f"[selftest 1] 1 unit/cluster: robust SE={r['se_robust']:.4f} == binomial SE={binom_se:.4f}  OK")

    # Case 2: all-concordant, 2 units/cluster -> robust SE > naive binomial SE
    # (dependence inflates the variance).
    y2 = [1, 1, 1, 1, 0, 0]
    c2 = [1, 1, 2, 2, 3, 3]
    r2 = cluster_robust_proportion(y2, c2, alternative="two-sided")
    binom_se2 = math.sqrt((4 / 6) * (2 / 6) / 6)
    assert r2["se_robust"] > binom_se2 - 1e-9, (r2["se_robust"], binom_se2)
    print(f"[selftest 2] 2 units/cluster concordant: robust SE={r2['se_robust']:.4f} > naive SE={binom_se2:.4f}  OK")

    # Case 3: p_hat and k/n consistent.
    assert abs(r["p_hat"] - 0.6) < 1e-9 and r["k"] == 3 and r["n"] == 5
    print(f"[selftest 3] p_hat={r['p_hat']:.2f}, k/n={r['k']}/{r['n']}  OK")

    # Case 4: bootstrap CI contains p_hat and is ordered.
    assert r["bootstrap_ci"][0] <= r["p_hat"] <= r["bootstrap_ci"][1]
    assert r["bootstrap_ci"][0] <= r["bootstrap_ci"][1]
    print(f"[selftest 4] bootstrap CI=[{r['bootstrap_ci'][0]:.3f},{r['bootstrap_ci'][1]:.3f}] contains p_hat  OK")

    print("ALL SELFTESTS PASSED")


if __name__ == "__main__":
    _selftest()
