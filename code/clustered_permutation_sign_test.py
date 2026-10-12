"""
clustered_permutation_sign_test.py
===================================

Confirmatory clustered-permutation sign test for directional effects when
experimental units are NOT statistically independent.

WHY THIS EXISTS
---------------
In the GRN-prior audit, many prior/ablation configurations share seeds and
training data, so their directional compression statistics are correlated.
A naive exact sign test over individual configurations therefore *overstates*
precision. The reviewer response prescribed a clustered-permutation test that
permutes the sign of each cluster's *collective* contribution -- i.e. flips the
whole cluster's direction together with probability 1/2. This preserves the
within-cluster correlation structure while breaking the association between
cluster identity and direction, yielding a valid cluster-adjusted p-value.

METHOD
------
Given signed statistics s_i (+1 compressed, -1 expanded, 0/NaN excluded) and a
cluster label c_i:
  * aggregate per cluster:  T_c = sum_{i in c} s_i
  * observed total:         T_obs = sum_c T_c
  * null: for each cluster independently flip its sign with prob 1/2; recompute
          T* = sum_c (+-1)_c * T_c
  * two-sided p = P(|T*| >= |T_obs|)

If the number of clusters is small (<= 12 by default) the null is enumerated
*exactly* over all 2^{#clusters} sign-flip combinations; otherwise a Monte-Carlo
approximation is used.

USAGE
-----
    from clustered_permutation_sign_test import clustered_permutation_p
    p_adj, n_null = clustered_permutation_p(signed_stats, cluster_ids)

Dependency-free (standard library only).
"""

import itertools
import math
from typing import Sequence, Tuple


def _aggregate(signed_stats, cluster_ids):
    clusters = sorted(set(cluster_ids))
    contrib = []
    for c in clusters:
        t = 0.0
        for s, cc in zip(signed_stats, cluster_ids):
            if cc == c and s is not None and not (isinstance(s, float) and math.isnan(s)):
                t += s
        contrib.append(t)
    return clusters, contrib


def clustered_permutation_p(
    signed_stats: Sequence[float],
    cluster_ids: Sequence,
    n_perm: int = 10000,
    exact: bool = None,
    seed: int = 0,
) -> Tuple[float, int]:
    """Return (cluster_adjusted_two_sided_p, n_null_samples)."""
    clusters, contrib = _aggregate(signed_stats, cluster_ids)
    if not contrib:
        raise ValueError("no non-empty clusters")
    T_obs = sum(contrib)

    if exact is None:
        exact = len(clusters) <= 12

    if exact:
        # enumerate every sign-flip combination of the cluster contributions
        T_null = [sum(s * c for s, c in zip(signs, contrib))
                  for signs in itertools.product([-1, 1], repeat=len(clusters))]
    else:
        import random
        rng = random.Random(seed)
        T_null = []
        for _ in range(n_perm):
            s = [rng.choice([-1, 1]) for _ in contrib]
            T_null.append(sum(a * b for a, b in zip(s, contrib)))

    # two-sided: fraction of null |T| at least as extreme as observed.
    # include the observed value itself for tie-consistent (>=) handling.
    count = sum(1 for t in T_null if abs(t) >= abs(T_obs) - 1e-12)
    p = count / len(T_null)
    return p, len(T_null)


def naive_exact_sign_test_p(signed_stats: Sequence[float]) -> float:
    """Naive two-sided exact binomial sign test treating units as independent.

    s_i should be +1 / -1 (excluded entries already removed).
    """
    vals = [s for s in signed_stats if s is not None and not math.isnan(s)]
    n = len(vals)
    k = sum(1 for v in vals if v > 0)
    # two-sided: P(X >= k) + P(X <= n-k) under Bin(n, 0.5)
    def binom_cdf(x, nn, p=0.5):
        return sum(math.comb(nn, i) * (p ** i) * ((1 - p) ** (nn - i))
                   for i in range(0, x + 1))
    if k > n - k:
        k = n - k
    # sum of both tails
    p_val = sum(math.comb(n, i) * (0.5 ** n) for i in range(0, k + 1))
    p_val = 2 * p_val
    return min(1.0, p_val)


def _selftest():
    """Validate against hand-computable cases (no external deps)."""
    # Case 1: 3 clusters, ONE unit each, all concordant compressed -> cluster == naive.
    s = [1, 1, 1]
    c = [1, 2, 3]
    p_adj, n = clustered_permutation_p(s, c, exact=True)
    p_naive = naive_exact_sign_test_p(s)
    # 1 unit/cluster removes the dependence structure, so the two must coincide.
    assert abs(p_adj - p_naive) < 1e-9, (p_adj, p_naive)
    print(f"[selftest 1] 1 unit/cluster: p_adj={p_adj:.4f} == p_naive={p_naive:.4f}  OK")

    # Case 1b: 3 clusters, TWO units each, all concordant -> cluster is MORE
    # conservative than naive (dependence reduces effective sample size).
    s1 = [1, 1, 1, 1, 1, 1]
    c1 = [1, 1, 2, 2, 3, 3]
    p_adj1, _ = clustered_permutation_p(s1, c1, exact=True)
    p_naive1 = naive_exact_sign_test_p(s1)
    assert p_adj1 > p_naive1 - 1e-9, (p_adj1, p_naive1)
    print(f"[selftest 1b] 2 units/cluster concordant: p_adj={p_adj1:.4f} > p_naive={p_naive1:.4f}  OK")

    # Case 2: two opposing clusters -> cluster adjustment weakens significance.
    # cluster A: 5 compressed; cluster B: 5 expanded. Observed |T|=0.
    s2 = [1, 1, 1, 1, 1, -1, -1, -1, -1, -1]
    c2 = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
    p_adj2, _ = clustered_permutation_p(s2, c2, exact=True)
    p_naive2 = naive_exact_sign_test_p(s2)
    # naive sees 5/10 -> p_naive ~ 1.0; cluster sees T_obs=0 which is the median
    # of {-10,0,0,0,0,0,0,0,0,0,10}? Actually 2 clusters -> signs in {-,+}:
    #   both + -> T=+10 ; both - -> T=-10 ; opp -> T=0. T_obs=0.
    #   |T|>=0 always -> p_adj = 1.0. So cluster p >> naive.
    assert p_adj2 >= p_naive2 - 1e-9, (p_adj2, p_naive2)
    print(f"[selftest 2] opposing clusters: p_adj={p_adj2:.4f}  p_naive={p_naive2:.4f}  OK")

    # Case 3: exact enumeration count equals 2^{#clusters}
    _, n3 = clustered_permutation_p([1, -1, 1], [1, 2, 3], exact=True)
    assert n3 == 8, n3
    print(f"[selftest 3] exact enumeration size = {n3} (=2^3)  OK")
    print("ALL SELFTESTS PASSED")


if __name__ == "__main__":
    _selftest()
