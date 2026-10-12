"""Power assessment for the like-for-like participation-ratio test, and the Ridge/Jacobian
dissociation under both estimators.

Recreated 2026-10-08 after the workspace cleanup incident.

Statistics over the 33 configurations:
  r_Z (latent) and r_W (decoder) are participation-ratio ratios against each configuration's
  vanilla baseline; the dissociation is d = log r_Z - log r_W tested against zero.  The 33
  configurations trace to three training seeds, so power is computed both i.i.d. and
  cluster-robust (t-test on the 3 seed means, df=2).

Reads  : data/kappa_c_pr_comparison.json, data/unagi_jacobian_reaudit_33.json
Writes : data/closing_gaps_power_and_bound.json, data/dissociation_jacobian_ridge.json
"""
import collections
import json
import os
import re
import sys
from statistics import NormalDist

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER1 = os.path.abspath(os.path.join(HERE, ".."))
PR = os.path.join(PAPER1, "data", "kappa_c_pr_comparison.json")
JAC = os.path.join(PAPER1, "data", "unagi_jacobian_reaudit_33.json")
RNG = np.random.default_rng(20261008)
NSIM = 20000
ALPHA = 0.05


def seed_of(cfg):
    m = re.search(r"seed[_-]?(\d+)", cfg) or re.search(r"_s(\d{2,3})\b", cfg)
    return int(m.group(1)) if m else None


def t_power(x, alpha=ALPHA):
    n = x.shape[0]
    sd = np.maximum(x.std(axis=0, ddof=1), 1e-12)
    t = x.mean(axis=0) / (sd / np.sqrt(n))
    z = NormalDist().inv_cdf(1 - alpha / 2)
    tcrit = z * (1.0 + (z * z + 1.0) / (4.0 * (n - 1)))
    return float(np.mean(np.abs(t) > tcrit))


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    rows = json.load(open(PR, encoding="utf-8"))["rows"]
    lat = {r["config"]: r["pr_ratio_Z"] for r in rows}
    dec = {r["config"]: r["pr_ratio_W"] for r in rows}
    tis = {r["config"]: r["tissue"] for r in rows}
    jac = {}
    for _t, blk in json.load(open(JAC, encoding="utf-8"))["tissues"].items():
        for c, v in blk["configs"].items():
            jac[c] = v["kappa_jac_A"]

    cs = [c for c in jac if c in lat and c in dec]
    L = np.array([lat[c] for c in cs])
    W = np.array([dec[c] for c in cs])
    J = np.array([jac[c] for c in cs])
    tn = np.array([tis[c] for c in cs])

    groups = collections.defaultdict(list)
    for c, v in zip(cs, np.log(L) - np.log(W)):
        s = seed_of(c)
        if s is not None:
            groups[s].append(v)
    k = len(groups)
    cmeans = np.array([np.mean(v) for v in groups.values()])
    between = float(cmeans.std(ddof=1))
    within = float(np.concatenate([np.array(v) - np.mean(v) for v in groups.values()]).std(ddof=1))
    icc = between ** 2 / (between ** 2 + within ** 2)
    design = 1.0 + (np.mean([len(v) for v in groups.values()]) - 1.0) * icc

    d_obs = float((np.log(L) - np.log(W)).mean())
    pooled = float((np.log(L) - np.log(W)).std(ddof=1))
    p_naive = t_power(RNG.normal(abs(d_obs), pooled, size=(len(cs), NSIM)))
    p_clust = t_power(RNG.normal(abs(d_obs), between, size=(k, NSIM)))

    def mde(fn, target=0.80):
        lo, hi = 1e-4, 2.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if fn(mid) >= target:
                hi = mid
            else:
                lo = mid
        return float(0.5 * (lo + hi))

    mde_naive = mde(lambda e: t_power(RNG.normal(e, pooled, size=(len(cs), NSIM))))
    mde_clust = mde(lambda e: t_power(RNG.normal(e, between, size=(k, NSIM))))

    print("n=%d  seeds=%d  ICC=%.3f  design effect=%.1f" % (len(cs), k, icc, design))
    print("  paired log-difference: %.4f (pooled sd %.4f)" % (d_obs, pooled))
    print("  power@observed: naive %.3f  cluster %.3f" % (p_naive, p_clust))
    print("  MDE@80%%: naive %.4f  cluster %.4f   observed/cluster %.2f"
          % (mde_naive, mde_clust, abs(d_obs) / mde_clust))

    power_out = {
        "test": "paired log-difference log(pr_ratio_Z) - log(pr_ratio_W), two-sided vs 0",
        "n_configurations": len(cs), "n_clusters": k,
        "seed_clusters": {int(s): len(v) for s, v in groups.items()},
        "observed": {"mean_pr_ratio_W": float(W.mean()), "mean_pr_ratio_Z": float(L.mean()),
                     "paired_log_difference_mean": d_obs, "pooled_sd": pooled,
                     "within_seed_sd": within, "between_seed_sd": between,
                     "implied_icc": float(icc), "design_effect": float(design)},
        "power_at_observed": {"naive_iid": p_naive, "cluster_k3": p_clust},
        "mde_80pct": {"naive_iid": mde_naive, "cluster": mde_clust},
        "verdict": ("Power at the observed effect is %.3f i.i.d. but %.3f cluster-robust; the "
                    "cluster MDE at 80%% power (%.3f) is about twice the observed effect (%.3f), "
                    "so the reported interval is not calibrated for a three-seed population."
                    % (p_naive, p_clust, mde_clust, abs(d_obs))),
    }

    dis = {"n_configurations": len(cs),
           "latent": {"mean": float(L.mean()), "median": float(np.median(L)),
                      "n_below_unity": int((L < 1).sum())},
           "decoder_ridge_pr": {"mean": float(W.mean()), "median": float(np.median(W)),
                                "min": float(W.min()), "max": float(W.max()),
                                "n_below_unity": int((W < 1).sum())},
           "decoder_jacobian": {"mean": float(J.mean()), "median": float(np.median(J)),
                                "min": float(J.min()), "max": float(J.max()),
                                "n_below_unity": int((J < 1).sum()),
                                "n_above_unity": int((J > 1).sum())},
           "concordance": {"direction_disagreements": int((((W - 1) * (J - 1)) < 0).sum()),
                           "spearman_overall": spearman(W, J), "per_tissue": {}}}
    for t in ("hemo", "neural"):
        m = tn == t
        dis["concordance"]["per_tissue"][t] = {"n": int(m.sum()), "spearman": spearman(W[m], J[m]),
                                              "ridge_range": [float(W[m].min()), float(W[m].max())],
                                              "jacobian_range": [float(J[m].min()), float(J[m].max())],
                                              "jacobian_n_above_unity": int((J[m] > 1).sum())}
    for nm, D in (("ridge", W), ("jacobian", J)):
        dd = np.log(L) - np.log(D)
        prod = D * L
        dis["dissociation_" + nm] = {
            "paired_log_difference_mean": float(dd.mean()),
            "paired_log_difference_sd": float(dd.std(ddof=1)),
            "naive_t": float(dd.mean() / (dd.std(ddof=1) / np.sqrt(len(dd)))),
            "product_median": float(np.median(prod)),
            "product_within_25pct_of_unity": int((np.abs(prod - 1) < 0.25).sum())}
    print("  dissociation vs Ridge: %+.4f (t %+.2f); vs Jacobian: %+.4f (t %+.2f)"
          % (dis["dissociation_ridge"]["paired_log_difference_mean"], dis["dissociation_ridge"]["naive_t"],
             dis["dissociation_jacobian"]["paired_log_difference_mean"], dis["dissociation_jacobian"]["naive_t"]))
    print("  direction disagreements %d/%d ; Spearman %.3f"
          % (dis["concordance"]["direction_disagreements"], len(cs), dis["concordance"]["spearman_overall"]))

    outs = {"--power-out=": power_out, "--dissoc-out=": dis}
    for arg in sys.argv[1:]:
        for key, payload in outs.items():
            if arg.startswith(key):
                p = arg.split("=", 1)[1]
                os.makedirs(os.path.dirname(p), exist_ok=True)
                json.dump(payload, open(p, "w", encoding="utf-8"), indent=1)
                print("wrote %s" % p)


if __name__ == "__main__":
    main()
