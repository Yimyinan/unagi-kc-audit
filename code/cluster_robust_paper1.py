# -*- coding: utf-8 -*-
"""
cluster_robust_paper1.py
========================

Cluster-robust binomial re-analysis of the Paper1 directional latent-Z claims,
replacing the near-vacuous seed-clustered permutation (3 seeds -> 2**3 = 8
flip-configurations -> maximum one-sided power 1/8) with a proper
cluster-robust binomial (Liang-Zeger sandwich SE + cluster-bootstrap CI) that
reports a confidence interval on the compressed proportion.

Claim families (same data source as compute_clustered_paper1.py):

  (A) 17/18 -- hematopoietic latent-Z participation ratio decreases iter0 -> iter4.
               Source: positive_result/z_spectrum_all.csv (pr column, iter 0&4),
               clustered by training seed.
  (B) kappa-C-Z directional latent compression (per tissue + pooled 22/29).
               Source: same CSV; a prior run is "directionally compressed" if its
               terminal pr < vanilla centre pr (per tissue); clustered by seed.

The script prints a report and writes data/cluster_robust_paper1_results.json.
The existing seed-clustered permutation p is kept alongside for comparison.
"""
from __future__ import annotations

import csv
import json
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "..")                                  # paper1_kc_diagnostic
SRC = os.path.join(REPO, "..", "positive_result", "z_spectrum_all.csv")
OUT = os.path.join(REPO, "data", "cluster_robust_paper1_results.json")

sys.path.insert(0, HERE)
from cluster_robust_binomial import cluster_robust_proportion
from clustered_permutation_sign_test import (
    clustered_permutation_p,
    naive_exact_sign_test_p,
)

TISSUE_LABEL = {"hemo": "hematopoietic", "neural": "neural crest", "lung": "lung fibrosis"}


def load_rows(path):
    return list(csv.DictReader(open(path, newline="")))


def terminal_iter(rows, run):
    return max(int(r["iteration"]) for r in rows if r["run"] == run)


def vanilla_centre(rows, tissue):
    van = [r for r in rows if r["tissue"] == tissue and r["role"] == "vanilla"]
    term = {r["run"]: int(r["iteration"]) for r in van}
    vals = [float(r["pr"]) for r in van if int(r["iteration"]) == term[r["run"]]]
    return float(sum(vals) / len(vals)), len(van)


def _summarize(label, signed, clusters):
    """Return both the old permutation p and the new cluster-robust binomial."""
    n = len(signed)
    k = sum(1 for s in signed if s > 0)
    p_naive = naive_exact_sign_test_p(signed)
    p_adj, _ = clustered_permutation_p(signed, clusters, exact=True)
    crb = cluster_robust_proportion(
        [1.0 if s > 0 else 0.0 for s in signed], clusters,
        p0=0.5, alternative="greater",
    )
    return {
        "claim": label,
        "n_units": n,
        "n_compressed": k,
        "compressed_proportion": round(k / n, 4) if n else None,
        "seed_clusters": {str(c): clusters.count(c) for c in sorted(set(clusters), key=str)},
        "naive_sign_test_p": p_naive,
        "seed_clustered_perm_p": p_adj,
        "cluster_robust": {
            "p_hat": round(crb["p_hat"], 4),
            "se_robust": round(crb["se_robust"], 4),
            "analytic_ci": [round(x, 4) for x in crb["analytic_ci"]],
            "bootstrap_ci": [round(x, 4) for x in crb["bootstrap_ci"]],
            "p_one_sided_normal": round(crb["p_value_normal"], 4),
            "p_one_sided_bootstrap": round(crb["p_value_bootstrap"], 4),
            "n_clusters": crb["n_clusters"],
            "n_boot": crb["n_boot"],
        },
    }


# ---------------------------------------------------------------- (A) 17/18
def analyze_17_18(rows):
    hema_prior = list(dict.fromkeys(
        r["run"] for r in rows
        if r["tissue"] == "hemo" and r["role"] in ("prior", "ablation")))
    signed, clusters = [], []
    for run in hema_prior:
        e0 = next((float(r["pr"]) for r in rows if r["run"] == run and int(r["iteration"]) == 0), None)
        e4 = next((float(r["pr"]) for r in rows if r["run"] == run and int(r["iteration"]) == 4), None)
        if e0 is None or e4 is None:
            continue
        seed = next(r["seed"] for r in rows if r["run"] == run)
        signed.append(1.0 if e4 < e0 else -1.0)
        clusters.append(seed)
    return _summarize("17/18 hema iter0->iter4", signed, clusters)


# ----------------------------------------------------- (B) kappa-C-Z per tissue
def analyze_kc_z(rows, tissue):
    centre, _ = vanilla_centre(rows, tissue)
    priors = [r for r in rows if r["tissue"] == tissue and r["role"] in ("prior", "ablation")]
    term = {r["run"]: int(r["iteration"]) for r in priors}
    signed, clusters = [], []
    for r in priors:
        if int(r["iteration"]) != term[r["run"]]:
            continue
        sign = 1.0 if float(r["pr"]) < centre else -1.0
        signed.append(sign)
        clusters.append(r["seed"])
    label = f"kappa-C-Z {TISSUE_LABEL.get(tissue, tissue)}"
    return _summarize(label, signed, clusters)


def main():
    rows = load_rows(SRC)
    out = {
        "source": os.path.relpath(SRC, REPO),
        "method": ("cluster-robust binomial (Liang-Zeger sandwich SE + cluster-bootstrap "
                   "CI) replacing seed-clustered permutation; H0: p<=0.5 (no directional preference)"),
        "results": {},
    }
    out["results"]["17_18"] = analyze_17_18(rows)
    kcz = {}
    for t in ["hemo", "neural", "lung"]:
        r = analyze_kc_z(rows, t)
        if r["n_units"] > 0:
            kcz[t] = r
    # pooled 22/29 (per-tissue runs, cluster label = tissue:seed)
    pooled_signed, pooled_clusters = [], []
    for t in ["hemo", "neural", "lung"]:
        centre, _ = vanilla_centre(rows, t)
        priors = [r for r in rows if r["tissue"] == t and r["role"] in ("prior", "ablation")]
        term = {r["run"]: int(r["iteration"]) for r in priors}
        for r in priors:
            if int(r["iteration"]) != term[r["run"]]:
                continue
            pooled_signed.append(1.0 if float(r["pr"]) < centre else -1.0)
            pooled_clusters.append(f"{t}:{r['seed']}")
    kcz["pooled_22_29"] = _summarize("kappa-C-Z pooled 22/29", pooled_signed, pooled_clusters)
    out["results"]["kappa_c_z"] = kcz

    json.dump(out, open(OUT, "w"), indent=2)

    # ---- human-readable report ----
    print("=" * 78)
    print("CLUSTER-ROBUST BINOMIAL RE-ANALYSIS (Paper1 directional latent-Z claims)")
    print("=" * 78)
    for key, res in [("17_18", out["results"]["17_18"])] + [(k, v) for k, v in out["results"]["kappa_c_z"].items()]:
        cr = res["cluster_robust"]
        print(f"\n[{res['claim']}]  n={res['n_units']}  compressed={res['n_compressed']}  "
              f"prop={res['compressed_proportion']}  clusters={cr['n_clusters']}")
        print(f"   seed clusters: {res['seed_clusters']}")
        print(f"   naive sign-test p      = {res['naive_sign_test_p']:.5f}")
        print(f"   seed-clustered perm p  = {res['seed_clustered_perm_p']:.5f}   (power <= 1/8)")
        print(f"   cluster-robust p_hat   = {cr['p_hat']:.3f}  SE={cr['se_robust']:.3f}")
        print(f"   analytic  95% CI       = [{cr['analytic_ci'][0]:.3f}, {cr['analytic_ci'][1]:.3f}]")
        print(f"   bootstrap 95% CI       = [{cr['bootstrap_ci'][0]:.3f}, {cr['bootstrap_ci'][1]:.3f}]")
        print(f"   one-sided p (normal)   = {cr['p_one_sided_normal']:.4f}")
        print(f"   one-sided p (boot)     = {cr['p_one_sided_bootstrap']:.4f}")
    print("\nWrote:", os.path.relpath(OUT, REPO))


if __name__ == "__main__":
    main()
