# -*- coding: utf-8 -*-
"""
Real-data clustered-permutation re-analysis for the Paper1 kappa-C audit.

Two claim families are re-tested with a seed-clustered permutation sign test
(cluster-level sign flips) instead of the naive exact sign test that the
manuscript already flags as "overstat[ing] precision" (main text L73).

  (A) 17/18  -- hematopoietic latent-Z participation-ratio decrease iter0 -> iter4.
                Source: positive_result/z_spectrum_all.csv (pr column, iter 0&4,
                clustered by training seed). NO stagedata needed; pr is
                already tabulated in the CSV.

  (B) 16/19 / 5/9 / 22/29 -- kappa-C-Z directional latent compression.
                Source: same CSV. Per tissue, vanilla centre = mean terminal
                pr (participation ratio) of vanilla runs; a prior run is
                "directionally compressed" if its terminal pr < vanilla centre;
                runs are clustered by seed.

The clustered permutation enumerates all 2^{#seed_clusters} sign-flip
assignments of cluster means and yields a permutation p that is the honest
confirmatory counterpart to the naive sign-test p.
"""
from __future__ import annotations
import csv, os, json, sys
from collections import defaultdict, Counter
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "..")                       # paper1_kc_diagnostic
SRC  = os.path.join(REPO, "..", "positive_result", "z_spectrum_all.csv")  # paper1/positive_result
OUT  = os.path.join(REPO, "data", "paper1_clustered_permutation_results.json")

sys.path.insert(0, HERE)
from clustered_permutation_sign_test import (
    clustered_permutation_p, naive_exact_sign_test_p,
)

TISSUE_LABEL = {"hemo": "hematopoietic", "neural": "neural crest", "lung": "lung fibrosis"}


def load_rows(path):
    return list(csv.DictReader(open(path, newline="")))


def terminal_iter(rows, run):
    """Max iteration recorded for a run."""
    return max(int(r["iteration"]) for r in rows if r["run"] == run)


def vanilla_centre(rows, tissue):
    van = [r for r in rows if r["tissue"] == tissue and r["role"] == "vanilla"]
    term = {r["run"]: int(r["iteration"]) for r in van}
    vals = [float(r["pr"]) for r in van
            if int(r["iteration"]) == term[r["run"]]]
    return float(sum(vals) / len(vals)), len(van)


# ---------------------------------------------------------------- (A) 17/18
def analyze_17_18(rows):
    hema_prior = list(dict.fromkeys(
        r["run"] for r in rows
        if r["tissue"] == "hemo" and r["role"] in ("prior", "ablation")))
    pairs = []
    for run in hema_prior:
        e0 = next((float(r["pr"]) for r in rows
                   if r["run"] == run and int(r["iteration"]) == 0), None)
        e4 = next((float(r["pr"]) for r in rows
                   if r["run"] == run and int(r["iteration"]) == 4), None)
        if e0 is None or e4 is None:
            continue
        seed = next(r["seed"] for r in rows if r["run"] == run)
        pairs.append((run, seed, e0, e4))
    signed, clusters = [], []
    for run, seed, e0, e4 in pairs:
        signed.append(1.0 if e4 < e0 else -1.0)
        clusters.append(seed)
    n = len(signed)
    k = sum(1 for s in signed if s > 0)
    p_naive = naive_exact_sign_test_p(signed)
    p_adj, _ = clustered_permutation_p(signed, clusters, exact=True)
    return {
        "claim": "17/18",
        "description": "hematopoietic latent-Z participation ratio decreases iter0->iter4",
        "n_configs": n,
        "n_compressed": k,
        "seed_clusters": {str(c): clusters.count(c) for c in sorted(set(clusters), key=str)},
        "naive_sign_test_p": p_naive,
        "cluster_adjusted_p": p_adj,
        "note": "Replaces the over-precise naive p=0.000145 with the seed-clustered permutation p.",
    }


# ---------------------------------------------------------------- (B) kappa-C-Z
def analyze_kc_z(rows, tissue):
    centre, n_van = vanilla_centre(rows, tissue)
    priors = [r for r in rows if r["tissue"] == tissue and r["role"] in ("prior", "ablation")]
    # terminal eff_rank per prior run
    term = {r["run"]: int(r["iteration"]) for r in priors}
    signed, clusters = [], []
    runs_meta = []
    for r in priors:
        if int(r["iteration"]) != term[r["run"]]:
            continue
        val = float(r["pr"])
        sign = 1.0 if val < centre else -1.0
        signed.append(sign)
        clusters.append(r["seed"])
        runs_meta.append((r["run"], r["seed"], round(val, 3), round(centre, 3), sign > 0))
    n = len(signed)
    k = sum(1 for s in signed if s > 0)
    if n == 0:
        return None
    p_naive = naive_exact_sign_test_p(signed)
    p_adj, _ = clustered_permutation_p(signed, clusters, exact=True)
    return {
        "claim": f"kappa-C-Z {tissue}",
        "description": f"{TISSUE_LABEL.get(tissue, tissue)} directional latent compression vs vanilla-only null",
        "n_vanilla": n_van,
        "vanilla_centre_pr": round(centre, 3),
        "n_prior_runs": n,
        "n_directionally_compressed": k,
        "seed_clusters": {str(c): clusters.count(c) for c in sorted(set(clusters), key=str)},
        "naive_sign_test_p": p_naive,
        "cluster_adjusted_p": p_adj,
    }


def main():
    rows = load_rows(SRC)
    out = {"source": os.path.relpath(SRC, REPO),
           "method": "seed-clustered exact permutation sign test (2^{#clusters} cluster sign-flips)",
           "results": {}}
    out["results"]["17_18"] = analyze_17_18(rows)
    kcz = {}
    for t in ["hemo", "neural", "lung"]:
        r = analyze_kc_z(rows, t)
        if r:
            kcz[t] = r
    # cross-tissue pooled 22/29
    all_signed, all_clusters = [], []
    for t, r in kcz.items():
        pass
    # rebuild pooled from the per-tissue runs
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
    k = sum(1 for s in pooled_signed if s > 0)
    kcz["pooled_22_29"] = {
        "claim": "kappa-C-Z pooled 22/29",
        "n_prior_runs": len(pooled_signed),
        "n_directionally_compressed": k,
        "naive_sign_test_p": naive_exact_sign_test_p(pooled_signed),
        "cluster_adjusted_p": clustered_permutation_p(pooled_signed, pooled_clusters, exact=True)[0],
        "seed_clusters": {str(c): pooled_clusters.count(c) for c in sorted(set(pooled_clusters), key=str)},
    }
    out["results"]["kappa_c_z"] = kcz
    json.dump(out, open(OUT, "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
