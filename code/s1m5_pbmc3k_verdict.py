# -*- coding: utf-8 -*-
"""S1m5 -- verdict for the scVI/pbmc3k cross-architecture pair.

Combines the three probes into one decision table:

  s1m3  mechanistic probe   -- labels_key leaves the latent bit-identical
                               (and the fitted parameter count unchanged);
                               categorical_covariate_keys does not.
  s1m   seed band           -- 10 vanilla seeds, label arm inert by construction;
                               cross-seed null = two vanilla fits.
  s1m4  injection band      -- 6 paired seeds with the prior registered the way the
                               Methods describe (categorical covariate).

The three questions this answers:

  Q1  Did the published pair inject anything?         (no)
  Q2  Is the published pair's separation real?        (no -- inside the seed null)
  Q3  With a genuine injection, does the audit fire?  (yes, decoder readout)

Also forms the paired difference D = kappa-C(Z) - kappa-C(W) per seed, which is the
dissociation-relevant statistic, and compares it against the cross-seed vanilla null
of the same difference.

Reads  : s1m_pbmc3k_seed_band.json, s1m4_pbmc3k_injection.json, s1m3_scvi_injection_probe.json
Writes : outputs/positive_result/s1m5_pbmc3k_verdict.json
"""
from __future__ import annotations

import itertools
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT = os.path.join(ROOT, "outputs", "positive_result")

PUB = dict(kappa_C_W_trace=0.872875208973927,
           kappa_C_Z=1.0778976280359904,
           eff_rank_vanilla=12.567126033416036,
           eff_rank_conditioned=15.660359435356279)


def load(n):
    with open(os.path.join(OUT, n), encoding="utf-8") as f:
        return json.load(f)


def tstat(v):
    v = np.asarray(v, float)
    return float(v.mean() / (v.std(ddof=1) / np.sqrt(v.size)))


def main():
    inert = load("s1m_pbmc3k_seed_band.json")
    inj = load("s1m4_pbmc3k_injection.json")
    probe = load("s1m3_scvi_injection_probe.json")

    rows_i = inj["per_seed"]
    kcw = np.array([r["kappa_corrected"] for r in rows_i])
    kcz = np.array([r["kappa_C_Z"] for r in rows_i])
    d_pair = kcz - kcw

    # null distribution of the same difference over vanilla cross-seed pairs
    sv = [r["sum_r2_vanilla"] for r in rows_i]
    tz = [r["trace_Z_vanilla"] for r in rows_i]
    d_null = []
    for a, b in itertools.combinations(range(len(rows_i)), 2):
        d_null.append((tz[a] / tz[b]) - (sv[a] / sv[b]))
    d_null = np.array(d_null)

    null_c = inj["cross_seed_vanilla_null"]            # categorical-covariate arm (Panel B / Q3)
    null_inert = inert["cross_seed_vanilla_null"]      # labels_key arm (Panel A / Q2, re-run as written)
    van_er = [r["eff_rank_vanilla"] for r in inert["per_seed"]]

    out = dict(
        environment=dict(scvi_tools=inj["architecture_scvi_tools"],
                         torch=inj.get("torch"), scanpy=inj.get("scanpy"),
                         gene_likelihood=inj["gene_likelihood"],
                         max_epochs=inj["max_epochs"], accelerator=inj["accelerator"]),
        Q1_injection_route=dict(
            asked="labels_key (what code/scvi_demo_benchmark.py passed)",
            latent_max_abs_dev=probe["arms"][1]["max_abs_dev_from_vanilla"],
            bitwise_identical=probe["arms"][1]["bitwise_identical_to_vanilla"],
            n_parameters_vanilla=probe["arms"][0]["n_parameters"],
            n_parameters_labels_key=probe["arms"][1]["n_parameters"],
            n_parameters_covariate=probe["arms"][2]["n_parameters"],
            verdict="INERT"),
        Q2_published_pair=dict(
            published_kappa_C_W_trace=PUB["kappa_C_W_trace"],
            # Panel A re-runs the published pair *as written* (prior registered with
            # labels_key), so its reference null must be the labels_key (inert) arm,
            # NOT the categorical-covariate arm used in Panel B / Q3.
            null_mean=null_inert["kappa_trace"]["mean"], null_sd=null_inert["kappa_trace"]["sd"],
            null_band=[null_inert["kappa_trace"]["lo2s"], null_inert["kappa_trace"]["hi2s"]],
            z=float((PUB["kappa_C_W_trace"] - null_inert["kappa_trace"]["mean"])
                    / null_inert["kappa_trace"]["sd"]),
            published_eff_rank_pair=[PUB["eff_rank_vanilla"], PUB["eff_rank_conditioned"]],
            vanilla_eff_rank_band=[float(min(van_er)), float(max(van_er))],
            vanilla_eff_rank_mean=float(np.mean(van_er)),
            vanilla_eff_rank_sd=float(np.std(van_er, ddof=1)),
            verdict="inside the seed null -- no evidence of an effect"),
        Q3_genuine_injection=dict(
            route="categorical_covariate_keys (as the Methods describe)",
            n_seeds=len(rows_i),
            kappa_C_W_corrected=dict(mean=float(kcw.mean()), sd=float(kcw.std(ddof=1)),
                                     min=float(kcw.min()), max=float(kcw.max()),
                                     per_seed=[float(x) for x in kcw]),
            kappa_C_W_trace=dict(mean=float(np.mean([r["kappa_trace"] for r in rows_i])),
                                 sd=float(np.std([r["kappa_trace"] for r in rows_i], ddof=1)),
                                 null_mean=null_c["kappa_trace"]["mean"],
                                 null_sd=null_c["kappa_trace"]["sd"]),
            kappa_C_Z=dict(mean=float(kcz.mean()), sd=float(kcz.std(ddof=1)),
                           null_mean=null_c["kappa_C_Z"]["mean"],
                           null_sd=null_c["kappa_C_Z"]["sd"]),
            z_vs_null_corrected=float((kcw.mean() - null_c["kappa_corrected"]["mean"])
                                      / null_c["kappa_corrected"]["sd"]),
            z_vs_null_trace=float((np.mean([r["kappa_trace"] for r in rows_i])
                                   - null_c["kappa_trace"]["mean"])
                                  / null_c["kappa_trace"]["sd"]),
            z_vs_null_kappa_C_Z=float((kcz.mean() - null_c["kappa_C_Z"]["mean"])
                                      / null_c["kappa_C_Z"]["sd"]),
            effective_rank_delta_pct=[r["eff_rank_delta_pct"] for r in rows_i],
            verdict="the audit fires: the decoder readout compresses"),
        dissociation=dict(
            definition="D = kappa-C(Z) - kappa-C(W), paired, corrected form",
            mean=float(d_pair.mean()), sd=float(d_pair.std(ddof=1)),
            t_one_sample=tstat(d_pair),
            null_mean=float(d_null.mean()), null_sd=float(d_null.std(ddof=1)),
            null_band=[float(d_null.mean() - 2 * d_null.std(ddof=1)),
                       float(d_null.mean() + 2 * d_null.std(ddof=1))],
            z_vs_null=float((d_pair.mean() - d_null.mean()) / d_null.std(ddof=1)),
        ),
        effective_rank_verdict=dict(
            mean_ratio=float(np.mean([r["eff_rank_ratio"] for r in rows_i])),
            null_mean=null_c["eff_rank_ratio"]["mean"],
            null_sd=null_c["eff_rank_ratio"]["sd"],
            z=float((np.mean([r["eff_rank_ratio"] for r in rows_i])
                     - null_c["eff_rank_ratio"]["mean"]) / null_c["eff_rank_ratio"]["sd"]),
            verdict="unchanged within seed scatter"),
    )

    q1, q2, q3 = out["Q1_injection_route"], out["Q2_published_pair"], out["Q3_genuine_injection"]
    dis = out["dissociation"]
    er = out["effective_rank_verdict"]

    print("=" * 100)
    print("scVI / pbmc3k cross-architecture pair -- verdict")
    print("=" * 100)
    print(f"Q1  did `labels_key` inject anything?")
    print(f"    fitted parameters: vanilla {q1['n_parameters_vanilla']:,}  "
          f"labels_key {q1['n_parameters_labels_key']:,}  "
          f"(categorical covariate {q1['n_parameters_covariate']:,})")
    print(f"    max|dZ| vs vanilla = {q1['latent_max_abs_dev']:.1f}  "
          f"bitwise identical = {q1['bitwise_identical']}  -> {q1['verdict']}")
    print()
    print(f"Q2  is the published pair real?")
    print(f"    published trace kappa-C(W) = {q2['published_kappa_C_W_trace']:.4f}  "
          f"vs seed null {q2['null_mean']:.4f} +- {q2['null_sd']:.4f} "
          f"(band {[round(x,3) for x in q2['null_band']]})  z = {q2['z']:+.2f}")
    print(f"    published eff_rank pair {q2['published_eff_rank_pair']} vs "
          f"10-seed range {[round(x,2) for x in q2['vanilla_eff_rank_band']]} "
          f"(mean {q2['vanilla_eff_rank_mean']:.2f} +- {q2['vanilla_eff_rank_sd']:.2f})")
    print(f"    -> {q2['verdict']}")
    print()
    print(f"Q3  with the prior registered as a categorical covariate (n={q3['n_seeds']} paired seeds):")
    print(f"    kappa-C(W) corrected = {q3['kappa_C_W_corrected']['mean']:.4f} "
          f"+- {q3['kappa_C_W_corrected']['sd']:.4f}  "
          f"range {q3['kappa_C_W_corrected']['min']:.3f}-{q3['kappa_C_W_corrected']['max']:.3f}"
          f"   z vs null = {q3['z_vs_null_corrected']:+.1f}")
    print(f"    kappa-C(W) trace     = {q3['kappa_C_W_trace']['mean']:.4f} "
          f"+- {q3['kappa_C_W_trace']['sd']:.4f}   z vs null = {q3['z_vs_null_trace']:+.2f} "
          f"(null sd {q3['kappa_C_W_trace']['null_sd']:.4f})")
    print(f"    kappa-C(Z)           = {q3['kappa_C_Z']['mean']:.4f} "
          f"+- {q3['kappa_C_Z']['sd']:.4f}   z vs null = {q3['z_vs_null_kappa_C_Z']:+.2f}")
    print(f"    effective rank deltas: "
          f"{[round(x,1) for x in q3['effective_rank_delta_pct']]} %  -> {er['verdict']} "
          f"(z = {er['z']:+.2f})")
    print()
    print(f"    dissociation D = kappa-C(Z) - kappa-C(W) = {dis['mean']:+.4f} "
          f"+- {dis['sd']:.4f}  (one-sample t = {dis['t_one_sample']:+.2f}, n={q3['n_seeds']})")
    print(f"      vanilla null of D: {dis['null_mean']:+.4f} +- {dis['null_sd']:.4f}  "
          f"-> z = {dis['z_vs_null']:+.2f}")

    with open(os.path.join(OUT, "s1m5_pbmc3k_verdict.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(f"\nwrote {os.path.join(OUT, 's1m5_pbmc3k_verdict.json')}")


if __name__ == "__main__":
    main()
