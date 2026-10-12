#!/usr/bin/env python3
"""compute_intervention_bands.py -- Execute the Block A.1 intervention-stability
band sweep for the Paper 1 geometric dual-spectrum extension.

Reads : paper1/kc_trajectory/trajectory.json
        (produced by scan_idrem_trajectory.py from
         F:/UNAGI_Project/UNAGI/decoder_out -- no new training, no F: copy needed)

Writes : paper1/kc_trajectory/intervention_bands.json

Bands computed (decoder-readout / pullback-metric spectrum only):
  (1) seed-resampling  : neural vanilla (R0 x {42,123,456}) and hematopoietic
        prior (R15 x {42,123,456}) -> spread of tr, eff_rank, top1_share, dim%
  (2) prior injection/removal : full PG-UNAGI (R1) vs vanilla (R0) per iteration
        -> geometric vanilla-equivalence band on tr and eff_rank (ratio)
  (3) training-iteration sweep : R0 within-model spread across iter 0..4

NOTE: R_eff(Sigma_Z) (data-covariance spectrum) is NOT available here --
trajectory.json holds only the decoder readout (W.T) spectrum. It would require
the latent Z covariance and is deferred (no new training, but needs Z).
"""
import json, os
import numpy as np

TRAJ = "F:/UNAGI_Project/paper1/kc_trajectory/trajectory.json"
OUT = "F:/UNAGI_Project/paper1/kc_trajectory/intervention_bands.json"

rows = json.load(open(TRAJ))


def by_config(names):
    return [r for r in rows if r["config"] in names]


def band(metric, sub):
    vals = [r[metric] for r in sub if r.get(metric) is not None]
    a = np.array(vals, dtype=float)
    if a.size == 0:
        return None
    return dict(n=int(a.size), min=float(a.min()), max=float(a.max()),
                mean=float(a.mean()), sd=float(a.std(ddof=0)),
                range=float(a.max() - a.min()))


METRICS = ["tr", "eff_rank", "top1_share", "top5_share",
           "dim1_pct", "dim3_pct", "dim55_pct"]

nv = by_config({"R0_seed42", "R0_seed123", "R0_seed456"})
hp = by_config({"R15_seed42", "R15_seed123", "R15_seed456"})

result = {
    "neural_vanilla_3seed_R0": {m: band(m, nv) for m in METRICS},
    "hemo_prior_3seed_R15": {m: band(m, hp) for m in METRICS},
}


def per_iter_cfg(name):
    d = {}
    for r in rows:
        if r["config"] == name and r["iteration"] is not None:
            d[r["iteration"]] = r
    return d


r1, r0 = per_iter_cfg("R1"), per_iter_cfg("R0")
ratios_tr, ratios_er = [], []
for it in sorted(set(r1) & set(r0)):
    if r1[it].get("tr") and r0[it].get("tr"):
        ratios_tr.append(r1[it]["tr"] / r0[it]["tr"])
    if r1[it].get("eff_rank") and r0[it].get("eff_rank"):
        ratios_er.append(r1[it]["eff_rank"] / r0[it]["eff_rank"])


def rng(x):
    a = np.array(x, float)
    return dict(min=float(a.min()), max=float(a.max()), mean=float(a.mean()))


result["prior_injection_R1_vs_R0"] = {
    "tr_ratio": rng(ratios_tr),
    "eff_rank_ratio": rng(ratios_er),
    "iterations": sorted(set(r1) & set(r0)),
}

r0sweep = by_config({"R0"})
result["iteration_sweep_R0"] = {m: band(m, r0sweep) for m in METRICS}

result["notes"] = ("Bands computed from decoder-readout (W.T) spectrum only. "
                   "R_eff(Sigma_Z) deferred: needs latent Z covariance, not in trajectory.json.")
json.dump(result, open(OUT, "w"), indent=2)
print(json.dumps(result, indent=2))
