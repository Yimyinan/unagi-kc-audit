# -*- coding: utf-8 -*-
"""S1m2 -- where the published pbmc3k numbers sit inside the family-matched seed band.

Why the cross-seed null is the right reference
----------------------------------------------
Section 2.9 contrasts a "vanilla" scVI with a "prior-injected" scVI.  The probe in
``code/s1m3_scvi_injection_probe.py`` shows the two are the *same model*: passing
``labels_key`` to ``setup_anndata`` leaves the latent bit-identical, because labels
reach scVI's generative model only through ``dispersion="gene-label"`` and the
default is ``dispersion="gene"``.  The published pair is therefore a draw of two
independent fits of one configuration, and the correct reference distribution is
the cross-seed null of two vanilla fits -- not the paired band (which is
degenerate at exactly 1.0000 precisely because nothing was injected).

Reads  : outputs/positive_result/s1m_pbmc3k_seed_band.json
Writes : outputs/positive_result/s1m2_band_summary.json
Usage  : python code/s1m2_band_summary.py
"""
from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT = os.path.join(ROOT, "outputs", "positive_result")

# section 2.9 as published (data/scvi_demo_results.json, 2026-09-05)
PUB = dict(kappa_C_W_trace=0.872875208973927,
           kappa_C_Z=1.0778976280359904,
           eff_rank_vanilla=12.567126033416036,
           eff_rank_conditioned=15.660359435356279,
           eff_rank_delta_pct=24.613689667115004)
# the faithful re-run of the same declared configuration (s1l2)
FAITHFUL_CORRECTED = 0.971860551549297


def z_of(x, band):
    return (x - band["mean"]) / band["sd"] if band["sd"] else float("nan")


def verdict(x, band):
    return "inside" if band["lo2s"] <= x <= band["hi2s"] else "OUTSIDE"


def main():
    with open(os.path.join(OUT, "s1m_pbmc3k_seed_band.json"), encoding="utf-8") as f:
        d = json.load(f)
    null = d["cross_seed_vanilla_null"]
    paired = d["paired_label_vs_vanilla"]
    n_pairs = null["kappa_trace"]["n"]

    pub_ratio = PUB["eff_rank_conditioned"] / PUB["eff_rank_vanilla"]
    rows = [
        ("kappa-C(W) trace  (published 0.873)", PUB["kappa_C_W_trace"], null["kappa_trace"]),
        ("kappa-C(W) corr.  (faithful re-run 0.972)", FAITHFUL_CORRECTED,
         null["kappa_corrected"]),
        ("kappa-C(Z)        (published 1.078)", PUB["kappa_C_Z"], null["kappa_C_Z"]),
        ("eff-rank ratio    (published 1.246)", pub_ratio, null["eff_rank_ratio"]),
    ]

    out = dict(
        source="s1m_pbmc3k_seed_band.json",
        reference=("cross-seed null of two VANILLA scVI fits (n=%d unordered pairs); "
                   "section 2.9's two arms are the same configuration, see s1m3"
                   % n_pairs),
        paired_band_is_degenerate=all(abs(paired[k]["sd"]) < 1e-12
                                      for k in ("kappa_trace", "kappa_corrected",
                                                "kappa_C_Z", "eff_rank_ratio")),
        n_seeds=len(d["per_seed"]),
        vanilla_eff_rank_per_seed=[round(r["eff_rank_vanilla"], 3) for r in d["per_seed"]],
        published=dict(kappa_C_W_trace=PUB["kappa_C_W_trace"], kappa_C_Z=PUB["kappa_C_Z"],
                       eff_rank_vanilla=PUB["eff_rank_vanilla"],
                       eff_rank_conditioned=PUB["eff_rank_conditioned"],
                       eff_rank_ratio=pub_ratio),
        positions={}, )

    print(f"{'quantity':<44}{'published':>10}{'null mean':>11}{'null sd':>9}"
          f"{'2sigma band':>20}{'z':>7}  verdict")
    print("-" * 108)
    for label, x, band in rows:
        z = z_of(x, band)
        out["positions"][label] = dict(published=x, null_mean=band["mean"],
                                       null_sd=band["sd"],
                                       band=[band["lo2s"], band["hi2s"]],
                                       z=z, verdict=verdict(x, band))
        print(f"{label:<44}{x:>10.4f}{band['mean']:>11.4f}{band['sd']:>9.4f}"
              f"{'[' + format(band['lo2s'], '.3f') + ', ' + format(band['hi2s'], '.3f') + ']':>20}"
              f"{z:>+7.2f}  {verdict(x, band)}")

    infl = null["kappa_trace"]["sd"] / null["kappa_corrected"]["sd"]
    out["trace_form_sd_inflation"] = float(infl)
    out["paired_ratios_exactly_unity"] = {k: paired[k]["mean"] for k in paired}

    shr = [r for r in d["per_seed"]]
    lo = min(r["eff_rank_vanilla"] for r in shr)
    hi = max(r["eff_rank_vanilla"] for r in shr)
    out["vanilla_eff_rank_range"] = [lo, hi]
    out["published_eff_rank_pair_inside_seed_range"] = bool(
        lo <= PUB["eff_rank_vanilla"] and PUB["eff_rank_conditioned"] <= hi)

    print()
    print(f"trace-form null sd / corrected-form null sd = {infl:.1f}x  "
          f"(same fits, same pairs)")
    print(f"vanilla eff_rank across {len(shr)} seeds: {lo:.2f} -- {hi:.2f}")
    print(f"  published vanilla 12.57 -> 'conditioned' 15.66 falls inside that range: "
          f"{out['published_eff_rank_pair_inside_seed_range']}")
    print(f"paired (labels_key added, same seed): "
          f"{ {k: round(v, 6) for k, v in out['paired_ratios_exactly_unity'].items()} }")
    print("  -> exactly unity: the label arm is not a different model")

    with open(os.path.join(OUT, "s1m2_band_summary.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(f"\nwrote {os.path.join(OUT, 's1m2_band_summary.json')}")


if __name__ == "__main__":
    main()
