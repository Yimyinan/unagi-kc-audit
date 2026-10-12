# -*- coding: utf-8 -*-
"""S1h -- recompute the family-matched vanilla-null BANDS in both statistics.

Why
---
``kappa_C(W)`` as released is the trace ratio ``tr(W'W)_i / tr(W'W)_j``; the
manuscript names ``sum_g R2_g`` (see S1F_0OF33_RECHECK.md).  The null BAND is the
mean +/- 2 sd of cross-readout kappa ratios inside a no-prior pool, so it must be
recomputed in the corrected statistic too -- a band calibrated on the trace form
inherits the basis-rotation inflation.

Two independent provenance issues surfaced while doing this and are reported, not
silently patched:

  (1) The published hemo pool (``data/unaug_vanilla_null_final.json``) stores
      R14_seed42/123/456 trH = 4333.06 / 4156.59 / 5272.91.  The stagedata of the
      runs those names resolve to (``F:/UNAGI_Project/UNAGI/result/R15/
      R14_hemato_seed*``) recomputes to 5160.92 / 9109.91 / 14272.94.  The neural
      pool DOES reproduce (R0_seed123 6792.02, R0_seed456 9648.78).  So the
      published hemo band rests on numbers that cannot be regenerated from disk.

  (2) The R2 form cannot be computed from a stored W matrix -- it needs (Z, X).
      Runs without iter_4 stagedata on disk (R0_seed42, R0_vanilla_seed42) are
      therefore unreachable in the corrected statistic and are dropped from the
      corrected pools.  This is a hard information loss of the trace form's
      convenience, and it is stated rather than hidden.

Reads  : data/_ridge_recompute_cache.json          (Ridge recompute from stagedata,
                                                     carries tr AND sum_r2)
         data/unaug_vanilla_null_final.json        (published pool, tr only)
         outputs/positive_result/s1f_0of33_recheck.json  (33 configs, both forms)
Writes : outputs/positive_result/s1h_null_bands.json

Usage: python code/s1h_null_band_recompute.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
PAPER1 = os.path.abspath(os.path.join(HERE, ".."))
ROOT = os.path.abspath(os.path.join(PAPER1, "..", ".."))
OUT = os.path.join(ROOT, "outputs", "positive_result")

CROSS_FAMILY_BAND = (0.78, 1.23)               # trace form, as released
CROSS_FAMILY_BAND_CORRECTED = (0.937, 1.064)   # corrected predictability form (the manuscript's)
LOCKED = 0.5

TR = "tr"
R2 = "sum_r2"


def load_json(p):
    with open(p, encoding="utf-8-sig") as f:
        return json.load(f)


def ordered_ratios(vals):
    v = np.asarray(list(vals), dtype=float)
    n = len(v)
    if n < 2:
        return None
    return np.array([v[i] / v[j] for i in range(n) for j in range(n) if i != j])


def band_from_ratios(r):
    """Summarise an array of ordered ratios."""
    r = np.asarray(r, dtype=float)
    return dict(n_members=None, n_pairs=int(r.size),
                mean=float(r.mean()), sd=float(r.std(ddof=1)),
                lo2s=float(r.mean() - 2 * r.std(ddof=1)),
                hi2s=float(r.mean() + 2 * r.std(ddof=1)),
                min=float(r.min()), max=float(r.max()),
                p05=float(np.percentile(r, 5)), p95=float(np.percentile(r, 95)))


def band_of(vals):
    r = ordered_ratios(vals)
    if r is None:
        return None
    out = band_from_ratios(r)
    out["n_members"] = len(vals)
    return out


def pool_vals(cache, names, stat, published=None):
    """Values for `names` from the stagedata cache, else from a published table."""
    out, src = {}, {}
    for nm in names:
        spec = cache.get(nm)
        if spec and "error" not in spec and stat in spec:
            out[nm] = spec[stat]
            src[nm] = "stagedata"
        elif published and nm in published:
            out[nm] = published[nm]
            src[nm] = "published(tr only)"
    return out, src


def classify(kappa, band):
    if kappa is None or not np.isfinite(kappa):
        return "n/a"
    if kappa < LOCKED:
        return "locked"
    if band is None:
        return "no-band"
    if kappa < band[0]:
        return "sub-vanilla"
    if kappa > band[1]:
        return "expanded"
    return "vanilla-equivalent"


def main():
    cache = load_json(os.path.join(PAPER1, "data", "_ridge_recompute_cache.json"))
    pub = load_json(os.path.join(PAPER1, "data", "unaug_vanilla_null_final.json"))
    s1f = load_json(os.path.join(OUT, "s1f_0of33_recheck.json"))
    pub_pool = pub["vanilla_trH"]

    # ---------------- pools ----------------
    HEMO_NAMES = ["R14_hemato_vanilla", "R14_hemato_seed42",
                  "R14_hemato_seed123", "R14_hemato_seed456"]
    NEURAL_NAMES = ["R0_vanilla_seed123", "R0_vanilla_seed456"]
    NEURAL_PUB_NAMES = ["R0_seed42", "R0_seed123", "R0_seed456"]

    pools = {}
    for stat in (TR, R2):
        pv, src = pool_vals(cache, HEMO_NAMES, stat)
        pools[f"hemo_mixed__{stat}"] = dict(band=band_of(pv.values()),
                                            n_src=len(pv), source=src,
                                            values={k: float(v) for k, v in pv.items()})
        pv, src = pool_vals(cache, NEURAL_NAMES, stat)
        pools[f"neural_mixed__{stat}"] = dict(band=band_of(pv.values()),
                                              n_src=len(pv), source=src,
                                              values={k: float(v) for k, v in pv.items()})
        # combined pool = POOLED RATIOS, not pooled raw magnitudes (a ratio band must
        # never mix readouts of different gene sets before taking ratios)
        comb_ratios = np.concatenate([
            ordered_ratios(pools[f"hemo_mixed__{stat}"]["values"].values()),
            ordered_ratios(pools[f"neural_mixed__{stat}"]["values"].values()),
        ])
        cb = band_from_ratios(comb_ratios)
        cb["n_members"] = (pools[f"hemo_mixed__{stat}"]["n_src"]
                           + pools[f"neural_mixed__{stat}"]["n_src"])
        pools[f"unagi_combined__{stat}"] = dict(band=cb, n_src=cb["n_members"],
                                                source={}, values={})

    # vehicle-only hemo pool (the honest seed-replicate null; excludes true vanilla)
    pv, src = pool_vals(cache, HEMO_NAMES[1:], TR)
    pools["hemo_vehicle_only__tr"] = dict(band=band_of(pv.values()), n_src=len(pv),
                                          source=src,
                                          values={k: float(v) for k, v in pv.items()})
    pv, src = pool_vals(cache, HEMO_NAMES[1:], R2)
    pools["hemo_vehicle_only__sum_r2"] = dict(band=band_of(pv.values()), n_src=len(pv),
                                              source=src,
                                              values={k: float(v) for k, v in pv.items()})

    # published pool, as stored (hemo 4 members / neural 3 members)
    pv, src = pool_vals(cache, HEMO_NAMES, TR, published=pub_pool.get("hemo"))
    pools["hemo_published__tr"] = dict(band=band_of(pv.values()), n_src=len(pv),
                                       source=src,
                                       values={k: float(v) for k, v in pv.items()})
    pv, src = pool_vals(cache, NEURAL_PUB_NAMES, TR, published=pub_pool.get("neural"))
    pools["neural_published__tr"] = dict(band=band_of(pv.values()), n_src=len(pv),
                                         source=src,
                                         values={k: float(v) for k, v in pv.items()})

    # ---------------- provenance check ----------------
    # published keys are short names ("R14_seed42"); cache keys are full run names
    ALIAS = {
        "R14": "R14_hemato_vanilla",
        "R14_seed42": "R14_hemato_seed42",
        "R14_seed123": "R14_hemato_seed123",
        "R14_seed456": "R14_hemato_seed456",
        "R0_seed123": "R0_vanilla_seed123",
        "R0_seed456": "R0_vanilla_seed456",
        "R0_seed42": "R0_seed42",
    }
    prov = {}
    for tissue in ("hemo", "neural"):
        for pk, a in (pub_pool.get(tissue) or {}).items():
            run = ALIAS.get(pk)
            spec = cache.get(run) if run else None
            if spec and "error" not in spec and TR in spec:
                b = float(spec[TR])
                prov[f"{tissue}:{pk}"] = dict(run=run, published_trH=float(a),
                                              stagedata_trH=b,
                                              rel_diff=abs(b - a) / max(a, 1e-9),
                                              reproduced=True)
            else:
                prov[f"{tissue}:{pk}"] = dict(run=run, published_trH=float(a),
                                              stagedata_trH=None, rel_diff=None,
                                              reproduced=False)
    prov["_note"] = ("published hemo pool values cannot be regenerated from the "
                     "stagedata of the runs they name; the neural pool does reproduce")

    # ---------------- reclassify the 33 configs ----------------
    rows = [r for r in s1f["rows"] if r.get("status") == "ok"]
    schemes = {
        "cross_family_tr": lambda r: classify(r["kappa_tr"], CROSS_FAMILY_BAND),
        "cross_family_r2": lambda r: classify(r["kappa_r2"], CROSS_FAMILY_BAND_CORRECTED),
        "published_family_tr": lambda r: classify(
            r["kappa_tr"],
            {"hemo": tuple([pools["hemo_published__tr"]["band"]["lo2s"],
                            pools["hemo_published__tr"]["band"]["hi2s"]]),
             "neural": tuple([pools["neural_published__tr"]["band"]["lo2s"],
                              pools["neural_published__tr"]["band"]["hi2s"]])}.get(r["tissue"])),
        "stagedata_family_r2": lambda r: classify(
            r["kappa_r2"],
            {"hemo": tuple([pools["hemo_mixed__sum_r2"]["band"]["lo2s"],
                            pools["hemo_mixed__sum_r2"]["band"]["hi2s"]]),
             "neural": tuple([pools["neural_mixed__sum_r2"]["band"]["lo2s"],
                              pools["neural_mixed__sum_r2"]["band"]["hi2s"]])}.get(r["tissue"])),
    }
    counts = {}
    for name, fn in schemes.items():
        c = {}
        for r in rows:
            v = fn(r)
            c[v] = c.get(v, 0) + 1
        counts[name] = c

    out = dict(
        protocol=("band = mean +/- 2sd of ordered cross-readout ratios X_i/X_j inside a "
                  "no-prior pool; kappa_C(W) trace = tr(W'W); corrected = sum_g R2_g"),
        cross_family_band=list(CROSS_FAMILY_BAND),
        cross_family_band_corrected=list(CROSS_FAMILY_BAND_CORRECTED),
        locked_threshold=LOCKED,
        pools=pools, provenance_check=prov, counts=counts,
        rows=[dict(config=r["config"], tissue=r["tissue"], kappa_tr=r["kappa_tr"],
                   kappa_r2=r["kappa_r2"],
                   v_cross_tr=schemes["cross_family_tr"](r),
                   v_cross_r2=schemes["cross_family_r2"](r),
                   v_pubfam_tr=schemes["published_family_tr"](r),
                   v_sdfam_r2=schemes["stagedata_family_r2"](r)) for r in rows],
    )
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "s1h_null_bands.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)

    # ---------------- report ----------------
    print("=" * 78)
    print("A. pools (band = mean +/- 2sd of ordered ratios)")
    print("=" * 78)
    print(f"{'pool':<26}{'n':>4}{'pairs':>7}{'mean':>9}{'sd':>9}"
          f"{'lo2s':>9}{'hi2s':>9}   source")
    for k, v in pools.items():
        b = v["band"]
        if b is None:
            print(f"{k:<26}{v['n_src']:>4}{'-':>7}   (single member -> band undefined)")
            continue
        srcs = sorted(set(v["source"].values())) if v["source"] else ["mixed"]
        print(f"{k:<26}{b['n_members']:>4}{b['n_pairs']:>7}{b['mean']:>9.4f}"
              f"{b['sd']:>9.4f}{b['lo2s']:>9.4f}{b['hi2s']:>9.4f}   {','.join(srcs)}")

    print()
    print("=" * 78)
    print("B. provenance: published pool vs stagedata of the same-named runs")
    print("=" * 78)
    print(f"{'pool key':<18}{'resolved run':<24}{'published':>11}{'stagedata':>11}"
          f"{'rel diff':>10}")
    for k, v in prov.items():
        if k.startswith("_"):
            continue
        st = f"{v['stagedata_trH']:.2f}" if v["stagedata_trH"] is not None else "ABSENT"
        rd = f"{v['rel_diff'] * 100:.1f}%" if v["rel_diff"] is not None else "-"
        print(f"{k:<18}{str(v['run']):<24}{v['published_trH']:>11.2f}{st:>11}{rd:>10}")

    print()
    print("=" * 78)
    print("C. reclassification of the 33 configurations (counts by verdict)")
    print("=" * 78)
    for name, c in counts.items():
        print(f"  {name:<22} {c}")

    print()
    print("D. corrected statistic, spread of the 33 configs (from S1f)")
    ktr = np.array([r["kappa_tr"] for r in rows])
    kr2 = np.array([r["kappa_r2"] for r in rows])
    print(f"  kappa_tr : mean {ktr.mean():.3f}  sd {ktr.std(ddof=1):.3f}  "
          f"range {ktr.min():.3f}-{ktr.max():.3f}")
    print(f"  kappa_r2 : mean {kr2.mean():.3f}  sd {kr2.std(ddof=1):.3f}  "
          f"range {kr2.min():.3f}-{kr2.max():.3f}")
    print(f"  n below 0.5 (deep compression): tr {int((ktr < 0.5).sum())}/33  "
          f"r2 {int((kr2 < 0.5).sum())}/33")

    print()
    print("=" * 78)
    print("E. per-configuration verdict under the corrected family-matched band")
    print("=" * 78)
    hb = pools["hemo_mixed__sum_r2"]["band"]
    nb = pools["neural_mixed__sum_r2"]["band"]
    print(f"  hemo  band [{hb['lo2s']:.4f}, {hb['hi2s']:.4f}]   "
          f"neural band [{nb['lo2s']:.4f}, {nb['hi2s']:.4f}]   lung: no family-matched band")
    print(f"  {'config':<14}{'tissue':<8}{'k_tr':>9}{'k_r2':>9}   {'v_cross_tr':<18}"
          f"{'v_cross_r2':<18}{'v_sdfam_r2':<18}")
    for r in sorted(rows, key=lambda x: (x["tissue"], x["kappa_r2"])):
        print(f"  {r['config']:<14}{r['tissue']:<8}{r['kappa_tr']:>9.3f}"
              f"{r['kappa_r2']:>9.4f}   {schemes['cross_family_tr'](r):<18}"
              f"{schemes['cross_family_r2'](r):<18}"
              f"{schemes['stagedata_family_r2'](r):<18}")

    print(f"\nwrote {os.path.join(OUT, 's1h_null_bands.json')}")


if __name__ == "__main__":
    main()
