# -*- coding: utf-8 -*-
"""Recompute the UNAGI family-matched null under the corrected Vanilla identity.

User decision 2026-09-16 ("做").

THE BUG
-------
The manuscript's family-matched null pools the single TRUE hematopoietic vanilla
together with the three VEHICLE seeds:

    vanilla_trH[hemo] = {R14: 5634.82, R14_seed42: 4333.06,
                         R14_seed123: 4156.59, R14_seed456: 5272.91}

R14_seed42/123/456 are the empty-vector VEHICLE arm (37-TF prior DB / 1054 boosted
genes), NOT further empty priors.  Pooling them with the vanilla mixes two variance
sources -- seed noise AND the vanilla-vs-vehicle offset -- into one null.

(The neural pool {R0_seed42, R0_seed123, R0_seed456} is legitimate: those ARE three
genuine vanilla readouts.)

An earlier revision had this right: data/vanilla_null_v2.json notes
"the UNAGI audit family has only one vanilla seed per tissue ... so a
UNAGI-family null is infeasible. R15_hemato_seed42/123/456 are PRIOR models".

WHAT THIS SCRIPT DOES
---------------------
Recomputes every candidate pool from the SAME numbers already on disk, then
re-scores the 33 audit configurations against each resulting band.

protocol (as in the original): trH = ||W||_F^2 ; kappa_ij = trH_i / trH_j over
ORDERED pairs i != j ; band = mean +/- 2 sd.
"""
import json, os
import numpy as np

os.chdir(r"F:/UNAGI_Project/paper1/paper1_kc_diagnostic")
OUT = r"F:/UNAGI_Project/outputs/positive_result"

FINAL = json.load(open("data/unaug_vanilla_null_final.json", encoding="utf-8-sig"))
RECALC = json.load(open("data/unaug_vanilla_null_recalc.json", encoding="utf-8-sig"))

TRH = FINAL["vanilla_trH"]           # decoder_out-generation readouts
POOL_MIXED = {t: v for t, v in TRH.items()}

VANILLA = {                          # the genuine no-prior readouts
    "hemo":   {"R14": POOL_MIXED["hemo"]["R14"]},
    "neural": dict(POOL_MIXED["neural"]),
}
VEHICLE = {                          # the 37-TF prior arm (NOT vanilla)
    "hemo": {k: v for k, v in POOL_MIXED["hemo"].items() if k != "R14"},
}


def null_of(pool):
    """kappa_ij = trH_i / trH_j over ordered i != j."""
    vals = np.array(list(pool.values()), float)
    n = len(vals)
    if n < 2:
        return None
    ratios = np.array([vals[i] / vals[j] for i in range(n) for j in range(n) if i != j])
    m, s = float(ratios.mean()), float(ratios.std(ddof=0))   # population sd (matches the stored nulls)
    return dict(n_seeds=n, n_pairs=len(ratios), mean=round(m, 4), std=round(s, 4),
                lo2s=round(m - 2 * s, 4), hi2s=round(m + 2 * s, 4),
                min=round(float(ratios.min()), 4), max=round(float(ratios.max()), 4))


CURRENT = {t: null_of(POOL_MIXED[t]) for t in POOL_MIXED}
CORRECT = {"hemo_vanilla_only": null_of(VANILLA["hemo"]),
           "hemo_vehicle_only": null_of(VEHICLE["hemo"]),
           "neural_vanilla": null_of(VANILLA["neural"])}

print("=" * 96)
print("POOLS (trH values, as on disk)")
print("=" * 96)
for t, p in POOL_MIXED.items():
    print(f"  current pool [{t}] = {p}")
print(f"  -> genuine VANILLA [hemo]   = {VANILLA['hemo']}   (1 readout)")
print(f"  -> VEHICLE arm      [hemo]  = {VEHICLE['hemo']}   ({len(VEHICLE['hemo'])} readouts)")
print(f"  -> genuine VANILLA [neural] = {VANILLA['neural']}")

print()
print("=" * 96)
print("NULLS")
print("=" * 96)
hdr = f"{'pool':<34}{'seeds':>6}{'pairs':>7}{'mean':>9}{'sd':>9}{'band (mean+/-2sd)':>26}"
print(hdr)
for label, d in [("CURRENT hemo (vanilla + vehicle)", CURRENT["hemo"]),
                 ("CURRENT neural", CURRENT["neural"]),
                 ("CORRECTED hemo, VANILLA only", CORRECT["hemo_vanilla_only"]),
                 ("CORRECTED hemo, VEHICLE only", CORRECT["hemo_vehicle_only"]),
                 ("CORRECTED neural (unchanged)", CORRECT["neural_vanilla"])]:
    if d is None:
        print(f"{label:<34}{'1':>6}{'-':>7}   NOT IDENTIFIABLE (a single readout cannot form a pair)")
    else:
        print(f"{label:<34}{d['n_seeds']:>6}{d['n_pairs']:>7}{d['mean']:>9.4f}{d['std']:>9.4f}"
              f"{'['+format(d['lo2s'],'.4f')+', '+format(d['hi2s'],'.4f')+']':>26}")

# ---- combined pools ----
# The original "combined" null is the CONCATENATION of the per-tissue ordered-pair
# sets (hemo 12 + neural 6 = 18 pairs), NOT the ordered pairs of a merged pool
# (which would cross hemo against neural).  Preserve that stratification.
def ordered_ratios(pool):
    vals = np.array(list(pool.values()), float)
    n = len(vals)
    return np.array([vals[i] / vals[j] for i in range(n) for j in range(n) if i != j])


def band_from(ratios):
    m, s = float(ratios.mean()), float(ratios.std(ddof=0))
    return dict(n_seeds=None, n_pairs=len(ratios), mean=round(m, 4), round_sd=round(s, 4),
                std=round(s, 4), lo2s=round(m - 2 * s, 4), hi2s=round(m + 2 * s, 4),
                min=round(float(ratios.min()), 4), max=round(float(ratios.max()), 4))


comb_now = band_from(np.concatenate([ordered_ratios(POOL_MIXED["hemo"]),
                                     ordered_ratios(POOL_MIXED["neural"])]))
comb_corr = band_from(np.concatenate([ordered_ratios(VEHICLE["hemo"]),
                                      ordered_ratios(VANILLA["neural"])]))
comb_neural_only = null_of(VANILLA["neural"])
for label, d in [("combined CURRENT (4 hemo + 3 neural)", comb_now),
                 ("combined CORRECTED (3 veh + 3 neural)", comb_corr)]:
    print(f"{label:<40}{'7' if d is comb_now else '6':>6}{d['n_pairs']:>7}"
          f"{d['mean']:>9.4f}{d['std']:>9.4f}"
          f"{'['+format(d['lo2s'],'.4f')+', '+format(d['hi2s'],'.4f')+']':>26}"
          f"   (concat of per-tissue ordered-pair sets)")

# ---- re-score the 33 audit configs against each band ----
rows_all = RECALC["audit_repro"]["rows"]
rows = [r for r in rows_all if "kappa_repro" in r]
print(f"[info] audit rows usable: {len(rows)} of {len(rows_all)} "
      f"(missing kappa_repro: {[r.get('config') for r in rows_all if 'kappa_repro' not in r]})")
VEH_TRH = {42: 5160.92, 123: 9109.905, 456: 14272.942}   # unified R14_s* (audit denominator)
TRUE_VAN_HEMO = POOL_MIXED["hemo"]["R14"]                # 5634.82
NEU_VAN = POOL_MIXED["neural"]                           # seed-matched


def resolv(row):
    """Return (kappa_vehicle_denom, kappa_true_vanilla_denom)."""
    k = row["kappa_repro"]
    if row["tissue"] == "hemo":
        s = 42 if row["config"].endswith("s42") else (123 if row["config"].endswith("s123") else 456)
        den = VEH_TRH.get(s, VEH_TRH[42])
        trp = k * den
        return k, trp / TRUE_VAN_HEMO
    den = NEU_VAN.get("R0_seed42", 5402.1)
    return k, k          # neural already uses a vanilla denominator


print()
print("=" * 96)
print("AUDIT RE-SCORE (33 configurations)")
print("=" * 96)
bands = {
    "G-VAE [0.78,1.23] (paper)": (0.78, 1.23),
    "CURRENT UNAGI combined": (comb_now["lo2s"], comb_now["hi2s"]),
    "CORRECTED UNAGI combined": (comb_corr["lo2s"], comb_corr["hi2s"]),
}
print(f"{'band':<30}{'denominator':<22}{'n outside':>10}{'n kappa<0.5':>13}")
for bname, (lo, hi) in bands.items():
    for dname, idx in [("vehicle (as recalc)", 0), ("TRUE vanilla (spec B)", 1)]:
        ks = [resolv(r)[idx] for r in rows]
        out = sum(1 for k in ks if not (lo <= k <= hi))
        deep = sum(1 for k in ks if k < 0.5)
        print(f"{bname:<30}{dname:<22}{out:>10}{deep:>13}")

# per-config detail with the correct denominator
ks_true = [resolv(r)[1] for r in rows]
print()
print(f"kappa with the TRUE hema vanilla: min={min(ks_true):.3f}  max={max(ks_true):.3f}  "
      f"n<0.5={sum(1 for k in ks_true if k < 0.5)}  n<1={sum(1 for k in ks_true if k < 1)}")
worst = sorted(zip(rows, ks_true), key=lambda x: x[1])[:8]
print("  lowest 8:", [(r["config"], round(k, 3)) for r, k in worst])

json.dump(dict(current=CURRENT, corrected=CORRECT,
               combined_current=comb_now, combined_corrected=comb_corr,
               combined_neural_only=comb_neural_only,
               pools=dict(mixed=POOL_MIXED, vanilla=VANILLA, vehicle=VEHICLE),
               rescore={b: dict(
                   vehicle_denom=dict(outside=sum(1 for k in [resolv(r)[0] for r in rows]
                                                  if not (lo <= k <= hi)),
                                      deep=sum(1 for k in [resolv(r)[0] for r in rows] if k < 0.5)),
                   true_vanilla_denom=dict(outside=sum(1 for k in ks_true if not (lo <= k <= hi)),
                                           deep=sum(1 for k in ks_true if k < 0.5)))
                        for b, (lo, hi) in bands.items()},
               kappa_true_vanilla=[round(k, 4) for k in ks_true]),
          open(os.path.join(OUT, "family_matched_null_recomputed.json"), "w"), indent=1)
print(f"\nwrote {OUT}/family_matched_null_recomputed.json")
