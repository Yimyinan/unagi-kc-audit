"""
recalc_unagi_band_final.py
===========================
Faithful UNAGI-family recalibration of the kappa-C(W) vanilla null band,
using the EXACT W definition of the published audit
(decoder_out/<cfg>/iter_4/loading_matrix.npy; kappa = tr(W^T W)_prior /
tr(W^T W)_vanilla = trH_i / trH_j).

The G-VAE band [0.78, 1.23] was built from 15 G-VAE vanilla seeds as
vanilla_vs_vanilla kappa ratios (mean +/- 2 sigma). Here we rebuild the
same quantity from UNAGI's OWN vanilla seeds, which ARE available in
decoder_out as seed variants:
    hemo vanilla seeds  : R14, R14_seed42, R14_seed123, R14_seed456
    neural vanilla seeds: R0,  R0_seed42,  R0_seed123,  R0_seed456
(lung has a single vanilla R10 -> no multi-seed null; kept out of band)

Outputs: ../data/unaug_vanilla_null_final.json  (+ prints summary)
"""
import os, glob, json
import numpy as np

DEC = r"F:/UNAGI_Project/UNAGI/decoder_out"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "data", "unaug_vanilla_null_final.json")
AUDIT = os.path.join(HERE, "..", "data", "kappa_complete_audit.json")
ITER = "iter_4"


def trh(cfg):
    p = os.path.join(DEC, cfg, ITER, "loading_matrix.npy")
    if not os.path.exists(p):
        return None
    W = np.load(p).astype(float)
    return float((W ** 2).sum())


# ---- UNAGI vanilla seed sets (independent training draws of the SAME
#      vanilla model). NOTE: the audit's vanilla denominators are
#      hemo=R14 (trH=5634.8), neural=R0_seed42 (trH=5402.1), lung=R10.
#      Bare R0 (trH=17695) is a 3x outlier (same shape 64x11932) and is a
#      DIFFERENT model, so it is EXCLUDED from the neural null. ----
VANILLA = {
    "hemo":   ["R14", "R14_seed42", "R14_seed123", "R14_seed456"],
    "neural": ["R0_seed42", "R0_seed123", "R0_seed456"],
}

# canonical vanilla denominator used by the audit (for cross-check)
CANON = {"hemo": "R14", "neural": "R0", "lung": "R10"}


def band_from_trH(trhs):
    trhs = [t for t in trhs if t]
    ratios = []
    for i in range(len(trhs)):
        for j in range(len(trhs)):
            if i == j:
                continue
            ratios.append(trhs[i] / trhs[j])
    arr = np.array(ratios)
    return dict(n_seeds=len(trhs), n_pairs=len(ratios),
                mean=round(float(arr.mean()), 4),
                std=round(float(arr.std()), 4),
                lo2s=round(float(arr.mean() - 2 * arr.std()), 4),
                hi2s=round(float(arr.mean() + 2 * arr.std()), 4),
                p05=round(float(np.percentile(arr, 5)), 4),
                p95=round(float(np.percentile(arr, 95)), 4),
                min=round(float(arr.min()), 4),
                max=round(float(arr.max()), 4))


def main():
    out = {"protocol": "kappa = trH(loading_matrix.npy)_i / trH_j; "
                        "band = mean +/- 2*sigma of vanilla-vs-vanilla ratios",
           "vanilla_trH": {}, "null": {}, "audit_recheck": {}}

    all_ratios = []
    for tissue, seeds in VANILLA.items():
        trhs = {}
        for s in seeds:
            t = trh(s)
            trhs[s] = round(t, 2) if t is not None else "MISSING"
            print(f"  {s:14s} trH={trhs[s]}")
        out["vanilla_trH"][tissue] = trhs
        valid = [trhs[s] for s in seeds if isinstance(trhs[s], float)]
        b = band_from_trH(valid)
        out["null"][tissue] = b
        # collect all ratios for combined band
        for i in range(len(valid)):
            for j in range(len(valid)):
                if i != j:
                    all_ratios.append(valid[i] / valid[j])

    arr = np.array(all_ratios)
    out["null"]["UNAGI_combined"] = dict(
        n_pairs=len(all_ratios),
        mean=round(float(arr.mean()), 4),
        std=round(float(arr.std()), 4),
        lo2s=round(float(arr.mean() - 2 * arr.std()), 4),
        hi2s=round(float(arr.mean() + 2 * arr.std()), 4),
        p05=round(float(np.percentile(arr, 5)), 4),
        p95=round(float(np.percentile(arr, 95)), 4))

    # ---- recheck the 33 audit configs against the NEW band ----
    audit = json.load(open(AUDIT))
    new_lo = out["null"]["UNAGI_combined"]["lo2s"]
    new_hi = out["null"]["UNAGI_combined"]["hi2s"]
    outside = 0
    deep = 0
    outside_list = []
    for c in audit:
        k = c["kappa_ratio"]
        if k < new_lo or k > new_hi:
            outside += 1
            outside_list.append(c["config"])
        if k < 0.5:           # deep compression C>50%
            deep += 1
    out["audit_recheck"] = dict(
        n=len(audit), n_outside_new_band=outside,
        outside_configs=outside_list,
        n_deep_compression_k_lt_0p5=deep,
        new_band=[new_lo, new_hi],
        GVAE_band=[0.78, 1.23],
        note="deep compression (C>50%) is an absolute kappa<0.5 threshold, "
             "independent of the band; band only changes the 'vanilla-"
             "equivalent' labelling.",
        bare_R0_excluded_as_outlier=True)

    out["comparison"] = dict(
        GVAE_band=[0.78, 1.23],
        UNAGI_hemo_band=[out["null"]["hemo"]["lo2s"], out["null"]["hemo"]["hi2s"]],
        UNAGI_neural_band=[out["null"]["neural"]["lo2s"], out["null"]["neural"]["hi2s"]],
        UNAGI_combined_band=[new_lo, new_hi])

    json.dump(out, open(OUT, "w"), indent=2)
    print("\n== UNAGI vanilla null bands (kappa = trH_i/trH_j) ==")
    for t, d in out["null"].items():
        print(f"  {t:16s}: lo2s={d['lo2s']}  hi2s={d['hi2s']}  "
              f"(mean={d.get('mean')}, std={d.get('std')}, n_pairs={d.get('n_pairs')})")
    print(f"\n  GVAE band:             [0.78, 1.23]")
    print(f"  audit recheck:         {outside}/33 outside new band; "
          f"{deep}/33 deep compression (k<0.5)")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
