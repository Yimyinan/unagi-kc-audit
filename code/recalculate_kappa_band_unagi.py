"""
recalculate_kappa_band_unagi.py
================================
Full κ-C(W) recalculation using UNAGI's OWN vanilla seeds (family-matched),
to replace the G-VAE-calibrated band [0.78, 1.23] that was previously applied
cross-family to the UNAGI audit.

Definitions (verbatim from manuscript Methods):
    kappa = tr(H_P) / tr(H_V)            # H = W^T W, W = decoder readout (D x G)
    C     = (1 - kappa) * 100 %
    deep compression: C > 50%  <=>  kappa < 0.5   (absolute, band-independent)

Two distinct things are recomputed:
  (A) The vanilla null BAND (the ±2σ "vanilla-equivalence region"), rebuilt from
      UNAGI vanilla cross-seed κ ratios. This is what the G-VAE [0.78,1.23] was.
  (B) The 33-config audit κ values, recomputed from raw .npy matrices via
      run_registry, to verify end-to-end reproducibility and re-count deep
      compression under the new (and old) band.

UNAGI vanilla seeds available on F: (per run_registry NAME_MAP)
    hematopoietic: R14  = decoder_unified/vanilla/R14_hemato_vanilla
                   (the TRUE VANILLA; R14_s42/R14_s123/R14_s456 are the
                    37-TF VEHICLE arm, NOT empty priors)
    neural crest : R0_s123,  R0_s456             (2 seeds; R0_s42 legacy/excluded)
    lung         : R10                                 (1 seed -> no cross-seed null)

So a UNAGI cross-seed null is feasible for hema (3 seeds) + neural (2 seeds),
and lung is single-seed (reported separately, cannot enter the null).

Writes: ../data/unaug_vanilla_null_recalc.json
"""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_registry as RR

OUT = os.path.join(HERE, "..", "data", "unaug_vanilla_null_recalc.json")

# ---- UNAGI vanilla seeds (per tissue) ----
VANILLA = {
    "hemo":   ["R14_s42", "R14_s123", "R14_s456"],
    "neural": ["R0_s123", "R0_s456"],
    "lung":   ["R10"],
}
# matched tissue denominator used by the published audit (tr_V column)
TISSUE_DEN = {"hemo": "R14_s42", "neural": "R0_s123", "lung": "R10"}


def trH(name):
    W = RR.get_matrix(name, strict=False)
    return float((W ** 2).sum())


def main():
    out = {"protocol": "trH = ||W||_F^2 of decoder_unified/loading .npy via run_registry; "
                       "kappa = trH_i/trH_j (vanilla cross-seed) or trH_prior/trH_vanilla (audit)",
           "vanilla_trH": {}, "vanilla_null": {}, "audit_repro": {}}

    # ---------- (A) vanilla trH per seed ----------
    for tissue, seeds in VANILLA.items():
        for s in seeds:
            try:
                out["vanilla_trH"].setdefault(tissue, {})[s] = round(trH(s), 3)
            except Exception as e:
                out["vanilla_trH"].setdefault(tissue, {})[s] = f"ERR {e}"

    # ---------- (A) UNAGI vanilla cross-seed null band ----------
    all_pairs = []
    per_tissue_pairs = {}
    for tissue, seeds in VANILLA.items():
        tr = [trH(s) for s in seeds]
        pairs = []
        # ordered pairs i != j (same construction as G-VAE 210 = 15*14)
        for i in range(len(seeds)):
            for j in range(len(seeds)):
                if i == j:
                    continue
                k = tr[i] / tr[j]
                pairs.append(round(k, 4))
                all_pairs.append(k)
        per_tissue_pairs[tissue] = pairs
        if len(pairs) >= 2:
            arr = np.array(pairs)
            out["vanilla_null"][tissue] = dict(
                n_seeds=len(seeds), n_pairs=len(pairs),
                mean=round(float(arr.mean()), 4), std=round(float(arr.std()), 4),
                lo2s=round(float(arr.mean() - 2 * arr.std()), 4),
                hi2s=round(float(arr.mean() + 2 * arr.std()), 4),
                p05=round(float(np.percentile(arr, 5)), 4),
                p95=round(float(np.percentile(arr, 95)), 4),
                min=round(float(arr.min()), 4), max=round(float(arr.max()), 4))
        else:
            out["vanilla_null"][tissue] = dict(
                n_seeds=len(seeds), n_pairs=len(pairs),
                note="single seed -> no cross-seed null possible")
    # combined hema+neural null (multi-seed tissues only)
    arr = np.array(all_pairs)
    out["vanilla_null"]["UNAGI_combined_hema_neural"] = dict(
        n_pairs=len(all_pairs),
        mean=round(float(arr.mean()), 4), std=round(float(arr.std()), 4),
        lo2s=round(float(arr.mean() - 2 * arr.std()), 4),
        hi2s=round(float(arr.mean() + 2 * arr.std()), 4),
        p05=round(float(np.percentile(arr, 5)), 4),
        p95=round(float(np.percentile(arr, 95)), 4),
        min=round(float(arr.min()), 4), max=round(float(arr.max()), 4))

    # ---------- (B) re-run the 33-config audit from raw matrices ----------
    audit = json.load(open(os.path.join(HERE, "..", "data", "kappa_complete_audit.json")))
    repro = []
    n_deep_old = n_deep_new = 0
    deep_thr = 0.5
    old_lo, old_hi = 0.78, 1.23
    new_lo = out["vanilla_null"]["UNAGI_combined_hema_neural"]["lo2s"]
    new_hi = out["vanilla_null"]["UNAGI_combined_hema_neural"]["hi2s"]
    for row in audit:
        cfg = row["config"]; tissue = row["tissue"]
        try:
            tr_P = trH(cfg)
            tr_V = trH(TISSUE_DEN[tissue])
        except Exception as e:
            repro.append(dict(config=cfg, tissue=tissue, err=str(e)))
            continue
        kappa = tr_P / tr_V
        C = (1 - kappa) * 100.0
        deep = kappa < deep_thr
        in_old_band = old_lo <= kappa <= old_hi
        in_new_band = new_lo <= kappa <= new_hi
        if deep:
            n_deep_old += 1; n_deep_new += 1
        repro.append(dict(
            config=cfg, tissue=tissue,
            kappa_repro=round(kappa, 4), C_repro=round(C, 2),
            kappa_audit=row["kappa_ratio"], C_audit=row["C"],
            deep=deep, in_old_band=in_old_band, in_new_band=in_new_band))
    out["audit_repro"] = dict(
        n=len([r for r in repro if "kappa_repro" in r]),
        deep_compression_kappa_lt_0p5=n_deep_old,   # absolute, band-independent
        n_in_old_band_GVAE=sum(1 for r in repro if r.get("in_old_band")),
        n_in_new_band_UNAGI=sum(1 for r in repro if r.get("in_new_band")),
        rows=repro)

    # ---------- comparison vs G-VAE band ----------
    out["comparison"] = dict(
        GVAE_band=[0.78, 1.23],
        UNAGI_combined_band=[new_lo, new_hi],
        note=("Deep-compression count (kappa<0.5) is band-independent and must be "
              "identical under both bands. The vanilla-equivalence region shifts to "
              "the UNAGI-calibrated values."))

    json.dump(out, open(OUT, "w"), indent=2)
    # ---- console summary ----
    print("== vanilla trH ==")
    for t, d in out["vanilla_trH"].items():
        print(f"  {t:7s}: {d}")
    print("== UNAGI vanilla null bands (kappa = trH_i/trH_j) ==")
    for t, d in out["vanilla_null"].items():
        print(f"  {t:32s}: {d}")
    print("== audit repro ==")
    print(f"  n={out['audit_repro']['n']}  deep(kappa<0.5)={n_deep_old}/33  "
          f"in_old_GVAE_band={out['audit_repro']['n_in_old_band_GVAE']}  "
          f"in_new_UNAGI_band={out['audit_repro']['n_in_new_band_UNAGI']}")
    # show max deviation from audit kappa for verification
    maxdev = max(abs(r["kappa_repro"] - r["kappa_audit"]) for r in repro if "kappa_repro" in r)
    print(f"  max |kappa_repro - kappa_audit| = {maxdev:.4f}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
