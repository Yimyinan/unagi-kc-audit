"""
run_registry.py
===============
Single source of truth mapping a logical run name to its loading matrix,
WITH a data-quality gate.

Why this exists
---------------
Two data-integrity failures were found while running the dose-response
analysis, and both silently corrupted results:

1. `decoder_out/R13/iter_4/loading_matrix.npy` (the seed-42 full-prior model)
   is DEGENERATE: Frobenius norm 7.57 versus 70-120 for every other matrix,
   and column-norm CV 1.96 versus ~0.6. The correct matrix is
   `decoder_unified/loading/R13_s42.npy` (norm 100.45, CV 0.62). A naive
   "try unified, else decoder_out" loader resolves the bare name `R13` to
   the degenerate file, because no unified file is literally called R13.npy.

2. `decoder_unified/loading/R15_s42.npy`, `R15_s123`, `R15_s456` are the SAME
   readout as the corresponding `R14_s*` files (identical norm, effective rank
   and column-norm CV) -- both naming schemes denote the **VEHICLE arm**
   (runs `result/R15/R14_hemato_seed{42,123,456}`: 37 TF prior database /
   1054 boosted genes, renamed out of the R14 family). They are therefore not
   an independent arm -- and they are emphatically NOT the empty prior.
   The hematopoietic empty prior is `R14_hemato_vanilla`
   (`result/R14/R14_hemato_vanilla`, readout tr(W'W) = 5634.8).

3. `decoder_out/R0` (bare name, neural vanilla) is a legacy run that is NOT the
   seed-42 vanilla: there is no `R0_vanilla_seed42` entry in the Z-spectrum
   table, and its effective rank (38.1) is an outlier against the seed-matched
   vanillas (24.0-27.0). Pairing the seed-42 neural prior `R1` against it
   produced a spurious z = +3.91; against the correct `R0_seed42` the same
   comparison gives z = +0.19.

4. CORRECTED 2026-09-16 -- the previous revision of this file had this
   identification INVERTED. It asserted that `decoder_out/R14/` (=
   `decoder_out/R14/vanilla/`, byte-identical) was a suspicious "different
   source" and that `R14_s42.npy` was "the seed-42 empty prior". The numbers it
   quoted were right (||W|| 75.07 vs 71.84) but the roles were swapped:

       decoder_out/R14/vanilla/iter_4    tr(W'W) = 5634.8  <-- EMPTY PRIOR
       decoder_unified/loading/R14_s42   tr(W'W) = 5160.9  <-- VEHICLE arm

   Proof: `decoder_out/R14/kappa_hematopoietic.json` records
   `config = R14_hemato_vanilla_vs_R15_hemato_seed42`, `delta_V = 478.2565`
   (exactly the VANILLA's lambda1-lambda2) and `E_norm = 1396.0445`; the
   recovered `||H_P - H_V||_2 = kappa * delta_V` equals 1396.0445 EXACTLY for
   both `R14_s42` and `R15_s42`, so those two are the PRIOR (P) arm. The
   published hemo tr_V is 5634.8 => the vanilla is `decoder_out/R14/vanilla`.
   The consequent mapping `"R14" -> "R14_s42"` pointed the empty-prior alias at
   the vehicle arm and is REMOVED; bare vanilla names now resolve through
   VANILLA_ALIAS to the canonical readouts in `decoder_unified/vanilla/`.
   (Same class of error affected the neural tissue: the published neural
   tr_V is 5402.1 = `decoder_out/R0_seed42`, not `R0_seed123` (6792.0).)

The quality gate below refuses to return a degenerate matrix instead of
silently handing back numbers.

Usage
-----
    from run_registry import get_matrix, get_gene_index, load_pair, get_vanilla
    W   = get_matrix("R13")           # -> unified R13_s42, quality-checked
    Wv  = get_vanilla("hemo")         # -> the EMPTY PRIOR (tr 5634.8)
    Am, Av, idx = load_pair("R13", "R14")
"""
import os
import numpy as np
import json

UNI = "F:/UNAGI_Project/UNAGI/decoder_unified/loading"
DEC = "F:/UNAGI_Project/UNAGI/decoder_out"

# Canonical vanilla readouts. Deliberately a SEPARATE directory: iterating
# `decoder_unified/loading/*.npy` is documented elsewhere as "30 readouts"
# (decoder_subspace_test.py:24), so the vanillas must not be added there.
VAN = "F:/UNAGI_Project/UNAGI/decoder_unified/vanilla"

# tissue -> canonical vanilla stem under VAN.
# The tr(W'W) values are the published anchors and are frozen in VAN/MANIFEST.json.
#   hemo 5634.8 / neural 5402.1 / lung 12568.7
VANILLA_STEM = {
    "hemo":   "R14_hemato_vanilla",
    "neural": "R0_neural_vanilla",
    "lung":   "R10_lung_vanilla",
}

# Logical vanilla aliases -> canonical stem under VAN.
# NOTE: "R14" means the EMPTY PRIOR. The vehicle arm (37-TF prior) is
# "R14_vehicle_seed*" / "R15_seed*" and is NOT a vanilla.
VANILLA_ALIAS = {
    "R14":                "R14_hemato_vanilla",
    "R14/vanilla":        "R14_hemato_vanilla",
    "R14_vanilla":        "R14_hemato_vanilla",
    "R14_hemato_vanilla": "R14_hemato_vanilla",
    "R0_seed42":          "R0_neural_vanilla",
    "R0_vanilla":         "R0_neural_vanilla",
    "R0_neural_vanilla":  "R0_neural_vanilla",
    "R10":                "R10_lung_vanilla",
    "R10_vanilla":        "R10_lung_vanilla",
    "R10_lung_vanilla":   "R10_lung_vanilla",
}

# logical name -> unified stem (preferred). If None, only decoder_out is used.
NAME_MAP = {
    # ---- hematopoietic, seed 42 / 123 / 456
    "R13":              "R13_s42",
    "R13_seed42":       "R13_s42",
    "R13_seed123":      "R13_s123",
    "R13_seed456":      "R13_s456",
    "R11_seed42":       "R11_s42",
    "R11_KLF1_TAL1_seed123": "R11_s123",
    "R11_KLF1_TAL1_seed456": "R11_s456",
    "R12_seed42":       "R12_s42",
    "R12_seed123":      "R12_s123",
    "R12_seed456":      "R12_s456",
    "R84_seed42":       "R84_s42",
    "R84_seed123":      "R84_s123",
    "R107_seed42":      "R107_s42",
    "R111_seed42":      "R111_s42",
    "R111_seed123":     "R111_s123",
    "R165_seed42":      "R165_s42",
    # ---- VEHICLE arm (37-TF prior DB / 1054 boosted genes) -- NOT the empty
    # prior. Underlying runs: result/R15/R14_hemato_seed{42,123,456}.
    # NOTE: the empty prior is NOT in NAME_MAP; use get_vanilla("hemo") or the
    # VANILLA_ALIAS names ("R14", "R14/vanilla").
    "R14_vehicle_seed42":  "R14_s42",
    "R14_vehicle_seed123": "R14_s123",
    "R14_vehicle_seed456": "R14_s456",
    "R15_seed42":          "R14_s42",
    "R15_seed123":         "R14_s123",
    "R15_seed456":         "R14_s456",
    "R15_s42":             "R14_s42",
    "R15_s123":            "R14_s123",
    "R15_s456":            "R14_s456",
    # DEPRECATED aliases kept for backward compatibility ONLY. These names used
    # to be read as "the seed-matched empty prior"; they are the vehicle arm.
    # They will be removed once the callers named below are migrated:
    #   dose_response_wiring.py, layer2_pooled_mwu.py, seed_dependence_explain.py,
    #   kappa_v2_partitioned.py, kappa_v2_seed_sweep.py, kappa_v2_vanilla_null.py,
    #   recalculate_kappa_band_unagi.py, wiring_cross_seed.py,
    #   prior_wiring_test.py, prior_specificity_controls.py
    "R14_seed42":          "R14_s42",
    "R14_seed123":         "R14_s123",
    "R14_seed456":         "R14_s456",
    "R13_KO_KLF1":      "R13_KO_KLF1",
    "R13_OE_KLF1":      "R13_OE_KLF1",
    # wrong-prior / mismatch: not in unified, decoder_out copy is healthy
    "R16":              None,
    # ---- neural. NOTE: bare R0 is a legacy run, deliberately NOT mapped.
    "R1":               None,
    "R1_seed123":       "R1_s123",
    "R1_seed456":       "R1_s456",
    "R0_seed42":        None,
    "R0_seed123":       "R0_s123",
    "R0_seed456":       "R0_s456",
}

# Known-bad data. Documented so the exclusion is auditable, not silent.
# (The R15_* names used to be listed here as "empty-prior duplicate". They are
#  now NAME_MAP aliases for the VEHICLE readout -- see docstring item 2.)
EXCLUDED = {
    "R0":  "legacy run, NOT the seed-42 neural vanilla: eff rank 38.1 vs "
           "24.0-27.0 for the seed-matched vanillas, and tr(W'W) = 17695.2 vs "
           "the published neural tr_V = 5402.1. Use 'R0_seed42' / 'R0_vanilla' "
           "(= decoder_out/R0_seed42) instead. Pairing R1 against it produced a "
           "spurious z = +3.91 (vs +0.19 against the correct baseline).",
}

# Quality thresholds calibrated on the 30 healthy unified matrices:
#   Frobenius norm 63-180, column-norm CV 0.49-0.65.
MIN_FRO = 30.0
MAX_COLNORM_CV = 0.90


class DataQualityError(Exception):
    pass


def _quality(W):
    s = np.linalg.svd(W, compute_uv=False)
    cn = np.linalg.norm(W, axis=0)
    return dict(fro=float(np.linalg.norm(W)),
                eff_rank=float((s.sum() ** 2) / (s ** 2).sum()),
                colnorm_cv=float(cn.std() / cn.mean()))


def _load_checked(path, label, strict):
    W = np.load(path).astype(float)
    q = _quality(W)
    if strict and (q["fro"] < MIN_FRO or q["colnorm_cv"] > MAX_COLNORM_CV):
        raise DataQualityError(
            f"{label}: degenerate matrix at {path} "
            f"(||W||={q['fro']:.2f}, colnorm CV={q['colnorm_cv']:.2f})")
    return W


def get_vanilla(tissue, strict=True):
    """Return the (64, G) canonical VANILLA (no-prior) readout for a tissue.

    Prefer this over any hand-written run name. The three published tr_V
    anchors are frozen in decoder_unified/vanilla/MANIFEST.json:
        hemo 5634.8 | neural 5402.1 | lung 12568.7
    """
    if tissue not in VANILLA_STEM:
        raise KeyError(f"unknown tissue {tissue!r}; expected one of {list(VANILLA_STEM)}")
    stem = VANILLA_STEM[tissue]
    path = os.path.join(VAN, stem + ".npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"vanilla readout missing for {tissue}: {path}")
    return _load_checked(path, f"vanilla[{tissue}]={stem}", strict)


def get_matrix(name, strict=True):
    """Return the (64, G) loading matrix for a logical run name."""
    if name in EXCLUDED:
        raise DataQualityError(f"{name} is excluded: {EXCLUDED[name]}")
    if name in VANILLA_ALIAS:
        path = os.path.join(VAN, VANILLA_ALIAS[name] + ".npy")
        if not os.path.exists(path):
            raise FileNotFoundError(f"vanilla readout missing: {path}")
        return _load_checked(path, f"{name} -> {VANILLA_ALIAS[name]}", strict)
    stem = NAME_MAP.get(name, name)
    cands = []
    if stem:
        cands.append(os.path.join(UNI, stem + ".npy"))
    cands.append(os.path.join(DEC, name, "iter_4", "loading_matrix.npy"))
    cands.append(os.path.join(DEC, stem or name, "iter_4", "loading_matrix.npy"))
    path = next((p for p in cands if p and os.path.exists(p)), None)
    if path is None:
        raise FileNotFoundError(f"no loading matrix for {name}")
    return _load_checked(path, name, strict)


def get_gene_index(G):
    name = "hema_gene_list.json" if G > 15000 else "neural_gene_list.json"
    gl = [str(x).upper() for x in json.load(open(os.path.join(DEC, name)))]
    assert len(gl) == G, f"gene list {len(gl)} != matrix {G}"
    return {g: i for i, g in enumerate(gl)}


def centred_unit(W):
    """Cross-axis centring then L2 normalisation of each gene column.

    The centring step is essential: without it, coupling and enrichment are
    dominated by the component shared by all latent axes.
    """
    A = W - W.mean(axis=0, keepdims=True)
    n = np.linalg.norm(A, axis=0, keepdims=True)
    n[n == 0] = 1.0
    return A / n


def load_pair(model, vanilla):
    """Return (A_model, A_vanilla, gene_index) with matched shapes."""
    Wm = get_matrix(model); Wv = get_matrix(vanilla)
    if Wm.shape != Wv.shape:
        raise ValueError(f"shape mismatch {model}{Wm.shape} vs {vanilla}{Wv.shape}")
    idx = get_gene_index(Wm.shape[1])
    return centred_unit(Wm), centred_unit(Wv), idx


def audit():
    """Print a quality table for every logical run; returns the bad ones."""
    bad = []
    print(f"{'name':24s} {'stem':14s} {'||W||':>8s} {'eff_rank':>9s} {'CV':>6s}  status")
    print("-" * 74)
    for name in sorted(NAME_MAP):
        if name in EXCLUDED:
            print(f"{name:24s} {'-':14s} {'-':>8s} {'-':>9s} {'-':>6s}  EXCLUDED")
            continue
        try:
            W = get_matrix(name, strict=False)
            q = _quality(W)
            ok = not (q["fro"] < MIN_FRO or q["colnorm_cv"] > MAX_COLNORM_CV)
            if not ok:
                bad.append(name)
            print(f"{name:24s} {str(NAME_MAP[name]):14s} {q['fro']:8.2f} "
                  f"{q['eff_rank']:9.2f} {q['colnorm_cv']:6.3f}  "
                  f"{'OK' if ok else 'DEGENERATE'}")
        except FileNotFoundError as e:
            bad.append(name)
            print(f"{name:24s} {str(NAME_MAP[name]):14s} {'-':>8s} {'-':>9s} "
                  f"{'-':>6s}  MISSING")
    return bad


if __name__ == "__main__":
    b = audit()
    print(f"\n{len(b)} problem runs: {b}")
