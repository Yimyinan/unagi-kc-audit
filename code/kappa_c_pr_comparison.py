#!/usr/bin/env python3
"""
kappa_c_pr_comparison.py -- like-for-like decomposition of the kappa-C dual
spectrum, recomputed FROM RAW STAGEDATA with the published protocol.

WHY THIS REWRITE
----------------
Earlier versions read `decoder_out/<cfg>/iter_4/loading_matrix.npy`.  That is
NOT the estimator the published 0/33 audit uses.  Provenance check
(`check_provenance.py`) establishes that `kappa_complete_audit.json` was built
from `decoder_unified/loading/<cfg>.npy`, and we verified numerically that
    tr(W'W) of decoder_unified/<cfg>.npy  ==  audit tr_P
for 22/22 overlapping configs (rel err <= 7e-6, i.e. pure 1-decimal rounding in
the audit JSON).  Those matrices are (64, G) Ridge loadings, not decoder
network weights -- shape (64, 20466) and coefficient magnitudes ~1.8.

Consequence: the audit readout can be recomputed from raw data.  We therefore
recompute it for ALL audited configurations, including R2-R7 whose weights were
never exported, using the exact published protocol
(`recalc_unagi_vanilla_null.py`):

    A = sum_s Z_s' Z_s ;  B = sum_s Z_s' X_s ;  A += alpha * I
    W = solve(A, B)                      # (64, G), alpha = 1.0, no intercept
    kappa-C(W) = tr(W'W)_prior / tr(W'W)_vanilla

Both sides are then summarised with the SAME functional (participation ratio),
the manuscript's effective-rank convention:

    pr = (sum lambda)^2 / sum lambda^2 ,   lambda = eig(W W')

so the decoder/latent comparison is like-for-like rather than trace-vs-rank.

QUESTIONS ANSWERED
------------------
Q1  Is the trace trade-off (kappa-C(W) * kappa-C(Z) == 1) conserved?
Q2  Does the dissociation survive when BOTH sides use participation ratio?
Q3  Does it reproduce across tissues, or is it hematopoietic-only?

OUTPUT
------
    ../data/kappa_c_pr_comparison.json
    ../data/_ridge_recompute_cache.json   (expensive Ridge recompute cache)
"""
from __future__ import annotations
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from provenance import attach as _attach_provenance

import argparse
import csv
import glob
import json
import os

import h5py
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
DATA = os.path.join(HERE, "..", "data")
CACHE = os.path.join(DATA, "_ridge_recompute_cache.json")

BASE = "F:/UNAGI_Project/UNAGI"
RESULT = os.path.join(BASE, "result")
AUDIT_JSON = os.path.join(DATA, "kappa_complete_audit.json")
CSV_Z = os.path.join(ROOT, "outputs", "positive_result", "z_spectrum_all.csv")

ALPHA = 1.0
LAT = "obsm/z"
ITER = "iter_4"

# audit config -> stagedata run directory name.  Run dirs carry timestamps /
# long descriptive names; audit configs use short names.  This is the mapping
# that was missing in earlier passes.
RUN_NAME = {
    "R107_s42":     "R107_erythroid_lymphoid_seed42",
    "R111_s123":    "R111_erythroid_temporal_seed123-1",
    "R111_s42":     "R111_erythroid_temporal_seed42",
    "R111_s456":    "R111_erythroid_temporal_seed456",
    "R11_s123":     "R11_KLF1_TAL1_seed123",
    "R11_s42":      "R11_KLF1_TAL1_seed42",
    "R11_s456":     "R11_KLF1_TAL1_seed456",
    "R12_s123":     "R12_GATA2_GFI1B_PAX5_seed123",
    "R12_s42":      "R12_GATA2_GFI1B_PAX5_seed42",
    "R12_s456":     "R12_GATA2_GFI1B_PAX5_seed456",
    "R13_KO_KLF1":  "R13_KO_KLF1_seed42",
    "R13_OE_KLF1":  "R13_OE_KLF1_seed42",
    "R13_s123":     "R13_hemato_trueprior_seed123",
    "R13_s42":      "R13_hemato_trueprior_seed42",
    "R13_s456":     "R13_hemato_trueprior_seed456",
    "R15_s123":     "R14_hemato_seed123",
    "R15_s42":      "R14_hemato_seed42",
    "R15_s456":     "R14_hemato_seed456",
    "R165_s42":     "R165_erythroid_myeloid_seed42",
    "R84_s123":     "R84_erythroid_seed123",
    "R84_s42":      "R84_erythroid_seed42",
    "R9":           "r9_backup",
    "R1_s123":      "R1_neural_seed123",
    "R1_s456":      "R1_neural_seed456",
    "R2_s42":       "R2_curriculum_only_20260531_180852",
    "R3_s42":       "R3_mixed_dist_only_20260601_025237",
    "R4_s42":       "R4_tf_prior_only_20260601_113012",
    "R5_s42":       "R5_curriculum_mixed_20260601_202222",
    "R6_s42":       "R6_curriculum_tfprior_20260602_042146",
    "R7_s42":       "R7_mixed_tfprior_20260602_110953",
    "R16_s42":       "R14_wrongprior_seed42",
    "R1_s42":        "R1_unagi_neuro_full_20260531_092524",
    "R8_s42":       "R8_string_20260613_002335",
}
# config -> manuscript CSV run name providing the LATENT side (same names here,
# kept separate so the two sides can diverge if the CSV is ever re-exported).
CSV_NAME = dict(RUN_NAME)

# tissue -> vanilla reference supplying the denominator.  The SAME run is used
# for the decoder and the latent side (using "first vanilla row in the CSV"
# silently mismatched the two sides).
#
# A candidate must satisfy BOTH (a) it exists as a RUN DIRECTORY -- the latent
# side is recomputed from that run's iter_4 stagedata by `find_stagedata` -- and
# (b) it reproduces the PUBLISHED tr_V for the tissue. For the neural tissue the
# two requirements CONFLICT; see VANILLA_NOTE.
VANILLA = {
    "hemo":   "R14_hemato_vanilla",   # tr 5634.8 == published
    "neural": "R0_vanilla_seed123",   # tr 6792.0 != published 5402.1 (see note)
    "lung":   "R10",                  # tr 12568.7 == published
}
# Candidates tried IN ORDER whenever the current pick fails to reproduce the
# published tr_V within TR_TOL -- not only when the pick is missing.  (The
# previous revision fell back on a MISSING candidate only, so a candidate that
# merely got the wrong tr -- e.g. a seed-123 vanilla in place of the published
# seed-42 one -- passed through silently.)
VANILLA_FALLBACK = {
    "hemo":   ["R14/vanilla", "R14_vanilla"],
    "neural": ["R0_vanilla_seed456", "R0_seed42", "R0_vanilla_seed42"],
    "lung":   ["R10_vanilla"],
}
# Published vanilla traces tr(W'W) at iter 4 -- the anchors quoted in the
# manuscript. Frozen in decoder_unified/vanilla/MANIFEST.json.
PUBLISHED_TR_V = {"hemo": 5634.8, "neural": 5402.1, "lung": 12568.7}
TR_TOL = 0.01     # 1 % relative tolerance
#
# VANILLA_NOTE (neural) -- an unresolved DATA GAP, surfaced not hidden:
# the published neural tr_V = 5402.1 belongs to decoder_out/R0_seed42, but
# result/R0 has NO `R0_seed42` directory (only R0_vanilla_seed123,
# R0_vanilla_seed456 and R0_vanilla_unagi_20260530_223153), so its iter_4
# stagedata is gone and the latent side cannot be recomputed for it.
# The script therefore falls back to a run that DOES have stagedata, whose
# decoder tr is ~25.7 % higher. That mismatch is reported in the log and in the
# output JSON under `vanilla_tr_check` -- it is never silent.


def log(*a):
    print(*a, flush=True)


# ----------------------------------------------------------------- Ridge ----
def find_stagedata(name):
    """Locate the iter_4 stagedata for a run directory name.

    Two layouts exist on disk:
        <run>/iter_4/stagedata/*.h5ad          (most runs)
        <run>/<stage>/stagedata/*.h5ad         (r9_backup: stages 0..4, no
                                                iter_* level at all)
    Only iter_4 (or, for the r9 layout, the stage directories) is used --
    accumulating iter_0..iter_4 would inflate N five-fold.
    """
    hits = glob.glob(os.path.join(RESULT, "**", name, ITER, "stagedata"),
                     recursive=True)
    hits = [h for h in hits if "stagedata" in h]
    if not hits:
        # r9-style: numeric *iteration* directories directly under the run.
        # These are iterations, NOT stages -- accumulating 0..4 inflates N
        # five-fold.  Only the terminal iteration 4 is the audited state
        # (verified: tr = 6511.64 == audit 6511.6, N = 19999).
        alt = glob.glob(os.path.join(RESULT, "**", name, "4", "stagedata"),
                        recursive=True)
        hits = sorted(alt)
    return hits


def ridge_spectrum(name, rowchunk=4096):
    """Recompute the published Ridge readout from raw stagedata.

    PROTOCOL (reverse-engineered and verified, see notes below):
        Zs = (Z - mean) / sd   per latent dim
        Xs = (X - mean) / sd   per gene
        W  = solve(Zs'Zs + alpha*I, Zs'Xs)      alpha = 1.0

    How this was established: `recalc_unagi_vanilla_null.py` documents the
    audit protocol as a Ridge fit on *raw* Z and X, but its own validation
    fails (`match: false`, 11305.49 vs the published 5634.8).  We tested all
    four standardisation combinations against `decoder_unified/loading/
    R15_s42.npy` (tr = 5160.92, pr = 7.8718) and only standardising BOTH
    sides reproduces it exactly:

        raw/raw    tr=11956.55  pr=4.4568  cos=+0.537
        raw/Xstd   tr=75672.70  pr=6.1093  cos=+0.774
        Zstd/raw   tr= 1005.74  pr=4.4315  cos=+0.718
        Zstd/Xstd  tr= 5160.92  pr=7.8718  cos=+1.000   <-- exact

    This also matches the documented behaviour of `ridge_readout` in the
    kappa-c-audit package ("standardises Z and X per column/gene").

    Sufficient statistics are accumulated in one streaming pass so the full
    dense X is never materialised.
    """
    sds = find_stagedata(name)
    if not sds:
        return None, "no iter_4 stagedata for %s" % name
    files = []
    for sd in sds:
        files += [os.path.join(sd, f) for f in sorted(os.listdir(sd))
                  if f.endswith(".h5ad") and f != "dataset.h5ad"]
    if not files:
        return None, "empty stagedata for %s" % name

    ZtZ = None
    ZtX = None
    sz = None
    sx = None
    sxx = None
    szz = None
    N = 0
    G = None
    d = None
    for path in files:
        with h5py.File(path, "r") as h:
            Z = np.asarray(h[LAT][:], dtype=np.float64)
            d = Z.shape[1]
            if ZtZ is None:
                ZtZ = np.zeros((d, d))
                sz = np.zeros(d)
                szz = np.zeros(d)
            Xh = h["X"]
            n, g = Xh.shape
            if G is None:
                G = g
                ZtX = np.zeros((d, g))
                sx = np.zeros(g)
                sxx = np.zeros(g)
            for s in range(0, n, rowchunk):
                e = min(s + rowchunk, n)
                Xb = np.asarray(Xh[s:e], dtype=np.float64)
                Zb = Z[s:e]
                ZtZ += Zb.T @ Zb
                ZtX += Zb.T @ Xb
                sz += Zb.sum(0)
                szz += (Zb ** 2).sum(0)
                sx += Xb.sum(0)
                sxx += (Xb ** 2).sum(0)
                del Xb
            N += n

    mz = sz / N
    mx = sx / N
    sdz = np.sqrt(np.clip(szz / N - mz ** 2, 1e-12, None))
    vx = sxx / N - mx ** 2
    sdx = np.sqrt(np.clip(vx, 1e-12, None))
    dead = vx <= 1e-16                      # constant genes -> zero loading

    Dz = np.diag(1.0 / sdz)
    Dx = np.diag(1.0 / sdx)
    As = Dz @ (ZtZ - N * np.outer(mz, mz)) @ Dz
    Bs = Dz @ (ZtX - N * np.outer(mz, mx)) @ Dx
    W = np.linalg.solve(As + ALPHA * np.eye(d), Bs)
    W[:, dead] = 0.0

    lam = np.linalg.eigvalsh(W @ W.T)
    lam = np.clip(lam, 0.0, None)
    tr = float(lam.sum())
    pr = float(lam.sum() ** 2 / (lam ** 2).sum())

    # --- added 2026-09-17 (S1f): the R2 form + the precondition diagnostic ---
    # tr(W'W) equals sum_g R^2_g ONLY when Corr(Z) ~ I.  With As the standardised
    # Z'Z and X standardised so that SS_tot_g = N:
    #     sum_g R^2_g = tr(As @ W W') / N
    # and Corr(Z) = As / N.  Measured on a real trained latent, Corr(Z) had
    # condition number 2.4e3 and the two forms diverged by up to 40 %; whitening
    # restored agreement exactly.  See
    # outputs/positive_result/S1_LIVE_PATH_RESULTS.md sections 2-3.
    sum_r2 = float(np.trace(As @ (W @ W.T)) / N)
    _ev = np.linalg.eigvalsh(As / N)
    corr_cond = float(_ev.max() / max(_ev.min(), 1e-300))
    n_small = int((_ev < 0.01).sum())

    return dict(run=name, stagedata=sd, n_cells=N, n_genes=G, d_latent=d,
                n_dead_genes=int(dead.sum()),
                tr=tr, pr=pr, sum_r2=sum_r2, corr_cond=corr_cond,
                n_eig_below_0p01=n_small,
                top1=float(lam.max() / lam.sum())), None


def latent_spectrum(name):
    """Latent-side spectrum from raw stagedata.

    Protocol verified against the manuscript's own CSV (`z_spectrum_all.csv`):
    lambda = eig(cov(Z)) with Z centred -- i.e. np.cov(Z, rowvar=False).
        R14_hemato_seed42   tr 178.611 == CSV 178.61
        R14_hemato_vanilla  tr 239.768 == CSV 239.77
        R0_vanilla_seed123  tr 325.550 == CSV 325.55
    and the vehicle pr = 5.1856 reproduces the 5.19 quoted in Fig. S11b.
    Recomputing here (rather than reading the CSV) lets R1_s42 in, whose
    latent row is absent from the CSV.
    """
    sds = find_stagedata(name)
    if not sds:
        return None, "no iter_4 stagedata for %s" % name
    blocks = []
    for sd in sds:
        for fn in sorted(os.listdir(sd)):
            if fn.endswith(".h5ad") and fn != "dataset.h5ad":
                with h5py.File(os.path.join(sd, fn), "r") as h:
                    blocks.append(np.asarray(h[LAT][:], dtype=np.float64))
    if not blocks:
        return None, "empty stagedata for %s" % name
    Z = np.vstack(blocks)
    lam = np.linalg.eigvalsh(np.cov(Z, rowvar=False))
    lam = np.clip(lam, 0.0, None)
    tr = float(lam.sum())
    pr = float(lam.sum() ** 2 / (lam ** 2).sum())
    return dict(run=name, n_cells=int(Z.shape[0]), d=Z.shape[1],
                tr=tr, pr=pr), None


def get_latent(name, cache):
    key = "LATENT::" + name
    if key in cache:
        return cache[key]
    spec, err = latent_spectrum(name)
    cache[key] = {"error": err} if spec is None else spec
    return cache[key]


def load_cache():
    if os.path.exists(CACHE):
        try:
            return json.load(open(CACHE))
        except Exception:
            return {}
    return {}


def get(name, cache):
    if name in cache:
        return cache[name]
    spec, err = ridge_spectrum(name)
    cache[name] = {"error": err} if spec is None else spec
    return cache[name]


# --------------------------------------------------------------- inputs ----
def load_audit():
    return json.load(open(AUDIT_JSON))


def load_csv():
    out = {}
    for r in csv.DictReader(open(CSV_Z)):
        if int(r["iteration"]) == 4:
            out[r["run"]] = dict(tissue=r["tissue"], role=r["role"],
                                 tr=float(r["tr"]), pr=float(r["pr"]))
    return out


# ----------------------------------------------------------------- main ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0,
                    help="only process the first N audit configs (debug)")
    ap.add_argument("--validate", action="store_true",
                    help="only run the published-number validation")
    a = ap.parse_args()

    audit = load_audit()
    cz = load_csv()
    cache = load_cache()

    # ---------- validation: does recomputation reproduce the audit? --------
    log("=" * 84)
    log("VALIDATION -- recomputed tr(W'W) must equal published tr_P")
    log("=" * 84)
    rel_errs = []
    for r in audit:
        cfg = r["config"]
        run = RUN_NAME.get(cfg)
        if not run:
            log("  %-14s %-34s %12.1f %12s  NO_MAPPING" %
                (cfg, "-", r["tr_P"], "-"))
            continue
        spec = get(run, cache)
        if "error" in spec:
            log("  %-14s %-34s %12.1f %12s  %s" %
                (cfg, run, r["tr_P"], "-", spec["error"]))
            continue
        rel = abs(spec["tr"] - r["tr_P"]) / max(r["tr_P"], 1e-9)
        rel_errs.append(rel)
        log("  %-14s %-34s %12.1f %12.2f  rel=%7.1e  %s" %
            (cfg, run, r["tr_P"], spec["tr"], rel,
             "OK" if rel < 1e-3 else "*** MISMATCH ***"))
    if rel_errs:
        log("")
        log("  configs validated: %d   max rel err: %.2e   within 1e-3: %d/%d"
            % (len(rel_errs), max(rel_errs),
               sum(1 for x in rel_errs if x < 1e-3), len(rel_errs)))
    json.dump(cache, open(CACHE, "w"), indent=1)
    if a.validate:
        return

    # ---------- vanilla denominators (decoder AND latent, same run) --------
    log("")
    log("=" * 84)
    log("VANILLA REFERENCE (denominator; identical run for both sides)")
    log("=" * 84)
    van = {}
    vlat = {}
    vcheck = {}
    for t, name in VANILLA.items():
        target = PUBLISHED_TR_V.get(t)
        order = [name] + VANILLA_FALLBACK.get(t, [])
        best = None            # (rel_err, candidate, spec)
        chosen = None          # (candidate, spec, rel_err)
        for cand in order:
            spec = get(cand, cache)
            if "error" in spec:
                continue
            rel = abs(spec["tr"] - target) / target if target else 0.0
            if best is None or rel < best[0]:
                best = (rel, cand, spec)
            if rel <= TR_TOL:
                chosen = (cand, spec, rel)
                break
        if chosen is None:
            if best is None:
                log("  %-7s decoder MISSING (no candidate resolvable from %s)"
                    % (t, order))
                continue
            rel, cand, spec = best
            log("  %-7s NO candidate reproduces published tr_V %.1f within %.0f%%;"
                % (t, target, TR_TOL * 100))
            log("          using the closest: %s (tr %.1f, %.1f%% off)"
                % (cand, spec["tr"], rel * 100))
            chosen = (cand, spec, rel)
        cand, spec, rel = chosen
        if cand != name:
            log("  %-7s decoder candidate -> %s" % (t, cand))
        lspec = get_latent(cand, cache)
        if "error" in lspec:
            log("  %-7s latent MISSING (%s)" % (t, lspec["error"]))
            continue
        van[t] = spec
        vlat[t] = lspec
        vcheck[t] = dict(run=cand, tr=float(spec["tr"]),
                         published_tr_V=float(target) if target else None,
                         rel_err=float(rel), ok=bool(target is not None
                                                     and rel <= TR_TOL))
        log("  %-7s %-24s decoder tr=%9.2f pr=%7.3f | latent tr=%8.2f pr=%7.3f%s"
            % (t, spec["run"], spec["tr"], spec["pr"], lspec["tr"], lspec["pr"],
               "" if (target is not None and rel <= TR_TOL)
               else "   <-- tr_V MISMATCH vs published"))

    # ---------- per-config ------------------------------------------------
    log("")
    log("=" * 84)
    log("PER-CONFIGURATION")
    log("=" * 84)
    rows = []
    skipped = []
    for r in (audit[: a.limit] if a.limit else audit):
        cfg, tis = r["config"], r["tissue"]
        if tis not in van:
            skipped.append((cfg, tis, "no vanilla reference"))
            continue
        run = RUN_NAME.get(cfg)
        if not run:
            skipped.append((cfg, tis, "no run-name mapping"))
            continue
        spec = get(run, cache)
        if "error" in spec:
            skipped.append((cfg, tis, spec["error"]))
            continue
        lspec = get_latent(run, cache)
        if "error" in lspec:
            skipped.append((cfg, tis, lspec["error"]))
            continue
        if tis not in vlat:
            skipped.append((cfg, tis, "no latent vanilla reference"))
            continue
        v, vz = van[tis], vlat[tis]
        rows.append(dict(
            config=cfg, tissue=tis, run=run,
            kappa_C_W_trace=spec["tr"] / v["tr"],
            kappa_C_Z_trace=lspec["tr"] / vz["tr"],
            pr_ratio_W=spec["pr"] / v["pr"],
            pr_ratio_Z=lspec["pr"] / vz["pr"],
            tr_W=spec["tr"], pr_W=spec["pr"],
            tr_Z=lspec["tr"], pr_Z=lspec["pr"],
            n_cells=spec["n_cells"],
            audit_tr_P=r["tr_P"], recomputed_tr=spec["tr"],
            tr_rel_err=abs(spec["tr"] - r["tr_P"]) / max(r["tr_P"], 1e-9),
        ))

    log("  %-14s %-7s %9s %9s %10s %10s %9s" %
        ("cfg", "tissue", "k-C(W)", "k-C(Z)", "pr-ratioW", "pr-ratioZ", "tr relerr"))
    for x in rows:
        log("  %-14s %-7s %9.3f %9.3f %10.3f %10.3f %9.1e" %
            (x["config"], x["tissue"], x["kappa_C_W_trace"], x["kappa_C_Z_trace"],
             x["pr_ratio_W"], x["pr_ratio_Z"], x["tr_rel_err"]))
    for cfg, tis, why in skipped:
        log("  SKIP %-14s %-7s  %s" % (cfg, tis, why))
    json.dump(cache, open(CACHE, "w"), indent=1)
    log("")
    log("usable: %d   skipped: %d" % (len(rows), len(skipped)))
    if not rows:
        return

    # ---------- Q1: trace conservation ------------------------------------
    n = len(rows)
    prods = np.array([x["kappa_C_W_trace"] * x["kappa_C_Z_trace"] for x in rows])
    lw = np.log([x["kappa_C_W_trace"] for x in rows])
    lz = np.log([x["kappa_C_Z_trace"] for x in rows])
    within = int(np.sum(np.abs(np.log(prods)) < np.log(1.25)))
    r_tr = float(np.corrcoef(lz, lw)[0, 1])
    slope_tr = float(np.polyfit(lz, lw, 1)[0])
    conc = float(np.mean(np.sign(lw) == np.sign(lz)))

    log("")
    log("=" * 84)
    log("Q1  TRACE CONSERVATION   kappa-C(W) * kappa-C(Z) == 1 ?")
    log("=" * 84)
    log("  n                       = %d" % n)
    log("  median product          = %.3f" % np.median(prods))
    log("  within +/-25%% of unity  = %d/%d" % (within, n))
    log("  pearson(logW, logZ)     = %+.3f" % r_tr)
    log("  slope logW ~ logZ       = %+.3f" % slope_tr)
    log("  sign concordance        = %.2f" % conc)
    verdict = "REJECTED" if within / n < 0.5 else "NOT_REJECTED"
    log("  --> trace conservation %s" % verdict)

    # ---------- Q2: like-for-like effective rank --------------------------
    lpW = np.log([x["pr_ratio_W"] for x in rows])
    lpZ = np.log([x["pr_ratio_Z"] for x in rows])
    mpW = float(np.exp(lpW).mean())
    mpZ = float(np.exp(lpZ).mean())
    r_er = float(np.corrcoef(lpZ, lpW)[0, 1])
    slope_er = float(np.polyfit(lpZ, lpW, 1)[0])
    nW = int(np.sum(lpW < 0))
    nZ = int(np.sum(lpZ < 0))
    log("")
    log("=" * 84)
    log("Q2  LIKE-FOR-LIKE EFFECTIVE RANK (both sides = participation ratio)")
    log("=" * 84)
    log("  mean pr-ratio decoder W = %.3f   (%d/%d below unity)" % (mpW, nW, n))
    log("  mean pr-ratio latent  Z = %.3f   (%d/%d below unity)" % (mpZ, nZ, n))
    log("  pearson(log prW, log prZ) = %+.3f   slope = %+.3f" % (r_er, slope_er))
    diss = "SURVIVES" if (mpZ < 0.95 and mpW > mpZ) else "DOES_NOT_SURVIVE"
    log("  --> dissociation %s (latent contracts, decoder does not)" % diss)

    # ---------- cross-statistic coupling ---------------------------------
    cross = dict(
        corr_traceW_prZ=float(np.corrcoef(lw, lpZ)[0, 1]),
        corr_prW_traceZ=float(np.corrcoef(lpW, lz)[0, 1]),
    )
    log("")
    log("  cross-statistic: corr(traceW, prZ) = %+.3f   corr(prW, traceZ) = %+.3f"
        % (cross["corr_traceW_prZ"], cross["corr_prW_traceZ"]))

    # ---------- Q3b: inference (bootstrap CI, sign tests, Holm) -----------
    rng = np.random.default_rng(20260912)
    B = 5000
    lpW = np.log([x["pr_ratio_W"] for x in rows])
    lpZ = np.log([x["pr_ratio_Z"] for x in rows])
    lw = np.log([x["kappa_C_W_trace"] for x in rows])
    lz = np.log([x["kappa_C_Z_trace"] for x in rows])

    def boot_ci(vals, fn=np.mean, alpha=0.05):
        idx = rng.integers(0, len(vals), size=(B, len(vals)))
        bs = fn(vals[idx], axis=1)
        return (float(np.percentile(bs, 100 * alpha / 2)),
                float(np.percentile(bs, 100 * (1 - alpha / 2))))

    def boot_ci_med(vals, alpha=0.05):
        idx = rng.integers(0, len(vals), size=(B, len(vals)))
        bs = np.median(vals[idx], axis=1)
        return (float(np.percentile(bs, 100 * alpha / 2)),
                float(np.percentile(bs, 100 * (1 - alpha / 2))))

    # pre-specified family of 3 confirmatory tests
    tests = []
    # T1: median trace product == 1 ?  (two-sided bootstrap test on the median)
    ci_prod = boot_ci_med(prods)
    lp = np.log(prods)
    bmed = np.median(lp[rng.integers(0, n, size=(B, n))], axis=1)
    obs = float(np.median(lp))
    p_prod = float(2 * min(np.mean(bmed <= 0), np.mean(bmed >= 0)))
    p_prod = min(1.0, max(p_prod, 1.0 / B))
    tests.append(dict(name="trace product differs from 1",
                      statistic="median log-product = %+.3f" % obs,
                      ci95=["%+.3f" % v for v in ci_prod], p=p_prod))
    # T2: does the latent side compress?  (Wilcoxon signed-rank on log ratio)
    st, p = stats.wilcoxon(lpZ, alternative="less")
    ci = boot_ci(np.exp(lpZ), fn=np.mean)
    tests.append(dict(name="latent pr-ratio < 1",
                      statistic="mean ratio = %.3f" % float(np.mean(np.exp(lpZ))),
                      ci95=["%.3f" % x for x in ci], p=float(p)))
    # T3: the dissociation itself -- PAIRED test that the latent contracts more
    # than the decoder.  This is the claim that matters; testing the decoder
    # alone is uninformative because its CI straddles 1.
    d = lpZ - lpW
    st, p = stats.wilcoxon(d, alternative="less")
    ci_d = boot_ci(d, fn=np.mean)
    tests.append(dict(name="latent contracts more than decoder (paired)",
                      statistic="mean log-diff = %+.3f" % float(np.mean(d)),
                      ci95=["%+.3f" % x for x in ci_d], p=float(p)))
    # Holm-Bonferroni across the family
    order = sorted(range(len(tests)), key=lambda i: tests[i]["p"])
    m = len(tests)
    running = 0.0
    for rank, i in enumerate(order):
        adj = min(1.0, (m - rank) * tests[i]["p"])
        running = max(running, adj)          # enforce monotonicity
        tests[i]["p_holm"] = running

    log("")
    log("=" * 84)
    log("Q4  INFERENCE (bootstrap CI, pre-specified family of %d, Holm)" % m)
    log("=" * 84)
    log("  %-30s %-26s %-18s %9s %9s" %
        ("test", "statistic", "95% CI (bootstrap)", "p", "p_Holm"))
    for t in tests:
        log("  %-30s %-26s %-18s %9.2e %9.2e" %
            (t["name"], t["statistic"], "[%s, %s]" % (t["ci95"][0], t["ci95"][1]),
             t["p"], t["p_holm"]))
    log("")
    log("  bootstrap: B=%d, seed=20260912. All %d tests survive Holm at 0.05: %s"
        % (B, m, all(t["p_holm"] < 0.05 for t in tests)))
    log("")
    log("  decoder side, reported descriptively (its CI straddles unity, so it")
    log("  is NOT claimed to compress):  mean pr-ratio = %.3f, 95%% CI [%.3f, %.3f]"
        % (float(np.mean(np.exp(lpW))), *boot_ci(np.exp(lpW), fn=np.mean)))

    # descriptive-only note: how many statistics are reported per configuration
    log("")
    log("  multiplicity note: %d configurations x 5 reported statistics "
        "(kappa, C, C_head, zeta, pr)." % n)
    log("  Only the %d pre-specified confirmatory tests above are used for the "
        "claims;" % m)
    log("  all remaining per-configuration values are reported descriptively "
        "and are not tested.")

    # ---------- Q3: per tissue -------------------------------------------
    per = {}
    for t in sorted({x["tissue"] for x in rows}):
        sub = [x for x in rows if x["tissue"] == t]
        per[t] = dict(
            n=len(sub),
            mean_pr_ratio_W=float(np.mean([x["pr_ratio_W"] for x in sub])),
            mean_pr_ratio_Z=float(np.mean([x["pr_ratio_Z"] for x in sub])),
            mean_kappa_C_W=float(np.mean([x["kappa_C_W_trace"] for x in sub])),
            mean_kappa_C_Z=float(np.mean([x["kappa_C_Z_trace"] for x in sub])),
            configs=[x["config"] for x in sub])
    log("")
    log("=" * 84)
    log("Q3  PER-TISSUE")
    log("=" * 84)
    log("  %-8s %4s %12s %12s %11s %11s" %
        ("tissue", "n", "pr-ratio W", "pr-ratio Z", "k-C(W)", "k-C(Z)"))
    for t, v in per.items():
        log("  %-8s %4d %12.3f %12.3f %11.3f %11.3f" %
            (t, v["n"], v["mean_pr_ratio_W"], v["mean_pr_ratio_Z"],
             v["mean_kappa_C_W"], v["mean_kappa_C_Z"]))

    # ---------- deep-compression check (the 0/33 claim) ------------------
    ndeep = int(np.sum(np.array([x["kappa_C_W_trace"] for x in rows]) < 0.5))
    log("")
    log("  decoder deep compression (kappa-C(W) < 0.5): %d/%d" % (ndeep, n))

    out = dict(
        protocol=("Ridge(alpha=1) Z->X recomputed from iter_4 stagedata; "
                  "kappa=tr(W'W)_prior/tr(W'W)_vanilla; "
                  "effective rank = participation ratio (sum l)^2/sum l^2"),
        source_of_truth=("F:/UNAGI_Project/UNAGI/result/<run>/iter_4/stagedata "
                         "(same protocol as recalc_unagi_vanilla_null.py); "
                         "audit estimator confirmed to be decoder_unified/loading "
                         "Ridge loadings, NOT decoder_out network weights"),
        n=n, skipped=[dict(config=c, tissue=t, reason=w) for c, t, w in skipped],
        validation=dict(n_validated=len(rel_errs),
                        max_rel_err=float(max(rel_errs)) if rel_errs else None,
                        n_within_1e_3=int(sum(1 for x in rel_errs if x < 1e-3))),
        trace_level=dict(median_product=float(np.median(prods)),
                         within_25pct=within, n=n,
                         pearson=r_tr, slope=slope_tr, sign_concordance=conc,
                         verdict=verdict),
        effective_rank_level=dict(mean_pr_ratio_W=mpW, mean_pr_ratio_Z=mpZ,
                                  n_below_unity_W=nW, n_below_unity_Z=nZ,
                                  pearson=r_er, slope=slope_er, verdict=diss),
        cross_statistic=cross,
        per_tissue=per,
        n_deep_compression=ndeep,
        inference=dict(bootstrap_B=B, seed=20260912, n_tests=m,
                       correction="Holm-Bonferroni",
                       all_survive_0p05=bool(all(t["p_holm"] < 0.05
                                                 for t in tests)),
                       tests=tests),
        rows=rows,
        vanilla_tr_check=vcheck,
        published_tr_V=PUBLISHED_TR_V,
        tr_tol=TR_TOL,
    )
    _inp = [os.path.join(DATA, "..", "..", "outputs", "positive_result",
            "s1f_0of33_recheck.json"),
            os.path.join(DATA, "z_spectrum_all.csv")]
    out = _attach_provenance(out, script=__file__,
                            inputs=[x for x in _inp if os.path.isfile(x)],
                            extra={"statistic": "corrected predictability form (pr_ratio); see REPORT_A_PROVENANCE_BINDING.md"})
    json.dump(out, open(os.path.join(DATA, "kappa_c_pr_comparison.json"), "w"),
              indent=1)
    log("")
    log("wrote ../data/kappa_c_pr_comparison.json")


if __name__ == "__main__":
    main()
