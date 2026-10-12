#!/usr/bin/env python3
"""check_form_consistency.py -- catch mixed-form and stale-count claims in paper1.

WHY THIS EXISTS
---------------
Four separate defects in this manuscript had one shared root cause: **the same quantity
was quoted in two different functional forms, or at two different scopes, and the two were
used as if interchangeable.**

  1. Introduction said "125 released run-iteration-stage files"; every other place said 796.
  2. Figure 4C said "40 / 40" (neural-crest subset only); the text said 796 (all arms).
  3. Figure 2 and three prose sites said "15/15 Jacobian re-audit"; the audit covers 33/33.
  4. Figure S10's product blended a corrected decoder with a TRACE-form latent
     (`kappa_r2 * kC_Z`), giving median 1.47 / 9 of 33, while the caption called it
     "the corrected predictability form". Same statistic, same form on both sides, is
     0.72 / 8 of 33.

Case 4 is the hard one: the number is *correct for the artifact* (`product_r2` really is
1.4722) but the artifact itself is an incoherent blend.  No amount of text-grepping finds
that.  So this checker recomputes every multiplicative claim in BOTH coherent forms and
reports when a quoted value matches ONLY the mixed form.

RULES
-----
  F1  Every product / ratio statistic quoted in the sources must equal a statistic computed
      from ONE coherent form (both factors from the same form).  A value that appears only
      in the mixed table is a failure.
  F2  Configuration/file counts quoted in the sources must belong to the maintained
      authoritative set.  A count that is not in that set is flagged as a possible stale
      reference.

USAGE
    python tools/check_form_consistency.py            # report only, always exit 0
    python tools/check_form_consistency.py --strict   # exit 1 on any F1/F2 failure
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER1 = os.path.dirname(HERE)                       # .../paper1_kc_diagnostic
PROJECT = os.path.dirname(os.path.dirname(PAPER1))   # .../UNAGI_Project

PR = os.path.join(PAPER1, "data", "kappa_c_pr_comparison.json")
S1G = os.path.join(PROJECT, "outputs", "positive_result", "s1g_corrected_summaries.json")
S1F = os.path.join(PROJECT, "outputs", "positive_result", "s1f_0of33_recheck.json")
INV = os.path.join(PROJECT, "outputs", "geneWeight_recovered_inventory.json")

# Sources scanned for claims.  Keep in sync with the \input graph of main.tex.
CLAIM_FILES = ["main.tex", "star_methods.tex", "supplementary.tex", "supp_addendum.tex",
               "latent_residence_conclusion.tex", "docs/si_extended_methods.tex"]

# Authoritative counts.  Anything outside this set is a candidate stale reference.
# Derived by `calibrate_counts` over the live sources: every value here was inspected in
# context and adjudicated legitimate.  Two historical defects are deliberately absent:
#   125 -> a stale stage-file count (the inventory says 796)
#   15  -> is present, because it is legitimate as "15 retained G-VAE seeds";
#          the STALE_COUNT_PHRASES table below catches it only in its stale phrasing.
AUTHORITATIVE = {
    2: "lung runs in the spectrum scan / small denominators",
    3: "seed clusters",
    4: "G-VAE configurations / tissues",
    5: "claim numerator",
    9: "claim denominator (different claim from 8)",
    15: "G-VAE haematopoietic vanilla seeds (null calibration)",
    18: "claim denominator (17/18)",
    19: "claim denominator (16/19)",
    22: "haematopoietic configurations",
    23: "configurations inside the cross-family band (22 haematopoietic + 1 neural-crest)",
    29: "retained run-iterations (stagedata on disk)",
    30: "archived UNAGI decoder readouts",
    33: "configuration audit size",
    24: "haematopoietic runs in the spectrum scan (data/z_spectrum_all.csv)",
    37: "runs with a computable Z-spectrum, all tissues (data/z_spectrum_all.csv)",
    8: "run logs terminating in an uncaught exception (16 contain a traceback)",
    39: "run directories available on disk (only 37 yield a Z-spectrum)",
    40: "random seeds in the constructive check",
    45: "neural-arm pair count",
    59: "retained run logs",
    70: "prior-quality-landscape configurations",
    90: "per-configuration readouts",
    105: "neural-crest stage files",
    210: "ordered seed-pair comparisons",
    456: "model seed label",
    796: "released run-iteration-stage files (all arms)",
}

# Phrasings that were factually wrong in an earlier revision.  Each entry is a regex; a match
# means the stale claim has come back.  Keep this list short and each entry undeniable.
STALE_COUNT_PHRASES = [
    (r"\b125\b[^\n]{0,40}(?:stage files|run--iterations?|run-iterations?)",
     "125 stage files -- the inventory records 796 (Introduction defect)"),
    (r"\b40\b\s+surviving\s+(?:stage\s+)?files",
     "40 surviving stage files -- superseded by 796 across all arms (Fig. 4C defect)"),
    (r"all\s+\$?40\$?\s*(?:surviving\s+)?stage files",
     "40 stage files -- superseded by 796"),
    (r"\b15\s*/\s*15\b",
     "15/15 Jacobian re-audit -- the audit covers all 33 configurations"),
    (r"All\s+15\s+configurations",
     "All 15 configurations -- the audit covers all 33"),
    (r"\b40\s*/\s*40\b",
     "40/40 files -- superseded by 796/796"),
    (r"\b18\s+(?:Ridge-only|Ridge)\b",
     "18 Ridge-only configurations -- no configuration is Ridge-only now"),
]


def load(path):
    with open(path, encoding="utf-8-sig") as f:
        return json.load(f)


def median(v):
    v = sorted(v)
    n = len(v)
    return v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2


def within(v, tol=0.25, log=False):
    import math
    if log:
        return sum(1 for x in v if abs(math.log(x)) < math.log(1 + tol))
    return sum(1 for x in v if abs(x - 1) < tol)


def read_claims():
    out = {}
    for rel in CLAIM_FILES:
        p = os.path.join(PAPER1, rel)
        if os.path.isfile(p):
            with open(p, encoding="utf-8", errors="replace") as f:
                out[rel] = f.read()
    return out


def build_forms(rows_pr, rows_s1g, rows_s1f):
    """Return {form_name: {'vals': [...], 'median': float, 'n': int,
                           'linear25': int, 'log25': int, 'mixed': bool}}."""
    s1g = {r["config"]: r for r in rows_s1g}
    s1f = {r["config"]: r for r in rows_s1f}
    cfg = [r["config"] for r in rows_pr]

    forms = {}
    # coherent: corrected decoder x corrected latent
    v = [s1f[c]["kappa_r2"] * rows_pr[i]["pr_ratio_Z"] for i, c in enumerate(cfg)]
    forms["corrected(W) x corrected(Z)"] = v
    # coherent: trace decoder x trace latent (authoritative pair)
    v = [rows_pr[i]["kappa_C_W_trace"] * rows_pr[i]["kappa_C_Z_trace"] for i in range(len(cfg))]
    forms["trace(W) x trace(Z)"] = v
    # coherent: the author's own trace-form product field
    v = [s1g[c]["product_tr"] for c in cfg]
    forms["product_tr (author field)"] = v
    # INCOHERENT: corrected decoder x trace latent  <- the S10 defect
    v = [s1g[c]["kappa_r2"] * rows_pr[i]["kappa_C_Z_trace"] for i, c in enumerate(cfg)]
    forms["MIXED corrected(W) x trace(Z)"] = v

    out = {}
    for name, vals in forms.items():
        out[name] = {"vals": vals, "median": median(vals), "n": len(vals),
                     "linear25": within(vals, 0.25, log=False),
                     "log25": within(vals, 0.25, log=True),
                     "mixed": name.startswith("MIXED")}
    return out


def near(a, b, tol):
    return abs(a - b) <= tol


def calibrate():
    """Regression test for the detector itself.

    A checker that never fires is worthless.  This mode runs STALE_COUNT_PHRASES against
    archived pre-fix revisions that are KNOWN to contain each defect, and asserts that every
    pattern is still matched by at least one of them.  If the live sources are ever reverted,
    the corresponding pattern fires in the main run; if a pattern silently stops matching, it
    fires here.
    """
    ARCHIVE = os.path.join(PAPER1, "releases_archive")
    CANDIDATES = [
        os.path.join(ARCHIVE, "_r79_incomplete_attempt", "source", "main.tex"),
        os.path.join(ARCHIVE, "_r79_incomplete_attempt", "source", "supplementary.tex"),
        os.path.join(ARCHIVE, "2026-10-07_s1-corrected-r87", "source", "supp_addendum.tex"),
        os.path.join(ARCHIVE, "2026-09-29_s1-corrected-r57", "source", "main.tex"),
    ]
    text = ""
    used = []
    for c in CANDIDATES:
        if os.path.isfile(c):
            with open(c, encoding="utf-8", errors="replace") as f:
                text += f.read() + "\n"
            used.append(os.path.basename(os.path.dirname(os.path.dirname(c))))
    print("calibration corpus: %d archived revision(s)" % len(used))
    unused = []
    for rx, why in STALE_COUNT_PHRASES:
        n = len(re.findall(rx, text, re.I | re.S))
        status = "fires (%d)" % n if n else "*** NEVER FIRES ***"
        print("  %-58s %s" % (rx[:58], status))
        if not n:
            unused.append(rx)
    print()
    if unused:
        print("FAIL: %d pattern(s) never matched the pre-fix corpus -- they cannot detect"
              " the defect they name" % len(unused))
        return 1
    print("PASS: every stale-phrase pattern still fires on the pre-fix corpus")
    return 0


# Scope note: this checker covers paper1 only.  The sibling paper2 package was assessed
# once (2026-10-09) and left out of scope by author decision, so no path to it is kept
# here -- a paper1 gate must not depend on, or report on, another manuscript.


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true", help="exit 1 on any failure")
    ap.add_argument("--calibrate", action="store_true",
                    help="regression-test the detector against archived pre-fix revisions")
    args = ap.parse_args()

    if args.calibrate:
        return calibrate()

    for path in (PR, S1G, S1F, INV):
        if not os.path.isfile(path):
            print("MISSING ARTIFACT: %s" % path)
            return 1 if args.strict else 0

    rows_pr = load(PR)["rows"]
    rows_s1g = load(S1G)["rows"]
    rows_s1f = load(S1F)["rows"]
    inv = load(INV)

    forms = build_forms(rows_pr, rows_s1g, rows_s1f)
    claims = read_claims()
    alltext = "\n".join(claims.values())

    fails = []

    # ---------------- F1: form agreement for product/ratio claims -----------------
    print("=" * 78)
    print("F1  multiplicative claims: does the quoted value match a COHERENT form?")
    print("=" * 78)
    print("  %-34s %8s %6s %8s %8s" % ("form", "median", "log25", "lin25", "mixed"))
    for name, f in forms.items():
        print("  %-34s %8.2f %5d/%-3d %5d/%-3d %8s"
              % (name, f["median"], f["log25"], f["n"], f["linear25"], f["n"],
                 "YES" if f["mixed"] else "-"))
    print()

    coherent = [f for n, f in forms.items() if not f["mixed"]]
    mixed = [f for n, f in forms.items() if f["mixed"]][0]

    # only product medians, i.e. "median product X.XX" or "median X.XX" where the sentence
    # is about the product.  A bare decimal elsewhere is not a product claim.
    quoted_medians = set()
    for m in re.finditer(r"median(?:\s+of\s+the)?\s+product\s*\$?(\d\.\d\d)\$?", alltext, re.I):
        quoted_medians.add(float(m.group(1)))
    for m in re.finditer(r"product[^.\n]{0,60}?median[^.\n]{0,12}?\$?(\d\.\d\d)\$?",
                         alltext, re.I):
        quoted_medians.add(float(m.group(1)))
    print("  product medians quoted in the sources: %s"
          % (sorted(quoted_medians) or "none"))
    diss_path = os.path.join(PAPER1, "data", "dissociation_jacobian_ridge.json")
    diss = load(diss_path) if os.path.isfile(diss_path) else None
    for q in sorted(quoted_medians):
        hits = [n for n, f in forms.items() if near(q, f["median"], 0.006)]
        if not hits and diss is not None and any(
                near(q, diss[k]["product_median"], 0.006)
                for k in ("dissociation_ridge", "dissociation_jacobian")):
            print("     %.2f -> dissociation product (ok, separate statistic)" % q)
            continue
        if not hits:
            fails.append("F1 quoted median %.2f matches NO form" % q)
            print("     %.2f -> *** FAIL: matches no coherent form ***" % q)
            continue
        good = [h for h in hits if not h.startswith("MIXED")]
        if good:
            print("     %.2f -> coherent: %s" % (q, ", ".join(good)))
        else:
            fails.append("F1 quoted median %.2f matches only the MIXED form" % q)
            print("     %.2f -> *** FAIL: matches ONLY the mixed form (%s) ***"
                  % (q, ", ".join(h for h in hits if h.startswith("MIXED"))))

    # "N of 33" / "N/33" counts are only product claims when the sentence says "within 25%";
    # "29/33 below unity" is a different statistic and must not be judged here.
    for m in re.finditer(r"\$?(\d{1,2})\s*(?:/|of)\s*33\$?[^\n]{0,40}?within", alltext, re.I):
        k = int(m.group(1))
        if k in (0, 33):
            continue
        good = [n for n, f in forms.items()
                if not f["mixed"] and (f["linear25"] == k or f["log25"] == k)]
        if good:
            print("     %d/33 within-25%% -> coherent: %s" % (k, ", ".join(sorted(set(good)))))
        else:
            mixonly = [n for n, f in forms.items()
                       if f["mixed"] and (f["linear25"] == k or f["log25"] == k)]
            fails.append("F1 quoted %d/33 within 25%% matches no coherent form" % k)
            print("     %d/33 -> *** FAIL: no coherent form; mixed=%s ***"
                  % (k, ", ".join(mixonly) or "none"))

    # ---------------- F2: authoritative counts -----------------------------------
    print()
    print("=" * 78)
    print("F2  configuration / file counts: any stale reference?")
    print("=" * 78)

    # (a) counts used in a count phrase must be in the authoritative set
    NOUN = (r"(?:configurations?|stage\s+files?|run-?iterations?|runs|logs|seeds|"
            r"checkpoints|readouts|ordered\s+seed-pair\s+comparisons)")
    flagged = set()
    for m in re.finditer(r"\$?(\d{1,4})\$?\s+(?:\w+\s+){0,2}?" + NOUN, alltext, re.I):
        v = int(m.group(1))
        if v not in AUTHORITATIVE:
            flagged.add(v)
    # (b) phrases that were factually wrong before
    for rx, why in STALE_COUNT_PHRASES:
        for m in re.finditer(rx, alltext, re.I | re.S):
            ln = alltext[:m.start()].count("\n") + 1
            fails.append("F2 stale phrase: %s" % why)
            print("     *** FAIL: %s" % why)
            print("        ...%s..." % " ".join(m.group(0).split())[:120])
    if flagged:
        for v in sorted(flagged):
            fails.append("F2 count %d is not in the authoritative set" % v)
            print("     %-6d *** FAIL: not an authoritative count ***" % v)
    else:
        print("     all count phrases inside the authoritative set")

    # ---------------- verdict ----------------------------------------------------

    print()
    print("=" * 78)
    if fails:
        print("FAIL: %d issue(s)" % len(fails))
        for f_ in fails:
            print("   - " + f_)
        print()
        print("   NOTE: the MIXED form above is the Figure S10 defect retained only for the")
        print("   record.  If a quoted value matches it, the sources and the artifact agree")
        print("   with each other but not with a coherent statistic.")
    else:
        print("PASS: no mixed-form or stale-count claims found")
    print("=" * 78)
    return 1 if (fails and args.strict) else 0


if __name__ == "__main__":
    sys.exit(main())
