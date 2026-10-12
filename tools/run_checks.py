#!/usr/bin/env python3
"""run_checks.py -- run the whole paper1 verification suite in one command.

Each tool below was added in response to a specific defect found during the audit.  They are
listed in the order a pre-submission pass should run them: cheap text checks first, then the
data-backed ones, then the build gate.

    python tools/run_checks.py            # run everything, print a table, exit 0
    python tools/run_checks.py --strict   # exit 1 if any GATING check fails

GATING means "must be clean before submission".  Items marked advisory are reported but do not
fail the run (for example the metadata gate is expected to fail until the repository URL and
Zenodo DOI are filled in).

WHAT EACH CHECK CAUGHT, HISTORICALLY
  check_text_integrity      placeholder/marker leakage into manuscript sources
  check_citations           dead or broken \\cite/\\bibitem pairs
  check_form_consistency    four mixed-form / stale-count defects (125 vs 796, 40 vs 796,
                            15/15 vs 33/33, and the Figure S10 mixed-form product 1.47 vs 0.72)
  si_reference_map          supplementary floats defined but never referenced
  check_submission_metadata unfilled [REPOSITORY_URL] / [ZENODO_DOI] sentinels
  latex_silent_linter       \\multicolumn without a wrapping column (the Fig. S8 19 cm overflow)
  test_figure_name_guard    the figure-name collision guard
  build logs                undefined references; overfull boxes
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER1 = os.path.dirname(HERE)


def run(cmd, cwd=None, timeout=900):
    # encoding/errors MUST be explicit: on a Chinese-locale Windows the default text codec is
    # GBK, and pdftotext emits UTF-8 curly quotes / em-dashes.  Without this the reader thread
    # dies with UnicodeDecodeError and returns NO output, which silently looks like a failure.
    try:
        p = subprocess.run(cmd, cwd=cwd or PAPER1, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=timeout)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except Exception as exc:                                  # noqa: BLE001
        return 99, "could not run: %s" % exc


def log_overfull(path):
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8", errors="replace") as f:
        t = f.read()
    return len(re.findall(r"Overfull \\hbox", t)), len(re.findall(r"undefined", t)), \
        len(re.findall(r"^! ", t, re.M))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--quick", action="store_true",
                    help="skip the tier-2 content hash of the raw outputs (~108 s for "
                         "120.2 GB); the tier-1 tree hash still runs")
    args = ap.parse_args()

    py = sys.executable
    rows = []          # (name, gating, ok, detail)

    def add(name, gating, ok, detail):
        rows.append((name, gating, ok, detail))

    # ---- text-level checks -------------------------------------------------------
    rc, out = run([py, "tools/check_text_integrity.py"])
    add("text integrity", True, "0 finding(s)" in out,
        out.strip().splitlines()[-1][:70] if out.strip() else "no output")

    rc, out = run([py, "tools/check_citations.py"])
    dead = re.search(r"DEAD \(defined, never cited\): \[(.*?)\]", out)
    broken = re.search(r"BROKEN \(cited, undefined\): \[(.*?)\]", out)
    ok = bool(dead) and bool(broken) and not dead.group(1).strip() and \
        not broken.group(1).strip()
    add("citations", True, ok, "dead/broken = 0" if ok else "see output")

    rc, out = run([py, "tools/check_form_consistency.py"])
    add("form consistency", True, "PASS:" in out,
        next((l.strip() for l in out.splitlines() if l.startswith("PASS") or
              l.startswith("FAIL")), "no verdict"))

    rc, out = run([py, "tools/si_reference_map.py"])
    # si_reference_map now distinguishes a float's OWN CAPTION from a real in-prose citation.
    # The old version counted the caption, so a never-cited float scored refs=1 and the check
    # reported "none" while 2 figures and 1 table were orphaned.  Read the explicit ORPHANS line.
    m = re.search(r"ORPHANS: figures=(.*?) tables=(.*)", out)
    if m:
        figs, tabs = m.group(1).strip(), m.group(2).strip()
        ok = figs == "none" and tabs == "none"
        add("SI reference map", True, ok,
            "no orphaned SI floats" if ok
            else "ORPHANED figures=%s tables=%s" % (figs, tabs))
    else:
        add("SI reference map", True, False, "could not parse ORPHANS line")

    rc, out = run([py, "code/latex_silent_linter.py"])
    live = [l for l in out.splitlines()
            if "[LOW]" in l and "releases_archive" not in l and "snapshots" not in l]
    add("silent-overflow linter", False, not live,
        "%d live advisory finding(s)" % len(live))

    rc, out = run([py, "code/test_figure_name_guard.py"])
    add("figure-name guard", True, "UNIT TEST: PASS" in out,
        "PASS" if "UNIT TEST: PASS" in out else "FAIL")

    # ---- submission gate ---------------------------------------------------------
    rc, out = run([py, "tools/check_submission_metadata.py", "--strict"])
    add("submission metadata gate", False, rc == 0,
        "clean" if rc == 0 else "unfilled sentinels (expected until URL/DOI are set)")

    # ---- raw-output provenance manifest -------------------------------------------
    # Report A step 4.3 records a (path,size) tree hash for every stagedata file of every
    # retained run-iteration, plus content hashes for the training logs.  Recomputing those
    # hashes catches any add/remove/rename/resize of the raw outputs that back the claims.
    # It does NOT catch an in-place content edit at constant size -- the manifest says so.
    manifest = os.path.join(os.path.dirname(os.path.dirname(PAPER1)), "_infra", "provenance",
                            "RUN_LEVEL_MANIFEST.json")
    if os.path.isfile(manifest):
        with open(manifest, encoding="utf-8") as f:
            man = json.load(f)
        root = man.get("root", "")
        bad, checked = [], 0
        if os.path.isdir(root):
            import hashlib as _h

            def _tree_hash(entries):
                hh = _h.sha256()
                for rel, size in sorted(entries):
                    hh.update(("%s\t%d\n" % (rel, size)).encode())
                return hh.hexdigest()

            for run_key, v in man.get("runs", {}).items():
                for ri, rec in v.get("run_iterations", {}).items():
                    sd = os.path.join(root, ri.replace("/", os.sep), "stagedata")
                    if not os.path.isdir(sd):
                        bad.append("%s MISSING" % ri)
                        continue
                    entries = []
                    for fn in os.listdir(sd):
                        fp = os.path.join(sd, fn)
                        if os.path.isfile(fp):
                            # key exactly as the manifest builder did: "stagedata/<file>"
                            entries.append(("stagedata/" + fn, os.path.getsize(fp)))
                    checked += 1
                    if _tree_hash(entries) != rec.get("tree_hash"):
                        bad.append("%s HASH-CHANGED (%s)" % (ri, run_key))
        else:
            bad.append("manifest root not found: %s" % root)
        add("raw-output manifest", True, not bad,
            "%d run-iterations verified" % checked if not bad
            else "%d changed/missing: %s" % (len(bad), "; ".join(bad[:3])))
    else:
        add("raw-output manifest", False, True,
            "not found (build with _infra/tmp/build_run_manifest.py)")

    # ---- raw-output content hashes (tier 2, default depth "all") -------------------
    # RUN_LEVEL_CONTENT_HASHES.json carries a byte-level sha256 for every stagedata file
    # and every training log.  Re-hashing costs ~108 s for 120.2 GB at ~1.1 GB/s, which
    # buys detection of content edits that preserve file size -- the blind spot the tree
    # hash cannot cover.  Skip with --quick.
    content = os.path.join(os.path.dirname(os.path.dirname(PAPER1)), "_infra", "provenance",
                           "RUN_LEVEL_CONTENT_HASHES.json")
    if args.quick:
        add("raw-output content hash", True, True, "skipped (--quick)")
    elif os.path.isfile(content):
        import hashlib as _hc

        def _sha(p, chunk=8 << 20):
            hh = _hc.sha256()
            with open(p, "rb") as fh:
                while True:
                    b = fh.read(chunk)
                    if not b:
                        break
                    hh.update(b)
            return hh.hexdigest()

        cdoc = json.load(open(content, encoding="utf-8"))
        root = (man.get("root", "") if os.path.isfile(manifest) else "")
        mismatched, missing, n_ok = [], [], 0
        if os.path.isdir(root):
            for ri, entries in cdoc.get("hashes", {}).items():
                sd = os.path.join(root, ri.replace("/", os.sep))
                for key, want in entries.items():
                    fp = os.path.join(sd, key.replace("/", os.sep))
                    if not os.path.isfile(fp):
                        missing.append(key)
                    elif _sha(fp) != want:
                        mismatched.append("%s/%s" % (ri, key))
                    else:
                        n_ok += 1
            for rel, rec in cdoc.get("logs", {}).items():
                fp = os.path.join(root, rel.replace("/", os.sep))
                if not os.path.isfile(fp):
                    missing.append(rel)
                elif _sha(fp) != rec.get("sha256"):
                    mismatched.append(rel)
                else:
                    n_ok += 1
            bad = mismatched + missing
            add("raw-output content hash", True, not bad,
                "%d files byte-identical (depth=%s)" % (n_ok, cdoc.get("depth", "?"))
                if not bad else "%d changed / %d missing: %s"
                % (len(mismatched), len(missing), "; ".join(bad[:3])))
        else:
            add("raw-output content hash", True, False, "manifest root not found")
    else:
        add("raw-output content hash", False, True,
            "not found (build with _infra/tmp/hash_all.py)")

    # ---- numeric claims vs artefacts ----------------------------------------------
    # Every other check here verifies FORMAT and CONSISTENCY.  This one verifies the
    # numbers themselves by recomputing them from the artefacts under data/,
    # outputs/positive_result/ and UNAGI/result.  It does not retrain anything, so it is
    # cheap enough to gate on.  A mismatch means a prose claim and its artefact have
    # drifted apart -- which is exactly the class of defect that section 10.1 of the audit
    # record had to fix by hand seven times.
    vnum = os.path.join(os.path.dirname(os.path.dirname(PAPER1)), "_infra", "tools",
                        "verify_numbers.py")
    if os.path.isfile(vnum):
        rc2, out2 = run([py, vnum, "--strict"])
        m = re.search(r"(\d+) checks, (\d+) failed", out2)
        if m:
            n_ck, n_bad = int(m.group(1)), int(m.group(2))
            add("numeric claims vs artefacts", True, n_bad == 0,
                "%d checks, %d failed" % (n_ck, n_bad) if n_bad == 0
                else "%d/%d FAILED: %s" % (
                    n_bad, n_ck,
                    "; ".join(l.strip() for l in out2.splitlines() if l.strip()
                              .startswith("FAIL"))[:90]))
        else:
            add("numeric claims vs artefacts", True, False,
                "verifier produced no summary (rc=%d)" % rc2)
    else:
        add("numeric claims vs artefacts", False, True,
            "not found at %s" % vnum)

    # ---- Briefings in Bioinformatics build ---------------------------------------
    # The BIB version is a separate document (OUP class).  Its log is only checked if it
    # has been built; the build itself needs -shell-escape (the OUP class runs texcount).
    bib_pdf = os.path.join(PAPER1, "submission", "main_bib.pdf")
    bib_log = os.path.join(PAPER1, "submission", "main_bib.log")
    if os.path.isfile(bib_log):
        with open(bib_log, encoding="utf-8", errors="replace") as f:
            bl = f.read()
        over = len(re.findall(r"Overfull \\hbox", bl))
        undef = len(re.findall(r"Citation .* undefined|Reference .* undefined", bl))
        err = len(re.findall(r"^! ", bl, re.M))
        add("BIB version build", True, undef == 0 and err == 0,
            "overfull=%d undefined=%d errors=%d" % (over, undef, err))
        bbl = os.path.join(PAPER1, "submission", "main_bib.bbl")
        if os.path.isfile(bbl):
            with open(bbl, encoding="utf-8", errors="replace") as f:
                nb = len(re.findall(r"\\bibitem", f.read()))
            add("BIB bibliography entries", True, nb == 46, "%d entries" % nb)
        # The OUP class silently drops an abstract label when the brace and the first label
        # share a line ("\abstract{%\textbf{Motivation:}"): the % comments the label out,
        # the build reports ZERO errors, and the label vanishes from the PDF.  Only the
        # rendered text can catch it, so check the rendered text.
        if os.path.isfile(bib_pdf):
            rc2, txt = run(["pdftotext", bib_pdf, "-"])
            if rc2 == 0:
                missing = [lb for lb in ("Motivation:", "Results:", "Availability:",
                                         "Contact:")
                           if lb not in txt]
                add("BIB abstract labels", True, not missing,
                    "all 4 present" if not missing else "MISSING " + ", ".join(missing))
    else:
        add("BIB version build", False, True,
            "not built (pdflatex -shell-escape submission/main_bib.tex)")

    # ---- graphical abstract ------------------------------------------------------
    # Acceptance checks from submission/GRAPHICAL_ABSTRACT_SPEC.md section 11, run against
    # the RENDERED text (the generator's docstring names the forbidden items in order to
    # prohibit them, so scanning the source gives false failures).
    ga_pdf = os.path.join(PAPER1, "figures", "graphical_abstract.pdf")
    if os.path.isfile(ga_pdf):
        rc3, gatxt = run(["pdftotext", ga_pdf, "-"])
        if rc3 == 0:
            forbidden = ["1.47", "max |delta|", "15/15", "40 / 40", "prior-driven",
                         "independent codebase"]
            hit = [f for f in forbidden if re.search(re.escape(f), gatxt, re.I)]
            add("graphical abstract: forbidden content", True, not hit,
                "clean" if not hit else "PRESENT " + ", ".join(hit))
            need = ["prior-agnostic", "796 / 796", "Audit delivery before attributing"]
            miss = [n for n in need if n not in gatxt]
            add("graphical abstract: required content", True, not miss,
                "all present" if not miss else "MISSING " + ", ".join(miss))
            nums = set(re.findall(r"(?<![\w.])\d[\d,]*(?:\.\d+)?(?![\w])", gatxt))
            allowed = {"0", "1", "2", "3", "4", "5", "6", "10", "11", "12", "13",
                       "21", "29", "33", "0.5", "0.72", "0.77", "0.95", "3.5", "203",
                       "325", "795", "796", "9396", "9,396", "0.873", "1.003", "0.684"}
            extra = sorted(n for n in nums if n not in allowed and n != "2026")
            add("graphical abstract: unlisted numbers", True, not extra,
                "clean" if not extra else ", ".join(extra))
    else:
        add("graphical abstract", False, True, "not built (code/make_graphical_abstract.py)")

    # ---- build logs --------------------------------------------------------------
    for name, logf in (("main build log", "main.log"),
                       ("supplementary build log", "supplementary.log"),
                       ("addendum build log", "supp_addendum.log"),
                       ("extended-methods build log", r"docs\si_extended_methods.log")):
        r = log_overfull(os.path.join(PAPER1, logf))
        if r is None:
            add(name, True, False, "log missing -- compile first")
        else:
            over, undef, err = r
            add(name, True, over == 0 and undef == 0 and err == 0,
                "overfull=%d undefined=%d errors=%d" % (over, undef, err))

    # ---- report ------------------------------------------------------------------
    print("=" * 84)
    print("%-30s %-9s %-6s %s" % ("check", "gating", "result", "detail"))
    print("=" * 84)
    nfail = 0
    for name, gating, ok, detail in rows:
        if not ok and gating:
            nfail += 1
        print("%-30s %-9s %-6s %s"
              % (name, "gating" if gating else "advisory",
                 "ok" if ok else "FAIL", detail[:34]))
    print("=" * 84)
    if nfail:
        print("FAIL: %d gating check(s) failed" % nfail)
    else:
        print("PASS: all gating checks clean")
    print()
    print("  detector calibration (does the form checker still fire on pre-fix sources?):")
    rc, out = run([py, "tools/check_form_consistency.py", "--calibrate"])
    for line in out.strip().splitlines():
        if "PASS" in line or "FAIL" in line:
            print("    " + line.strip())
    return 1 if (nfail and args.strict) else 0


if __name__ == "__main__":
    sys.exit(main())
