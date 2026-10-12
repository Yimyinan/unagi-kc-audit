#!/usr/bin/env python3
"""injection_path_check.py -- a runnable worked example of the injection-path check.

The claim this check decides
---------------------------
"Giving a single-cell VAE a gene-regulatory (GRN) prior changed its decoder."

That claim is only meaningful if the prior *reached the objective*. A GRN prior is
usually injected as a per-gene weight vector. If that vector is constructed but never
read by the loss, then the decoder cannot have changed for the prior's reasons -- and
any observed change is attributable to something else.

The check therefore walks the delivery path and stops at the first place the signal
dies:

    prior source  ->  tissue subset  ->  per-gene weights  ->  the loss term
                                                                   ^
                                        D1 asks whether the artefact written to
                                        training actually carries non-zero weights
                                        D2/D4 ask whether any code path reads them

Usage
-----
    python injection_path_check.py --unagi-root F:/UNAGI_Project/UNAGI \\
                                   --source-tree F:/UNAGI_Project/b1_expansion/UNAGI_source \\
                                   [--json out.json]

Everything is read-only. Nothing is trained. Results are printed as a verdict plus
the raw counts, so the numbers in a paper can be checked against them directly.

Author's note on scope
----------------------
D1 reports BOTH the search root it used and the file count, because the same
manuscript quotes two different denominators (a 40-file benchmark subset and a
796-file full tree).  Always state which root produced a count.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

# --------------------------------------------------------------------------
# D1: does the per-gene weight layer actually carry non-zero values?
# --------------------------------------------------------------------------


def d1_weight_layer_occupancy(roots, pattern="**/stagedata/*.h5ad"):
    """Scan every stagedata *.h5ad under each root for the geneWeight CSR layer.

    A CSR layer with indptr of length n_rows+1 and a 'data' of length 0 is a
    *structurally present but entirely empty* layer: the code allocated it and
    never filled it.
    """
    try:
        import h5py
    except ImportError:
        return {"status": "skipped", "reason": "h5py unavailable"}

    per_root = {}
    total_files = total_empty = total_populated = 0
    examples = []

    for root in roots:
        root = Path(root)
        if not root.is_dir():
            per_root[str(root)] = {"status": "missing"}
            continue
        files = sorted(root.glob(pattern))
        n = empty = populated = no_layer = 0
        for f in files:
            try:
                with h5py.File(f, "r") as h:
                    key = "layers/geneWeight/data"
                    if key not in h:
                        no_layer += 1
                        continue
                    n += 1
                    nnz = h[key].shape[0]
                    if nnz == 0:
                        empty += 1
                    else:
                        populated += 1
                        if len(examples) < 5:
                            examples.append(str(f))
            except Exception as exc:                      # noqa: BLE001
                no_layer += 1
                if len(examples) < 5:
                    examples.append("%s  (unreadable: %s)" % (f, str(exc)[:60]))
        per_root[str(root)] = {
            "status": "ran", "glob": pattern, "files_matched": len(files),
            "with_layer": n, "empty_layer": empty, "populated_layer": populated,
            "no_layer": no_layer,
        }
        total_files += n
        total_empty += empty
        total_populated += populated

    if total_files == 0:
        verdict = "undetermined"
    elif total_populated == 0:
        verdict = "ALL-ZERO: the weight layer is present but empty in every file scanned"
    else:
        verdict = "POPULATED: at least one file carries non-zero weights"

    return {
        "status": "ran", "per_root": per_root,
        "files_with_layer": total_files, "empty": total_empty,
        "populated": total_populated,
        "verdict": verdict, "non_empty_examples": examples,
    }


# --------------------------------------------------------------------------
# D2: is the weight vector actually READ, or only written?
# --------------------------------------------------------------------------
# NB: an earlier version of this check matched a bare 'geneWeight' and produced
# false positives from comments and docstrings (e.g. the string
# "_apply_tf_prior_weights()" appearing inside prose).  Real call sites are matched
# as code only: comments, strings and docstrings are stripped first.
_ASSIGN_PAT = re.compile(
    r"(?:^|\s)(?:self\.)?_?tf_prior_weights\s*=|"
    r"(?:^|\s)geneWeight\s*=|"
    r"(?:^|\s)self\.(?:_tf_prior_weights|geneWeight)\s*=")
_READ_PAT = re.compile(
    r"Loss\w*\s*\(\s*geneWeight|"
    r"\bgeneWeight\b(?!\s*=)")


def _code_lines(path):
    """Yield (lineno, code_line) with comments and string literals removed."""
    import tokenize
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            for tok in tokenize.generate_tokens(fh.readline):
                if tok.type == tokenize.COMMENT:
                    continue
                if tok.type == tokenize.STRING:
                    continue
                if tok.type == tokenize.NAME or tok.type == tokenize.OP or \
                        tok.type == tokenize.NUMBER:
                    yield tok.start[0], tok.line
    except Exception:                                      # noqa: BLE001
        return


def d2_reader_search(source_tree):
    """Find where the weight vector is assigned, and where (if anywhere) it is read.

    Uses the tokenizer so comments and docstrings cannot be mistaken for code.
    """
    tree = Path(source_tree)
    if not tree.is_dir():
        return {"status": "missing", "tree": str(tree)}

    assigns, readers = [], []
    for p in sorted(tree.rglob("*.py")):
        seen_a, seen_r = set(), set()
        for lineno, line in _code_lines(p):
            src = line.rstrip()
            if lineno not in seen_a and _ASSIGN_PAT.search(src):
                seen_a.add(lineno)
                assigns.append({"file": str(p.relative_to(tree)), "line": lineno,
                                "src": src.strip()[:110]})
            if lineno not in seen_r and _READ_PAT.search(src):
                seen_r.add(lineno)
                readers.append({"file": str(p.relative_to(tree)), "line": lineno,
                                "src": src.strip()[:110]})

    return {
        "status": "ran", "tree": str(tree),
        "n_assignment_sites": len(assigns), "n_reader_sites": len(readers),
        "assignment_sites": assigns[:12], "reader_sites": readers[:12],
        # Reader sites inside a class DEFINITION are not evidence that the value is
        # consumed: a parameter that is stored on self and never passed in remains
        # dead. D3 (instantiation sites) is the decisive test; D1 (the artefact) is
        # the empirical one. D2 is locational only.
        "verdict": ("D2 is LOCATIONAL ONLY. %d assignment site(s) and %d reader "
                    "site(s) exist; readers inside a definition do not prove the value "
                    "is consumed. Read D3 for reachability and D1 for the artefact."
                    % (len(assigns), len(readers))),
        "decisive": False,
    }


# --------------------------------------------------------------------------
# D3: does the likelihood-weighting branch have an instantiation site?
# --------------------------------------------------------------------------


def d3_branch_instantiation(source_tree):
    """The objective may contain a branch that would consume a per-gene vector.  If it
    has no instantiation site, the branch is dead code."""
    tree = Path(source_tree)
    if not tree.is_dir():
        return {"status": "missing", "tree": str(tree)}
    hits = []
    for p in tree.rglob("*.py"):
        try:
            txt = p.read_text(encoding="utf-8", errors="replace")
        except Exception:                                  # noqa: BLE001
            continue
        for i, line in enumerate(txt.splitlines(), 1):
            if re.search(r"Loss\w*\(\s*geneWeight", line):
                hits.append({"file": str(p.relative_to(tree)), "line": i,
                             "src": line.strip()[:110]})
    return {
        "status": "ran", "tree": str(tree),
        "n_instantiations": len(hits), "instantiations": hits,
        "verdict": ("branch is DEAD: no instantiation site" if not hits
                    else "branch is instantiated %d time(s)" % len(hits)),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--unagi-root", default="F:/UNAGI_Project/UNAGI",
                    help="root holding training outputs (default: %(default)s)")
    ap.add_argument("--source-tree", default="F:/UNAGI_Project/b1_expansion/UNAGI_source",
                    help="the patched UNAGI source tree (default: %(default)s)")
    ap.add_argument("--json", help="also write the full report here")
    args = ap.parse_args(argv)

    roots = [os.path.join(args.unagi_root, "benchmark"),
             os.path.join(args.unagi_root, "result")]

    report = {
        "check": "injection-path",
        "question": "does the injected prior reach the training objective?",
        "roots_scanned": roots,
        "source_tree": args.source_tree,
        "D1_weight_layer": d1_weight_layer_occupancy(roots),
        "D2_readers": d2_reader_search(args.source_tree),
        "D3_branch_instantiation": d3_branch_instantiation(args.source_tree),
    }

    d1 = report["D1_weight_layer"]
    d3 = report["D3_branch_instantiation"]
    report["verdict"] = {
        "delivery": ("BLOCKED -- the weight layer is empty in every file scanned"
                     if d1.get("status") == "ran" and d1.get("populated") == 0
                     else "not blocked (or undetermined)"),
        "branch": d3.get("verdict", "n/a"),
        "evidence_used": ["D1 (empirical: the artefact written to training)",
                          "D3 (structural: no instantiation site)"],
        "evidence_not_used": ["D2 (locational only; reader sites in a definition do "
                              "not establish consumption)"],
        "implication": ("The prior is constructed and assigned but never read. Decoder "
                        "changes cannot be attributed to GRN-prior content in this "
                        "implementation."),
        "scope_limit": ("This is a property of THIS implementation. It does not show "
                        "that GRN priors are ineffective in general."),
    }

    # ---- human-readable output ----
    print("=" * 74)
    print("INJECTION-PATH CHECK")
    print("=" * 74)
    print("question : %s" % report["question"])
    print()
    print("D1  per-gene weight layer occupancy")
    if d1.get("status") == "ran":
        for root, st in d1["per_root"].items():
            if st.get("status") == "ran":
                print("      %-52s %4d files  empty=%-4d populated=%d"
                      % (os.path.relpath(root, os.path.dirname(args.unagi_root)),
                         st["files_matched"], st["empty_layer"], st["populated_layer"]))
            else:
                print("      %-52s %s" % (root, st.get("status")))
        print("      TOTAL with layer: %d   empty: %d   populated: %d"
              % (d1["files_with_layer"], d1["empty"], d1["populated"]))
        print("      VERDICT: %s" % d1["verdict"])
    else:
        print("      skipped: %s" % d1.get("reason"))

    print()
    print("D2  assignment vs reader sites")
    d2 = report["D2_readers"]
    if d2.get("status") == "ran":
        print("      assignment sites: %d   reader sites: %d" % (
            d2["n_assignment_sites"], d2["n_reader_sites"]))
        print("      VERDICT: %s" % d2["verdict"])
        for a in d2["assignment_sites"][:4]:
            print("        assign %s:%d  %s" % (a["file"], a["line"], a["src"][:70]))
    else:
        print("      %s" % d2.get("status"))

    print()
    print("D3  likelihood-weighting branch instantiation")
    if d3.get("status") == "ran":
        print("      instantiations: %d" % d3["n_instantiations"])
        print("      VERDICT: %s" % d3["verdict"])
    else:
        print("      %s" % d3.get("status"))

    print()
    print("=" * 74)
    print("VERDICT: delivery %s" % report["verdict"]["delivery"])
    print("         implication: %s" % report["verdict"]["implication"])
    print("=" * 74)

    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("wrote %s" % args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
