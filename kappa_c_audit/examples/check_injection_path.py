#!/usr/bin/env python3
"""check_injection_path.py -- run the injection-path check on YOUR OWN pipeline.

The question
------------
If your model injects a gene-regulatory (or any per-gene) prior as a weight vector, a
change in the decoder is only attributable to that prior if the prior actually reached
the objective. This script walks the delivery path and tells you where it stops:

    prior source -> tissue subset -> per-gene weights -> the loss term
                                                            ^
        D1  does the artefact written to training carry non-zero weights?   (empirical)
        D2  where is the weight assigned, and where is it read?             (locational)
        D3  is the consuming branch instantiated anywhere?                  (structural)

D1 and D3 are decisive. D2 is locational only: a read inside a class definition does
not prove the stored value is consumed.

Usage
-----
    python check_injection_path.py --artifacts /path/to/run_outputs \
                                   --source    /path/to/your/training/code \
                                   [--layer-key layers/geneWeight/data] \
                                   [--weight-name geneWeight] \
                                   [--roots benchmark result] \
                                   [--pattern '**/stagedata/*.h5ad'] \
                                   [--json report.json]

Then read the printed verdict. If it says BLOCKED, no decoder change in that pipeline
can be attributed to the prior's content.

What you need to supply
-----------------------
1. `--artifacts`: the directory tree holding what training saved. The script globs
   `--pattern` beneath it. Keep whatever sub-directory names you actually use in
   `--roots`; each is reported separately, because a subset count and a full-tree count
   are different denominators and conflating them is an easy way to overstate coverage.
2. `--source`: the training source tree, scanned for the weight identifier.
3. `--layer-key`: the HDF5 path of the weight vector. Default assumes AnnData with a
   layer named geneWeight; change it for your format.
4. `--weight-name`: the Python name the vector goes by in your source.

Interpreting the two outcomes
-----------------------------
BLOCKED   The weight layer is present but empty in every artefact scanned, and/or the
          consuming branch has no instantiation site. Any decoder change you observed
          came from something other than the prior's content.
not blocked
          At least one artefact carries non-zero weights. This does NOT mean your
          prior is effective -- only that it arrived, so downstream claims are at least
          attributable. Proceed to the compression check.

Exit codes: 0 = ran and the path is not blocked, 1 = ran and it IS blocked,
            2 = could not run (bad paths, missing h5py).
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Run the injection-path check on your own pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Interpreting")[0])
    ap.add_argument("--artifacts", required=True,
                    help="root of the saved training artefacts")
    ap.add_argument("--source", required=True,
                    help="root of the training source tree")
    ap.add_argument("--layer-key", default="layers/geneWeight/data",
                    help="HDF5 path of the per-gene weight vector (default: %(default)s)")
    ap.add_argument("--weight-name", default="geneWeight",
                    help="identifier for the weight vector in the source "
                         "(default: %(default)s)")
    ap.add_argument("--roots", nargs="*", default=None,
                    help="sub-directories of --artifacts to scan; default scans the "
                         "root itself and the usual benchmark/result split")
    ap.add_argument("--pattern", default="**/*.h5ad",
                    help="glob for artefacts beneath each root (default: %(default)s)")
    ap.add_argument("--json", help="write the full report here")
    args = ap.parse_args(argv)

    if not os.path.isdir(args.artifacts):
        print("ERROR: --artifacts is not a directory: %s" % args.artifacts)
        return 2
    if not os.path.isdir(args.source):
        print("ERROR: --source is not a directory: %s" % args.source)
        return 2

    # import from the installed package, falling back to this directory's parent
    try:
        from kappa_c_audit.checks import injection_path_check
    except ImportError:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        try:
            from kappa_c_audit.checks import injection_path_check
        except ImportError:
            print("ERROR: could not import kappa_c_audit.checks.")
            print("       pip install kappa-c-audit, or run this from the package root.")
            return 2

    # Default roots: scan the root itself plus the conventional split, so that a
    # caller who does not care about layout still gets a per-root breakdown.
    if args.roots is None:
        cand = [d for d in ("benchmark", "result") if os.path.isdir(
            os.path.join(args.artifacts, d))]
        roots = cand if cand else [""]
    else:
        roots = args.roots

    res = injection_path_check(
        unagi_root=args.artifacts, source_tree=args.source,
        pattern=args.pattern, roots=roots,
        layer_key=args.layer_key, weight_name=args.weight_name)

    print("=" * 74)
    print("INJECTION-PATH CHECK")
    print("=" * 74)
    print("artifacts  : %s" % args.artifacts)
    print("source     : %s" % args.source)
    print("layer key  : %s" % args.layer_key)
    print("weight name: %s" % args.weight_name)
    print()
    print("D1  per-gene weight layer occupancy  (DECISIVE, empirical)")
    any_root = False
    for name, st in res.get("per_root", {}).items():
        if st.get("status") in (None,):
            any_root = True
            print("      %-24s %5d matched  with-layer=%-5d empty=%-5d populated=%d"
                  % (name or "(root)", st["files_matched"], st["with_layer"],
                     st["empty_layer"], st["populated_layer"]))
        else:
            print("      %-24s %s" % (name or "(root)", st.get("status")))
    if not any_root:
        print("      no artefacts matched -- check --pattern and --roots")
    tot = res.get("weight_layer_totals", {})
    print("      totals: with-layer=%s empty=%s populated=%s"
          % (tot.get("with_layer"), tot.get("empty"), tot.get("populated")))
    print("      VERDICT: %s" % res.get("weight_layer_verdict"))
    print()
    rd = res.get("readers", {})
    print("D2  assignment / reader sites       (locational only -- NOT evidence)")
    print("      assignment=%s  reader=%s" % (rd.get("n_assignment_sites"),
                                             rd.get("n_reader_sites")))
    print()
    br = res.get("branch", {})
    print("D3  consuming-branch instantiations (DECISIVE, structural)")
    print("      instantiations: %s" % br.get("n_instantiations"))
    print("      VERDICT: %s" % br.get("verdict"))
    for inst in (br.get("instantiations") or [])[:5]:
        print("        %s:%s  %s" % (inst.get("file"), inst.get("line"),
                                     inst.get("src", "")[:70]))
    print()
    print("=" * 74)
    delivery = res.get("verdict", {}).get("delivery")
    print("VERDICT: %s" % delivery)
    print(res.get("verdict", {}).get("implication", ""))
    print()
    print("SCOPE: %s" % res.get("verdict", {}).get("scope_limit", ""))
    print("=" * 74)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=2)
        print("wrote %s" % args.json)

    return 1 if delivery == "BLOCKED" else 0


if __name__ == "__main__":
    sys.exit(main())
