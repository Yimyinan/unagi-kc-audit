# -*- coding: utf-8 -*-
"""audit_guard.py -- edit/commit gate for the manuscript number audit.

Runs code/audit_sec29_numbers.py (which extracts numbers FROM main.tex and
compares them to the JSON artifacts) and exits non-zero if the manuscript is
NOT internally consistent. Intended to be wired as a git pre-commit hook (see
code/hooks/pre-commit and code/hooks/pre-commit.bat) or run manually before
freezing a release.

Exit codes
----------
  0  all claims consistent ("ALL CONSISTENT")
  1  inconsistencies found (or audit script errored)
  2  audit script missing / could not run
"""
from __future__ import annotations

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
AUDIT = os.path.join(HERE, "audit_sec29_numbers.py")


def main() -> int:
    if not os.path.isfile(AUDIT):
        print(f"ERROR: audit script not found: {AUDIT}", file=sys.stderr)
        return 2
    py = sys.executable or "python"
    try:
        proc = subprocess.run([py, AUDIT], capture_output=True, text=True)
    except Exception as exc:  # pragma: no cover
        print(f"ERROR: could not run audit: {exc}", file=sys.stderr)
        return 2

    out = proc.stdout + proc.stderr
    sys.stdout.write(out)
    if proc.returncode != 0:
        print("\n[audit_guard] audit_sec29_numbers.py exited non-zero.", file=sys.stderr)
        return 1
    if "ALL CONSISTENT" not in out or "MISMATCH" in out:
        print("\n[audit_guard] manuscript NOT consistent -- blocking.", file=sys.stderr)
        return 1
    print("\n[audit_guard] OK: manuscript number audit consistent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
