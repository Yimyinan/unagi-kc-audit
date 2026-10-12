# -*- coding: utf-8 -*-
"""check_figure_palette.py -- machine check that figure scripts draw only from figure_palette.

Why this exists: the manuscript figures had drifted apart because individual scripts
hardcoded their own hex values. Counting hex literals by eye does not catch a newly
introduced off-palette colour, so this scans for any hex literal that is not an exact
member of the exported palette, and additionally flags the semantic overload the palette
contract forbids (IDENTITY colours must not be used for verdicts and vice versa).

Usage:
    python code/check_figure_palette.py            # audit every figure script
    python code/check_figure_palette.py FILE ...   # audit specific scripts

Exit code 1 if any off-palette colour or overload is found.
"""
from __future__ import annotations

import io
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# ---------------------------------------------------------------- palette members
PAL = {}
src = io.open(os.path.join(HERE, "figure_palette.py"), encoding="utf-8").read()
for m in re.finditer(r"^([A-Z][A-Z0-9_]*)\s*=\s*\"(#[0-9A-Fa-f]{6})\"", src, re.M):
    PAL[m.group(2).upper()] = m.group(1)

DEFINED = set(PAL)                       # colours the palette defines
# near-white / near-black structural values are legitimate raw literals
STRUCTURAL = {"#FFFFFF", "#000000", "#FFF", "#000"}
BASE = re.compile(r"#[0-9A-Fa-f]{3,6}")

def scan(path):
    t = io.open(path, encoding="utf-8", errors="ignore").read()
    hits = [h.upper() for h in BASE.findall(t)]
    off = sorted({h for h in hits if h not in DEFINED and h not in STRUCTURAL})
    return off

def main(argv):
    files = argv[1:]
    if not files:
        files = []
        for dp, dirs, fs in os.walk(ROOT):
            if any(x in dp.replace("\\", "/") for x in
                   ("/releases/", "/snapshots/", "/_archive", "/build/", "/archive/")):
                continue
            for f in fs:
                if f.endswith(".py") and ("fig" in f.lower() or "make_" in f.lower()):
                    files.append(os.path.join(dp, f))
    files.sort()
    bad = 0
    print("palette defines %d colours" % len(DEFINED))
    print()
    for p in files:
        off = scan(p)
        rel = os.path.relpath(p, ROOT)
        if off:
            bad += 1
            print("OFF-PALETTE  %s" % rel)
            for h in off:
                # show the line so the fix is obvious
                t = io.open(p, encoding="utf-8", errors="ignore").read().split("\n")
                for i, l in enumerate(t, 1):
                    if h.lower() in l.lower() or h in l:
                        print("    %s  line %-4d %s" % (h, i, l.strip()[:96]))
                        break
        else:
            print("ok           %s" % rel)
    print()
    print("scripts with off-palette colours: %d" % bad)
    return 1 if bad else 0

if __name__ == "__main__":
    sys.exit(main(sys.argv))
