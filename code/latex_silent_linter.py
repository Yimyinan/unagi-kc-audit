# -*- coding: utf-8 -*-
"""latex_silent_linter.py -- catch LaTeX failures that emit NO error.

These are the manuscript traps that compile cleanly yet render wrong (recorded
in project memory as "these failures emit no error"):

  HIGH   \\pm written outside math mode  -> renders as literal "pm" text
  MEDIUM hardcoded hex colour not in figure_palette.py -> style drift
  LOW    \\multicolumn with only l/c/r spec (no p{}) -> cell overflows silently
  INFO   matplotlib tight_layout() in a figure script -> canvas can expand;
         the generated figure must be opened and inspected by eye

Usage: python code/latex_silent_linter.py [ROOT]
Scans every *.tex under ROOT (skips archive/, snapshots/, releases/).
Exit code is 1 if any HIGH finding, else 0.

The \\pm check uses per-line $ parity (ignoring escaped \$) plus display-math
environment tracking -- robust against \\command sequences inside $...$, unlike
a naive global char-by-char state machine.
"""
from __future__ import annotations

import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))

# Allowed colour hexes -- MUST mirror code/figure_palette.py (single source of truth).
PALETTE = {
    "#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b2",
    "#888888", "#9a9a9a", "#3b7a3b", "#eef2f7", "#2f2f2f",
}

MATHENV = {
    "equation", "equation*", "align", "align*", "gather", "gather*",
    "multline", "displaymath", "math", "eqnarray", "flalign", "alignat",
}

NUM = re.compile(r"[-+]?\d*\.\d+|\d+")
MULTICOLUMN = re.compile(r"\\multicolumn\{(\d+)\}\{([^}]*)\}")
HEX_IN_TEX = re.compile(r"(?:\\definecolor\{[^}]*\}\{[^}]*\}\{([0-9A-Fa-f]{6})\})"
                        r"|(?:\[HTML\]\{([0-9A-Fa-f]{6})\})"
                        r"|(?:HTML\}\{([0-9A-Fa-f]{6})\})")
TIGHT_LAYOUT = re.compile(r"\.tight_layout\s*\(")
BEGIN_ENV = re.compile(r"\\begin\{([a-zA-Z*]+)\}")
END_ENV = re.compile(r"\\end\{([a-zA-Z*]+)\}")


def _dollar_parity_before(s: str, pos: int) -> int:
    """Count unescaped '$' in s[:pos] (even => currently outside inline math)."""
    cnt = 0
    k = 0
    while k < pos:
        if s[k] == "\\" and k + 1 < pos and s[k + 1] == "$":
            k += 2
            continue
        if s[k] == "$":
            cnt += 1
        k += 1
    return cnt % 2


def lint_file(path: str, rel: str, findings: list[tuple]) -> None:
    text = open(path, encoding="utf-8", errors="ignore").read()
    lines = text.split("\n")

    # A file may make a maths-only command legal in text mode by redefining it through
    # \ensuremath in its OWN preamble:
    #     \let\origpm\pm
    #     \renewcommand{\pm}{\ensuremath{\origpm}}
    # Then `0.73\pm0.05` outside $...$ renders correctly and is NOT a defect. Without this the
    # generated Chinese manuscript reported 9 false HIGH findings.
    text_safe = set(re.findall(r"\\renewcommand\{\\([A-Za-z]+)\}\{\\ensuremath", text))
    text_safe |= set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}\{\\ensuremath", text))

    in_math_env = False
    for ln, line in enumerate(lines, 1):
        # update display-math environment state (toggle on begin/end of a math env)
        for m in BEGIN_ENV.finditer(line):
            if m.group(1) in MATHENV:
                in_math_env = True
        for m in END_ENV.finditer(line):
            if m.group(1) in MATHENV:
                in_math_env = False
        # HIGH: \\pm outside math mode (per-line inline parity OR display env)
        for m in re.finditer(r"\\pm", line):
            if "pm" in text_safe:
                continue
            inline_math = _dollar_parity_before(line, m.start()) == 1
            if not (in_math_env or inline_math):
                findings.append((rel, ln, "HIGH",
                                 r"\pm outside math mode -- renders as literal 'pm'"))
        # MEDIUM: hardcoded hex colour not in palette
        for m in HEX_IN_TEX.finditer(line):
            hexv = next(g for g in m.groups() if g)
            if hexv.lower() not in PALETTE:
                findings.append((rel, ln, "MEDIUM",
                                 f"non-palette colour #{hexv} (not in figure_palette.py)"))
        # LOW: \\multicolumn spanning >=2 columns with only non-wrapping spec
        #      (l/c/r, no p/m/b/X). A {1}{c} single-cell merge is just an alignment
        #      idiom, NOT an overflow risk -- skip it (it was pure noise).
        for m in MULTICOLUMN.finditer(line):
            n_cols = int(m.group(1))
            spec = m.group(2)
            letters = re.findall(r"[a-zA-Z]", spec)
            wrapping = any(ch in letters for ch in "pmbXwP")
            if n_cols >= 2 and letters and not wrapping:
                findings.append((rel, ln, "LOW",
                                 f"\\multicolumn{{{n_cols}}}{{{spec}}} spans {n_cols} cols "
                                 f"with no wrapping column (l/c/r only) -- long content "
                                 f"overflows silently; use p{{}}"))


def main() -> int:
    root = sys.argv[1] if len(sys.argv) > 1 else ROOT
    findings: list[tuple] = []
    tex_files = []
    for dirpath, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in ("archive", "snapshots", "releases")]
        for f in files:
            if f.endswith(".tex"):
                tex_files.append(os.path.join(dirpath, f))
    for p in sorted(tex_files):
        lint_file(p, os.path.relpath(p, root), findings)

    # INFO: tight_layout in figure scripts
    tight_hits = []
    for base in (os.path.join(root, "code"), os.path.join(root, "figures")):
        if not os.path.isdir(base):
            continue
        for dp, _, fs in os.walk(base):
            for f in fs:
                if f.endswith(".py"):
                    fp = os.path.join(dp, f)
                    if TIGHT_LAYOUT.search(open(fp, encoding="utf-8", errors="ignore").read()):
                        tight_hits.append(os.path.relpath(fp, root))

    order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "INFO": 3}
    findings.sort(key=lambda t: (order[t[2]], t[0], t[1]))
    for rel, ln, sev, msg in findings:
        print(f"{rel}:{ln}  [{sev}] {msg}")
    for h in tight_hits:
        print(f"{h}  [INFO] uses tight_layout() -- open the generated figure to "
              f"confirm aspect ratio / no silent canvas expansion")

    counts = {s: sum(1 for t in findings if t[2] == s) for s in order}
    print(f"\n{len(findings)} latex findings  "
          f"(HIGH={counts['HIGH']} MEDIUM={counts['MEDIUM']} "
          f"LOW={counts['LOW']}); tight_layout scripts: {len(tight_hits)}")
    return 1 if counts["HIGH"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
