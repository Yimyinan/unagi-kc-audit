"""
check_text_integrity.py
=======================
Static guards for the two failure modes that have actually corrupted this
manuscript, both caused by editing LaTeX source through a shell:

  1. SHELL EXPANSION. Text passed through a bash double-quoted `python -c`
     string has `$VAR` expanded. `$0.5$` became `/usr/bin/bash.5$` and reached
     main.pdf before it was caught. Tokens like `$0`--`$9`, `${`, `$@`, `$*`
     are the dangerous ones.

  2. UNBALANCED MATH DELIMITERS. A single lost `$` silently swallows the rest
     of a line into math mode. Odd count of unescaped `$` on a non-comment line
     is the tell.

Also flags non-printable control characters (the other recurring corruption:
a literal TAB or CR written into a LaTeX body).

Usage
-----
    python tools/check_text_integrity.py            # scan the standard file set
    python tools/check_text_integrity.py --strict   # exit 1 on any finding

Exit code 0 = clean, 1 = findings (only with --strict).
"""
import io
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

FILES = [
    "main.tex",
    "star_methods.tex",
    "supplementary.tex",
    "supp_addendum.tex",
    "latent_residence_conclusion.tex",
    "submission_metadata.tex",
    "table_s1_audit.tex",
]
FILES = [f for f in FILES if os.path.exists(os.path.join(ROOT, f))]
FILES += [os.path.join("supplementary", f)
          for f in sorted(os.listdir(os.path.join(ROOT, "supplementary")))
          if f.endswith(".tex")] if os.path.isdir(os.path.join(ROOT, "supplementary")) else []

SHELL_TOKENS = re.compile(r"/usr/bin/|/bin/(?:ba)?sh|\$\{[A-Za-z_]")
# `$` handling: legitimate LaTeX is full of `$2.3$` and multi-line display math,
# so a naive odd-count test is all false positives. Instead strip every balanced
# math construct and see what `$` survives.
MATH_ENVS = re.compile(
    r"\\begin\{(?:equation|align|gather|multline|eqnarray)\*?\}.*?"
    r"\\end\{(?:equation|align|gather|multline|eqnarray)\*?\}", re.S)
DISPLAY = re.compile(r"\\\[.*?\\\]", re.S)
INLINE = re.compile(r"(?<!\\)\$(?:[^$]|\$(?!\$))*?(?<!\\)\$", re.S)


def unbalanced_math(s):
    """Return (line, context) for `$` that no pairing rule can account for."""
    stripped = MATH_ENVS.sub(lambda m: "\n" * m.group(0).count("\n"), s)
    stripped = DISPLAY.sub(lambda m: "\n" * m.group(0).count("\n"), stripped)
    prev = None
    while prev != stripped:
        prev = stripped
        stripped = INLINE.sub(lambda m: "\n" * m.group(0).count("\n"), stripped)
    out = []
    for i, line in enumerate(stripped.split("\n"), 1):
        if line.lstrip().startswith("%"):
            continue
        n = len(re.findall(r"(?<!\\)\$", line))
        if n % 2 == 1:
            out.append((i, line.strip()[:90]))
    return out


def scan(path):
    s = io.open(path, encoding="utf-8").read()
    findings = []

    # 1. shell-expansion tokens
    for i, line in enumerate(s.split("\n"), 1):
        if line.lstrip().startswith("%"):
            continue
        m = SHELL_TOKENS.search(line)
        if m:
            findings.append(("SHELL", i, m.group(0), line.strip()[:90]))

    # 2. math delimiters that no pairing rule can account for
    for line, text in unbalanced_math(s):
        findings.append(("MATH", line, "unpaired $", text))

    # 3. control characters
    bad = sorted({hex(ord(c)) for c in s if ord(c) < 32 and c not in "\n"})
    if bad:
        findings.append(("CTRL", 0, " ".join(bad), "non-printable characters present"))

    return findings


def main():
    strict = "--strict" in sys.argv
    total = 0
    for rel in FILES:
        p = os.path.join(ROOT, rel)
        if not os.path.exists(p):
            continue
        f = scan(p)
        total += len(f)
        tag = "OK  " if not f else "FAIL"
        print("%s  %-44s %d finding(s)" % (tag, rel, len(f)))
        for kind, line, detail, text in f:
            print("      [%s] L%-5d %-22s %s" % (kind, line, detail, text))
    print("\n%s: %d finding(s) across %d file(s)"
          % ("CLEAN" if total == 0 else "FINDINGS", total, len(FILES)))
    return 1 if (strict and total) else 0


if __name__ == "__main__":
    sys.exit(main())
