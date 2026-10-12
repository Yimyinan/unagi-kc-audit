# -*- coding: utf-8 -*-
"""Author-line update:  Yim Yinan  ->  Yan Yinan ,  and the author separator loses "and".

Scope: LIVE files only. archive/ , releases/ , snapshots/ , _archive/ and the existing
submission package are immutable copies -- the package is re-synced from a new release instead.

Separator choice per language convention:
  * English manuscript : `Yan Yinan$^{1}$, Lan Wang$^{1,*}$`   (ASCII comma, standard English)
  * Chinese documents  : `Yan Yinan，Lan Wang`                 (full-width comma)
The phrase "Lead contact and corresponding author" is left alone -- that "and" joins two roles,
not two authors.
"""
import io, os, re

BASE = r"F:/UNAGI_Project/paper1/paper1_kc_diagnostic"
EXCL = ("/archive/", "/releases/", "/snapshots/", "/build/", "_archive", "SUBMISSION_PACKAGE",
        "/zh_tex/", "paper1_zh_正文.md", "paper1_zh_补充材料.md")

EDITS = [
    # (file, old, new, label)
    ("main.tex",
     r"\author{Yim Yinan$^{1}$ and Lan Wang$^{1,*}$}",
     r"\author{Yan Yinan$^{1}$, Lan Wang$^{1,*}$}",
     "EN manuscript author line"),
    ("PATTERNS_SUBMISSION.md",
     "Yim Yinan (on behalf of the authors)",
     "Yan Yinan (on behalf of the authors)",
     "patterns submission doc"),
    ("submission/cover_letter_patterns.md",
     "on behalf of the authors: Yim Yinan (first author), Lan Wang",
     "on behalf of the authors: Yan Yinan (first author), Lan Wang",
     "cover letter (md)"),
    ("submission/cover_letter_patterns.md",
     "- **Authors.** Title page carries Yim Yinan (first author) and Lan Wang",
     "- **Authors.** Title page carries Yan Yinan (first author), Lan Wang",
     "cover letter authors note (md)"),
    ("submission/cover_letter_patterns.tex",
     "on behalf of the authors: Yim Yinan (first author), Lan Wang",
     "on behalf of the authors: Yan Yinan (first author), Lan Wang",
     "cover letter (tex)"),
    ("zh_translation/01_frontmatter.md",
     "Yim Yinan$^{1}$、Lan Wang$^{1,*}$",
     "Yan Yinan，Lan Wang",
     "ZH front matter authors"),
    ("zh_translation/build_tex.py",
     "作者： & Yim Yinan；Lan Wang \\\\",
     "作者： & Yan Yinan，Lan Wang \\\\",
     "ZH LaTeX cover authors"),
]

print("%-38s %s" % ("file", "result"))
for rel, old, new, label in EDITS:
    p = os.path.join(BASE, rel)
    if not os.path.isfile(p):
        print("%-38s MISSING" % label)
        continue
    t = io.open(p, encoding="utf-8").read()
    n = t.count(old)
    if n == 0:
        print("%-38s not found (already updated?)" % label)
        continue
    t = t.replace(old, new)
    io.open(p, "w", encoding="utf-8", newline="\n").write(t) if p.endswith(".py") \
        else io.open(p, "w", encoding="utf-8").write(t)
    print("%-38s %d replacement(s)" % (label, n))

print()
print("=== residual 'Yim' in live files ===")
hits = 0
for dp, dirs, fs in os.walk(BASE):
    rel = dp.replace("\\", "/")
    if any(x.strip("/") in rel for x in ("archive", "releases", "snapshots", "build", "_archive")):
        continue
    for f in fs:
        if not f.endswith((".tex", ".md", ".py")):
            continue
        p = os.path.join(dp, f)
        try:
            t = io.open(p, encoding="utf-8", errors="ignore").read()
        except Exception:
            continue
        if "Yim" in t:
            for i, l in enumerate(t.split("\n"), 1):
                if "Yim" in l:
                    print("   %s:%d  %s" % (os.path.relpath(p, BASE), i, l.strip()[:110]))
                    hits += 1
print("   (none)" if not hits else "   total: %d" % hits)
