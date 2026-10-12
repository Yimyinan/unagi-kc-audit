# -*- coding: utf-8 -*-
"""Measure translatable volume for the full paper1 r23 (main + SI + addendum)."""
import io, os, re

ROOT = r"F:/UNAGI_Project/paper1/paper1_kc_diagnostic"
EXCLUDE = ("snapshots", "archive", "releases", "build", "dist", "__pycache__",
           "_archive_20260919", "_unused")

def rd(f):
    return io.open(os.path.join(ROOT, f), encoding="utf-8-sig", errors="replace").read()

def strip(t):
    t = re.sub(r"\\begin\{thebibliography\}.*?\\end\{thebibliography\}", " ", t, flags=re.S)
    t = re.sub(r"\\bibitem(?:\[[^\]]*\])?\{[^}]*\}.*?(?=\\bibitem|\\end\{thebibliography\}|\Z)", " ", t, flags=re.S)
    t = re.sub(r"%[^\n]*", " ", t)
    t = re.sub(r"\\(?:cite|ref|label|input|include|includegraphics|includepdf)\*?(?:\[[^\]]*\])*\{[^}]*\}", " ", t)
    t = re.sub(r"\\[a-zA-Z]+\*?", " ", t)
    t = re.sub(r"[{}$&\\~^_]", " ", t)
    return t

def wc(t):
    return len(strip(t).split())

def closure(entry):
    seen, stack = set(), [entry]
    while stack:
        f = stack.pop()
        if f in seen:
            continue
        seen.add(f)
        fp = os.path.join(ROOT, f)
        if not os.path.isfile(fp):
            continue
        for m in re.finditer(r"\\(?:input|include)\{([^}]*)\}", rd(f)):
            t = m.group(1).strip()
            if not t.endswith(".tex"):
                t += ".tex"
            stack.append(t)
    return sorted(x for x in seen if not any(s in x.split("/") for s in EXCLUDE))

print("=" * 78)
print("MAIN.tex closure")
print("=" * 78)
tot = 0
for f in closure("main.tex"):
    n = wc(rd(f)) if not f.endswith("main.tex") else None
    if n is None:
        t = rd(f)
        t = t[t.find(r"\begin{abstract}"):t.find(r"\begin{thebibliography}")]
        n = wc(t)
    print("  %-34s %6d words" % (f, n))
    tot += n
print("  %-34s %6d words" % ("MAIN TOTAL", tot))

print()
print("=" * 78)
print("SUPPLEMENTARY.tex closure")
print("=" * 78)
s_tot = 0
for f in closure("supplementary.tex"):
    n = wc(rd(f))
    print("  %-46s %6d words" % (f, n))
    s_tot += n
print("  %-46s %6d words" % ("SI TEXT TOTAL", s_tot))

add = "supp_addendum.tex"
a_n = wc(rd(add))
print()
print("  %-46s %6d words" % ("supp_addendum.tex (separate PDF)", a_n))

print()
print("GRAND TOTAL translatable: %d words" % (tot + s_tot + a_n))
print()
print("=== SI structure inventory ===")
sup = rd("supplementary.tex")
print("  includepdf  :", len(re.findall(r"\\includepdf", sup)))
print("  includegraphics :", len(re.findall(r"\\includegraphics", sup)))
print("  figure envs :", len(re.findall(r"\\begin\{figure\}", sup)))
print("  table envs  :", len(re.findall(r"\\begin\{table\}", sup)))
print("  captions    :", len(re.findall(r"\\caption\*?\{", sup)))
print("  input files :", len(re.findall(r"\\input\{", sup)))
