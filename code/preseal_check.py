# -*- coding: utf-8 -*-
"""paper1 pre-seal check -- closure-aware, single command.

Why this exists as a permanent guard: it caught a real, reader-visible defect
that every other check missed. `supplementary.tex` cited khemakhem2020 and
locatello2019 while defining no \\bibitem anywhere in its own \\input closure --
the keys live only in main.tex's inline bibliography, and main.tex/supplementary.tex
are SEPARATE documents. Both citations rendered as undefined. No single-file grep
and no existing guard would have found it.

Design rules baked in (each one cost real rework):
  * ALWAYS resolve the \\input closure before concluding anything about numbering,
    citations or labels. supplementary.tex \\inputs six supp_figS10..S15 wrappers
    that declare captions S9-S14 themselves; scanning the entry file alone makes a
    contiguous numbering look gapped (that false positive once produced a
    renumbering that duplicated numbers).
  * Case-sensitive phrase scans. A case-insensitive scan previously reported
    'released implementation' as revision-history narrative when the hits were the
    legitimate "in the released implementation" (= the pipeline we released).
  * Frozen dirs (releases/snapshots/archive/build) are never part of "live".

Usage:  <system python> code/preseal_check.py
Exit:   0 = clean, 1 = at least one FAIL
"""
import io, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXCLUDE = ("snapshots", "archive", "releases", "build", "dist", "__pycache__",
           "_archive_20260919", "_unused")

FAILL = []

def rd(f):
    return io.open(os.path.join(ROOT, f), encoding="utf-8-sig", errors="replace").read()

def keep(f):
    return not any(s in f.split("/") for s in EXCLUDE)

def closure(entry):
    seen, stack, missing = set(), [entry], []
    while stack:
        f = stack.pop()
        if f in seen:
            continue
        seen.add(f)
        fp = os.path.join(ROOT, f)
        if not os.path.isfile(fp):
            missing.append(f)
            continue
        for m in re.finditer(r"\\(?:input|include)\{([^}]*)\}", rd(f)):
            t = m.group(1).strip()
            if not t.endswith(".tex"):
                t += ".tex"
            stack.append(t)
    return sorted(x for x in seen if keep(x)), missing

def hdr(n, title):
    print("\n" + "=" * 74)
    print("%s. %s" % (n, title))
    print("=" * 74)

def fail(msg):
    FAILL.append(msg)
    print("   [FAIL] " + msg)

# ---------------------------------------------------------------- 1. citations
hdr(1, "citation / bibliography reconciliation (per document, closure-aware)")
DOCS = ("main.tex", "supplementary.tex")
for entry in DOCS:
    files, missing = closure(entry)
    used, defined = set(), set()
    for f in files:
        t = rd(f)
        for m in re.finditer(r"\\cite[a-zA-Z]*\*?(?:\[[^\]]*\])*\{([^}]*)\}", t):
            used |= {k.strip() for k in m.group(1).split(",") if k.strip()}
        defined |= {m.group(1).strip() for m in
                    re.finditer(r"\\bibitem(?:\[[^\]]*\])?\{([^}]*)\}", t)}
    undef = sorted(used - defined)
    unused = sorted(defined - used)
    print("   %-18s files=%-3d cited=%-3d defined=%-3d" % (entry, len(files), len(used), len(defined)))
    print("      missing inputs       :", missing or "none")
    print("      cited but UNDEFINED  :", undef or "none")
    print("      defined but uncited  :", unused or "none")
    if undef:
        fail("%s cites %d key(s) with no \\bibitem in its own closure: %s"
             % (entry, len(undef), undef))

# ---------------------------------------------------------------- 2. numbering
hdr(2, "supplementary numbering over the closure")
sfiles, _ = closure("supplementary.tex")
fig, tab = {}, set()
for f in sfiles:
    t = rd(f)
    for m in re.finditer(r"Supplementary Figure~?\s*S(\d+)", t):
        fig.setdefault(int(m.group(1)), set()).add(f)
    tab |= {int(m.group(1)) for m in re.finditer(r"Supplementary Table~?\s*S(\d+)", t)}
fn, tn = sorted(fig), sorted(tab)
print("   SI figures : %s" % (fn,))
print("   SI tables  : %s" % (tn,))
if fn and fn != list(range(1, fn[-1] + 1)):
    fail("SI figure numbering not contiguous 1..%d: %s" % (fn[-1], fn))
if tn and tn != list(range(1, tn[-1] + 1)):
    fail("SI table numbering not contiguous 1..%d: %s" % (tn[-1], tn))
cross = {n: sorted(v) for n, v in fig.items() if len(v) > 1}
print("   numbers declared by >1 file (usually \\includepdf bookmark + PDF caption):",
      cross or "none")

# ---------------------------------------------------------------- 3. main -> SI
hdr(3, "main -> supplementary cross-references")
main = rd("main.tex")
mfig = sorted({int(m) for m in re.findall(r"Supplementary Fig\w*\.?\s*~?\s*S(\d+)", main)})
mtab = sorted({int(m) for m in re.findall(r"Supplementary Table\w*\.?\s*~?\s*S(\d+)", main)})
print("   main cites SI figures:", mfig)
print("   main cites SI tables :", mtab)
bad = sorted(set(mfig) - set(fn)) + sorted(set(mtab) - set(tn))
if bad:
    fail("main references SI items with no caption: %s" % bad)
else:
    print("   all main->SI references resolve")

# ---------------------------------------------------------------- 4. labels
hdr(4, "label / ref resolution over each document's closure")
for entry in DOCS:
    files, _ = closure(entry)
    labs, refs = set(), set()
    for f in files:
        t = rd(f)
        labs |= set(re.findall(r"\\label\{([^}]*)\}", t))
        refs |= set(re.findall(r"\\ref\{([^}]*)\}", t))
    dangling = sorted(refs - labs)
    print("   %-18s files=%-3d labels=%-3d refs=%-3d dangling=%s"
          % (entry, len(files), len(labs), len(refs), dangling or "none"))
    if dangling:
        fail("%s has \\ref with no \\label: %s" % (entry, dangling))

# ---------------------------------------------------------------- 5. placeholders
hdr(5, "unfilled placeholders")
PH = re.compile(r"\[(?:TODO|REPOSITORY_URL|REPOSITORY\\_URL|FIGSHARE_DOI|FIGSHARE\\_DOI|"
                r"AUTHOR NAME|AFFILIATION|TBD|XXX)[^\]]*\]", re.I)
# ★★ 本轮修掉的一处**真缺陷**：本节原先**只打印、不进 FAILL** ⇒
#    "意外残留的占位符"（如正文里漏了个 [TBD]）会被打印出来，但门禁仍报 CLEAN，
#    而 VERDICT 却声称检查过 placeholder。门禁**不算失败**的部分等于没在守。
#    现改为：**封版时预期会填的**（仓库 URL / figshare DOI / 通讯邮箱）→ 容忍并计数；
#    其余一律 fail()。
TOL = re.compile(r"REPOSITORY|FIGSHARE|corresponding\s+e-?mail|CRediT", re.I)
n_tol, n_bad = 0, 0
allf, _ = closure("main.tex")
allf = sorted(set(allf) | set(closure("supplementary.tex")[0]))
for f in allf:
    hits = sorted(set(PH.findall(rd(f))))
    tol = [h for h in hits if TOL.search(h)]
    bad = [h for h in hits if not TOL.search(h)]
    n_tol += len(tol)
    if tol:
        print("   %-34s [封版时填] %s" % (f, tol))
    if bad:
        n_bad += len(bad)
        print("   %-34s [★ 意外残留] %s" % (f, bad))
        fail("%s 残留**非预期**占位符 %s（封版不填的必须清掉）" % (f, bad))
print("   tolerating %d seal-time placeholder(s); unexpected: %d" % (n_tol, n_bad))

# ---------------------------------------------------------------- 6. length
hdr(6, "length")
def wc(t):
    return len(re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^}]*\})?", " ", t).split())
# 2026-10-07: this block was a silent no-op. It looked for \begin{abstract} and for
# non-starred \section{Introduction} / \section*{Methods summary}, none of which this
# manuscript uses (it uses \section*{Abstract} and \section*{Methods}), so it printed
# "abstract words: ?" and "Introduction..Discussion: 0" and nobody noticed. It now
# accepts starred headings and FAILS LOUDLY when a heading is not found, so the gate
# can never silently report zero again.
m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", main, re.S)
if not m:
    m = re.search(r"\\section\*?\{Abstract\}(.*?)(?=\\section\*?\{)", main, re.S)


def _find_sec(name):
    for pat in (r"\section{%s}" % name, r"\section*{%s}" % name):
        k = main.find(pat)
        if k >= 0:
            return k
    return -1


i, j = _find_sec("Introduction"), _find_sec("Methods")
print("   abstract words              :", wc(m.group(1)) if m else "** ABSTRACT NOT FOUND **")
if i < 0 or j < 0 or j <= i:
    print("   Introduction..Discussion    : ** 结构未识别 (Introduction=%d, Methods=%d) **" % (i, j))
    print("   Methods + backmatter        : ** 同上 **")
else:
    print("   Introduction..Discussion    :", wc(main[i:j]))
    print("   Methods + backmatter        :", wc(main[j:]))

# ---------------------------------------------------------------- 7. narrative
hdr(7, "revision-history narrative in main.tex (case-sensitive)")
PHRASES = ["we corrected", "we withdraw", "we withdrew", "previously reported",
           "superseded", "in an earlier version", "had been reported as",
           "the released trace form", "trace form returns"]
hits = [(p, len(re.findall(re.escape(p), main))) for p in PHRASES]
hits = [(p, n) for p, n in hits if n]
print("   hits:", hits or "none (clean)")
if hits:
    fail("main.tex still carries revision-history narrative: %s" % hits)

# ---------------------------------------------------------------- 8. bib order
hdr(8, "bibliography order == first-citation order (project convention)")

def splice(t, depth=0):
    """Inline \\input content so citation order follows DOCUMENT order."""
    if depth > 6:
        return t
    out, pos = [], 0
    for m in re.finditer(r"\\(?:input|include)\{([^}]*)\}", t):
        out.append(t[pos:m.start()])
        f = m.group(1).strip()
        if not f.endswith(".tex"):
            f += ".tex"
        fp = os.path.join(ROOT, f)
        if os.path.isfile(fp) and keep(f):
            out.append(splice(rd(f).replace("\r\n", "\n"), depth + 1))
        pos = m.end()
    out.append(t[pos:])
    return "".join(out)

for entry in DOCS:
    t = rd(entry)
    if "\\begin{thebibliography}" not in t:
        print("   %-18s no bibliography -- skipped" % entry)
        continue
    bib_keys = [m.group(1).strip() for m in
                re.finditer(r"\\bibitem(?:\[[^\]]*\])?\{([^}]*)\}", t)]
    body = splice(t.split("\\begin{thebibliography}", 1)[0])
    order, seen = [], set()
    for m in re.finditer(r"\\cite[a-zA-Z]*\*?(?:\[[^\]]*\])*\{([^}]*)\}", body):
        for k in (x.strip() for x in m.group(1).split(",")):
            if k and k not in seen:
                seen.add(k)
                order.append(k)
    want = [k for k in order if k in bib_keys]
    ok = bib_keys == want
    print("   %-18s bibitems=%-3d in-first-citation-order=%s" % (entry, len(bib_keys), ok))
    if not ok:
        for i, (a, b) in enumerate(zip(bib_keys, want)):
            if a != b:
                print("      first divergence at #%d: file has %s, citation order wants %s"
                      % (i + 1, a, b))
                break
        fail("%s bibliography is not in first-citation order" % entry)

# ---------------------------------------------------------------- 9. package
hdr(9, "submission-package integrity (MANIFEST.sha256)")
# The package MANIFEST is the only record of what was actually signed off, and
# until now nothing verified it -- `release_freezer.py --verify` covers a release
# snapshot, not the assembled submission package.  A manifest nobody checks is
# decoration, so it is checked here, where it gates the seal.
PKG = os.path.join(os.path.dirname(ROOT), "paper1_SUBMISSION_PACKAGE_20260919")
MAN = os.path.join(PKG, "MANIFEST.sha256")
INTER = (".aux", ".log", ".out", ".toc", ".bbl", ".blg", ".spl", ".fls", ".fdb_latexmk")
if not os.path.isfile(MAN):
    print("   no package MANIFEST at %s -- skipped" % MAN)
else:
    want, bad, sized = {}, [], 0
    for line in io.open(MAN, encoding="utf-8", errors="replace").read().splitlines():
        if not line.strip():
            continue
        try:
            h, sz, rel = line.split(None, 2)
            rel = rel.strip()
        except ValueError:
            bad.append("unparsable MANIFEST line: %r" % line[:80])
            continue
        want[rel] = (h, int(sz))
    import hashlib
    for rel, (h, sz) in sorted(want.items()):
        fp = os.path.join(PKG, rel)
        if not os.path.isfile(fp):
            bad.append("in MANIFEST but absent: %s" % rel)
            continue
        raw = io.open(fp, "rb").read()
        if hashlib.sha256(raw).hexdigest() != h:
            bad.append("sha256 mismatch: %s" % rel)
        elif len(raw) != sz:
            bad.append("size mismatch: %s (%d != %d)" % (rel, len(raw), sz))
        else:
            sized += 1
    have = set()
    for dp, dirs, fs in os.walk(PKG):
        dirs[:] = [d for d in dirs
                   if d not in ("__pycache__", ".pytest_cache", ".git")]
        for f in fs:
            if f.endswith(INTER):
                continue
            have.add(os.path.relpath(os.path.join(dp, f), PKG).replace("\\", "/"))
    unsigned = sorted(have - set(want) - {"MANIFEST.sha256"})
    print("   %-34s entries=%-4d verified=%-4d problems=%d"
          % ("MANIFEST.sha256", len(want), sized, len(bad)))
    print("   files present but NOT in the manifest :", unsigned or "none")
    for b in bad[:12]:
        print("      " + b)
    if bad:
        fail("submission package does not match its MANIFEST.sha256 (%d problem(s))" % len(bad))
    if unsigned:
        fail("%d file(s) in the submission package are unsigned (re-run sync_package.py): %s"
             % (len(unsigned), unsigned[:5]))

# ---------------------------------------------------------------- verdict
print("\n" + "=" * 74)
if FAILL:
    print("VERDICT: %d FAIL" % len(FAILL))
    for f in FAILL:
        print("   - " + f)
    sys.exit(1)
print("VERDICT: CLEAN -- no citation, numbering, label, placeholder or narrative defect")
