# -*- coding: utf-8 -*-
"""Sync paper1's submission package to a frozen release closure, then rebuild the
package MANIFEST. Idempotent; safe to re-run after every freeze.

Usage:  <system python> code/sync_package.py [release-dir-name]
        default release dir name: the newest 2026-*-s1-corrected-r* under releases/

Why a script and not a one-off: this has been needed after every narrative revision
(r17 -> r18 -> r20 -> r21). It also *verifies* the copy (missing / extra / hash
mismatch), which a plain copy would not.
"""
import io, os, shutil, hashlib, sys, glob

PAPER1 = os.environ.get("UNAGI_PAPER1_DIR", r"F:/UNAGI_Project/paper1")
ROOT = PAPER1 + "/paper1_kc_diagnostic"
PKG = PAPER1 + "/paper1_SUBMISSION_PACKAGE_20260919"
DST = PKG + "/01_manuscript"

INTERMEDIATE = (".aux", ".log", ".out", ".toc", ".bbl", ".blg", ".spl", ".fls", ".fdb_latexmk")

def newest_release():
    dirs = glob.glob(ROOT + "/releases/*s1-corrected-r*")
    def rnum(d):
        try:
            return int(os.path.basename(d).rsplit("r", 1)[1])
        except Exception:
            return -1
    return sorted(dirs, key=rnum)[-1]

rel = sys.argv[1] if len(sys.argv) > 1 else newest_release()
SRC = os.path.join(rel, "source") if os.path.isdir(os.path.join(rel, "source")) else rel
print("release :", os.path.basename(rel))
print("source  :", SRC)
assert os.path.isdir(SRC), "no source dir"
assert os.path.isfile(os.path.join(SRC, "main.tex")), "no main.tex in source"

def dig(p):
    return hashlib.sha256(io.open(p, "rb").read()).hexdigest()

# Sync WITHOUT a bulk directory delete: `shutil.rmtree` on a 70-file dir trips
# [safe-delete] SAFE_DELETE_BULK_CONFIRM_REQUIRED (>50 files). Overwrite in place
# and remove only the few stale files individually.
#
# Intermediates are skipped on the way IN as well as deleted on the way OUT.  The
# frozen snapshots carry their own build leftovers (main.aux/.log/.out,
# supplementary.toc, ...), so copying them and then deleting them made this script
# report a constant "7 copied, 7 removed" on every single run -- which reads like
# the sync is not converging when in fact it is.
copied, removed, skipped_int = 0, 0, 0
for dp, _d, fs in os.walk(SRC):
    rel_dir = os.path.relpath(dp, SRC)
    dst_dir = DST if rel_dir == "." else os.path.join(DST, rel_dir)
    os.makedirs(dst_dir, exist_ok=True)
    for f in fs:
        if f.endswith(INTERMEDIATE):
            skipped_int += 1
            continue
        src_f, dst_f = os.path.join(dp, f), os.path.join(dst_dir, f)
        if os.path.isfile(dst_f) and dig(dst_f) == dig(src_f):
            continue
        shutil.copy2(src_f, dst_f)
        copied += 1

# Remove DST files the frozen snapshot does not have (a revision can drop a
# figure or a section file; leaving it behind would silently keep it alive).
for dp, _d, fs in os.walk(DST):
    for f in fs:
        rel = os.path.relpath(os.path.join(dp, f), DST)
        if not os.path.isfile(os.path.join(SRC, rel)):
            os.remove(os.path.join(dp, f))
            removed += 1
print("synced: %d copied, %d stale removed, %d intermediates not copied"
      % (copied, removed, skipped_int))

removed = 0
for dp, _d, fs in os.walk(DST):
    for f in fs:
        if f.endswith(INTERMEDIATE):
            os.remove(os.path.join(dp, f))
            removed += 1
print("intermediates removed:", removed)

def index(root):
    out = {}
    for dp, _d, fs in os.walk(root):
        for f in fs:
            if f.endswith(INTERMEDIATE):
                continue
            p = os.path.join(dp, f)
            out[os.path.relpath(p, root).replace("\\", "/")] = dig(p)
    return out

a, b = index(SRC), index(DST)
missing, extra = sorted(set(a) - set(b)), sorted(set(b) - set(a))
mismatch = [k for k in sorted(set(a) & set(b)) if a[k] != b[k]]
print("closure files src=%d dst=%d | missing=%s extra=%s mismatch=%s"
      % (len(a), len(b), missing or "0", extra or "0", mismatch or "0"))

# Caches must never enter the integrity record: their contents change on every
# test run, so the MANIFEST becomes non-reproducible, and a `.pytest_cache/` in
# the delivered package is simply wrong.  Skipping them *here* -- rather than
# only cleaning the tree by hand -- is what stops this from coming back.
CACHE_DIRS = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".git"}
CACHE_SUFFIX = (".pyc", ".pyo")

entries, skipped = [], []
for dp, dirs, fs in os.walk(PKG):
    skipped += [os.path.join(dp, d) for d in dirs if d in CACHE_DIRS]
    dirs[:] = [d for d in dirs if d not in CACHE_DIRS]
    for f in sorted(fs):
        if f == "MANIFEST.sha256" or f.endswith(CACHE_SUFFIX):
            continue
        p = os.path.join(dp, f)
        entries.append((dig(p), os.path.getsize(p),
                        os.path.relpath(p, PKG).replace("\\", "/")))
entries.sort(key=lambda t: t[2])
if skipped:
    rel_skip = sorted(os.path.relpath(s, PKG).replace("\\", "/") for s in skipped)
    print("cache dirs skipped (not indexed): %d | %s%s"
          % (len(rel_skip), ", ".join(rel_skip[:5]),
             " ..." if len(rel_skip) > 5 else ""))
io.open(PKG + "/MANIFEST.sha256", "w", encoding="utf-8", newline="\n").write(
    "\n".join("%s  %10d  %s" % t for t in entries) + "\n")

# page counts straight from the copied PDFs
for n in ("main", "supplementary"):
    fp = os.path.join(DST, n + ".pdf")
    if os.path.isfile(fp):
        d = io.open(fp, "rb").read()
        print("  package %-14s %d pages" % (n, d.count(b"/Type /Page") - d.count(b"/Type /Pages")))

print("MANIFEST.sha256 rebuilt: %d entries | %.1f MB"
      % (len(entries), sum(s for _h, s, _r in entries) / 1024 / 1024))
