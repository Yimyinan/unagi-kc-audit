"""Submission gate: fail if unfilled metadata placeholders remain.

WHY
---
main.tex and supplementary.tex previously carried literal ``[REPOSITORY_URL]``
and ``[ZENODO_DOI]`` placeholders.  They are now centralised in
``submission_metadata.tex`` so the two documents can never disagree, but a
centralised placeholder is still a placeholder: it is easy to compile a PDF,
skim the availability section, and submit with the sentinel still in it.

This script is the tripwire.  It scans every ``.tex`` source in the project and
exits non-zero if any unfilled sentinel survives, printing exactly where.

IMPORTANT -- why there is NO exemption list
-------------------------------------------
An earlier version exempted ``submission_metadata.tex`` itself ("that file is
*supposed* to contain the sentinels").  That made the gate a tautology: the only
file that could ever hold a sentinel was excluded from the scan, so the gate
reported CLEAN while both manuscripts were still unfilled.  It could never fail.

The correct invariant is simpler and falsifiable: **once the values are filled in,
no LIVE ``.tex`` source contains a sentinel** -- the definition file included,
because filling it *replaces* the sentinel.  So the definition file is scanned
like any other, and the gate fires exactly when it should.

Frozen historical copies are excluded (not "exempted").  ``snapshots/``,
``archive/`` and ``releases/`` hold immutable snapshots cut *before* the values
were known; they will always contain the sentinels, and they are not submission
sources.  Scanning them (the original behaviour) made the gate unable to ever
pass once the first snapshot existed -- a false blocker.  Excluding them does NOT
reintroduce the tautology above, because the LIVE definition file is still
scanned: if ``submission_metadata.tex`` is unfilled, the gate still fires.

USAGE
-----
    python tools/check_submission_metadata.py            # report only
    python tools/check_submission_metadata.py --strict   # exit 1 if unfilled

Exit codes: 0 = clean (or non-strict), 1 = unfilled placeholders found.
"""
from __future__ import annotations

import argparse
import io
import os
import re
import sys

# <project>/paper1/paper1_kc_diagnostic/tools/this_file -> walk up to <project>
# so that BOTH paper1_kc_diagnostic/ and paper2/ are covered by one gate.
_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
DEFINITION_FILE = "submission_metadata.tex"

# sentinels that mean "not filled in yet"
SENTINELS = ["[REPOSITORY\\_URL]", "[REPOSITORY_URL]",
             "[ZENODO\\_DOI]", "[ZENODO_DOI]",
             "[AUTHOR NAME]", "[AFFILIATION]"]

# Directory segments that are never submission sources.  ``snapshots``, ``archive``
# and ``releases`` are immutable frozen copies cut before the values were known;
# they will always carry the sentinels, so scanning them is a permanent false
# blocker.  The LIVE definition file is still scanned (see module docstring).
EXCLUDED_DIRS = {".git", "__pycache__", "dist", "build",
                 "snapshots", "archive", "releases",
                 # quarantined material is by definition not a submission source
                 "_quarantine", "_archive_20260919", "_archive_20261008",
                 # workspace noise: dependency trees and scratch dirs are NOT our metadata
                 ".venvs", ".venv", "site-packages", "node_modules",
                 "_audit_tmp2", "_tmp", "_scratch_gwas", "outputs", "gears_src",
                 "si_sources", "_figview",
                 # non-submission trees that nonetheless carry stale copies of the
                 # definition file: infrastructure/backups/scratch, build byproducts,
                 # and dated point-in-time snapshots (matched by pattern below).
                 "_infra", "output"}

# A dated directory such as ``2026-09-03-18-47-34`` is a frozen point-in-time copy,
# never a submission source -- the same reason ``snapshots``/``archive`` are excluded.
# Scanning it produced permanent false positives for files that can never be filled in.
_DATED_DIR_RE = re.compile(r"^\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}$")


def _excluded(dirpath):
    segs = dirpath.replace("\\", "/").split("/")
    if any(seg in EXCLUDED_DIRS for seg in segs):
        return True
    return any(_DATED_DIR_RE.match(seg) for seg in segs)


def tex_sources(root):
    out = []
    for dirpath, _dirnames, filenames in os.walk(root):
        if _excluded(dirpath):
            continue
        for fn in filenames:
            if fn.endswith(".tex"):
                out.append(os.path.relpath(os.path.join(dirpath, fn), root))
    return sorted(out)


# --------------------------------------------------------------------------- #
# Repository metadata: the .tex gate above cannot see these files, yet they are
# what Zenodo PUBLISHES (the DOI description is immutable once released).  Added
# after an audit found `version: v1.0` against a 0.1.2 wheel, and an author name
# (`Yim`) that the manuscript had already renamed (`Yan`).
# --------------------------------------------------------------------------- #
REPO_METADATA_NAMES = {"pyproject.toml", ".zenodo.json", "CITATION.cff"}


def repo_metadata_files(root):
    out = []
    for dirpath, _dirnames, filenames in os.walk(root):
        if _excluded(dirpath):
            continue
        for fn in filenames:
            if fn in REPO_METADATA_NAMES:
                out.append(os.path.relpath(os.path.join(dirpath, fn), root))
    return sorted(out)


def license_files(root):
    """★ 第 27 轮新增：LICENSE 也要扫占位符。

    为什么单列而不是塞进 `REPO_METADATA_NAMES`：那套名字会送进 `repo_metadata_checks()`
    的一致性检查（解析 TOML/JSON/YAML、比版本与作者），LICENSE 喂进去只会误报。
    这里只做**占位符扫描**，复用同一份 SENTINELS。

    背景（真实缺陷）：全项目 **4 个 LICENSE** 都写着
    `Copyright (c) 2026 [AUTHOR NAME]`，而门禁此前**根本不看 LICENSE** ⇒ 会随 Zenodo 发布。
    """
    out = []
    for dirpath, _dirnames, filenames in os.walk(root):
        if _excluded(dirpath):
            continue
        for fn in filenames:
            if fn == "LICENSE" or fn.startswith("LICENSE."):
                out.append(os.path.relpath(os.path.join(dirpath, fn), root))
    return sorted(out)


def _norm_person(s):
    """'Yan Yinan' / 'Yan, Yinan' / family+given -> comparable token set."""
    return frozenset(t for t in re.split(r"[^A-Za-z]+", s.lower()) if t)


# The leading separator is load-bearing: without it `cff-version: 1.2.0` in
# CITATION.cff is read as the *software* version and matches nothing.
# Two-component versions must be accepted as well -- the stale `.zenodo.json`
# this check was written to catch carried `"version": "v1.0"`, and a
# three-component-only pattern silently skips exactly that case.
_VER_RE = re.compile(r'(?:^|[\s",{])version"?\s*[:=]\s*"?v?([0-9]+\.[0-9]+(?:\.[0-9]+)?)')


def _declared(rel, text):
    """(version, person-token-set) declared by one metadata file; None if absent."""
    base = os.path.basename(rel)
    m = _VER_RE.search(text)
    person = None
    if base == ".zenodo.json":
        a = re.search(r'"creators"\s*:\s*\[\s*\{\s*"name"\s*:\s*"([^"]+)"', text, re.S)
        if a:
            person = a.group(1)
    elif base == "CITATION.cff":
        fam = re.search(r'family-names:\s*"([^"]+)"', text)
        giv = re.search(r'given-names:\s*"([^"]+)"', text)
        if fam or giv:
            person = (fam.group(1) if fam else "") + " " + (giv.group(1) if giv else "")
    elif base == "pyproject.toml":
        a = re.search(r'authors\s*=\s*\[.*?name\s*=\s*"([^"]+)"', text, re.S)
        if a:
            person = a.group(1)
    return (m.group(1) if m else None), (_norm_person(person) if person else None)


def _toml_name(text):
    m = re.search(r'(?:^|\n)\s*name\s*=\s*"([^"]+)"', text)
    return m.group(1).replace("_", "-").lower() if m else None


def _parse_wheel(fn):
    """'kappa_c_audit-0.1.2-py3-none-any.whl' -> ('kappa-c-audit', '0.1.2').

    The version is the first dash-separated tag that starts with a digit *and*
    still leaves the three tag components behind it.  Splitting and taking index
    1 breaks on hyphenated distributions: ``scvi-tools-1.5.0`` would yield
    ``tools``.
    """
    if not fn.endswith(".whl"):
        return None
    parts = fn[:-4].split("-")
    for i, p in enumerate(parts):
        if i >= 1 and re.match(r"^\d", p) and len(parts) - i >= 3:
            return "-".join(parts[:i]).replace("_", "-").lower(), p
    return None


# `dist/` must be walked (that is where build output lives); everything else the
# tex scan already treats as non-source is pruned here too.
_WHEEL_PRUNE = EXCLUDED_DIRS - {"dist"}


def dist_wheels(root):
    """{dist_name: {version: relpath}} for wheels sitting directly in a ``dist/``.

    Deliberately narrow.  A first cut globbed ``*.whl`` anywhere under a release
    root and so picked up ``archive/scvi_probe/scvi_tools-1.5.0-py3-none-any.whl``
    -- a downloaded *third-party* wheel -- reporting it as "our" shipped wheel and
    inventing a version mismatch against ``version = "0.1.2"``.  Only a
    build-output directory is evidence of what this package ships.
    """
    out = {}
    for dirpath, dirnames, filenames in os.walk(root):
        if any(seg in _WHEEL_PRUNE for seg in dirpath.replace("\\", "/").split("/")):
            dirnames[:] = []
            continue
        if os.path.basename(dirpath.replace("\\", "/")) != "dist":
            continue
        for fn in filenames:
            parsed = _parse_wheel(fn)
            if not parsed:
                continue
            name, ver = parsed
            out.setdefault(name, {})[ver] = os.path.relpath(
                os.path.join(dirpath, fn), root)
    return out


def repo_metadata_checks(root):
    """Returns (sentinel_hits, consistency_problems).

    Three checks, all scoped to artefacts that actually ship:

    1. **per release root** (a directory holding a ``.zenodo.json``): the version
       and author declared there must agree with each other, with any
       ``CITATION.cff``, and with the wheel built under that root.  A sibling
       package with no wheel is ignored, so ``kappac/`` cannot create a mismatch;
    2. **stale wheels**: one ``dist/`` holding two versions of the same
       distribution is an invitation to upload the wrong one;
    3. **cross-copy agreement**: the same distribution declared in more than one
       ``pyproject.toml`` must declare the same version and author.  This is the
       silent-staleness trap -- the working tree and the assembled release drift
       apart and the published copy is whichever was uploaded first.

    Divergent ``.zenodo.json`` copies are not cross-checked (they carry no package
    name to key on); the remedy there is to keep exactly one copy.
    """
    sent_hits, problems = [], []

    texts = {}
    for rel in repo_metadata_files(root):
        try:
            texts[rel] = io.open(os.path.join(root, rel),
                                 encoding="utf-8", errors="replace").read()
        except OSError as exc:
            # ★ 第 45 轮：原先是 `pass` ⇒ 读不到的仓库元数据（.zenodo.json / CITATION.cff /
            #   pyproject.toml）**既不报占位符也不报错**，一致性检查悄悄少一份输入仍可能通过。
            #   现在记成 problems（它本来就通向失败路径），让"读不到"降级结论。
            problems.append("%s: 读不到（%s）⇒ 该文件的占位符与一致性**未被检查**" % (rel, exc))
    for rel, text in sorted(texts.items()):
        for i, line in enumerate(text.splitlines(), 1):
            for sent in SENTINELS:
                if sent in line:
                    sent_hits.append((rel, i, sent, line.strip()[:100]))

    wheels = dist_wheels(root)

    # ---- 1 & 2: per release root ----------------------------------------- #
    roots = {}
    for rel in sorted(texts):
        if os.path.basename(rel) == ".zenodo.json":
            r = os.path.dirname(os.path.join(root, rel)).replace("\\", "/")
            roots[r] = []
    for rel in sorted(texts):
        full = os.path.join(root, rel).replace("\\", "/")
        cand = [r for r in roots if full.startswith(r + "/")]
        if cand:
            roots[max(cand, key=len)].append(rel)

    for r, rels in sorted(roots.items()):
        own = {}
        for n, versions in wheels.items():
            keep = {v: p for v, p in versions.items()
                    if os.path.join(root, p).replace("\\", "/").startswith(r + "/")}
            if keep:
                own[n] = keep

        # which distribution(s) does this root describe?
        subjects = sorted({_toml_name(texts[x]) for x in rels
                           if os.path.basename(x) == "pyproject.toml"} & set(own))
        if not subjects:
            subjects = sorted(own)

        vers, names = {}, {}
        for rel in rels:
            base = os.path.basename(rel)
            if base == "pyproject.toml" and _toml_name(texts[rel]) not in subjects:
                continue                     # a sibling package with no wheel: not ours
            v, p = _declared(rel, texts[rel])
            if v:
                vers[rel] = v
            if p:
                names[rel] = p

        for n in subjects:
            want = max(own[n])               # highest build is the shipped one
            for rel, v in sorted(vers.items()):
                if v != want:
                    problems.append("%s: version %s disagrees with the wheel built "
                                    "there, %s (%s)" % (rel, v, own[n][want], want))

        for n, versions in sorted(own.items()):
            if len(versions) > 1:
                problems.append("%s: stale wheel(s) for '%s' beside the current build "
                                "-- %s" % (r, n, ", ".join(sorted(versions.values()))))

        if len(set(names.values())) > 1:
            problems.append("%s: author name mismatch %s"
                            % (r, {k: sorted(v) for k, v in sorted(names.items())}))

    # ---- 3: cross-copy agreement ----------------------------------------- #
    copies = {}
    for rel in sorted(texts):
        if os.path.basename(rel) != "pyproject.toml":
            continue
        n = _toml_name(texts[rel])
        if n:
            copies.setdefault(n, []).append((rel,) + _declared(rel, texts[rel]))
    for n, rows in sorted(copies.items()):
        if len(rows) < 2:
            continue
        if len({v for _r, v, _p in rows}) > 1:
            problems.append("%s: version declared differently across copies -- %s"
                            % (n, {r: v for r, v, _p in rows}))
        if len({p for _r, _v, p in rows if p}) > 1:
            problems.append("%s: author declared differently across copies -- %s"
                            % (n, {r: sorted(p) for r, _v, p in rows if p}))

    return sent_hits, problems


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero when placeholders remain")
    ap.add_argument("--root", default=ROOT, help="project root to scan")
    a = ap.parse_args()

    hits = []
    scanned = 0
    unreadable = []          # ★ 第 45 轮：读不到的文件（不许静默缩小扫描范围）
    for rel in tex_sources(a.root):
        path = os.path.join(a.root, rel)
        try:
            text = io.open(path, encoding="utf-8", errors="replace").read()
        except OSError:
            # ★ 第 45 轮：读不到的 .tex 原先是 `continue` ⇒ 扫描范围静默缩小、仍可能报 CLEAN。
            #   现在登记下来，并让 CLEAN 分支失效（--strict 下退 1）。
            unreadable.append(rel)
            continue
        scanned += 1
        for i, line in enumerate(text.splitlines(), 1):
            for sent in SENTINELS:
                if sent in line:
                    hits.append((rel, i, sent, line.strip()[:100]))

    repo_sent, repo_problems = repo_metadata_checks(a.root)

    # ★ 第 27 轮：LICENSE 的占位符（此前完全没被扫到）
    for rel in license_files(a.root):
        path = os.path.join(a.root, rel)
        try:
            text = io.open(path, encoding="utf-8", errors="replace").read()
        except OSError:
            unreadable.append(rel)          # ★ 第 45 轮：LICENSE 读不到同样不许静默跳过
            continue
        for i, line in enumerate(text.splitlines(), 1):
            for sent in SENTINELS:
                if sent in line:
                    repo_sent.append((rel, i, sent, line.strip()[:100]))

    if not hits and not repo_sent and not repo_problems and not unreadable:
        print("submission metadata: CLEAN -- none of the %d .tex sources under %s "
              "contains an unfilled placeholder, and the repo metadata files are "
              "placeholder-free and self-consistent." % (scanned, a.root))
        return 0

    if unreadable:
        print()
        print("submission metadata: **无法判定** -- %d 个文件读不到，扫描范围被静默缩小："
              % len(unreadable))
        for rel in unreadable[:15]:
            print("  %s" % rel)
        print("  ★ 「读不到」不等于「没问题」；先修权限/路径，再跑本门禁。")

    if hits:
        print("submission metadata: %d unfilled placeholder(s) found in .tex (scanned %d files):"
              % (len(hits), scanned))
        for rel, i, sent, snippet in hits:
            print("  %s:%d  %s" % (rel, i, sent))
            print("      %s" % snippet)
        print()
        print("Fill the values ONCE in %s (one copy per manuscript), then recompile."
              % DEFINITION_FILE)

    if repo_sent:
        print()
        print("submission metadata: %d unfilled placeholder(s) in REPO metadata "
              "(these are what Zenodo publishes; the DOI description is immutable):"
              % len(repo_sent))
        for rel, i, sent, snippet in repo_sent:
            print("  %s:%d  %s" % (rel, i, sent))
            print("      %s" % snippet)

    if repo_problems:
        print()
        print("submission metadata: %d repo-metadata CONSISTENCY problem(s):"
              % len(repo_problems))
        for p in repo_problems:
            print("  %s" % p)

    # ★★ 第 24 轮：把"非 strict 恒 exit 0"这件事**说出来**。
    #    本工具历来有"恒真"前科（见 reviews/GAP_REGISTER_2026-09-12.md：
    #    曾因豁免定义文件本身而**永远报 CLEAN**）。现在 report 模式仍是 exit 0（设计如此），
    #    但它**不是门禁**。这里显式声明，避免有人把 report 的绿当成通过。
    if not a.strict:
        print()
        print("!! REPORT MODE -- exit code is 0 by design and does NOT gate.")
        print("   To gate, re-run with:  %s --strict" % os.path.basename(__file__))

    return 1 if a.strict else 0


if __name__ == "__main__":
    sys.exit(main())
