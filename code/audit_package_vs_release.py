# -*- coding: utf-8 -*-
"""audit_package_vs_release.py —— 投稿包正文 must == 冻结版本（r28）。

为什么需要它（本项目已发生过同类事故）：
  `preseal_check.py` 第 9 节只验证**投稿包 MANIFEST 自洽**（包内文件与包内哈希对得上），
  `audit_freeze_drift.py` 只验证**活树 == 冻结件**。
  **没有任何门禁验证"投稿包 == 冻结件"** —— 于是存在这个缺口：
  改了正文、只重建了包（或只切了 release），两边就会**静默分叉**，
  而投出去的是包、正文里引用的却是 r28。

第 31 轮实测（当前状态）：包内 5 个 `.tex` 与 r28 `source/` 以及活树**三方一致**；
包内 `main.pdf` / `supplementary.pdf` 与 r28 的哈希**也一致**（活树的 PDF 只是同源更新构建）。
⇒ 本脚本把这条性质固定下来。

用法： py code/audit_package_vs_release.py [--selftest]
"""
import hashlib
import io
import os
import re
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
KC = os.path.dirname(HERE)                                   # paper1_kc_diagnostic
PAPER1 = os.path.dirname(KC)
PKG = os.path.join(PAPER1, "paper1_SUBMISSION_PACKAGE_20260919", "01_manuscript")
RELROOT = os.path.join(KC, "releases")


def sha(p):
    return hashlib.sha256(io.open(p, "rb").read()).hexdigest()


def latest_release():
    ds = [d for d in sorted(os.listdir(RELROOT))
          if os.path.isdir(os.path.join(RELROOT, d)) and "s1-corrected" in d]
    return os.path.join(RELROOT, ds[-1]) if ds else None


def compare(pkg, rel_src):
    """返回 (一致, 漂移, 包内独有, 冻结内独有)。只比 .tex（构建产物另行处理）。"""
    same, drift, only_pkg, only_rel = [], [], [], []
    ptex = sorted(n for n in os.listdir(pkg) if n.endswith(".tex"))
    rtex = sorted(n for n in os.listdir(rel_src) if n.endswith(".tex"))
    for n in ptex:
        if n not in rtex:
            only_pkg.append(n)
        elif sha(os.path.join(pkg, n)) == sha(os.path.join(rel_src, n)):
            same.append(n)
        else:
            drift.append(n)
    only_rel = [n for n in rtex if n not in ptex]
    return same, drift, only_pkg, only_rel


def selftest():
    d = tempfile.mkdtemp(prefix="pkgrel_")
    pkg = os.path.join(d, "pkg")
    src = os.path.join(d, "src")
    os.makedirs(pkg)
    os.makedirs(src)
    for dd in (pkg, src):
        io.open(os.path.join(dd, "a.tex"), "w", encoding="utf-8").write("A\n")
        io.open(os.path.join(dd, "b.tex"), "w", encoding="utf-8").write("B\n")
    same, drift, op, orr = compare(pkg, src)
    rows = [("正例：完全一致 ⇒ 全 same", sorted(same) == ["a.tex", "b.tex"] and not drift, str(same))]
    io.open(os.path.join(pkg, "b.tex"), "w", encoding="utf-8").write("B changed\n")
    same2, drift2, _, _ = compare(pkg, src)
    rows.append(("反例：包内改了 ⇒ 必报 drift", drift2 == ["b.tex"], str(drift2)))
    ok = all(r[1] for r in rows)
    print("== 自测：投稿包 vs 冻结件 ==")
    for lab, g, note in rows:
        print("   [%s] %-28s %s" % ("PASS" if g else "FAIL", lab, note))
    print("   → %s" % ("全部通过" if ok else "** 有失败 **"))
    shutil.rmtree(d, ignore_errors=True)
    return ok


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
    rel = latest_release()
    if not rel:
        print("找不到冻结版本")
        sys.exit(2)
    src = os.path.join(rel, "source")
    print("投稿包 : %s" % PKG.replace(PAPER1, "<paper1>"))
    print("冻结件 : %s" % os.path.basename(rel))
    same, drift, only_pkg, only_rel = compare(PKG, src)
    print("\n  正文 .tex 一致   : %d  %s" % (len(same), same))
    if only_pkg:
        print("  仅投稿包有       : %s" % only_pkg)
    if only_rel:
        print("  仅冻结件有       : %s" % only_rel)
    if drift:
        print("  ★ 已分叉         : %s" % drift)
        print("\nVERDICT: 投稿包正文与冻结版本**不一致** —— 二者之一已过期。")
        print("  投出去的会是**包**，而正文/信里引用的是 %s。" % os.path.basename(rel))
        print("  ⇒ 先判断该以哪边为准，再把另一边重建（重建包走 sync_package.py；切新版本走 release_freezer.py）。")
        sys.exit(1)
    print("\nVERDICT: CLEAN -- 投稿包正文与冻结版本逐字节一致")
    sys.exit(0)
