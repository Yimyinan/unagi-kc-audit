# -*- coding: utf-8 -*-
"""audit_trrust_scope.py —— 检查 TRRUST 口径是否**声明完整**（目标 (2) 的守卫）。

统一口径（中文侧实测为准）：
    原始文件      : 9,396 条非空行 / 795 个 TF
    去除 27 自环后: **9,369 条边 / 794 个 TF / 2,491 个靶基因**  ← 论文使用的口径

缺陷模式：**只引原始值而不说明清洗口径**（或只引清洗值而不说明来源），
使读者无法判断 9,396 与 9,369 的差别从何而来。
合法模式：① 两个口径同现；② 明确标注"原始文件"/"raw"/"未清洗"；③ 位于历史文档。

用法： py code/audit_trrust_scope.py [--selftest]
"""
import io
import os
import re
import sys
import tempfile

P1 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(P1))
SCAN = [P1, os.path.join(ROOT, "paper2"), os.path.join(ROOT, "paper3"),
        os.path.join(ROOT, "thesis")]

EXCL = ("/releases/", "/snapshots/", "/_backup", "/_quarantine/", "/archive/", "/_archive",
        "/build/", "/output/", "/.git/", "/_infra/", "/repo_kappa-c-audit/", "/.workbuddy/",
        "/unagi_custom_data/")          # 原始数据文件本身不算"文档声明"
HIST = re.compile(r"superseded|archive|backup|reviews?[/\\]|AUDIT_NOTES|_review|CHANGELOG|"
                  r"REVISION_NOTES|_PLAN|REVIEW_")
EXT = (".tex", ".md", ".txt", ".rst")

RAW = re.compile(r"9[,，{]?396|9,396|795\s*(?:个\s*TF|TF|个转录因子|个转录因子)")
CLEAN = re.compile(r"9[,，{]?369|9,369|794\s*(?:个\s*TF|TF|个转录因子)|2[,，{]?491|2,491")
QUAL = re.compile(r"原始|raw|清洗|自环|self-?loop|去除|after removing|未清洗")


# ★ 第 44 轮：读不到的文件（OSError）在此登记 ——
#   原来的 `except OSError: continue` 会让扫描范围**静默缩小**，而守卫照样报 CLEAN。
UNREADABLE = []


def scan(paths=SCAN):
    live, hist = [], []
    for base in paths:
        for dp, _dn, fn in os.walk(base):
            if any(x in dp.replace("\\", "/") + "/" for x in EXCL):
                continue
            for n in fn:
                if not n.endswith(EXT):
                    continue
                p = os.path.join(dp, n)
                try:
                    lines = io.open(p, encoding="utf-8", errors="replace").read().split("\n")
                except OSError:
                    UNREADABLE.append(p)   # ★ 第 44 轮：读不到 ≠ 通过
                    continue
                is_hist = bool(HIST.search(p.replace("\\", "/")))
                for i, ln in enumerate(lines):
                    if not RAW.search(ln):
                        continue
                    ctx = " ".join(lines[max(0, i - 1):i + 2])
                    if CLEAN.search(ctx) or QUAL.search(ctx):
                        continue
                    (hist if is_hist else live).append((p, i + 1, ln.strip()))
    return live, hist


def selftest():
    d = tempfile.mkdtemp(prefix="trrust_")
    io.open(os.path.join(d, "bare.md"), "w", encoding="utf-8").write(
        "The prior ships 9,396 human edges from TRRUST v2 (795 TFs).\n")
    io.open(os.path.join(d, "both.md"), "w", encoding="utf-8").write(
        "TRRUST v2 human: 9,396 edges / 795 TFs (9,369 / 794 after removing 27 self-loops).\n")
    io.open(os.path.join(d, "rawlabelled.md"), "w", encoding="utf-8").write(
        "the raw file has 9,396 non-empty lines and 795 TFs.\n")
    live, _h = scan([d])
    names = {os.path.basename(p) for p, _i, _l in live}
    rows = [("反例：只引原始值 ⇒ 必报", "bare.md" in names, "bare.md"),
            ("正例：两口径同现 ⇒ 不报", "both.md" not in names, "both.md"),
            ("正例：标明'raw file' ⇒ 不报", "rawlabelled.md" not in names, "rawlabelled.md")]
    ok = all(r[1] for r in rows)
    print("== 自测：TRRUST 口径审计器 ==")
    for lab, g, note in rows:
        print("   [%s] %-30s %s" % ("PASS" if g else "FAIL", lab, note))
    print("   → %s" % ("全部通过" if ok else "** 有失败 **"))
    import shutil
    shutil.rmtree(d, ignore_errors=True)
    return ok


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
    live, hist = scan()
    print("统一口径：原始 9,396 / 795 TF → 清洗后 9,369 / 794 TF / 2,491 靶")
    for p, i, _l in hist:
        print("   [历史] %s:%d" % (p.replace(ROOT, "<root>"), i))
    if UNREADABLE:
        print("\nVERDICT: **无法判定** -- 有 %d 个文件读不到，扫描范围被静默缩小：" % len(UNREADABLE))
        for _p in UNREADABLE[:10]:
            print("   - %s" % _p)
        print("  ★ 「读不到」不等于「没问题」；先修权限/路径，再跑本守卫。")
        sys.exit(3)
    if not live:
        print("VERDICT: CLEAN -- 没有只引原始值而不说明口径之处")
        sys.exit(0)
    print("VERDICT: %d 处缺陷 —— 只引 TRRUST 原始值、未说明清洗口径：" % len(live))
    for p, i, ln in live:
        print("   %s:%d\n      %s" % (p.replace(ROOT, "<root>"), i, ln[:130]))
    sys.exit(1)
