# -*- coding: utf-8 -*-
"""audit_void_band.py —— 扫描**活文档**里是否还在把「作废的等价带」当现行值用。

背景（真实缺陷，2026-09-27 发现）：
  迹形标定 `kappa = 1.01 +- 0.11` / 带 `[0.78,1.23]` 已被**撤回**（它把造血种子间散布虚高 56 倍、
  把带中心位移 24%）。论文 `supplementary.tex` 表 S6 早已改为可预测性形的
  `1.001 +- 0.032` / `[0.937,1.064]`，但 `docs/math_derivations_{en,zh}.md`
  （自称"论文每条定量推导的独立摘录，按论文节号以保持可追溯"）**仍在把作废值当现行值**。

判定规则：
  出现作废带值的行，**必须**同时带有"更正/反例"语境关键词，否则判为缺陷。
  这条规则本身是启发式，所以脚本自带**正反例自测**（见 --selftest）。

用法：
  py code/audit_void_band.py            # 扫描
  py code/audit_void_band.py --selftest # 正反例自测
"""
import io
import os
import re
import sys
import tempfile

P1 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))          # paper1_kc_diagnostic
ROOT = os.path.dirname(os.path.dirname(P1))                                # F:/UNAGI_Project
SCAN = [P1, os.path.join(ROOT, "paper2"), os.path.join(ROOT, "paper3"),
        os.path.join(ROOT, "thesis")]

# 冻结/快照/构建/暂存 —— 不是"活文档"，历史值在此出现属正常
EXCL = ("/releases/", "/snapshots/", "/_backup", "/_quarantine/", "/archive/", "/_archive",
        "/build/", "/output/", "/.workbuddy/", "/_infra/", "/.git/", "/figures/")

VOID = re.compile(r"0\.78\s*,\s*1\.23|1\.23\]|\[0\.78")
CTX = re.compile(r"retract|retired|counter-?example|superseded|legacy|historical|void|"
                 r"撤回|反例|作废|过时|已改为|更正前")
# ★ 带日期的评审/审计记录**本来就该**引用当时的值；把它们与"活文档"分开报，
#   否则扫描器会被淹没在合理的历史引用里（本轮实测 28 处里 26 处属此类）。
HIST = re.compile(r"reviews?[/\\]|AUDIT_NOTES|_review|review_|peer_review|neutral_review|"
                  r"SKILL_REVIEW|paper\.md|NEWCOMPUTE|superseded")
EXT = (".md", ".tex", ".rst", ".txt")


# ★ 第 44 轮：读不到的文件（OSError）在此登记 ——
#   原来的 `except OSError: continue` 会让扫描范围**静默缩小**，而守卫照样报 CLEAN。
UNREADABLE = []


def scan(paths=SCAN):
    """返回 (活文档命中, 历史记录命中)，各为 [(文件, 行号, 行文本)]。

    ★ 语境按**前一行 + 本行 + 后一行**判定：更正说明可能写在上一行
      （`kappa_c_audit/README.md` 的 "superseded"）或下一行（`math_derivations_zh.md`
      的"该形式已撤回"）。只看本行会两头误报 —— 两种形态本轮都真实踩到过。
    """
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
                    if not VOID.search(ln):
                        continue
                    ctx = " ".join(lines[max(0, i - 1):i + 2])
                    if CTX.search(ctx):
                        continue
                    (hist if is_hist else live).append((p, i + 1, ln.strip()))
    return live, hist


def selftest():
    """正例（带语境 ⇒ 不报）+ 反例（无语境 ⇒ 必报）。任一不符即 exit 1。"""
    d = tempfile.mkdtemp(prefix="voidband_")
    good = os.path.join(d, "good.md")
    bad = os.path.join(d, "bad.md")
    io.open(good, "w", encoding="utf-8").write(
        "band `[0.78, 1.23]` is **retracted** and kept only as a counter-example.\n")
    io.open(bad, "w", encoding="utf-8").write(
        "the vanilla-equivalent band is `[0.78, 1.23]` (mean 1.01 +- 0.11).\n")
    rows = []
    g, h = scan([d])
    got_good = any(os.path.basename(p) == "good.md" for p, _i, _l in g + h)
    got_bad = any(os.path.basename(p) == "bad.md" for p, _i, _l in g)
    rows.append(("正例：作废值带语境 ⇒ 不报", not got_good, "good.md 是否被误报"))
    rows.append(("反例：作废值无语境 ⇒ 必报", got_bad, "bad.md 是否被抓（活文档）"))
    # 语境在**上一行**时也必须算有语境（本轮真实误报的形态）
    cross = os.path.join(d, "cross.md")
    io.open(cross, "w", encoding="utf-8").write(
        "the trace-form band was **superseded**;\nband `[0.78, 1.23]` is not current.\n")
    g2, h2 = scan([d])
    got_cross = any(os.path.basename(p) == "cross.md" for p, _i, _l in g2)
    rows.append(("正例：语境在上一行 ⇒ 不报", not got_cross, "cross.md 是否被误报"))
    ok = all(r[1] for r in rows)
    print("== 自测：作废带扫描器 ==")
    for lab, good_, note in rows:
        print("   [%s] %-28s %s" % ("PASS" if good_ else "FAIL", lab, note))
    print("   → %s" % ("全部通过" if ok else "** 有失败 **"))
    import shutil
    shutil.rmtree(d, ignore_errors=True)
    return ok


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
    live, hist = scan()
    print("扫描活文档 %d 个目录（已排除 releases/snapshots/build/output/_infra）" % len(SCAN))
    if hist:
        print("\n-- 历史记录（带日期的评审/审计文档，**本应**引用当时的值；不计为缺陷）%d 处 --" % len(hist))
        for p, i, _ln in hist:
            print("   %s:%d" % (p.replace(ROOT, "<root>"), i))
    if UNREADABLE:
        print("\nVERDICT: **无法判定** -- 有 %d 个文件读不到，扫描范围被静默缩小：" % len(UNREADABLE))
        for _p in UNREADABLE[:10]:
            print("   - %s" % _p)
        print("  ★ 「读不到」不等于「没问题」；先修权限/路径，再跑本守卫。")
        sys.exit(3)
    if not live:
        print("\nVERDICT: CLEAN -- 活文档里没有把作废等价带当现行值使用之处")
        sys.exit(0)
    print("\nVERDICT: %d 处缺陷 —— 活文档里作废带值出现在无更正语境的行：" % len(live))
    for p, i, ln in live:
        print("   %s:%d\n      %s" % (p.replace(ROOT, "<root>"), i, ln[:130]))
    sys.exit(1)
