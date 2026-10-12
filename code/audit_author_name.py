# -*- coding: utf-8 -*-
"""audit_author_name.py —— 全文本扩展名扫描**旧署名**是否仍出现在活文档里。

为什么单列（本轮真实发现的盲区）：
  统一的署名口径是 **Yan Yinan**（见 CHANGELOG / 各 .zenodo.json / CITATION.cff）。
  2026-09-27 那次审计**只扫了 `.tex` 与 `.md`**，于是漏掉了一个 `.txt`：
  `paper1_kc_diagnostic.txt`（2026-09-05 的 PDF 文本抽取物）第 3 行仍是旧署名
  `Yim Yinan`，且它还带着已撤回的等价带。
  ⇒ 教训：**审计的扩展名集合本身就是审计对象**。

判定：活文档里出现旧署名 = 缺陷；文件名含 superseded/archive/backup 或位于
reviews//releases/ 等处的历史记录单独列出，不计为缺陷。

用法： py code/audit_author_name.py [--selftest]
"""
import io
import os
import re
import sys
import tempfile

P1 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(P1))
SCAN = [os.path.join(ROOT, "paper1"),          # ← 含 paper1_kc_diagnostic **与投稿包**（后者原先没被扫到）
        os.path.join(ROOT, "paper2"), os.path.join(ROOT, "paper3"),
        os.path.join(ROOT, "thesis")]

EXCL = ("/releases/", "/snapshots/", "/_backup", "/_quarantine/", "/archive/", "/_archive",
        "/build/", "/output/", "/.git/", "/_infra/", "/repo_kappa-c-audit/", "/.workbuddy/")
HIST = re.compile(r"superseded|archive|backup|reviews?[/\\]|AUDIT_NOTES|_review|CHANGELOG|"
                  r"REPO_AUDIT|REPO_SETUP")     # ← 后两者**本身就是**记录旧→新改名的档案
OLD = re.compile(r"Yim\s+Yinan|\bYim\b")
# ★ 扩展名集合刻意放宽：上一轮就是因为只查 .tex/.md 才漏掉 .txt
EXT = (".tex", ".md", ".txt", ".rst", ".cff", ".json", ".yaml", ".yml", ".toml", ".bib")


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
                    txt = io.open(p, encoding="utf-8", errors="replace").read()
                except OSError:
                    UNREADABLE.append(p)   # ★ 第 44 轮：读不到 ≠ 通过
                    continue
                is_hist = bool(HIST.search(p.replace("\\", "/")))
                for i, ln in enumerate(txt.split("\n"), 1):
                    if OLD.search(ln):
                        (hist if is_hist else live).append((p, i, ln.strip()[:120]))
    return live, hist


def selftest():
    d = tempfile.mkdtemp(prefix="authname_")
    io.open(os.path.join(d, "live.txt"), "w", encoding="utf-8").write("Yim Yinan1,*\n")
    io.open(os.path.join(d, "old_superseded.md"), "w", encoding="utf-8").write("Yim Yinan\n")
    live, hist = scan([d])
    gl = any(os.path.basename(p) == "live.txt" for p, _i, _l in live)
    hl = any(os.path.basename(p) == "old_superseded.md" for p, _i, _l in hist)
    rows = [("反例：活文档含旧署名 ⇒ 必报", gl, "live.txt"),
            ("正例：superseded 文件名 ⇒ 归入历史", hl, "old_superseded.md"),
            ("正例：历史记录不计为缺陷", not any(
                os.path.basename(p) == "old_superseded.md" for p, _i, _l in live), "")]
    ok = all(r[1] for r in rows)
    print("== 自测：署名审计器 ==")
    for lab, g, note in rows:
        print("   [%s] %-32s %s" % ("PASS" if g else "FAIL", lab, note))
    print("   → %s" % ("全部通过" if ok else "** 有失败 **"))
    import shutil
    shutil.rmtree(d, ignore_errors=True)
    return ok


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
    live, hist = scan()
    print("统一署名口径：Yan Yinan；扫描扩展名 %s" % (", ".join(EXT),))
    for p, i, _l in hist:
        print("   [历史] %s:%d" % (p.replace(ROOT, "<root>"), i))

    # ★★ 投稿信署名口径：**每一封** cover letter 都必须含同一行
    #    （目标 (1) 要求"署名口径统一"——历史上 paper2 的空 `\author{}` 与
    #      `PATTERNS_SUBMISSION.md` 的单人落款都说明"名单对"不等于"文件里写对了"）
    SIG = "on behalf of the authors: Yan Yinan (first author), Lan Wang"
    cls = []
    for base in SCAN:
        for dp, _dn, fn in os.walk(base):
            if any(x in dp.replace("\\", "/") + "/" for x in EXCL):
                continue
            for n in fn:
                if n.startswith("cover_letter") and n.endswith(".tex"):
                    cls.append(os.path.join(dp, n))
    if cls:
        print("\n投稿信署名口径（共 %d 封）：" % len(cls))
        for p in cls:
            body = io.open(p, encoding="utf-8", errors="replace").read()
            good = SIG in body
            print("   [%s] %s" % ("PASS" if good else "FAIL", p.replace(ROOT, "<root>")))
            if not good:
                live.append((p, 0, "缺少统一署名行：%s" % SIG))

    if UNREADABLE:
        print("\nVERDICT: **无法判定** -- 有 %d 个文件读不到，扫描范围被静默缩小：" % len(UNREADABLE))
        for _p in UNREADABLE[:10]:
            print("   - %s" % _p)
        print("  ★ 「读不到」不等于「没问题」；先修权限/路径，再跑本守卫。")
        sys.exit(3)
    if not live:
        print("VERDICT: CLEAN -- 活文档里没有旧署名，且所有投稿信署名口径一致")
        sys.exit(0)
    print("VERDICT: %d 处缺陷 —— 活文档仍含旧署名：" % len(live))
    for p, i, ln in live:
        print("   %s:%d\n      %s" % (p.replace(ROOT, "<root>"), i, ln))
    sys.exit(1)
