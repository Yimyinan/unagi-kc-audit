# -*- coding: utf-8 -*-
"""audit_freeze_drift.py —— 「冻结件 vs 活树」漂移检查（目标里的"冻结纪律"维度）。

为什么需要它：
  `release_freezer.py --verify` 只证明**冻结件自身完整**（MANIFEST 自洽），
  **不证明活树仍等于冻结件**。于是存在一个盲区：
  若冻结后又改了 `main.tex`，活树就悄悄**跑到了 r28 之外** —— 此时投出去的是新稿，
  而正文/信里引用的却是 r28，二者不一致且**没有任何门禁会响**。

判据：
  * 活树 == 冻结件  ⇒ 冻结是最新的（CLEAN）
  * 活树 != 冻结件  ⇒ **不是"坏了"，而是"该切新版本了"**（r29）
    两种处置都是正常流程，但**必须被发现**，所以本脚本对漂移报 exit 1 并给出该措辞。
  * 构建产物（.pdf/.log/.aux/.out/.toc）**天然逐次不同**，不参与比较。

用法： py code/audit_freeze_drift.py [--release <目录名>] [--selftest]
"""
import hashlib
import io
import os
import re
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
KC = os.path.dirname(HERE)
RELROOT = os.path.join(KC, "releases")
BUILD_EXT = (".pdf", ".log", ".aux", ".out", ".toc", ".synctex.gz", ".fls", ".fdb_latexmk")


def sha(p):
    return hashlib.sha256(io.open(p, "rb").read()).hexdigest()


def latest_release():
    if not os.path.isdir(RELROOT):
        return None
    ds = [d for d in sorted(os.listdir(RELROOT))
          if os.path.isdir(os.path.join(RELROOT, d)) and "s1-corrected" in d]
    return os.path.join(RELROOT, ds[-1]) if ds else None


def compare(rel):
    """返回 (一致列表, 漂移列表, 仅冻结件列表)。"""
    src = os.path.join(rel, "source")
    same, drift, only = [], [], []
    for n in sorted(os.listdir(src)):
        fp = os.path.join(src, n)
        if not os.path.isfile(fp) or n.endswith(BUILD_EXT):
            continue
        live = os.path.join(KC, n)
        if not os.path.exists(live):
            only.append(n)
        elif sha(fp) == sha(live):
            same.append(n)
        else:
            drift.append(n)
    return same, drift, only


def selftest():
    """正例（同内容 ⇒ 不报）+ 反例（改了内容 ⇒ 必报）。"""
    d = tempfile.mkdtemp(prefix="drift_")
    src = os.path.join(d, "source")
    os.makedirs(src)
    io.open(os.path.join(src, "a.tex"), "w", encoding="utf-8").write("A\n")
    io.open(os.path.join(src, "b.log"), "w", encoding="utf-8").write("noise1\n")
    io.open(os.path.join(d, "a.tex"), "w", encoding="utf-8").write("A\n")
    io.open(os.path.join(d, "b.log"), "w", encoding="utf-8").write("noise2\n")
    global KC
    old, KC = KC, d
    same, drift, only = compare(d)
    KC = old
    rows = [("正例：内容相同 ⇒ 不报", "a.tex" in same and "a.tex" not in drift, str(same)),
            ("正例：构建产物不参与", "b.log" not in same + drift, "b.log excluded"),
            ("反例：内容不同 ⇒ 必报", True, "见下轮 in-place 变异")]
    # 真正的反例：改活树内容
    io.open(os.path.join(d, "a.tex"), "w", encoding="utf-8").write("B\n")
    KC = d
    _s2, drift2, _o2 = compare(d)
    KC = old
    rows[2] = ("反例：内容不同 ⇒ 必报", "a.tex" in drift2, str(drift2))
    ok = all(r[1] for r in rows)
    print("== 自测：冻结漂移检查器 ==")
    for lab, g, note in rows:
        print("   [%s] %-24s %s" % ("PASS" if g else "FAIL", lab, note))
    print("   → %s" % ("全部通过" if ok else "** 有失败 **"))
    import shutil
    shutil.rmtree(d, ignore_errors=True)
    return ok


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)

    rel = None
    if "--release" in sys.argv:
        rel = os.path.join(RELROOT, sys.argv[sys.argv.index("--release") + 1])
    rel = rel or latest_release()
    if not rel or not os.path.isdir(rel):
        print("找不到冻结版本目录")
        sys.exit(2)

    same, drift, only = compare(rel)
    print("冻结版本：%s" % os.path.basename(rel))
    print("  源文件一致   : %d  %s" % (len(same), same))
    if only:
        print("  仅冻结件有   : %d  %s" % (len(only), only))
    if drift:
        print("  ★ 已漂移     : %d  %s" % (len(drift), drift))
        print("\nVERDICT: 活树已**跑到该冻结版本之外** —— 这不是故障，而是"
              "**需要切新版本**（例如 r29）。在切之前，正文/投稿信若仍引用 %s 即不一致。"
              % os.path.basename(rel))
        sys.exit(1)
    print("\nVERDICT: CLEAN -- 活树与冻结版本一致（冻结纪律成立；构建产物不参与比较）")
    sys.exit(0)
