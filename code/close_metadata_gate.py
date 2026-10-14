# -*- coding: utf-8 -*-
"""close_metadata_gate.py —— 把「建 repo + 取 DOI」的成果**一次性**落进项目并验证。

为什么必须"一次性"（项目纪律，付过代价）：
  `kappa_c_audit/pyproject.toml` 的 `[project.urls]` Homepage 会在**构建时**烙进 wheel 的
  `METADATA`，而 wheel 随 DOI **永久归档**。⇒ 填 URL 而不重建 wheel = 发布了坏元数据；
    事后改 pyproject **无效**。本脚本把「填 URL → 重建 wheel → 重编 → 同步包 → 跑门禁」
    绑成一次动作。

★ 值不可编造：必须由人工在 GitHub 建库、figshare 取 **concept DOI**（"cite all versions"）后提供。
   本脚本默认 **dry-run**，只打印计划；`--apply` 才落盘。

用法：
  python code/close_metadata_gate.py --repo-url https://github.com/<user>/kappa-c-audit \
                                     --doi 10.6084/m9.figshare.XXXXXXX
  （加 --apply 才真正执行）
"""
import argparse
import glob
import io
import os
import re
import subprocess
import sys

PAPER1 = os.environ.get("UNAGI_PAPER1_DIR", r"F:/UNAGI_Project/paper1")
KC = os.path.join(PAPER1, "paper1_kc_diagnostic")
PKG = os.path.join(PAPER1, "paper1_SUBMISSION_PACKAGE_20260919")
PAPER2 = os.environ.get("UNAGI_PAPER2_DIR", r"F:/UNAGI_Project/paper2")
REPO = os.path.join(PAPER1, "repo_kappa-c-audit")          # 09-27 暂存仓库

URL_RE = re.compile(r"^https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
DOI_RE = re.compile(r"^10\.\d{4,9}/[A-Za-z0-9._;()/:-]+$")

ap = argparse.ArgumentParser()
ap.add_argument("--repo-url", required=True)
ap.add_argument("--doi", required=True)
ap.add_argument("--copyright", default="",
                help="LICENSE 的版权人写法（如 'Yan Yinan'）。★ 不设默认值："
                     "版权人是法律表述，必须由人给出；不给则该 4 份 LICENSE 保持占位、门禁仍 exit 1")
ap.add_argument("--apply", action="store_true")
A = ap.parse_args()

print("=" * 68)
print("模式:", "APPLY（会写盘）" if A.apply else "DRY-RUN（只打印计划）")
print("repo URL:", A.repo_url)
print("DOI     :", A.doi)
print("=" * 68)

# ---------- 0. 校验（值不可编造，格式必须先过关） ----------
ok = True
if not URL_RE.match(A.repo_url):
    print("!! repo URL 不像公开 GitHub 仓库地址（应形如 https://github.com/<user>/<repo>）")
    ok = False
if "/kappa-c-audit" not in A.repo_url:
    print("   [warn] 仓库名不含 'kappa-c-audit'；REPO_SETUP.md 建议用该名，确认无误再继续")
if not DOI_RE.match(A.doi):
    print("!! DOI 格式不对（应形如 10.6084/m9.figshare.1234567）")
    ok = False
if "figshare" not in A.doi.lower():
    print("   [warn] DOI 里没有 'figshare'——确认这是 figshare 的归档 DOI")
if not ok:
    sys.exit(2)

# ---------- 1. submission_metadata.tex（两篇 + 投稿包那份） ----------
TEX = []
for pat in (os.path.join(KC, "submission_metadata.tex"),
            os.path.join(PKG, "01_manuscript", "submission_metadata.tex"),
            os.path.join(PAPER2, "submission_metadata.tex")):
    if os.path.isfile(pat):
        TEX.append(pat)
print("\n[1] submission_metadata.tex 副本 %d 份:" % len(TEX))
for t in TEX:
    print("     ", t.replace(PAPER1, "<paper1>").replace(PAPER2, "<paper2>"))
if len(TEX) < 3:
    print("   [warn] 少于此前的 3 份（paper1 活树 / 投稿包 / paper2），请核对")

# ---------- 2. pyproject.toml（4 份，含暂存仓库） ----------
PYPROJ = sorted(set(glob.glob(os.path.join(PAPER1, "**", "kappa_c_audit", "pyproject.toml"), recursive=True)))
print("\n[2] kappa_c_audit/pyproject.toml 副本 %d 份:" % len(PYPROJ))
for p in PYPROJ:
    print("     ", p.replace(PAPER1, "<paper1>"))

# ---------- 3. wheel（URL 会烙进 METADATA ⇒ 必须重建） ----------
WHLS = sorted(set(glob.glob(os.path.join(PAPER1, "**", "kappa_c_audit", "dist", "*.whl"), recursive=True)))
print("\n[3] 需重建的 wheel %d 个（URL 在构建时烙进 METADATA，事后改 pyproject 无效）:" % len(WHLS))
for w in WHLS:
    print("     ", os.path.basename(w))

if not A.apply:
    print("\n[4] 计划（--apply 时执行）:")
    print("     4.1 改写 %d 份 submission_metadata.tex 的 \\RepositoryURL / \\FigshareDOI" % len(TEX))
    print("     4.2 给 %d 份 pyproject.toml 加/更新 [project.urls] Homepage" % len(PYPROJ))
    print("     4.2b 把 4 份 LICENSE 的 `Copyright (c) 2026 [AUTHOR NAME]` 换成你给的值"
          "（**需要 --copyright**；不给则跳过、门禁仍 exit 1）")
    print("     4.3 重建 %d 个 wheel（`python -m build --wheel`）" % len(WHLS))
    print("     4.4 重编 paper1 main.tex / supplementary.tex（各 3 遍）")
    print("     4.5 跑 sync_package.py 同步投稿包 + 重建 MANIFEST")
    print("     4.6 跑 check_submission_metadata.py --strict，期望 **exit 0**")
    print("")
    print("     ⚠ 注意（第 27 轮起）：该门禁**现在还扫 LICENSE**。全项目 4 个 LICENSE 的"
          " `Copyright (c) 2026 [AUTHOR NAME]`")
    print("       会被它拦下 ⇒ 想拿到 exit 0，必须**先决定版权人**并填好这 4 个文件：")
    print("         paper1/paper1_kc_diagnostic/LICENSE")
    print("         paper1/paper1_SUBMISSION_PACKAGE_20260919/03_code_release/LICENSE"
          "   ← 在投稿包 MANIFEST 内 ⇒ 填完要跑 sync_package.py 重建 MANIFEST")
    print("         paper1/repo_kappa-c-audit/LICENSE"
          "                            ← 在暂存仓库里 ⇒ 填完要补一次 commit")
    print("         paper2/LICENSE")
    print("       ★ 版权人是谁（作者 / 单位 / 两者）属**法律表述**，本脚本不代为编造。")
    print("\n(dry-run 结束；确认上面清单无误后加 --apply)")
    sys.exit(0)

# =========================== APPLY ===========================
def rd(p):
    return io.open(p, encoding="utf-8").read()


def wr(p, t):
    io.open(p, "w", encoding="utf-8", newline="").write(t)


# 4.1 tex
n = 0
for t in TEX:
    s = rd(t)
    s2 = re.sub(r"(\\newcommand\{\\RepositoryURL\}\{)[^}]*(?=\})", r"\g<1>" + A.repo_url, s)
    s2 = re.sub(r"(\\newcommand\{\\FigshareDOI\}\{)[^}]*(?=\})", r"\g<1>" + A.doi, s2)
    if s2 != s:
        wr(t, s2)
        n += 1
        print("   tex  已填:", os.path.basename(os.path.dirname(t)) + "/" + os.path.basename(t))
print("[4.1] 已改写 %d 份 tex" % n)

# 4.2 pyproject
n = 0
for p in PYPROJ:
    s = rd(p)
    block = '[project.urls]\nHomepage = "%s"\n' % A.repo_url
    if "[project.urls]" in s:
        s2 = re.sub(r"\[project\.urls\][^\[]*", block, s)
    else:
        s2 = s.rstrip("\n") + "\n\n" + block
    if s2 != s:
        wr(p, s2)
        n += 1
        print("   pyproject 已填:", p.replace(PAPER1, "<paper1>"))
print("[4.2] 已改写 %d 份 pyproject" % n)

# 4.2b LICENSE 的版权人（★ 第 42 轮新增）
#   为什么必须补这一步：门禁**还扫 LICENSE**，而全项目 4 个 LICENSE 都写着
#   `Copyright (c) 2026 [AUTHOR NAME]`。**只填 URL/DOI 的话，门禁拿不到 exit 0**
#   —— 第 42 轮实测确认本脚本此前**完全不碰 LICENSE**（只打印一句警告）。
#   ★ 版权人是法律表述，**不设默认值**：不给 --copyright 就明确告诉你还差这一项。
CR_OLD = "Copyright (c) 2026 [AUTHOR NAME]"
LICENSES = []
for base in (PAPER1, PAPER2):
    for dp, dn, fn in os.walk(base):
        seg = dp.replace("\\", "/").split("/")
        if any(x in seg for x in ("releases", "archive", "snapshots", ".git", "__pycache__")):
            continue
        for f in fn:
            if f == "LICENSE" or f.startswith("LICENSE."):
                LICENSES.append(os.path.join(dp, f))
LICENSES = sorted(set(LICENSES))
print("\n[4.2b] LICENSE 副本 %d 份（含 [AUTHOR NAME] 的）：" % len(LICENSES))
for p in LICENSES:
    print("     ", p.replace(PAPER1, "<paper1>").replace(PAPER2, "<paper2>"))
if not A.copyright:
    print("   !! **未提供 --copyright** ⇒ 这些文件会保持 [AUTHOR NAME]")
    print("      ⇒ 门禁**仍然 exit 1**（这不是 bug，是版权人尚未确定）")
    print("      想一次做齐：--copyright \"<版权人标准写法，如 Yan Yinan>\"")
else:
    n = 0
    for p in LICENSES:
        s = rd(p)
        if CR_OLD in s:
            wr(p, s.replace(CR_OLD, "Copyright (c) 2026 %s" % A.copyright))
            n += 1
            print("   LICENSE 已填:", p.replace(PAPER1, "<paper1>").replace(PAPER2, "<paper2>"))
    print("[4.2b] 已改写 %d 份 LICENSE" % n)
    if n:
        print("   ⚠ 投稿包那份在 MANIFEST 内 ⇒ 后面 4.5 的 sync_package 会重建 MANIFEST")
        print("   ⚠ 暂存仓库那份在 git 里 ⇒ 需要你补一次 commit")

# 4.3 wheel
for w in WHLS:
    d = os.path.dirname(os.path.dirname(w))
    print("[4.3] 重建 wheel 于", d.replace(PAPER1, "<paper1>"))
    r = subprocess.run([sys.executable, "-m", "build", "--wheel"], cwd=d,
                       capture_output=True, text=True)
    print("      exit=%d %s" % (r.returncode, (r.stdout or r.stderr).strip().splitlines()[-1:]))

# 4.4 重编
PDFLATEX = r"D:\MiKTeX\miktex\bin\x64\pdflatex.exe"
for tex in ("main.tex", "supplementary.tex"):
    print("[4.4] 编译", tex)
    for _ in range(3):
        rr = subprocess.run([PDFLATEX, "-interaction=nonstopmode", "-halt-on-error", tex],
                            cwd=KC, capture_output=True, text=True)
        if rr.returncode != 0:
            print("      !! 编译失败，见", tex.replace(".tex", ".log"))
            break

# 4.5 同步投稿包
print("[4.5] sync_package.py")
r = subprocess.run([sys.executable, os.path.join(KC, "code", "sync_package.py")],
                   cwd=KC, capture_output=True, text=True)
print("      " + "\n      ".join((r.stdout or r.stderr).strip().splitlines()[-4:]))

# 4.6 门禁
print("[4.6] check_submission_metadata.py --strict")
r = subprocess.run([sys.executable, os.path.join(KC, "tools", "check_submission_metadata.py"), "--strict"],
                   cwd=KC, capture_output=True, text=True)
print("      exit =", r.returncode, "（期望 0）")
print((r.stdout or "")[-600:])
