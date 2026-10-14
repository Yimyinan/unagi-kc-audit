# -*- coding: utf-8 -*-
"""selftest_metadata_gate.py —— `check_submission_metadata.repo_metadata_checks` 的回归自测。

**为什么需要它**：这个守卫的作用是"在不可撤销的 DOI 发布之前拦住错元数据"，
一个**静默放行**的守卫比没有守卫更危险。本文件对每条规则同时给出**正例（必须静默）**
与**反例（必须开火）**，任何一侧不符即 exit 1。

用法：``python tools/selftest_metadata_gate.py``   （exit 0 = 全部符合）

已抓到的真实缺陷（留作记录）：
  * 版本正则只认三段式 ⇒ 过期 `.figshare.json` 里的 ``"version": "v1.0"`` 被静默跳过；
  * wheel 版本解析 ``split("-")[1]`` ⇒ ``scvi-tools-1.5.0`` 解析成 ``tools``；
  * wheel 发现递归全根 ⇒ 把 ``archive/`` 下第三方下载的 wheel 当成本包产物。
"""
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import check_submission_metadata as G

PYPROJECT = ('[project]\nname = "kappa-c-audit"\nversion = "%s"\n'
             'authors = [{ name = "%s" }]\n')
FIGSHARE_JSON = '{"version": "%s", "creators": [{"name": "%s"}]}'
CFF = 'cff-version: 1.2.0\nversion: "%s"\nfamily-names: "%s"\ngiven-names: "%s"\n'

FAILURES = []


def check(label, ok, detail=""):
    print("%-4s %-52s %s" % ("OK" if ok else "FAIL", label, detail))
    if not ok:
        FAILURES.append(label)


def build(tmp, ver="0.1.2", person="Yan Yinan", wheels=("0.1.2",),
          zen=None, cffv="0.1.2", zen_person=None):
    """A miniature release root: pyproject + .figshare.json + CITATION.cff + dist/."""
    root = os.path.join(tmp, "rel")
    dist = os.path.join(root, "kappa_c_audit", "dist")
    os.makedirs(dist, exist_ok=True)
    for v in wheels:
        open(os.path.join(dist, "kappa_c_audit-%s-py3-none-any.whl" % v), "wb").close()
    open(os.path.join(root, "kappa_c_audit", "pyproject.toml"), "w").write(
        PYPROJECT % (ver, person))
    open(os.path.join(root, ".figshare.json"), "w").write(
        FIGSHARE_JSON % (zen if zen is not None else ver, zen_person or person))
    fam, giv = person.split()
    open(os.path.join(root, "CITATION.cff"), "w").write(CFF % (cffv, fam, giv))
    return root


def fires(label, expect_fire, **kw):
    tmp = tempfile.mkdtemp()
    try:
        probs = G.repo_metadata_checks(build(tmp, **kw))[1]
        check(label, bool(probs) == expect_fire, probs if probs else "(clean)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


print("--- wheel 文件名解析（含连字符的发行名） ---")
for fn, want in [("kappa_c_audit-0.1.2-py3-none-any.whl", ("kappa-c-audit", "0.1.2")),
                 ("scvi_tools-1.5.0-py3-none-any.whl", ("scvi-tools", "1.5.0")),
                 ("scvi-tools-1.5.0-py3-none-any.whl", ("scvi-tools", "1.5.0")),
                 ("not_a_wheel.tar.gz", None)]:
    check(fn, G._parse_wheel(fn) == want, "-> %s" % (G._parse_wheel(fn),))

print("--- 版本提取 ---")
for text, want in [(FIGSHARE_JSON % ("0.1.2", "Yan Yinan"), "0.1.2"),
                   (FIGSHARE_JSON % ("v1.0", "Yan Yinan"), "1.0"),          # 两段式，旧文件形态
                   (CFF % ("0.1.2", "Yan", "Yinan"), "0.1.2"),
                   (PYPROJECT % ("0.1.2", "Yan Yinan"), "0.1.2")]:
    got = G._declared("x", text)[0]
    check("version -> %s" % want, got == want, "got %s" % got)
check("cff-version: 1.2.0 不被误读为软件版本",
      G._declared("CITATION.cff", CFF % ("0.1.2", "Yan", "Yinan"))[0] == "0.1.2")

print("--- 行为：正例须静默、反例须开火 ---")
fires("全部一致", False)
fires("pyproject 版本与 wheel 不符", True, ver="0.1.3")
fires(".figshare 版本与 wheel 不符（三段式）", True, zen="1.0.0")
fires(".figshare 版本与 wheel 不符（两段式 v1.0）", True, zen="v1.0")
fires("CITATION.cff 版本与 wheel 不符", True, cffv="0.1.0")
fires(".figshare 作者名漂移", True, zen_person="Yim, Yinan")
fires("dist/ 残留两个版本 wheel", True, wheels=("0.1.1", "0.1.2"))

print("--- 外部 wheel 必须被忽略 ---")
tmp = tempfile.mkdtemp()
try:
    root = build(tmp)
    arc = os.path.join(root, "archive", "scvi_probe")
    os.makedirs(arc, exist_ok=True)
    open(os.path.join(arc, "scvi_tools-1.5.0-py3-none-any.whl"), "wb").close()
    check("archive/ 下的第三方 wheel 不参与校验",
          not G.repo_metadata_checks(root)[1])
    check("dist_wheels 只认 dist/", sorted(G.dist_wheels(root)) == ["kappa-c-audit"],
          sorted(G.dist_wheels(root)))
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print("\n%s（%d 项不符）" % ("全部通过" if not FAILURES else "存在不符项", len(FAILURES)))
sys.exit(1 if FAILURES else 0)
