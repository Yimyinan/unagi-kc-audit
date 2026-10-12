# -*- coding: utf-8 -*-
"""audit_repo_sync.py —— 待发布仓库 must 覆盖 code release 的全部文件（除经批准的例外）。

为什么需要它：
  `repo_kappa-c-audit/` 是**待发布的干净副本**，由 `03_code_release/` 经
  ① 排除投稿包记账文件、② 出库前路径净化 得到（见第 2 轮）。
  但**没有任何门禁保证"包里的文件都被带上了"** ——
  若日后往 `03_code_release/` 加了脚本，待发布仓库不会自动获得，也**没人会响**。

判据：`repo_kappa-c-audit` 的文件集必须 ⊇ `03_code_release` 的文件集，
减去**显式批准**的例外（下表，每条带理由）。少一个即报缺陷。

第 32 轮实测：204 vs 205，唯一差异是 `data/_ridge_recompute_cache.json`
（`.gitignore` 已排除的 scratch 缓存；且它含 37 处绝对路径，第 2 轮**有意**不带出）。
⇒ 属批准例外，当前 CLEAN。

用法： py code/audit_repo_sync.py [--selftest]
"""
import io
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
KC = os.path.dirname(HERE)
PAPER1 = os.path.dirname(KC)
SRC = os.path.join(PAPER1, "paper1_SUBMISSION_PACKAGE_20260919", "03_code_release")
DST = os.path.join(PAPER1, "repo_kappa-c-audit")

# 经批准的例外。★ 自测抓到首版只做**精确路径**匹配，表达不了"data/_* 是 scratch"这一意图
#    （真实树恰好因文件名完全一致而通过，属"碰巧对"）⇒ 拆成"精确名"与"前缀"两类。
EXACT = {
    "FILE_AUDIT.md": "投稿包记账文件：内部路径相对包根，放进纯代码库会指不到",
    "README_PACKAGE.md": "同上",
    "MANIFEST.sha256": "同上（且仓库会由 git 自己保证完整性）",
    "data/_ridge_recompute_cache.json":
        "gitignored 的 scratch 缓存；且含 37 处绝对路径 ⇒ 出库前有意不带出（第 2 轮）",
}
PREFIXES = [
    ("data/_", "data/ 下以 _ 开头者是 .gitignore 明确排除的 scratch/中间结果，一律不带出"),
]
SANCTIONED_LABEL = list(EXACT) + [p for p, _ in PREFIXES]


def is_sanctioned(rel):
    if rel in EXACT:
        return True
    return any(rel.startswith(p) for p, _ in PREFIXES)


def rels(base):
    out = set()
    for dp, dn, fn in os.walk(base):
        parts = dp.replace("\\", "/").split("/")
        if ".git" in parts or "__pycache__" in parts:
            continue
        for n in fn:
            out.add(os.path.relpath(os.path.join(dp, n), base).replace("\\", "/"))
    return out


def diff(s, d):
    """返回 (缺失, 额外) —— 缺失 = 在 SRC 且非批准例外、但不在 DST。"""
    missing = sorted(x for x in s if x not in d and not is_sanctioned(x))
    extra = sorted(d - s)
    return missing, extra


def selftest():
    d = tempfile.mkdtemp(prefix="reposync_")
    a, b = os.path.join(d, "src"), os.path.join(d, "dst")
    os.makedirs(a)
    os.makedirs(b)
    for f in ("x.py", "MANIFEST.sha256", "data/_cache.json"):
        p = os.path.join(a, f)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        io.open(p, "w", encoding="utf-8").write("1\n")
    io.open(os.path.join(b, "x.py"), "w", encoding="utf-8").write("1\n")
    m, e = diff(rels(a), rels(b))
    rows = [("正例：记账文件与 scratch 不算缺失", m == [], str(m)),
            ("正例：无额外文件", e == [], str(e))]
    io.open(os.path.join(a, "y.py"), "w", encoding="utf-8").write("2\n")
    m2, _ = diff(rels(a), rels(b))
    rows.append(("反例：漏带 y.py ⇒ 必报", m2 == ["y.py"], str(m2)))
    ok = all(r[1] for r in rows)
    print("== 自测：待发布仓库 vs code release ==")
    for lab, g, note in rows:
        print("   [%s] %-32s %s" % ("PASS" if g else "FAIL", lab, note))
    print("   → %s" % ("全部通过" if ok else "** 有失败 **"))
    shutil.rmtree(d, ignore_errors=True)
    return ok


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
    if not (os.path.isdir(SRC) and os.path.isdir(DST)):
        print("找不到 %s 或 %s" % (SRC, DST))
        sys.exit(2)
    s, d = rels(SRC), rels(DST)
    missing, extra = diff(s, d)
    print("03_code_release : %d 个文件" % len(s))
    print("repo_kappa-c-audit : %d 个文件" % len(d))
    print("批准例外 %d 条：" % len(SANCTIONED_LABEL))
    for k, why in EXACT.items():
        print("   - %-44s (%s) %s" % (k, "存在" if k in s else "不存在", why))
    for pre, why in PREFIXES:
        n = len([x for x in s if x.startswith(pre)])
        print("   - %-44s (%d 个) %s" % (pre + "*", n, why))
    if extra:
        print("\n  额外（仓库里有、包里没有）：")
        for x in extra[:10]:
            print("     + " + x)
    if missing:
        print("\nVERDICT: 待发布仓库**漏带** %d 个文件：" % len(missing))
        for x in missing[:20]:
            print("   - " + x)
        print("  ⇒ 要么补进仓库，要么补进仓库，要么把它加进 EXACT/PREFIXES 并写明理由。")
        sys.exit(1)
    print("\nVERDICT: CLEAN -- 待发布仓库覆盖 code release 的全部文件（除 %d 条批准例外）"
          % len(SANCTIONED_LABEL))
    sys.exit(0)
