# -*- coding: utf-8 -*-
"""assemble_code_release.py —— 把 03_code_release/ 补全成与它自己的描述相符的完整代码发布包。

内容（约 2.1 MB）：
  kappa_c_audit/   工具包（已就位，含 wheel / tests / examples）
  code/            分析脚本（133 个 .py；**排除** code/data/ 里的 h5/h5ad 二进制）
  data/            精选结果表（54 个 .json，共 600 KB；**排除** 322 MB 的 .npz 与 h5/xlsx）
  README.md        重写为如实描述
  CHANGELOG.md     承接原 README 里的修订史（公开仓库的规范做法）
  CITATION.cff     新增
  .gitignore       新增
  LICENSE / .figshare.json  已有（后者本次修正元数据）

原则：只搬不改；幂等；每个复制都做 sha256 校验。
"""
import hashlib, io, os, shutil

PAPER1 = r"F:/UNAGI_Project/paper1"
ROOT = PAPER1 + "/paper1_kc_diagnostic"
DEST = PAPER1 + "/paper1_SUBMISSION_PACKAGE_20260919/03_code_release"

CODE_EXT = (".py", ".sh", ".md", ".txt", ".json", ".yaml", ".yml", ".cfg", ".toml", ".rst")
SKIP_DIRS = {"__pycache__", "data", ".ipynb_checkpoints", ".pytest_cache"}


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def copy_checked(s, d):
    os.makedirs(os.path.dirname(d), exist_ok=True)
    shutil.copy2(s, d)
    assert sha(s) == sha(d), "copy mismatch: " + d


# ---------------------------------------------------------------- code/
n = 0
for dp, dirs, fs in os.walk(os.path.join(ROOT, "code")):
    dirs[:] = [x for x in dirs if x not in SKIP_DIRS]
    for f in fs:
        if not f.endswith(CODE_EXT):
            continue
        s = os.path.join(dp, f)
        rel = os.path.relpath(s, os.path.join(ROOT, "code"))
        copy_checked(s, os.path.join(DEST, "code", rel))
        n += 1
print("code/  : %d 个文件" % n)

# ---------------------------------------------------------------- data/
n = 0
for f in sorted(os.listdir(os.path.join(ROOT, "data"))):
    if not f.endswith(".json"):
        continue
    copy_checked(os.path.join(ROOT, "data", f), os.path.join(DEST, "data", f))
    n += 1
print("data/  : %d 个文件" % n)

# ---------------------------------------------------------------- 报告
tot = 0
cnt = 0
for dp, _d, fs in os.walk(DEST):
    if "__pycache__" in dp:
        continue
    for f in fs:
        tot += os.path.getsize(os.path.join(dp, f))
        cnt += 1
print("release: %d 个文件 | %.1f MB" % (cnt, tot / 1024 / 1024))
