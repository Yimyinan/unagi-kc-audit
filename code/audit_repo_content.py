# -*- coding: utf-8 -*-
"""audit_repo_content.py —— 公开代码库出库前扫描（绝对路径 / 凭据 / 个人信息）。

为什么需要：`REPO_AUDIT_20260923.md` 的"无绝对路径、无凭据"结论只覆盖了**工具包**
（13 个文件）。09-23 装配包把 `code/`(134 .py) + `data/`(54 .json) 也并了进来，
**这两块从未做过同样扫描** —— 公开前必须补。

用法： python audit_repo_content.py [repo_dir]
"""
import io
import os
import re
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else r"F:/UNAGI_Project/paper1/repo_kappa-c-audit"

PATHS = [
    ("Windows 绝对路径 F:", re.compile(r"[Ff]:[\\/](?:UNAGI|Projects|data)")),
    ("Windows 绝对路径 C:", re.compile(r"[Cc]:[\\/]Users[\\/]")),
    ("*nix 绝对路径", re.compile(r"/(?:home|Users|mnt|media)/[A-Za-z0-9_.-]+/")),
    ("原工作区名", re.compile(r"WorkBuddy|2026-09-03-18-47-34")),
]
SECRETS = [
    ("私钥块", re.compile(r"BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY")),
    ("GitHub token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}")),
    ("AWS key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("通用 api_key/secret 赋值", re.compile(r"(?i)\b(?:api[_-]?key|secret|password|passwd|token)\s*[=:]\s*[\"'][^\"']{8,}")),
    ("Zenodo token", re.compile(r"(?i)zenodo.*(?:token|access)")),
]
PERSONAL = [
    ("邮箱", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")),
]

text_ext = {".py", ".json", ".md", ".txt", ".yml", ".yaml", ".toml", ".cff", ".cfg", ".sh", ".in"}
files = []
for dirpath, _dirs, names in os.walk(ROOT):
    if ".git" in dirpath:
        continue
    for n in names:
        if os.path.splitext(n)[1].lower() in text_ext:
            files.append(os.path.join(dirpath, n))

print("扫描 %d 个文本文件于 %s\n" % (len(files), ROOT))

hits = {label: [] for label, _ in PATHS + SECRETS + PERSONAL}
for p in files:
    try:
        t = io.open(p, encoding="utf-8", errors="replace").read()
    except Exception:
        continue
    rel = os.path.relpath(p, ROOT)
    for label, rx in PATHS + SECRETS + PERSONAL:
        for m in rx.finditer(t):
            line = t.count("\n", 0, m.start()) + 1
            hits[label].append((rel, line, m.group(0)[:60]))

for label, _ in PATHS + SECRETS + PERSONAL:
    hs = hits[label]
    flag = "!!!" if label in [x[0] for x in SECRETS] and hs else ("!" if hs else " ")
    print("%-24s %s %d" % (label, flag, len(hs)))
    if hs:
        by = {}
        for rel, line, sample in hs:
            by.setdefault(rel, []).append((line, sample))
        for rel in sorted(by)[:6]:
            print("      %s  (%d 处, 例 L%d: %s)" % (rel, len(by[rel]), by[rel][0][0], by[rel][0][1]))
        if len(by) > 6:
            print("      … 另 %d 个文件" % (len(by) - 6))

secrets_total = sum(len(hits[l]) for l, _ in SECRETS)
paths_total = sum(len(hits[l]) for l, _ in PATHS)
print("\n小结: 绝对路径/环境泄漏 %d 处 | 凭据 %d 处 | 邮箱 %d 处"
      % (paths_total, secrets_total, len(hits["邮箱"])))
sys.exit(1 if secrets_total else 0)
