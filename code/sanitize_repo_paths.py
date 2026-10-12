# -*- coding: utf-8 -*-
"""sanitize_repo_paths.py (v2) —— 出库前把**暂存仓库**里的硬编码绝对路径改为环境变量解析。

v1 的三个 bug（已由"全量 py_compile + 复扫"抓出，v2 修掉）：
  ① **`r` 前缀在引号之前**，v1 的匹配从引号开始 ⇒ 前缀残留，`Path(r"C:/…")` 变成 `Path(ros.environ…)`；
  ② **`import os` 插在 `from __future__` 之前** ⇒ `SyntaxError: from __future__ imports must occur
     at the beginning of the file`（8 个文件编译失败）；
  ③ 只覆盖了 `WorkBuddy/2026-09-03-18-47-34` 一个会话目录，漏了 `2026-08-30-…` 等（残留 49 处）。

v2 规则：`C:/Users/12901/WorkBuddy/<任意会话>/…` -> `UNAGI_SESSION`；`F:/UNAGI_Project/…` -> `UNAGI_ROOT`。

用法： python sanitize_repo_paths.py [--apply]
"""
import io
import os
import re
import sys

REPO = r"F:/UNAGI_Project/paper1/repo_kappa-c-audit"
APPLY = "--apply" in sys.argv

ROOT_RE = r"F:[\\/]UNAGI_Project"
SESS_RE = r"C:[\\/]Users[\\/]12901[\\/]WorkBuddy[\\/][0-9][0-9\-]*"
# ③ 本机用户目录的其他用法（如文档里的系统解释器路径 C:/Users/12901/AppData/...）
USER_RE = r"[Cc]:[\\/]Users[\\/]12901"
NIX_RE = r"/(?:Users|home)/12901"

# ① 前缀与引号一起匹配（关键修正）
LIT = re.compile(
    r"(?P<pfx>[rRbBuU]{0,2})(?P<q>[\"'])(?P<path>" + ROOT_RE + r"|" + SESS_RE + r")"
    r"(?P<rest>[^\"']*)(?P=q)"
)


def to_expr(path, rest):
    env = "UNAGI_SESSION" if "WorkBuddy" in path else "UNAGI_ROOT"
    rest = rest.replace("\\", "/")
    base = 'os.environ.get("%s", ".")' % env
    return base + (' + "%s"' % rest if rest else "")


def ensure_import_os(t):
    if re.search(r"(?m)^\s*import os\s*$", t) or re.search(r"(?m)^import [^\n]*\bos\b", t):
        return t, False
    lines = t.split("\n")
    # ② 必须落在 __future__ 之后
    last_future = -1
    for i, ln in enumerate(lines):
        if re.match(r"^from __future__ import", ln):
            last_future = i
    if last_future >= 0:
        lines.insert(last_future + 1, "import os  # noqa: E402  (paths from env; see README)")
        return "\n".join(lines), True
    for i, ln in enumerate(lines):
        if re.match(r"^(import |from )", ln):
            lines.insert(i, "import os  # noqa: E402  (paths from env; see README)")
            return "\n".join(lines), True
    return "import os  # noqa: E402  (paths from env; see README)\n" + t, True


changed = 0
for dirpath, _dirs, names in os.walk(REPO):
    if ".git" in dirpath:
        continue
    for n in names:
        # ★ v2 只处理 .py/.md/.sh ⇒ **漏掉了 data/*.json 里的路径**（残留 39 处）。v3 补上。
        if not n.endswith((".py", ".md", ".sh", ".json")):
            continue
        p = os.path.join(dirpath, n)
        t = io.open(p, encoding="utf-8", errors="replace").read()
        orig = t
        t, n_lit = LIT.subn(lambda m: to_expr(m.group("path"), m.group("rest")), t)
        t, n_b1 = re.subn(ROOT_RE, "<UNAGI_ROOT>", t)
        t, n_b2 = re.subn(SESS_RE, "<UNAGI_SESSION>", t)
        t, n_b3 = re.subn(USER_RE, "<LOCAL_USER>", t)
        t, n_b4 = re.subn(NIX_RE, "<LOCAL_USER>", t)
        added = False
        if n_lit and n.endswith(".py"):
            t, added = ensure_import_os(t)
        if t != orig:
            changed += 1
            if changed <= 6 or n_lit > 3:
                print("  %-44s literals=%-3d bare=%-3d import_os=%s"
                      % (os.path.relpath(p, REPO), n_lit, n_b1 + n_b2 + n_b3 + n_b4, added))
            if APPLY:
                io.open(p, "w", encoding="utf-8", newline="").write(t)

print("\n%s: %d 个文件" % ("已改写" if APPLY else "将改写", changed))
if not APPLY:
    print("re-run with --apply")
