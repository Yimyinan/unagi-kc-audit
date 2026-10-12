# 报告 A — 审计源码与 796 个产物的绑定

**日期** 2026-10-09 · **性质** 只读调查 · **未改动 paper1 正文**
**目的** 回答一个问题：**现在能否把 paper1 的审计结论绑定到产生它们的那次运行？**

---

## 0. 结论（先说要点）

**不能，以今天的仓库状态不能。** 而且这不是"差一点材料"的程度问题，是三处结构性缺失：

| # | 缺失 | 证据 |
|---|---|---|
| 1 | **审计代码从未进入版本控制** | 11 个产出文件里 **10 个存在、0 个被 git 跟踪**；`paper1/paper1_kc_diagnostic/data/` 被 `.gitignore:33`（`data/`）忽略 |
| 2 | **产物不含任何运行元数据** | `geneWeight_recovered_inventory.json` 的顶层键只有 `local_reconstruction_iterations / runs / total_files / total_nonzero_rows`——**没有时间戳、没有 commit、没有工具版本** |
| 3 | **最高危：两个产出脚本比它自己的输出更新** | `code/kappa_c_pr_comparison.py` 改于 **2026-10-07 19:27**，而 `data/kappa_c_pr_comparison.json` 写于 **2026-09-17 16:47**（晚 20 天）；`prior_injection_chain_audit.py` 与其产物同理 |

第 3 条是关键：**即使今天把代码打了个 tag，也不能证明现在这份代码文本就是当初产生那些数字的代码。**

git 状态本身也说明问题：全仓 **10 个 commit**，最新一次在 **2026-08-09**，而 raw stagedata 产生于 **2026-05-31 → 06-29**、派生产物产生于 **2026-09-17 → 10-07**。**最新 commit 比审计工作早两个月，比 raw 数据晚一个多月——它跟这条证据链无关。**

---

## 1. 已建立的（可用的部分）

我已经把**当前**状态封存成一份可引用的 manifest：

**`F:\UNAGI_Project\_infra\provenance\AUDIT_CODE_PROVENANCE.json`**

内容：

- `git`：HEAD `a6ced8bad27c`、日期 `2026-08-09 09:43:15 +0800`、commit 数 10、`audit_code_tracked: false`
- `code[]`：10 个产出/校验文件的 **sha256 + 字节数 + mtime + 是否被跟踪/被何规则忽略**
- `artifacts[]`：9 个关键产物的 **sha256 + 字节数 + mtime**
- `gaps[]`：自动检出的 **"产出脚本比产物新"** 两处（附 implication 字段）
- `stagedata`：1700 个文件的 **最早/最晚 mtime 与跨度（693 小时）**

这份 manifest 的价值：它把"**今天的代码长这样、今天的产物长这样**"钉死。
它**不能**证明两者之间的因果——那正是缺失的部分。

---

## 2. 证据链的四段，以及断在哪

```
  ① raw 训练输出            ② 派生产物            ③ 产出代码          ④ 论文主张
  UNAGI/result/            data/*.json          code/*.py          main.tex
  1700 文件                9 个关键件            10 个文件           0/33 等
  2026-05-31→06-29         2026-09-17→10-07      mtime 混杂
                             ↑                     ↑                  ↑
                             └──────── ② 与 ③ 之间无可靠绑定 ────────┘
                                        （且 ① 被 git 跟踪但 ②③ 没有）
```

- **① → ②**：`prior_injection_chain_audit.py` / `kappa_c_pr_comparison.py` 消费 ①产生 ②，但脚本 mtime 晚于产物 → **顺序可疑**
- **② → ③**：无 commit、无 manifest、无 run-id
- **③ → ④**：论文**没有**引用任何 commit 或运行标识（`submission_metadata.tex` 里 `[REPOSITORY_URL]`/`[ZENODO_DOI]` 仍是占位符）

---

## 3. 我**没有**做的判断（避免上一轮的错）

- 我**不主张**产物是错的。raw 数据存在、产物存在、数值我先前逐项复算过（0/33、796/29、Table 1 全范围），**它们是可复现的**——只是**不可追溯**。
- 我**不主张** `kappa_c_pr_comparison.py` 被篡改过。更可能的解释是：脚本长期迭代，产物在 09-17 冻结，之后脚本又改过（10-07 那批修改很可能就是为应对审计发现的 corrected/`pr_ratio` 口径问题）。**但"很可能"正是问题所在——无法证明。**

---

## 4. 要补什么才算绑定（按性价比排序）

### 4.1 立刻可做，成本最低（30 分钟）

**给审计代码打第一个 tag 并提交。**

代码就在 `paper1/paper1_kc_diagnostic/code/`、`tools/`，当前**完全未跟踪**。需要注意 `data/` 被忽略——产物应单独处理（见 4.2）。

```powershell
cd F:\UNAGI_Project
# 先确认 .gitignore 不会把代码也吞掉
git check-ignore -v paper1/paper1_kc_diagnostic/code/kappa_c_pr_comparison.py
# 只加代码与工具，不加 data/
git add paper1/paper1_kc_diagnostic/code paper1/paper1_kc_diagnostic/tools
git commit -m "paper1: freeze audit code as used for the 0/33 and 796/29 claims"
git tag -a paper1-audit-v1 -m "Audit code snapshot; see _infra/provenance/AUDIT_CODE_PROVENANCE.json"
```

**但必须同时声明**：由于第 3 条（脚本晚于产物），这个 tag 只能标注为
*"代码的当前状态"*，**不能**写成 *"产生这些产物的代码"*。

### 4.2 让产物自带来源（需要重跑一次，1–2 小时）

在每个产出脚本的写盘处加一个 provenance 块：

```python
PROVENANCE = {
    "script": "code/kappa_c_pr_comparison.py",
    "script_sha256": "<自哈希>",
    "git_head": subprocess.run(["git","rev-parse","HEAD"],...).stdout.strip(),
    "git_dirty": bool(subprocess.run(["git","status","--porcelain"],...).stdout.strip()),
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "inputs": {p: sha256(p) for p in INPUTS},      # stagedata / csv / 上游 json
    "python": sys.version.split()[0],
}
# 写进 JSON 顶层，与数值并列
```

**重跑一次**把这段烘进产物——这是唯一能真正闭合 ②→③ 的做法。
代价：要重新生成 9 个派生产物（**不需要**重训模型，raw stagedata 已在盘上）。

### 4.3 把 raw 训练也钉住（成本最高，收益最大）

`UNAGI/result` **已被 git 跟踪**，但 stagedata 是二进制大文件。建议：

- 用 `git rev-parse HEAD` + `git hash-object` 记录每个 run 的 `stagedata/` 树哈希
- 或产出 run 级 manifest：`run → {commit, config hash, seed, start/end mtime, stage file hashes}`

这能闭合 ①→②，把论文的"可复现"从**结论级**提到**运行级**。

### 4.4 论文侧（一行字，但需你同意）

`submission_metadata.tex` 里填 `[REPOSITORY_URL]` / `[ZENODO_DOI]` 时，
**同时把 tag 名写进 Data and code availability**，例如：

> Audit code frozen at tag `paper1-audit-v1` (commit `<hash>`, dated <date>).

---

## 5. 我建议的措辞（若你要在论文里如实交代）

当前论文对可复现性的表述是**结论级**的（"inputs recomputed from raw stagedata rather than checkpoints"）。
若要更诚实，可加一句限定——**这需要你确认**：

> The audit inputs are recomputed from raw stagedata rather than from checkpoints.
> The audit code and derived artefacts were not under version control while the analysis
> was developed; `AUDIT_CODE_PROVENANCE.json` records their hashes and modification times
> as released.

**我不建议**把它写成"we release the exact code that produced the results"——除非先做完 4.2 的重跑。

---

## 6. 一句话总结

**证据是可复现的，但不是可追溯的。** 补上 4.1（tag）+ 4.2（产物自带 provenance）之后，
"可复现"这个核心卖点才算完全落地；4.3 是加固，4.4 是如实交代。
