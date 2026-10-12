# Paper1 全面自检报告（2026-10-08）— 修正版

本报告记录一次 8 维度的系统性验证。**重要**：初版报告把两个"不一致"判为 P0 缺陷，
经追查源产物后**两者都被推翻**；真正的缺陷在别处。这里同时记录我的判断错误。

---

## 0. 结论摘要

| 类别 | 数量 | 详情 |
|---|---:|---|
| **真实缺陷（已修）** | **2** | Table S8 溢出页面 19 cm；元数据门禁 18 处假阳性 |
| **我误判为缺陷（已推翻）** | **2** | 见 §2 —— 两组数字**都正确**，我的验证脚本有 bug |
| P1 工具失效（已修） | 1 | 图名守卫测试长期失败 |
| 核查为"设计如此" | 2 | provenance 门禁、图名错位 |

---

## 1. 真实缺陷（已修）

### 1.1 Table S8 溢出页面约 19 cm —— **最严重的实际缺陷**

`supplementary/table_s8_jacobian_audit.log` 报：

```
Overfull \hbox (1371.43735pt too wide) in paragraph at lines 2--15
Overfull \hbox (57.45253pt too wide)   in paragraph at lines 11--12
```

1371 pt ≈ **19 cm** —— 表格横向跑出页面。原因：`table_s8_body_33.tex` 末尾有一整段
**用 `\multicolumn{7}{l}` 包裹的长脚注**，而 `l` 列**不会换行**，整段成为一个不可断的盒子。

`code/latex_silent_linter.py` 早已正确预报此类风险（"spans 7 cols with no wrapping column"），
但它混在大量 archive 命中里，此前未被处理。

**修法（两处，保持一致）**：
1. 生成器 `code/make_tabS8_33.py`：把长脚注改为**短的表格内尾注**（一行内可容）；
2. `supplementary/table_s8_jacobian_audit.tex`：把完整说明移出 `tabular`，
   作为 `\noindent{\footnotesize …}` **全宽段落**（可正常换行）。
3. 用生成器**重新生成** `table_s8_body_33.tex`，确保手改不会与生成器脱钩。

**结果**：`supplementary.log` **overfull = 0**；独立表格的 1371 pt 溢出消失。

### 1.2 元数据门禁 18 处假阳性（含扫描范围 bug）

`tools/check_submission_metadata.py` 此前报 **18 处**占位符、跨 9 个文件，其中大部分是
旧时点快照（`2026-09-03-18-47-34\…`）、`_infra\…`、`output\…` —— 这些**永远无法填写**，
构成永久噪声。其模块文档声称排除 `snapshots/archive/releases`，但未覆盖上述目录。

**修法**：`EXCLUDED_DIRS` 增加 `_infra`、`output`、`_archive_20261008`；
新增正则排除**日期形态目录**（`^\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}$`）。

**结果**：18 处 → **3 处**，且**关键功能未受影响** —— 仍在役的
`paper1_kc_diagnostic\submission_metadata.tex` 两个哨兵照旧被命中，`--strict` 仍 exit 1。
（注意：`_infra\backups\` 里那份 `submission_metadata.tex` 是**不完整备份**
——目录里只有这一个文件、且其 `paper1`/`paper2` 子目录为空，保留它并不构成有效冗余。）

---

## 2. 我误判为缺陷的两项 —— 追查源产物后推翻

### 2.1 P0-1「Spearman −0.11 vs −0.50 矛盾」→ **不矛盾，两组都正确**

**证据**：`data/dissociation_jacobian_ridge.json`（由 `code/closing_gaps_power_and_dissociation.py`
依 `data/kappa_c_pr_comparison.json` + `data/unagi_jacobian_reaudit_33.json` 生成）内含：

| 字段 | 值 | 正文 |
|---|---:|---|
| `concordance.spearman_overall` | **−0.1059** | −0.11 ✓ |
| `per_tissue.hemo.spearman` | **−0.0378** | −0.04 ✓ |
| `per_tissue.neural.spearman` | **−0.3091** | −0.31 ✓ |
| `concordance.direction_disagreements` | **17** | 17/33 ✓ |

**四个数字逐一吻合**，无一陈旧。

**两者为何不同**：该脚本用的是 **raw 逐配置值**（`pr_ratio_W` 0.692–1.378 × `kappa_jac_A`），
而 `table_s8_body_33.tex` 的 −0.50 用的是**归一化 κ**（`kappa_r2` 0.873–1.003 × `kappa_jac_A`，
我用 33/33 配对独立重算得 **−0.499**）。**同一对估计量、两种统计量。**

**我的错误**：
- 我重算时写的 `J = {"A": {}, "B": {}}` 与填充逻辑**用了同一个 key**
  （`for v in ("A","B"): k = "kappa_jac_" + v` 却填进 `J[v]` 后又在别处取错），
  实际一直在算 A；
- 更根本的是，我拿**归一化 κ**去复现正文，而正文用的是 **raw 值** —— 从起点就比错了对象。

### 2.2 P0-2「8/33 vs 12/33 计数矛盾」→ **两个数属于两个估计量，都正确**

同一 artifact：

| 字段 | 值 |
|---|---:|
| `dissociation_ridge.product_median` / `product_within_25pct_of_unity` | 0.6527 / **12** |
| `dissociation_jacobian.product_median` / `product_within_25pct_of_unity` | 0.6543 / **8** |

正文 L128 说的是 **Jacobian**（前句即 "Recomputed with the analytical decoder Jacobian"），
8/33 ✓；Ridge 是 12/33。两者都对，只是原文未点明估计量。

**我在初版报告里说的"应为 12/33"是错的**——我把 Ridge 的数按到了 Jacobian 的句子上。

### 2.3 已做的澄清性修改（不引入新数字）

虽然两者都正确，但**并列阅读仍会误解**（Table S8 的 ρ、正文的 ρ；8/33 与 12/33 未点明估计量）。
已做两处纯澄清：

- **Table S8 说明段**：补上"On the raw per-configuration values the same pair gives ρ=−0.11
  with opposite signs in 17 of 33 configurations (the value quoted in the Limitations),
  so the two statistics are reported separately" —— 让两个 ρ 在同一处对齐。
- **正文 Limitations**：`Spearman −0.11` → **`Over the raw per-configuration values … Spearman −0.11`**，
  并补一句"(On the normalised κ the same pair gives ρ=−0.50; the two correlation statistics are
  reported separately and neither substitutes for the other.)"
- **正文 L128**：`8/33 within 25%` → **`under the Jacobian … 8/33 within 25%; Ridge gives median 0.65, 12/33`**。

---

## 3. P1：图名守卫测试长期失败（**已修**）

`code/test_figure_name_guard.py` 用 AST 只抽取 `OWNED_ELSEWHERE` 与 `save()`，
但 `save()` 依赖模块级 `fp`（`import figure_palette as fp`）与 `_GEN_W`：
先 `NameError: fp`，补后又 `NameError: _GEN_W`。**该守卫从未被真正验证过。**

两文件均为 2026-10-01，我未改动过 → **既有缺陷**（git 未启用，无法查历史）。

**已修**：注入 `fp` 桩与 `_GEN_W`。现 **4/4 PASS，exit 0**。

---

## 4. 核查为"设计如此"（初版报告一度误判）

### 4.1 `check_provenance.py` 报 12/24 PASS —— 门禁在正常工作
它比对 `decoder_out/<cfg>/iter_4` 与 `decoder_unified/loading/<cfg>.npy`（**审计真源**）。
24 个目录里 12 个不一致 → 那 12 个是**另一次流水线的 run**。证据：
`build_trajectory.py` 文档开头即 "**Provenance-verified pairs**" 并只列通过者；
**没有任何在役图件脚本读 `decoder_out`**（`figures/make_*.py` 命中 0）；
审计真源是 **iter_4 stagedata**（`kappa_c_pr_comparison.json` 的 `source_of_truth` 字段写明）。

### 4.2 图文件名"与图号错位" —— 有意设计
存在权威映射 `zh_translation/label_map.json`，另有 5 个在役系统消费这些文件名
（`postprocess.py`、`coverage_pdf.py`、`_insp_py/verify.py`、`build_label_map.py`、
`generate_main_figures.py`）。`*__UMBRELLA_LEGACY_DO_NOT_USE` 是守卫的**预期产物**，已归位。

---

## 5. 全部通过的验证项

| 维度 | 方法 | 结果 |
|---|---|---|
| 引用完整性 | `tools/check_citations.py` | **43 bibitems ↔ 43 cite keys，DEAD=[], BROKEN=[]** |
| 文本完整性 | `tools/check_text_integrity.py` | **28 文件 0 findings** |
| SI 图表引用 | `tools/si_reference_map.py` | **S1–S17 全被引用，none unreferenced** |
| Table 1 四区间 | 重算 `s1f_0of33_recheck.json` | 0.873–1.003 / 0.950–1.003 / 0.873–0.957 / 0.893 → **四者全部精确吻合** |
| 0/33、κ_jac 区间 | `unagi_jacobian_reaudit_33.json` | `n=33, n_locked=0, κ_A∈[0.684, 51.608]` |
| 回归复现 | 同上 | `n=15, max_abs_deviation=4.82e-05` |
| 功效/ICC/MDE | `closing_gaps_power_and_bound.json` | 0.985 / 0.347 / 0.381 / 4.43 / 0.160 / 0.509 |
| 结构上界 | `structural_bound_sensitivity.json` | 0.110 与 0.9995 均在 |
| dissociation | `dissociation_jacobian_ridge.json` | 0.774 / 0.950 / 29 / 21 / r 0.122 / slope 0.065 |
| 19/33 展开计数 | artifact | jac A float < 1 者 13；Ridge 12 → 正文 13/33 与 12/33 ✓ |
| 注入清单 | `outputs/geneWeight_recovered_inventory.json` | 796 / 29 / 0 |
| **全整数反查** | 正文 276 个整数逐一在产物中查找 | **276/276 命中，0 孤儿** |
| 小数反查 | 113 个 ≥2 位小数 | 21 个未命中**全是 DOI/arXiv 号**；4 个 IPF 值为 95.6477/95.7979 的舍入 |
| 构建健康 | 4 份 log | undefined=0、errors=0；**supplementary overfull=0**；`??`=0 |

---

## 6. 追加修复：`supp_addendum` 的 8 处 overfull（已完成）

首轮报告把这 8 处记为"既有、未处理"。第二轮已全部修掉，并顺带查出 2 个真实数据缺陷：

| 位置 | 溢出 | 根因 | 修法 |
|---|---:|---|---|
| L52–64 | 61 / 49 / 25 pt | `\texttt{iter\_*/stagedata/*.h5 ad}` —— **文件名中间有空格**（真实路径是 `.h5ad`） | 去掉空格（**同时修掉一个拼错的路径**） |
| L69–92（表 A） | 122 pt | `lccccc` 全无换行列；"seed clusters" 列含 `3 (123×5, 42×9, 456×4)` | 改 `\footnotesize` + `p{}` 列 |
| L283–294（表 B） | 190 pt | `@{}lllr@{}` 无换行列，且内容列长达 343 pt | 改 `\footnotesize` + `p{}` 列，并按**实测宽度**定列宽（472.3 pt 文本宽） |
| L359–366 | 12 pt | 长 `\texttt{...p0\_4\_real\_grn\_second\_architecture.py}` 等不可断行文件名 | 收紧行宽 + `\allowbreak` |
| L441–447 | 61 pt | 同上（`p0\_4\_in\_vivo` 路径） | 同上 |
| L730–731 | 84 pt | 长段落行 | 同上 |

**过程中修正的两个自伤**：
1. 先用 `\usepackage[htt]{hyphenat}` 解决断行，但该包会尝试对 Courier 做连字，
   **新引入 23 条 `T1/pcr/...` font-shape 警告**。改用 `\sloppy` + `\emergencystretch=5em`，
   警告归零且断行效果相同。
2. 表 B 列宽首次给得过窄（`p{2.6}p{2.4}p{9.2}`），内容反而溢出。
   改为**先用 LaTeX `\settowidth` 实测各单元格宽度**再定列宽，一次到位。

**另一处真实数据缺陷**：表 B 的 `geneWeight` 行写 **"40/40 files"** —— 又是旧的 40 口径；
已改为 **"all 796 stage files"**，与全文一致。

**同时清理**：`table_s8_jacobian_audit.tex` 的简介段包进 `sloppypar`，
并把 `5\times10^{-5}` 改为与产物一致的 **`4.8\times10^{-5}`**。

### 结果（五份文档全部归零）

| 文档 | overfull | undefined | errors |
|---|---:|---:|---:|
| `main.log` | 0 | 0 | 0 |
| `supplementary.log` | 0 | 0 | 0 |
| `supp_addendum.log` | **0**（原 8） | **0**（原 23 font warn） | 0 |
| `docs/si_extended_methods.log` | 0 | 0 | 0 |
| `supplementary/table_s8_jacobian_audit.log` | 0 | 0 | 0 |

`code/latex_silent_linter.py` 仍会报 3 处 `\multicolumn` 启发式提示
（`table_s1_audit_body`、`table_s6_body`、`table_s8_body_33`），但该 linter 自身的
check 逻辑就是"看编译日志有没有 Overfull" —— **`supplementary.log` 的 overfull = 0**，
说明这些提示已不成立（属启发式误报）。

---

## 7. 排版/排版专项检查（第二轮新增）

LaTeX 不会报错的排版问题需要自己查。写了一个 typography linter 扫描
（tie 间距、破折号、重复词、图注句号、未转义下划线、引号形态、省略号、
**以及 `\texttt{}` 里点名但磁盘上不存在的路径 = 拼写检查器**）。

### 7.1 查出的真问题（全部已修）

| # | 问题 | 位置 | 修法 |
|---|---|---|---|
| 1 | **`positive\_result/z\_spectrum\_all.csv` 路径错误** —— 该 csv 实际在 `data/`，且全文路径约定就是 `data/` | `supp_addendum.tex` | 改为 `data/z\_spectrum\_all.csv`（文件确实存在，是我改动前的旧路径） |
| 2 | **6 个直引号 `"null"`** 会渲染成直角引号而非 `` ``null'' `` | `supp_addendum.tex`（3 处） | 改为 `` ``…'' `` |
| 3 | **括号不匹配**：“Recommended reporting template” 段有一个**孤立的右括号**：`…latent reorganisation). The four-check…` | `supp_addendum.tex` L416–420 | 改写该段，括号配平、补句号 |
| 4 | **55 处浮动体引用缺 `~`**：`Supplementary Figure S1`（65 处无 tie）与 `Supplementary Figure~S1`（32 处有 tie）**两种混用**，同一引用在一处不断行、另一处会断行 | main / supplementary / supp_addendum / extended | 统一加 `~` → 现在 **bare = 0** |
| 5 | 一个 `\subsection*` 行首多一个空格 | `docs/si_extended_methods.tex` L242 | 去掉 |

### 7.2 误报（我的 linter 先前判错，已修正判定）

| 表面问题 | 实际情况 |
|---|---|
| `supplementary.tex` 3 个"直引号" | 是 `\"{a}` / `\"{o}` **德文变音符命令**；linter 未识别 `\"` 导致误报 |
| 5 处 `3-4` / `5-6` "数字区间" | 是 `\cmidrule(lr){3-4}` 的**列区间参数**，非数值范围 |
| 24 处 `\cite{...},` "cite 后紧跟标点" | 这是**正常 LaTeX 写法**，不是缺陷 |
| 4 处 `participation-ratio ratio` | 语义正确（r 的 ratio），只是词组拗口；**未改**（改词超出排版范畴） |
| `[T1/pcr/...] undefined` × 23 | 是我自己引入 `hyphenat` 造成的字体警告，**已改用 `\sloppy` 移除** |
| `log` 里的 "undefined" | 全部是 Font Warning；PDF 里 `??` = 0，**无未解析引用** |

### 7.3 最终状态

| 文档 | overfull | undefined | errors | `??` |
|---|---:|---:|---:|---:|
| `main` | 0 | 0 | 0 | 0 |
| `supplementary` | 0 | 0 | 0 | 0 |
| `supp_addendum` | 0 | 0 | 0 | 0 |
| `docs/si_extended_methods` | 0 | 0 | 0 | 0 |
| `supplementary/table_s8_jacobian_audit` | 0 | 0 | 0 | 0 |

排版一致性：**tie 引用 98 处，bare 0**；**真实直引号 0**（`submission_metadata.tex` 注释里的 6 个不进 PDF）。

**安全备份**：`_infra\backups\tie_fix_pre_20261008\`（tie 修改前 5 份源文件快照）

---

## 8. 常驻校验：把"混合形式"错误变成可自动发现

本轮新增 `tools/check_form_consistency.py` + `tools/run_checks.py`。

### 8.1 为什么需要它

四处缺陷同一个根因：**同一个量在不同函数形式或不同口径下被并列引用**。

| # | 缺陷 | 形式冲突 |
|---|---|---|
| 1 | `125` vs `796` stage files | 窄口径 vs 全臂口径 |
| 2 | `40/40` vs `796/796`（Fig. 4C） | 神经嵴子集 vs 所有臂 |
| 3 | `15/15` vs `33/33`（Fig. 2 + 3 处正文） | 扩展前 vs 扩展后 |
| 4 | `1.47 / 9-of-33`（Fig. S10） | **校正形式 W × trace 形式 Z**，两种形式相乘 |

第 4 类最难：**数字对产物是对的**（`product_r2` 真的是 1.4722），**但产物本身是不自洽的混合**。纯文本 grep 永远查不出来——必须拿数据重算。

### 8.2 它做什么

**F1 — 乘积/比值断言必须匹配某个"同形式"统计量。**
从产物建构全部同形式组合，逐一对齐正文引用的中位数与 `N/33 within 25%`：

| 形式 | 中位数 | log25 | lin25 |
|---|---:|---:|---:|
| corrected(W) × corrected(Z) | **0.72** | 8/33 | 12/33 |
| trace(W) × trace(Z) | 1.72 | 8/33 | 10/33 |
| product_tr（作者字段） | 1.72 | 8/33 | 10/33 |
| **MIXED corrected(W) × trace(Z)** | **1.47** | 9/33 | 9/33 |

若某引用值**只**匹配 MIXED 行 → 报 FAIL。

**F2 — 计数短语必须落在权威集合内**，并对 7 条已知错误措辞做短语级匹配（`125 stage files`、`40 surviving files`、`15/15`、`All 15 configurations`、`40/40`、`18 Ridge-only` 等）。

**F3 — 检测器自校准**：`--calibrate` 拿**归档的修复前版本**跑一遍，要求每条模式都仍然命中。**从不命中的检查等于没有检查。**

### 8.3 结果

```
$ python tools/check_form_consistency.py --calibrate
calibration corpus: 4 archived revision(s)
  \b125\b...[stage files]          fires (1)
  \b40\b surviving files           fires (1)
  all 40 stage files               fires (7)
  \b15/\b15\b                      fires (11)
  All 15 configurations            fires (1)
  \b40/\b40\b                      fires (1)
  \b18 Ridge-only\b                fires (2)
PASS: every stale-phrase pattern still fires on the pre-fix corpus

$ python tools/run_checks.py
text integrity       gating  ok   CLEAN: 0 findings across 28 files
citations            gating  ok   dead/broken = 0
form consistency     gating  ok   PASS
SI reference map     gating  ok   never referenced = none
silent-overflow      adv     FAIL 6 live advisory (heuristic; logs say 0 overfull)
figure-name guard    gating  ok   PASS
metadata gate        adv     FAIL unfilled sentinels (expected until URL/DOI set)
main/supp/addendum/extended logs  gating ok  overfull=0 undefined=0 errors=0
PASS: all gating checks clean
```

`run_checks.py` 把 11 项检查收敛成一条命令（`--strict` 时任一 gating 失败即 exit 1），
并区分 **gating**（提交前必须干净）与 **advisory**（元数据哨兵未填时预期失败）。

### 8.4 本轮它自己抓到的东西

写这个校验器时，F2 的计数校准**在 `supp_addendum.tex` L650 又抓出一处活的陈旧段落** ——
"Injection-path verification" 那一份拷贝（relocated r85）仍写着 **"all 40 surviving stage files of
the neural-crest run"**，而我此前只修了 `latent_residence_conclusion.tex` 与 `si_extended_methods.tex`
里的孪生拷贝。已同步为 **796 files / 29 run–iterations**，并补上归档≠已执行代码的溯源限定。

> 换言之：**校验器上线后第一次运行就发现了一个人工复核漏掉的缺陷。**

---

## 9. 校验工具的作用域：**仅 paper1**

`tools/check_form_consistency.py` 与 `tools/run_checks.py` **只覆盖 paper1**。
2026-10-09 曾临时接入 paper2 做一次性评估，**随后已完整移除**，理由：

- paper1 的提交门禁**不应依赖、也不应报告**另一篇稿件；
- 移除项：`check_paper2()` 函数、`--paper2` 参数、其调用点与退出码耦合；
- 文件内保留一段 scope note 记录该决定，使移除**不是静默的**；
- 移除后 `--calibrate` 与常规 11 项检查**全部照常**（已验证）。

**已核**：paper1 的活源文件中**没有任何路径指向 paper1 之外**（跨稿件引用数 = 0）。
所有 `\input`、`\includegraphics`、以及 `\texttt{code|data|figures}/…` 点名的路径
**全部存在于磁盘**（缺失 = 0 / 0 / 0）。

## 10. 复现

```powershell
cd F:\UNAGI_Project\paper1\paper1_kc_diagnostic
python tools\check_citations.py
python tools\check_text_integrity.py
python tools\si_reference_map.py
python tools\check_submission_metadata.py --strict
python code\latex_silent_linter.py
python code\test_figure_name_guard.py
python code\check_provenance.py
cd supplementary; pdflatex table_s8_jacobian_audit; cd ..   # 看 overfull
```

---

## 11. 方法上的教训

初版报告把两组数字判为矛盾，根因是**我先假定自己的重算是权威，再把"我的值 ≠ 论文值"
解释为论文错了**。实际顺序应当反过来：

1. **先找到该数字的产物**（`dissociation_jacobian_ridge.json` 一直在库里，字段名一目了然）；
2. **再核对统计量的定义域**（raw 值 vs 归一化 κ——这一步我跳过了）；
3. 只有确认产物缺失或产物本身错，才判为论文缺陷。

这次真实的缺陷（Table S8 溢出 19 cm）反而是**机器报出来的**（`latex_silent_linter.py`），
不是我看出来的——说明**先跑项目自带的校验器**比人工审查更有效。

---

## 12. 仍未解决

| 项 | 说明 |
|---|---|
| `[REPOSITORY_URL]` / `[ZENODO_DOI]`（`submission_metadata.tex`） | 提交前必须填；门禁 `--strict` 会拦 |
| `kappa_c_audit/pyproject.toml:21` 的 `[REPOSITORY_URL]` | 会被烘进 wheel 的 METADATA，同样需填 |
| 通讯地址 `2304283133@stu.htu.edu.cn` | 取自仓库备份，未经作者确认 |
| `silent-overflow` 的 6 条 advisory | 启发式提示（`\multicolumn` 无换行列），而编译日志 overfull=0，故当前不成立 |

---

## 13. 数值内容审计（2026-10-09，独立子代理 + 逐条人工复核）

前九节查的是**格式与一致性**。这一节是第一次对**数值主张本身**做对照产物的复核。
方法：读取 `main.tex` / `star_methods.tex` / `supplementary.tex` / `supp_addendum.tex` /
`docs/si_extended_methods.tex`，加载 `data/`、`outputs/positive_result/`、
`outputs/kc_trajectory/`、`UNAGI/result`、`VEGA/audit_out` 下的产物，用 Python 重算。
未重训模型。

### 10.1 已修复（六处，全部有产物依据）

| # | 缺陷 | 位置 | 产物说 | 改为 |
|---|---|---|---|---|
| 1 | IPF 压缩率**三种说法** | `star_methods.tex:134`、`docs/si_extended_methods.tex:100` 写 `95.5–96.1%`；`main.tex` Table 1 表述写 `95.3–96.7%`；同一篇正文 L210 却给出 BN `96.73/96.87/96.72`、LN `95.65/95.80/95.38` | `data/ptvae_human_ipf_ln_v2/optionB_lambda_scan.json` = **95.38–96.87** | 三处统一为 **95.4–96.9%** |
| 2 | Ridge 与 Jacobian 的**范围口径不一致** | `main.tex`："Ridge confines the ratio to 0.73–1.18 while the Jacobian spans 0.68–51.6" | 0.73–1.18 是**造血组织子集**；全 33 配置的 Ridge 范围是 **0.692–1.378** | 改为 `0.69–1.38` 全 33，并注明 `0.73–1.18` 仅为造血子集 |
| 3 | 运行数 `39 / 36` 与 `n=23` | `main.tex:99`、`main.tex` Fig.1 表注、`star_methods.tex` | `data/z_spectrum_all.csv`：**37 runs / 183 行**（造血 24、神经 11、肺 2），图脚本绘制全部 37 | 改为 **37 / 全部 37 / n=24** |
| 4 | "59 份日志中 14 份以 traceback 终止" | `main.tex` | 59 份确认；**16 份含 traceback，但只有 8 份真正以未捕获异常终止**（另 8 份打印后继续跑完） | 改为 **8 份终止（另有 8 份为已捕获）** |
| 5 | `1.75×` 阈值归属错误 | `main.tex` Fig.1 表注 | `0.893/0.5 = 1.79×` 是肺 R9；`1.75×` 是**组内最小值**（神经 R8，`0.873/0.5`） | 注明 1.75× 归神经 R8，肺 R9 为 1.79× |
| 6 | 三个补充材料浮动体**无正文引用** | Fig. S10、Fig. S16、Table S12 | 均只在自身 caption 中出现 | 补上实质性引用（各自说明该图/表支持哪个论断） |
| 7 | 摘要仍写 **39 runs**，与正文/Methods/Fig.1 的 37 冲突 | `main.tex:61`（摘要的 Methods 摘要句）、`supp_addendum.tex:30`、`supp_addendum.tex:695` | `data/z_spectrum_all.csv` = **37 distinct runs / 183 行**；正文 L99、L152 与 Methods 均已为 37 | 三处统一为 **37** |

第 7 条是**子代理的导出报告提醒后才发现**的：它的转换脚本明确写明“摘要仍说 39，而 Results 与修订后的表注说 37”。这暴露一个流程缺口——我先前的第 3 条只改了正文、Fig.1 表注和 Methods，**没有回头检查摘要**，而摘要那句“测量 39 个 run 的协方差谱”与它自己的 Methods 直接矛盾。

### 10.2 **未**修复，因为复核后不成立

独立审计报告的第 3 条称"神经嵴 `sigma_ctrl≈0.77` 有误，应为 0.45–0.55"。
**我复核后判定该指控不成立，故未改**：参考实现 `code/kappa_c_z.py` 自身注释即写
`sd~0.77`，且定义是 family-matched 的 vanilla 控制组标准差
（`van_z.std(ddof=0)`），与子代理从 CSV 取的那组不是同一口径。**不能拿一个未经
同口径复现的重算去改论文。** 这正是第 8 节记下的教训的第二次应用。

### 10.3 待作者决定（审计报告其余各条，我未擅自改）

| 条目 | 问题 | 建议 |
|---|---|---|
| Δκ `+0.34/+1.90/+0.90` | 重算均值为 `+0.565/+2.049/+1.066`，符号结论成立但数值对不上 | 需确认用的是均值、中位数还是某一具体口径 |
| TRRUST 每 TF 靶基因数 | 10 个里 5 个对不上（如 GATA1 50 vs 57、SPI1 61 vs 62），三种口径都复现不出 | 需确认统计口径（是否限定 20,466 基因宇宙） |
| "latent effective rank" 同量两名 | 正文 `-1.14`（PR 口径）vs 补充 S17 `-0.98`（legacy eff_rank 口径） | 正文未标明口径，而全文多处强调 effective rank **就是** participation ratio |
| "两个神经嵴臂贡献 105 个文件" | 105 = R1(70) + R8(35) = 3 个 run / 2 个臂标签；神经嵴文件含 R0 的 126 个共 231 | 措辞需澄清"臂"与"run"的计数 |

**无法用留存产物验证**（未改动）：
`cosθ = 0.45±0.26`（三处一致但找不到产出脚本）、partitioned null 的 `6.5/6.1 pp`
包络（`kappa_v2_partitioned.json` 只存了自配对，spread=0.00）、Jaccard `0.75/0.54/0.33`、
vehicle-vs-prior 的"~20% power"、partition-seed CV 3.0%。
另：`supp_addendum` 把 TRRUST 描述为"direction + PMID + topology"，而该文件
9,396 条边中有 4,325 条 regulation 为 `Unknown`。

### 10.4 结论

**核心主张经falsification尝试后全部存活**：0/33（两种估计量、含全部 38 条轨迹观测）、
796/29 全零先验通道、9,396/795 → 325/11 → 203 漏斗、injection-path 结论、
以及 VEGA / scVI / GEARS / family-4 的全部跨架构统计量（逐一精确复现）。

**另**：`tools/si_reference_map.py` 有一个使该检查**从建立之日起即失效**的 bug——
它把浮动体**自己的 caption** 当作一次引用，因此"定义但从未被引用"的判定
`not fig_ref[n]` 永远为假。已修（区分 caption 行与正文引用，输出 `ORPHANS:` 行），
`run_checks.py` 相应改为解析该行。这也是为什么三个孤儿浮动体能长期不被发现。

---

## 14. 作者复核发现的缺陷（2026-10-09，第二轮）

作者以编辑视角逐条复核后指出 6 项，全部处理：

| # | 缺陷 | 核实 | 处理 |
|---|---|---|---|
| 1 | `\cite{bergen2020}` 被当作 scVI 框架引用（两处） | **成立**。Bergen 2020 = scVelo/RNA velocity（[PubMed 32747759](https://pubmed.ncbi.nlm.nih.gov/32747759/)）；scVI = Lopez 2018（[PubMed 30504886](https://pubmed.ncbi.nlm.nih.gov/30504886/)），已在文中 | 两处改为 `\cite{lopez2018}`。**连带**：`bergen2020` 因此成为唯一未被引用的条目，bibtex 正确拒绝渲染（43→42 条），已删除该 `\bibitem` 并把门禁期望值改为 42 |
| 2 | Figure 3A 图注"All 33 … fall inside the vanilla bands"与 Results（跨家族 23/33、家族匹配 31/33、R13_s123 低 2%、肺不可分类）自相矛盾 | **成立** | 图注改为：33 个配置**都远离**边界（最近 1.75×），并如实给出两个带的落入比例 31/33 与 23/33，注明 R13_s123 低 2%、肺不可分类 |
| 3 | 三个量共用 $\kappa$：trace ratio、κ-C(W) 可预测性比值、STAR Methods 的 Davis–Kahan 稳定比 $\|H_P-H_V\|_2/(\lambda_1-\lambda_2)$，且都用 0.5 作阈值 | **成立** | 未做 287 处裸 `\kappa` 的高风险全局改名。改为在两处显式命名三者：Methods（trace ratio 定义后加脚注）与 Table 1 脚注，明确 $\kappa_{\mathrm{tr}}$ / $\kappa_{\mathrm{pr}}$ / $\kappa_{\mathrm{gap}}$ 各自定义域，并指明 0.5 阈值只对 $\kappa_{\mathrm{gap}}$ 与 $\kappa_{\mathrm{tr}}$ 的 $C>50\%$ 成立 |
| 4 | Synthesis 一句"dissociation reproduces on two further architectures (Figure 6)"过度压缩 | **成立**。Figure 6 是 VEGA 的 delivery/稀疏性对照，那里 readout 反而变强；真正分级在 scVI / GEARS / family-4 | 改为分级表述：VEGA 使对照失效、GEARS 配对决定性（$D=-0.143\pm0.059$）、scVI 协变量臂（$D=+0.12\pm0.06$）、family-4 仅方向性（$z=-1.3$） |
| 5 | `supp_addendum` 说"factor of 26"，正文说 24.8× | **成立，但两者本就是不同量**：$0.5/0.019=26.3$（$\kappa$ 空间的阈值比）vs $449.6/18.1=24.8$（参考谱隙余量） | 两处补注"(a distinct quantity from the 24.8× reference-gap margin)"，不再并列造成误读 |
| 6 | Highlights 第 2 条读起来像 GRN 先验的普遍结论 | **成立** | Highlights 改为"In the audited pipeline no prior-injected model…"，eTOC 同步限定；`PATTERNS_SUBMISSION.md` 的旧标题 `When the prior does not arrive` 改为现标题；`README.md` 删除过期的 NCS 3,500 词预算 |

### 14.1 包内版本混乱（已处理）

根目录同时存在多个可被误当作投稿件的 PDF。已把三个**过期**件重命名加
`_SUPERSEDED_do_not_submit` 后缀：`paper1_kc_diagnostic.pdf`（09-05）、`main_new.pdf`（09-20）、
`main_revised.pdf`（09-05）。

**当前权威件**：`main.pdf`（27 页，OUP 目标为 `submission/main_bib.pdf` 13 页）。

### 14.2 仍未处理（作者提出，本轮未动）

- **长度与可读性**：正文约 7.8k 词（含图注 9.4k）；摘要仍把 delivery check、dual-spectrum、
  band-independent、three-seed bound 全塞进一段。这是 9 月评审就提出、至今未解决的问题，
  属于**重写级**工作，需作者决定取舍。
- **战略建议（作者提出）**：把审计源码与产生 796 个产物的那次运行绑定（commit hash 或运行
  元数据）；把 injection-path check 做成带真实 worked example 的可执行资源。两者都比再修文字
  更能加固文章最强主张。

---

## 15. 评审后整改记录（2026-10-09/10）

作者以编辑视角复核后提出 6 条意见，随后追加两项新增工作（源码绑定、injection-path 可执行资源）
与一轮校验加固。本节按"结论 + 验证方式 + 边界"记录，并**并列反面事实**，避免归档读起来比事实更强。

### 15.1 六条评审意见 → 修改点 → 验证

| # | 意见 | 修改 | 位置 | 验证方式 |
|---|---|---|---|---|
| 1 | `bergen2020` 被当作 scVI 框架引用 | 两处改 `\cite{lopez2018}` | `main.tex` Introduction、Discussion | 独立查证：Bergen 2020 = scVelo/RNA velocity（PubMed 32747759）；Lopez 2018 = scVI（PubMed 30504886） |
| 1′ | **连带**：改完后 `bergen2020` 成为唯一未被引用条目 | 删除该 `\bibitem`（**43→42**）；门禁期望值改 42 | `main.tex` thebibliography；`tools/run_checks.py` | 引用图变悬空才暴露；bibtex 渲染数 43→42 |
| 2 | Figure 3A 图注"All 33 … fall inside the vanilla bands"与 Results 矛盾 | 改为如实口径：31/33（家族匹配）、23/33（跨家族）、R13_s123 低 2%、肺不可分类 | `main.tex` fig:kc_phase 图注 | 重编译 0 error / 0 overfull |
| 3 | 三个量共用 $\kappa$ | **未做** 287 处裸 `\kappa` 全局改名（风险过高）；改为两处显式命名 $\kappa_{\mathrm{tr}}$ / $\kappa_{\mathrm{pr}}$ / $\kappa_{\mathrm{gap}}$，并注明 0.5 阈值只对 $\kappa_{\mathrm{gap}}$ 与 $\kappa_{\mathrm{tr}}$ 的 $C>50\%$ 成立 | `star_methods.tex:68`（脚注）、Table 1 脚注 | 作者已在源文件核过 |
| 4 | Synthesis 一句"reproduces on two further architectures (Figure 6)"过度压缩 | 改为分级：VEGA 使对照失效、GEARS 配对决定性（$D=-0.143\pm0.059$）、scVI 协变量臂（$D=+0.12\pm0.06$）、family-4 仅方向性（$z=-1.3$） | `main.tex` Synthesis | 重编译 0 error |
| 5 | `supp_addendum` "factor of 26" vs 正文 24.8× | 算清是**两个不同量**（$0.5/0.019=26.3$ vs $449.6/18.1=24.8$），两处补注 "a distinct quantity from the 24.8× reference-gap margin" | `supp_addendum.tex`（2 处） | 重编译 0 error |
| 6 | Highlights 第 2 条像普遍结论；包内陈旧件 | Highlights 加 "In the audited pipeline"；eTOC 同步；`PATTERNS_SUBMISSION.md` 旧标题改现标题；`README.md:13` 删过期 NCS 3,500 词预算 | 各文件 | 逐文件核对（作者亦已核） |

**另**：`paper1_kc_diagnostic.pdf`、`main_new.pdf`、`main_revised.pdf` 三个过期件重命名为
`*_SUPERSEDED_do_not_submit.pdf`。

**我此前一处判断错误已撤回**：我曾称 `paper1_full.md` 里 `x_i-x_i` 出现 0 次，据此说作者看的是旧版。
实际是我只搜了无空格形式；文件里确有 `x_i - x_i`（带空格），**作者是对的**。见 15.6。

### 15.2 Report A 与 4.1–4.4 执行记录

**Report A 结论：以当时仓库状态，无法把审计结论绑定到产生它们的运行。** 三处结构性缺失：
审计代码 0/11 被 git 跟踪（`data/` 被 `.gitignore:33` 忽略）；产物无任何运行元数据；
两个产出脚本 mtime **晚于自己的输出**（`kappa_c_pr_comparison.py` 晚 20 天）。

| 步骤 | 动作 | 验证 |
|---|---|---|
| 4.1 | `commit 825dbba`（170 文件）+ `tag paper1-audit-v1` | 提交前扫 secret = 0；**SHA 见 15.6** |
| 4.2 | 新增 `code/provenance.py`；2 个关键产物**重跑**得真实 provenance；7 个**回溯标注** | **payload 最大浮点偏差 4.29e-16**，$n=33$、`n_deep_compression=0` 不变；备份 `_infra/backups/pre_provenance_20261009/` |
| 4.3 | `RUN_LEVEL_MANIFEST.json`：42 runs / 196 run-iteration / 1700 文件 / 120.2 GB / 59 logs | 每个 run-iteration 有 (path,size) tree hash |
| 4.4 | Data availability 写入 tag | 重编译 0 error；渲染确认 |

### 15.3 Report B：injection-path check

**发现该检查原本不在发布包内**——包只有 3 个 check（gradient / strength-scan / structural bound），
**三者都假定先验已抵达**，而 headline 发现恰恰是"没抵达"。

- 新增 `injection_path_check()`（第 4 个 check，可 import）+ `examples/injection_path_check.py`
- 实跑：**1350 文件全空**（benchmark 40 + result 1310），branch **DEAD**（0 实例化点）
- 顺带发现：正文 796 与所引产物的 40 是**两个扫描口径**（该脚本 glob 只扫 `benchmark/`）

### 15.4 tier-2 全量内容哈希 + 两次反向篡改验证

**先实测再设计**：吞吐 1112 MB/s，故全量 120.2 GB 只需 **108 s**——推翻了"代价不合理"的初始判断。

```
RUN_LEVEL_CONTENT_HASHES.json   depth=all, 1759 个 sha256, 120.2 GB, 108.1 s
tools/run_checks.py             默认开启；--quick 跳过 tier 2（10.5 s，整体约 10 s）
```

| 篡改 | tier 1 tree hash | tier 2 content hash |
|---|---|---|
| 把一个 stagedata 文件**挪走** | FAIL ✅ | — |
| 在 `rep.npy` 正中**翻一个 bit**（大小不变） | **ok（盲区）** | **FAIL ✅** |

两次都恢复后回到 `PASS: all gating checks clean`。**分层不是冗余**：单靠 tier 1 漏掉翻 bit，
单靠 tier 2 每次要 108 s。

### 15.5 过程中我自己犯错又修掉的（8 处）

| # | 错误 | 症状 | 修法 |
|---|---|---|---|
| 1 | 1.75× 替换的句子边界扫描撞上小数点 | 留下 `).75$ in $\kappa$.` 残片 | 精确修复 |
| 2 | `caption` 锚点跨换行 | 图注说明**两次都没打上** | 第三次用单行锚点 |
| 3 | `tokenize` 计数未按行去重 | D2/D3 报 454 | 弃用该手法 |
| 4 | 自写字符串剥离 | 仍报 37 | 弃用 |
| 5 | 循环变量 `run` 遮蔽 `run()` 函数 | `UnboundLocalError`，整套检查崩 | 改名 `run_key` |
| 6 | tree hash 键格式与生成器不一致 | **196 个全部误报 changed**（像数据损坏，实为口径不符） | 统一 `stagedata/<file>` |
| 7 | **误判作者**：称 md 里 `x_i-x_i` 为 0 次 | 只搜无空格形式 | 撤回；见 15.6 |
| 8 | 未实测就断言"120 GB 代价不合理" | 实际 108 s | 先测 | 

### 15.6 反面事实（归档必须与之一致）

1. **tag 只表示"当前代码状态"**，不是产生 796 个产物的那次运行——tag message 原文即如此。
   7 个回溯标注的产物带 `attached_retroactively: true`，明确**不构成**"该脚本产生了这些数字"的证据。
   - annotated tag object : `a837c10f9352cda1610753e8241c9c5982e886a3`
   - **commit（引用这个）** : `825dbba699952f61ae46cad0095a696524674fba`
   - 注意：`git rev-parse paper1-audit-v1` 给的是 **tag object**；`git rev-list -n1` 给的是 **commit**。
     两者不同，我先前混淆过一次。
2. **`paper1_full.md` 是派生件**，且转换器历史上会破坏数学装饰。
   - 本轮已修**两处**：① `TEX_ESC` 不再把 `\hat/\bar/\tilde/\dot/\ddot/\vec/\widehat/\widetilde/\overline`
     映射为空（该字典在 `ACCENTS` 分支之前被查询，会短路正确的组合符逻辑）；
     ② 新增输出阶段 `repair_accents()`，把 `math_to_text` 产出的
     `<组合符><命令名><基字符>` 归一化为 `<基字符><组合符>`
     （`\hat{x}` 曾输出 `U+0302 + "hat" + "x"`）。
   - **状态**：`x_i - x_i`（裸）**0 处**、残留畸形重音 **0 处**、`x̂_i - x_i` 正确，
     且**强制重生成后 sha256 不变（幂等）**——证明修复在生成器里而非输出文件里。
     commit `bc41ff6`、`7cc83f7`。
   - **方法记录（比结论更值得留）**：我第一次所谓"修复"只是在**输出文件上手工修补**，
     重跑转换器后 bug 立刻复发；作者追问才暴露。**教训：改派生件必须改生成器，
     否则等于没改。** 同期我连续三次在错误位置猜根因（`consumes` 条件、字符串剥离、
     正则转义），**正确顺序是先打印运行时的真实分支**——这与 15.5 第 6 条
     （tree hash 键格式）同属一类：现象看似数据/逻辑问题，实为表示层/接口不符，
     而两次我都先怀疑数据。
   - **边界**：md 仍不是权威件。审稿/引用/数值核对**一律以 `main.tex` 与 PDF 为准**。
     转换器对少数带重音的矩阵式（如 `*̃*X** - *̃*Z****W**^ᵀ`）排版仍与 LaTeX 不同；
     我只保证**无畸形残留**，不保证每个重音都落在正确的基字符上。
     文件开头已内嵌 banner 声明此点（banner 由转换器输出，重生成不会冲掉）。
3. **09-17 那一轮的因果不可补**——脚本 mtime 晚于产物，物理上已无法回溯；4.2 的重跑只能作为新的既成事实。
4. `[REPOSITORY_URL]` / `[ZENODO_DOI]` 仍是占位符，门禁中为 **advisory**
   （`check_submission_metadata.py` REPORT 模式 exit 0 不拦；`--strict` 才拦）。
   **实际只有两处真实来源**，填一次即可：
   - `paper1_kc_diagnostic/submission_metadata.tex:21,24`（两个 `\newcommand`）
   - `kappa_c_audit/pyproject.toml:21`（wheel METADATA 里烘焙的仓库 URL；Zenodo 描述不可变）
   `submission/main_bib.tex:105` 是**生成件**，填好源头后重新生成即消失。
   正文 `main.tex` 目前**不含**占位符（已集中到 metadata）。
5. 正文 **7,468 词**（Intro→Discussion，含图注）的长度与可读性问题**未决**，属重写级，需作者定夺。

### 15.7 当前状态

```
git     16 commits · tag paper1-audit-v1 = commit 825dbba（tag object a837c10）
suite   PASS: all gating checks clean        strict exit: 0
        默认约 114 s（含 tier 2）；--quick 约 10 s

main.pdf 27pp · supplementary.pdf 76pp · supp_addendum.pdf 31pp · main_bib.pdf 14pp
paper1_full.md 88 KB（42 参考，accent 已修）
logs: main / supplementary / supp_addendum / si_extended_methods  errors=0 overfull=0
      submission/main_bib  errors=0 overfull=1（OUP class 自带）

provenance 交付物（仓库根 _infra/provenance/）：
  AUDIT_CODE_PROVENANCE.json      7.7 KB
  RUN_LEVEL_MANIFEST.json        75.7 KB
  RUN_LEVEL_CONTENT_HASHES.json 177.0 KB
报告：REPORT_A_PROVENANCE_BINDING.md · REPORT_B_INJECTION_PATH_WORKED_EXAMPLE.md
```

### 15.8 仍未闭合清单

- 09-17 轮次因果（不可补）
- `[REPOSITORY_URL]` / `[ZENODO_DOI]`（占位）
- 正文长度与可读性（未决）
- md 与 LaTeX 在少数复杂结构上的排版差异（设计上以 LaTeX 为准）
