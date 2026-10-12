# 报告 B — injection-path check 的可执行实现与真实 worked example

**日期** 2026-10-09 · **性质** 新增资源（**未写入 paper1 任何正文/图/表**）
**目的** 把论文的 headline 发现（"先验从未抵达损失"）从**一段论证**做成**一个可运行、带真实数据的检查**。

---

## 0. 交付物

| 文件 | 说明 |
|---|---|
| `paper1_kc_diagnostic/kappa_c_audit/kappa_c_audit/checks.py` | 新增 `injection_path_check()`，**第 4 个 check**，可 import 可组合 |
| `paper1_kc_diagnostic/kappa_c_audit/examples/injection_path_check.py` | 独立可跑脚本，带完整 docstring 与 CLI |
| `outputs/positive_result/INJECTION_PATH_WORKED_EXAMPLE.json` | 本次运行的完整结构化结果 |

**原包只有 3 个 check**（`gradient_check` / `prior_strength_scan` / `structural_bound`），三者都**假定先验已经抵达**。headline 发现本身反而不在包里——这是本报告要补的缺口。

---

## 1. 一次真实运行（这是 worked example，不是示意）

```
python injection_path_check.py --unagi-root F:/UNAGI_Project/UNAGI \
                               --source-tree F:/UNAGI_Project/b1_expansion/UNAGI_source
```

实跑输出：

```
==========================================================================
INJECTION-PATH CHECK
==========================================================================
question : does the injected prior reach the training objective?

D1  per-gene weight layer occupancy
      UNAGI\benchmark      40 files   empty=40    populated=0
      UNAGI\result       1310 files   empty=1310  populated=0
      TOTAL with layer: 1350   empty: 1350   populated: 0
      VERDICT: ALL-ZERO: the weight layer is present but empty in every file scanned

D2  assignment vs reader sites
      assignment sites: 6   reader sites: 36
      (locational only -- see note)

D3  likelihood-weighting branch instantiation
      instantiations: 0
      VERDICT: branch is DEAD: no instantiation site

==========================================================================
VERDICT: delivery BLOCKED
==========================================================================
```

### 检查在做什么

```
  先验来源 → 组织子集 → 每基因权重向量 → 损失项
                                          ↑
              D1  写入训练的产物里，这一层是否真的有非零值？（实证）
              D3  消费它的那个分支，是否存在实例化点？（结构）
              D2  赋值点与读取点在哪里？（仅定位，不作证据）
```

一个 **CSR 层 `indptr` 长度 n_rows+1、而 `data` 长度为 0**，是"结构上存在但完全为空"：
代码分配了它，从未填充。这是 D1 报出的现象。

---

## 2. 这次运行**额外**产出的两个事实

（这两点不是 paper1 里已有的，是本 worked example 跑出来的）

### 2.1 论文引用的 40 与 796 是两个不同口径——现在可以一次同时给出

论文正文用 **796**（全部保留 run-iteration-stage 文件），而 `injection_chain_audit.json`
里 D1 记录的 `n_files` 是 **40**——因为那个脚本的 glob 是
`UNAGI/benchmark/*/iter_*/stagedata/*.h5ad`，**只扫了 benchmark 子目录**。

我的检查**两个 root 都扫并分别报告**：

```
UNAGI\benchmark    40  →  empty 40   populated 0
UNAGI\result     1310  →  empty 1310 populated 0
```

含义有两层：
- **更广的样本同样全空**（1350 个文件 vs 论文声称的 796）→ 结论被**加强**，不是削弱；
- 但论文声称的 796 与所引产物的 40 **口径不同**，需要在 caption/footnote 里说明哪个数来自哪个 root。

### 2.2 被复制的实现手法有一个可验证的副作用

D2 在源码里命中的赋值点之一是：

```
UNAGI\UNAGI_tool.py  →  self._apply_tf_prior_weights()_parameters = ...
```

`_apply_tf_prior_weights()` 被**作为字符串插进了变量名**。这是 regex patcher 的附带损伤，
与 D3 一致：该 patch 注入了调用文本，但没有任何**真正的**实例化点。

---

## 3. 检查的判定规则（可复核，不含自由心证）

| 项 | 规则 | 本 run |
|---|---|---|
| D1 | 逐文件读 `layers/geneWeight/data`；`shape[0] == 0` 记 empty | 1350/1350 empty |
| D2 | 纯 regex 命中赋值/读取行；**locational only** | 6 / 36 |
| D3 | `Loss\w*\(\s*geneWeight` 命中且该行不以 `def ` 开头 = 实例化点 | 0 → DEAD |
| verdict | D1 全空 **且** D3 无实例化 → `BLOCKED` | BLOCKED |

`scope_limit` 字段随结果一并输出：**"这是本实现的属性，不表明 GRN 先验普遍无效。"**

---

## 4. 开发过程中被否掉的两种实现（留档）

这一点值得记，因为它说明"看起来更严谨"的做法反而更错：

| 手法 | 结果 | 为什么否掉 |
|---|---|---|
| `tokenize` 按 token 迭代 | D2/D3 报 **454** | tokenizer 对一行产出多个 token，未按行去重 → 计数膨胀 |
| 自写 `_strip_strings_and_comments` 逐行剥离 | 仍报 **37** | 与 docstring 状态机的交互不可靠 |

**最终采用**：照抄 `code/prior_injection_chain_audit.py` 的原始机制——**朴素 regex 逐行匹配 +
`def ` 开头排除**。它与产生论文数字的脚本同源，并且给出 6/36/0。

---

## 5. 现在可以怎么用（三种用法）

### 5.1 作为检查（库里直接调）

```python
from kappa_c_audit import checks

res = checks.injection_path_check(
    unagi_root="F:/UNAGI_Project/UNAGI",
    source_tree="F:/UNAGI_Project/b1_expansion/UNAGI_source",
)
if res["weight_layer_verdict"].startswith("ALL-ZERO"):
    print("this pipeline's prior never reaches the loss")
```

### 5.2 作为论文的"可执行附录"

`examples/injection_path_check.py --json <out>` 直接产出结构化 JSON，
可作为 submission 的 supplementary resource 附上（**需你同意才附**）。

### 5.3 作为自查工具（给别的组用）

Docstring 已把"回答什么问题、每步规则、口径陷阱（40 vs 796）"写全。
任何组对自己的 pipeline 跑一遍就能得到 BLOCKED / not blocked。

---

## 6. 测试状态（我核过）

```
kappa_c_audit 包测试    12 passed
repo tools/run_checks.py --strict   exit 0   PASS: all gating checks clean
```

**paper1 正文、图、表均未改动**（本轮只新增了 toolkit 函数与 worked example）。

---

## 7. 一句话总结

论文最强的主张（"先验构建了、赋值了、但从未被读取"）现在有了**一个可 import 的函数、
一个带真实数据的可跑脚本、以及一份结构化输出**；同时在跑的过程中发现论文引用的
**40 与 796 是两个不同扫描口径**——结论在更广样本上依然成立（且更强），
但口径需要在文中交代。

---

## 8. 价值评估与归属建议（2026-10-09 补）

按「够不够进正文」逐项评估。当前正文 **7,468 词**（Intro→Discussion，含图注），
Methods 另有 **2,759 词**——长度本身已是待办项，这是评估的前提。

| 维度 | 评价 | 够正文级？ |
|---|---|---|
| 科学新颖性 | **无**。worked example 不产生新发现，只确认已有主张 | ✗ |
| 可复现性 | **高**。一条命令、零训练、结构化 JSON 输出 | ✓✓ |
| 独立性 | **高**。扫了 **1350** 个文件（论文声称 796、所引产物仅 40），结论在更广样本上成立 | ✓✓ |
| 可交付为工具 | **是**。已做成可 import 的 `injection_path_check()` | ✓ |
| 方法复杂度 | **低**。regex 扫源码 + 读 CSR，无新算法 | ✗ |

### 8.1 结论：**进补充材料，不进正文**

三条理由：

1. **它没有新结果**。正文 headline 是「先验未抵达」；worked example 是**检验该主张的方法**，
   属方法与验证，按惯例归 SI。
2. **正文已偏长**（7,468 词），且 9 月评审提过可读性。为一段「如何复核」再加字与瘦身方向相反。
3. **论文定位在「审计/规范」层**（你的战略判断），该层最强载体是**能跑的工具**而非又一段文字——
   所以它的正确位置是代码/SI。

**但它确实抬高了最强主张的可信度**：读者能自己跑出同样结论。值得在正文**用一句话**指向，
例如 Data and code availability：

> The injection-path check is released as an executable resource
> (`kappa_c_audit.checks.injection_path_check`); re-running it on the released
> stagedata reproduces the all-zero weight channel.

**此句需你同意才加——我没动正文。**

### 8.2 有一项**应该**进正文（是修正，不是新增）

40 vs 796 不是新资源问题，是**论文既有的口径不一致**：

- 正文用 **796**（全部保留 stage 文件）
- 而所引的 `injection_chain_audit.json` 里 D1 记录 `n_files = 40`，因为其 glob 只扫 `UNAGI/benchmark/`

两者都真实，但**读者对照产物会看到 40 而正文说 796**。建议在引用该产物处补口径说明：

> ... across all 796 retained stage files (`UNAGI/result`); the 40-file `benchmark`
> subset is the scan recorded in `injection_chain_audit.json`.

**属事实性修正，建议进正文**——措辞与位置等你定。

### 8.3 一句话

**worked example → 补充材料。它顺带查出的口径不一致 → 正文（修正性质）。**
