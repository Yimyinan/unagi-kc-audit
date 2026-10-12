# paper1 图件改造方案 —— 借鉴 bbag / VEGA 的视觉语法

**性质**：方案文档。**未改动任何图件或正文。**
**日期**：2026-10-10

---

## 0. 目标：一句话

把"对照必须与注入形式匹配"从**散落在 SI 的表格**变成**一张主图 + 一套统一视觉语法**，
同时用「图换表」回收篇幅，直接回应 9 月评审"把分级证据压成二元"那条。

---

## 1. 现有资产（已核实）

| 目标 | 现文件 | 现状 |
|---|---|---|
| 重画 Figure 2B | `figures/fig5_injection_path.pdf` | 已有注入链图 |
| 四检查协议 | `figures/fig0_three_check_flowchart.pdf` | **决策树**形态 |
| 跨架构配对底子 | `figures/figS22_norman_threeway.pdf`（60 KB） | **已含 paired 面板**（scVI $1.02\pm0.15$ $n{=}4$；family-4 $0.868\pm0.032$ $n{=}5$） |
| 剂量–响应 | `figures/fig4_prior_activity.pdf` | $\lambda$ 扫描，三点都平 |
| 约束/理论 | 无（散在正文与 Fig 5 几格） | 需新建 |
| 工作流 | `figures/fig0_workflow.pdf` + Box 1 文字 | Box 未图化 |

SI 表：`tab:scvi`(supplementary.tex:266)、`tab:norman`、`tab:norman_null`。

**好消息**：第 1 项（配对主图）的底子**已经在 `figS22` 里了**——它已经是 paired 设计，
只需换形态与配色，不需要新算数据。

---

## 2. ⚠️ 一处必须先纠正的表述

你的第 2 项写：

> scVI 的 `labels_key` 是 no-op

**这与论文当前证据不符，写进图里会引来审稿人。** 实测：

- `labels_key` 在 `main.tex` / `star_methods.tex` / 两份 SI 中**出现 0 次**。论文用的措辞是
  "cell-type clusters as a **decoder-side categorical covariate**"（`star_methods.tex:96`）。
- 论文的实际结论**更强也更有意思**：把同一批 k-means 聚类注册为 **decoder 侧协变量后，
  审计确实动了**。`supp_addendum.tex:549` 原文：

  > "Because the covariate enters the decoder as an additional input, part of the readout
  > response is **architectural**; what the audit establishes is that a conditioning which
  > **actually reaches the decoder** does move the audit."

**所以正确的对照语义是：**

| 路线 | 结果 |
|---|---|
| curated GRN → 每基因权重 | **无投递**（权重层全空、分支 DEAD） |
| scVI → decoder 侧协变量 | **投递了，审计动了**，但增益部分是架构的 |
| VEGA 增益 | 被 **degree-preserving shuffle** 复现 |

这比"no-op"更有力：它证明**投递与否是分界线**，而不是"先验都没用"。
建议图注直接沿用 `supp_addendum` 的措辞。

**另外**：你说的 `figS22_norman_threeway.pdf` 实际挂在 **Supplementary Figure S12**
（`supplementary.tex` 里 caption 写的是 `Supplementary Figure~S12`），
文件名与图号不一致——重画时顺手统一，避免引用时对不上。

---

## 3. 六项方案，按「收益 ÷ 成本」排序

### ① 配对差异主图（你的第 1 项）—— **最高收益**

**做**：新 `fig_crossarch_paired.pdf`，三列形态：

```
(A) 配对分布            (B) 家族匹配 null 带        (C) 分级标注
  violin + 连线            shaded band                 逐架构结论
  真条件 vs 随机          真实点 vs null 区间          （非二元）
```

**底子**：`figS22` 的 paired 面板 + `tab:norman` / `tab:norman_null`。
**消掉的评审意见**：分级证据被压成二元。
**回收篇幅**：S11–S13 的表可移入 SI 附录或直接删除，主文相应段落缩短。
**风险**：低——数据已在，只需换表现形式。

### ② 四检查改配对面板（你的第 2 项）—— **次高收益**

**做**：`fig0_three_check_flowchart.pdf` 决策树 → **三列配对图**：

```
注入路线          ×   形式匹配的 null         ×   该路线结果
curated GRN            家族匹配 shuffle           无投递
VEGA 增益              degree-preserving shuffle  增益被复现
scVI 协变量            ——（架构）                审计动了，部分架构
```

**为什么值得**：这是论文最想说的那句话的**视觉证明**，比决策树直接。
**风险**：低（无新数据），但需要一张新图脚本。

### ③ 剂量–响应标"实际工作点"（你的第 3 项）—— **高性价比**

**做**：在 `fig4_prior_activity.pdf` 的 $\lambda$ 扫描上加一条竖线/虚线，
标出**该先验实际生效的强度**（对应 bbag 那条粉色 sparsity 线）。

**收益**：结论从"扫的三个点平"升级为"**在实际工作点上也平**"。
**成本**：极低——只需在现有扫描数据里定位真实工作点。
**前置**：需要确认该工作点的**可计算定义**（例如使权重层达到某占用率的 $\lambda$）。

### ④ 约束方程面板（你的第 4 项）—— **最高说服力，成本最高**

**做**：新一张理论面板，写清"目标函数允许什么 / 不允许什么"：

| 机制 | 约束 | 后果 |
|---|---|---|
| ranking transform | 相对重加权 $\le 1.99\times$ | 表达性封顶 |
| ENCODE 先验 | 无方向列 | 无法表达抑制 |
| 拓扑丢弃 | 边权未进损失 | GRN 结构不可达 |

**收益**：把"prior 到不了 loss"从**证据陈述**变成**可推导结论**。
**成本**：高——需要把散在正文的论证形式化成约束式，并保证每条约束都有出处。
**风险**：中——若某条约束推导不严密，会成为攻击点。建议先出草稿给你审。

### ⑤ 工作流图化（你的第 5 项）—— **中等，取决于投稿定位**

**做**：Box 1 的"五行代码 + 四检查"做成流程图（data→matrix→embedding→randomisation→optimisation）。
**注意**：**Patterns 已放弃**（你早先的决定），所以"Resource 类文章靠截图传播"这个理由
**不再适用**。这条的收益取决于最终投哪本刊。
**建议**：**降级为低优先**，除非确定投 Resource 型刊物。

### ⑥ 首图注入位置解剖条（你的第 6 项）—— **值得做，成本低**

**做**：`fig2_z_spectrum_3tissue.pdf` 上方加一条"先验能进入的三个位置（loss / 架构 / 数据）"
的解剖条，标注本文审计了哪些、各自结论。

**收益**：读者**一开始**就明白标题 "never reached the loss" 说的是**投递问题**，
不是"先验本身无效"。这是防御性最强的一处。
**成本**：低（一条横向示意图）。

---

## 4. 统一视觉语法（你的贯穿性建议）—— **采纳**

```
粉色 = 生物先验（真条件）
绿色 = 形式匹配的随机对照
成对 = 每个面板都是「观测 vs 匹配 null」
角标 = 通过 / 失败
```

**收益**：Figure 2B（协议）、Figure 4（IPF）、Figure 6（VEGA）看起来是
**同一台仪器的四次使用**，而不是四个独立实验。

**实现注意**：现有 `figure_palette.py` 已用 Okabe-Ito（色盲安全）。
粉色/绿色需**在该色板内选取**，否则会破坏已有的色盲安全约束——现有
`check_figure_palette.py` 会拦。**这一点不能妥协**。

---

## 5. 执行批次（2026-10-10 最终，按作者决定）

**作者决定**：⑤ 删除；**③ 删除**；④ 进 SI（**已完成**）。

| 批次 | 内容 | 状态 |
|---|---|---|
| **1** | ①配对主图 + ②四检查配对面板 | 可做——数据已在，**未动** |
| **2** | ⑥首图解剖条 | 可做——低风险，**未动** |
| ~~3~~ | ~~③剂量–响应工作点~~ | **已删除**，见 §9 |
| **4** | ④约束面板 | ✅ **已写入 SI**：`docs/si_extended_methods.tex` §AII |
| ~~5~~ | ~~⑤工作流图~~ | **已删除**，见 §5.1 |

### 5.1 两项删除的理由

**⑤ 工作流图**：前提是"Patterns Resource 靠截图传播"，**Patterns 已放弃**，前提失效。

**③ 剂量–响应工作点**：作者裁定——"**先验实际生效我们现在没有办法定义，不宜出现**"。
我实测后同意，且证据支持这个裁定（§9）：扫描数据里没有占用率轴、只有三个 $\lambda$、
而论文的 "target" 是 $\kappa_{\mathrm{gap}}=2$ 而非占用率。**定义一个说不清的量并画进图里，
比不画它更糟。**

### 5.2 ④ 的落地位置与取舍
**已写入** `docs/si_extended_methods.tex` 新增 **§AII "What the objective could express,
had the prior arrived"**，紧接 §AI（方向信息被丢弃）之后、文献表之前。

**没有进正文**——理由：L1 投递主张已由实证承担，L2 是反事实加固；
而正文 7,468 词偏长。放 SI 保留价值且不动正文长度。

**构建验证**：`si_extended_methods.pdf` **19 页**（原 18），errors=0 / overfull=0 / undefined=0，
新增节标题与 `1.99`、`KLF1` 均已渲染。

**一处我未做**：$1.99\times$ 的**闭式推导**。正文只有结论，我不臆造；
SI 里明写 "a closed-form derivation of the constant is not given here"。
若要写成等式，需你给推导形式。

---

## 9. ③ 已删除：为什么"先验实际生效"当前不可定义

作者裁定删除，并给出理由："先验实际生效我们现在没有办法定义，不宜出现。"
实测证据支持该裁定，三条：

### 9.1 扫描数据不含占用率

`data/ptvae_human_ipf_ln_v2/optionB_lambda_scan.json` 每个 $\lambda$ 只记录：

```
seed, architecture, kappa, C, trH, C_H, tau, cos_lat, cos_head, R
```

**没有占用率、非零数、稀疏度**——无法从现有产物定位"达到某占用率的 $\lambda$"。

### 9.2 只有三个点，且 $C$ 已平

```
lam0.1  BN k=0.0100 C=96.73   LN k=0.0211 C=95.65
lam1    BN k=0.0090 C=96.87   LN k=0.0199 C=95.80
lam10   BN k=0.0104 C=96.72   LN k=0.0240 C=95.38
```

跨**两个数量级**，$C$ 变化 $<0.5$ 个百分点（正文 `main.tex:209` 已如实写）。
加点需要**重训**。

### 9.3 概念冲突

正文的 "target" 是 **$\kappa_{\mathrm{gap}}=2$**（`main.tex:386`），**不是**权重层占用率。
故"该先验实际生效的工作点"在论文里**目前没有已定义的概念**。

**结论**：删除。若要恢复此项，需先有一个**可计算且已定义**的"生效强度"概念——
这属于新增工作量，不在本轮范围。



---

## 6. 关于篇幅（那件真正挡投稿的事）

「图换表」确实能回收篇幅，但**要老实说**：把 S11–S13 的表移走只回收约 **0.3–0.5k 词**，
而正文 **7,468 词**的问题**主要不在图注，在 Results 的叙述密度**。

**图件改造能帮你把段落讲得更短**（因为图承担了论证），但它**不是**长度问题的解药。
长度决策仍需你定：砍哪一段、砍多少。

---

## 7. 需要你决定的三件事

1. **第 ⑤ 项是否直接删除？**（Patterns 已放弃，我倾向删除）
2. **第 ③ 项"实际工作点"如何定义？** 我可以用"使权重层达到某占用率的 $\lambda$"，
   但需你确认这是否符合你对"该先验实际生效"的定义。
3. **第 ④ 项约束面板**要不要我先出**文字草稿**（不画图），你审推导后再进图？

---

## 8. 我不建议借的（与你一致）

- bbag Fig 2 圆形领域普查 —— 只审 3 个 family，做不出也不必要
- bbag Fig 5 特征重要性 —— null 是关于可预测性比值的，无可比 ground truth
- VEGA latent 热图 / UMAP —— 会把注意力拉回**已降级**的三种子 $p=0.25$ 那一半

**最后一条我特别同意**：正文刚把那一半降级，**视觉上不该再抬回来**。


---

## 10. 最终状态（2026-10-10 结案）

### 10.1 已入稿

| 图 | 文件 | 位置 |
|---|---|---|
| **Figure 1** | `fig_entry_points.pdf` | 首图：先验能进入的三个位置，标注标题所指的 Entry 1 |
| **Figure 5** | `fig_route_null.pdf` | 替换原决策树四面板：route × 匹配 null × 结果 |
| **Figure 6** | `fig_crossarch_paired.pdf` | 配对跨架构，graded 非二元 |

正文现有 **8 图**（fig:entries / zspect / workflow / kc_phase / injpath / crossarch /
prior_activity / crossimpl），8 个 label 全部被引用。

### 10.2 篇幅决定：**接受现状（选项 E）**

作者裁定不再削减。依据是**按 float 剔除后的真实散文字数**：

```
正文散文合计         6,457 词
图注                约 1,600 词
实测计入口径          8,079 词
```

**结论**：6,457 词散文对这篇的证据量不算冗长；测得的 8k 主要由**图注密度**贡献，
而图注换来的是论证透明度。若要减，最有效的是精简图注而非正文。

### 10.3 两项删除

- **⑤ 工作流图**：前提（Patterns Resource）已失效
- **③ 剂量–响应工作点**：作者裁定"先验实际生效无法定义，不宜出现"；实测支持（扫描无占用率轴、
  仅三点、论文 target 是 κ_gap=2 而非占用率）

### 10.4 ④ 约束面板：已进 SI（§AII）

`docs/si_extended_methods.tex` §AII，19 页，0 error。**未进正文**——L1 投递主张已由实证承担，
L2 是反事实加固。**1.99× 的闭式推导未给**，SI 明写"not given here"。

### 10.5 过程中我犯的错误（留档）

| # | 错误 | 发现方式 | 修法 |
|---|---|---|---|
| 1 | ① 首稿把家族匹配 null 带当基准（带宽 0.381–1.723，含 unity） | 画完看图 | 带子画淡 + 改报配对统计量 |
| 2 | ② panel A 三行重叠；panel B 文字压标题 | 看图 | 重排 |
| 3 | **执行 B 时删掉两张图**（fig:prior_activity、fig:crossimpl） | **构建后 undefined reference 0→4** | 从 HEAD 恢复；新边界以 heading 或 float 为界；写盘前断言 figure/label 数不变 |
| 4 | 用「到下一个 section」做切割边界 | 第 3 条 | 同上 |
| 5 | 早期按 section 统计字数把图注算进散文，误得 876 词 | 重算 | 提取时剔除 float |

**第 3 条最值得记**：如果构建没有报 undefined reference，我会提交一个少两张图的稿子。
**"删内容后没有报错"不等于"没有删错"**——需要一个独立的结构不变量（这里是 figure/label 计数）来兜底。
