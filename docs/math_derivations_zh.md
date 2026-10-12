# 数学推导摘录 —— `paper1`（GRN 先验注入 VAE 谱审计）

> 对 *"GRN-prior injection reorganizes single-cell VAE latent spectra without restructuring the decoder readout: a 33-configuration spectral audit"* 一文中全部定量/形式化推导的独立抽取。
> 来源：`main.tex`（Results + Methods）与 `supplementary.tex`。符号与原文逐字一致；下列小节编号与稿件对应以便溯源。

---

## 1. 符号表

| 符号 | 含义 |
|---|---|
| $\mathbf{x}\in\mathbb{R}^G$ | 基因表达输入（单细胞） |
| $\mathbf{z}\in\mathbb{R}^D$ | 潜变量编码（编码器输出均值） |
| $\mathbf{X}\in\mathbb{R}^{N\times G}$ | $N$ 个细胞的表达矩阵 |
| $\mathbf{Z}\in\mathbb{R}^{N\times D}$ | $N$ 个细胞的潜变量矩阵 |
| $\mathcal{D}(\cdot)$ | 解码器（基因表达生成器） |
| $\mathbf{W}\in\mathbb{R}^{G\times D}$ | **解码器读出**（$\mathbf{Z}\!\to\!\mathbf{X}$ 的 Ridge 回归器） |
| $\mathbf{H}=\mathbf{W}^\top\mathbf{W}\in\mathbb{R}^{D\times D}$ | 解码器读出的 **Hessian 代理** |
| $\mathbf{J}=\partial\mathcal{D}/\partial\mathbf{z}$ | 解析解码器 Jacobian |
| $P$ 先验注入模型；$V$ 无先验（vanilla）基线 | |
| $\kappa$ 耦合系数（迹比）；$C$ 全局压缩（%） | |
| $\mathrm{tr}(\cdot)$ 矩阵迹；$\mathrm{cov}\,\mathbf{Z}$ 潜变量协方差 | |

---

## 2. 解码器读出与 Hessian 代理

### 2.1 Ridge 回归（闭式解）
分段线性（ReLU）解码器把输入空间划分成线性区域，因此**单个**线性近似在局部成立。我们用解码器 Jacobian 的 Ridge 读出拟合它：

$$
\mathbf{W}
= \argmin_{\mathbf{W}}\;
\bigl\|\tilde{\mathbf{X}}-\tilde{\mathbf{Z}}\mathbf{W}^\top\bigr\|_F^{2}
+ \alpha\,\|\mathbf{W}\|_F^{2},
\qquad \alpha = 1.0.
\tag{1}
$$

闭式解（标准 Ridge）：

$$
\mathbf{W} = \bigl(\mathbf{Z}^\top\mathbf{Z}+\alpha\mathbf{I}\bigr)^{-1}\mathbf{Z}^\top\mathbf{X}.
\tag{2}
$$

$\tilde{\mathbf{X}},\tilde{\mathbf{Z}}$ 为（StandardScaler）标准化后的矩阵；$\mathbf{Z}$ 做零均值/单位方差标准化，$\mathbf{X}$ 保持零均值、做单位方差缩放（不中心化）。

### 2.2 Hessian 代理
拟合得到的读出给出 Hessian 代理

$$
\mathbf{H} = \mathbf{W}^\top\mathbf{W}\in\mathbb{R}^{D\times D},
\qquad
\mathrm{tr}(\mathbf{H}) = \mathrm{tr}(\mathbf{W}\mathbf{W}^\top),
\tag{3}
$$

即 $\mathbf{H}$ 的迹等于读出的平方 Frobenius 范数 —— 解码器响应增益的代理。

### 2.3 为何用代理（而非字面解码器权重）
$\kappa$-$C$ 读取的是**解码器 Jacobian 的 Ridge 读出**（被诱导出的响应几何），*而非*字面解码器权重矩阵 $\mathbf{W}_{\mathrm{dec}}$。注入损失作用于 $\mathbf{W}_{\mathrm{dec}}$，但诊断读出的是被诱导出的响应映射。Ridge 读出与解析 Jacobian 作为方向估计仅弱一致（$\cos\theta = 0.45\pm0.26$）；因此用解析 Jacobian（$\partial\mathcal{D}/\partial\mathbf{z}$，autodiff 验证，相对误差 $2\times10^{-8}$）对每一可达 checkpoint 重新计算以做验证。全部 15 个可达配置（33 个中的 45%）在两种估计器下均保持临界值以下（$\kappa_{\mathrm{jac}}>0.5$）。

---

## 3. $\kappa$-$C$ 解码器读出诊断

### 3.1 耦合系数
$$
\kappa = \frac{\mathrm{tr}(\mathbf{H}_P)}{\mathrm{tr}(\mathbf{H}_V)},
\tag{4}
$$
其中 $\mathbf{H}_P$（先验注入）与 $\mathbf{H}_V$（vanilla，同种子同数据集）为 Hessian 代理。按构造，vanilla 的 $\kappa=1.0$。

### 3.2 全局压缩
$$
C = \frac{\mathrm{tr}(\mathbf{H}_V)-\mathrm{tr}(\mathbf{H}_P)}{\mathrm{tr}(\mathbf{H}_V)}\times100\%
= (1-\kappa)\times100\%.
\tag{5}
$$
$C>0$ ⇒ 压缩（Hessian 迹减小）；$C<0$ ⇒ 扩张。由于 $C$ 是 $\kappa$ 的有符号重缩放，二者在基于迹的审计中携带完全相同的信息；*方向性*坍缩由 $C_{\text{head}}$ 单独度量（§9）。

### 3.3 分类阈值
- **锁定（深度压缩）**：$C>50\%$ ⇔ $\kappa<0.5$（与带无关）；
- **vanilla 等价**：落在家族匹配的**更正后**带内（见 §4）——G-VAE 跨家族参照带为 $[0.937,1.064]$；
- **谱扩张**：高于该带。

---

## 4. 噪声标定 —— vanilla 等价带

> ★ **更正：迹形 → 可预测性形（用任何带之前请先读这段）。**
> 本文件早先版本携带的是**迹形**标定（$\kappa = 1.01 \pm 0.11$，带 $[0.78,1.23]$）。
> **该形式已撤回。** 它把造血的种子间散布虚高 **56 倍**（标准差 $0.817$ 对 $0.015$），
> 并把带中心从 1 位移 **24%**。此处**仅作反例保留**，不得用于分类。
> 权威出处：`supplementary.tex` 表 S6（`tab:unagi_band`）；原始值
> `outputs/positive_result/s1h_null_bands.json`。迹形只有在潜码被白化时才等于可预测性形。

在**可预测性形**（$\sum_g R^2_g$，两个统计量取自*同一批*读出）下比较独立训练的 vanilla G-VAE 种子，得零分布

$$
\kappa = 1.001 \pm 0.032 \quad\text{（均值}\pm\text{标准差；210 个有序种子对比较，15 个保留的造血检查点）},
\tag{6}
$$

故 G-VAE 跨家族 vanilla 等价带为 $\kappa\in[0.937,1.064]$（±2σ）。若用 $5{,}000$ 个细胞而非
$2{,}000$ 重推，得 $1.001\pm0.035$ ⇒ 带宽**不**敏感于子集：该流程的读出可预测性在种子间稳定到
$\approx1.5\%$。

**家族匹配 UNAGI 重新标定**（可预测性形；剔除裸 R0 这个 $3\times$ tr$H$ 离群点，17,695 vs 5,402–9,649）：

| 组织 | 种子对数 | $\sum_g R^2_g$（逐种子） | 均值 $\kappa$ | 标准差 | 带（±2σ） |
|---|---|---|---|---|---|
| 造血 | 12 | 1716, 1692, 1684, 1677 | 1.000 | 0.015 | $[0.971,1.029]$ |
| 神经嵴 | 2 | 478, 431 | 1.005 | 0.146 | $[0.713,1.298]$ |
| 合并（UNAGI） | 14 | — | 1.001 | 0.043 | $[0.915,1.086]$ |

**33 个受审配置中有 31 个**落在家族匹配的更正后带内。深度压缩结论
（$C>50\%\Leftrightarrow\kappa<0.5$）与带无关，在*两种*带、*两种*统计量下均为 **0/33**。
注意更正后的 UNAGI 带比 G-VAE 跨家族参照带**更窄**，而非更宽——更正前的文字断言正好相反，
这是迹形→可预测性形更正带来的方向性翻转。

---

## 5. 潜变量谱与有效秩

### 5.1 协方差谱
对每个 run / IDREM 迭代，中心化 $\mathbf{Z}$ 并计算

$$
\mathbf{C} = \frac{\mathbf{Z}^\top\mathbf{Z}}{N-1}\in\mathbb{R}^{D\times D},
\qquad
\lambda_1\ge\cdots\ge\lambda_D \text{ 为其特征值}.
\tag{7}
$$

### 5.2 有效谱秩（参与比 participation ratio）
$$
\text{eff\_rank} = \mathrm{PR} = \frac{(\sum_{i=1}^{D}\lambda_i)^2}{\sum_{i=1}^{D}\lambda_i^2} = \frac{1}{\sum_{i=1}^{D} p_i^2},
\qquad
p_i = \frac{\lambda_i}{\sum_j\lambda_j}.
\tag{8}
$$
主导轴占比 $= \lambda_1/\sum_i\lambda_i$；谱熵按 $\log D$ 归一化；潜变量总方差 $=\mathrm{tr}(\mathbf{C})$。（备选的谱 perplexity 度量 $\exp(-\sum_i p_i\log p_i)$ 作为遗留 `eff\_rank` 列保留于 `z\_spectrum\_all.csv` 以兼容旧流程，但不用于任何报告统计量；审计全程使用参与比。）

> **度量约定（已写入正文）。** 论文全程报告**单一、一致**的有效秩度量：参与比 $\mathrm{PR}=R_{\mathrm{eff}}^{(\mathrm{PR})}=1/\sum_i p_i^2 = (\sum_i\lambda_i)^2/\sum_i\lambda_i^2$，即 VAE / 单细胞文献的通用约定（PG-UNAGI 潜在谱审计、$\kappa$-$C$-$Z$ 诊断、scVI 跨架构验证、`kappa_c_z.py`、`perturbation_benchmark.py` 与 `kappac` 均消费 `z_spectrum_all.csv` 的 `pr` 列）。
>
> 备选的谱 perplexity 度量 $R_{\mathrm{eff}}^{(\mathrm{sp})}=\exp(H_{\mathrm{spec}})=\exp(-\sum_i p_i\log p_i)$ 作为遗留 `eff_rank` 列保留于 `z_spectrum_all.csv` 以兼容旧流程，但**不**用于任何报告统计量。由于度量跨分析完全一致，有效秩的绝对值**可**直接比较。

### 5.3 组织依赖的重组织（特征）
- 造血：18 个先验配置中 **17 个压缩**（中位数 $\Delta\log(\mathrm{PR})=-0.758$）；朴素双侧符号检验 $p=0.000145$；种子聚类 $p=0.25$。
- 神经嵴：无平均压缩（$2^3$ 析因，全部 $|t|<0.3$，$p>0.39$）。
- 肺：$n=1$（轶事性）。
- 关键在于，几乎每个 run（含 vanilla）中 $\mathrm{tr}(\mathbf{Z}^\top\mathbf{Z})$ 都*增加*，因此真正的信号是总方差**增加**与有效秩**下降**的**联合**发生。

---

## 6. $\kappa$-$C$-$Z$ 伴随诊断（潜变量谱）

用同一协方差谱审计潜变量编码器输出 $\mathbf{Z}$：

$$
\zeta_{\mathbf{Z}}
= \frac{\mu_{\mathrm{ctrl}}-\mathrm{PR}}{\sigma_{\mathrm{ctrl}}},
\tag{9}
$$

其中 $\mu_{\mathrm{ctrl}},\sigma_{\mathrm{ctrl}}$ 为**家族匹配 vanilla 对照**有效秩的均值/标准差。$\zeta_{\mathbf{Z}}$ 对潜变量旋转与轴置换不变；它度量的是相对于一个*分布*（而非单个基线）的压缩。当效应量阈值 $\zeta_{\mathbf{Z}}\ge 1.5$ 时标记为 $\mathbf{Z}$ 重组织。

零分布设计：神经嵴用 3 个 vanilla 种子（$\sigma_{\mathrm{ctrl}}\approx0.77$）；造血（R14）与肺（R10）各自磁盘上仅 1 个 vanilla → 退化为方向性符号检验。在 29 个有 $\mathbf{Z}$ 数据的 run 中，**22/29** 显示方向性压缩，而 $\kappa$-$C(W)=0/29$。

---

## 7. Weyl 不等式基础

令 $\lambda_1\ge\cdots\ge\lambda_D$ 为 $\mathbf{H}$ 的特征值。**Weyl 不等式**界定加性扰动下特征值的变化：

$$
\bigl|\lambda_i(\mathbf{H}_P)-\lambda_i(\mathbf{H}_V)\bigr|
\le \|\mathbf{H}_P-\mathbf{H}_V\|_2.
\tag{10}
$$

迹 $\mathrm{tr}(\mathbf{H})=\sum_i\lambda_i$ 给出谱重构的累积度量；**Davis–Kahan 定理**进一步界定特征向量旋转，将迹层面的变化与方向性重组织联系起来。

---

## 8. 广义 Rayleigh 商（GRQ）扰动

寻找潜变量方向 $\mathbf{v}^*\in\mathbb{R}^D$ 以最大化靶标特异性、同时最小化脱靶变化。设解码器 Jacobian 为 $\mathbf{J}$，靶标基因索引集为 $\mathcal{T}$：

$$
\mathbf{v}^*
= \argmax_{\mathbf{v}}\;
\frac{\mathbf{v}^\top(\mathbf{J}_{\mathcal{T}}^\top\mathbf{J}_{\mathcal{T}})\mathbf{v}}
{\mathbf{v}^\top(\mathbf{J}^\top\mathbf{J}+\epsilon\mathbf{I})\mathbf{v}},
\qquad \epsilon=0.01.
\tag{11}
$$

这是 $(\mathbf{J}_{\mathcal{T}}^\top\mathbf{J}_{\mathcal{T}},\;\mathbf{J}^\top\mathbf{J}+\epsilon\mathbf{I})$ 的**主广义特征向量**，用 `scipy.linalg.eigh` 以 $O(D^3)$ 求解。扰动施加以 $\mathbf{z}'=\mathbf{z}+\alpha\mathbf{v}^*$。

**特异性**（靶标 vs 脱靶平均绝对表达变化之比）：

$$
\text{Specificity}
= \frac{\frac{1}{|\mathcal{T}|}\sum_{i\in\mathcal{T}}|\Delta x_i|}
{\frac{1}{G-|\mathcal{T}|}\sum_{i\notin\mathcal{T}}|\Delta x_i|}.
\tag{12}
$$

在 4 个 G-VAE 种子上，沿 $\mathbf{v}^*$ 的解码器遍历达到均值 $13.4\times$ 红系特异性（范围 11.3–16.8$\times$）。最优方向是种子特异的（两两 $|\cos\theta|\le0.26$）。

---

## 9. 对抗预算与方向性坍缩

对抗预算（G-VAE）：

$$
B = \frac{\Delta\lambda}{\mathrm{tr}(\mathbf{H})},
\qquad
\Delta\lambda = \lambda_1(\mathbf{H}_V)-\lambda_2(\mathbf{H}_V),
\tag{13}
$$

即 G-VAE 训练为锁定所必须对抗的 vanilla 谱功率占比。原报道的 $C_{\text{head}}=77.6\%\pm0.7\%$（64/64 对抗方向）从现存 checkpoint **无法复现**：重新计算得到 **0/15** 个种子 $C_{\text{head}}>50\%$ → 撤回。

---

## 10. G-VAE 谱整形目标（JVP 惩罚）

$$
\mathcal{L}_{\mathrm{GVAE}}
= \sum_i w_i(\hat{x}_i-x_i)^2
+ \beta\,\mathcal{L}_{\mathrm{KL}}
+ \lambda_{\mathrm{JVP}}\,
\frac{1}{\mathrm{tr}(\mathbf{H}_V)}\,
\mathbb{E}_{\mathbf{v}}\!\left[
\Bigl\|
\frac{\mathcal{D}(\mathbf{z}+\varepsilon\mathbf{v})-\mathcal{D}(\mathbf{z})}{\varepsilon}
\Bigr\|_2^{2}
\right],
\tag{14}
$$

采用 Hutchinson 估计（3 个随机单位方向 $\mathbf{v}$），$\varepsilon=2.0$，$\lambda_{\mathrm{JVP}}$ 跨阶段从 $8$ 升到 $12$。JVP 项可微，惩罚解码器 Jacobian 范数 —— 真正的谱整形机制。*$\kappa$-$C$ 量在此实现中是诊断量，而非可微损失项*（它们在 no-grad 语义下计算；见 §13）。

---

## 11. 细胞类型条件扰动（CTCP）与收缩

CTCP 为每个细胞类型 $c$ 计算 $\mathbf{v}^*_c$，并以标签混排置换（排除测试集）做零分布。收缩强度：

$$
\lambda = \frac{n}{n+200},
\tag{15}
$$

其中 $n$ = 靶标类型细胞数；$\tau=200$ 由期望 Frobenius 风险导出。真实 vs 混排特异性：$19.2\times$ vs $19.4\times$，配对 $p=0.52$ ⇒ 表观条件增益是样本量再平衡假象。

---

## 12. 解码器直推扰动（DDP，流形外）

从平均潜变量 $\bar{\mathbf{z}}$ 出发，不重新编码地解码 $\bar{\mathbf{z}}+\alpha\mathbf{v}^*$，$\alpha\in\{5,10,20,50,100\}$。在 $\|\alpha\mathbf{v}^*\|\in[5,20]$ 时特异性达 $92\text{–}156\times$；$\|\alpha\mathbf{v}^*\|\ge50$ 时非红系输出变负 ⇒ 流形外，属上界压力测试，而非流形上预测。

---

## 13. 先验活性：$C$ **不**度量什么（三检验结构界）

高 $C$ 常被误读为"先验被使用了"。三检验证明并非如此；其中第三项是**结构界**。

在人 IPF 上，vanilla 主导轴已占 $\mathrm{tr}(\mathbf{H}_V)$ 的 $91.2\%$（间隙 $9{,}663 / 10{,}590$）。由于所有 Hessian 特征值非负，

$$
\kappa_{\mathrm{gap}}
\le \frac{\mathrm{tr}(\mathbf{H}_P)}{\mathrm{gap}_V}
= \frac{(1-C)\,\mathrm{tr}(\mathbf{H}_V)}{\mathrm{gap}_V},
\quad\text{故}\quad
\kappa_{\mathrm{gap}}\le 0.11\ \text{当}\ C=90\%.
\tag{16}
$$

训练目标 $\kappa_{\mathrm{gap}}=2$ 因此在原始计数 IPF 上**结构上不可行**。经验上（BN/LN 解码器，惰性 $\kappa$-$C$ 项或可微 JVP 惩罚）所有模型收敛到 $C=95.3\text{–}96.7\%$ 且 $\kappa_{\mathrm{gap}}=0.017\text{–}0.023$，且 $\lambda$ 扫描（$0.1,1,10$）平坦（$C$ 变化 $<0.5$ 个百分点）。坍缩是**数据固有**的，而非先验驱动。

*前两项检验：*（i）**梯度检验** —— $\kappa$-$C$ 项由 no-grad 的 Ridge 读出计算（对模型参数零梯度的叶常量），故无法训练模型；（ii）**强度扫描** —— 被使用的先验应随权重缩放（事实并非如此）。

---

## 14. 统计校正

- **朴素双侧精确符号检验**：对 $k$ 个 run（如 17/18 压缩，$p=0.000145$）。
- **种子聚类置换检验**（补充附录 Note 6）：18（或 29）个配置*不*独立 —— 它们仅追溯到 **3 个训练种子**（种子 123：5；种子 42：9；种子 456：4）。该检验以种子簇为整体翻转。仅 3 个种子时，聚类检验的最大单侧功效为 $1/8$（$2^3$ 中唯一一个全一致的符号翻转），因此非显著的聚类结果（$p=0.25$）**不**意味着效应缺失 —— 跨种子的方向性一致才是实质发现。
- **Welch $t$ 检验**：组间比较（如纯系 vs 混合系 $C$：$p=0.74$）。
- **分层 5 折逻辑回归**：表型/扰动身份分类。

---

## 15. 速查表（全部核心公式）

| 量 | 公式 |
|---|---|
| Ridge 读出 | $\mathbf{W}=(\mathbf{Z}^\top\mathbf{Z}+\alpha\mathbf{I})^{-1}\mathbf{Z}^\top\mathbf{X}$ |
| Hessian 代理 | $\mathbf{H}=\mathbf{W}^\top\mathbf{W}$ |
| 耦合系数 | $\kappa=\mathrm{tr}(\mathbf{H}_P)/\mathrm{tr}(\mathbf{H}_V)$ |
| 压缩 | $C=(1-\kappa)\times100\%$ |
| vanilla 带（更正后，§4） | $\kappa\in[0.937,1.064]$（G-VAE 跨家族）、$[0.971,1.029]$（UNAGI 造血）；迹形 $[0.78,1.23]$ **已撤回** |
| 有效秩 | $1/\sum_i p_i^2=(\sum_i\lambda_i)^2/\sum_i\lambda_i^2$ —— **参与率**（§5.2）；困惑度形 $\exp(-\sum p_i\log p_i)$ 仅为遗留列，不用于任何报告统计量 |
| $\kappa$-$C$-$Z$ 标记 | $\zeta_{\mathbf{Z}}=(\mu_{\mathrm{ctrl}}-\text{eff\_rank})/\sigma_{\mathrm{ctrl}}\ge1.5$ |
| GRQ 方向 | $\argmax_{\mathbf{v}}\mathbf{v}^\top\mathbf{J}_{\mathcal{T}}^\top\mathbf{J}_{\mathcal{T}}\mathbf{v}\,/\,\mathbf{v}^\top(\mathbf{J}^\top\mathbf{J}+\epsilon\mathbf{I})\mathbf{v}$ |
| 结构界 | $\kappa_{\mathrm{gap}}\le(1-C)\,\mathrm{tr}(\mathbf{H}_V)/\mathrm{gap}_V$ |
| 对抗预算 | $B=\Delta\lambda/\mathrm{tr}(\mathbf{H})$ |
| CTCP 收缩 | $\lambda=n/(n+200)$ |
