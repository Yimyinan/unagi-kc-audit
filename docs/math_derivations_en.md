# Mathematical Derivations — `paper1` (GRN-prior-injected VAE spectral audit)

> Standalone extraction of every quantitative / formal derivation used in *"GRN-prior injection reorganizes single-cell VAE latent spectra without restructuring the decoder readout: a 33-configuration spectral audit"*.
> Source: `main.tex` (Results + Methods) and `supplementary.tex`. Notation follows the paper verbatim; section numbers below mirror the manuscript for traceability.

---

## 1. Notation

| Symbol | Meaning |
|---|---|
| $\mathbf{x}\in\mathbb{R}^G$ | gene-expression input (per cell) |
| $\mathbf{z}\in\mathbb{R}^D$ | latent code (encoder output mean) |
| $\mathbf{X}\in\mathbb{R}^{N\times G}$ | expression matrix over $N$ cells |
| $\mathbf{Z}\in\mathbb{R}^{N\times D}$ | latent matrix over $N$ cells |
| $\mathcal{D}(\cdot)$ | decoder (gene-expression generator) |
| $\mathbf{W}\in\mathbb{R}^{G\times D}$ | **decoder readout** (Ridge regressor $\mathbf{Z}\!\to\!\mathbf{X}$) |
| $\mathbf{H}=\mathbf{W}^\top\mathbf{W}\in\mathbb{R}^{D\times D}$ | **Hessian proxy** of the decoder readout |
| $\mathbf{J}=\partial\mathcal{D}/\partial\mathbf{z}$ | analytical decoder Jacobian |
| $P$ | prior-injected model; $V$ | prior-free (vanilla) baseline |
| $\kappa$ | coupling coefficient (trace ratio); $C$ | global compression (%) |
| $\mathrm{tr}(\cdot)$ | matrix trace; $\mathrm{cov}\,\mathbf{Z}$ | latent covariance |

---

## 2. Decoder readout and Hessian proxy

### 2.1 Ridge regression (closed form)
A piecewise-linear (ReLU) decoder partitions input space into linear regions, so a *single* linear approximation is locally valid. We fit it with a Ridge readout of the decoder Jacobian:

$$
\mathbf{W}
= \argmin_{\mathbf{W}}\;
\bigl\|\tilde{\mathbf{X}}-\tilde{\mathbf{Z}}\mathbf{W}^\top\bigr\|_F^{2}
+ \alpha\,\|\mathbf{W}\|_F^{2},
\qquad \alpha = 1.0.
\tag{1}
$$

Closed-form solution (standard Ridge):

$$
\mathbf{W} = \bigl(\mathbf{Z}^\top\mathbf{Z}+\alpha\mathbf{I}\bigr)^{-1}\mathbf{Z}^\top\mathbf{X}.
\tag{2}
$$

$\tilde{\mathbf{X}},\tilde{\mathbf{Z}}$ are the (StandardScaler-)standardized matrices; $\mathbf{Z}$ is zero-mean/unit-variance standardized, $\mathbf{X}$ is scaled preserving zero mean with unit variance (no mean centering).

### 2.2 Hessian proxy
The fitted readout yields the Hessian proxy

$$
\mathbf{H} = \mathbf{W}^\top\mathbf{W}\in\mathbb{R}^{D\times D},
\qquad
\mathrm{tr}(\mathbf{H}) = \mathrm{tr}(\mathbf{W}\mathbf{W}^\top),
\tag{3}
$$

i.e. the trace of $\mathbf{H}$ is the squared Frobenius norm of the readout — a proxy for decoder response gain.

### 2.3 Why a proxy (not the literal decoder weight)
$\kappa$-$C$ is read from the **Ridge readout of the decoder Jacobian** (the induced response geometry), *not* from the literal decoder weight matrix $\mathbf{W}_{\mathrm{dec}}$. The injection loss acts on $\mathbf{W}_{\mathrm{dec}}$, but the diagnostic reads out the induced response map. The Ridge readout agrees only weakly with the analytical Jacobian as a directional estimator ($\cos\theta = 0.45\pm0.26$); it was therefore validated by recomputing every accessible checkpoint with the analytical Jacobian ($\partial\mathcal{D}/\partial\mathbf{z}$, autodiff-verified, relative error $2\times10^{-8}$). All 15 accessible configurations (45 % of 33) remain subcritical ($\kappa_{\mathrm{jac}}>0.5$) under both estimators.

---

## 3. The $\kappa$-$C$ decoder-readout diagnostic

### 3.1 Coupling coefficient
$$
\kappa = \frac{\mathrm{tr}(\mathbf{H}_P)}{\mathrm{tr}(\mathbf{H}_V)},
\tag{4}
$$
where $\mathbf{H}_P$ (prior-injected) and $\mathbf{H}_V$ (vanilla, same seed & dataset) are Hessian proxies. By construction $\kappa=1.0$ for vanilla.

### 3.2 Global compression
$$
C = \frac{\mathrm{tr}(\mathbf{H}_V)-\mathrm{tr}(\mathbf{H}_P)}{\mathrm{tr}(\mathbf{H}_V)}\times100\%
= (1-\kappa)\times100\%.
\tag{5}
$$
$C>0$ ⇒ compression (Hessian trace reduced); $C<0$ ⇒ expansion. Since $C$ is a signed rescaling of $\kappa$, the two carry identical information in the trace-based audit; *directional* collapse is measured separately by $C_{\text{head}}$ (§9).

### 3.3 Classification thresholds
- **locked (deep compression)**: $C>50\%$ ⇔ $\kappa<0.5$ (band-independent);
- **vanilla-equivalent**: inside the family-matched corrected band (§4) — the G-VAE cross-family reference is $[0.937, 1.064]$;
- **spectrally expanded**: above that band.

---

## 4. Noise calibration — the vanilla-equivalence band

> ★ **Correction — trace form → predictability form (read before using any band).**
> Earlier versions of this document carried the **trace-form** calibration
> ($\kappa = 1.01 \pm 0.11$, band $[0.78, 1.23]$). **That form is retracted.** It inflates the
> hematopoietic between-seed scatter **56-fold** (SD $0.817$ versus $0.015$) and displaces the band
> centre from unity by $24\%$. It is retained here **only as a documented counter-example** and must
> not be used for classification. Canonical source: `supplementary.tex` Table~S6
> (`tab:unagi_band`); raw values `outputs/positive_result/s1h_null_bands.json`. The trace form equals
> the predictability form only when the latent code is whitened.

Comparing independently trained vanilla G-VAE seeds in the **predictability form** ($\sum_g R^2_g$,
both statistics computed from the *same* readouts) gives the null

$$
\kappa = 1.001 \pm 0.032 \quad\text{(mean}\pm\text{SD; 210 ordered seed-pair comparisons, 15 retained hematopoietic checkpoints)},
\tag{6}
$$

so the G-VAE cross-family vanilla-equivalent band is $\kappa\in[0.937,1.064]$ (±2σ). Re-deriving with
$5{,}000$ cells instead of $2{,}000$ gives $1.001\pm0.035$, so the width is **not** subset-sensitive:
the pipeline's readout predictability is stable to $\approx1.5\%$ across seeds.

**Family-matched UNAGI recalibration** (predictability form; bare R0 excluded as a $3\times$ tr$H$
outlier, 17,695 vs 5,402–9,649):

| Tissue | Seed pairs | $\sum_g R^2_g$ (per seed) | Mean $\kappa$ | SD | Band (±2σ) |
|---|---|---|---|---|---|
| Hematopoietic | 12 | 1716, 1692, 1684, 1677 | 1.000 | 0.015 | $[0.971,1.029]$ |
| Neural crest | 2 | 478, 431 | 1.005 | 0.146 | $[0.713,1.298]$ |
| Combined (UNAGI) | 14 | — | 1.001 | 0.043 | $[0.915,1.086]$ |

**31 of the 33** audited configurations fall inside the family-matched corrected band. The
deep-compression conclusion ($C>50\%\Leftrightarrow\kappa<0.5$) is band-independent and remains
**0/33** under *either* band and *either* statistic. Note the corrected UNAGI bands are **narrower**
than the G-VAE cross-family reference, not wider — the pre-correction text asserted the opposite,
which the trace→predictability correction reverses.

---

## 5. Latent-space spectrum and effective rank

### 5.1 Covariance spectrum
For each run/IDREM iteration, centre $\mathbf{Z}$ and compute

$$
\mathbf{C} = \frac{\mathbf{Z}^\top\mathbf{Z}}{N-1}\in\mathbb{R}^{D\times D},
\qquad
\lambda_1\ge\cdots\ge\lambda_D \text{ its eigenvalues}.
\tag{7}
$$

### 5.2 Effective spectral rank (participation ratio)
$$
\text{eff\_rank} = \mathrm{PR} = \frac{(\sum_{i=1}^{D}\lambda_i)^2}{\sum_{i=1}^{D}\lambda_i^2} = \frac{1}{\sum_{i=1}^{D} p_i^2},
\qquad
p_i = \frac{\lambda_i}{\sum_j\lambda_j}.
\tag{8}
$$
Leading-axis share $= \lambda_1/\sum_i\lambda_i$; spectral entropy normalized by $\log D$; total latent variance $=\mathrm{tr}(\mathbf{C})$. (The alternative spectral-perplexity measure $\exp(-\sum_i p_i\log p_i)$ is retained as the legacy `eff\_rank` column in `z\_spectrum\_all.csv` for backward compatibility but is not used in any reported statistic; the audit uses the participation ratio throughout.)

> **Metric convention (now stated explicitly in the manuscript).** The paper reports a *single, consistent* effective-rank measure throughout: the **participation ratio** $\mathrm{PR}=R_{\mathrm{eff}}^{(\mathrm{PR})}=1/\sum_i p_i^2 = (\sum_i\lambda_i)^2/\sum_i\lambda_i^2$, the convention standard in the VAE / single-cell literature (used by the PG-UNAGI latent-spectrum audit, the $\kappa$-$C$-$Z$ diagnostic, scVI cross-architecture validation, `kappa_c_z.py`, `perturbation_benchmark.py` and `kappac`).
>
> The alternative spectral-perplexity measure $R_{\mathrm{eff}}^{(\mathrm{sp})}=\exp(H_{\mathrm{spec}})=\exp(-\sum_i p_i\log p_i)$ is retained as the legacy `eff\_rank` column in `z_spectrum_all.csv` for backward compatibility, but is **not** used in any reported statistic. Because the metric is identical across all the above analyses, absolute effective-rank values *are* directly comparable.

### 5.3 Tissue-dependent reorganization (signatures)
- Hematopoiesis: 17/18 prior runs **compress** (median $\Delta\log(\mathrm{PR})=-0.758$); naive two-sided sign test $p=0.000145$; seed-clustered $p=0.25$.
- Neural crest: no mean compression ($2^3$ factorial, all $|t|<0.3$, $p>0.39$).
- Lung: $n=1$ (anecdotal).
- Crucially $\mathrm{tr}(\mathbf{Z}^\top\mathbf{Z})$ *increases* in essentially every run (including vanilla), so the signature is the **joint** increase in total variance **and** decrease in effective rank.

---

## 6. The $\kappa$-$C$-$Z$ companion diagnostic (latent spectrum)

Audits the latent encoder output $\mathbf{Z}$ with the same covariance spectrum:

$$
\zeta_{\mathbf{Z}}
= \frac{\mu_{\mathrm{ctrl}}-\mathrm{PR}}{\sigma_{\mathrm{ctrl}}},
\tag{9}
$$

where $\mu_{\mathrm{ctrl}},\sigma_{\mathrm{ctrl}}$ are the mean/std of effective rank across **family-matched vanilla controls**. $\zeta_{\mathbf{Z}}$ is invariant to latent rotation and axis permutation; it measures compression relative to a *distribution*, not a single baseline. A run is flagged $\mathbf{Z}$-reorganized at the effect-size threshold $\zeta_{\mathbf{Z}}\ge 1.5$.

Null design: neural crest uses 3 vanilla seeds ($\sigma_{\mathrm{ctrl}}\approx0.77$); hematopoietic (R14) and lung (R10) each have a single vanilla on disk → fall back to a directional sign test. Across 29 runs with $\mathbf{Z}$ data, **22/29** show directional compression while $\kappa$-$C(W)=0/29$.

---

## 7. Grounding in Weyl's inequality

Let $\lambda_1\ge\cdots\ge\lambda_D$ be eigenvalues of $\mathbf{H}$. **Weyl's inequality** bounds eigenvalue perturbation under additive change:

$$
\bigl|\lambda_i(\mathbf{H}_P)-\lambda_i(\mathbf{H}_V)\bigr|
\le \|\mathbf{H}_P-\mathbf{H}_V\|_2.
\tag{10}
$$

The trace $\mathrm{tr}(\mathbf{H})=\sum_i\lambda_i$ gives a cumulative measure of spectral restructuring; the **Davis–Kahan theorem** further bounds eigenvector rotation, connecting trace-level changes to directional reorganization.

---

## 8. Generalized Rayleigh Quotient (GRQ) perturbation

Find a latent direction $\mathbf{v}^*\in\mathbb{R}^D$ maximizing target specificity while minimizing off-target change. With decoder Jacobian $\mathbf{J}$ and target-gene index set $\mathcal{T}$:

$$
\mathbf{v}^*
= \argmax_{\mathbf{v}}\;
\frac{\mathbf{v}^\top(\mathbf{J}_{\mathcal{T}}^\top\mathbf{J}_{\mathcal{T}})\mathbf{v}}
{\mathbf{v}^\top(\mathbf{J}^\top\mathbf{J}+\epsilon\mathbf{I})\mathbf{v}},
\qquad \epsilon=0.01.
\tag{11}
$$

This is the dominant **generalized eigenvector** of $(\mathbf{J}_{\mathcal{T}}^\top\mathbf{J}_{\mathcal{T}},\;\mathbf{J}^\top\mathbf{J}+\epsilon\mathbf{I})$, solved via `scipy.linalg.eigh` in $O(D^3)$. Perturbation applied as $\mathbf{z}'=\mathbf{z}+\alpha\mathbf{v}^*$.

**Specificity** (target vs off-target mean absolute expression change):

$$
\text{Specificity}
= \frac{\frac{1}{|\mathcal{T}|}\sum_{i\in\mathcal{T}}|\Delta x_i|}
{\frac{1}{G-|\mathcal{T}|}\sum_{i\notin\mathcal{T}}|\Delta x_i|}.
\tag{12}
$$

On four G-VAE seeds, decoder traversal along $\mathbf{v}^*$ reaches mean $13.4\times$ erythroid specificity (range 11.3–16.8$\times$). The optimal direction is seed-specific (pairwise $|\cos\theta|\le0.26$).

---

## 9. Opposition budget and directional collapse

Opposition budget (G-VAE):

$$
B = \frac{\Delta\lambda}{\mathrm{tr}(\mathbf{H})},
\qquad
\Delta\lambda = \lambda_1(\mathbf{H}_V)-\lambda_2(\mathbf{H}_V),
\tag{13}
$$

the fraction of vanilla spectral power the G-VAE training must oppose to lock. The originally reported $C_{\text{head}}=77.6\%\pm0.7\%$ (64/64 opposed directions) was **not reproducible** from surviving checkpoints: recomputation yields **0/15** seeds with $C_{\text{head}}>50\%$ → withdrawn.

---

## 10. G-VAE spectral-shaping objective (JVP penalty)

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

with Hutchinson estimator (3 random unit directions $\mathbf{v}$), $\varepsilon=2.0$, $\lambda_{\mathrm{JVP}}$ ramping $8\to12$ across phases. The JVP term is differentiable and penalizes the decoder Jacobian norm — the actual spectral-shaping mechanism. *The $\kappa$-$C$ quantities are diagnostics, not differentiable loss terms in this implementation* (they are computed under no-grad semantics; see §13).

---

## 11. Cell-type-conditional perturbation (CTCP) and shrinkage

CTCP computes $\mathbf{v}^*_c$ per cell type $c$ and is nulled via label-shuffling permutation (test-set excluded). Shrinkage intensity:

$$
\lambda = \frac{n}{n+200},
\tag{15}
$$

where $n$ = cells of the target type; $\tau=200$ derived from expected Frobenius risk. Real vs shuffled specificity: $19.2\times$ vs $19.4\times$, paired $p=0.52$ ⇒ apparent conditional gain is a sample-size rebalancing artifact.

---

## 12. Decoder-direct perturbation (DDP, off-manifold)

From mean latent $\bar{\mathbf{z}}$, decode $\bar{\mathbf{z}}+\alpha\mathbf{v}^*$ at $\alpha\in\{5,10,20,50,100\}$ without re-encoding. Specificity reaches $92\text{–}156\times$ for $\|\alpha\mathbf{v}^*\|\in[5,20]$; non-erythroid output turns negative for $\|\alpha\mathbf{v}^*\|\ge50$ ⇒ off-manifold, an upper-bound stress test, not an on-manifold prediction.

---

## 13. Prior-activity: what $C$ does **not** measure (three-check structural bound)

High $C$ is frequently misread as "the prior was used." Three checks establish otherwise; the third is a **structural bound**.

On human IPF the vanilla top axis already accounts for $91.2\%$ of $\mathrm{tr}(\mathbf{H}_V)$ (gap $9{,}663$ of $10{,}590$). Since all Hessian eigenvalues are non-negative,

$$
\kappa_{\mathrm{gap}}
\le \frac{\mathrm{tr}(\mathbf{H}_P)}{\mathrm{gap}_V}
= \frac{(1-C)\,\mathrm{tr}(\mathbf{H}_V)}{\mathrm{gap}_V},
\quad\text{so}\quad
\kappa_{\mathrm{gap}}\le 0.11\ \text{at}\ C=90\%.
\tag{16}
$$

The training target $\kappa_{\mathrm{gap}}=2$ is therefore **structurally infeasible** on raw-count IPF. Empirically (BN/LN decoders, inert $\kappa$-$C$ term or differentiable JVP penalty) all models converge to $C=95.3\text{–}96.7\%$ with $\kappa_{\mathrm{gap}}=0.017\text{–}0.023$, and a $\lambda$-scan ($0.1,1,10$) is flat ($C$ varies $<0.5$ pp). The collapse is **data-inherent**, not prior-driven.

*The first two checks:* (i) **gradient check** — the $\kappa$-$C$ term is computed from a no-grad Ridge readout (a leaf constant with zero gradient), so it cannot train the model; (ii) **strength scan** — a used prior must scale with weight (it does not).

---

## 14. Statistical corrections

- **Naive two-sided exact sign test** over $k$ runs (e.g. 17/18 compressions, $p=0.000145$).
- **Seed-clustered permutation test** (Supplementary Addendum, Note 6): the 18 (or 29) configurations are *not* independent — they trace to only **3 training seeds** (seed 123: 5; seed 42: 9; seed 456: 4). The test flips each seed cluster as a unit. With only 3 seeds the clustered test attains a maximum one-sided power of $1/8$ (one all-concordant sign-flip out of $2^3$), so a non-significant clustered result ($p=0.25$) does **not** imply an absent effect — the cross-seed directional concordance is the substantive finding.
- **Welch $t$-test** for group comparisons (e.g. pure vs mixed lineage $C$: $p=0.74$).
- **Stratified 5-fold logistic regression** for phenotype/perturbation-identity classification.

---

## 15. Quick reference (all headline formulas)

| Quantity | Formula |
|---|---|
| Ridge readout | $\mathbf{W}=(\mathbf{Z}^\top\mathbf{Z}+\alpha\mathbf{I})^{-1}\mathbf{Z}^\top\mathbf{X}$ |
| Hessian proxy | $\mathbf{H}=\mathbf{W}^\top\mathbf{W}$ |
| Coupling | $\kappa=\mathrm{tr}(\mathbf{H}_P)/\mathrm{tr}(\mathbf{H}_V)$ |
| Compression | $C=(1-\kappa)\times100\%$ |
| Vanilla band (corrected, §4) | $\kappa\in[0.937,1.064]$ (G-VAE cross-family), $[0.971,1.029]$ (UNAGI hematopoietic); trace form $[0.78,1.23]$ is **retracted** |
| Effective rank | $1/\sum_i p_i^2=(\sum_i\lambda_i)^2/\sum_i\lambda_i^2$ — **participation ratio** (§5.2); the perplexity form $\exp(-\sum p_i\log p_i)$ is legacy-only and used in no reported statistic |
| $\kappa$-$C$-$Z$ flag | $\zeta_{\mathbf{Z}}=(\mu_{\mathrm{ctrl}}-\text{eff\_rank})/\sigma_{\mathrm{ctrl}}\ge1.5$ |
| GRQ direction | $\argmax_{\mathbf{v}}\mathbf{v}^\top\mathbf{J}_{\mathcal{T}}^\top\mathbf{J}_{\mathcal{T}}\mathbf{v}\,/\,\mathbf{v}^\top(\mathbf{J}^\top\mathbf{J}+\epsilon\mathbf{I})\mathbf{v}$ |
| Structural bound | $\kappa_{\mathrm{gap}}\le(1-C)\,\mathrm{tr}(\mathbf{H}_V)/\mathrm{gap}_V$ |
| Opposition budget | $B=\Delta\lambda/\mathrm{tr}(\mathbf{H})$ |
| CTCP shrinkage | $\lambda=n/(n+200)$ |
