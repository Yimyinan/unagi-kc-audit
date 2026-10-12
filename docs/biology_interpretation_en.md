# Biological Interpretation — `paper1` (GRN-prior-injected VAE spectral audit)

> Standalone extraction of the *biological reading* of the results in *"GRN-prior injection reorganizes single-cell VAE latent spectra without restructuring the decoder readout: a 33-configuration spectral audit"*.
> What the numbers *mean* for virtual perturbation, for how VAEs absorb injected structure, and for the GRN-prior assumption. Source: `main.tex` (Results + Discussion) and `supplementary.tex`.

---

## 1. The central thesis: decoder / latent spectral decoupling

The field's default assumption is that **injecting a GRN prior into a VAE organizes the latent space for virtual perturbation**. The audit shows this is only *partially* supported. The core finding is a **decoder/latent spectral decoupling**:

> GRN prior injection reorganizes the **latent encoder-output spectrum** ($\mathbf{Z}$) while leaving the **decoder readout** ($\mathbf{W}$) invariant. The standard decoder-only $\kappa$-$C$ diagnostic therefore *misses* the latent effect entirely.

This is the paper's main biological message: a model can look "unchanged" at the decoder readout yet be substantially reorganized in the geometry it actually uses for perturbation.

---

## 2. Why the decoder-only $\kappa$-$C$ audit is null — and why that is *not* "nothing happened"

A decoder move $\Delta\mathbf{z}$ produces interpretable gene-expression change only if the decoder's response geometry is organized around separable programs. But when $\mathbf{Z}$ contracts, $\mathbf{W}$ **must expand** to reconstruct the *same* expression space (the prior-injected and vanilla models are trained to comparable MSE — a constant-reconstruction-loss constraint). This is a **scalar trace trade-off**:

$$
\mathrm{tr}(\mathbf{W}^\top\mathbf{W})\ \text{rises as}\ \mathrm{tr}(\mathbf{Z}^\top\mathbf{Z})\ \text{falls}.
\tag{*}
$$

So the decoder-centric audit stays null ($0/22$ hematopoietic, $0/10$ neural-crest, $0/1$ lung reach $C>50\%$) **precisely because** the latent space reorganized. The $0/33$ null is the *corollary* of the decoupling, not evidence of an absent effect. $\kappa$-$C$ measures decoder **gain**, not perturbation-readiness.

---

## 3. Tissue-dependent latent reorganization

Prior injection does not act uniformly; it reorganizes the latent spectrum **in a tissue-dependent manner**:

| Tissue | Latent effect | Interpretation |
|---|---|---|
| **Hematopoiesis** | 17/18 prior runs compress effective rank (median $\Delta\log(\mathrm{PR})=-0.758$; naive $p=0.000145$, clustered $p=0.25$) | Prior "wins" geometric territory — latent manifold collapses onto a lower-dimensional, program-aligned subspace. |
| **Neural crest** | No mean compression ($2^3$ factorial, all $p>0.39$) | Cortex prior (R1) mismatches the neural-crest trajectory. |
| **Lung fibrosis** | R9 modestly lower than R10 ($12.35$ vs $13.80$); $n=1$ | Anecdotal, underpowered. |

The effect is **directional and tissue-specific**, not a uniform training artifact.

---

## 4. Biological meaning of latent compression

In hematopoiesis, the hematopoietic prior (R13) is a curated transcription-factor cascade driving the erythroid/lymphoid differentiation trajectory. The latent compression is most parsimoniously read as:

> the latent manifold **collapsing onto a lower-dimensional, program-aligned organizing subspace** — the prior has "won" geometric territory **without restructuring the decoder**.

This is the biologically interesting part: the reorganization is *coupled to real biology*, not a numerical drift.

---

## 5. Prior **content** vs prior **effect** — the purity paradox

A natural hypothesis is that *lineage purity* governs the spectral response. The data falsify this:

- **R111** (pure erythroid TFs, 111 edges): $\kappa=1.71$, $C=-71.4\%$.
- **R107** (mixed erythroid+lymphoid, 107 edges): $\kappa=0.99$, $C=0.7\%$.

A 4-edge (3.6%) difference cannot explain a 72-percentage-point gap in $C$. At the **group level**, pure ($n=8$) vs mixed ($n=13$) configurations do **not** separate (Welch $p=0.74$). The paired contrast implicates **prior content** — not lineage purity per se — as the feature that tracks the spectral response.

Consistent with this, **edge count does not predict the spectral response** (Spearman $\rho=-0.21$, $p=0.70$ across 8 configurations; dose–response ladder 21–292 edges also flat). Latent reorganization is a property of **prior–data alignment**, not prior size.

---

## 6. Internalisation is real but fragile, gated by program-matching

On marrow data, the true hematopoietic prior (R13) wires its curated edges **above** vanilla (per-edge Mann–Whitney $z=+2.85$, $p=0.0044$). But the mismatched cortex prior (R16) wires **neither** its own nor the marrow edges, yet **still compresses the latent spectrum almost as much as R13**. So internalisation is a real but fragile, program-matching-gated event — and it is not driven by prior size.

A **specificity ladder** in seed 42 shows effective-rank compression is *graded by prior–tissue congruence* (vanilla 3.10; empty-vector vehicle 5.19; wrong-tissue neural prior 2.30; curated hematopoietic prior 1.75). Stronger biological specificity → lower effective rank at larger total latent variance.

---

## 7. Cross-architecture validation: the sign flips, the phenomenon holds

On **scVI** (independent VAE, different decoder parameterization), the dissociation reproduces exactly:

- pbmc3k (cell-type structural prior): decoder readout invariant ($\kappa$-$C(W)=0.87$, inside band) **yet** latent expands ($\kappa$-$C(Z)=1.08$; effective rank $12.6\to15.7$, $+24.6\%$).
- The **sign** of the latent response differs from UNAGI: label-conditioning forces the latent to **expand and separate**, whereas the GRN prior compels it to **compress**. But the governing phenomenon is identical — decoder and latent spectra move *independently*.
- A third architecture (plain MLP VAE, decoder-only conditioning, encoder condition-blind "family 4") reproduces the same qualitative pattern on the identical Norman data: decoder readout vanilla-equivalent ($\kappa$-$C(W)=0.95$) while latent contracts ($\kappa$-$C(Z)=0.89$, $-2.8\%$).

The decoupling is therefore a property of **how VAEs absorb injected structure**, not a quirk of one training objective.

---

## 8. The perturbation benchmark: conditioning absorbs the perturbation into the decoder

On the **Norman 2019 GEARS** CRISPRi screen (40,964 cells, 105 perturbations), a third decoder/latent pattern appears:

- decoder readout **adapts** to the expression-modulating prior ($\kappa$-$C(W)=1.22$, $C=-22\%$);
- latent **contracts** ($\kappa$-$C(Z)=0.75$; effective rank $8.80\to6.15$, $-30\%$);
- conditioning on the perturbation covariate **absorbs the perturbation effect into the decoder pathway**, so the conditioned latent no longer encodes perturbation identity above the majority baseline ($0.099$ vs $0.179$ stratified-5-fold accuracy) — the dual of the pbmc3k regime.

**Crucially, the reorganization is biologically coupled**: projecting the latent perturbation-shift through the decoder readout predicts the *direction* of real perturbation-induced differential expression with positive Spearman $\rho\approx0.20$ (range $-0.01$ to $0.35$, $n=8$). So $\kappa$-$C$-$Z$ tracks genuine perturbation biology — a non-circular validation.

---

## 9. GRQ closure: target specificity ≠ response prediction

A **GRQ virtual-perturbation closure** on Norman data sharpens the distinction:

- **GRQ direction**: highly target-specific ($\approx6.5\times$ enrichment over observed perturbed cells) **but does not predict** the real whole-genome response direction ($\rho\approx0$).
- **Mean latent-shift direction**: *does* predict the response ($\rho\approx0.22$), and is **even stronger** from the condition-blind vanilla latent ($\rho\approx0.27$) than from the conditioned latent.

> **Conclusion:** optimizing target specificity is **not** the same as predicting perturbation response, and conditioning the latent on perturbation identity can **weaken** its response-predictive signal.

This is the biologically cleanest statement of the paper's warning: virtual-perturbation tools measure different objectives than the one a biologist cares about.

---

## 10. High $C$ on IPF is data-inherent, not prior activity

On human IPF, the same deep-compression regime ($C=95$–$97\%$) arises with a **fully inert prior** (BatchNorm/LayerNorm decoders; inert $\kappa$-$C$ term *or* differentiable JVP penalty; $\lambda$-scan flat). A structural bound shows the collapse is set by the data: the vanilla top axis already captures $\approx91\%$ of the decoder trace, so the target $\kappa_{\mathrm{gap}}=2$ is structurally infeasible.

**Takeaway for the field:** studies should report a **gradient check, a strength scan, and a structural bound** before attributing spectral collapse to a prior. Without these, $C$ is a *monitoring quantity*, not a mechanism.

---

## 11. Encoder–decoder co-adaptation: seed-specific encoders, convergent decoders

Independently trained seeds have **divergent encoders** in weight space (cross-seed encoder correlation $r<0.02$) yet **converge to near-identical decoder sensitivity maps** (decoder correlation $r=0.96$). Hence the *same* prior yields *different* $\kappa$ across seeds even with functionally identical decoders. The global trace is also blind to trace-preserving rearrangements (the partitioned $K=8$ metric detects no prior-specific reorganisation). This is why the audit reports *directional patterns* rather than universal significance.

---

## 12. The three-check protocol as a philosophy

The paper's operative contribution is turning spectral claims from **rhetorical into verifiable**:

1. **Gradient check** — does the credited term actually have a gradient path to parameters?
2. **Strength scan** — does the spectral response scale with prior weight?
3. **Structural bound** — is the claimed effect even feasible given the data geometry?

Readiness is adjudicated by this protocol (and downstream GRQ readouts), **not** by $\kappa$-$C$ being a readiness metric. Operationally, a model is treated as perturbation-ready only when (i) the latent manifold is low-rank and program-aligned **and** (ii) the decoder readout supports separable perturbation responses. Prior injection in this audit satisfies neither on its own.

---

## 13. Why the $0/33$ null is informative, not a failure

The originally reported G-VAE directional-locking positive control (**21/21**) is **withdrawn**: not reproducible from surviving checkpoints (0/15 seeds reach $C_{\text{head}}>50\%$; fresh 3-seed reproduction locks only 1/3, transiently). The two are different failures:

- the **0/33 null** is a uniform, end-to-end, auditable recomputation (single registry; Jacobian re-audit of 15/33);
- the **21/21** was an unreproducible historical positive control.

Withdrawing it **strengthens**, rather than undermines, confidence in the null.

---

## 14. Bottom line for perturbation biology

- "Inject a GRN prior" is **not, as currently measured, demonstrated** to be a standalone mechanism for perturbation-readiness.
- The decoder-gain null establishes only that the prior does **not** reorganize the decoder *readout*.
- The $\kappa$-$C$-$Z$ companion diagnostic reveals a real, tissue-dependent, biologically-coupled latent reorganization that the decoder-only lens is blind to.
- Virtual perturbation is possible in principle; the three-check protocol and GRQ readouts tell you *whether* and *how* a given model can do it.
