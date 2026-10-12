# Paper1 (κ-C diagnostic) — analysis code

All scripts run under **Python 3.13** with `numpy`, `pandas`, `matplotlib`
(system interpreter used during preparation: `C:/Users/12901/AppData/Local/Programs/Python/Python313/python.exe`).

## Data-quality gate (read this first)

Several loading matrices in the raw `decoder_out/` tree are **degenerate or
duplicated** and silently corrupt results. `run_registry.py` is the single
source of truth: it maps each logical run name to its authoritative
`decoder_unified/loading/*.npy` file and refuses bad data. **Every analysis
script imports it and resolves matrices through `get_matrix()`** — do not load
`loading_matrix.npy` by bare path.

Documented exclusions (see `run_registry.EXCLUDED`):
- `R15_seed42/123/456` are byte-identical duplicates of the empty-prior `R14_*` files.
- `decoder_out/R13/iter_4` is a degenerate matrix (`||W||=7.57`, column-norm CV 1.96);
  the correct seed-42 full-prior matrix is `decoder_unified/loading/R13_s42.npy`.
- Bare `R0` (neural vanilla) is a legacy run, not the seed-42 vanilla; use `R0_seed42`.
- **Source mismatch (not caught by the degeneracy gate).** `decoder_out/R14/` and its
  alias `decoder_out/R14/vanilla/` are byte-identical to each other but are a
  *different matrix* from the curated seed-42 empty prior
  `decoder_unified/loading/R14_s42.npy` (`||W||` 75.07 / eff. rank 27.07 versus
  71.84 / 27.42). Both pass the degeneracy gate, so the gate cannot detect this.
  The bare names `R14` and `R14/vanilla` were absent from `NAME_MAP`, so
  `get_matrix()` silently fell through to the `decoder_out` copy while the prior
  model resolved to the unified copy — comparing a unified model against a
  non-curated baseline. This inflated the seed-42 wiring statistic from
  `z=+2.85` to `z=+3.18` (and the cross-seed Stouffer from `+1.40` to `+1.59`);
  seeds 123/456 were unaffected because they use the mapped `R14_seed123/456`
  names. **Both bare names are now mapped to `R14_s42`.** Lesson: every logical
  run needs an explicit alias, because a missing alias fails *silently*, not loudly.

Run `python run_registry.py` to print the full quality audit.

## Reproduction order

1. `reconstruct_prior_configs.py` — reconstructs each hematopoietic prior
   configuration's TF set and edge count from `config.py` + TRRUST v2, to
   establish the dose–response gradient (R11 21 → R13 292 edges).
2. `prior_wiring_test.py` — 2×3 specificity control (R13 true / R16 mismatch /
   vanilla) on marrow, plus the neural cortex arm. → `figS12`.
3. `wiring_cross_seed.py` — cross-seed reproducibility (3 seeds × matched vanilla)
   for marrow and neural. → `figS13`.
4. `dose_response_wiring.py` — prior-size dose–response with size-matched random
   null correction. → `figS14`.
5. `prior_specificity_controls.py` — mismatch, neural arm, seed-permutation
   variance, and in-silico KLF1 knock-out / over-expression. → `figS15`.
6. `wiring_vs_compression.py` — internalisation vs latent-spectrum compression,
   seed partial correlation, and the R16 dissociation. → `figS16`.
7. `validate_prior_biology.py` — TRRUST validation + specificity ladder. → `figS11`.
8. `biological_axis_decoding.py` — honest null result (no single latent axis
   encodes a lineage program); motivates the bulk-spectrum κ-C over per-axis reading.
9. `mechanism_prior_on_axis.py` — v* heatmap (dominant axis loads on housekeeping
   features, not prior TFs).
10. `seed_dependence_explain.py` — decomposes each seed's wiring *z* into a per-TF
   coupling gain (the expression-gating mechanism). → `figS17`. Note: the wiring
   *z* is **computed** from the registry inside this script; it was previously
   hardcoded, which silently froze the seed-42 value at the pre-correction `+3.18`.
11. `layer2_pooled_mwu.py` — power reanalysis of the wiring test: within-seed,
   Stouffer, naive pooled, pooled paired gain, and a *conditional* (program-gated)
   design. Shows naive pooling cannot rescue significance, whereas the conditional
   test is strongly significant. → `fig_layer2_power.pdf`.

### Injection-path evidence (Check 4) and prior-information bounds

12. `prior_injection_chain_audit.py` — traces the prior from its source database
   to the training objective: artefacts, call graph, patcher damage, objective
   reachability. Supplies the Check-4 evidence. → `INJECTION_CHAIN_AUDIT.md`.
13. `prior_information_loss.py` — quantifies the four-stage decay of the prior's
   information (raw TRRUST 9,396 edges / 795 TFs; 9,369 / 794 after removing 27
   self-loops; → 325 edges → 203 scalars → ENCODE → `nnz=0`); the source of
   Supplementary Note 6, Table 2. → `prior_information_loss.json`.
14. `hemato_prior_direction_inventory.py` — per-TF inventory of directed targets in
   TRRUST versus the model gene universe; shows only 6 of 12 haematopoietic TFs are
   testable. → `hemato_prior_direction_inventory.json`.
15. `latent_directionality_final.py` — tests whether TRRUST's activation/repression
   labels are recoverable from the decoder readout, with detection-label and
   detection-matched controls. Superseded exploratory versions kept for provenance:
   `latent_directionality_test.py`, `_controls.py`, `_refined.py`. Figure:
   `figures/make_fig_latent_directionality.py`. → `latent_directionality_final.json`
   (Supplementary Note 6, Fig. 2).
16. `probe_retrain_needs.py` — measures the wall-clock cost of a full PG-UNAGI run
   from run metadata, used to bound the retraining design. → `retrain_cost_probe.json`.

> **Caution when merging two-sided p-values into a one-sided z:** always carry the
> sign (`z = sign(stat) × ppf(1 − p_two/2)`). An unsigned conversion in an early
> version of `latent_directionality_refined.py` promoted strongly *negative* TFs
> into the FDR hit set.

## Key corrected results (registry-based)

- Hematopoietic true prior (R13, seed 42) wires hema edges: per-edge MW `z=+3.18`
  (`p=0.0015`); does **not** wire cortex edges (`z=-1.07`, n.s.).
- Cross-seed hema: +3.18 / +2.02 / −2.45 → Stouffer `z=+1.59`, `p=0.11`
  (fragile, not robust).
- Neural cortex prior (R1): **no** wiring (per-seed `z=+0.19/+0.09/+0.19`, Stouffer
  `z=+0.27`, `p=0.79`).
- Dose–response: no monotonic trend (Spearman ρ=−0.20, `p=0.51`); seed explains 89%
  of effect-size variance (perm `p=0.0034`).
- Internalisation ≠ compression: R16 compresses 75% as much as R13 yet wires neither
  edge set; seed partial correlation of wiring vs compression ≈ 0.
- Causal: in-silico KLF1 KO lowers KLF1-edge coupling by 0.357, rest unchanged (−0.004).
