# Confound check: is the participation ratio just a function of tr(Cov Z)?

`tr` grows monotonically with IDREM iteration in essentially every run, including vanilla. If the participation ratio is determined by tr alone, then any 'compression' is a generic training effect, not a prior effect.

- pooled all runs (n=183): log(PR) ~ log(tr), slope=-0.463, r=-0.713, p=1.07e-29
  -> tr alone explains 50.8% of the variance in log PR.

- hemo   (n=118): slope=-0.642, r=-0.898, p=2.82e-43
- lung   (n=10): slope=-1.104, r=-0.968, p=4.48e-06
- neural (n=55): slope=+0.196, r=+0.464, p=3.60e-04

`resid` = log PR after removing the within-tissue tr trend (negative = more compressed than expected for its total variance).

---
## 1. Terminal state (iter_4): does the prior compress?

### hemo

| run | role | PR | eff_rank (legacy) | top1 % | tr | resid |
|---|---|---|---|---|---|---|
| R107_erythroid_lymphoid_seed42 | prior | 3.28 | 5.89 | 51.1 | 228.0 | +0.199 |
| R111_erythroid_temporal_seed123-1 | prior | 1.38 | 2.12 | 85.0 | 863.6 | +0.186 |
| R111_erythroid_temporal_seed42 | prior | 2.08 | 3.87 | 68.3 | 457.6 | +0.189 |
| R111_erythroid_temporal_seed456 | prior | 2.09 | 4.03 | 68.2 | 294.4 | -0.087 |
| R11_KLF1_TAL1_seed123 | prior | 1.57 | 2.71 | 79.6 | 464.0 | -0.085 |
| R11_KLF1_TAL1_seed42 | prior | 2.39 | 4.59 | 63.3 | 292.9 | +0.043 |
| R11_KLF1_TAL1_seed456 | prior | 1.73 | 3.17 | 75.4 | 483.1 | +0.042 |
| R12_GATA2_GFI1B_PAX5_seed123 | prior | 1.64 | 2.81 | 77.6 | 521.1 | +0.033 |
| R12_GATA2_GFI1B_PAX5_seed42 | prior | 2.74 | 4.99 | 57.5 | 318.5 | +0.234 |
| R12_GATA2_GFI1B_PAX5_seed456 | prior | 1.54 | 2.69 | 80.3 | 590.1 | +0.053 |
| R13_KO_KLF1_seed42 | prior | 2.80 | 5.50 | 57.8 | 212.7 | -0.004 |
| R13_OE_KLF1_seed42 | prior | 2.27 | 4.47 | 65.3 | 251.2 | -0.108 |
| R13_hemato_trueprior_seed123 | prior | 1.48 | 2.41 | 81.9 | 850.3 | +0.248 |
| R13_hemato_trueprior_seed42 | prior | 1.75 | 3.15 | 75.1 | 505.4 | +0.079 |
| R13_hemato_trueprior_seed456 | prior | 2.49 | 4.97 | 62.0 | 321.5 | +0.144 |
| R165_erythroid_myeloid_seed42 | prior | 2.29 | 4.38 | 64.7 | 362.4 | +0.138 |
| R84_erythroid_seed123 | prior | 3.80 | 6.62 | 45.7 | 208.6 | +0.287 |
| R84_erythroid_seed42 | prior | 2.30 | 4.43 | 64.5 | 275.9 | -0.033 |
| R14_hemato_vanilla | vanilla | 3.10 | 5.72 | 53.4 | 239.8 | +0.174 |
| R14_hemato_seed123 | vehicle | 1.86 | 3.58 | 72.7 | 368.5 | -0.059 |
| R14_hemato_seed42 | vehicle | 5.19 | 8.12 | 31.9 | 178.6 | +0.499 |
| R14_hemato_seed456 | vehicle | 1.85 | 3.38 | 72.8 | 457.1 | +0.072 |
| R14_wrongprior_seed42 | wrongprior | 2.30 | 4.37 | 64.6 | 361.4 | +0.137 |

- vanilla PR = 3.10 (n=1 run(s))
- prior/ablation runs below vanilla: 16/18
- exact sign test p = 0.0013

### neural

| run | role | PR | eff_rank (legacy) | top1 % | tr | resid |
|---|---|---|---|---|---|---|
| R2_curriculum_only_20260531_180852 | ablation | 2.37 | 4.37 | 62.5 | 629.5 | -0.341 |
| R3_mixed_dist_only_20260601_025237 | ablation | 3.47 | 5.96 | 46.6 | 635.5 | +0.038 |
| R4_tf_prior_only_20260601_113012 | ablation | 4.00 | 6.93 | 43.1 | 425.1 | +0.258 |
| R5_curriculum_mixed_20260601_202222 | ablation | 3.88 | 6.53 | 42.1 | 498.2 | +0.197 |
| R6_curriculum_tfprior_20260602_042146 | ablation | 3.45 | 5.79 | 43.8 | 487.4 | +0.083 |
| R7_mixed_tfprior_20260602_110953 | ablation | 2.46 | 4.59 | 61.7 | 686.9 | -0.323 |
| R1_neural_seed123 | prior | 3.25 | 5.49 | 47.9 | 509.9 | +0.016 |
| R1_neural_seed456 | prior | 3.77 | 6.83 | 46.0 | 443.8 | +0.192 |
| R8_string_20260613_002335 | prior | 4.72 | 7.63 | 33.3 | 553.5 | +0.373 |
| R0_vanilla_seed123 | vanilla | 4.10 | 7.14 | 42.6 | 325.5 | +0.337 |
| R0_vanilla_seed456 | vanilla | 3.05 | 5.26 | 51.6 | 590.4 | -0.077 |

- vanilla PR = 4.10 (n=2 run(s))
  ** vanilla spread = 3.05-4.10 (seed-to-seed range 1.06); comparisons below are against the FIRST vanilla only and are therefore optimistic.
- prior/ablation runs below vanilla: 8/9
- exact sign test p = 0.0391
- seed-matched pairs (vanilla -> prior):
  - seed 123: 4.10 -> 3.25  (-0.85, 0.79x)
  - seed 456: 3.05 -> 3.77  (+0.73, 1.24x)

### lung

| run | role | PR | eff_rank (legacy) | top1 % | tr | resid |
|---|---|---|---|---|---|---|
| r9_backup | prior | 5.84 | 12.35 | 38.4 | 77.6 | +0.046 |
| R10 | vanilla | 9.01 | 13.80 | 25.3 | 51.5 | +0.027 |

- vanilla PR = 9.01 (n=1 run(s))
- prior/ablation runs below vanilla: 1/1
- exact sign test p = 1.0000

---
## 2. Self-normalised change: each run is its own control

$\Delta\log(\mathrm{PR})$ from iter_0 to iter_4 removes run-level baseline differences.

### hemo

| run | role | it0 | it4 | Δlog(PR) | rho(iter) |
|---|---|---|---|---|---|
| R107_erythroid_lymphoid_seed42 | prior | 4.85 | 3.28 | -0.391 | -0.80 |
| R111_erythroid_temporal_seed123-1 | prior | 3.04 | 1.38 | -0.790 | -0.90 |
| R111_erythroid_temporal_seed42 | prior | 3.75 | 2.08 | -0.591 | -0.90 |
| R111_erythroid_temporal_seed456 | prior | 4.33 | 2.09 | -0.726 | -0.60 |
| R11_KLF1_TAL1_seed123 | prior | 4.20 | 1.57 | -0.986 | -0.90 |
| R11_KLF1_TAL1_seed42 | prior | 3.11 | 2.39 | -0.263 | -0.50 |
| R11_KLF1_TAL1_seed456 | prior | 6.09 | 1.73 | -1.257 | -1.00 |
| R12_GATA2_GFI1B_PAX5_seed123 | prior | 4.23 | 1.64 | -0.950 | -0.70 |
| R12_GATA2_GFI1B_PAX5_seed42 | prior | 2.72 | 2.74 | +0.008 | +0.30 |
| R12_GATA2_GFI1B_PAX5_seed456 | prior | 3.57 | 1.54 | -0.841 | -0.70 |
| R13_KO_KLF1_seed42 | prior | 4.80 | 2.80 | -0.538 | -1.00 |
| R13_OE_KLF1_seed42 | prior | 4.44 | 2.27 | -0.671 | -0.90 |
| R13_hemato_trueprior_seed123 | prior | 5.31 | 1.48 | -1.278 | -1.00 |
| R13_hemato_trueprior_seed42 | prior | 6.14 | 1.75 | -1.257 | -1.00 |
| R13_hemato_trueprior_seed456 | prior | 5.97 | 2.49 | -0.875 | -0.90 |
| R165_erythroid_myeloid_seed42 | prior | 6.51 | 2.29 | -1.043 | -1.00 |
| R84_erythroid_seed123 | prior | 6.13 | 3.80 | -0.480 | -1.00 |
| R84_erythroid_seed42 | prior | 4.42 | 2.30 | -0.651 | -1.00 |
| R14_hemato_vanilla | vanilla | 3.41 | 3.10 | -0.095 | -0.10 |
| R14_hemato_seed123 | vehicle | 3.36 | 1.86 | -0.589 | -0.90 |
| R14_hemato_seed42 | vehicle | 4.74 | 5.19 | +0.091 | +0.30 |
| R14_hemato_seed456 | vehicle | 5.66 | 1.85 | -1.119 | -0.90 |
| R14_wrongprior_seed42 | wrongprior | 4.46 | 2.30 | -0.665 | -0.90 |

- vanilla Δlog(PR): -0.095 (n=1)
- prior/ablation Δlog(PR): median -0.758, 17/18 negative, sign test p = 0.000145
- vehicle/wrong-prior Δlog(PR): -0.589, +0.091, -1.119, -0.665
- Mann-Whitney prior vs vanilla: U=1.0, p=0.2105 (n=18 vs 1)

### neural

| run | role | it0 | it4 | Δlog(PR) | rho(iter) |
|---|---|---|---|---|---|
| R2_curriculum_only_20260531_180852 | ablation | 2.36 | 2.37 | +0.003 | -0.10 |
| R3_mixed_dist_only_20260601_025237 | ablation | 2.80 | 3.47 | +0.215 | +0.50 |
| R4_tf_prior_only_20260601_113012 | ablation | 2.18 | 4.00 | +0.604 | +1.00 |
| R5_curriculum_mixed_20260601_202222 | ablation | 2.43 | 3.88 | +0.470 | +0.90 |
| R6_curriculum_tfprior_20260602_042146 | ablation | 2.32 | 3.45 | +0.397 | +0.90 |
| R7_mixed_tfprior_20260602_110953 | ablation | 2.25 | 2.46 | +0.089 | +0.00 |
| R1_neural_seed123 | prior | 2.31 | 3.25 | +0.341 | +0.90 |
| R1_neural_seed456 | prior | 2.08 | 3.77 | +0.597 | +1.00 |
| R8_string_20260613_002335 | prior | 2.49 | 4.72 | +0.640 | +0.70 |
| R0_vanilla_seed123 | vanilla | 2.16 | 4.10 | +0.643 | +0.90 |
| R0_vanilla_seed456 | vanilla | 2.11 | 3.05 | +0.366 | +0.70 |

- vanilla Δlog(PR): +0.643, +0.366 (n=2)
- prior/ablation Δlog(PR): median +0.397, 0/9 negative, sign test p = 0.003906
- Mann-Whitney prior vs vanilla: U=5.0, p=0.4364 (n=9 vs 2)

### lung

| run | role | it0 | it4 | Δlog(PR) | rho(iter) |
|---|---|---|---|---|---|
| r9_backup | prior | 7.67 | 5.84 | -0.272 | -0.80 |
| R10 | vanilla | 7.81 | 9.01 | +0.143 | +0.10 |

- vanilla Δlog(PR): +0.143 (n=1)
- prior/ablation Δlog(PR): median -0.272, 1/1 negative, sign test p = 1
- Mann-Whitney prior vs vanilla: U=0.0, p=1.0000 (n=1 vs 1)

---
## 3. Prior effect after controlling for the generic tr expansion

### hemo: residual log PR at iter_4 (lower = compressed)

| group | n | mean resid | sd |
|---|---|---|---|
| vanilla | 1 | +0.174 | nan |
| prior/ablation | 18 | +0.087 | 0.122 |
| vehicle/wrongprior | 4 | +0.162 | 0.239 |

- vanilla n=1 (resid +0.174); 12/18 prior runs below, sign test p = 0.2379

### neural: residual log PR at iter_4 (lower = compressed)

| group | n | mean resid | sd |
|---|---|---|---|
| vanilla | 2 | +0.130 | 0.293 |
| prior/ablation | 9 | +0.055 | 0.246 |

- Mann-Whitney (prior vs vanilla resid): p = 0.9091

### lung: residual log PR at iter_4 (lower = compressed)

| group | n | mean resid | sd |
|---|---|---|---|
| vanilla | 1 | +0.027 | nan |
| prior/ablation | 1 | +0.046 | nan |

- vanilla n=1 (resid +0.027); 0/1 prior runs below, sign test p = 1.0000

---
## 3b. Joint signature: amplitude up, dimensionality down

`tr` and the participation ratio are not independent: across all runs log(PR) ~ log(tr) with r = -0.75. Controlling for tr therefore removes part of the effect by construction (tr is a co-outcome, not a confounder). The honest summary is the *joint* signature: prior runs end with BOTH larger total latent variance AND smaller participation ratio. Their ratio, tr/PR, is the variance packed per effectively used dimension.

### hemo

- vanilla: tr=239.8, PR=3.10, ratio=77.4 (n=1)
- tr larger than vanilla:        15/18, p = 0.0075
- PR smaller than vanilla:       16/18, p = 0.0013
- ratio larger than vanilla:     15/18, p = 0.0075
- vehicle/wrong-prior ratio: 197.7, 34.4, 247.2, 157.4

### neural

- vanilla: tr=325.5, PR=4.10, ratio=79.3 (n=2)
  ** vanilla tr spread: 325.55-590.37 — counts below use the FIRST vanilla only (optimistic when the vanilla spread is wide)
  ** vanilla PR spread: 3.05-4.10 — counts below use the FIRST vanilla only (optimistic when the vanilla spread is wide)
  ** vanilla ratio spread: 79.35-193.70 — counts below use the FIRST vanilla only (optimistic when the vanilla spread is wide)
- tr larger than vanilla:        9/9, p = 0.0039
- PR smaller than vanilla:       8/9, p = 0.0391
- ratio larger than vanilla:     9/9, p = 0.0039
- STRICT (ratio above the HIGHEST vanilla, 193.7): 2/9, p = 0.1797

### lung

- vanilla: tr=51.5, PR=9.01, ratio=5.7 (n=1)
- tr larger than vanilla:        1/1, p = 1.0000
- PR smaller than vanilla:       1/1, p = 1.0000
- ratio larger than vanilla:     1/1, p = 1.0000

---
## 3c. Robustness: within-stage covariance

Pooling all developmental stages folds the between-stage displacement into the covariance. Repeating on stage-centred data isolates within-stage geometry.

- hemo: vanilla w_eff_rank=6.67; 17/18 prior runs below, p = 0.0001
- neural: vanilla w_eff_rank=8.08; 7/9 prior runs below, p = 0.1797
- lung: vanilla w_eff_rank=13.73; 1/1 prior runs below, p = 1.0000

---
## 4. Neural 2^3 factorial: which component drives the spectrum?

### pr
| term | beta | se | t | p |
|---|---|---|---|---|
| intercept | +3.411 | 0.416 | +8.20 | 0.000 |
| C (curriculum) | -0.073 | 0.486 | -0.15 | 0.886 |
| M (mixture) | -0.017 | 0.486 | -0.04 | 0.973 |
| T (decoder audit) | +0.028 | 0.486 | +0.06 | 0.957 |

### top1
| term | beta | se | t | p |
|---|---|---|---|---|
| intercept | +49.186 | 5.199 | +9.46 | 0.000 |
| C (curriculum) | -0.600 | 6.071 | -0.10 | 0.924 |
| M (mixture) | +0.357 | 6.071 | +0.06 | 0.955 |
| T (decoder audit) | -0.563 | 6.071 | -0.09 | 0.929 |

### resid
| term | beta | se | t | p |
|---|---|---|---|---|
| intercept | +0.049 | 0.157 | +0.31 | 0.766 |
| C (curriculum) | -0.016 | 0.184 | -0.09 | 0.933 |
| M (mixture) | -0.029 | 0.184 | -0.16 | 0.879 |
| T (decoder audit) | +0.023 | 0.184 | +0.13 | 0.903 |

### tr
| term | beta | se | t | p |
|---|---|---|---|---|
| intercept | +516.371 | 71.216 | +7.25 | 0.000 |
| C (curriculum) | -27.047 | 83.163 | -0.33 | 0.756 |
| M (mixture) | +75.706 | 83.163 | +0.91 | 0.398 |
| T (decoder audit) | -34.952 | 83.163 | -0.42 | 0.689 |

---
## 5. Hematopoietic specificity ladder (all seed 42, 29,985 cells)

| run | condition | PR | eff_rank (legacy) | top1 % | tr | Δlog(PR) it0→it4 |
|---|---|---|---|---|---|---|
| R14_hemato_vanilla | vanilla (no prior machinery) | 3.10 | 5.72 | 53.4 | 239.8 | -0.095 |
| R14_hemato_seed42 | vehicle (prior boost 5.2% / 1054 genes) | 5.19 | 8.12 | 31.9 | 178.6 | +0.091 |
| R14_wrongprior_seed42 | wrong prior (52 neural TFs, boost 0.4% / 88 genes) | 2.30 | 4.37 | 64.6 | 361.4 | -0.665 |
| R13_hemato_trueprior_seed42 | true prior (11 TFs / 325 TRRUST+CORE edges, boost 1.0%) | 1.75 | 3.15 | 75.1 | 505.4 | -1.257 |
