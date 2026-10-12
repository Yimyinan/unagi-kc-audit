# kappa-c-audit

When the prior does not arrive: a dual-spectrum audit of **GRN-prior injection in single-cell VAEs**,
plus the **three-check prior-activity protocol**.

It answers two questions for any trained generative model with a differentiable
decoder:

1. **Does prior injection reach the decoder readout?** — the decoder audit
   `kappa-C(W)`.
2. **Did the latent code reorganise even if the decoder did not?** — the companion
   latent audit `kappa-C(Z)`.

## Install

```bash
pip install kappa_c_audit-0.1.2-py3-none-any.whl   # from the release wheel
# or, from a checkout:
pip install -e .
```

Only `numpy`, `scipy`, `scikit-learn` are required. Verified to install and run
in a clean virtual environment on Python 3.13 (7/7 package tests pass).

## Quick start

```python
from kappa_c_audit import audit

res = audit(Z_vanilla, X_vanilla, Z_prior, X_prior)
# Z_*: latent codes (cells x latent dims); X_*: expression (cells x genes)
print(res["verdict"], res["kappa_C_W"], res["kappa_C_Z"])
# -> 'vanilla-equivalent' 0.99 1.08
```

Verdicts returned by `classify`: `locked` (`kappa < 0.5`, i.e. `C > 50%`),
`sub-vanilla` (`kappa < band lower` but not locked), `vanilla-equivalent`
(`kappa in [0.937, 1.064]`, indistinguishable from seed noise), `amplified`
(`kappa > 1.064`), and `baseline-only` when no prior checkpoint is supplied.

## Checking a model you did not train

`audit` is architecture-agnostic: point it at the latent code and expression
matrix of a vanilla checkpoint and a prior-injected / fine-tuned one, over the
same cells. `examples/checkup_external_model.py` is a ready-made CLI for this:

```bash
python examples/checkup_external_model.py --vanilla RUN_DIR --prior RUN_DIR2
# RUN_DIR must contain Z.npy (cells x latent) and X.npy (cells x genes) --
# extract them however the model exposes them (get_latent_representation(),
# model.encode(), or any projection).

python examples/checkup_external_model.py --self-test
# fabricates the two canonical outcomes and must print
# '(a) vanilla-equivalent' for a latent-only reorganisation and
# '(b) locked' for a genuine decoder compression.
```

It prints a six-line report and is safe to wire into a CI gate: the exit status
reflects whether the audit *ran*, the judgement lives in the verdict.

## What the decoder audit actually measures

`ridge_readout` standardises `Z` and `X` per column/gene. The audited quantity is

```
kappa-C(W) = sum_g R2_g(perturbed) / sum_g R2_g(vanilla)
```

i.e. **kappa-C(W) is an aggregate latent->expression predictability ratio**. It is
**invariant to per-gene rescaling** of the decoder and to **attenuation of
latent->gene directions**; it fires only when a model's latent code predicts its
expression differently well. This is stated explicitly so the decoder null is not
credited with more than it shows: a null decoder audit means *"prior injection does
not change how well the latent predicts expression"*, not *"the decoder is
unaltered"*.

### The trace form is a different quantity

An earlier release returned the **trace form**
`tr(W^T W)_perturbed / tr(W^T W)_vanilla`. For standardised `Z`, `X`,

```
sum_g R2_g = sum_g W[:,g]^T Corr(Z) W[:,g]     # predictability form
tr(W^T W)  = sum_g W[:,g]^T W[:,g]             # trace form
```

so the two agree **if and only if the latent code is whitened**. Trained latent
codes are not: in the published UNAGI audit the condition number of `Corr(Z)` had
median 3.3e3, and the precondition failed in 33/33 configurations. Two consequences
matter in practice:

* the trace form is **not invariant to the latent basis** — rotating `Z` leaves
  every `R2_g`, and therefore the audited quantity, unchanged, yet moves the trace
  ratio (1.000 to 1.247 on a real trained latent);
* it **inflates between-model scatter** — for a family-matched no-prior pool of four
  hematopoietic readouts the trace form gives `1.24 +/- 0.82` where the
  predictability form gives `1.000 +/- 0.015`, a 56-fold difference, with the band
  centre displaced from unity by 24%. At that noise level a real effect is invisible.

`kappa_C_W(..., form="trace")` and `audit()["kappa_C_W_trace"]` reproduce the legacy
estimator; the default and `audit()["kappa_C_W"]` are the predictability form.
`predictability_total()` returns a model's `sum_g R2_g`, and `predictability_ratio()`
is the ratio of two of them (identical to the default by construction). Note that
`sum_g R2_g` **cannot** be recovered from `W` alone — it needs the latent correlation
matrix or the data — which is why `audit_from_readout()` can only offer the trace
form and says so.

## The four checks

```python
from kappa_c_audit import (gradient_check, prior_strength_scan, structural_bound,
                           injection_path_check)

gradient_check(spectral_response, prior_gradient_norm)   # does the response track the gradient?
prior_strength_scan({0.1: k1, 1.0: k2, 10.0: k3})        # is the collapse lambda-invariant?
structural_bound(C_percent, trH_vanilla, gap_vanilla)    # is the compression even feasible?
injection_path_check(artifacts_root, source_root)        # did the prior REACH the loss?
```

A high compression `C` is **not** evidence that the prior was used: the
`structural_bound` and a flat `prior_strength_scan` can both be satisfied by a
data-inherent collapse with an inert prior.

### Check 4: did the prior arrive?

Checks 1–3 all assume the prior reached the objective. If it did not, a change in the
decoder is attributable to something else and no downstream analysis can recover that.
`injection_path_check` tests delivery directly, on saved artefacts, with no retraining:

```
prior source → subsetting → per-gene weights → the loss term
                                                  │
            D1  is the weight vector non-zero in what training saved?   (empirical, decisive)
            D2  where is it assigned / read?                            (locational only)
            D3  is the consuming branch ever instantiated?              (structural, decisive)
```

It returns `BLOCKED` when the layer is empty in every artefact and/or the consuming
branch has no instantiation site. Both paths are covered by tests, so the check is not
only ever seen failing.

Point it at your own pipeline:

```bash
python examples/check_injection_path.py \
    --artifacts /path/to/saved_training_outputs \
    --source    /path/to/your/training/code
# exit 0 = not blocked, 1 = BLOCKED, 2 = could not run (usable directly in CI)
```

`--layer-key` and `--weight-name` name the two things it must look for in your layout.
**Full walkthrough, including where the check can mislead you, in
[`docs/RUN_THE_INJECTION_PATH_CHECK.md`](docs/RUN_THE_INJECTION_PATH_CHECK.md).**

## Reproducibility

All definitions follow the manuscript protocol, and the legacy trace-form estimator
of release 0.1.0 remains available so previously published numbers still reproduce
exactly (`form="trace"`, `audit()["kappa_C_W_trace"]`). Release 0.1.1 changed the
default to the predictability form and corrected `predictability_ratio()`, which in
0.1.0 averaged per-output `score()` values (rather than summing) and used an
intercept on unstandardised data, so it did not match `kappa_C_W` as its docstring
claimed. Release 0.1.2 updates `DEFAULT_BAND` to the corrected predictability-form
vanilla-equivalence band `[0.937, 1.064]` (1.001 +/- 0.032 over 15 G-VAE hematopoietic
seeds / 210 ordered seed-pair comparisons); it had carried the superseded trace-form
band `[0.78, 1.23]`. This changes only the `vanilla-equivalent` vs `amplified` /
`sub-vanilla` boundaries -- the band-independent deep-compression verdict
(`kappa < 0.5`) is unaffected.

## Citation

If you use this tool, please cite the accompanying manuscript (dual-spectrum
diagnostic for auditing GRN-prior injection).

## License

MIT.
