# Running the injection-path check on your own pipeline

A worked guide. Ten minutes, no training, read-only.

**What this answers:** if your model injects a per-gene weight vector — a
gene-regulatory prior, a gene-importance mask, a likelihood weighting — did that vector
actually reach the objective? If it did not, any change you measured in the decoder
came from something else, and no amount of downstream analysis will make it
attributable.

This check is cheap, it runs on saved artefacts, and it either passes or it does not.
It is worth running *before* you build an interpretation on top of an observed change.

---

## 1. The idea in one picture

```
   prior source  →  subsetting  →  per-gene weights  →  the loss term
                                                            │
                          D1  is the weight vector non-zero in what training saved?
                          D2  where is it assigned, and where is it read?
                          D3  is the branch that would consume it ever instantiated?
```

A weight vector can be **built, stored, and never read**. That is not a hypothetical
failure mode: it is the one this check was written to catch. Storing a vector and
consuming it are separate acts, and only the second one can change a decoder.

**D1 and D3 are decisive. D2 is locational only** — a read inside a class definition
does not prove the stored value is consumed. The tool labels this explicitly so it
cannot be over-read.

---

## 2. Run it

```bash
pip install kappa-c-audit h5py

python examples/check_injection_path.py \
    --artifacts /path/to/saved_training_outputs \
    --source    /path/to/your/training/code
```

Two paths in, one verdict out. Four things are configurable, and you will likely need
at least the last two:

| Flag | Default | When you change it |
|---|---|---|
| `--artifacts` | required | root of what training saved |
| `--source` | required | root of your training source tree |
| `--roots` | auto | sub-directories to scan; **each is reported separately** |
| `--pattern` | `**/*.h5ad` | if your artefacts are not `.h5ad` |
| `--layer-key` | `layers/geneWeight/data` | the HDF5 path of your weight vector |
| `--weight-name` | `geneWeight` | the Python name it goes by in your source |

Add `--json report.json` to keep the full structured result.

**Keep `--roots` honest.** If you pass two roots they are counted and reported
separately, because a subset count and a full-tree count are different denominators.
Conflating them is the easiest way to overstate your coverage — the paper this check
comes from quotes both 40 and 796 for the same pool of artefacts, from two different
search roots, and had to say so explicitly.

---

## 3. Read the output

Real output, from a synthetic pipeline that **does** deliver its prior:

```
D1  per-gene weight layer occupancy  (DECISIVE, empirical)
      benchmark                    1 matched  with-layer=1     empty=0     populated=1
      result                       1 matched  with-layer=1     empty=1     populated=0
      totals: with-layer=2 empty=1 populated=1
      VERDICT: POPULATED: at least one artefact carries non-zero weights

D2  assignment / reader sites       (locational only -- NOT evidence)
      assignment=2  reader=2

D3  consuming-branch instantiations (DECISIVE, structural)
      instantiations: 1
      VERDICT: instantiated 1 time(s)
        model.py:6  return GVAELoss(geneWeight=self.geneWeight)

VERDICT: not blocked
```

And from one that does **not**:

```
D1  VERDICT: ALL-ZERO: layer present but empty in every file scanned
D3  VERDICT: DEAD: no instantiation site
VERDICT: BLOCKED
```

### The three outcomes

| Verdict | Means | What to do |
|---|---|---|
| **BLOCKED** | The layer is empty in every artefact scanned, and/or the consuming branch has no instantiation site. | Do not attribute any decoder change to the prior. Find where the signal dies: `populated=0` with `with-layer>0` means the vector was allocated and never filled; `DEAD` means nothing ever constructs the consuming object. |
| **not blocked** | At least one artefact carries non-zero weights. | The prior arrived. This says **nothing** about whether it is effective — only that downstream claims are now attributable. Move on to a compression check. |
| **undetermined** | Nothing matched, or no layer was found at all. | Your `--pattern`, `--roots` or `--layer-key` is wrong. This is a setup error, not a result. |

Exit codes: `0` not blocked, `1` blocked, `2` could not run. That makes it usable
directly in CI:

```bash
python examples/check_injection_path.py --artifacts ... --source ...
# fails the build if a refactor silently breaks delivery
```

---

## 4. Where this can mislead you

Stated plainly, because a check you over-trust is worse than no check.

1. **`populated=1` is not a pass on effectiveness.** A prior can arrive and still do
   nothing useful. This check is about *attribution*, not *efficacy*.
2. **`DEAD` from a regex is weaker evidence than `ALL-ZERO` from the artefacts.** D3
   greps source text; it can be fooled by dynamic dispatch, a different wrapper name,
   or a factory that builds the branch without mentioning the identifier. D1 reads the
   actual bytes. **If D1 and D3 disagree, believe D1.**
3. **A hand-written `--weight-name` that does not appear in your source produces a
   silent `DEAD`.** Before trusting a `DEAD`, grep for your identifier yourself:
   `grep -rn "yourName" /path/to/source | head`. If the tool found 0 assignment sites,
   the name is probably wrong, not the pipeline.
4. **An unreadable artefact is counted as `no_layer`, not as an error.** If
   `files_matched` is much larger than `with-layer + no_layer`, check for a corrupted
   or differently-formatted file.
5. **A property of one implementation is not a property of a method.** A `BLOCKED`
   result here says this pipeline never delivered this prior. It does **not** say GRN
   priors are ineffective in general, and it does not generalise to other codebases.

---

## 5. Reproduce this guide's output

The two synthetic fixtures above are built by
`tests/test_injection_path.py`, which asserts a passing case, a failing case, and the
parameterisation. Run:

```bash
pytest tests/ -q
```

If you want to see the check find a real failure, point it at any pipeline whose stored
weight channel is empty — including, for the record, the one this toolkit ships with:

```bash
python examples/check_injection_path.py \
    --artifacts /path/to/UNAGI --source /path/to/UNAGI_source \
    --roots benchmark result --pattern '**/stagedata/*.h5ad'
# -> VERDICT: BLOCKED
```
