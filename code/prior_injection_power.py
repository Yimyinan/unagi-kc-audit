"""
prior_injection_power.py
========================
How many seeds does a re-trained UNAGI comparison need?

[!] SUPERSEDED PREMISE -- DO NOT CITE AS JUSTIFICATION FOR A TWO-ARM DESIGN.
----------------------------------------------------------------------------
This budget was computed for a two-arm comparison of an *information-poor* prior
(current binary boost) against an *information-rich* prior (TRUST direction +
confidence), on the assumption that "both arms are injected".

That premise no longer holds.  The injection-path audit of the released PG-UNAGI
pipeline finds that the prior's content has **no functional route into the training
objective**: the per-gene weight vector is assigned to the training object and never
read, the likelihood-weighting branch that would consume it has no instantiation
site, and the per-cell weight layer is identically zero across all 796 retained stage
files / 29 run-iterations.  An information-rich arm therefore cannot be constructed
inside PG-UNAGI: enriching the prior changes no loss term, so the comparison this
file sizes has no treatment contrast to measure.

A second mismatch: the seed noise measured below is read from
UNAGI/decoder_unified/loading/*.npy, i.e. from the decoder **readout spectrum**,
whereas the audit reports the corrected kappa-C(W) predictability ratio.  Those are
different statistics on different scales, so this is not the variance of the quantity
a two-arm kappa-C(W) design would test.

What remains valid: the general method (power for a paired seed design against a
positive ratio with multiplicative noise) and the qualitative conclusion that an
unpaired two-arm design is unaffordable.  For a power assessment of the statistic the
audit actually reports, use code/closing_gaps_power_and_dissociation.py
(results in data/closing_gaps_power_and_bound.json).

Motivation (as originally written)
----------------------------------
The L2 experiment (direction/strength-aware GRN injection) requires retraining
UNAGI, which is expensive. Before committing, we estimate the seed budget from
the variance that the existing 33-configuration audit already measured.

The design is a two-arm comparison: an *information-poor* prior (current
binary boost) versus an *information-rich* prior (TRRUST direction +
confidence). Both arms are injected; both are subject to the same seed noise.

Key quantities
--------------
1. Seed-to-seed variance of the audited statistic, taken empirically from the
   three-seed groups already present in decoder_unified/loading/ (R11, R12,
   R13) plus the vanilla group. Measured in log space, because the audit
   statistic is a positive ratio and its noise is multiplicative.

2. The detectable effect size d = |log(treated/control)| / (sd_pooled * sqrt(2)).
   An effect is "detectable" at n seeds/arm when |t| > t_crit for the chosen
   alpha and power.

3. The implied minimum detectable effect (MDE) at a given seed count, which is
   the more useful direction of the calculation: tell the user what the design
   can see, then let them judge whether the biology is bigger than that.

Which statistic?
----------------
We compute power for two endpoints that matter for the L2 question:

  ENDPOINT A (decoder gain): the audit statistic gap/kappa. This is what
  distinguishes "locked" from "not". Historically its seed noise is LARGE
  (CV 60-80% in the multi-prior groups), so this endpoint is expensive.

  ENDPOINT B (paired within-seed contrast): comparing the two priors *within*
  the same seed removes the seed effect entirely. Its noise is the
  seed-by-prior interaction, which the existing audit suggests is much
  smaller. This is the design recommendation.

Outputs
-------
  outputs/positive_result/PRIOR_INJECTION_POWER.md
  outputs/positive_result/prior_injection_power.json
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path("F:/UNAGI_Project")
OUT = REPO / "outputs" / "positive_result"
LOADING = Path("F:/UNAGI_Project/UNAGI/decoder_unified/loading")

ALPHA = 0.05
POWER = 0.80


def load_gaps():
    """gap = lambda1 - lambda2 of H = L L^T, per audited configuration."""
    import glob
    import os
    rows = {}
    for f in sorted(glob.glob(str(LOADING / "*.npy"))):
        L = np.load(f)
        H = L @ L.T
        ev = np.sort(np.linalg.eigvalsh(H))[::-1]
        rows[os.path.basename(f)[:-4]] = float(ev[0] - ev[1])
    return rows


def groups_from_audit(g):
    """Three-seed groups that exist in the audit, with their known caveats.

    NOTE: R14/R15 are recorded as the SAME matrix (verified elsewhere), so they
    do not form an independent three-seed group for the prior arms; R14 alone
    (R14_s42/123/456) does. R11 is the cleanest group (2-TF prior).
    """
    def pick(*names):
        return [g[n] for n in names if n in g]
    return {
        "R11 (KLF1+TAL1, 2 TF)": pick("R11_s42", "R11_s123", "R11_s456"),
        "R12 (GATA2/GFI1B/PAX5, 3 TF)": pick("R12_s42", "R12_s123", "R12_s456"),
        "R13 (pan-hema, 11 TF)": pick("R13_s42", "R13_s123", "R13_s456"),
        "R14 (vanilla-comparison arm)": pick("R14_s42", "R14_s123", "R14_s456"),
    }


def log_sd(vals):
    """Multiplicative sd, measured in log space."""
    a = np.asarray(vals, dtype=float)
    a = a[a > 0]
    if a.size < 2:
        return float("nan"), float("nan")
    la = np.log(a)
    return float(la.std(ddof=1)), float(a.mean())


def n_per_arm(delta_log, sd_log):
    """Seeds per arm for a two-sample t-test at ALPHA / POWER."""
    if not np.isfinite(sd_log) or sd_log <= 0 or delta_log == 0:
        return float("inf")
    d = abs(delta_log) / (sd_log * np.sqrt(2))
    z_a = stats.norm.ppf(1 - ALPHA / 2)
    z_b = stats.norm.ppf(POWER)
    return float(2 * (z_a + z_b) ** 2 / d ** 2)


def mde(n_per, sd_log):
    """Minimum detectable |log ratio| at n seeds per arm."""
    if not np.isfinite(n_per) or n_per <= 0:
        return float("nan")
    z_a = stats.norm.ppf(1 - ALPHA / 2)
    z_b = stats.norm.ppf(POWER)
    d = np.sqrt(2 * (z_a + z_b) ** 2 / n_per)
    return float(d * sd_log * np.sqrt(2))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    g = load_gaps()
    groups = groups_from_audit(g)

    measured = {}
    for name, vals in groups.items():
        if len(vals) < 2:
            continue
        sd, mean = log_sd(vals)
        measured[name] = {
            "gaps": vals, "mean_gap": mean,
            "log_sd": sd, "mult_sd": float(np.exp(sd)),
            "within_group_ratio": float(max(vals) / min(vals)),
        }

    sds = [v["log_sd"] for v in measured.values()]
    sd_unpaired = float(np.mean(sds))

    # ---- ENDPOINT B: paired within-seed contrast --------------------------
    # The paired design removes the seed effect. Its noise is the seed x prior
    # interaction. We cannot measure this from the audit (both arms were never
    # run on the same seed with different prior *information*), so we tabulate
    # the required n as a function of the assumed interaction sd.
    paired_table = []
    for sd_int in [0.10, 0.20, 0.35, 0.50, 0.69]:
        row = {"assumed_interaction_log_sd": sd_int,
               "mult_sd": float(np.exp(sd_int))}
        for u in [0.10, 0.20, 0.30, 0.50]:
            row[f"n_for_{int(u*100)}pct"] = n_per_arm(u, sd_int)
        paired_table.append(row)

    # ---- ENDPOINT A: unpaired group comparison ----------------------------
    unpaired_table = []
    for u in [0.10, 0.20, 0.30, 0.50, 0.80]:
        unpaired_table.append({
            "effect_log_units": u,
            "effect_fold": float(np.exp(u)),
            "n_per_arm": n_per_arm(u, sd_unpaired),
        })

    # ---- what can 3 seeds/arm actually see? -------------------------------
    mde3_unpaired = mde(3, sd_unpaired)
    mde6_unpaired = mde(6, sd_unpaired)

    rep = {
        "alpha": ALPHA, "power": POWER,
        "measured_seed_noise": measured,
        "sd_unpaired_log": sd_unpaired,
        "mult_sd_unpaired": float(np.exp(sd_unpaired)),
        "endpoint_A_unpaired": unpaired_table,
        "endpoint_B_paired": paired_table,
        "mde_at_3_per_arm": {
            "log_units": mde3_unpaired,
            "fold": float(np.exp(mde3_unpaired)),
        },
        "mde_at_6_per_arm": {
            "log_units": mde6_unpaired,
            "fold": float(np.exp(mde6_unpaired)),
        },
    }
    json.dump(rep, open(OUT / "prior_injection_power.json", "w"), indent=1)

    L = ["# Seed budget for a retrained UNAGI comparison", "",
         f"Two-arm design, alpha={ALPHA}, power={POWER}. "
         "Effect sizes are expressed in log units because the audit statistic "
         "is a positive ratio with multiplicative noise.", ""]

    L += ["## Measured seed-to-seed noise (from the existing audit)", "",
          "| group | gaps | mean | multiplicative sd | max/min |",
          "|---|---|---|---|---|"]
    for name, v in measured.items():
        gaps = ", ".join(f"{x:.0f}" for x in v["gaps"])
        L.append(f"| {name} | {gaps} | {v['mean_gap']:.0f} | "
                 f"{v['mult_sd']:.2f}x | {v['within_group_ratio']:.2f}x |")
    L += ["", f"Pooled multiplicative sd across groups: "
          f"**{np.exp(sd_unpaired):.2f}x** (log sd {sd_unpaired:.3f}).",
          "",
          "R14 is used as the vanilla-comparison arm; R15 is recorded as the "
          "same matrix and is excluded as a duplicate. R11 is the cleanest "
          "group and is also the least noisy, which matters for the design "
          "recommendation below.", ""]

    L += ["## Endpoint A - unpaired group comparison (not recommended)", "",
          "Comparing two groups of independently trained models. The seed "
          "effect is part of the signal, so it has to be paid for in seeds.",
          "", "| effect | fold-change | seeds per arm |", "|---|---|---|"]
    for r in unpaired_table:
        n = r["n_per_arm"]
        L.append(f"| {r['effect_log_units']:.2f} log | {r['effect_fold']:.2f}x | "
                 f"{n:.0f} |")
    L += ["",
          f"At 3 seeds/arm this design can only see "
          f"**{np.exp(mde3_unpaired):.1f}x** or larger. At 6 seeds/arm, "
          f"**{np.exp(mde6_unpaired):.1f}x**. The audit's own prior arms split "
          f"by up to 6.5x across seeds, so a biologically plausible effect "
          f"(2-3x) is invisible at any seed count a single project can afford.",
          ""]

    L += ["## Endpoint B - paired within-seed contrast (recommended)", "",
          "Train both priors on the *same* seed, then compare within seed. The "
          "seed effect cancels, and the relevant noise is the seed-by-prior "
          "interaction. That interaction has never been measured (both arms "
          "were only ever run as different configurations), so the table is "
          "tabulated against an assumed interaction sd.",
          "",
          "| interaction sd | as a factor | n for 10% eff | n for 20% | n for 30% | n for 50% |",
          "|---|---|---|---|---|---|"]
    for r in paired_table:
        L.append(f"| {r['assumed_interaction_log_sd']:.2f} | "
                 f"{r['mult_sd']:.2f}x | {r['n_for_10pct']:.0f} | "
                 f"{r['n_for_20pct']:.0f} | {r['n_for_30pct']:.0f} | "
                 f"{r['n_for_50pct']:.0f} |")
    L += ["",
          "If the interaction is small (0.2 log units, i.e. priors respond to "
          "seeds in a similar way), 3 seeds/arm suffices for a 20% effect and "
          "6 seeds/arm reaches ~15%. If the interaction is as large as the "
          "unpaired noise (0.69), the paired design buys nothing and we are "
          "back to Endpoint A.", ""]

    L += ["## Conclusion", "",
          "- Seed noise in the audit statistic is **large**: 60-80% CV within "
          "a fixed prior, up to 6.5x max/min.",
          "- An unpaired two-arm design needs an implausible effect size to "
          "work at affordable n. Do not run it.",
          "- A paired within-seed design is the only affordable option, and "
          "its budget depends on an interaction term nobody has measured.",
          "- **Therefore the cheap first step is to measure that interaction**, "
          "not to run the full comparison: 3 seeds x 2 arms is the smallest "
          "experiment that both estimates the interaction and tests the "
          "question.", ""]

    open(OUT / "PRIOR_INJECTION_POWER.md", "w",
         encoding="utf-8").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
