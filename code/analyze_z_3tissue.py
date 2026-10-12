# -*- coding: utf-8 -*-
"""Statistics for the three-tissue latent-spectrum scan.

Everything here is pre-specified by tissue: replicate only where the design
actually has replication, and report n for every test. No pooling across
tissues unless explicitly labelled as such.

Metric convention (2026-09): the reported effective-rank measure is the
*participation ratio* ``pr`` (VAE / single-cell convention, shared by the
PG-UNAGI latent audit, the kappa-C-Z diagnostic, and the scVI validation).
The legacy spectral-perplexity column ``eff_rank`` is retained in
``z_spectrum_all.csv`` for backward compatibility but is NOT used in any
reported statistic. All headline numbers below are therefore computed in
``pr``; ``eff_rank`` is shown only as a secondary legacy column.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# Primary reported metric (participation ratio) and legacy column.
PR = "pr"
LEG = "eff_rank"

OUT = Path(r"F:/UNAGI_Project/outputs/positive_result")
df = pd.read_csv(OUT / "z_spectrum_all.csv")

L: list[str] = []
def w(s: str = "") -> None:
    L.append(s)
    print(s)


def sign_test(k: int, n: int) -> float:
    """Two-sided exact binomial p for k/n successes under H0 p=0.5."""
    return stats.binomtest(k, n, 0.5).pvalue


# ------------------------------------------------------------------ confound
w("# Confound check: is the participation ratio just a function of tr(Cov Z)?")
w()
w("`tr` grows monotonically with IDREM iteration in essentially every run, "
  "including vanilla. If the participation ratio is determined by tr alone, "
  "then any 'compression' is a generic training effect, not a prior effect.")
w()
allr = df.dropna(subset=[PR, "tr"])
sl, ic, rv, pv, se = stats.linregress(np.log(allr.tr), np.log(allr[PR]))
w(f"- pooled all runs (n={len(allr)}): log(PR) ~ log(tr), "
  f"slope={sl:+.3f}, r={rv:+.3f}, p={pv:.2e}")
w(f"  -> tr alone explains {rv**2*100:.1f}% of the variance in log PR.")
w()
df["resid"] = np.nan
for tis, g in df.groupby("tissue"):
    sl_t, ic_t, rv_t, pv_t, _ = stats.linregress(np.log(g.tr), np.log(g[PR]))
    df.loc[g.index, "resid"] = np.log(g[PR]) - (sl_t * np.log(g.tr) + ic_t)
    w(f"- {tis:6s} (n={len(g)}): slope={sl_t:+.3f}, r={rv_t:+.3f}, p={pv_t:.2e}")
w()
w("`resid` = log PR after removing the within-tissue tr trend "
  "(negative = more compressed than expected for its total variance).")
w()

# ------------------------------------------------------- 1. terminal per tissue
w("---")
w("## 1. Terminal state (iter_4): does the prior compress?")
w()
for tis in ["hemo", "neural", "lung"]:
    sub = df[(df.tissue == tis) & (df.iteration == 4)]
    van = sub[sub.role == "vanilla"]
    pri = sub[sub.role.isin(["prior", "ablation"])]
    if len(van) == 0:
        w(f"### {tis}: no vanilla run -> untestable")
        continue
    w(f"### {tis}")
    w()
    w("| run | role | PR | eff_rank (legacy) | top1 % | tr | resid |")
    w("|---|---|---|---|---|---|---|")
    for r in sub.sort_values(["role", "run"]).itertuples():
        w(f"| {r.run} | {r.role} | {r.pr:.2f} | {r.eff_rank:.2f} | "
          f"{r.top1:.1f} | {r.tr:.1f} | {r.resid:+.3f} |")
    w()
    v = van[PR].iloc[0]
    n_below = int((pri[PR] < v).sum())
    w(f"- vanilla PR = {v:.2f} (n={len(van)} run(s))")
    if len(van) > 1:
        w(f"  ** vanilla spread = {van[PR].min():.2f}-{van[PR].max():.2f} "
          f"(seed-to-seed range {van[PR].max()-van[PR].min():.2f}); "
          f"comparisons below are against the FIRST vanilla only and are "
          f"therefore optimistic.")
    w(f"- prior/ablation runs below vanilla: {n_below}/{len(pri)}")
    if len(pri):
        w(f"- exact sign test p = {sign_test(n_below, len(pri)):.4f}")
    # seed-matched pairs where available
    pairs = []
    for s in pri.seed.dropna().unique():
        vp = van[van.seed == s]
        pp = pri[pri.seed == s]
        if len(vp) and len(pp):
            pairs.append((s, vp[PR].iloc[0], pp[PR].iloc[0]))
    if pairs:
        w("- seed-matched pairs (vanilla -> prior):")
        for s, a, b in pairs:
            w(f"  - seed {int(s)}: {a:.2f} -> {b:.2f}  ({b-a:+.2f}, {b/a:.2f}x)")
    w()

# ------------------------------------------- 2. self-normalised dose-response
w("---")
w("## 2. Self-normalised change: each run is its own control")
w()
w(r"$\Delta\log(\mathrm{PR})$ from iter_0 to iter_4 removes run-level baseline differences.")
w()
rows = []
for run, g in df.groupby("run"):
    g = g.sort_values("iteration")
    if len(g) < 5 or g.iteration.min() != 0:
        continue
    a, b = g[PR].iloc[0], g[PR].iloc[-1]
    rho = stats.spearmanr(g.iteration, g[PR]).correlation
    rows.append(dict(run=run, tissue=g.tissue.iloc[0], role=g.role.iloc[0],
                     seed=g.seed.iloc[0], it0=a, it4=b,
                     dlog=float(np.log(b / a)), rho=rho))
ch = pd.DataFrame(rows)
for tis in ["hemo", "neural", "lung"]:
    s = ch[ch.tissue == tis]
    if s.empty:
        continue
    w(f"### {tis}")
    w()
    w("| run | role | it0 | it4 | Δlog(PR) | rho(iter) |")
    w("|---|---|---|---|---|---|")
    for r in s.sort_values(["role", "run"]).itertuples():
        w(f"| {r.run} | {r.role} | {r.it0:.2f} | {r.it4:.2f} | "
          f"{r.dlog:+.3f} | {r.rho:+.2f} |")
    van = s[s.role == "vanilla"]
    pri = s[s.role.isin(["prior", "ablation"])]
    veh = s[s.role.isin(["vehicle", "wrongprior"])]
    w()
    if len(van):
        w(f"- vanilla Δlog(PR): "
          f"{', '.join(f'{v:+.3f}' for v in van.dlog)} (n={len(van)})")
    if len(pri):
        neg = int((pri.dlog < 0).sum())
        w(f"- prior/ablation Δlog(PR): median {pri.dlog.median():+.3f}, "
          f"{neg}/{len(pri)} negative, sign test p = {sign_test(neg, len(pri)):.4g}")
    if len(veh):
        w(f"- vehicle/wrong-prior Δlog(PR): "
          f"{', '.join(f'{v:+.3f}' for v in veh.dlog)}")
    if len(van) and len(pri):
        u = stats.mannwhitneyu(pri.dlog, van.dlog, alternative="two-sided")
        w(f"- Mann-Whitney prior vs vanilla: U={u.statistic:.1f}, p={u.pvalue:.4f} "
          f"(n={len(pri)} vs {len(van)})")
    w()

# ------------------------------------------------------ 3. tr-controlled test
w("---")
w("## 3. Prior effect after controlling for the generic tr expansion")
w()
for tis in ["hemo", "neural", "lung"]:
    sub = df[(df.tissue == tis) & (df.iteration == 4)]
    van = sub[sub.role == "vanilla"]
    pri = sub[sub.role.isin(["prior", "ablation"])]
    veh = sub[sub.role.isin(["vehicle", "wrongprior"])]
    if len(van) == 0 or len(pri) == 0:
        continue
    w(f"### {tis}: residual log PR at iter_4 (lower = compressed)")
    w()
    w("| group | n | mean resid | sd |")
    w("|---|---|---|---|")
    for nm, g in (("vanilla", van), ("prior/ablation", pri),
                  ("vehicle/wrongprior", veh)):
        if len(g) == 0:
            continue
        w(f"| {nm} | {len(g)} | {g.resid.mean():+.3f} | {g.resid.std(ddof=1):.3f} |")
    if len(van) > 1:
        u = stats.mannwhitneyu(pri.resid, van.resid, alternative="two-sided")
        w(f"\n- Mann-Whitney (prior vs vanilla resid): p = {u.pvalue:.4f}")
    else:
        w(f"\n- vanilla n=1 (resid {van.resid.iloc[0]:+.3f}); "
          f"{(pri.resid < van.resid.iloc[0]).sum()}/{len(pri)} prior runs below, "
          f"sign test p = "
          f"{sign_test(int((pri.resid < van.resid.iloc[0]).sum()), len(pri)):.4f}")
    w()

# ------------------------------------------------------ 3b. joint signature
w("---")
w("## 3b. Joint signature: amplitude up, dimensionality down")
w()
w("`tr` and the participation ratio are not independent: across all runs "
  "log(PR) ~ log(tr) with r = -0.75. Controlling for tr therefore removes "
  "part of the effect by construction (tr is a co-outcome, not a confounder). "
  "The honest summary is the *joint* signature: prior runs end with BOTH "
  "larger total latent variance AND smaller participation ratio. "
  "Their ratio, tr/PR, is the variance packed per effectively used "
  "dimension.")
w()
for tis in ["hemo", "neural", "lung"]:
    sub = df[(df.tissue == tis) & (df.iteration == 4)].copy()
    van = sub[sub.role == "vanilla"]
    pri = sub[sub.role.isin(["prior", "ablation"])]
    veh = sub[sub.role.isin(["vehicle", "wrongprior"])]
    if len(van) == 0 or len(pri) == 0:
        continue
    sub["ratio"] = sub.tr / sub[PR]
    van = sub[sub.role == "vanilla"]
    pri = sub[sub.role.isin(["prior", "ablation"])]
    veh = sub[sub.role.isin(["vehicle", "wrongprior"])]
    v = van.ratio.iloc[0]
    up_tr = int((pri.tr > van.tr.iloc[0]).sum())
    dn_er = int((pri[PR] < van[PR].iloc[0]).sum())
    up_ra = int((pri.ratio > v).sum())
    w(f"### {tis}")
    w()
    w(f"- vanilla: tr={van.tr.iloc[0]:.1f}, PR={van[PR].iloc[0]:.2f}, "
      f"ratio={v:.1f} (n={len(van)})")
    multi = len(van) > 1
    if multi:
        for nm, col in (("tr", "tr"), ("PR", PR), ("ratio", "ratio")):
            w(f"  ** vanilla {nm} spread: {van[col].min():.2f}-{van[col].max():.2f}"
              f" — counts below use the FIRST vanilla only (optimistic when the "
              f"vanilla spread is wide)")
    w(f"- tr larger than vanilla:        {up_tr}/{len(pri)}, "
      f"p = {sign_test(up_tr, len(pri)):.4f}")
    w(f"- PR smaller than vanilla:       {dn_er}/{len(pri)}, "
      f"p = {sign_test(dn_er, len(pri)):.4f}")
    w(f"- ratio larger than vanilla:     {up_ra}/{len(pri)}, "
      f"p = {sign_test(up_ra, len(pri)):.4f}")
    if multi:
        # strict version: outside the vanilla range entirely
        hi, lo = van.ratio.max(), van.ratio.min()
        strict = int(((pri.ratio > hi)).sum())
        w(f"- STRICT (ratio above the HIGHEST vanilla, {hi:.1f}): "
          f"{strict}/{len(pri)}, p = {sign_test(strict, len(pri)):.4f}")
    if len(veh):
        w(f"- vehicle/wrong-prior ratio: "
          f"{', '.join(f'{x:.1f}' for x in veh.ratio)}")
    w()

# ------------------------------------------------------ 3c. within-stage check
w("---")
w("## 3c. Robustness: within-stage covariance")
w()
w("Pooling all developmental stages folds the between-stage displacement into "
  "the covariance. Repeating on stage-centred data isolates within-stage "
  "geometry.")
w()
for tis in ["hemo", "neural", "lung"]:
    sub = df[(df.tissue == tis) & (df.iteration == 4)]
    van = sub[sub.role == "vanilla"]
    pri = sub[sub.role.isin(["prior", "ablation"])]
    if len(van) == 0 or len(pri) == 0:
        continue
    v = van.w_eff_rank.iloc[0]
    k = int((pri.w_eff_rank < v).sum())
    w(f"- {tis}: vanilla w_eff_rank={v:.2f}; {k}/{len(pri)} prior runs below, "
      f"p = {sign_test(k, len(pri)):.4f}")
w()

# --------------------------------------------------- 4. neural 2^3 factorial
w("---")
w("## 4. Neural 2^3 factorial: which component drives the spectrum?")
w()
neu = df[(df.tissue == "neural") & (df.iteration == 4)].copy()
fac = neu[neu.role.isin(["vanilla", "prior", "ablation"]) &
          neu.C.notna() & (neu.run.str.startswith(("R0_", "R1_", "R2_", "R3_",
                                                   "R4_", "R5_", "R6_", "R7_")))]
for metric in ["pr", "top1", "resid", "tr"]:
    y = fac[metric].to_numpy(float)
    X = np.column_stack([np.ones(len(fac)), fac["C"], fac["M"], fac["T"]])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = len(fac) - 4
    se = np.sqrt(np.sum(resid ** 2) / dof * np.diag(np.linalg.pinv(X.T @ X)))
    w(f"### {metric}")
    w("| term | beta | se | t | p |")
    w("|---|---|---|---|---|")
    for nm, b, s_ in zip(["intercept", "C (curriculum)", "M (mixture)",
                          "T (decoder audit)"], beta, se):
        t = b / s_ if s_ > 0 else np.nan
        p = 2 * stats.t.sf(abs(t), dof) if np.isfinite(t) else np.nan
        w(f"| {nm} | {b:+.3f} | {s_:.3f} | {t:+.2f} | {p:.3f} |")
    w()

# ------------------------------------------- 5. hemo specificity ladder @s42
w("---")
w("## 5. Hematopoietic specificity ladder (all seed 42, 29,985 cells)")
w()
ladder = [
    ("R14_hemato_vanilla", "vanilla", "no prior machinery"),
    ("R14_hemato_seed42", "vehicle", "prior boost 5.2% / 1054 genes"),
    ("R14_wrongprior_seed42", "wrong prior", "52 neural TFs, boost 0.4% / 88 genes"),
    ("R13_hemato_trueprior_seed42", "true prior", "11 TFs / 325 TRRUST+CORE edges, boost 1.0%"),
]
sub = df[(df.tissue == "hemo") & (df.iteration == 4)]
w("| run | condition | PR | eff_rank (legacy) | top1 % | tr | Δlog(PR) it0→it4 |")
w("|---|---|---|---|---|---|---|")
for run, cond, desc in ladder:
    r = sub[sub.run == run]
    if r.empty:
        continue
    r = r.iloc[0]
    c = ch[ch.run == run]
    d = f"{c.dlog.iloc[0]:+.3f}" if len(c) else "n/a"
    w(f"| {run} | {cond} ({desc}) | {r.pr:.2f} | {r.eff_rank:.2f} | "
      f"{r.top1:.1f} | {r.tr:.1f} | {d} |")
w()

(OUT / "Z_SPECTRUM_3TISSUE.md").write_text("\n".join(L), encoding="utf-8")
print("\nwrote", OUT / "Z_SPECTRUM_3TISSUE.md")
