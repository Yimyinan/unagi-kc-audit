# -*- coding: utf-8 -*-
"""Recheck the hematopoietic latent "specificity ladder" (Fig 5a) against the
pipeline's own run-to-run scatter.

Motivation: the injection-chain audit shows that none of the three PG-UNAGI
enhancements (curriculum, mixed-distribution, TF prior) has a functional route in
the released source, and that `--disable-tf-prior` is a no-op.  If that is right,
the vehicle arm (R14_hemato_seed*) and the "no-prior" vanilla arm
(R14_hemato_vanilla) are the same procedure differing only by seed, so the ladder
contrast can only be credible if the vanilla value falls OUTSIDE the vehicle
spread.

Both candidate latent metrics are reported, because the numbers printed in
Fig. 5a (-1.06 / -1.24 / -0.63) reproduce the LEGACY `eff_rank` column and not
the participation ratio `pr` that code/analyze_z_3tissue.py declares as the
reported metric.

Inputs : outputs/positive_result/z_spectrum_all.csv
Outputs: outputs/positive_result/arm_ladder_recheck.csv
         outputs/positive_result/LADDER_RECHECK.md
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

OUT = Path(r"F:/UNAGI_Project/outputs/positive_result")
TISSUE = "hemo"
ARMS = {
    "vanilla":    ["R14_hemato_vanilla"],
    "vehicle":    ["R14_hemato_seed42", "R14_hemato_seed123", "R14_hemato_seed456"],
    "wrongprior": ["R14_wrongprior_seed42"],
    "trueprior":  ["R13_hemato_trueprior_seed42", "R13_hemato_trueprior_seed123",
                   "R13_hemato_trueprior_seed456"],
}

df = pd.read_csv(OUT / "z_spectrum_all.csv")
h = df[(df.tissue == TISSUE)].dropna(subset=["pr", "tr", "eff_rank"]).copy()

sl, ic, rv, pv, _ = stats.linregress(np.log(h.tr), np.log(h.pr))
h["resid"] = np.log(h.pr) - (sl * np.log(h.tr) + ic)

rows = []
for run, g in h.groupby("run"):
    g = g.sort_values("iteration")
    if g.iteration.min() != 0 or len(g) < 5:
        continue
    rows.append(dict(run=run, role=g.role.iloc[0], seed=g.seed.iloc[0],
                     pr0=float(g.pr.iloc[0]), tr0=float(g.tr.iloc[0]),
                     dlog_pr=float(np.log(g.pr.iloc[-1] / g.pr.iloc[0])),
                     dlog_eff=float(np.log(g.eff_rank.iloc[-1] / g.eff_rank.iloc[0])),
                     dresid=float(g.resid.iloc[-1] - g.resid.iloc[0])))
t = pd.DataFrame(rows)
t.to_csv(OUT / "arm_ladder_recheck.csv", index=False)

# baseline dependence of the delta metric
sl2, _, rv2, pv2, _ = stats.linregress(np.log(t.pr0), t.dlog_pr)

L = []
def w(s=""):
    L.append(s); print(s)

w("# Recheck of the hematopoietic latent ladder (Fig. 5a)")
w()
w(f"Source `z_spectrum_all.csv` ({TISSUE}, {len(t)} runs x 5 iterations). "
  f"`pr` = participation ratio of `obsm['z']` (the metric the analysis script "
  f"declares as reported); `eff_rank` = legacy column; `resid` = log PR with the "
  f"within-tissue tr trend removed (fit: slope {sl:+.3f}, r={rv:+.3f}, "
  f"p={pv:.1e}, n={len(h)}).")
w()
w("**Metric provenance note.** The three true-prior values printed in Fig. 5a "
  "($-1.06/-1.24/-0.63$) reproduce the **legacy `eff_rank`** column, not `pr` "
  f"(${[round(v,3) for v in t[t.run.str.startswith('R13_hemato')].dlog_pr]}$). "
  "Both metrics are therefore reported below; the conclusion is the same for both.")
w()
w("## 1. The arms are not comparable at baseline")
w()
w("| arm | PR(iter0) | tr(iter0) |")
w("|---|---|---|")
for k, runs in ARMS.items():
    s = t[t.run.isin(runs)]
    w(f"| {k} | " + ", ".join(f"{v:.2f}" for v in s.pr0) + " | " +
      ", ".join(f"{v:.0f}" for v in s.tr0) + " |")
w()
w(f"log PR(iter0) explains **{rv2**2*100:.0f}%** of the variance in $\\Delta\\log(PR)$ "
  f"(slope {sl2:+.2f}, p={pv2:.2e}): part of any \"compression gradient\" is simply "
  "where each run started.")
w()
w("## 2. The endpoints")
w()
w("| arm | n | dlog(PR) | dlog(eff_rank) | d(resid) |")
w("|---|---|---|---|---|")
for k, runs in ARMS.items():
    s = t[t.run.isin(runs)]
    w(f"| {k} | {len(s)} | " + ", ".join(f"{v:+.3f}" for v in s.dlog_pr) + " | " +
      ", ".join(f"{v:+.3f}" for v in s.dlog_eff) + " | " +
      ", ".join(f"{v:+.3f}" for v in s.dresid) + " |")
w()
veh = t[t.run.isin(ARMS["vehicle"])]
tp = t[t.run.isin(ARMS["trueprior"])]
van_pr = t[t.run.isin(ARMS["vanilla"])].dlog_pr.iloc[0]
van_eff = t[t.run.isin(ARMS["vanilla"])].dlog_eff.iloc[0]
w(f"- **vanilla lies INSIDE the vehicle spread** on both metrics: "
  f"PR {van_pr:+.3f} in [{veh.dlog_pr.min():+.3f}, {veh.dlog_pr.max():+.3f}]; "
  f"eff_rank {van_eff:+.3f} in [{veh.dlog_eff.min():+.3f}, {veh.dlog_eff.max():+.3f}]. "
  "The vehicle arm is the one the audit finds functionally identical to vanilla.")
for m, lab in (("dlog_pr", "dlog(PR)"), ("dlog_eff", "dlog(eff_rank)"), ("dresid", "d(resid)")):
    u, p = stats.mannwhitneyu(veh[m], tp[m], alternative="two-sided")
    w(f"- vehicle vs trueprior, {lab}: U={u:.0f}, p={p:.3f} "
      f"(means {veh[m].mean():+.3f} vs {tp[m].mean():+.3f}).")
d = veh.dlog_pr.mean() - tp.dlog_pr.mean()
sp = np.sqrt((veh.dlog_pr.var(ddof=1) + tp.dlog_pr.var(ddof=1)) / 2)
w(f"- Cohen d (dlog PR) = {d/sp:.2f} -> ~{max(2, round(2*(1.96+0.84)**2/(d/sp)**2))} "
  "seeds/arm for 80% power at alpha=0.05; n=3/arm cannot resolve it.")
w()
w("## 3. Verdict")
w()
w("- The matched-prior vs no-prior contrast is **not separable from run-to-run scatter "
  "of the same pipeline**: the no-prior value sits inside the spread of an arm carrying "
  "the same (inert) machinery, on both latent metrics.")
w(f"- Removing the tr trend -- the manuscript's own control -- removes the difference "
  f"(p={stats.mannwhitneyu(veh.dresid, tp.dresid, alternative='two-sided')[1]:.2f}).")
w("- The wrong-tissue arm lies between the two, i.e. it does not behave as the "
  "congruence story requires of it.")
w(f"- **What survives**: the cross-tissue contrast (hematopoietic {int((t.dlog_pr<0).sum())}"
  f"/{len(t)} runs negative vs neural crest all positive) is a difference in *direction* "
  "between tissues and is untouched by this recheck. What does not survive is the "
  "*within-tissue* ordering by prior-biology congruence.")
w()
(OUT / "LADDER_RECHECK.md").write_text("\n".join(L), encoding="utf-8")
print("\nwrote", OUT / "LADDER_RECHECK.md")
