# -*- coding: utf-8 -*-
"""Data-authenticity audit for the positive-biology demo (validate_prior_biology.py).

Checks, independently of the original script:
  A. The 4 ladder runs' stagedata h5ad files really exist on disk (real model outputs).
  B. The ladder dlog values reproduce from z_spectrum_all.csv, and the raw iter0/iter4
     eff_rank curves look like genuine training dynamics (not flat/fake).
  C. Multi-seed support: all 3 true-prior seeds and all 3 vehicle seeds.
  D. hema_prior.json / mouse_fibrosis_prior.json structure & counts.
  E. TRRUST v2 file format, size, sample, and the validation numbers recompute.
"""
from __future__ import annotations
import json, glob, re
from pathlib import Path
import numpy as np
import pandas as pd

RES = Path(r"F:/UNAGI_Project/UNAGI/result")
OUT = Path(r"F:/UNAGI_Project/outputs/positive_result")
DATA = Path(r"F:/UNAGI_Project/next_phase/data")
TRR = Path(r"F:/UNAGI_Project/unagi_custom_data/trrust_raw_human.txt")

print("="*78); print("A. STAGEDATA PROVENANCE  (do the ladder runs have real h5ad outputs?)")
print("="*78)
ladder_paths = {
    "vanilla  (R14_hemato_vanilla)":        RES/"R14"/"R14_hemato_vanilla",
    "vehicle  (R15/R14_hemato_seed42)":     RES/"R15"/"R14_hemato_seed42",
    "wrongprior (R16/R14_wrongprior_seed42)":RES/"R16"/"R14_wrongprior_seed42",
    "trueprior (R13/R13_hemato_trueprior_seed42)": RES/"R13"/"R13_hemato_trueprior_seed42",
}
for name, root in ladder_paths.items():
    if not root.exists():
        print(f"  [MISSING DIR] {name} -> {root}"); continue
    per_iter = []
    ok = True
    for it in range(5):
        sd = root / f"iter_{it}" / "stagedata"
        h5 = glob.glob(str(sd / "*.h5ad")) if sd.exists() else []
        h5 = [h for h in h5 if not h.endswith("dataset.h5ad")]
        per_iter.append(len(h5))
        if not h5: ok = False
    print(f"  {'OK ' if ok else 'GAP'} {name:42s} h5ad/stage per iter0-4: {per_iter}")

print(); print("="*78); print("B. LADDER REPRODUCIBILITY  (recompute dlog from CSV, show raw curves)")
print("="*78)
df = pd.read_csv(OUT/"z_spectrum_all.csv")
def dlog(sub):
    sub_df = df[df["run"].str.contains(sub, na=False)]
    if sub_df.empty: return float("nan"), []
    i0 = sub_df[sub_df["iteration"]==0]["eff_rank"]; i4 = sub_df[sub_df["iteration"]==4]["eff_rank"]
    # raw mirror: show the per-seed/per-run iter0 & iter4 means
    raw = []
    for run, g in sub_df.groupby("run"):
        r0 = g[g["iteration"]==0]["eff_rank"].mean(); r4 = g[g["iteration"]==4]["eff_rank"].mean()
        raw.append((run, r0, r4))
    return float(np.log(i4.mean())-np.log(i0.mean())), raw

ladder = {
    "vanilla":    "R14_hemato_vanilla",
    "vehicle":    "R14_hemato_seed42",
    "wrongprior": "R14_wrongprior_seed42",
    "trueprior":  "R13_hemato_trueprior_seed42",
}
for k, sub in ladder.items():
    v, raw = dlog(sub)
    print(f"  {k:10s} dlog={v:+.3f}   (iter0->iter4 eff_rank per run:)")
    for run, r0, r4 in raw:
        print(f"        {run:42s} {r0:8.3f} -> {r4:8.3f}")

print(); print("C. MULTI-SEED SUPPORT for true-prior and vehicle conditions")
for sub in ["R13_hemato_trueprior_seed", "R14_hemato_seed"]:
    sub_df = df[df["run"].str.contains(sub, na=False)]
    print(f"  pattern '{sub}':")
    for run, g in sub_df.groupby("run"):
        r0 = g[g["iteration"]==0]["eff_rank"].mean(); r4 = g[g["iteration"]==4]["eff_rank"].mean()
        extra = sub_df[sub_df["run"]==run]
        # iteration coverage
        its = sorted(extra["iteration"].unique())
        v = np.log(r4)-np.log(r0)
        print(f"     {run:42s} iters={its} dlog={v:+.3f}")

print(); print("="*78); print("D. PRIOR JSON STRUCTURE")
print("="*78)
hema = json.load(open(DATA/"hema_prior.json"))
hk = list(hema["tf_db"].keys())
he = [(tf, t) for tf, ts in hema["tf_db"].items() for t in ts]
print(f"  hema_prior.json: tf_db keys={len(hk)}  edges={len(he)}  keys sample={hk[:6]}")
lung = json.load(open(DATA/"mouse_fibrosis_prior.json"))
lt = lung.get("tfs", lung.get("tf_list", None))
print(f"  mouse_fibrosis_prior.json: tfs field type={type(lt).__name__}  n={len(lt) if hasattr(lt,'__len__') else '?'}")
if isinstance(lt, list):
    print(f"     sample={lt[:6]}")

print(); print("="*78); print("E. TRRUST v2 FILE  +  validation recompute")
print("="*78)
trrust = {}
nlines = 0
with open(TRR) as f:
    for line in f:
        if line.startswith("#"): continue
        nlines += 1
        p = line.rstrip("\n").split("\t")
        if len(p) < 3: continue
        tf, tgt = p[0].upper(), p[1].upper()
        trrust.setdefault(tf, set()).add(tgt)
print(f"  file lines={nlines}  TFs={len(trrust)}  edges={sum(len(v) for v in trrust.values())}")
sample = list(trrust.items())[:3]
for tf, tg in sample:
    print(f"     {tf} -> {len(tg)} targets e.g. {sorted(tg)[:4]}")

# recompute the validation pieces used by the figure
HEMATO_MR = {"GATA1","GATA2","KLF1","SPI1","CEBPA","CEBPB","PAX5","RUNX1","TAL1",
             "LMO2","GFI1","GFI1B","NFE2","MEF2C","IKZF1","LYL1"}
hema_tfs = set(t.upper() for t in hk)
hema_deg = [len(trrust[t]) for t in hema_tfs if t in trrust]
def jac(a,b): a,b=set(a),set(b); return len(a&b)/len(a|b) if (a|b) else 0.0
print(f"  hema prior TF-degree median={np.median(hema_deg):.0f}  MR Jaccard(hema)={jac(hema_tfs,HEMATO_MR):.2f}")
hema_cov = sum(1 for tf,t in he if t in trrust.get(tf,()))/len(he)
print(f"  hema edge coverage in TRRUST = {hema_cov*100:.0f}%  ({sum(1 for tf,t in he if t in trrust.get(tf,()))}/{len(he)})")
