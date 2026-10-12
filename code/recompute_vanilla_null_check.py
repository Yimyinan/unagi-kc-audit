# -*- coding: utf-8 -*-
"""Recompute the vanilla-vs-vanilla kappa null from the per-seed values that ARE persisted,
instead of trusting the mean/std a JSON reports about itself.

  * vanilla_null_v2.json stores 15 per-seed trH for the G-VAE hematopoietic vanilla seeds
    -> the TRACE-form null is recomputable exactly, as all C(15,2)=210 ordered ratios.
  * the CORRECTED (aggregate-predictability) null needs per-seed (Z, X); the stagedata and
    loading matrices are gone from this machine (0 hits workspace-wide), so it is NOT
    recomputable here -- this script says so plainly rather than guessing.
"""
import io, json, itertools, statistics, os, glob

P = r"F:/UNAGI_Project/paper1/paper1_kc_diagnostic"

def band(m, s):
    return (round(m - 2 * s, 4), round(m + 2 * s, 4))

# ---------------------------------------------------------------- trace-form null
f = os.path.join(P, "data", "vanilla_null_v2.json")
d = json.load(io.open(f, encoding="utf-8-sig"))
trH = d["trH"]
seeds = list(trH)
vals = [trH[s] for s in seeds]
print("=== G-VAE hematopoietic vanilla: trace-form null ===")
print("  seeds stored      :", len(seeds), "->", ", ".join(map(str, seeds)))
print("  n ordered pairs   :", len(seeds) * (len(seeds) - 1), "(reported: %s)" % d.get("n_pairs"))
ratios = [a / b for a, b in itertools.permutations(vals, 2)]
m, s = statistics.mean(ratios), statistics.stdev(ratios)
print("  recomputed mean   : %.4f   (json says %.4f)" % (m, d["mean"]))
print("  recomputed sd     : %.4f   (json says %.4f)" % (s, d["std"]))
print("  recomputed +-2sd  : %s     (json says %s)" % (band(m, s), (round(d["lo2s"], 4), round(d["hi2s"], 4))))
print("  -> TRACE-form band = [%.3f, %.3f]" % band(m, s))
print()

# ---------------------------------------------------------------- is a corrected null persisted?
print("=== searching for a persisted CORRECTED-statistic null ===")
hits = []
for p in glob.glob(r"F:/UNAGI_Project/**/*.json", recursive=True):
    if "/.venvs/" in p or "/node_modules/" in p:
        continue
    try:
        t = io.open(p, encoding="utf-8-sig", errors="ignore").read()
    except Exception:
        continue
    low = t.lower()
    if ("0.032" in t or "0.937" in t or "1.064" in t) and ("null" in low or "band" in low or "vanilla" in low):
        hits.append((os.path.relpath(p, r"F:/UNAGI_Project"), len(t)))
if hits:
    for h, n in hits:
        print("   %s  (%d bytes)" % (h, n))
else:
    print("   NONE -- no JSON persists a corrected-statistic vanilla null")
print()

# ---------------------------------------------------------------- data availability
print("=== are the inputs still on this machine? ===")
for pat in ("stagedata", "loading_matrix"):
    n = len(glob.glob(r"F:/UNAGI_Project/**/" + pat, recursive=True))
    print("   %-16s -> %d hits" % (pat, n))
print()
print("CONCLUSION: the trace-form null is reproducible from stored per-seed values;")
print("            the corrected-statistic null is NOT reproducible here (inputs removed).")
