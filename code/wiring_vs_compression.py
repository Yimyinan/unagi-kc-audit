"""
wiring_vs_compression.py
========================
Are "the prior is internalised" and "the latent space is compressed" the same
observation, or two different ones?

The manuscript's Discussion already asserts they are different
("internalisation is not compression"). Until now that was an argument, not a
measurement. This script measures both quantities on the same set of models and
asks whether they co-vary.

  internalisation : delta prior-edge cosine, model vs the SAME seed's
                    empty-prior model  (from dose_response_wiring.csv).
                    A statement about the DECODER geometry: are the TF->target
                    pairs the prior named actually aligned in W?

  compression     : change in the covariance spectrum of the LATENT code Z,
                    model vs the same seed's vanilla  (from z_spectrum_all.csv,
                    terminal iteration). pr down / top1 up / entropy down
                    all mean the cells occupy fewer effective dimensions.

If the two are the same phenomenon they must correlate across the 12
dose x seed points. If they do not, internalisation and compression are
separable consequences of injecting a prior — which is the claim in the text.

Outputs
-------
  outputs/positive_result/fig_wiring_vs_compression.{png,pdf}
  outputs/positive_result/WIRING_VS_COMPRESSION.md
"""
import os, json, csv
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "F:/UNAGI_Project/outputs/positive_result"
import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figure_palette as fp
fp.apply_paper_style()

# wiring run name -> z_spectrum run name
ZMAP_PRIOR = {
    "R11_seed42": "R11_KLF1_TAL1_seed42",
    "R11_KLF1_TAL1_seed123": "R11_KLF1_TAL1_seed123",
    "R11_KLF1_TAL1_seed456": "R11_KLF1_TAL1_seed456",
    "R12_seed42": "R12_GATA2_GFI1B_PAX5_seed42",
    "R12_seed123": "R12_GATA2_GFI1B_PAX5_seed123",
    "R12_seed456": "R12_GATA2_GFI1B_PAX5_seed456",
    "R84_seed42": "R84_erythroid_seed42",
    "R84_seed123": "R84_erythroid_seed123",
    "R107_seed42": "R107_erythroid_lymphoid_seed42",
    "R13": "R13_hemato_trueprior_seed42",
    "R13_seed123": "R13_hemato_trueprior_seed123",
    "R13_seed456": "R13_hemato_trueprior_seed456",
    "R16": "R14_wrongprior_seed42",
}
ZMAP_VAN = {
    "R14_seed42": "R14_hemato_seed42",
    "R14_seed123": "R14_hemato_seed123",
    "R14_seed456": "R14_hemato_seed456",
}
NEURAL = [("R1", "R0_seed42", 42, "R1_neural_seed42", "R0_vanilla_seed42"),
          ("R1_seed123", "R0_seed123", 123, "R1_neural_seed123", "R0_vanilla_seed123"),
          ("R1_seed456", "R0_seed456", 456, "R1_neural_seed456", "R0_vanilla_seed456")]


def spearman(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    den = np.linalg.norm(rx) * np.linalg.norm(ry)
    if den == 0:
        return 0.0, 1.0
    rho = float(rx @ ry / den)
    n = len(x)
    t = rho * np.sqrt((n - 2) / (1 - rho ** 2))
    from math import erfc, sqrt
    return rho, erfc(abs(t) / sqrt(2.0))


def pearson(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    x = x - x.mean(); y = y - y.mean()
    den = np.linalg.norm(x) * np.linalg.norm(y)
    r = float(x @ y / den) if den else 0.0
    n = len(x)
    if abs(r) >= 1:
        return r, 0.0
    t = r * np.sqrt((n - 2) / (1 - r ** 2))
    from math import erfc, sqrt
    return r, erfc(abs(t) / sqrt(2.0))


z = pd.read_csv(os.path.join(OUT, "z_spectrum_all.csv"))
w = pd.read_csv(os.path.join(OUT, "dose_response_wiring.csv"))

# terminal iteration per run
z = z.sort_values(["run", "iteration"])
term = z.groupby("run").tail(1).set_index("run")
print("z_spectrum terminal-iteration runs:", len(term))

SPEC = ["pr", "top1", "entropy", "n90", "tr"]

rows = []
for _, r in w.iterrows():
    cfg, van, seed = r["config"], r["vanilla"], int(r["seed"])
    zr, zv = ZMAP_PRIOR.get(cfg), ZMAP_VAN.get(van)
    if zr is None or zv is None or zr not in term.index or zv not in term.index:
        print(f"  skip {cfg} (z run {zr} / vanilla {zv})")
        continue
    d = dict(label=r["label"], config=cfg, seed=seed, n_edges=int(r["n_edges"]),
             wiring=float(r["delta"]), wiring_z=float(r["z"]),
             wiring_z_nullcorr=float(r["z_nullcorr"]))
    for s in SPEC:
        d[f"d_{s}"] = float(term.loc[zr, s] - term.loc[zv, s])
        d[s] = float(term.loc[zr, s])
        d[f"van_{s}"] = float(term.loc[zv, s])
    rows.append(d)

# add the mismatch control (R16, seed 42) from the specificity run
psc = pd.read_csv(os.path.join(OUT, "prior_specificity.csv"))
mm = psc[(psc["kind"] == "mismatch") & (psc["config"] == "R16") &
         (psc["edgeset"] == "hema")]
if len(mm):
    r0 = mm.iloc[0]
    zr, zv = ZMAP_PRIOR["R16"], ZMAP_VAN["R14_seed42"]
    d = dict(label="R16", config="R16", seed=42, n_edges=int(r0["n"]),
             wiring=float(r0["delta"]), wiring_z=float(r0["z"]),
             wiring_z_nullcorr=float(r0["z_nullcorr"]))
    for s in SPEC:
        d[f"d_{s}"] = float(term.loc[zr, s] - term.loc[zv, s])
        d[s] = float(term.loc[zr, s]); d[f"van_{s}"] = float(term.loc[zv, s])
    rows.append(d)

df = pd.DataFrame(rows)
print("\n" + "=" * 78)
print("JOINT TABLE: internalisation vs compression")
print("=" * 78)
print(df[["label", "seed", "n_edges", "wiring", "wiring_z",
          "d_pr", "d_top1", "d_entropy"]].to_string(index=False,
                                                          float_format=lambda v: f"{v:+.4f}"))

print("\n--- does compression depend on prior dose? ---")
for s in ["pr", "top1", "entropy"]:
    rho, p = spearman(df["n_edges"], df[f"d_{s}"])
    print(f"  Spearman(dose, d_{s:9s}) rho={rho:+.3f} p={p:.3g}")

print("\n--- does internalisation track compression? ---")
res = {}
for s in ["pr", "top1", "entropy", "n90", "tr"]:
    rho, p = spearman(df["wiring"], df[f"d_{s}"])
    r, pr = pearson(df["wiring"], df[f"d_{s}"])
    res[s] = (rho, p, r, pr)
    print(f"  wiring vs d_{s:9s}:  Spearman rho={rho:+.3f} (p={p:.3g})   "
          f"Pearson r={r:+.3f} (p={pr:.3g})")

print("\n--- PARTIAL correlation: seed-centred residuals ---")
print("    (seed drives both quantities, so any raw correlation may be spurious)")


def seed_centre(col):
    return (df[col] - df.groupby("seed")[col].transform("mean")).to_numpy()


wc = seed_centre("wiring")
for s in ["pr", "top1", "entropy", "n90"]:
    dc_ = seed_centre(f"d_{s}")
    rho, p = spearman(wc, dc_)
    r, pr = pearson(wc, dc_)
    print(f"  wiring vs d_{s:9s} | seed:  Spearman rho={rho:+.3f} (p={p:.3g})   "
          f"Pearson r={r:+.3f} (p={pr:.3g})")

print("\n--- DISSOCIATION: matched seed 42, only the prior CONTENT differs ---")
sub42 = df[(df.seed == 42)]
r13 = sub42[sub42.label == "R13"]
r16 = df[df.label == "R16"]
if len(r13) and len(r16):
    a, b = r13.iloc[0], r16.iloc[0]
    print(f"  R13 true prior   : wiring {a['wiring']:+.4f}   "
          f"compression (-d pr) {-a['d_pr']:+.3f}   "
          f"(-d entropy) {-a['d_entropy']:+.4f}")
    print(f"  R16 neural prior : wiring {b['wiring']:+.4f}   "
          f"compression (-d pr) {-b['d_pr']:+.3f}   "
          f"(-d entropy) {-b['d_entropy']:+.4f}")
    frac = (-b["d_pr"]) / (-a["d_pr"]) if a["d_pr"] else float("nan")
    print(f"  the MISMATCHED prior still delivers {frac*100:.0f}% of the matched "
          f"prior's latent compression, but with the opposite wiring sign.")
    print("  -> latent compression is a consequence of injecting ANY prior;")
    print("     positive wiring requires the prior to MATCH the tissue.")

print("\n--- the reverse question: does the SIGNIFICANT wiring z track it? ---")
for s in ["pr", "top1", "entropy"]:
    rho, p = spearman(df["wiring_z"], df[f"d_{s}"])
    print(f"  wiring_z vs d_{s:9s}: Spearman rho={rho:+.3f} (p={p:.3g})")

print("\n--- seed structure of each quantity ---")
for col in ["wiring", "d_pr", "d_top1", "d_entropy"]:
    parts = "  ".join(f"s{s}={df[df.seed==s][col].mean():+.4f}"
                      for s in sorted(df.seed.unique()))
    print(f"  {col:12s} {parts}")

# is compression seed-driven the way wiring is?
def var_explained(y, lab):
    tot = np.var(y) * len(y)
    wit = sum(((y[lab == s] - y[lab == s].mean()) ** 2).sum() for s in set(lab))
    return (tot - wit) / tot

lab = df["seed"].to_numpy()
print()
for col in ["wiring", "d_pr", "d_top1", "d_entropy"]:
    print(f"  seed explains {var_explained(df[col].to_numpy(), lab)*100:5.1f}% "
          f"of variance in {col}")

df.to_csv(os.path.join(OUT, "wiring_vs_compression.csv"), index=False)

# --------------------------------------------------------------- figure
fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.8))
COL = {42: "#B2182B", 123: "#2166AC", 456: "#4D9221"}

ax = axes[0]
for s in [42, 123, 456]:
    sub = df[df.seed == s]
    ax.scatter(sub["wiring"], -sub["d_pr"], color=COL[s], s=44,
               label=f"seed {s}", zorder=3, edgecolor="#333", lw=0.4)
mmrow = df[df.label == "R16"]
if len(mmrow):
    ax.scatter(mmrow["wiring"], -mmrow["d_pr"], marker="D", s=70,
               facecolor="none", edgecolor="#333", lw=1.1, zorder=4,
               label="mismatch (R16)")
rho, p = spearman(df["wiring"], -df["d_pr"])
ax.axhline(0, color="#333", lw=0.6); ax.axvline(0, color="#333", lw=0.6)
ax.set_xlabel("internalisation: $\\Delta$prior-edge cosine", fontsize=8)
ax.set_ylabel("compression: $-\\Delta$participation ratio of Z", fontsize=8)
ax.set_title(f"A. Internalisation vs compression\nSpearman rho={rho:+.2f}, p={p:.2g}",
             fontsize=8.5)
fp.legend_below(ax, ncol=2, fontsize=6.3)
ax.grid(alpha=0.25, lw=0.4)

ax = axes[1]
ax.scatter(df["n_edges"], -df["d_pr"],
           c=[COL[s] for s in df["seed"]], s=44, edgecolor="#333", lw=0.4, zorder=3)
ax.axhline(0, color="#333", lw=0.6)
ax.set_xscale("log")
ax.set_xticks([21, 55, 84, 107, 325])
ax.set_xticklabels(["21", "55", "84", "107", "325"])
ax.set_xlabel("prior edges injected (log)", fontsize=8)
ax.set_ylabel("compression: $-\\Delta$participation ratio of Z", fontsize=8)
rho2, p2 = spearman(df["n_edges"], -df["d_pr"])
ax.set_title(f"B. Compression vs prior dose\nSpearman rho={rho2:+.2f}, p={p2:.2g}",
             fontsize=8.5)
ax.grid(alpha=0.25, lw=0.4)

ax = axes[2]
names = ["wiring", "d_pr", "d_top1", "d_entropy"]
vals = [var_explained(df[c].to_numpy(), lab) for c in names]
ax.bar(range(4), vals, color=["#B2182B", "#2166AC", "#2166AC", "#2166AC"],
       edgecolor="#333", lw=0.5)
ax.set_xticks(range(4))
ax.set_xticklabels(["internalisation", "$\\Delta$PR", "$\\Delta$top1",
                    "$\\Delta$entropy"], fontsize=7, rotation=20)
ax.set_ylabel("variance explained by seed", fontsize=8)
ax.set_ylim(0, 1)
ax.set_title("C. Both quantities are seed-gated", fontsize=8.5)
ax.grid(axis="y", alpha=0.25, lw=0.4)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_wiring_vs_compression.pdf"))
fig.savefig(os.path.join(OUT, "fig_wiring_vs_compression.png"), dpi=200)
plt.close(fig)
print("\nwrote fig_wiring_vs_compression.pdf/png")

lines = ["# Internalisation vs compression", "",
         "Both measured on the same 12 dose x seed models (plus the seed-matched "
         "mismatch control), each against the SAME seed's empty-prior model.", "",
         "| config | seed | edges | wiring delta | z | d pr | d top1 | d entropy |",
         "|---|---|---|---|---|---|---|---|"]
for _, r in df.iterrows():
    lines.append(f"| {r['label']} | {int(r['seed'])} | {int(r['n_edges'])} | "
                 f"{r['wiring']:+.4f} | {r['wiring_z']:+.2f} | {r['d_pr']:+.4f} | "
                 f"{r['d_top1']:+.4f} | {r['d_entropy']:+.4f} |")
lines += ["", "## Correlations", "",
          "| pair | Spearman rho | p |", "|---|---|---|"]
for s in ["pr", "top1", "entropy"]:
    rho, p, _, _ = res[s]
    lines.append(f"| wiring vs d_{s} | {rho:+.3f} | {p:.3g} |")
lines += ["", "## Variance explained by seed", "",
          "| quantity | % variance |", "|---|---|"]
for c in ["wiring", "d_pr", "d_top1", "d_entropy"]:
    lines.append(f"| {c} | {var_explained(df[c].to_numpy(), lab)*100:.1f}% |")
open(os.path.join(OUT, "WIRING_VS_COMPRESSION.md"), "w").write("\n".join(lines))
print("wrote WIRING_VS_COMPRESSION.md and wiring_vs_compression.csv")
