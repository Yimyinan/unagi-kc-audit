# -*- coding: utf-8 -*-
"""
kappa-C-Z : a latent-space (Z) companion diagnostic mirroring the W-only kappa-C.

V2 -- tightened null (2026-09-05)
---------------------------------
* Added the one extra vanilla seed found on F: (R0_vanilla_unagi_20260530_223153),
  so the neural-crest family now has 3 vanilla controls instead of 2.
* Replaced the mixed "pooled-control / vanilla-only envelope" labelling with ONE
  consistent VANILLA-ONLY null (matches kappa-C's vanilla cross-seed band and the
  manuscript's own Z analysis). vehicle / wrongprior are NOT used as a Z null
  because they are different training configurations that would inflate the band.
* The Z statistic is the rotation-invariant directional compaction
  zeta_Z = (mu_vanilla - eff_rank) / sd_vanilla (where >=2 vanilla exist). Positive
  = more compressed than the vanilla centre (lower effective rank). With a single
  vanilla the flag falls back to directional (prior eff_rank < vanilla).
* "Z-reorganized" is an EFFECT-SIZE flag (zeta_Z >= Z_THRESH), reported with the
  empirical one-sided permutation p = (#controls <= run's eff_rank + 1)/(M+1).
  Because control counts are small (M=1..5), the permutation p is reported for
  transparency but the flag is deliberately effect-size based -- we state this.
* kappa-C (W) = 0/N is cited from the manuscript (NOT recomputed: prior_design.json
  name mapping / C values are unreliable). The "W misses what Z catches" contrast
  is therefore exact.

Outputs
-------
  paper1/positive_result/KAPPA_C_Z.md
  paper1/positive_result/z_spectrum_all_augmented.csv   (incl. new vanilla)
  paper1/paper1_kc_diagnostic/figures/fig_kappa_c_z.png / .pdf
"""
from __future__ import annotations
import csv, os, glob, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import figure_palette as fp
SRC = os.path.join(HERE, "..", "..", "positive_result", "z_spectrum_all.csv")
OUT_MD = os.path.join(HERE, "..", "..", "positive_result", "KAPPA_C_Z.md")
OUT_CSV = os.path.join(HERE, "..", "..", "positive_result", "z_spectrum_all_augmented.csv")
FIG_PNG = os.path.join(HERE, "..", "figures", "fig_kappa_c_z.png")
FIG_PDF = os.path.join(HERE, "..", "figures", "fig_kappa_c_z.pdf")

Z_FEAT = "pr"  # participation ratio -- the single effective-rank metric (VAE / single-cell convention)
CONTROL_ROLES = {"vanilla", "vehicle", "wrongprior"}
PRIOR_ROLES = {"prior", "ablation"}
TISSUE_LABEL = {"hemo": "hematopoietic", "neural": "neural crest", "lung": "lung fibrosis"}
Z_THRESH = 1.5          # primary effect-size flag
SENS = [1.34, 1.5, 2.0] # sensitivity thresholds
F_BASE = r"F:/UNAGI_Project/UNAGI/result"
NEW_VANILLA = "R0/R0_vanilla_unagi_20260530_223153"


# ----------------------------------------------------------- h5py z reader
def read_z_from_h5ad(run_rel: str, n_iters: int = 5):
    """Return list of (iteration, Z, stage) using h5py only (no anndata/scanpy)."""
    import h5py
    root = os.path.join(F_BASE, run_rel)
    out = []
    for it in range(n_iters):
        sd = os.path.join(root, f"iter_{it}", "stagedata")
        if not os.path.isdir(sd):
            sd = os.path.join(root, str(it), "stagedata")
        if not os.path.isdir(sd):
            continue
        blocks, sids = [], []
        for s, h in enumerate(sorted(glob.glob(os.path.join(sd, "*.h5ad")))):
            if h.endswith("dataset.h5ad"):
                continue
            with h5py.File(h, "r") as f:
                if "obsm" not in f or "z" not in f["obsm"]:
                    continue
                z = np.asarray(f["obsm"]["z"][:], dtype=np.float64)
            blocks.append(z); sids.append(np.full(z.shape[0], s))
        if blocks:
            out.append((it, np.vstack(blocks), np.concatenate(sids)))
    return out


def spectrum(Z: np.ndarray) -> float:
    """Pooled effective spectral rank (matches scan_z_spectrum.py pooled metric)."""
    Zc = Z - Z.mean(axis=0)
    cov = np.cov(Zc, rowvar=False)
    ev = np.clip(np.linalg.eigvalsh(cov)[::-1], 0.0, None)
    tot = ev.sum()
    p = ev / tot
    pnz = p[p > 0]
    # participation ratio (VAE / single-cell convention), the single
    # effective-rank metric now used throughout the manuscript
    return float(1.0 / (pnz ** 2).sum())


def add_new_vanilla(rows: list[dict]) -> list[dict]:
    """Compute and append the extra neural vanilla rows (iters 0..4)."""
    if any(r["run"] == "R0_vanilla_unagi_20260530_223153" for r in rows):
        return rows
    for it, Z, _stage in read_z_from_h5ad(NEW_VANILLA, 5):
        rows.append(dict(run="R0_vanilla_unagi_20260530_223153", tissue="neural",
                         role="vanilla", seed="unagi", iteration=it,
                         eff_rank=spectrum(Z), pr=spectrum(Z)))
    return rows


def load_runs(path):
    rows = list(csv.DictReader(open(path, newline="")))
    rows = add_new_vanilla(rows)
    runs = {}
    for r in rows:
        key = r["run"]; it = int(r["iteration"])
        runs.setdefault(key, {"tissue": r["tissue"], "role": r["role"],
                              "seed": r["seed"], "iters": {}})
        runs[key]["iters"][it] = float(r[Z_FEAT])
    out = []
    for key, info in runs.items():
        term = max(info["iters"])
        out.append({"run": key, "tissue": info["tissue"], "role": info["role"],
                    "seed": info["seed"], "iter": term, "z": info["iters"][term]})
    return out


def main():
    runs = load_runs(SRC)
    tissues = ["hemo", "neural", "lung"]
    report = ["# kappa-C-Z : latent-space (Z) audit (tightened null, V2)\n"]
    report.append("Built from the same master table as kappa-C (z_spectrum_all.csv), "
                  "now with the one extra vanilla seed found on F: "
                  "(R0_vanilla_unagi_20260530_223153) added to the neural-crest family. "
                  "Z statistic = directional compaction zeta_Z = (mu_ctrl - eff_rank)/sd_ctrl, "
                  "derived from the Z covariance eigenvalue spectrum and therefore invariant "
                  "to latent rotation. 'Z-reorganized' is an effect-size flag at "
                  f"zeta_Z >= {Z_THRESH}.\n")
    report.append("> kappa-C (W) side is cited from the manuscript (0/33 null); not recomputed "
                  "here (prior_design.json name mapping / C values unreliable). The contrast "
                  "is therefore exact.\n")
    report.append("**Scope (Paper1):** kappa-C-Z audits the PG-UNAGI latent Z only. The "
                  "separate 15-seed G-VAE hematopoietic vanilla ensemble is a next-paper "
                  "resource (G-VAE prescriptive) and is NOT mixed into this null -- combining "
                  "two different models' nulls would be unsound.")

    all_meta = []
    summary = {}
    for t in tissues:
        truns = [r for r in runs if r["tissue"] == t]
        van = [r for r in truns if r["role"] == "vanilla"]
        priors = [r for r in truns if r["role"] in PRIOR_ROLES]
        if not van or not priors:
            continue
        # vanilla-only null (matches the manuscript's Z analysis + kappa-C's
        # vanilla cross-seed band). vehicle/wrongprior are NOT a Z null: they are
        # different training configurations and would inflate the band.
        van_z = np.array([v["z"] for v in van])
        mu_v = float(van_z.mean())
        sd_v = float(van_z.std(ddof=0)) if len(van) >= 2 else None
        M = len(van)

        n_prior = len(priors)
        per_run = []
        for r in priors:
            if sd_v is not None:
                zz = (mu_v - r["z"]) / sd_v
                p = (np.sum(van_z <= r["z"]) + 1) / (M + 1)
                flag = zz >= Z_THRESH
                method = "vanilla_sigma"
            else:
                zz = float("nan")
                p = float("nan")
                flag = r["z"] < mu_v          # directional vs single vanilla
                method = "single_vanilla_dir"
            per_run.append((r, zz, p, flag, method))
            all_meta.append((t, r["run"], r["role"], (r["z"] < mu_v),
                             zz, p, bool(flag), method))

        n_z = sum(1 for _, _, _, f, _ in per_run if f)
        k = sum(1 for r in priors if r["z"] < mu_v)
        from math import comb
        p_dir = float(sum(comb(n_prior, i) for i in range(k, n_prior + 1)) / 2.0 ** n_prior)
        sens = {th: sum(1 for _, zz, _, _, _ in per_run
                        if (not np.isnan(zz)) and zz >= th) for th in SENS}
        summary[t] = dict(n_van=M, n_prior=n_prior, n_z=n_z, mu=mu_v, sd=sd_v,
                          k=k, p_dir=p_dir, sens=sens, method=("sigma" if sd_v is not None else "dir"))

        rep_method = (f"vanilla-only null, {M} vanilla"
                      + (f", sd={sd_v:.2f}" if sd_v is not None else " (single vanilla -> directional only)"))
        report.append(f"\n## {TISSUE_LABEL.get(t, t)}  (tissue={t})\n")
        report.append(f"- null: **{rep_method}**  (centre eff_rank = {mu_v:.2f})")
        report.append(f"- prior-injected runs audited: {n_prior}")
        if sd_v is not None:
            report.append(f"- **kappa-C-Z (Z): {n_z}/{n_prior} reorganized** "
                          f"(sigma flag zeta_Z >= {Z_THRESH})")
            report.append(f"- sensitivity (flag threshold -> #reorganized): " +
                          ", ".join(f"zeta>={th}: {sens[th]}/{n_prior}" for th in SENS))
        else:
            report.append(f"- **kappa-C-Z (Z): {n_z}/{n_prior} directionally compressed** "
                          f"(prior eff_rank < single vanilla; no sigma flag, underpowered)")
        report.append(f"- directional signal: {k}/{n_prior} priors more compressed than "
                      f"vanilla; one-sided binomial p={p_dir:.2e}")
        report.append(f"- kappa-C (W): **0/{n_prior}** (manuscript 0/33) -> these Z "
                      f"reorganizations are INVISIBLE to kappa-C")
        report.append("\n  | run | role | zeta_Z | empirical p | flag |")
        report.append("  |---|---|---|---|---|")
        for r, zz, p, f, _m in sorted(per_run, key=lambda x: -(x[1] if not np.isnan(x[1]) else -9)):
            zzs = f"{zz:+.2f}" if not np.isnan(zz) else "n/a"
            ps = f"{p:.3f}" if not np.isnan(p) else "n/a"
            report.append(f"  | {r['run']} | {r['role']} | {zzs} | {ps} | "
                          f"{'YES' if f else '-'} |")

    tot_p = sum(s["n_prior"] for s in summary.values())
    tot_k = sum(s["k"] for s in summary.values())                 # directionally compressed
    tot_sigma = sum(s["n_z"] for f, s in summary.items() if s["sd"] is not None)
    tot_sigma_n = sum(s["n_prior"] for f, s in summary.items() if s["sd"] is not None)
    report.append("\n## Cross-tissue totals\n")
    report.append(f"- prior-injected runs audited: **{tot_p}**")
    report.append(f"- **kappa-C-Z (Z) directionally compressed (prior eff_rank < vanilla): "
                  f"{tot_k}/{tot_p}**  (this is the signal kappa-C(W)=0/33 cannot see)")
    report.append(f"- **kappa-C-Z (Z) strong sigma-flag (>= {Z_THRESH} sigma, vanilla-only "
                  f"null families only): {tot_sigma}/{tot_sigma_n}**")
    report.append(f"- kappa-C (W) reorganized: **0/{tot_p}** (manuscript) "
                  f"-> every one of the {tot_k} directionally-compressed runs is INVISIBLE "
                  f"to the W-only lens")

    report.append("\n## Z-reorganized runs (the signal kappa-C misses)\n")
    dec = [m for m in all_meta if m[6]]
    if dec:
        report.append("| tissue | run | role | method | zeta_Z | empirical p |")
        report.append("|---|---|---|---|---|---|")
        for t, run, role, _dir, zz, p, _f, method in dec:
            zzs = f"{zz:+.2f}" if not (isinstance(zz, float) and np.isnan(zz)) else "n/a"
            ps = f"{p:.3f}" if not (isinstance(p, float) and np.isnan(p)) else "n/a"
            report.append(f"| {TISSUE_LABEL.get(t,t)} | {run} | {role} | {method} | {zzs} | {ps} |")
    else:
        report.append("_none_")

    report.append("\n## Honest limitations (V2)\n")
    report.append("- Null is **vanilla-only** (matches the manuscript's own Z analysis and "
                  "kappa-C's vanilla cross-seed band). vehicle/wrongprior are NOT used as a Z "
                  "null: they are different training configurations and would inflate the band. "
                  "neural-crest now has 3 vanilla (added R0_vanilla_unagi_20260530_223153), "
                  "giving a tight sigma-flag null (sd~0.77). hematopoietic and lung each have "
                  "only 1 vanilla on disk, so they fall back to the manuscript's directional "
                  "sign test (prior eff_rank < single vanilla).")
    report.append("- **Scope (Paper1):** kappa-C-Z audits the PG-UNAGI latent Z only. The "
                  "separate 15-seed G-VAE hematopoietic vanilla ensemble is a resource for the "
                  "NEXT paper (G-VAE prescriptive) and is intentionally NOT mixed into this "
                  "PG-UNAGI null -- doing so would conflate two different models. Within "
                  "PG-UNAGI, hematopoietic and lung each have only 1 vanilla on disk, so they "
                  "fall back to the manuscript's directional sign test (prior eff_rank < "
                  "single vanilla); only neural-crest (3 vanilla) supports a real sigma-flag.")
    report.append("- **PG-UNAGI vanilla mapping (on disk):** neural = R0 (R0_vanilla_seed123 / "
                  "R0_vanilla_seed456 / R0_vanilla_unagi_20260530_223153); lung = R10; "
                  "hematopoietic = R14 (R14_hemato_vanilla). The folders R9 (lung IPF prior) "
                  "and R11 (KLF1+TAL1 prior) contain the PRIOR-injected treatments, NOT the "
                  "vanilla controls, so they are correctly excluded from the null.")
    report.append(f"- 'Z-reorganized' (sigma-flag) is an EFFECT-SIZE flag (zeta_Z >= {Z_THRESH}), "
                  "not a significance test; permutation p is reported for transparency but power "
                  "is limited by the small vanilla count (M).")
    report.append("- zeta_Z uses eff_rank only (the manuscript's positive signal). A richer "
                  "rotation-invariant spectrum (full eigenvalue vector) would need raw "
                  "stagedata; deferred.")
    report.append("- kappa-C (W) = 0/N is cited from the manuscript, not recomputed "
                  "(prior_design.json name mapping / C values are unreliable here).")

    open(OUT_MD, "w", encoding="utf-8").write("\n".join(report))
    # augmented csv (incl. new vanilla) for reproducibility
    rows_all = list(csv.DictReader(open(SRC, newline="")))
    rows_all = add_new_vanilla(rows_all)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_all[0].keys()))
        w.writeheader(); w.writerows(rows_all)
    print("\n".join(report))
    print(f"\nwrote {OUT_MD}\nwrote {OUT_CSV}")

    # -------------------------------------------------- figure (2 panels)
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fams = list(summary.keys())
        labels = [TISSUE_LABEL.get(f, f) for f in fams]
        # panel A: grouped bars per family (W=0, Z directional, Z sigma-flag)
        z_dir = [s["k"] for s in summary.values()]
        z_sig = [s["n_z"] if s["sd"] is not None else 0 for s in summary.values()]
        n_vals = [s["n_prior"] for s in summary.values()]
        fig = plt.figure(figsize=(11.0, 4.6))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.35], wspace=0.34)
        axA = fig.add_subplot(gs[0])
        x = np.arange(len(fams)); wbar = 0.26
        axA.bar(x - wbar, [0]*len(fams), wbar, color="#5F5E5A", label="kappa-C (W): 0")
        axA.bar(x, z_dir, wbar, color="#E8A0A8", label="kappa-C-Z: directional")
        axA.bar(x + wbar, z_sig, wbar, color="#C44E52", label="kappa-C-Z: sigma-flag")
        for i, (d, s, n) in enumerate(zip(z_dir, z_sig, n_vals)):
            axA.text(i - wbar, d + 0.15, f"{d}/{n}", ha="center", fontsize=8, color="#B05060")
            if s: axA.text(i + wbar, s + 0.15, f"{s}/{n}", ha="center", fontsize=8, color="#C44E52")
        axA.set_xticks(x); axA.set_xticklabels(labels, fontsize=9)
        axA.set_ylabel("runs (out of N audited)")
        axA.set_ylim(0, max(z_dir) + 3)
        axA.set_title("A.  W-only kappa-C misses the Z reorganization", fontsize=10, fontweight="bold")
        axA.legend(fontsize=7.5, loc="upper left")
        axA.text(0.5, -0.22, "W = decoder readout (manuscript 0/33); Z = latent spectrum (this diagnostic). "
                 "Directional = prior eff_rank < vanilla; sigma-flag = >=1.5 sigma below vanilla (vanilla-only null).",
                 transform=axA.transAxes, ha="center", fontsize=7, color="#555")
        # panel B: eff_rank strip per family with tightened vanilla band
        axB = fig.add_subplot(gs[1])
        palette = {"hemo": fp.C_HEMA, "neural": fp.C_NEURAL, "lung": fp.C_LUNG}
        offset = 0; xticks = []; xlabels = []
        for fi, t in enumerate(fams):
            s = summary[t]
            # vanilla points
            van_runs = [r for r in runs if r["tissue"] == t and r["role"] == "vanilla"]
            for v in van_runs:
                axB.scatter([offset], [v["z"]], c="#222222", marker="*", s=90,
                            edgecolors="white", linewidths=0.5, zorder=5)
            pri_runs = [r for r in runs if r["tissue"] == t and r["role"] in PRIOR_ROLES]
            ys = [r["z"] for r in pri_runs]
            axB.scatter([offset + 0.18]*len(pri_runs), ys, c=palette[t], s=24, zorder=4,
                        edgecolors="white", linewidths=0.4)
            # band
            axB.hlines(s["mu"], offset - 0.35, offset + 0.55, colors="#444", lw=1.0, ls="-")
            if s["sd"] is not None:
                axB.axhspan(s["mu"] - Z_THRESH*s["sd"], s["mu"], color=palette[t],
                            alpha=0.10, zorder=0)
                axB.hlines(s["mu"] - Z_THRESH*s["sd"], offset - 0.35, offset + 0.55,
                           colors=palette[t], lw=1.0, ls="--")
            xticks.append(offset + 0.09); xlabels.append(f"{labels[fi]}\n(n={s['n_prior']})")
            offset += 1.4
        axB.set_xticks(xticks); axB.set_xticklabels(xlabels, fontsize=8)
        axB.set_ylabel("effective spectral rank of Z  (lower = more compressed)")
        axB.invert_yaxis()
        axB.set_title("B.  tightened vanilla-only null: prior below the band = reorganized",
                      fontsize=10, fontweight="bold")
        axB.text(0.5, -0.16, "black * = vanilla centre;   shaded/dashed band = vanilla - 1.5 sigma "
                 "(only where >=2 vanilla exist). Lower = more compressed.",
                 transform=axB.transAxes, ha="center", fontsize=7, color="#555")
        fig.tight_layout()
        fig.savefig(FIG_PNG, dpi=150, bbox_inches="tight"); fig.savefig(FIG_PDF, bbox_inches="tight")
        print(f"wrote {FIG_PNG}\nwrote {FIG_PDF}")
    except Exception as e:
        print("figure skipped:", e)


if __name__ == "__main__":
    main()
