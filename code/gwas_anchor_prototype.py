#!/usr/bin/env python3
"""gwas_anchor_prototype.py — feasibility prototype.

Question: is the latent axis a GRN prior organises ENRICHED for human-genetic
evidence of erythroid traits? Ground truth = GWAS (not perturbation prediction,
which failed in Fig. S22).

Gene sets from the GWAS Catalog (mapped genes):
  ERY : erythroid traits (RBC count, MCV, MCH, RDW, haematocrit, haemoglobin)
  MYE : non-erythroid BLOOD traits (platelet, WBC, eosinophil, monocyte,
        lymphocyte, neutrophil) -- matched for tissue AND ascertainment, so a
        generic "blood gene" effect cannot produce an ERY-specific result.

Two statistics are reported, and the difference between them is the point:
  (1) auc        raw  P(|g|_set > |g|_background)      -- CONFOUNDED
  (2) matched    mean within-expression-decile rank    -- the usable one
For a latent direction v the induced gene profile is g = W^T v, with W the
audit's Ridge readout (64 x G). Null = 200 random unit directions in R^64.

Run configs follow code/discover_stagedata.py (latent in obsm['z']);
the vanilla baseline is R15/R14_hemato_seed{42,123,456}.
"""
import csv
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULT = "F:/UNAGI_Project/UNAGI/result"
LOADING = "F:/UNAGI_Project/UNAGI/decoder_unified/loading"
# GWAS Catalog flat download (~58 MB). Kept OUT of the paper repo; re-fetch with
#   curl -sL "https://www.ebi.ac.uk/gwas/api/search/downloads?q=blood+cell+trait&facet=association&efo=true" -o <path>
# or point GWAS_TSV at an existing copy.
GWAS_TSV = os.environ.get(
    "GWAS_TSV",
    "F:/UNAGI_Project/_scratch_gwas/gwas_catalog_blood_traits.tsv")
OUT = os.path.join(ROOT, "data", "gwas_anchor_prototype.json")

ERY_TRAITS = {"Red blood cell count", "Mean corpuscular volume",
              "Mean corpuscular hemoglobin", "Red cell distribution width",
              "Hematocrit", "Hemoglobin levels"}
MYE_TRAITS = {"Platelet count", "White blood cell count", "Eosinophil count",
              "Monocyte count", "Lymphocyte count", "Neutrophil count"}

CONFIGS = {
    "R13_s42":     ("R13/R13_hemato_trueprior_seed42", "prior"),
    "R13_s123":    ("R13/R13_hemato_trueprior_seed123", "prior"),
    "R13_s456":    ("R13/R13_hemato_trueprior_seed456", "prior"),
    "R14_s42":     ("R15/R14_hemato_seed42", "vanilla"),
    "R14_s123":    ("R15/R14_hemato_seed123", "vanilla"),
    "R14_s456":    ("R15/R14_hemato_seed456", "vanilla"),
    "R13_KO_KLF1": ("R13/R13_KO_KLF1_seed42", "klf1_ko"),
    "R13_OE_KLF1": ("R13/R13_OE_KLF1_seed42", "klf1_oe"),
}


def load_gwas_sets(space):
    ery, mye = set(), set()
    with open(GWAS_TSV, encoding="utf-8", errors="replace") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            trait = row.get("DISEASE/TRAIT", "") or ""
            tgt = ery if trait in ERY_TRAITS else (mye if trait in MYE_TRAITS else None)
            if tgt is None:
                continue
            for g in (row.get("MAPPED_GENE") or "").split(";"):
                g = g.strip().upper()
                if g:
                    tgt.add(g)
    return ery & space, mye & space


def load_run(run, it=4):
    import anndata as ad
    files = sorted(glob.glob(os.path.join(RESULT, run, "iter_%d" % it, "stagedata", "[0-9].h5ad")))
    if not files:
        return None, None, None
    Zs, Xs, genes = [], [], None
    for f in files:
        a = ad.read_h5ad(f)
        Zs.append(np.asarray(a.obsm["z"], dtype=float))
        X = a.X
        X = np.asarray(X.todense() if hasattr(X, "todense") else X, dtype=np.float32)
        Xs.append(X)
        if genes is None:
            genes = np.array([g.upper() for g in a.var_names])
    return np.vstack(Zs), np.vstack(Xs), genes


def auc(vals, mask):
    r = np.argsort(np.argsort(vals)).astype(float)
    r /= max(len(r) - 1, 1)
    a, b = r[mask], r[~mask]
    if a.size == 0 or b.size == 0:
        return float("nan")
    n1, n2 = a.size, b.size
    u = a.sum() * n2 - n1 * (n2 - 1) / 2.0
    return float(u / (n1 * n2))


def matched_score(prof, mean_expr, mask, n_bins=10):
    """Mean WITHIN-expression-decile percentile rank of |profile| for `mask`.

    A VAE latent axis's raw |loading| ranking is dominated by highly expressed /
    high-variance genes (this paper's own dominant axis loads on housekeeping and
    ribosomal features), while GWAS-implicated genes are typically lowly
    expressed and regulatory. Comparing them without matching on expression
    therefore measures gene class rather than biological relevance -- exactly
    what the unmatched statistic showed (AUC ~ 0.10 for every configuration,
    prior and vanilla alike). Ranking within expression deciles removes that
    confound; null expectation is 0.5.
    """
    edges = np.quantile(mean_expr, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    pct = np.full(prof.shape, np.nan)
    for i in range(n_bins):
        sel = (mean_expr >= edges[i]) & (mean_expr < edges[i + 1])
        if sel.sum() < 20:
            continue
        r = np.argsort(np.argsort(prof[sel])).astype(float)
        pct[sel] = r / max(int(sel.sum()) - 1, 1)
    ok = np.isfinite(pct)
    m = ok & mask
    return float(np.nanmean(pct[m])) if m.any() else float("nan")


def main():
    rng = np.random.default_rng(20260915)
    Zref, Xref, genes = load_run(CONFIGS["R13_s42"][0])
    if genes is None:
        raise SystemExit("could not load stagedata for the reference config")
    mean_expr = np.asarray(Xref.mean(0)).ravel()
    space = set(genes.tolist())
    ery, mye = load_gwas_sets(space)
    ery_mask, mye_mask = np.isin(genes, list(ery)), np.isin(genes, list(mye))
    print("gene space %d | ERY in space %d | MYE in space %d | background %d"
          % (len(space), len(ery), len(mye), int((~(ery_mask | mye_mask)).sum())))
    print("confound check: corr(|W| , mean expression) will be reported per config\n")

    out = {"gene_space": len(space), "n_ery": len(ery), "n_mye": len(mye),
           "ery_traits": sorted(ERY_TRAITS), "mye_traits": sorted(MYE_TRAITS),
           "note": ("statistic 'auc' is confounded by expression level; "
                    "'matched_*' ranks within expression deciles (null = 0.5)"),
           "configs": {}}

    for cfg, (run, role) in CONFIGS.items():
        lf = os.path.join(LOADING, cfg + ".npy")
        if not os.path.exists(lf):
            print("  [skip] no loading:", cfg); continue
        W = np.load(lf).astype(float)
        Z, X, g2 = load_run(run)
        if Z is None:
            print("  [skip] no stagedata:", cfg); continue
        if Z.shape[1] != W.shape[0] or len(g2) != W.shape[1]:
            print("  [skip] shape mismatch", Z.shape, W.shape, len(g2)); continue
        Zc = Z - Z.mean(0)
        C = Zc.T @ Zc / max(len(Zc) - 1, 1)
        ev, V = np.linalg.eigh(C)
        v1 = V[:, -1]
        prof = np.abs(W.T @ v1)

        a_ery, a_mye = auc(prof, ery_mask), auc(prof, mye_mask)
        m_ery = matched_score(prof, mean_expr, ery_mask)
        m_mye = matched_score(prof, mean_expr, mye_mask)

        ne, nm, ke, km = [], [], [], []
        for _ in range(200):
            u = rng.normal(size=W.shape[0]); u /= np.linalg.norm(u)
            p = np.abs(W.T @ u)
            ne.append(auc(p, ery_mask)); nm.append(auc(p, mye_mask))
            ke.append(matched_score(p, mean_expr, ery_mask))
            km.append(matched_score(p, mean_expr, mye_mask))
        ne, nm, ke, km = map(np.array, (ne, nm, ke, km))
        r_conf = float(np.corrcoef(prof, mean_expr)[0, 1])

        out["configs"][cfg] = {
            "role": role, "run": run, "n_cells": int(Z.shape[0]),
            "corr_profile_vs_expr": r_conf,
            "auc_ery": a_ery, "auc_mye": a_mye,
            "p_auc_ery": float((ne >= a_ery).mean()),
            "matched_ery": m_ery, "matched_mye": m_mye,
            "p_matched_ery": float((ke >= m_ery).mean()),
            "p_matched_mye": float((km >= m_mye).mean()),
            "null_matched_ery_mean": float(ke.mean()), "null_matched_ery_sd": float(ke.std()),
            "matched_ery_minus_mye": m_ery - m_mye,
        }
        print("  %-13s %-8s n=%6d  r(|W|,expr)=%+.2f | auc_ery=%.3f | MATCHED ery=%.3f (p=%.3f)"
              " mye=%.3f (p=%.3f)  diff=%+.3f"
              % (cfg, role, Z.shape[0], r_conf, a_ery, m_ery,
                 float((ke >= m_ery).mean()), m_mye, float((km >= m_mye).mean()),
                 m_ery - m_mye))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print("\n[saved]", OUT)


if __name__ == "__main__":
    main()
