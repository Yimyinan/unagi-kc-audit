# -*- coding: utf-8 -*-
"""
Three-tissue scan: does the PG-UNAGI prior compress the LATENT spectrum (Z)?

Why the Z spectrum and not the decoder spectrum
-----------------------------------------------
kappa-C measures tr(W^T W) of the Ridge map Z -> X (decoder side) and reports
0/33 "locked". The original hypothesis -- the prior constrains cells onto a
lower-dimensional manifold -- is a statement about the spectrum of Z.

The covariance spectrum of Z is invariant to any orthogonal rotation of the
latent axes, so it is directly comparable across independently trained models.
That makes it a well-posed observable where W is not (W needs axis alignment).

Design
------
Neural   2^3 factorial: C (curriculum) x M (mixture prior) x T (decoder audit)
         R0 = --- (vanilla), R1 = CMT (full), R2..R7 = the six partial arms,
         R8 = STRING prior. 21,078 cells, 7 stages, 64 dims.
Lung     R10 = vanilla, R9 = prior. 5 stages.
Hemo     R14 = vanilla, R15 = vehicle (C+M, empty TF edge list),
         R16 = WRONG prior (neural TF edges on marrow),
         R13 = true prior x 3 seeds, R11/R12/R84/R107/R111/R165 = other edge sets.
         29,985 cells, 5 stages.

Outputs -> outputs/positive_result/
    z_spectrum_all.csv      tidy: every run x iteration
    Z_SPECTRUM_3TISSUE.md   report with statistics
"""

from __future__ import annotations

import glob
import json
import re
import warnings
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

BASE = Path(r"F:/UNAGI_Project/UNAGI/result")
# The pre-migration working copy is still the canonical location of the raw
# IDREM stage files; override with UNAGI_SESSION when it has been relocated.
OUT = Path(os.environ.get("UNAGI_SESSION",
                          r"F:/UNAGI_Project")) / "outputs" / "positive_result"
OUT.mkdir(parents=True, exist_ok=True)

ITERS = 5   # compare on iterations 0..4 uniformly

# ---------------------------------------------------------------- design table
# (relative path, tissue, role, seed, C, M, T, note)
DESIGN = [
    # ---------------- neural: 2^3 factorial ----------------
    ("R0/R0_vanilla_seed123", "neural", "vanilla", 123, 0, 0, 0, "vanilla seed123"),
    ("R0/R0_vanilla_seed456", "neural", "vanilla", 456, 0, 0, 0, "vanilla seed456"),
    ("R1/R1_neural_seed123", "neural", "prior", 123, 1, 1, 1, "full PG-UNAGI seed123"),
    ("R1/R1_neural_seed456", "neural", "prior", 456, 1, 1, 1, "full PG-UNAGI seed456"),
    ("R2-R7/R2_R7_all/R2_curriculum_only_20260531_180852", "neural", "ablation", None, 1, 0, 0, "C only"),
    ("R2-R7/R2_R7_all/R3_mixed_dist_only_20260601_025237", "neural", "ablation", None, 0, 1, 0, "M only"),
    ("R2-R7/R2_R7_all/R4_tf_prior_only_20260601_113012", "neural", "ablation", None, 0, 0, 1, "T only"),
    ("R2-R7/R2_R7_all/R5_curriculum_mixed_20260601_202222", "neural", "ablation", None, 1, 1, 0, "C+M"),
    ("R2-R7/R2_R7_all/R6_curriculum_tfprior_20260602_042146", "neural", "ablation", None, 1, 0, 1, "C+T"),
    ("R2-R7/R2_R7_all/R7_mixed_tfprior_20260602_110953", "neural", "ablation", None, 0, 1, 1, "M+T"),
    ("R8/R8_string_20260613_002335", "neural", "prior", None, 1, 1, 1, "STRING prior"),
    # ---------------- lung ----------------
    ("R10", "lung", "vanilla", None, 0, 0, 0, "lung vanilla"),
    ("R9/r9_results/r9_backup", "lung", "prior", None, 1, 1, 1, "lung IPF prior"),
    # ---------------- hematopoietic ----------------
    ("R14/R14_hemato_vanilla", "hemo", "vanilla", None, 0, 0, 0, "hemo vanilla"),
    ("R15/R14_hemato_seed123", "hemo", "vehicle", 123, 1, 1, 0, "C+M, empty TF list"),
    ("R15/R14_hemato_seed42", "hemo", "vehicle", 42, 1, 1, 0, "C+M, empty TF list"),
    ("R15/R14_hemato_seed456", "hemo", "vehicle", 456, 1, 1, 0, "C+M, empty TF list"),
    ("R16/R14_wrongprior_seed42", "hemo", "wrongprior", 42, 1, 1, 1, "WRONG TF (neural on marrow)"),
    ("R13/R13_hemato_trueprior_seed123", "hemo", "prior", 123, 1, 1, 1, "true prior"),
    ("R13/R13_hemato_trueprior_seed42", "hemo", "prior", 42, 1, 1, 1, "true prior"),
    ("R13/R13_hemato_trueprior_seed456", "hemo", "prior", 456, 1, 1, 1, "true prior"),
    ("R13/R13_KO_KLF1_seed42", "hemo", "prior", 42, 1, 1, 1, "KO-KLF1 prior"),
    ("R13/R13_OE_KLF1_seed42", "hemo", "prior", 42, 1, 1, 1, "OE-KLF1 prior"),
    ("R11/R11_KLF1_TAL1_seed123", "hemo", "prior", 123, 1, 1, 1, "KLF1+TAL1"),
    ("R11/R11_KLF1_TAL1_seed42", "hemo", "prior", 42, 1, 1, 1, "KLF1+TAL1"),
    ("R11/R11_KLF1_TAL1_seed456", "hemo", "prior", 456, 1, 1, 1, "KLF1+TAL1"),
    ("R12/R12_GATA2_GFI1B_PAX5_seed123", "hemo", "prior", 123, 1, 1, 1, "GATA2+GFI1B+PAX5"),
    ("R12/R12_GATA2_GFI1B_PAX5_seed42", "hemo", "prior", 42, 1, 1, 1, "GATA2+GFI1B+PAX5"),
    ("R12/R12_GATA2_GFI1B_PAX5_seed456", "hemo", "prior", 456, 1, 1, 1, "GATA2+GFI1B+PAX5"),
    ("R84/R84_erythroid_seed123", "hemo", "prior", 123, 1, 1, 1, "erythroid"),
    ("R84/R84_erythroid_seed42", "hemo", "prior", 42, 1, 1, 1, "erythroid"),
    ("R107_erythroid_lymphoid_seed42", "hemo", "prior", 42, 1, 1, 1, "erythroid+lymphoid"),
    ("R111/R111_erythroid_temporal_seed123-1", "hemo", "prior", 123, 1, 1, 1, "erythroid temporal"),
    ("R111/R111_erythroid_temporal_seed42", "hemo", "prior", 42, 1, 1, 1, "erythroid temporal"),
    ("R111/R111_erythroid_temporal_seed456", "hemo", "prior", 456, 1, 1, 1, "erythroid temporal"),
    ("R165/R165_erythroid_myeloid_seed123", "hemo", "prior", 123, 1, 1, 1, "erythroid+myeloid (3 iters)"),
    ("R165/R165_erythroid_myeloid_seed42", "hemo", "prior", 42, 1, 1, 1, "erythroid+myeloid"),
]


def iter_dirs(root: Path) -> list[Path]:
    cands = sorted(glob.glob(str(root / "iter_*" / "stagedata")))
    if not cands:
        cands = sorted(glob.glob(str(root / "*" / "stagedata")))

    def key(p: str) -> int:
        m = re.search(r"(\d+)", Path(p).parent.name)
        return int(m.group(1)) if m else -1

    return [Path(c) for c in sorted(cands, key=key)]


def load_z(sd: Path) -> np.ndarray | None:
    """Concatenate every stage's latent matrix in one iteration."""
    import scanpy as sc

    h5 = [h for h in sorted(glob.glob(str(sd / "*.h5ad")))
          if not h.endswith("dataset.h5ad")]
    if not h5:
        return None
    blocks, stage_id = [], []
    for s, h in enumerate(h5):
        try:
            ad = sc.read_h5ad(h, backed="r")
        except Exception:
            continue
        if "z" not in ad.obsm:
            ad.file.close()
            continue
        blocks.append(np.asarray(ad.obsm["z"], dtype=np.float64))
        stage_id.append(np.full(blocks[-1].shape[0], s))
        ad.file.close()
    if not blocks:
        return None
    return np.vstack(blocks), np.concatenate(stage_id)


def spectrum(Z: np.ndarray, stage: np.ndarray) -> dict:
    """Rotation-invariant spectral summaries of the latent covariance."""
    def _moments(M: np.ndarray) -> dict:
        ev = np.clip(np.linalg.eigvalsh(M)[::-1], 0.0, None)
        tot = ev.sum()
        p = ev / tot
        pnz = p[p > 0]
        H = float(-(pnz * np.log(pnz)).sum())
        return dict(
            tr=float(tot),
            top1=float(p[0] * 100),
            top5=float(p[:5].sum() * 100),
            eff_rank=float(np.exp(H)),
            pr=float(tot ** 2 / np.sum(ev ** 2)),          # participation ratio
            entropy=float(H / np.log(len(ev))),            # normalised to [0,1]
            n90=int(np.searchsorted(np.cumsum(p), 0.90) + 1),
        )

    Zc = Z - Z.mean(axis=0)
    out_pooled = _moments(np.cov(Zc, rowvar=False))

    # within-stage covariance: removes between-stage developmental displacement
    Zw = Z.copy()
    for s in np.unique(stage):
        m = stage == s
        Zw[m] -= Zw[m].mean(axis=0)
    out_within = _moments(np.cov(Zw, rowvar=False))

    res = dict(n_cells=int(Z.shape[0]), n_stages=int(stage.max() + 1))
    for k, v in out_pooled.items():
        res[k] = v
    for k, v in out_within.items():
        res["w_" + k] = v
    return res


def main() -> None:
    rows, missing = [], []
    for rel, tissue, role, seed, C, M, T, note in DESIGN:
        root = BASE / rel
        its = iter_dirs(root)
        if not its:
            missing.append((rel, "no stagedata"))
            continue
        for it in range(min(ITERS, len(its))):
            got = load_z(its[it])
            if got is None:
                missing.append((f"{rel} iter_{it}", "no obsm['z']"))
                continue
            Z, stage = got
            rows.append(dict(run=rel.split("/")[-1], tissue=tissue, role=role,
                             seed=seed, C=C, M=M, T=T, note=note,
                             iteration=it, **spectrum(Z, stage)))
        print(f"  {tissue:6s} {rel.split('/')[-1][:44]:44s} "
              f"{min(ITERS, len(its))} iters, {rows[-1]['n_cells']} cells")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "z_spectrum_all.csv", index=False)
    print(f"\nrows: {len(df)}   missing: {len(missing)}")
    for m in missing:
        print("   MISSING:", m)

    (OUT / "z_spectrum_all.json").write_text(
        json.dumps(dict(n_rows=len(df), missing=[list(m) for m in missing]),
                   indent=1), encoding="utf-8")
    print("wrote", OUT / "z_spectrum_all.csv")


if __name__ == "__main__":
    main()
