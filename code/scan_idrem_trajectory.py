#!/usr/bin/env python3
"""
Scan UNAGI decoder_out for per-IDREM-iteration linear-decoder readouts and
compute the kappa-C spectral quantities under the paper's UNIFIED protocol.

Background
----------
`scripts/analysis/batch_decoder.py` (in F:/UNAGI_Project/UNAGI) produced
`decoder_out/<config>/iter_<k>/loading_matrix.npy`. That script uses exactly
the kappa-C unified protocol:

    Z  <- obsm['X_unagi'];  StandardScaler()               (zero mean, unit var)
    X  <- adata.X;          StandardScaler(with_mean=False) (unit var, no centring)
    W  <- Ridge(alpha=1.0, fit_intercept=True).fit(Z, X).coef_   # (G, D)
    saved matrix A = W.T                                          # (D, G)

So the kappa-C Hessian proxy is  H = W^T W = A A^T  (D x D), and

    tr(H) = sum_ij A_ij^2 = ||A||_F^2

IMPORTANT: never compute A.T @ A -- that materialises a (G x G) matrix
(11932^2 * 8 bytes = 1.1 GB). Always use A @ A.T, which is 64 x 64.

Outputs
-------
  outputs/kc_trajectory/trajectory.csv        one row per (config, iteration)
  outputs/kc_trajectory/trajectory.json       same + per-config spectra summary
  outputs/kc_trajectory/cross_validation.md   agreement with kappa_complete_audit.json
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np

DECODER_OUT = Path(r"F:/UNAGI_Project/UNAGI/decoder_out")
AUDIT_JSON = Path(
    r"F:/UNAGI_Project/next_phase/submission_packages/paper1_kc_diagnostic/"
    r"data/kappa_complete_audit.json"
)
OUTDIR = Path(r"F:/UNAGI_Project/outputs/kc_trajectory")

# --- configuration metadata -------------------------------------------------
# The 2^3 PG-UNAGI factorial ablation (mouse neural, 21,078 cells x 11,932 genes).
# C = curriculum, M = mixture prior, T = decoder-audit (linear decoder) transparency.
ABLATION = {
    "R0": dict(C=0, M=0, T=0, label="vanilla"),
    "R1": dict(C=1, M=1, T=1, label="C+M+T (full PG-UNAGI)"),
    "R2": dict(C=1, M=0, T=0, label="C only"),
    "R3": dict(C=0, M=1, T=0, label="M only"),
    "R4": dict(C=0, M=0, T=1, label="T only"),
    "R5": dict(C=1, M=1, T=0, label="C+M"),
    "R6": dict(C=1, M=0, T=1, label="C+T"),
    "R7": dict(C=0, M=1, T=1, label="M+T"),
    "R8": dict(C=None, M=None, T=None, label="string-network prior"),
}

# prior size in edges, for the hematopoietic GRN ladder
PRIOR_EDGES = {
    "R11": 21, "R84": 84, "R107": 107, "R111": 111,
    "R165": 165, "R13": 325, "R15": None, "R12": None, "R14": None, "R16": None,
}

TISSUE_BY_PREFIX = {
    "R9": "lung", "R10": "lung",
    "R0": "neural", "R1": "neural", "R2": "neural", "R3": "neural", "R4": "neural",
    "R5": "neural", "R6": "neural", "R7": "neural", "R8": "neural",
}


def parse_config(dirname: str) -> dict:
    """Split 'R13_OE_KLF1_seed42' -> (base='R13', variant='OE_KLF1', seed=42)."""
    seed = None
    m = re.search(r"_seed(\d+)$", dirname)
    if m:
        seed = int(m.group(1))
        dirname = dirname[: m.start()]
    m = re.match(r"^(R\d+)(?:_(.+))?$", dirname)
    if not m:
        return dict(base=dirname, variant=None, seed=seed)
    base, variant = m.group(1), m.group(2)
    if variant in ("KLF1_TAL1", "neuro", "lung", "hema"):
        variant = None
    # dirs without an explicit _seedNNN suffix are the seed-42 runs
    if seed is None and base in TISSUE_BY_PREFIX:
        seed = 42
    return dict(base=base, variant=variant, seed=seed)


def tissue_of(base: str) -> str:
    if base in TISSUE_BY_PREFIX:
        return TISSUE_BY_PREFIX[base]
    return "hemo"


def find_matrices(root: Path):
    """Yield (config_dir, iter_index, path) for every loading_matrix.npy."""
    out = []
    for cfg_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for it_dir in sorted(cfg_dir.glob("iter_*")):
            m = re.search(r"iter_(\d+)", it_dir.name)
            if not m:
                continue
            f = it_dir / "loading_matrix.npy"
            if f.exists():
                out.append((cfg_dir.name, int(m.group(1)), f))
        # configs that stored matrices without iter_* subdirs
        if not list(cfg_dir.glob("iter_*")):
            for f in sorted(cfg_dir.glob("**/loading_matrix.npy")):
                out.append((cfg_dir.name, None, f))
    return out


def spectrum_of(A: np.ndarray) -> dict:
    """A is (D, G). Return kappa-C spectral quantities without ever forming G x G."""
    H = A @ A.T                       # (D, D) -- cheap
    tr = float(np.trace(H))
    ev = np.linalg.eigvalsh(H)[::-1]  # descending
    ev = np.clip(ev, 0.0, None)
    share1 = float(ev[0] / tr) if tr > 0 else np.nan
    # participation ratio: effective number of latent directions carrying the trace
    pr = float(tr ** 2 / np.sum(ev ** 2)) if tr > 0 else np.nan
    # per-dimension loading share (L1 mass of each latent row)
    row_l1 = np.abs(A).sum(axis=1)
    tot = row_l1.sum()
    share = row_l1 / tot if tot > 0 else row_l1
    return dict(
        tr=tr,
        top1_share=share1,
        top5_share=float(ev[:5].sum() / tr) if tr > 0 else np.nan,
        eff_rank=pr,
        entropy=float(-np.sum((ev / tr) * np.log(ev / tr + 1e-300))) if tr > 0 else np.nan,
        dim1_pct=float(share[1] * 100),
        dim3_pct=float(share[3] * 100),
        dim55_pct=float(share[55] * 100),
        top1_dim=int(np.argmax(share)),
        n_latent=int(A.shape[0]),
        n_genes=int(A.shape[1]),
        ev=ev.tolist(),
        share=share.tolist(),
    )


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    entries = find_matrices(DECODER_OUT)
    print(f"found {len(entries)} loading matrices under {DECODER_OUT}")

    rows = []
    t0 = time.time()
    for i, (cfg, it, path) in enumerate(entries):
        A = np.load(path)
        if A.shape[0] > A.shape[1]:          # safety: ensure (D, G)
            A = A.T
        s = spectrum_of(A)
        meta = parse_config(cfg)
        base = meta["base"]
        ab = ABLATION.get(base, {})
        # optional per-gene R2 summary emitted by the same pipeline
        summ = {}
        for cand in sorted(path.parent.glob("*summary.json")):
            try:
                summ = json.loads(cand.read_text(encoding="utf-8"))
            except Exception:
                pass
        rows.append(dict(
            config=cfg, base=base, variant=meta["variant"], seed=meta["seed"],
            tissue=tissue_of(base), iteration=it,
            C=ab.get("C"), M=ab.get("M"), T=ab.get("T"), ab_label=ab.get("label"),
            prior_edges=PRIOR_EDGES.get(base),
            median_r2=summ.get("median_r2"), mean_r2=summ.get("mean_r2"),
            r2_gt_0p5=summ.get("r2_gt_0.5", summ.get("genes_r2_gt_05")),
            n_cells=summ.get("n_cells"), n_genes_sum=summ.get("n_genes"),
            path=str(path.relative_to(DECODER_OUT)),
            **{k: v for k, v in s.items() if k not in ("ev", "share")},
        ))
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(entries)}  ({time.time()-t0:.1f}s)")

    rows.sort(key=lambda r: (r["tissue"], r["base"], r["seed"] or 0,
                             r["variant"] or "", r["iteration"] if r["iteration"] is not None else -1))

    # ---------- persist ----------
    import csv
    cols = [c for c in rows[0].keys() if c != "path"]
    with open(OUTDIR / "trajectory.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols + ["path"])
        w.writeheader()
        w.writerows(rows)
    with open(OUTDIR / "trajectory.json", "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=1)

    print(f"\nwrote {len(rows)} rows -> {OUTDIR/'trajectory.csv'}")

    # ---------- cross-validation against the published audit ----------
    audit = json.loads(AUDIT_JSON.read_text(encoding="utf-8"))
    amap = {a["config"]: a for a in audit}

    def audit_key(r):
        b, s, v = r["base"], r["seed"], r["variant"]
        if r["tissue"] == "lung":
            return b
        if v:
            return f"{b}_{v}"
        return f"{b}_s{s}" if s else b

    lines = ["# Cross-validation: decoder_out trajectory vs. kappa_complete_audit.json",
             "",
             "Terminal iteration of each config, recomputed from `loading_matrix.npy`",
             "(`tr = ||A||_F^2`, A is the (D,G) Ridge readout saved by batch_decoder.py),",
             "compared against the `tr_P` column of the published audit.",
             "",
             "| audit config | tissue | iter | tr (decoder_out) | tr_P (audit) | ratio | kappa (audit) |",
             "|---|---|---|---|---|---|---|"]
    matched = 0
    ratios = []
    for r in rows:
        k = audit_key(r)
        a = amap.get(k)
        if not a:
            continue
        # keep only the highest iteration available for this config
        sibs = [x for x in rows if audit_key(x) == k]
        if not sibs:
            continue
        last = max(sibs, key=lambda x: (x["iteration"] if x["iteration"] is not None else -1))
        if r is not last:
            continue
        tr, trP = last["tr"], a["tr_P"]
        ratio = tr / trP if trP else np.nan
        ratios.append(ratio)
        matched += 1
        lines.append(f"| {k} | {a['tissue']} | {last['iteration']} | {tr:,.1f} | "
                     f"{trP:,.1f} | {ratio:.3f} | {a['kappa_ratio']:.3f} |")
    lines.append("")
    if ratios:
        rr = np.array(ratios)
        lines.append(f"**Matched {matched} configs.** "
                     f"ratio = tr(decoder_out)/tr_P(audit): "
                     f"median {np.median(rr):.3f}, mean {rr.mean():.3f}, "
                     f"range {rr.min():.3f}--{rr.max():.3f}.")
        lines.append("")
        lines.append("A median ratio of ~1.0 means the iteration matrices reproduce the "
                     "published terminal-state trace, i.e. the trajectory analysis is "
                     "on the same footing as the 0/33 audit.")
    (OUTDIR / "cross_validation.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[-6:]))


if __name__ == "__main__":
    sys.exit(main())
