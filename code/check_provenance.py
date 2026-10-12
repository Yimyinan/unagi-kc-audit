#!/usr/bin/env python3
"""
Provenance gate: decide WHICH per-iteration trajectories in `decoder_out` may be
plotted as the trajectory leading to the audited terminal state.

Question
--------
`decoder_unified/loading/<cfg>.npy` is the terminal (iter_4) readout that
`kappa_complete_audit.json` was built from.  `decoder_out/<cfg>/iter_*/loading_matrix.npy`
are per-iteration readouts.  Are they the *same* pipeline run?

If decoder_out@iter_4 == decoder_unified (same trace AND row-wise cosine ~ 1),
then iter_0..iter_3 in that directory are a legitimate trajectory ending at the
audited point.  If not, the directory is a different run and its trajectory must
NOT be presented as leading to the published number.

Note: the in-repo script `xcheck_unified.py` attempted this test but its second
block silently no-op'd -- it looks for `decoder_out/R84_s42/iter_4/...` while the
real directory is `decoder_out/R84_seed42/iter_4/...`.  Nothing was ever printed.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

BASE = Path(r"F:/UNAGI_Project/UNAGI")
DEC = BASE / "decoder_out"
UNI = BASE / "decoder_unified" / "loading"
AUDIT = Path(
    r"F:/UNAGI_Project/next_phase/submission_packages/paper1_kc_diagnostic/"
    r"data/kappa_complete_audit.json"
)
OUTDIR = Path(r"F:/UNAGI_Project/outputs/kc_trajectory")

# decoder_out directory name -> decoder_unified file stem
NAME_MAP = {
    "R0_seed123": "R0_s123", "R0_seed456": "R0_s456",
    "R1_seed123": "R1_s123", "R1_seed456": "R1_s456",
    "R11_seed42": "R11_s42",
    "R11_KLF1_TAL1_seed123": "R11_s123",
    "R11_KLF1_TAL1_seed456": "R11_s456",
    "R12_seed42": "R12_s42", "R12_seed123": "R12_s123", "R12_seed456": "R12_s456",
    "R13_seed123": "R13_s123", "R13_seed456": "R13_s456",
    "R14_seed42": "R14_seed42", "R14_seed123": "R14_seed123", "R14_seed456": "R14_seed456",
    "R15_seed42": "R15_s42", "R15_seed123": "R15_s123", "R15_seed456": "R15_s456",
    "R84_seed42": "R84_s42", "R84_seed123": "R84_s123",
    "R107_seed42": "R107_s42",
    "R111_seed42": "R111_s42", "R111_seed123": "R111_s123",
    "R111_seed456": "R111_s456",
    "R165_seed42": "R165_s42",
}


def rowcos(a: np.ndarray, b: np.ndarray) -> float:
    """Mean row-wise cosine between two (D, G) matrices."""
    num = np.sum(a * b, axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12
    return float(np.mean(num / den))


def trace(A: np.ndarray) -> float:
    return float(np.einsum("ij,ij->", A, A))   # = tr(A A^T), never forms G x G


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    audit = {a["config"]: a for a in json.loads(AUDIT.read_text(encoding="utf-8"))}

    rows = []
    for dec_name, uni_name in NAME_MAP.items():
        dec_dir = DEC / dec_name
        uni_file = UNI / f"{uni_name}.npy"
        if not uni_file.exists():
            continue
        # highest iteration present in decoder_out
        it_dirs = sorted(dec_dir.glob("iter_*"),
                         key=lambda p: int(re.search(r"iter_(\d+)", p.name).group(1)))
        if not it_dirs:
            rows.append(dict(dec=dec_name, uni=uni_name, status="no iterations in decoder_out"))
            continue
        last = it_dirs[-1]
        it_last = int(re.search(r"iter_(\d+)", last.name).group(1))
        A = np.load(last / "loading_matrix.npy")
        U = np.load(uni_file)
        if A.shape != U.shape:
            rows.append(dict(dec=dec_name, uni=uni_name, status=f"shape mismatch {A.shape} vs {U.shape}"))
            continue
        rows.append(dict(
            dec=dec_name, uni=uni_name, iter=it_last,
            n_iters=len(it_dirs),
            status="ok" if it_last == 4 else f"terminal iter is {it_last}, not 4",
            tr_dec=trace(A), tr_uni=trace(U),
            tr_ratio=trace(A) / trace(U) if trace(U) else np.nan,
            rowcos=rowcos(A, U),
            fro_rel=float(np.linalg.norm(A - U) / np.linalg.norm(U)),
        ))

    print(f"{'decoder_out':26s} {'unified':14s} it  n  {'tr(dec)':>10s} {'tr(uni)':>10s} "
          f"{'ratio':>7s} {'rowcos':>7s} {'||dU||/||U||':>11s}")
    print("-" * 100)
    for r in rows:
        if "tr_dec" not in r:
            print(f"{r['dec']:26s} {r['uni']:14s} --  {r['status']}")
            continue
        print(f"{r['dec']:26s} {r['uni']:14s} {r['iter']:2d} {r['n_iters']:2d} "
              f"{r['tr_dec']:10,.1f} {r['tr_uni']:10,.1f} {r['tr_ratio']:7.3f} "
              f"{r['rowcos']:7.3f} {r['fro_rel']:11.4f}")

    verdict = []
    for r in rows:
        if "tr_dec" not in r:
            continue
        same = (abs(r["tr_ratio"] - 1.0) < 0.02) and (r["rowcos"] > 0.98)
        r["verdict"] = "PASS" if same else "FAIL"
        verdict.append(r)

    npass = sum(1 for r in verdict if r["verdict"] == "PASS")
    print(f"\nPASS {npass} / {len(verdict)}   "
          f"(PASS = terminal iteration reproduces the audited unified matrix: "
          f"|tr ratio - 1| < 0.02 AND row-wise cosine > 0.98)")

    (OUTDIR / "provenance.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")

    lines = ["# Provenance gate: decoder_out trajectories vs. audited terminal state",
             "",
             "Test: does `decoder_out/<cfg>/iter_<max>/loading_matrix.npy` reproduce",
             "`decoder_unified/loading/<cfg>.npy` (the matrix the 33-config audit used)?",
             "",
             "PASS criterion: `|tr ratio - 1| < 0.02` **and** mean row-wise cosine > 0.98.",
             "",
             "| decoder_out | unified | iter | n_iters | tr(decoder_out) | tr(unified) | ratio | rowcos | verdict |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if "tr_dec" not in r:
            lines.append(f"| {r['dec']} | {r['uni']} | - | - | - | - | - | - | {r['status']} |")
        else:
            lines.append(f"| {r['dec']} | {r['uni']} | {r['iter']} | {r['n_iters']} | "
                         f"{r['tr_dec']:,.1f} | {r['tr_uni']:,.1f} | {r['tr_ratio']:.3f} | "
                         f"{r['rowcos']:.3f} | **{r['verdict']}** |")
    lines += ["", f"**{npass} / {len(verdict)} PASS.**", "",
              "Only PASS configs may be shown as a trajectory ending at the published "
              "number. FAIL configs are a different run (or a different pipeline "
              "generation) and their trajectories are not comparable to the 0/33 audit."]
    (OUTDIR / "provenance.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwrote {OUTDIR/'provenance.md'}")


if __name__ == "__main__":
    main()
