# -*- coding: utf-8 -*-
"""Discover every run under UNAGI/result that carries stagedata/*.h5ad with obsm['z'].

Prints a table so the three-tissue matched design can be assembled deliberately,
not by guessing directory names.
"""

from __future__ import annotations

import glob
import os
import re
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

BASE = Path(r"F:/UNAGI_Project/UNAGI/result")


def iter_dirs(root: Path) -> list[Path]:
    """Return stagedata dirs in ascending iteration order."""
    cands = sorted(glob.glob(str(root / "iter_*" / "stagedata")))
    if not cands:
        cands = sorted(glob.glob(str(root / "*" / "stagedata")))
    out = []

    def key(p: str) -> int:
        stem = Path(p).parent.name          # 'iter_3' or '3'
        m = re.search(r"(\d+)", stem)
        return int(m.group(1)) if m else -1

    for c in sorted(cands, key=key):
        out.append(Path(c))
    return out


def probe(rel: str) -> dict | None:
    root = BASE / rel
    its = iter_dirs(root)
    if not its:
        return None
    import scanpy as sc

    h5 = sorted(glob.glob(str(its[0] / "*.h5ad")))
    h5 = [h for h in h5 if not h.endswith("dataset.h5ad")]
    if not h5:
        return None
    ad = sc.read_h5ad(h5[0], backed="r")
    keys = list(ad.obsm.keys())
    if "z" not in keys:
        ad.file.close()
        return None
    Z = np.asarray(ad.obsm["z"], dtype=np.float64)
    n_stage = len(h5)
    ad.file.close()
    return dict(rel=rel, n_iter=len(its), n_stage=n_stage,
                cells=int(Z.shape[0]), dims=int(Z.shape[1]), obsm=keys)


def walk() -> list[str]:
    """Every directory that directly contains iter_N/stagedata (any nesting)."""
    roots = set()
    for p in glob.glob(str(BASE / "**" / "stagedata"), recursive=True):
        roots.add(Path(p).parent.parent)   # stagedata -> iter_N -> run root
    return sorted(str(r.relative_to(BASE)).replace("\\", "/") for r in roots)


def main() -> None:
    rels = walk()
    rows = []
    for rel in rels:
        r = probe(rel)
        if r:
            rows.append(r)
    print(f"{'run':52s} {'iters':>5s} {'stages':>6s} {'cells':>8s} {'dims':>5s}  obsm")
    print("-" * 100)
    for r in rows:
        print(f"{r['rel']:52s} {r['n_iter']:5d} {r['n_stage']:6d} "
              f"{r['cells']:8d} {r['dims']:5d}  {r['obsm']}")
    print(f"\ntotal usable runs: {len(rows)} / {len(rels)} discovered roots")


if __name__ == "__main__":
    main()
