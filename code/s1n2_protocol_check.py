# -*- coding: utf-8 -*-
"""S1n2 -- protocol control for the Norman null: one vanilla arm, single 200-epoch call.

Why
---
The Norman null in s1n_norman_null.json trains baselines in four 50-epoch chunks, so
that a late-training divergence still leaves a queryable model (scvi refuses to serve
a latent whose trainer never completed a fit).  Seed 42's arm, however, came from an
earlier revision that used one single 200-epoch call, and it is the lowest of the
three (effective rank 6.899 against 10.624 and 11.705).  Before treating that spread
as seed scatter, this control trains the SAME seed with a single call, so the
protocol difference is measured rather than assumed.

Usage: python code/s1n2_protocol_check.py --seed 43        # single 200-epoch call
Writes: outputs/positive_result/s1n2_protocol_check.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER1 = os.path.abspath(os.path.join(HERE, ".."))
ROOT = os.path.abspath(os.path.join(PAPER1, "..", ".."))
OUT = os.path.join(ROOT, "outputs", "positive_result")
sys.path.insert(0, HERE)

import s1n_norman_null as S                                   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--out", default="s1n2_protocol_check.json")
    a = ap.parse_args()

    import scvi
    import torch
    adata, X, pert_col, control = S.load_norman()

    ts = time.time()
    scvi.settings.seed = a.seed
    torch.manual_seed(a.seed)
    copy = adata.copy()
    scvi.model.SCVI.setup_anndata(copy)
    m = scvi.model.SCVI(copy, n_latent=S.N_LATENT, n_layers=S.N_LAYERS,
                        dropout_rate=S.DROPOUT, gene_likelihood=S.GENE_LIKELIHOOD)
    status = "completed"
    try:
        m.train(max_epochs=a.epochs, accelerator="cpu")          # one single call
    except Exception as e:                                       # noqa: BLE001
        status = f"diverged: {type(e).__name__}: {str(e)[:160]}"
    try:
        Z = m.get_latent_representation(copy).astype(np.float64)
    except Exception as e:                                       # noqa: BLE001
        print(f"latent unavailable after a single call: {e}")
        Z = None
    rec = dict(seed=a.seed, protocol="single_200_epoch_call", status=status,
               seconds=time.time() - ts)
    if Z is not None:
        rec.update(S.metrics(Z, X))
    with open(os.path.join(OUT, a.out), "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=1, ensure_ascii=False)

    # compare with the chunked protocol at the same seed, if recorded
    jsonl = os.path.join(OUT, "s1n_norman_null.jsonl")
    chunked = None
    if os.path.exists(jsonl):
        with open(jsonl, encoding="utf-8") as f:
            for l in f:
                if l.strip():
                    r = json.loads(l)
                    if r["seed"] == a.seed and r["arm"] == "vanilla":
                        chunked = r
    print(f"[single-call] seed {a.seed}: {json.dumps({k: rec.get(k) for k in ('sum_r2','trace','eff_rank','trace_Z')}, default=str)}")
    if chunked:
        print(f"[chunked    ] seed {a.seed}: "
              f"{json.dumps({k: chunked.get(k) for k in ('sum_r2','trace','eff_rank','trace_Z')}, default=str)}")
        for k in ("sum_r2", "trace", "eff_rank", "trace_Z"):
            if k in rec and k in chunked:
                rel = abs(rec[k] - chunked[k]) / abs(chunked[k])
                print(f"   {k:<10} relative protocol difference {rel * 100:6.2f}%")
    print(f"wrote {os.path.join(OUT, a.out)}")


if __name__ == "__main__":
    main()
