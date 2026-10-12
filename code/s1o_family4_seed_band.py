# -*- coding: utf-8 -*-
"""S1o -- seed band for the family-4 MLP VAE (section 2.9.2's third architecture).

Why
---
Section 2.9.2 reports the family-4 pair as a single run: ``vae_family4_results.json``
contains a beta sweep (2 betas, only beta=0.1 clearing the latent-health gate), and
the reported numbers (kappa-C(W)=0.942 corrected, kappa-C(Z)=0.89, effective rank
21.66 -> 21.04) are one draw of that one configuration.  The scVI/pbmc3k control
showed what a single draw is worth, so this script repeats the family-4 training
over several seeds and reports the paired effect against a cross-seed null.

Seeding follows the published script exactly: ``vae_family4_decoupling.py`` seeds
once at import (SEED=42) and then trains the vanilla and the conditioned model in
sequence on one continuing random stream.  Re-seeding between the two arms changes
the conditioned arm's initialisation and breaks the seed-42 replication check -- the
first version of this script made exactly that mistake (seed 42 then gave
effective rank 22.04 -> 17.15, -22.2%, against the published 21.66 -> 21.04, -2.8%).

Reuses the published training code verbatim by importing
``vae_family4_decoupling`` (MLP VAE, perturbation identity concatenated to the
decoder input only; X = the cached Norman matrix, per-gene z-scored; 150 epochs).

Three readouts are reported per seed, because they are not the same protocol:
  kappa_trace         tr(W1^T W1)/tr(W0^T W0) with the paper's ridge readout
                      (per-column standardisation, alpha=1, no intercept)
  kappa_corrected     sum_g R2_g ratio, same readout (the manuscript's corrected form)
  kappa_r2_internal   the ratio of ``Ridge(alpha=1).score()`` values that the
                      published family-4 script itself records -- unstandardised
                      inputs, an intercept, and score() returns the per-gene MEAN

Robustness: every finished seed is appended to a ``.jsonl`` sidecar and skipped on
re-runs, so a kill never discards completed work; aggregates are only formed once
two seeds exist (the first version crashed on ``describe([])`` before writing).

Writes : outputs/positive_result/s1o_family4_seed_band.json  (+ .jsonl sidecar)
Usage  : python code/s1o_family4_seed_band.py --seeds 5
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")


def log(*a):
    print(*a, flush=True)


HERE = os.path.dirname(os.path.abspath(__file__))
PAPER1 = os.path.abspath(os.path.join(HERE, ".."))
ROOT = os.path.abspath(os.path.join(PAPER1, "..", ".."))
OUT = os.path.join(ROOT, "outputs", "positive_result")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(PAPER1, "kappa_c_audit"))

os.environ.setdefault("VAE4_EPOCHS", "150")
os.environ.setdefault("VAE4_BETAS", "0.1")

import torch                                                   # noqa: E402
import vae_family4_decoupling as f4                            # noqa: E402
from kappa_c_audit.core import predictability_total            # noqa: E402


def describe(v):
    v = np.asarray(v, float)
    if v.size == 0:
        return dict(n=0, mean=float("nan"), sd=float("nan"), lo2s=float("nan"),
                    hi2s=float("nan"), min=float("nan"), max=float("nan"))
    sd = float(v.std(ddof=1)) if v.size > 1 else float("nan")
    return dict(n=int(v.size), mean=float(v.mean()), sd=sd,
                lo2s=float(v.mean() - 2 * sd), hi2s=float(v.mean() + 2 * sd),
                min=float(v.min()), max=float(v.max()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--seed-start", type=int, default=42)
    ap.add_argument("--beta", type=float, default=0.1,
                    help="the beta the published run selected as usable")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--out", default="s1o_family4_seed_band.json")
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    t0 = time.time()

    jsonl = os.path.join(OUT, a.out.replace(".json", ".jsonl"))
    os.makedirs(OUT, exist_ok=True)
    rows = []
    if os.path.exists(jsonl):
        with open(jsonl, encoding="utf-8") as f:
            rows = [json.loads(l) for l in f if l.strip()]
        log(f"[resume] {len(rows)} seed rows loaded from {os.path.basename(jsonl)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = getattr(f4.pb, "_cache", None) or os.path.join(PAPER1, "data",
                                                           "_norman_latents.npz")
    d = np.load(cache, allow_pickle=True)
    X = d["X"].astype(np.float32)
    labels = np.array([str(x) for x in d["pert_labels"]])
    if f4.STANDARDIZE:
        mu, sd = X.mean(0), X.std(0)
        sd[sd < 1e-8] = 1.0
        X = ((X - mu) / sd).astype(np.float32)
    uniq = np.unique(labels)
    cond_idx = np.array([{c: i for i, c in enumerate(uniq)}[l] for l in labels],
                        dtype=np.int64)
    log(f"[data] X{X.shape}  conditions={len(uniq)}  device={device}  "
        f"beta={a.beta}  epochs={f4.EPOCHS}  threads={torch.get_num_threads()}")

    out = None
    for s in range(a.seed_start, a.seed_start + a.seeds):
        if any(r["seed"] == s for r in rows):
            log(f"seed {s} already present, skipping")
            continue
        ts = time.time()
        # seed ONCE per run, exactly as the published script does, so that the two
        # arms share one continuing random stream
        torch.manual_seed(s)
        np.random.seed(s)
        mv = f4.train_model(X, cond_idx, 0, device, f"van_s{s}", a.beta)
        Zv = f4.get_latent(mv, X, device)
        mp = f4.train_model(X, cond_idx, len(uniq), device, f"cond_s{s}", a.beta)
        Zp = f4.get_latent(mp, X, device)
        rec = f4.evaluate(Zv, Zp, X)
        tr_v, tr_p = float(predictability_total(Zv, X)), float(predictability_total(Zp, X))
        row = dict(seed=s, kappa_trace=rec["kappa_C_W"],
                   kappa_corrected=tr_p / tr_v, sum_r2_vanilla=tr_v,
                   sum_r2_conditioned=tr_p,
                   kappa_r2_internal=rec["ridge_r2_conditioned"] / rec["ridge_r2_vanilla"],
                   kappa_C_Z=rec["kappa_C_Z"],
                   trace_Z_vanilla=float(np.trace(np.cov(Zv, rowvar=False))),
                   eff_rank_vanilla=rec["eff_rank_vanilla"],
                   eff_rank_conditioned=rec["eff_rank_conditioned"],
                   eff_rank_ratio=rec["eff_rank_conditioned"] / rec["eff_rank_vanilla"],
                   eff_rank_delta_pct=rec["eff_rank_change_percent"],
                   latent_usable=bool(rec["latent_usable"]),
                   seconds=(time.time() - ts))
        rows.append(row)
        with open(jsonl, "a", encoding="utf-8") as f:          # never lose a seed
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        log(f"seed {s}: k_tr={row['kappa_trace']:.4f} k_r2={row['kappa_corrected']:.4f} "
            f"(internal {row['kappa_r2_internal']:.4f}) k_Z={row['kappa_C_Z']:.4f} "
            f"eff_rank {row['eff_rank_vanilla']:.2f}->{row['eff_rank_conditioned']:.2f} "
            f"({row['eff_rank_delta_pct']:+.1f}%) usable={row['latent_usable']} "
            f"[{row['seconds']:.0f}s]")

        rows.sort(key=lambda r: r["seed"])
        null = [dict(kappa_trace=ri["kappa_trace"] / rj["kappa_trace"],
                     kappa_corrected=ri["sum_r2_vanilla"] / rj["sum_r2_vanilla"],
                     kappa_C_Z=ri["trace_Z_vanilla"] / rj["trace_Z_vanilla"],
                     eff_rank_ratio=ri["eff_rank_vanilla"] / rj["eff_rank_vanilla"])
                for ri, rj in itertools.combinations(rows, 2)]
        out = dict(architecture="family-4 MLP VAE (imported from "
                                "code/vae_family4_decoupling.py)",
                   beta=a.beta, epochs=f4.EPOCHS, warmup=f4.WARMUP,
                   n_cells=int(X.shape[0]), n_genes=int(X.shape[1]),
                   n_conditions=int(len(uniq)), standardised=bool(f4.STANDARDIZE),
                   seed_policy=("seeded once per run, both arms on the continuing "
                                "stream -- as the published script does"),
                   note="the published run is seed 42",
                   partial=len(rows) < 2,
                   seeds=[r["seed"] for r in rows], per_seed=rows,
                   paired_conditioned_vs_vanilla={
                       k: describe([r[k] for r in rows])
                       for k in ("kappa_trace", "kappa_corrected", "kappa_r2_internal",
                                 "kappa_C_Z", "eff_rank_ratio")},
                   cross_seed_vanilla_null={
                       k: describe([x[k] for x in null])
                       for k in ("kappa_trace", "kappa_corrected", "kappa_C_Z",
                                 "eff_rank_ratio")},
                   published=dict(kappa_trace=0.9473690390586853,
                                  kappa_corrected=0.9419,
                                  kappa_C_Z=0.8902077374844006,
                                  eff_rank_vanilla=21.655498280838927,
                                  eff_rank_conditioned=21.04238213427831))
        with open(os.path.join(OUT, a.out), "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1, ensure_ascii=False)

    log(f"[done] {len(rows)} seeds in {(time.time() - t0) / 60:.1f} min")
    if out:
        log("paired (conditioned vs vanilla):")
        for k, v in out["paired_conditioned_vs_vanilla"].items():
            log(f"  {k:<20}{v['mean']:>9.4f}{v['sd']:>9.4f}  [{v['lo2s']:.3f}, {v['hi2s']:.3f}]")
        log("cross-seed null (two baseline fits):")
        for k, v in out["cross_seed_vanilla_null"].items():
            log(f"  {k:<20}{v['mean']:>9.4f}{v['sd']:>9.4f}  [{v['lo2s']:.3f}, {v['hi2s']:.3f}]")


if __name__ == "__main__":
    main()
