# -*- coding: utf-8 -*-
"""Norman injected-arm seed band: the missing half of the paired design.

Why this exists
---------------
The published Norman pair is a single run of each arm. Judging it needed a
family-matched **null**, which only requires baseline fits, so the earlier work
(`code/s1n_norman_null.py`) fit baselines and stopped there. That answers "is the
point estimate bigger than the seed scatter of a baseline fit", but not "how much
does the *effect* vary between seeds".

This script fits the injected arm (`categorical_covariate_keys=[pert_col]`) at seeds
that already have a protocol-matched baseline fit, so the paired ratio
`metric_injected(s) / metric_vanilla(s)` can be formed per seed. That removes the
baseline scatter entirely and gives the effect its own standard error.

Protocol
--------
Single 200-epoch call, exactly as the published run, CPU, seed set per arm. Chunked
training is explicitly NOT used here: at one seed it differs from the single call by
10% in readout predictability, 19% in effective rank and 142% in the latent trace, so
a chunked arm would not be comparable to the published one.

Divergence
----------
The `normal` gene likelihood diverges late in training on this numerical stack
(observed at epochs 187-193 in ~2/5 of single-call fits). scvi refuses to serve a
latent whose trainer never completed a fit, so extraction is retried with the trainer
flag forced; the arm is tagged with its `train_status` and the analysis is reported
both over all arms and over completed arms only.

Writes : outputs/positive_result/<out>          summary
         outputs/positive_result/<out>.jsonl    one row per arm, appended as it finishes
Usage  : python code/s1n3_norman_injected_band.py --seeds 3 --seed-start 42
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER1 = os.path.abspath(os.path.join(HERE, ".."))
ROOT = os.path.abspath(os.path.join(PAPER1, "..", ".."))
OUT = os.path.join(ROOT, "outputs", "positive_result")


def log(*a):
    print(*a, flush=True)


def _load_sibling():
    """Import s1n_norman_seed_band as a module so the loader/fit code is not duplicated."""
    p = os.path.join(HERE, "s1n_norman_seed_band.py")
    spec = importlib.util.spec_from_file_location("s1n_norman_seed_band", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)          # __name__ != "__main__", so main() does not run
    return mod


NB = _load_sibling()


def extract_latent(model, adata):
    """Latent of the reached state, with the untained-model guard worked around.

    Returns (Z or None, status).  scvi raises if the trainer never completed a fit,
    which is exactly what a late divergence produces, so we set the flag and take the
    state the model actually reached; the caller decides whether to trust it.
    """
    try:
        return model.get_latent_representation(adata).astype(np.float64), None
    except Exception as e:                                     # noqa: BLE001
        first = f"{type(e).__name__}"
        try:
            model.trainer.is_trained = True
            Z = model.get_latent_representation(adata).astype(np.float64)
            return Z, f"forced-after-{first}"
        except Exception as e2:                                # noqa: BLE001
            return None, f"unextractable: {type(e2).__name__}: {str(e2)[:120]}"


def fit_injected(adata, seed, pert_col, epochs):
    """One injected arm under the published protocol (single call)."""
    import scvi
    import torch
    scvi.settings.seed = seed
    torch.manual_seed(seed)
    a = adata.copy()
    a.obs[pert_col] = a.obs[pert_col].astype(str)
    scvi.model.SCVI.setup_anndata(a, categorical_covariate_keys=[pert_col])
    m = scvi.model.SCVI(a, n_latent=NB.N_LATENT, n_layers=NB.N_LAYERS,
                        dropout_rate=NB.DROPOUT, gene_likelihood=NB.GENE_LIKELIHOOD)
    t0 = time.time()
    status = "completed"
    try:
        m.train(max_epochs=epochs, accelerator="cpu")
    except Exception as e:                                     # noqa: BLE001
        status = f"diverged: {type(e).__name__}: {str(e)[:150]}"
        log(f"    [warn] training raised: {status}")
    seconds = time.time() - t0
    Z, note = extract_latent(m, a)
    if note:
        status = f"{status} | {note}"
    finite = bool(Z is not None and np.isfinite(Z).all())
    return Z, dict(train_status=status, seconds=round(seconds, 1), z_finite=finite)


def load_vanilla_rows():
    """Single-protocol vanilla rows from the baseline band."""
    p = os.path.join(OUT, "s1n_norman_null.jsonl")
    if not os.path.exists(p):
        return {}
    out = {}
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("arm") == "vanilla" and r.get("protocol", "single") == "single":
                out[r["seed"]] = r
    return out


def tstat(v):
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    if v.size < 2:
        return dict(n=int(v.size), mean=float(v.mean()) if v.size else float("nan"),
                    sd=float("nan"), t=float("nan"))
    sd = float(v.std(ddof=1))
    return dict(n=int(v.size), mean=float(v.mean()), sd=sd,
                t=float((v.mean() - 1.0) / (sd / np.sqrt(v.size))) if sd > 0 else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed-start", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=NB.N_EPOCHS)
    ap.add_argument("--pert-col", default=None)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--analyse-only", action="store_true")
    ap.add_argument("--out", default="s1n3_norman_injected_band.json")
    a = ap.parse_args()

    if not a.analyse_only:
        if a.threads:
            import torch
            torch.set_num_threads(a.threads)

        adata, X, pert_col, control = NB.load_norman(a.pert_col)
        log(f"cells={adata.shape[0]} genes={X.shape[1]} pert_col={pert_col} "
            f"control={control} n_labels={adata.obs[pert_col].nunique()}")
        if a.check_only:
            return

        jsonl = os.path.join(OUT, a.out.replace(".json", ".jsonl"))
        done = set()
        if os.path.exists(jsonl):
            with open(jsonl, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        done.add(json.loads(line)["seed"])
            log(f"[resume] {len(done)} injected rows already present: {sorted(done)}")

        os.makedirs(OUT, exist_ok=True)
        for s in range(a.seed_start, a.seed_start + a.seeds):
            if s in done:
                log(f"seed {s} already present, skipping")
                continue
            log(f"=== seed {s}: injected arm (single protocol, {a.epochs} epochs) ===")
            Z, info = fit_injected(adata, s, pert_col, a.epochs)
            row = dict(seed=s, arm="injected", protocol="single", **info)
            if Z is not None:
                row.update(**NB.metrics(Z, X))
            with open(jsonl, "a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            log(f"    seed {s} done: status={row['train_status'][:60]} "
                f"seconds={info['seconds']:.0f} z_finite={info['z_finite']}")

    # ---------------- analysis ----------------
    jsonl = os.path.join(OUT, a.out.replace(".json", ".jsonl"))
    rows = []
    if os.path.exists(jsonl):
        with open(jsonl, encoding="utf-8") as f:
            rows = [json.loads(l) for l in f if l.strip()]
    van = load_vanilla_rows()
    log(f"\n{len(rows)} injected arms, {len(van)} single-protocol vanilla baselines")

    usable_all, usable_done = [], []
    for r in rows:
        v = van.get(r["seed"])
        if v is None or not r.get("z_finite"):
            continue
        rec = dict(seed=r["seed"],
                   kappa_C_W_trace=r["trace"] / v["trace"],
                   kappa_C_W_corrected=r["sum_r2"] / v["sum_r2"],
                   kappa_C_Z=r["trace_Z"] / v["trace_Z"],
                   eff_rank_ratio=r["eff_rank"] / v["eff_rank"])
        rec["D"] = rec["kappa_C_Z"] - rec["kappa_C_W_corrected"]
        usable_all.append(rec)
        if r["train_status"].startswith("completed"):
            usable_done.append(rec)

    out = dict(epochs=a.epochs, n_injected=len(rows),
               n_paired_all=len(usable_all), n_paired_completed=len(usable_done),
               rows=rows, per_seed=usable_all,
               per_seed_completed_only=usable_done)
    for tag, per in (("all", usable_all), ("completed", usable_done)):
        out[f"paired_{tag}"] = {}
        for k in ("kappa_C_W_trace", "kappa_C_W_corrected", "kappa_C_Z",
                  "eff_rank_ratio"):
            out[f"paired_{tag}"][k] = tstat([r[k] for r in per])
        D = [r["D"] for r in per]
        if len(D) >= 2:
            D = np.array(D)
            out[f"paired_{tag}"]["D"] = dict(
                n=len(D), mean=float(D.mean()), sd=float(D.std(ddof=1)),
                t=float(D.mean() / (D.std(ddof=1) / np.sqrt(len(D)))))

    log("\npaired ratios (same seed, injected / vanilla):")
    for tag, label in (("all", "all"), ("completed", "completed-only")):
        per = out.get(f"per_seed{'_completed_only' if tag == 'completed' else ''}") or []
        if not per:
            continue
        log(f"  -- {label} (n={len(per)})")
        for k in ("kappa_C_W_corrected", "kappa_C_W_trace", "kappa_C_Z",
                  "eff_rank_ratio", "D"):
            d = out[f"paired_{tag}"].get(k)
            if not d:
                continue
            t = d.get("t_vs_1", d.get("t"))
            tt = f"{t:+.2f}" if t is not None and np.isfinite(t) else "n/a (n<2)"
            log(f"     {k:<22} {d['mean']:+.4f} +- {d.get('sd', float('nan')):.4f}"
                f"  t = {tt}")

    with open(os.path.join(OUT, a.out), "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    log(f"\nwrote {os.path.join(OUT, a.out)}")


if __name__ == "__main__":
    main()
