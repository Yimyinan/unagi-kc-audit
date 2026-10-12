# -*- coding: utf-8 -*-
"""Recompute the G-VAE hematopoietic vanilla-vs-vanilla kappa null FROM RAW DATA.

Why three functionals: the manuscript's two candidate bands come from DIFFERENT readouts, not
just different functionals of one readout.

  (a) decoder-surrogate  W = ridge(z -> decoder(z)).T   [ptvae GVAE.get_W]      trace functional
  (b) audit Ridge        W = ridge(z_std -> X_std).T    [the audit readout]     trace functional
  (c) audit Ridge        same W as (b)                                          predictability functional

  (a) is what vanilla_null_v2.json used -> the [0.78, 1.23] band.
  (c) is what the Results/captions use    -> the [0.937, 1.064] band.
  (b) is the missing cell of the 2x2: the trace functional applied to the AUDIT readout, which is
      the comparison the manuscript's correction narrative actually implies.

Run with the SYSTEM python (has torch/scanpy/sklearn):
    "C:/Users/12901/AppData/Local/Programs/Python/Python313/python.exe" code/recompute_gvae_vanilla_null.py
"""
import io, json, os, sys, time
import numpy as np

sys.path.insert(0, r"F:\UNAGI_Project\next_phase")
import torch
import scanpy as sc
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from ptvae.gvae_model import GVAE

N_CELLS = int(os.environ.get("N_CELLS", "5000"))
H5AD = r"F:\UNAGI_Project\UNAGI\result\R14\R14_hemato_vanilla\iter_4\stagedata\dataset.h5ad"
SCAN = r"F:\UNAGI_Project\next_phase\results\gvae_hema_scan"
V13 = r"F:\UNAGI_Project\next_phase\results\gvae_hema_v13"
OUT = r"F:/UNAGI_Project/paper1/paper1_kc_diagnostic/data/gvae_vanilla_null_recomputed.json"

CKPTS = [(os.path.join(SCAN, "vanilla_s%d.pt" % s), "scan_s%d" % s) for s in range(1, 13)]
CKPTS += [(os.path.join(V13, "vanilla_s%d.pt" % s), "v13_s%d" % s) for s in (42, 123, 456)]

print("loading h5ad ...", flush=True)
ad = sc.read_h5ad(H5AD)
n_all = ad.n_obs
X = ad.X[:N_CELLS]
X = np.asarray(X.todense(), dtype=np.float32) if hasattr(X, "todense") else np.asarray(X, dtype=np.float32)
n_genes = X.shape[1]
print("  X = %s (of %d cells), %d genes" % (X.shape, n_all, n_genes), flush=True)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
print("  device:", DEV, flush=True)

tr_dec, tr_aud, sum_r2 = {}, {}, {}
t0 = time.time()
for path, name in CKPTS:
    if not os.path.isfile(path):
        print("  MISSING", path, flush=True)
        continue
    ck = torch.load(path, map_location=DEV, weights_only=False)
    model = GVAE(n_genes=n_genes, n_latent=64, n_hidden=512, n_layers=2).to(DEV)
    model.load_state_dict(ck["model"])
    model.eval()
    x_t = torch.tensor(X, dtype=torch.float32).to(DEV)
    with torch.no_grad():
        z = model.get_latent(x_t)                     # (N, 64)
        W_dec = model.get_W(z)                        # (G, 64) decoder surrogate
    W_dec = np.asarray(W_dec)
    tr_dec[name] = float(np.sum(W_dec * W_dec))

    # audit readout: standardised Ridge from the latent to the OBSERVED expression
    z_np = z.detach().cpu().numpy()
    zs = StandardScaler().fit_transform(z_np)
    Xs = StandardScaler(with_mean=False).fit_transform(X)
    rid = Ridge(alpha=1.0, fit_intercept=True).fit(zs, Xs)
    W_aud = rid.coef_.T                               # (G, 64)
    tr_aud[name] = float(np.sum(W_aud * W_aud))

    pred = rid.predict(zs)
    resid = Xs - pred
    ss_res = np.einsum("ij,ij->j", resid, resid)
    mu = np.asarray(Xs.mean(axis=0)).ravel()
    ss_tot = np.einsum("ij,ij->j", Xs - mu, Xs - mu)
    r2 = 1.0 - ss_res / np.maximum(ss_tot, 1e-12)
    sum_r2[name] = float(np.sum(r2))

    print("  %-10s tr_dec=%12.1f  tr_aud=%12.1f  sum_r2=%10.1f   [%.0fs]"
          % (name, tr_dec[name], tr_aud[name], sum_r2[name], time.time() - t0), flush=True)
    del model
    if DEV == "cuda":
        torch.cuda.empty_cache()


def null_from(d):
    ks = sorted(d)
    vals = [d[b] / d[a] for a in ks for b in ks if a != b]
    v = np.array(vals)
    return {"n": len(v), "mean": float(v.mean()), "sd": float(v.std(ddof=1)),
            "lo2s": float(v.mean() - 2 * v.std(ddof=1)),
            "hi2s": float(v.mean() + 2 * v.std(ddof=1)),
            "min": float(v.min()), "max": float(v.max())}


res = {"n_cells": int(X.shape[0]), "n_cells_available": int(n_all), "n_genes": int(n_genes),
       "n_seeds": len(tr_dec), "ckpt_src": "gvae_hema_scan vanilla_s1-12 + gvae_hema_v13 vanilla_s42/123/456",
       "per_seed": {"tr_decoder_surrogate": tr_dec, "tr_audit_readout": tr_aud,
                    "sum_r2_audit_readout": sum_r2},
       "null": {"a_trace_decoder_surrogate": null_from(tr_dec),
                "b_trace_audit_readout": null_from(tr_aud),
                "c_predictability_audit_readout": null_from(sum_r2)}}
io.open(OUT, "w", encoding="utf-8").write(json.dumps(res, indent=2))

print()
print("=" * 74)
print("VANILLA-VANILLA kappa NULL  (n_cells=%d, %d seeds, %d ordered pairs)"
      % (X.shape[0], len(tr_dec), max(0, len(tr_dec) * (len(tr_dec) - 1))))
print("=" * 74)
for key, lab in (("a_trace_decoder_surrogate", "(a) trace, decoder-surrogate W  <- old [0.78,1.23]"),
                 ("b_trace_audit_readout", "(b) trace, AUDIT readout W"),
                 ("c_predictability_audit_readout", "(c) predictability, AUDIT W  <- [0.937,1.064]")):
    n = res["null"][key]
    print("%-44s mean=%.4f sd=%.4f  +-2s=[%.4f, %.4f]"
          % (lab, n["mean"], n["sd"], n["lo2s"], n["hi2s"]))
print()
sd_ratio = res["null"]["b_trace_audit_readout"]["sd"] / res["null"]["c_predictability_audit_readout"]["sd"]
print("trace-vs-predictability sd inflation on the SAME readout: %.2fx" % sd_ratio)
print("saved", OUT)
