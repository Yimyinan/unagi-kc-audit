"""Verify the installed `kappac` package reproduces the published Norman
kappa-C numbers on the REAL cached latents, and persist the result as a JSON
artifact (so the reproduction is citable, not just a log line).

Run:
    python kappac_verify_realdata.py
"""
import json
import os
import numpy as np
from kappac import audit

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "..", "data", "_norman_latents.npz")
OUT = os.path.join(HERE, "..", "data", "kappac_realdata_verification.json")

d = np.load(CACHE, allow_pickle=True)
Zv, Zp, X = d["Zv"], d["Zp"], d["X"]
print("[verify] Zv%s Zp%s X%s" % (np.asarray(Zv).shape, np.asarray(Zp).shape, np.asarray(X).shape))

# manuscript convention: vanilla (Zv) is the reference, conditioned (Zp) is Z1
res = audit(np.asarray(Zv), np.asarray(X), np.asarray(Zp), np.asarray(X), latent_dim=30)
res["source"] = "Norman 2019 GEARS cached latents (scVI vanilla vs scVI perturbation-conditioned)"
res["published_reference"] = dict(kappa_C_W=1.219, kappa_C_Z=0.746,
                                  eff_rank_vanilla=8.80, eff_rank_model=6.15)
res["reproduces_published"] = bool(
    abs(res["kappa_C_W"] - 1.219) < 0.01 and abs(res["kappa_C_Z"] - 0.746) < 0.01)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(res, f, indent=2)
print("[saved] %s" % os.path.abspath(OUT))
print("kappa-C(W)=%.4f  kappa-C(Z)=%.4f  eff_rank %.2f->%.2f (%+.1f%%)  reproduces=%s"
      % (res["kappa_C_W"], res["kappa_C_Z"], res["eff_rank_vanilla"],
         res["eff_rank_model"], res["eff_rank_change_percent"], res["reproduces_published"]))
print("pattern:", res["pattern"])
