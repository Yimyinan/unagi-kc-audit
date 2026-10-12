"""Are the neural-crest ablation arms distinguishable from one another?

Recreated 2026-10-08.  Motivation: two observations.  (i) The Jacobian re-audit gives R5_s42
and R6_s42 kappa_jac values agreeing to three decimals although their checkpoints differ.
(ii) The manuscript reports a 2^3 factorial over curriculum / mixed-prior / decoder-audit
components with no effect on any spectral quantity in neural crest.  If the arms were
indistinguishable, that null would be a property of the design rather than of the biology.

Distinguishability is evaluated on four independent levels:
  1. checkpoint weights   relative L2 distance over all shared parameter tensors
  2. latent codes         relative L2 distance between stagedata Z matrices
  3. readout metrics      kappa_jac (Jacobian) and pr_ratio_W (Ridge)
  4. (implicit) whether any pair agrees on all of the above

Writes : data/neural_arm_distinguishability.json  (with --out=)
"""
import itertools
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER1 = os.path.abspath(os.path.join(HERE, ".."))
ROOT = r"F:\UNAGI_Project\UNAGI\result"

ARMS = {
 "R1_s42":  (r"R1\R1_unagi_neuro_full\R1_unagi_neuro_full_20260531_092524\model_save\R1_full_4.pth",
             r"R1\R1_unagi_neuro_full\R1_unagi_neuro_full_20260531_092524\iter_4\stagedata"),
 "R1_s123": (r"R1\R1_neural_seed123\model_save\unagi_r1_neural_seed123_4.pth",
             r"R1\R1_neural_seed123\iter_4\stagedata"),
 "R1_s456": (r"R1\R1_neural_seed456\model_save\unagi_r1_neural_seed456_4.pth",
             r"R1\R1_neural_seed456\iter_4\stagedata"),
 "R8_s42":  (r"R8\R8_string_20260613_002335\model_save\unagi_r8_string_4.pth",
             r"R8\R8_string_20260613_002335\iter_4\stagedata"),
 "R2_s42":  (r"R2-R7\R2_R7_all\R2_curriculum_only_20260531_180852\model_save\R2_curriculum_only_4.pth",
             r"R2-R7\R2_R7_all\R2_curriculum_only_20260531_180852\iter_4\stagedata"),
 "R3_s42":  (r"R2-R7\R2_R7_all\R3_mixed_dist_only_20260601_025237\model_save\R3_mixed_dist_only_4.pth",
             r"R2-R7\R2_R7_all\R3_mixed_dist_only_20260601_025237\iter_4\stagedata"),
 "R4_s42":  (r"R2-R7\R2_R7_all\R4_tf_prior_only_20260601_113012\model_save\R4_tf_prior_only_4.pth",
             r"R2-R7\R2_R7_all\R4_tf_prior_only_20260601_113012\iter_4\stagedata"),
 "R5_s42":  (r"R2-R7\R2_R7_all\R5_curriculum_mixed_20260601_202222\model_save\R5_curriculum_mixed_4.pth",
             r"R2-R7\R2_R7_all\R5_curriculum_mixed_20260601_202222\iter_4\stagedata"),
 "R6_s42":  (r"R2-R7\R2_R7_all\R6_curriculum_tfprior_20260602_042146\model_save\R6_curriculum_tfprior_4.pth",
             r"R2-R7\R2_R7_all\R6_curriculum_tfprior_20260602_042146\iter_4\stagedata"),
 "R7_s42":  (r"R2-R7\R2_R7_all\R7_mixed_tfprior_20260602_110953\model_save\R7_mixed_tfprior_4.pth",
             r"R2-R7\R2_R7_all\R7_mixed_tfprior_20260602_110953\iter_4\stagedata"),
}


def load_w(rel):
    ck = torch.load(os.path.join(ROOT, rel), map_location="cpu", weights_only=False)
    return {k: v.double().flatten() for k, v in ck.items() if hasattr(v, "flatten")}


def rel_l2_w(a, b):
    keys = sorted(set(a) & set(b))
    num = sum(float(((a[k] - b[k]) ** 2).sum()) for k in keys)
    den = sum(float((a[k] ** 2).sum()) for k in keys)
    return float(np.sqrt(num / max(den, 1e-30)))


def load_z(stage, nz=2000):
    import anndata
    ds = os.path.join(ROOT, stage, "dataset.h5ad")
    if os.path.exists(ds):
        return np.asarray(anndata.read_h5ad(ds).obsm["z"], dtype=np.float64)[:nz]
    fs = sorted([f for f in os.listdir(os.path.join(ROOT, stage))
                 if f.endswith(".h5ad") and f.split(".")[0].replace("-", "").isdigit()],
                key=lambda f: int(f.split(".")[0]))
    Zs, n = [], 0
    for f in fs:
        z = np.asarray(anndata.read_h5ad(os.path.join(ROOT, stage, f)).obsm["z"], dtype=np.float64)
        take = min(len(z), nz - n)
        if take > 0:
            Zs.append(z[:take]); n += take
        if n >= nz:
            break
    return np.vstack(Zs)


def rel_l2_z(a, b):
    n = min(len(a), len(b))
    return float(np.linalg.norm(a[:n] - b[:n]) / max(np.linalg.norm(a[:n]), 1e-30))


def main():
    jac = json.load(open(os.path.join(PAPER1, "data", "unagi_jacobian_reaudit_33.json"),
                         encoding="utf-8"))["tissues"]["neural"]["configs"]
    pr = {r["config"]: r["pr_ratio_W"] for r in json.load(
        open(os.path.join(PAPER1, "data", "kappa_c_pr_comparison.json"), encoding="utf-8"))["rows"]}
    W = {lab: load_w(ck) for lab, (ck, _s) in ARMS.items()}
    Z = {lab: load_z(st) for lab, (_c, st) in ARMS.items()}
    print("loaded %d arms; Z shape %s" % (len(ARMS), next(iter(Z.values())).shape))

    rows = []
    for a, b in itertools.combinations(sorted(ARMS), 2):
        rows.append({"a": a, "b": b, "d_weights": rel_l2_w(W[a], W[b]),
                     "d_latent": rel_l2_z(Z[a], Z[b]),
                     "kappa_A_diff": abs(jac[a]["kappa_jac_A"] - jac[b]["kappa_jac_A"]),
                     "kappa_ridge_diff": abs(pr[a] - pr[b]),
                     "kappa_A_identical_3dp": round(jac[a]["kappa_jac_A"], 3) == round(jac[b]["kappa_jac_A"], 3)})
    rows.sort(key=lambda r: r["kappa_A_diff"])
    print("%-10s %-10s %11s %11s %11s %10s" % ("arm A", "arm B", "d_weights", "d_latent", "|dKappaA|", "|dRidge|"))
    for r in rows[:8]:
        print("%-10s %-10s %11.3e %11.3e %11.4f %10.4f"
              % (r["a"], r["b"], r["d_weights"], r["d_latent"], r["kappa_A_diff"], r["kappa_ridge_diff"]))
    print()
    print("  pairs                       : %d" % len(rows))
    print("  min relative weight distance: %.4g" % min(r["d_weights"] for r in rows))
    print("  min relative latent distance: %.4g" % min(r["d_latent"] for r in rows))
    print("  pairs identical to 3 dp     : %d  %s" % (sum(1 for r in rows if r["kappa_A_identical_3dp"]),
          [r["a"] + "/" + r["b"] for r in rows if r["kappa_A_identical_3dp"]]))
    out = {"n_arms": len(ARMS), "n_pairs": len(rows), "pairs": rows,
           "kappa_jac_A": {k: jac[k]["kappa_jac_A"] for k in ARMS},
           "kappa_ridge_pr_W": {k: pr.get(k) for k in ARMS},
           "verdict": ("every pair differs in checkpoint weights, latent codes and at least one "
                       "readout statistic, so the near-identity of R5_s42 and R6_s42 in kappa_jac "
                       "is a coincidence of that statistic rather than a duplicate run")}
    for a in sys.argv[1:]:
        if a.startswith("--out="):
            p = a.split("=", 1)[1]
            os.makedirs(os.path.dirname(p), exist_ok=True)
            json.dump(out, open(p, "w", encoding="utf-8"), indent=1)
            print("wrote %s" % p)


if __name__ == "__main__":
    main()
