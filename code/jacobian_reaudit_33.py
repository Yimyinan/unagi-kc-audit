"""Extend the analytical-Jacobian re-audit from 15 of the 33 audited configurations to all 33.

Recreated 2026-10-08 after the workspace cleanup incident.  Method is identical to
next_phase/recompute_unagi_jacobian.py, the script that produced the published 15 (which is
recomputed here as a regression control).

  Z, X  = iter_4 stagedata (dataset.h5ad if present, else stacked numeric .h5ad)
  xstd  = X[:2000].std(0); xstd[xstd==0]=1 ; gs = 1/xstd            ("first2000" scale)
  a1 = z @ W3.T + b3 ; h3 = softplus(a1) ; Dh = sigmoid(a1) ; mu = h3 @ W4.T + b4
  A: mup = clamp(mu, min=-1); Dmu = (mu>-1)*sigmoid(mup); J = Dmu*(W4 @ (Dh*W3))
  B: J = W4 @ (Dh*W3)
  Jbar = mean over the first NK=1000 latent rows ; trJ = trace(Jbar^T Jbar)
  kappa_jac = trJ_prior / trJ_vanilla_tissue        (vanilla = organ: hemo R14, neural R0, lung R10)

Writes : data/unagi_jacobian_reaudit_33.json  (with --out=)
"""
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = r"F:\UNAGI_Project\UNAGI\result"
NK, NSC = 1000, 2000

T = {
 "hemo": {
  "v_stage": r"R14\R14_hemato_vanilla\iter_4\stagedata",
  "v_ck": r"R14\R14_hemato_vanilla\model_save\R14_vanilla_seed42_4.pth",
  "c": [
   ("R11_s42", r"R11\R11_KLF1_TAL1_seed42\iter_4\stagedata", r"R11\R11_KLF1_TAL1_seed42\model_save\R11_KLF1_TAL1_4.pth", True),
   ("R11_s123", r"R11\R11_KLF1_TAL1_seed123\iter_4\stagedata", r"R11\R11_KLF1_TAL1_seed123\model_save\R11_KLF1_TAL1_4.pth", True),
   ("R12_s42", r"R12\R12_GATA2_GFI1B_PAX5_seed42\iter_4\stagedata", r"R12\R12_GATA2_GFI1B_PAX5_seed42\model_save\R12_GATA2_GFI1B_PAX5_4.pth", True),
   ("R13_s42", r"R13\R13_hemato_trueprior_seed42\iter_4\stagedata", r"R13\R13_hemato_trueprior_seed42\model_save\R13_hemato_trueprior_seed42_4.pth", True),
   ("R15_s42", r"R15\R14_hemato_seed42\iter_4\stagedata", r"R15\R14_hemato_seed42\model_save\unagi_r14_hemato_seed42_4.pth", True),
   ("R107_s42", r"R107_erythroid_lymphoid_seed42\iter_4\stagedata", r"R107_erythroid_lymphoid_seed42\model_save\R107_erythroid_lymphoid_4.pth", True),
   ("R111_s42", r"R111\R111_erythroid_temporal_seed42\iter_4\stagedata", r"R111\R111_erythroid_temporal_seed42\model_save\R111_erythroid_temporal_4.pth", True),
   ("R84_s42", r"R84\R84_erythroid_seed42\iter_4\stagedata", r"R84\R84_erythroid_seed42\model_save\R84_erythroid_4.pth", True),
   ("R165_s42", r"R165\R165_erythroid_myeloid_seed42\iter_4\stagedata", r"R165\R165_erythroid_myeloid_seed42\model_save\R165_erythroid_myeloid_4.pth", True),
   ("R16_s42", r"R16\R14_wrongprior_seed42\iter_4\stagedata", r"R16\R14_wrongprior_seed42\model_save\R14_wrongprior_seed42_4.pth", True),
   ("R11_s456", r"R11\R11_KLF1_TAL1_seed456\iter_4\stagedata", r"R11\R11_KLF1_TAL1_seed456\model_save\R11_KLF1_TAL1_4.pth", False),
   ("R12_s123", r"R12\R12_GATA2_GFI1B_PAX5_seed123\iter_4\stagedata", r"R12\R12_GATA2_GFI1B_PAX5_seed123\model_save\R12_GATA2_GFI1B_PAX5_4.pth", False),
   ("R12_s456", r"R12\R12_GATA2_GFI1B_PAX5_seed456\iter_4\stagedata", r"R12\R12_GATA2_GFI1B_PAX5_seed456\model_save\R12_GATA2_GFI1B_PAX5_4.pth", False),
   ("R13_s123", r"R13\R13_hemato_trueprior_seed123\iter_4\stagedata", r"R13\R13_hemato_trueprior_seed123\model_save\R13_hemato_trueprior_seed123_4.pth", False),
   ("R13_s456", r"R13\R13_hemato_trueprior_seed456\iter_4\stagedata", r"R13\R13_hemato_trueprior_seed456\model_save\R13_hemato_trueprior_seed456_4.pth", False),
   ("R13_KO_KLF1", r"R13\R13_KO_KLF1_seed42\iter_4\stagedata", r"R13\R13_KO_KLF1_seed42\model_save\R13_KO_KLF1_seed42_4.pth", False),
   ("R13_OE_KLF1", r"R13\R13_OE_KLF1_seed42\iter_4\stagedata", r"R13\R13_OE_KLF1_seed42\model_save\R13_OE_KLF1_seed42_4.pth", False),
   ("R15_s123", r"R15\R14_hemato_seed123\iter_4\stagedata", r"R15\R14_hemato_seed123\model_save\unagi_r14_hemato_seed123_4.pth", False),
   ("R15_s456", r"R15\R14_hemato_seed456\iter_4\stagedata", r"R15\R14_hemato_seed456\model_save\unagi_r14_hemato_seed456_4.pth", False),
   ("R84_s123", r"R84\R84_erythroid_seed123\iter_4\stagedata", r"R84\R84_erythroid_seed123\model_save\R84_erythroid_4.pth", False),
   ("R111_s123", r"R111\R111_erythroid_temporal_seed123-1\iter_4\stagedata", r"R111\R111_erythroid_temporal_seed123-1\model_save\R111_erythroid_temporal_4.pth", False),
   ("R111_s456", r"R111\R111_erythroid_temporal_seed456\iter_4\stagedata", r"R111\R111_erythroid_temporal_seed456\model_save\R111_erythroid_temporal_4.pth", False),
  ]},
 "neural": {
  "v_stage": r"R0\R0_vanilla_unagi_20260530_223153\iter_4\stagedata",
  "v_ck": r"R0\R0_vanilla_unagi_20260530_223153\model_save\R0_vanilla_4.pth",
  "c": [
   ("R1_s42", r"R1\R1_unagi_neuro_full\R1_unagi_neuro_full_20260531_092524\iter_4\stagedata", r"R1\R1_unagi_neuro_full\R1_unagi_neuro_full_20260531_092524\model_save\R1_full_4.pth", True),
   ("R1_s123", r"R1\R1_neural_seed123\iter_4\stagedata", r"R1\R1_neural_seed123\model_save\unagi_r1_neural_seed123_4.pth", True),
   ("R1_s456", r"R1\R1_neural_seed456\iter_4\stagedata", r"R1\R1_neural_seed456\model_save\unagi_r1_neural_seed456_4.pth", True),
   ("R8_s42", r"R8\R8_string_20260613_002335\iter_4\stagedata", r"R8\R8_string_20260613_002335\model_save\unagi_r8_string_4.pth", True),
   ("R2_s42", r"R2-R7\R2_R7_all\R2_curriculum_only_20260531_180852\iter_4\stagedata", r"R2-R7\R2_R7_all\R2_curriculum_only_20260531_180852\model_save\R2_curriculum_only_4.pth", False),
   ("R3_s42", r"R2-R7\R2_R7_all\R3_mixed_dist_only_20260601_025237\iter_4\stagedata", r"R2-R7\R2_R7_all\R3_mixed_dist_only_20260601_025237\model_save\R3_mixed_dist_only_4.pth", False),
   ("R4_s42", r"R2-R7\R2_R7_all\R4_tf_prior_only_20260601_113012\iter_4\stagedata", r"R2-R7\R2_R7_all\R4_tf_prior_only_20260601_113012\model_save\R4_tf_prior_only_4.pth", False),
   ("R5_s42", r"R2-R7\R2_R7_all\R5_curriculum_mixed_20260601_202222\iter_4\stagedata", r"R2-R7\R2_R7_all\R5_curriculum_mixed_20260601_202222\model_save\R5_curriculum_mixed_4.pth", False),
   ("R6_s42", r"R2-R7\R2_R7_all\R6_curriculum_tfprior_20260602_042146\iter_4\stagedata", r"R2-R7\R2_R7_all\R6_curriculum_tfprior_20260602_042146\model_save\R6_curriculum_tfprior_4.pth", False),
   ("R7_s42", r"R2-R7\R2_R7_all\R7_mixed_tfprior_20260602_110953\iter_4\stagedata", r"R2-R7\R2_R7_all\R7_mixed_tfprior_20260602_110953\model_save\R7_mixed_tfprior_4.pth", False),
  ]},
 "lung": {
  "v_stage": r"R10\4\stagedata",
  "v_ck": r"R10\model_save\unagi_lung_r10_4.pth",
  "c": [("R9", r"R9\r9_results\r9_backup\4\stagedata", r"R9\r9_results\r9_backup\unagi_lung_r9_4.pth", True)]},
}


def load_zx(stage, need_x=True, nz=None):
    import anndata
    ds = os.path.join(ROOT, stage, "dataset.h5ad")
    if os.path.exists(ds):
        ad = anndata.read_h5ad(ds)
        Z = np.asarray(ad.obsm["z"], dtype=np.float64)
        if nz is not None:
            Z = Z[:nz]
        if not need_x:
            return Z, None
        X = ad.X.toarray() if hasattr(ad.X, "toarray") else np.array(ad.X)
        X = np.asarray(X, dtype=np.float32)
        return Z, (X[:nz] if nz is not None else X)
    fs = sorted([f for f in os.listdir(os.path.join(ROOT, stage))
                 if f.endswith(".h5ad") and f.split(".")[0].replace("-", "").isdigit()],
                key=lambda f: int(f.split(".")[0]))
    Zs, Xs = [], []
    for f in fs:
        ad = anndata.read_h5ad(os.path.join(ROOT, stage, f))
        Z = np.asarray(ad.obsm["z"], dtype=np.float64)
        if nz is not None:
            Z = Z[:max(0, nz - sum(len(z) for z in Zs))]
        Zs.append(Z)
        if need_x:
            X = ad.X.toarray() if hasattr(ad.X, "toarray") else np.array(ad.X)
            X = np.asarray(X, dtype=np.float32)
            if nz is not None:
                X = X[:max(0, nz - sum(len(x) for x in Xs))]
            Xs.append(X)
        if nz is not None and sum(len(z) for z in Zs) >= nz:
            break
    return (np.vstack(Zs), np.vstack(Xs)) if need_x else (np.vstack(Zs), None)


def load_dec(rel):
    ck = torch.load(os.path.join(ROOT, rel), map_location="cpu", weights_only=False)
    return (ck["decoder.fc3.weight"].double(), ck["decoder.fc3.bias"].double(),
            ck["decoder.fc4.weight"].double(), ck["decoder.fc4.bias"].double())


def jac_trace(model, z, gs, variant, chunk=32):
    W3, b3, W4, b4 = model
    g = torch.tensor(np.asarray(gs), dtype=torch.float64)
    S = None
    n = 0
    for i in range(0, len(z), chunk):
        zb = torch.tensor(np.asarray(z[i:i + chunk]), dtype=torch.float64)
        a1 = zb @ W3.T + b3
        h3 = torch.nn.functional.softplus(a1)
        Dh = torch.sigmoid(a1)
        mu = h3 @ W4.T + b4
        A = W4 @ (Dh.unsqueeze(2) * W3.unsqueeze(0))
        if variant == "A":
            mup = torch.clamp(mu, min=-1.0)
            J = ((mu > -1.0).double() * torch.sigmoid(mup)).unsqueeze(2) * A
        else:
            J = A
        J = J * g[None, :, None]
        S = J.sum(dim=0) if S is None else S + J.sum(dim=0)
        n += len(zb)
    Jb = S / n
    return float((Jb.T @ Jb).trace())


def main():
    t0 = time.time()
    pub = json.load(open(r"F:\UNAGI_Project\next_phase\results\unagi_jacobian_recompute.json",
                         encoding="utf-8-sig"))["tissues"]
    out = {"protocol": "identical to next_phase/recompute_unagi_jacobian.py; first2000 scale; "
                       "NK=1000; variants A and B; extended to all 33 configurations", "tissues": {}}
    worst = 0.0
    nreg = 0
    for tis, spec in T.items():
        print("[%s] vanilla stagedata" % tis, flush=True)
        Zv, Xv = load_zx(spec["v_stage"], True)
        xs = Xv[:NSC].std(0)
        xs[xs == 0] = 1.0
        gs = 1.0 / xs
        mv = load_dec(spec["v_ck"])
        trv = {"A": jac_trace(mv, Zv[:NK], gs, "A"), "B": jac_trace(mv, Zv[:NK], gs, "B")}
        pv = pub[tis]["vanilla"]["first2000"]
        print("  vanilla A=%.3f (pub %.3f)  B=%.3f (pub %.3f)" % (trv["A"], pv["A"], trv["B"], pv["B"]))
        del Zv, Xv
        confs = {}
        for name, stage, ck, old in spec["c"]:
            Zp, _ = load_zx(stage, False, NK)
            mp = load_dec(ck)
            a = jac_trace(mp, Zp, gs, "A")
            b = jac_trace(mp, Zp, gs, "B")
            kA, kB = a / trv["A"], b / trv["B"]
            confs[name] = {"trJ_prior_A": a, "trJ_prior_B": b, "kappa_jac_A": kA,
                           "kappa_jac_B": kB, "C_jac_pct_A": (1 - kA) * 100.0,
                           "C_jac_pct_B": (1 - kB) * 100.0,
                           "previously_treated": bool(old), "n_latent_rows": int(len(Zp))}
            print("  %s%-14s kappa_A=%.4f kappa_B=%.4f" % ("OLD " if old else "NEW ", name, kA, kB), flush=True)
            if old:
                worst = max(worst, abs(kA - pub[tis]["configs"][name]["kappa_jac"]["first2000_A"]))
                nreg += 1
            del Zp, mp
        out["tissues"][tis] = {"vanilla": trv, "configs": confs}
        print("  [%s] done %.0fs" % (tis, time.time() - t0), flush=True)
    out["regression"] = {"n": nreg, "max_abs_deviation": worst}
    allk = [v["kappa_jac_A"] for b in out["tissues"].values() for v in b["configs"].values()]
    out["verdict"] = {"n_configs": len(allk), "n_locked": int(sum(1 for k in allk if k < 0.5)),
                      "kappa_A_min": min(allk), "kappa_A_max": max(allk)}
    print("\nREGRESSION max |dev| over %d published = %.3g" % (nreg, worst))
    print("N=%d  locked=%d  kappa_A %.4f-%.4f" % (len(allk), out["verdict"]["n_locked"], min(allk), max(allk)))
    for arg in sys.argv[1:]:
        if arg.startswith("--out="):
            p = arg.split("=", 1)[1]
            os.makedirs(os.path.dirname(p), exist_ok=True)
            json.dump(out, open(p, "w", encoding="utf-8"), indent=1)
            print("wrote %s" % p)


if __name__ == "__main__":
    main()
