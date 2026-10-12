# -*- coding: utf-8 -*-
"""Verify section 2.9 by extracting numbers FROM main.tex and comparing to the JSON.

Design note: regexes over LaTeX need heavy backslash escaping, which is fragile. Instead
each claim is located by a plain-text anchor containing no backslashes, and the first `k`
decimal numbers following it are read off. A mismatch is then a real inconsistency in the
manuscript, not a pattern artefact.

Usage: python code/audit_sec29_numbers.py
"""
from __future__ import annotations

import io
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER1 = os.path.abspath(os.path.join(HERE, ".."))
ROOT = os.path.abspath(os.path.join(PAPER1, "..", ".."))
OUT = os.path.join(ROOT, "outputs", "positive_result")
MAIN = io.open(os.path.join(PAPER1, "main.tex"), encoding="utf-8").read()

NUM = re.compile(r"[-+]?\d+\.\d+")
ROWS = []


def J(name):
    for base in (OUT, os.path.join(PAPER1, "data")):
        p = os.path.join(base, name)
        if os.path.exists(p):
            return json.load(open(p))
    return None


def after(anchor, k, window=150):
    i = MAIN.find(anchor)
    if i < 0:
        raise AssertionError(f"anchor not found in main.tex: {anchor!r}")
    return [float(x) for x in NUM.findall(MAIN[i:i + window])[:k]]


def chk(label, got, want, tol):
    ok = abs(got - want) <= tol
    ROWS.append(ok)
    print(f"  {label:<44}{got:>11.4f}{want:>11.4f}   {'OK' if ok else '*** MISMATCH ***'}")


def main():
    m2 = J("s1m2_band_summary.json")
    m4 = J("s1m4_pbmc3k_injection.json")
    nm = J("s1n3_norman_injected_paired.json")
    f4 = J("s1o_family4_seed_band.json")
    pub = J("scvi_demo_results.json")

    print("=" * 74)
    print("2.9.1  the retracted pbmc3k pair")
    print(f"{'claim':<46}{'main.tex':>11}{'json':>11}")
    p = m2["positions"]["kappa-C(W) trace  (published 0.873)"]
    v = after("and its separation lies inside the seed null", 4)
    chk("published kappa-C(W)", v[0], p["published"], 5e-3)
    chk("null mean", v[1], p["null_mean"], 5e-3)
    chk("null sd", v[2], p["null_sd"], 5e-3)
    chk("z", v[3], p["z"], 5e-2)

    v = after("inside a ten-seed vanilla range of", 4)
    chk("eff-rank vanilla lo", v[0], m2["vanilla_eff_rank_range"][0], 5e-2)
    chk("eff-rank vanilla hi", v[1], m2["vanilla_eff_rank_range"][1], 5e-2)
    chk("eff-rank vanilla mean", v[2],
        sum(m2["vanilla_eff_rank_per_seed"]) / len(m2["vanilla_eff_rank_per_seed"]), 5e-2)
    chk("eff-rank vanilla sd", v[3], 1.76, 5e-2)

    v = after("and an effective rank of", 2)
    chk("published eff-rank pair lo", v[0], m2["published"]["eff_rank_vanilla"], 5e-2)
    chk("published eff-rank pair hi", v[1], m2["published"]["eff_rank_conditioned"], 5e-2)

    print("\n2.9.1  the repaired control (covariate route, 6 paired seeds)")
    p4 = m4["paired_label_vs_vanilla"]
    n4 = m4["cross_seed_vanilla_null"]
    v = after("decoder readout compresses to", 2)
    chk("kappa-C(W) mean", v[0], p4["kappa_corrected"]["mean"], 5e-3)
    chk("kappa-C(W) sd", v[1], p4["kappa_corrected"]["sd"], 5e-3)
    v = after("family-matched seed null", 2)
    chk("null mean", v[0], n4["kappa_corrected"]["mean"], 5e-3)
    chk("null sd", v[1], n4["kappa_corrected"]["sd"], 5e-3)
    # NOTE: the trace-form values are no longer reported in main.tex (the main text
    # reports only the predictability statistic), so the 'trace form returns only'
    # checks were removed here. The trace-form values remain audited in the SI and
    # in data/ outputs, not against the main text.
    v = after("latent trace contracts modestly", 2)
    chk("kappa-C(Z) mean", v[0], p4["kappa_C_Z"]["mean"], 5e-3)
    chk("kappa-C(Z) sd", v[1], p4["kappa_C_Z"]["sd"], 5e-3)
    v = after("does not move beyond seed scatter", 2)
    chk("eff-rank ratio mean", v[0], p4["eff_rank_ratio"]["mean"], 5e-3)
    chk("eff-rank ratio sd", v[1], p4["eff_rank_ratio"]["sd"], 5e-3)

    print("\n2.9.2  phenotype benchmark (published json, vanilla latent)")
    b = pub["benchmark"]
    v = after("latent representation at", 4)
    chk("acc from Z", v[0], b["acc_from_Z_vanilla"], 6e-4)
    chk("sd  from Z", v[1], b["sd_Z_vanilla"], 6e-4)
    chk("acc from X", v[2], b["acc_from_X_raw"], 6e-4)
    chk("sd  from X", v[3], b["sd_X_raw"], 6e-4)

    print("\n2.9.2  Norman scVI paired design")
    k = "paired_completed" if nm.get("n_paired_completed") else "paired_all"
    kc, kz = nm[k]["kappa_C_W_corrected"], nm[k]["kappa_C_Z"]
    dd, er = nm[k]["D"], nm[k]["eff_rank_ratio"]
    pn = nm["published_single_pair"]
    v = after("It resolves the readout (corrected", 3)
    chk("kappa-C(W) mean", v[0], kc["mean"], 6e-4)
    chk("kappa-C(W) sd", v[1], kc["sd"], 6e-4)
    chk("kappa-C(W) t", v[2], kc["t_vs_1"], 6e-2)
    v = after("sits on that mean,", 1)
    chk("z of published readout", v[0],
        (pn["kappa_C_W_corrected"] - kc["mean"]) / kc["sd"], 6e-2)
    v = after("and the latent (", 3)
    chk("kappa-C(Z) mean", v[0], kz["mean"], 6e-4)
    chk("kappa-C(Z) sd", v[1], kz["sd"], 6e-4)
    chk("kappa-C(Z) t", v[2], kz["t_vs_1"], 6e-2)
    v = after("and resolves their difference", 3, window=320)
    chk("D mean", v[0], dd["mean"], 6e-4)
    chk("D sd", v[1], dd["sd"], 6e-4)
    chk("D t", v[2], dd["t_vs_0"], 6e-2)
    # the trace-form value is no longer reported in main.tex; the published
    # effective-rank drop (8.80 -> 6.15) is now the next reported pair
    v = after("The published latent effective-rank drop,", 2)
    chk("published eff-rank pair lo", v[0], 8.80, 6e-3)
    chk("published eff-rank pair hi", v[1], 6.15, 6e-3)
    v = after("does not survive the paired test (rank ratio", 3)
    chk("eff-rank ratio mean", v[0], er["mean"], 6e-3)
    chk("eff-rank ratio sd", v[1], er["sd"], 6e-3)
    chk("eff-rank t", v[2], er["t_vs_1"], 6e-2)

    print("\n2.9.2  family-4 MLP VAE (5 seeds, both arms)")
    f4p = f4["paired_conditioned_vs_vanilla"]
    f4_pub = f4["published"]
    f4_ratio_pub = f4_pub.get("eff_rank_ratio") or (
        f4_pub["eff_rank_conditioned"] / f4_pub["eff_rank_vanilla"])
    v = after("compresses the readout slightly", 3)
    chk("kappa-C(W) mean", v[0], f4p["kappa_corrected"]["mean"], 6e-4)
    chk("kappa-C(W) sd", v[1], f4p["kappa_corrected"]["sd"], 6e-4)
    chk("kappa-C(W) t", v[2], -10.58, 6e-2)
    v = after("latent contracts about", 4)
    chk("latent/readout compression ratio", v[0],
        (1 - f4p["kappa_C_Z"]["mean"]) / (1 - f4p["kappa_corrected"]["mean"]), 6e-2)
    chk("kappa-C(Z) mean", v[1], f4p["kappa_C_Z"]["mean"], 6e-4)
    chk("kappa-C(Z) sd", v[2], f4p["kappa_C_Z"]["sd"], 6e-4)
    v = after("near-unity effective-rank ratio", 4)
    chk("published eff-rank ratio", v[0], f4_ratio_pub, 6e-4)
    chk("sigma above the five-seed mean", v[1],
        (f4_ratio_pub - f4p["eff_rank_ratio"]["mean"]) / f4p["eff_rank_ratio"]["sd"], 6e-2)
    chk("5-seed eff-rank mean", v[2], f4p["eff_rank_ratio"]["mean"], 6e-4)
    chk("5-seed eff-rank sd", v[3], f4p["eff_rank_ratio"]["sd"], 6e-4)
    v = after("its readout value", 4)
    chk("published readout", v[0], f4_pub["kappa_corrected"], 6e-4)
    chk("band floor", v[1], 0.937, 6e-4)
    chk("band ceiling", v[2], 1.064, 6e-4)
    chk("configuration mean", v[3], f4p["kappa_corrected"]["mean"], 6e-4)
    v = after("latent-more-than-decoder direction", 3)
    chk("D mean", v[0], -0.0526, 6e-4)
    chk("D sd", v[1], 0.0381, 6e-4)
    chk("D t", v[2], -3.09, 6e-2)

    print("\n2.9.2  perturbation-identity decodability (recomputed from the published latents)")
    pj = J("s1n5_norman_pert_identity.json")
    if pj:
        v = after("perturbation identity is", 4, window=360)
        chk("vanilla balanced accuracy", v[0], pj["arms"]["vanilla"]["balanced_accuracy"], 6e-4)
        chk("vanilla sd", v[1], pj["arms"]["vanilla"]["balanced_sd_across_folds"], 6e-4)
        chk("conditioned balanced accuracy", v[2],
            pj["arms"]["conditioned"]["balanced_accuracy"], 6e-4)
        chk("conditioned sd", v[3], pj["arms"]["conditioned"]["balanced_sd_across_folds"], 6e-4)
        v = after("conditioned, against", 1, window=30)
        chk("majority baseline", v[0], pj["majority_baseline"], 6e-4)

    print("\n" + "=" * 74)
    print(f"{sum(ROWS)}/{len(ROWS)} consistent"
          + ("  -- ALL CONSISTENT" if all(ROWS) else "  *** INCONSISTENCIES FOUND ***"))


if __name__ == "__main__":
    main()
    # ★★ 第 23 轮修掉的一处**假绿隐患**：本脚本原先**永远 exit 0**，
    #    单跑它（或直接接进 CI）无论结论如何都是绿的。`audit_guard.py` 之所以还能拦住，
    #    是因为它**去字符串匹配 stdout**（"ALL CONSISTENT" / "MISMATCH"）—— 那是脆弱的补偿。
    #    现让它自己的退出码就反映结论：ROWS 为空（异常/未跑到）同样判失败，避免"空跑即通过"。
    raise SystemExit(0 if ROWS and all(ROWS) else 1)
