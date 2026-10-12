"""S1f -- does the 0/33 headline survive when kappa_C(W) is computed in the
correct (R^2) form?

Background
----------
``data/kappa_complete_audit.json`` stores, for each of the 33 audited configs,
``kappa_ratio = tr_P / tr_V`` -- i.e. the headline classification "no
configuration reaches deep decoder compression (0/33)" is a threshold decision
on the **tr(W'W) ratio**.

S1 (see outputs/positive_result/S1_LIVE_PATH_RESULTS.md sections 2-3) showed
that tr(W'W) equals the quantity the manuscript *names*, ``sum_g R^2_g``, ONLY
when Corr(Z) ~ I.  For a real trained latent Corr(Z) had condition number 2.4e3
and the two forms diverged by up to 40 %; whitening Z restored agreement
exactly.  ``kappa_c_pr_comparison.ridge_spectrum`` now also returns ``sum_r2``
and ``corr_cond``, so both forms can be scored on the published 33 configs.

This script reports, per config, both kappa values, both verdicts, and the
conditioning diagnostic, then summarises how many configs change class.

Usage
    python code/s1f_recheck_0of33.py
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER1 = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(PAPER1, "kappa_c_audit"))

import kappa_c_pr_comparison as K  # noqa: E402
from kappa_c_audit import DEFAULT_BAND, LOCKED_THRESHOLD, classify  # noqa: E402

OUT = r"F:/UNAGI_Project/outputs/positive_result"


def resolve_vanilla(cache):
    """Replicate kappa_c_pr_comparison.main(): first candidate whose tr
    reproduces PUBLISHED_TR_V[tissue] within TR_TOL, else the primary pick."""
    picked, note = {}, {}
    for t, primary in K.VANILLA.items():
        target = K.PUBLISHED_TR_V.get(t)
        cands = [primary] + list(K.VANILLA_FALLBACK.get(t, []))
        best = None
        for c in cands:
            spec = cache.get(c)
            if not spec or "error" in spec:
                continue
            if target is None:
                best = (c, spec); break
            rel = abs(spec["tr"] - target) / max(target, 1e-9)
            if rel <= K.TR_TOL:
                best = (c, spec); note[t] = f"{c} reproduces published tr_V"; break
            if best is None:
                best = (c, spec); note[t] = (f"{c} closest ({rel*100:.1f}% off, "
                                             f"tr={spec['tr']:.1f} vs {target})")
        picked[t] = best
    return picked, note


def main():
    audit = json.load(open(os.path.join(PAPER1, "data", "kappa_complete_audit.json"),
                           encoding="utf-8-sig"))
    cache = K.load_cache()
    print(f"audit configs: {len(audit)}   cache entries: {len(cache)}")
    print(f"band={DEFAULT_BAND}  locked threshold={LOCKED_THRESHOLD}\n")

    vp, vnote = resolve_vanilla(cache)
    print("vanilla denominators resolved:")
    for t in K.VANILLA:
        ent = vp.get(t)
        if not ent:
            print(f"  {t:<8} UNRESOLVED")
            continue
        nm, spec = ent
        print(f"  {t:<8} {nm:<24} tr={spec['tr']:9.1f} "
              f"sum_r2={spec.get('sum_r2', float('nan')):8.2f} "
              f"corr_cond={spec.get('corr_cond', float('nan')):9.1f}"
              f"   {vnote.get(t, '')}")

    rows = []
    for r in audit:
        cfg, tis = r["config"], r["tissue"]
        run = K.RUN_NAME.get(cfg)
        if not run:
            rows.append(dict(config=cfg, tissue=tis, status="NO_MAPPING")); continue
        spec = cache.get(run)
        v = vp.get(tis, (None, None))[1]
        if not spec or "error" in spec or v is None:
            rows.append(dict(config=cfg, tissue=tis, status="MISSING")); continue
        k_tr = spec["tr"] / v["tr"]
        sr, sv = spec.get("sum_r2"), v.get("sum_r2")
        k_r2 = (sr / sv) if (sr and sv) else float("nan")
        rows.append(dict(
            config=cfg, tissue=tis, run=run, status="ok",
            kappa_tr=k_tr, kappa_r2=k_r2,
            verdict_tr=classify(k_tr), verdict_r2=classify(k_r2),
            corr_cond=spec.get("corr_cond"), n_eig_small=spec.get("n_eig_below_0p01"),
            ratio=k_tr / k_r2 if k_r2 else float("nan"),
        ))

    ok = [r for r in rows if r["status"] == "ok"]
    hdr = (f"{'config':<14}{'tissue':<7}{'k_tr(pub)':>10}{'k_r2(corr)':>12}"
           f"{'k_tr/k_r2':>10}{'verdict_tr':>19}{'verdict_r2':>19}{'corrCond':>10}")
    print("\n" + hdr); print("-" * len(hdr))
    for r in sorted(ok, key=lambda x: (x["tissue"], x["config"])):
        print(f"{r['config']:<14}{r['tissue']:<7}{r['kappa_tr']:>10.3f}"
              f"{r['kappa_r2']:>12.3f}{r['ratio']:>10.3f}"
              f"{r['verdict_tr']:>19}{r['verdict_r2']:>19}"
              f"{r['corr_cond']:>10.1f}")

    print("\n" + "=" * 78)
    n = len(ok)
    lock_tr = [r for r in ok if r["kappa_tr"] < LOCKED_THRESHOLD]
    lock_r2 = [r for r in ok if r["kappa_r2"] < LOCKED_THRESHOLD]
    changed = [r for r in ok if r["verdict_tr"] != r["verdict_r2"]]
    print(f"configs scored            : {n}  (of {len(audit)})")
    print(f"HEADLINE  kappa < {LOCKED_THRESHOLD}      : tr form {len(lock_tr)}/{n}"
          f"   ->   R2 form {len(lock_r2)}/{n}")
    if lock_r2:
        print("   NEWLY LOCKED under the R2 form: "
              + ", ".join(f"{r['config']}({r['kappa_r2']:.3f})" for r in lock_r2))
    print(f"verdict class changes     : {len(changed)}/{n}")
    for r in changed:
        print(f"   {r['config']:<14}{r['tissue']:<7} {r['verdict_tr']:<19}"
              f" -> {r['verdict_r2']:<19} "
              f"(k_tr {r['kappa_tr']:.3f} vs k_r2 {r['kappa_r2']:.3f})")
    rr = [r["ratio"] for r in ok if r["ratio"] == r["ratio"]]
    if rr:
        import numpy as np
        rr = np.array(rr)
        print(f"tr/r2 ratio               : median {np.median(rr):.3f}  "
              f"range [{rr.min():.3f}, {rr.max():.3f}]")
    cc = [r["corr_cond"] for r in ok if r.get("corr_cond")]
    if cc:
        import numpy as np
        cc = np.array(cc)
        print(f"Corr(Z) condition number  : median {np.median(cc):.1f}  "
              f"range [{cc.min():.1f}, {cc.max():.1f}]   "
              f"(all >> 1 => tr form precondition violated in every config)")

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "s1f_0of33_recheck.json"), "w") as f:
        json.dump(dict(rows=rows, band=list(DEFAULT_BAND),
                       locked_threshold=LOCKED_THRESHOLD,
                       vanilla={t: (v[0], vnote.get(t)) for t, v in vp.items()}),
                  f, indent=1)
    print("\nsaved -> outputs/positive_result/s1f_0of33_recheck.json")


if __name__ == "__main__":
    main()
