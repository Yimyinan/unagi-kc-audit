"""S1g -- which summary numbers in paper1 depend on the (invalid) tr form?

``data/kappa_c_pr_comparison.json`` reports a family of summary statistics built
on ``kappa_C_W_trace``.  S1f showed that quantity is inflated by latent
ill-conditioning (median inflation 1.25x, up to 5.8x, Corr(Z) condition number
572-24159 in every one of the 33 configs).  This script recomputes the same
summaries with the corrected R^2-form decoder factor, leaving the latent-side
statistics untouched (they are Z-spectrum quantities and involve no Ridge
readout).

Usage: python code/s1g_corrected_summaries.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER1 = os.path.dirname(HERE)
OUT = r"F:/UNAGI_Project/outputs/positive_result"


def main():
    comp = json.load(open(os.path.join(PAPER1, "data", "kappa_c_pr_comparison.json"),
                          encoding="utf-8-sig"))
    re_ = json.load(open(os.path.join(OUT, "s1f_0of33_recheck.json")))
    rep = {r["config"]: r for r in re_["rows"] if r.get("status") == "ok"}

    rows = []
    for r in comp["rows"]:
        cfg = r["config"]
        if cfg not in rep:
            continue
        rows.append(dict(
            config=cfg, tissue=r["tissue"],
            kappa_C_W_tr=r["kappa_C_W_trace"], kappa_r2=rep[cfg]["kappa_r2"],
            kC_Z=r["kappa_C_Z_trace"],
            kappa_tr=r["kappa_C_W_trace"],
            product_tr=r["kappa_C_W_trace"] * r["kappa_C_Z_trace"],
            product_r2=rep[cfg]["kappa_r2"] * r["kappa_C_Z_trace"],
            cond=rep[cfg]["corr_cond"],
        ))
    print(f"configs matched: {len(rows)} / {len(comp['rows'])}\n")

    def summarise(key):
        v = np.array([r[key] for r in rows], float)
        lw = np.log(np.array([r["kappa_tr"] if key == "product_tr"
                              else r["kappa_r2"] for r in rows], float))
        lz = np.log(np.array([r["kC_Z"] for r in rows], float))
        lv = np.log(v)
        return dict(median_product=float(np.median(v)),
                    within_25pct=int((np.abs(v - 1) <= 0.25).sum()),
                    pearson=float(np.corrcoef(lw, lz)[0, 1]),
                    slope=float(np.polyfit(lz, lw, 1)[0]),
                    sign_concordance=float(
                        (np.sign(np.log(np.array([r["kappa_tr"] if key == "product_tr"
                                                  else r["kappa_r2"] for r in rows])))
                         == np.sign(lz)).mean()),
                    vals=[round(float(x), 4) for x in v])

    a, b = summarise("product_tr"), summarise("product_r2")
    print("Q1  TRACE LEVEL -- 'is the product of the two ratios ~ 1?'")
    print(f"{'metric':<22}{'as published (tr)':>20}{'corrected (R2)':>18}")
    for k in ("median_product", "within_25pct", "pearson", "slope",
              "sign_concordance"):
        print(f"{k:<22}{a[k]:>20.4f}{b[k]:>18.4f}")

    print("\nQ3  PER-TISSUE mean kappa_C(W)")
    print(f"{'tissue':<8}{'n':>4}{'as published':>16}{'corrected':>13}")
    tis = {}
    for r in rows:
        tis.setdefault(r["tissue"], []).append(r)
    for t, rs in sorted(tis.items()):
        mt = np.mean([x["kappa_tr"] for x in rs])
        mr = np.mean([x["kappa_r2"] for x in rs])
        mn, mx = min(x["kappa_r2"] for x in rs), max(x["kappa_r2"] for x in rs)
        print(f"{t:<8}{len(rs):>4}{mt:>16.3f}{mr:>13.3f}   (R2 range {mn:.3f}-{mx:.3f})")

    allr2 = np.array([r["kappa_r2"] for r in rows])
    print(f"\nALL configs, corrected kappa_C(W): min {allr2.min():.3f}  "
          f"max {allr2.max():.3f}  mean {allr2.mean():.3f}")
    print(f"  inside the vanilla-equivalence band [0.937, 1.064]: "
          f"{int(((allr2 >= 0.937) & (allr2 <= 1.064)).sum())}/{len(allr2)}")
    print(f"  below the locking threshold 0.5                : "
          f"{int((allr2 < 0.5).sum())}/{len(allr2)}")

    cond = np.array([r["cond"] for r in rows], float)
    print(f"\nCorr(Z) condition number: median {np.median(cond):.0f}  "
          f"min {cond.min():.0f}  max {cond.max():.0f}  "
          f"(>1 in {int((cond > 100).sum())}/{len(cond)} configs)")

    json.dump(dict(rows=rows, trace_level_published=a, trace_level_corrected=b),
              open(os.path.join(OUT, "s1g_corrected_summaries.json"), "w"), indent=1)
    print("\nsaved -> outputs/positive_result/s1g_corrected_summaries.json")


if __name__ == "__main__":
    main()
