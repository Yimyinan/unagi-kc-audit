"""
celloracle_cross_architecture_sensitivity.py
============================================
Worst-case sensitivity for the cross-architecture claim.

`_hessian.compute_kappa` uses `loading_hessian()`, not a raw L L^T, and the
original CellOracle audit script was not archived. So we do not claim to have
reproduced the published vanilla gap (478.2565) exactly -- the closest
decoder_unified config (R14_s42 / R15_s42, a 464.0 gap) differs by 3%.

Instead of pretending, bound the conclusion. The question that matters is:

    how large would the vanilla gap have to be before any of the 8
    CellOracle-style priors became critical?

Since kappa = ||H_P - H_V||_2 / gap, and the numerator is fixed by the audit's
own matrices, the required gap is simply

    gap* = ||H_P - H_V||_2 / 0.5

If gap* is small compared with every plausible reference gap, the subcritical
verdict is safe regardless of which vanilla we anchor to.

Outputs
-------
  outputs/positive_result/celloracle_sensitivity.json
"""
import json
from pathlib import Path

import numpy as np

REPO = Path("F:/UNAGI_Project")
OUT = REPO / "outputs" / "positive_result"
CO = Path("F:/UNAGI_Project/UNAGI/celloracle_audit/output")
KAPPA_CRITICAL = 0.5


def main():
    ka = json.load(open(CO / "kappa_alpha_batch.json", encoding="utf-8-sig"))
    gap_audit = ka["vanilla_gap"]

    rows = []
    for r in ka["tf_results"]:
        num = r["kappa"] * gap_audit            # ||H_P - H_V||_2
        gap_star = num / KAPPA_CRITICAL         # gap needed to reach critical
        rows.append({
            "tf": r["tf"],
            "kappa_original": r["kappa"],
            "numerator_norm": num,
            "gap_required_for_critical": gap_star,
            "margin_factor_vs_audit_gap": gap_audit / gap_star,
        })

    gap_star_max = max(r["gap_required_for_critical"] for r in rows)
    # every reference gap we measured / recorded
    refs = {
        "audit_own": gap_audit,
        "R14-R15_s42": 464.00918298039994,
        "R107_s42": 502.8765483510069,
        "R1_s456": 449.607226,
        "R14_s123": 526.303,
        "R0_s123": 1163.010,
        "R0_s456": 1727.837,
    }
    smallest_ref = min(refs.values())
    safety = smallest_ref / gap_star_max

    rep = {
        "rows": rows,
        "gap_required_for_critical_max": gap_star_max,
        "smallest_reference_gap": smallest_ref,
        "safety_factor": safety,
        "verdict": ("robust" if safety > 10 else
                    "marginal" if safety > 1 else "fails"),
        "claim_boundary": (
            "We do NOT claim to have reproduced the published vanilla gap "
            f"({gap_audit:.4f}); the closest decoder_unified reference differs "
            f"by {abs(464.00918298039994 - gap_audit) / gap_audit * 100:.1f}%. "
            "The subcritical verdict is instead reported as a bounded "
            "statement: even the smallest plausible reference gap exceeds the "
            f"gap required for criticality by a factor of {safety:.0f}."),
    }
    json.dump(rep, open(OUT / "celloracle_sensitivity.json", "w"), indent=1)

    print(f"max gap required for criticality : {gap_star_max:.3f}")
    print(f"smallest plausible reference gap : {smallest_ref:.3f}")
    print(f"safety factor                    : {safety:.1f}x")
    print(f"verdict                          : {rep['verdict']}")
    print()
    for r in rows:
        print(f"  {r['tf']:7s} kappa={r['kappa_original']:.5f} "
              f"gap*={r['gap_required_for_critical']:7.3f} "
              f"margin={r['margin_factor_vs_audit_gap']:6.1f}x")


if __name__ == "__main__":
    main()
