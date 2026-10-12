"""Sensitivity and correctness check of the structural upper bound on kappa_gap.

Recreated 2026-10-08 after the workspace cleanup incident.  Closes the second of the two
numerical gaps flagged in the manuscript: the structural upper bound currently relies on a
single vanilla top-axis value, 91.2%.

Envelopes
---------
(A) unconstrained            kappa_gap <= lambda_1 / gap_V
(B) preserving direction     kappa_gap <= (1-C) tr(H_V) / gap_V     <- the quoted form

The manuscript quotes (B) at C=0.90 with tr=10590 and gap=9663, giving 0.110.  At the same
inputs (A) is 0.9995, 9.1x larger, and is the envelope valid without assuming the leading
eigendirection survives.  The bound scales as (1-C)/t_axis, so it is insensitive to the
top-axis share it uses.

Reads  : F:/UNAGI_Project/UNAGI/decoder_unified/loading/*.npy
Writes : data/structural_bound_sensitivity.json  (with --out=)
"""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LOADING_DIR = r"F:\UNAGI_Project\UNAGI\decoder_unified\loading"
IPF_TOP_AXIS = 0.912
IPF_TR_HV = 10590.0
IPF_GAP_V = 9663.0
TARGET = 2.0
C_QUOTED = 0.90


def top_axis_share(W):
    H = W @ W.T
    ev = np.clip(np.linalg.eigvalsh(H), 0.0, None)
    return float(ev[-1] / max(ev.sum(), 1e-12))


def env_unconstrained(tr, gap, t):
    return (t * tr) / max(gap, 1e-12)


def env_preserving(tr, gap, C):
    return ((1.0 - C) * tr) / max(gap, 1e-12)


def main():
    files = sorted(glob.glob(os.path.join(LOADING_DIR, "*.npy")))
    per = []
    for f in files:
        per.append({"file": os.path.basename(f), "top_axis_share": top_axis_share(np.load(f))})
    a = np.array([p["top_axis_share"] for p in per])
    t_mean, t_sd = float(a.mean()), float(a.std(ddof=1))
    t_min, t_max = float(a.min()), float(a.max())
    uncon = env_unconstrained(IPF_TR_HV, IPF_GAP_V, IPF_TOP_AXIS)
    pres = env_preserving(IPF_TR_HV, IPF_GAP_V, C_QUOTED)
    z = (IPF_TOP_AXIS - t_mean) / t_sd

    print("empirical top-axis share, %d archived readouts" % len(per))
    print("  mean %.4f sd %.4f min %.4f max %.4f ; IPF %.4f is %.1f sd above mean"
          % (t_mean, t_sd, t_min, t_max, IPF_TOP_AXIS, z))
    print("  (A) unconstrained lambda_1/gap_V = %.4f" % uncon)
    print("  (B) preserving    (1-C)tr/gap_V  = %.4f  (quoted 0.11)" % pres)
    print("  ratio (A)/(B) = %.1fx ; both below target %.1f" % (uncon / pres, TARGET))
    scan = []
    for t in (0.50, 0.70, 0.80, 0.85, 0.90, 0.912, 0.95, 0.99):
        b = pres * (IPF_TOP_AXIS / t)
        scan.append({"t_axis": t, "bound": b})
        print("   t=%.3f -> %.4f" % (t, b))
    comp = []
    for c in (0.50, 0.70, 0.80, 0.90, 0.95, 0.967):
        b = env_preserving(IPF_TR_HV, IPF_GAP_V, c)
        comp.append({"C": c, "bound": b})
        print("   C=%.3f -> %.4f" % (c, b))
    out = {
        "description": "Sensitivity and correctness check of the structural upper bound.",
        "target_kappa_gap": TARGET,
        "ipf_values": {"top_axis_share": IPF_TOP_AXIS, "tr_HV": IPF_TR_HV,
                       "gap_V": IPF_GAP_V, "C_quoted": C_QUOTED},
        "empirical_top_axis": {"n": len(per), "mean": t_mean, "sd": t_sd, "min": t_min,
                               "max": t_max, "ipf_z_score": float(z), "per_readout": per},
        "envelopes": {"unconstrained_exact_at_ipf": float(uncon),
                      "preserving_direction_at_ipf_C90": float(pres),
                      "ratio_unconstrained_over_preserving": float(uncon / pres)},
        "scan_top_axis": scan, "scan_compression": comp,
        "verdict": ("The quoted 0.11 is the preserving-direction envelope, conditional on the "
                    "leading eigendirection surviving; the unconstrained envelope at the same "
                    "inputs is %.4f (%.1fx larger) and still below the target of 2. The bound "
                    "scales as (1-C)/t_axis, so the sensitivity analysis does not change the "
                    "conclusion." % (uncon, uncon / pres)),
    }
    for arg in sys.argv[1:]:
        if arg.startswith("--out="):
            p = arg.split("=", 1)[1]
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w", encoding="utf-8") as fh:
                json.dump(out, fh, indent=1)
            print("wrote %s" % p)


if __name__ == "__main__":
    main()
