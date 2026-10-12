# -*- coding: utf-8 -*-
"""S1p -- consolidated verdict for the three cross-architecture pairs (section 2.9).

The pbmc3k control turned out to be two fits of one model, so every reported
cross-architecture pair has to be judged against the scatter of two independent fits
of the same configuration.  This script collects that judgement for all three pairs.

  pbmc3k scVI   : code/s1m_pbmc3k_seed_band.py + s1m4_pbmc3k_injection.json
                  (10 vanilla seeds; the label arm is bit-identical to vanilla, and the
                  prior was then re-registered as a decoder-side covariate)
  Norman scVI   : code/s1n_norman_null.py
                  (3 of this environment's vanilla fits + the published cached one;
                  the published pair's own latents are recomputed from the cache)
  family-4 MLP  : code/s1o_family4_seed_band.py (5 seeds, both arms trained)

Two statistics decide each pair:
  * the published values against the family-matched null of two baseline fits
  * D = kappa-C(Z) - kappa-C(W), paired within seed, against the same difference
    computed over baseline pairs -- the dissociation-relevant quantity

Reads  : s1m2_band_summary.json, s1n_norman_null.json, s1o_family4_seed_band.json
Writes : outputs/positive_result/s1p_crossarch_bands_summary.json
Usage  : python code/s1p_crossarch_band_summary.py
"""
from __future__ import annotations

import itertools
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT = os.path.join(ROOT, "outputs", "positive_result")


def load(n):
    p = os.path.join(OUT, n)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def desc(v):
    v = np.asarray(v, float)
    if v.size == 0:
        return dict(n=0, mean=float("nan"), sd=float("nan"),
                    lo2s=float("nan"), hi2s=float("nan"))
    sd = float(v.std(ddof=1)) if v.size > 1 else float("nan")
    return dict(n=int(v.size), mean=float(v.mean()), sd=sd,
                lo2s=float(v.mean() - 2 * sd), hi2s=float(v.mean() + 2 * sd),
                min=float(v.min()), max=float(v.max()))


def position(x, b):
    z = (x - b["mean"]) / b["sd"] if b.get("sd") else float("nan")
    inside = bool(b["lo2s"] <= x <= b["hi2s"]) if b["n"] else None
    return dict(published=float(x), null_mean=b["mean"], null_sd=b["sd"],
                band=[b["lo2s"], b["hi2s"]], n=b["n"], z=z,
                verdict=("inside" if inside else "OUTSIDE") if inside is not None else "n/a")


def tstat(v):
    v = np.asarray(v, float)
    return float(v.mean() / (v.std(ddof=1) / np.sqrt(v.size))) if v.size > 1 else float("nan")


def rows_of(band, arm="van", inj="inj"):
    return [r for r in band["per_seed"]]


def main():
    out = {}
    print("=" * 100)
    print("cross-architecture pairs -- family-matched seed nulls")
    print("=" * 100)

    # ---------------- pbmc3k (already summarised by s1m2) -------------------
    if (m2 := load("s1m2_band_summary.json")):
        print("\n### pbmc3k scVI (10 vanilla seeds; label arm bit-identical to vanilla)")
        for k, v in m2["positions"].items():
            print(f"  {k:<44} published {v['published']:>8.4f}  null {v['null_mean']:.4f}"
                  f" +- {v['null_sd']:.4f}  z {v['z']:+.2f}  {v['verdict']}")
        out["pbmc3k"] = m2["positions"]

    # ---------------- Norman scVI ------------------------------------------
    if (nm := load("s1n_norman_null.json")):
        print("\n### Norman-2019 scVI (3 own vanilla fits; published latents recomputed)")
        arms = nm["arms"]
        proto = nm.get("protocol", "single")
        # protocol-matched only: a chunked fit is a materially different model (seed 43:
        # 10% readout, 19% effective rank, 142% latent trace away from the single call),
        # so pooled arms would inflate the null.
        mine = [r for r in arms if r["arm"] == "vanilla"
                and r.get("protocol", "single") == proto]
        print(f"  own vanilla eff_rank ({proto} protocol): "
              f"{[round(r['eff_rank'], 3) for r in mine]}"
              f"   (published cached baseline {nm['published']['eff_rank_vanilla']:.3f})")
        out["norman"] = dict(null_mine=nm["null_mine"],
                            null_with_published=nm["null_with_published"],
                            published_z=nm["published_z"],
                            own_vanilla=[{k: r[k] for k in
                                          ("seed", "sum_r2", "trace", "eff_rank", "trace_Z",
                                           "train_status")} for r in mine],
                            published=nm["published"])
        for name, tab in nm["published_z"].items():
            print(f"  -- {name} (n_pairs={nm['null_mine']['kappa_trace']['n'] if 'mine' in name else nm['null_with_published']['kappa_trace']['n']})")
            for k, v in tab.items():
                print(f"     {k:<16} published {v['published']:>8.4f}  null {v['null_mean']:.4f}"
                      f" +- {v['null_sd']:.4f}  z {v['z']:+.2f}  {v['verdict']}")
        # D statistic
        D_pub = nm["published"]["kappa_C_Z"] - nm["published"]["kappa_corrected"]
        members = [(r["trace_Z"], r["sum_r2"]) for r in mine]
        members.append((nm["published"]["van"]["trace_Z"], nm["published"]["van"]["sum_r2"]))
        D_null = np.array([a[0] / b[0] - a[1] / b[1]
                           for a, b in itertools.combinations(members, 2)])
        bD = desc(D_null)
        print(f"  D = kappa-C(Z) - kappa-C(W) published {D_pub:+.4f} | baseline-pair null "
              f"{bD['mean']:+.4f} +- {bD['sd']:.4f}  z "
              f"{(D_pub - bD['mean']) / bD['sd']:+.2f}")
        out["norman"]["D"] = dict(published=D_pub, null=bD,
                                  z=(D_pub - bD["mean"]) / bD["sd"])

    # ---------------- family-4 ---------------------------------------------
    if (f4 := load("s1o_family4_seed_band.json")):
        print("\n### family-4 MLP VAE (5 seeds, both arms trained)")
        paired = f4["paired_conditioned_vs_vanilla"]
        null = f4["cross_seed_vanilla_null"]
        rows = f4["per_seed"]
        pub = f4["published"]
        f4_ratio = pub["eff_rank_conditioned"] / pub["eff_rank_vanilla"]
        print(f"  baseline spread across seeds:")
        for k in ("sum_r2_vanilla", "eff_rank_vanilla", "trace_Z_vanilla"):
            v = np.array([r[k] for r in rows])
            print(f"    {k:<20}{v.min():9.3f} - {v.max():9.3f}  (CV {v.std(ddof=1) / v.mean() * 100:5.1f}%)")
        for k, v in paired.items():
            print(f"  paired {k:<18}{v['mean']:>8.4f} +- {v['sd']:.4f}  [{v['lo2s']:.3f}, {v['hi2s']:.3f}]")
        print(f"  {'published quantity':<36}{'value':>9}{'null mean':>11}{'null sd':>9}{'z':>7}  verdict")
        pos = {}
        for label, x, key in (("kappa-C(W) trace  (0.9474)", pub["kappa_trace"], "kappa_trace"),
                              ("kappa-C(W) corrected (0.9419)", pub["kappa_corrected"], "kappa_corrected"),
                              ("kappa-C(Z)        (0.8902)", pub["kappa_C_Z"], "kappa_C_Z"),
                              ("eff-rank ratio    (0.9717)", f4_ratio, "eff_rank_ratio")):
            b = paired[key]
            pos[label] = position(x, b)
            print(f"  {label:<36}{x:>9.4f}{b['mean']:>11.4f}{b['sd']:>9.4f}"
                  f"{pos[label]['z']:>+7.2f}  {pos[label]['verdict']}")
        Dp = [r["kappa_C_Z"] - r["kappa_corrected"] for r in rows]
        Dn = np.array([ri["trace_Z_vanilla"] / rj["trace_Z_vanilla"]
                       - ri["sum_r2_vanilla"] / rj["sum_r2_vanilla"]
                       for ri, rj in itertools.combinations(rows, 2)])
        bD = desc(Dn)
        print(f"  D = kappa-C(Z) - kappa-C(W): paired {desc(Dp)['mean']:+.4f} +- {desc(Dp)['sd']:.4f}"
              f" (t={tstat(Dp):+.2f}, n={len(Dp)}) | baseline-pair null {bD['mean']:+.4f} +- "
              f"{bD['sd']:.4f}  z {(desc(Dp)['mean'] - bD['mean']) / bD['sd']:+.2f}")
        out["family4"] = dict(paired=paired, null=null, published_positions=pos,
                              published=dict(pub, eff_rank_ratio=f4_ratio),
                              D=dict(paired=desc(Dp), t=tstat(Dp), null=bD,
                                     z=(desc(Dp)["mean"] - bD["mean"]) / bD["sd"]),
                              baseline_cv={k: float(np.array([r[k] for r in rows]).std(ddof=1)
                                                    / np.array([r[k] for r in rows]).mean() * 100)
                                           for k in ("sum_r2_vanilla", "eff_rank_vanilla",
                                                     "trace_Z_vanilla")})

    with open(os.path.join(OUT, "s1p_crossarch_bands_summary.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(f"\nwrote {os.path.join(OUT, 's1p_crossarch_bands_summary.json')}")


if __name__ == "__main__":
    main()
