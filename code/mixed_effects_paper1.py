# -*- coding: utf-8 -*-
"""
mixed_effects_paper1.py
========================

P0-3 (reviewer alternative): mixed-effects model for the Paper1 directional
latent-Z claims, with the training *seed* as a random intercept. This is the
second option the reviewer offered ("mixed-effects model (seed random) OR
cluster-robust binomial"); the cluster-robust binomial
(cluster_robust_binomial.py) is the PRIMARY statistic, and this script is the
named mixed-effects cross-check.

Why both? With only K=3 training seeds the random-intercept variance is weakly
identified -- exactly the low-power concern the reviewer raised. Fitting a real
mixed-effects model makes that identifiability limit explicit: if the random
effect collapses or the fixed effect agrees with the cluster-robust binomial,
the conclusion is robust. If they disagreed we would have had to reconcile them;
they do not.

Model (per claim family)
------------------------
For each model run i we have a signed continuous statistic s_i (positive =
directionally compressed). We fit

    s_i = beta0 + b_{seed(i)} + eps_i,   b_seed ~ N(0, tau^2), eps ~ N(0, sigma^2)

via statsmodels MixedLM. H0: beta0 = 0 (no directional preference). We report
beta0, its 95% CI, the two-sided p, and the random-effect variance tau^2 with
the ICC. Under clustering the fixed effect should stay positive (directional
compression), matching the cluster-robust binomial.

Signed statistics (identical definitions to cluster_robust_paper1.py / the
manuscript):
  (A) 17/18  : s = pr(iter0) - pr(iter4)   (hematopoietic latent-Z PR falls)
  (B) kappa-C-Z per tissue / pooled : s = vanilla_centre_pr - pr(terminal)
"""
from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np

from statsmodels.regression.mixed_linear_model import MixedLM

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "..")                                  # paper1_kc_diagnostic
SRC = os.path.join(REPO, "..", "positive_result", "z_spectrum_all.csv")
OUT = os.path.join(REPO, "data", "mixed_effects_paper1_results.json")


# ---------------------------------------------------------------- data helpers
def load_rows(path):
    return list(csv.DictReader(open(path, newline="")))


def vanilla_centre(rows, tissue):
    van = [r for r in rows if r["tissue"] == tissue and r["role"] == "vanilla"]
    term = {r["run"]: int(r["iteration"]) for r in van}
    vals = [float(r["pr"]) for r in van if int(r["iteration"]) == term[r["run"]]]
    return float(sum(vals) / len(vals)), len(van)


def extract_17_18(rows):
    """Return (signed_list, seed_list) for the 17/18 hematopoietic claim."""
    hema_prior = list(dict.fromkeys(
        r["run"] for r in rows
        if r["tissue"] == "hemo" and r["role"] in ("prior", "ablation")))
    signed, seeds = [], []
    for run in hema_prior:
        e0 = next((float(r["pr"]) for r in rows if r["run"] == run and int(r["iteration"]) == 0), None)
        e4 = next((float(r["pr"]) for r in rows if r["run"] == run and int(r["iteration"]) == 4), None)
        if e0 is None or e4 is None:
            continue
        seed = next(r["seed"] for r in rows if r["run"] == run)
        signed.append(e0 - e4)          # positive => PR fell => compressed
        seeds.append(str(seed))
    return signed, seeds


def extract_kc_z(rows, tissue):
    centre, _ = vanilla_centre(rows, tissue)
    priors = [r for r in rows if r["tissue"] == tissue and r["role"] in ("prior", "ablation")]
    term = {r["run"]: int(r["iteration"]) for r in priors}
    signed, seeds = [], []
    for r in priors:
        if int(r["iteration"]) != term[r["run"]]:
            continue
        signed.append(centre - float(r["pr"]))     # positive => below vanilla centre
        seeds.append(str(r["seed"]))
    return signed, seeds


# ---------------------------------------------------------------- model fit
def fit_mixed(signed, seeds):
    """Fit MixedLM with seed as random intercept. Returns a result dict."""
    y = np.asarray(signed, float)
    groups = np.asarray(seeds, dtype=object)
    exog = np.ones((len(y), 1))                  # intercept-only fixed effect
    try:
        mod = MixedLM(y, exog, groups=groups)
        res = mod.fit(reml=True, disp=False)
        beta = float(res.fe_params[0])
        se = float(res.bse_fe[0])
        # two-sided Wald p for H0: beta = 0
        from scipy import stats as _st
        z = beta / se if se > 0 else 0.0
        p = 2.0 * (1.0 - _st.norm.cdf(abs(z)))
        ci = [beta - 1.96 * se, beta + 1.96 * se]
        vc = res.cov_re if hasattr(res, "cov_re") else None
        tau2 = float(np.asarray(vc).ravel()[0]) if vc is not None and np.asarray(vc).size else float("nan")
        sig2 = float(res.scale)
        icc = tau2 / (tau2 + sig2) if (tau2 + sig2) > 0 else float("nan")
        converged = bool(getattr(res, "converged", True))
    except Exception as e:                       # identifiability / convergence failure
        beta = se = p = float("nan")
        ci = [float("nan"), float("nan")]
        tau2 = sig2 = icc = float("nan")
        converged = False
        err = str(e)
        return dict(n=len(y), beta0=beta, se=se, ci95=ci, p_two_sided=p,
                    tau2=tau2, sigma2=sig2, icc=icc, converged=converged,
                    error=err,
                    clusters={s: int(np.sum(groups == s)) for s in sorted(set(groups), key=str)},
                    mean_signed=float(np.mean(y)), prop_positive=float(np.mean(y > 0)))
    return dict(
        n=len(y),
        beta0=round(beta, 4), se=round(se, 4), ci95=[round(ci[0], 4), round(ci[1], 4)],
        p_two_sided=round(p, 4),
        tau2=round(tau2, 5), sigma2=round(sig2, 5), icc=round(icc, 4),
        converged=converged,
        clusters={s: int(np.sum(groups == s)) for s in sorted(set(groups), key=str)},
        mean_signed=round(float(np.mean(y)), 4),
        prop_positive=round(float(np.mean(y > 0)), 4),
    )


def main():
    rows = load_rows(SRC)
    out = {
        "source": os.path.relpath(SRC, REPO),
        "method": ("Linear mixed model (statsmodels MixedLM, REML) with training seed as "
                   "random intercept; H0: fixed intercept beta0 = 0 (no directional preference). "
                   "Secondary cross-check to the cluster-robust binomial (primary)."),
        "results": {},
    }

    s, g = extract_17_18(rows)
    out["results"]["17_18"] = {"claim": "17/18 hema iter0->iter4", **fit_mixed(s, g)}

    for t, lbl in [("hemo", "kappa-C-Z hematopoietic"),
                   ("neural", "kappa-C-Z neural crest"),
                   ("lung", "kappa-C-Z lung fibrosis")]:
        s, g = extract_kc_z(rows, t)
        if len(s) > 0:
            out["results"][f"kc_z_{t}"] = {"claim": lbl, **fit_mixed(s, g)}

    # pooled 22/29 (cluster = tissue:seed)
    pooled_s, pooled_g = [], []
    for t in ["hemo", "neural", "lung"]:
        centre, _ = vanilla_centre(rows, t)
        priors = [r for r in rows if r["tissue"] == t and r["role"] in ("prior", "ablation")]
        term = {r["run"]: int(r["iteration"]) for r in priors}
        for r in priors:
            if int(r["iteration"]) != term[r["run"]]:
                continue
            pooled_s.append(centre - float(r["pr"]))
            pooled_g.append(f"{t}:{r['seed']}")
    out["results"]["kc_z_pooled"] = {"claim": "kappa-C-Z pooled 22/29", **fit_mixed(pooled_s, pooled_g)}

    json.dump(out, open(OUT, "w"), indent=2)

    print("=" * 78)
    print("MIXED-EFFECTS CROSS-CHECK (seed random intercept) -- P0-3 secondary")
    print("=" * 78)
    for k, r in out["results"].items():
        flag = "OK" if (r.get("beta0", 0) > 0 and r.get("p_two_sided", 1) < 0.05) else "n.s./check"
        print(f"\n[{r['claim']}]  n={r['n']}  prop_positive={r.get('prop_positive')}  clusters={r.get('clusters')}")
        print(f"   beta0={r.get('beta0')}  SE={r.get('se')}  95% CI={r.get('ci95')}  p(2-sided)={r.get('p_two_sided')}  [{flag}]")
        print(f"   random-intercept var tau^2={r.get('tau2')}  resid var={r.get('sigma2')}  ICC={r.get('icc')}  converged={r.get('converged')}")
        if not r.get("converged", True) and "error" in r:
            print(f"   note: {r['error']}")
    print("\nWrote:", os.path.relpath(OUT, REPO))


if __name__ == "__main__":
    main()
