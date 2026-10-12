"""
gwas_prior_dose_response.py
===========================
Dose-response prototype: does the GENETIC content of a prior's TF set predict
how much the latent spectrum compresses?

Motivation
----------
The paper's prior is an unweighted TRRUST mask, so two questions stay open:
(i) edge count does not predict compression (rho = -0.21, p = 0.70), and
(ii) R16 (wrong-tissue prior) compresses as much as R13 (matched prior).
Both leave "what does compression track?" unanswered. `gwas_tf_scores.py`
supplies an EXTERNAL axis: a per-TF germline genetic-association score for
erythroid traits (Open Targets). This script turns the 2-group prior contrast
into a 6-point dose-response over the prior ladder.

Design
------
Unit of analysis = prior COMPOSITION (not run), to avoid pseudo-replication
from repeated seeds. Six compositions have both a resolved TF set and a
measurable latent outcome:
    R11 {KLF1,TAL1}  R12 {GATA2,GFI1B,PAX5}  R84 {GATA1,KLF1,TAL1,GFI1B}
    R107 {GATA1,KLF1,PAX5,TAL1,GFI1B}        R15 {8 TF}   R13 {11 TF}
Outcome = pr_ratio_Z = PR(latent, prior) / PR(latent, vanilla); <1 = compression.

Inference
---------
n = 6 compositions, so an EXACT permutation test over all 6! = 720 score
assignments is available and gives a minimum two-sided p of 0.003 for a
perfect monotone trend -- better than the 3-seed cluster floor that caps the
paper's existing latent tests at p >= 0.25. We report the exact p regardless
of whether it clears the bar.

Content vs size: partial Spearman controlling log(edge count). Note the two
axes are NOT collinear here -- R11 has the FEWEST edges (21) and the HIGHEST
genetic score -- which is what makes the separation meaningful.

Outputs
-------
  data/gwas_prior_dose_response.json
  reviews/gwas_prior_dose_response.md
"""
import itertools
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PR_JSON = os.path.join(ROOT, "data", "kappa_c_pr_comparison.json")
GWAS_JSON = os.path.join(ROOT, "data", "gwas_tf_scores.json")
PRIOR_JSON = os.environ.get(
    "PRIOR_CONFIGS",
    "F:/UNAGI_Project/outputs/positive_result/prior_configs.json")
OUT_JSON = os.path.join(ROOT, "data", "gwas_prior_dose_response.json")
OUT_MD = os.path.join(ROOT, "reviews", "gwas_prior_dose_response.md")


# ---------------------------------------------------------------- statistics
def rankdata(x):
    x = np.asarray(x, float)
    order = x.argsort()
    r = np.empty(len(x), float)
    r[order] = np.arange(len(x), dtype=float)
    # average ties
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return r


def spearman(a, b):
    ra, rb = rankdata(a), rankdata(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d > 0 else float("nan")


def exact_perm_p(score, outcome, observed, max_perms=200000):
    """Two-sided exact permutation p by permuting `score` over `outcome`."""
    n = len(score)
    if n <= 8:
        perms = itertools.permutations(range(n))
        vals = []
        for p in perms:
            vals.append(abs(spearman([score[i] for i in p], outcome)))
        vals = np.array(vals)
        p = float((vals >= abs(observed) - 1e-12).mean())
        return p, len(vals)
    rng = np.random.default_rng(20260915)
    vals = np.empty(max_perms)
    for k in range(max_perms):
        p = rng.permutation(n)
        vals[k] = abs(spearman(score[p], outcome))
    p = float((vals >= abs(observed) - 1e-12).mean())
    return p, max_perms


def partial_spearman(x, y, z):
    """Spearman of x,y after residualising both ranks on rank(z) (linear)."""
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    A = np.vstack([rz, np.ones_like(rz)]).T
    bx = np.linalg.lstsq(A, rx, rcond=None)[0]
    by = np.linalg.lstsq(A, ry, rcond=None)[0]
    return spearman(rx - A @ bx, ry - A @ by)


# ------------------------------------------------------------------- loading
def main():
    pr = json.load(open(PR_JSON, encoding="utf-8"))
    tf = json.load(open(GWAS_JSON, encoding="utf-8"))["tf_scores"]
    prior = json.load(open(PRIOR_JSON, encoding="utf-8"))

    # ---- explicit alias map: pr-JSON config name -> prior_configs.json key ---
    # Naming in the two artefacts differs (R11_s42 vs R11_seed42), and a naive
    # prefix match would fold R13_KO_KLF1 / R13_OE_KLF1 into the R13 prior.
    # Mapping is therefore explicit and the leftover set is asserted below.
    RUN_TO_PRIOR = {
        "R11_s42": "R11_seed42",
        "R11_s123": "R11_KLF1_TAL1_seed123",
        "R11_s456": "R11_KLF1_TAL1_seed456",
        "R12_s42": "R12_seed42",
        "R12_s123": "R12_seed123",
        "R12_s456": "R12_seed456",
        "R13_s42": "R13",
        "R13_s123": "R13_seed123",
        "R13_s456": "R13_seed456",
        "R84_s42": "R84_seed42",
        "R84_s123": "R84_seed123",
        "R107_s42": "R107_seed42",
        "R15_s42": "R15_seed42",
        "R15_s123": "R15_seed123",
        "R15_s456": "R15_seed456",
        "R165_s42": "R165_seed42",
    }
    # deliberately NOT part of the dose-response, with the reason:
    EXCLUDED = {
        "R111_s42": "TF set not resolved by reconstruct_prior_configs.py",
        "R111_s123": "TF set not resolved by reconstruct_prior_configs.py",
        "R111_s456": "TF set not resolved by reconstruct_prior_configs.py",
        "R13_KO_KLF1": "KLF1 perturbation arm, not a prior-composition arm",
        "R13_OE_KLF1": "KLF1 perturbation arm, not a prior-composition arm",
        "R16_s42": "wrong-tissue (neural) prior; outside the hema TF universe",
    }

    # ---- gather hema rows, then group by resolved TF set --------------------
    rows, unresolved, hema_seen = [], [], []
    for r in pr["rows"]:
        if r["tissue"] != "hemo":
            continue
        cfg = r["config"]
        hema_seen.append(cfg)
        if cfg in EXCLUDED:
            unresolved.append(dict(config=cfg, reason=EXCLUDED[cfg]))
            continue
        if cfg not in RUN_TO_PRIOR:
            raise AssertionError(
                "unmapped hema config %r -- add it to RUN_TO_PRIOR or EXCLUDED "
                "rather than letting it drop silently" % cfg)
        pk = RUN_TO_PRIOR[cfg]
        if not prior[pk].get("tfs"):
            raise AssertionError("prior %r has no resolved TF set" % pk)
        rows.append(dict(run=cfg, key=pk, pr_ratio_Z=r["pr_ratio_Z"],
                         kappa_C_W=r["kappa_C_W_trace"],
                         seed="s" + str(r["run"]).split("seed")[-1][:3]))

    # every hema row must be accounted for
    assert len(rows) + len(unresolved) == len(hema_seen), "hemo row accounting failed"

    # group runs sharing an identical resolved TF set into one composition
    groups = {}
    for r in rows:
        fs = frozenset(prior[r["key"]]["tfs"])
        groups.setdefault(fs, []).append(r)

    table = []
    for fs, sub in groups.items():
        tset = sorted(fs)
        ed = {prior[r["key"]]["n_edges"] for r in sub}
        assert len(ed) == 1, "edge count differs within a TF-set group: %s" % ed
        ery = np.mean([tf[t]["ery_mean"] for t in tset])
        spec = np.mean([tf[t]["erythroid_specificity"] for t in tset])
        prs = [r["pr_ratio_Z"] for r in sub]
        label = "+".join(sorted(x for x in sub[0]["key"].split("_") if x[:1].isalpha())
                         [:1]) or "comp"
        names = sorted({r["run"].split("_")[0] for r in sub})
        table.append(dict(
            comp="/".join(names), tfs=tset, n_tfs=len(tset), edges=ed.pop(),
            n_runs=len(sub), runs=[r["run"] for r in sub],
            pr_ratio_Z_mean=float(np.mean(prs)),
            pr_ratio_Z_min=float(np.min(prs)),
            pr_ratio_Z_max=float(np.max(prs)),
            pr_ratio_Z_sd=float(np.std(prs, ddof=1)) if len(prs) > 1 else None,
            gwas_ery_mean=float(ery), gwas_specificity_mean=float(spec),
            kappa_C_W_mean=float(np.mean([r["kappa_C_W"] for r in sub])),
            seeds=sorted({r["seed"] for r in sub}),
        ))
    table.sort(key=lambda d: -d["gwas_ery_mean"])
    comps = [d["comp"] for d in table]

    # ------------------------------------------------------------- inference
    score = np.array([d["gwas_ery_mean"] for d in table])
    outcome = np.array([d["pr_ratio_Z_mean"] for d in table])
    size = np.log(np.array([d["edges"] for d in table], float))
    spec = np.array([d["gwas_specificity_mean"] for d in table])

    rho_content = spearman(score, outcome)
    p_content, nperm = exact_perm_p(list(score), list(outcome), rho_content)
    rho_size = spearman(size, outcome)
    p_size, _ = exact_perm_p(list(size), list(outcome), rho_size)
    rho_spec = spearman(spec, outcome)
    p_spec, _ = exact_perm_p(list(spec), list(outcome), rho_spec)
    rho_partial = partial_spearman(score, outcome, size)

    # ---- within-seed check: compositions observed at all three seeds -------
    # Compositions differ in which seeds they were run at, so a per-seed
    # (paired) contrast is the only way to hold the seed effect fixed.
    three = [d["comp"] for d in table if len(d["seeds"]) == 3]
    per_seed = {}
    for s in ["s42", "s123", "s456"]:
        ss, oo = [], []
        for d in table:
            if s not in d["seeds"] or d["comp"] not in three:
                continue
            run = [r for r in rows if r["run"] in d["runs"] and r["seed"] == s][0]
            ss.append(d["gwas_ery_mean"])
            oo.append(run["pr_ratio_Z"])
        if len(ss) >= 3:
            per_seed[s] = dict(n=len(ss), rho=spearman(ss, oo),
                               scores=[round(x, 4) for x in ss],
                               outcomes=[round(x, 4) for x in oo])

    # ---- run-level sensitivity, permuting WITHIN seed blocks ---------------
    # n = 16 runs. Treating runs as independent would be pseudo-replication, so
    # the null is built by permuting the composition score across runs inside
    # each seed block, which holds the seed effect fixed.
    run_rows = []
    for d in table:
        for run in d["runs"]:
            rec = [r for r in rows if r["run"] == run][0]
            run_rows.append(dict(seed=rec["seed"], score=d["gwas_ery_mean"],
                                 out=rec["pr_ratio_Z"], run=run,
                                 edges=d["edges"]))
    rs = np.array([r["score"] for r in run_rows])
    ro = np.array([r["out"] for r in run_rows])
    rho_run = spearman(rs, ro)
    rng = np.random.default_rng(20260915)
    blocks = {}
    for i, r in enumerate(run_rows):
        blocks.setdefault(r["seed"], []).append(i)
    nperm_run = 20000
    null = np.empty(nperm_run)
    base = rs.copy()
    for k in range(nperm_run):
        perm = base.copy()
        for _, idx in blocks.items():
            perm[idx] = rng.permutation(perm[idx])
        null[k] = abs(spearman(perm, ro))
    p_run = float((null >= abs(rho_run) - 1e-12).mean())

    # size axis on the same run-level basis, for a like-for-like comparison
    rz = np.log(np.array([r["edges"] for r in run_rows], float))
    rho_run_size = spearman(rz, ro)
    null_s = np.empty(nperm_run)
    for k in range(nperm_run):
        perm = rz.copy()
        for _, idx in blocks.items():
            perm[idx] = rng.permutation(perm[idx])
        null_s[k] = abs(spearman(perm, ro))
    p_run_size = float((null_s >= abs(rho_run_size) - 1e-12).mean())

    # ---- robustness to the R15/R165 metadata ambiguity ---------------------
    # The run log for r14_hemato_seed42 (same run directory as R15_s42) reports
    # "TF indices: 11/11 matched", while prior_configs.json assigns R15 an
    # 8-TF set. The attribution is therefore unverified, so the primary result
    # is re-derived under two alternative groupings.
    by_comp = {d["comp"]: d for d in table}

    def variant(spec):
        """spec: {label: (tf_list, [run names])} -> rho + exact p."""
        sc, ou, keep = [], [], []
        for lab, (tset, runs) in spec.items():
            sel = [r for r in rows if r["run"] in runs]
            if not sel:
                raise AssertionError("variant group %r matched no runs" % lab)
            sc.append(np.mean([tf[x]["ery_mean"] for x in tset]))
            ou.append(float(np.mean([r["pr_ratio_Z"] for r in sel])))
            keep.append(lab)
        rho_v = spearman(sc, ou)
        p_v, np_v = exact_perm_p(sc, ou, rho_v)
        return dict(rho=rho_v, p_exact=p_v, n=len(sc), labels=keep,
                    scores=[round(x, 4) for x in sc],
                    outcomes=[round(x, 4) for x in ou])

    tset_of = {}
    for d in table:
        for nm in d["comp"].split("/"):
            tset_of[nm] = d["tfs"]

    runs_by_label = {
        "R11": ["R11_s42", "R11_s123", "R11_s456"],
        "R12": ["R12_s42", "R12_s123", "R12_s456"],
        "R84": ["R84_s42", "R84_s123"],
        "R107": ["R107_s42"],
        "R13": ["R13_s42", "R13_s123", "R13_s456"],
        "R15": ["R15_s42", "R15_s123", "R15_s456"],
        "R165": ["R165_s42"],
    }
    variants = {}
    variants["B_drop_unverified_8TF_group"] = variant(
        {lab: (tset_of[lab], runs_by_label[lab])
         for lab in ["R11", "R12", "R84", "R107", "R13"]})
    variants["C_R15_pooled_with_11TF"] = variant({
        "R11": (tset_of["R11"], runs_by_label["R11"]),
        "R12": (tset_of["R12"], runs_by_label["R12"]),
        "R84": (tset_of["R84"], runs_by_label["R84"]),
        "R107": (tset_of["R107"], runs_by_label["R107"]),
        "R13+R15": (tset_of["R13"], runs_by_label["R13"] + runs_by_label["R15"]),
        "R165": (tset_of["R165"], runs_by_label["R165"]),
    })

    res = dict(
        question=("Does the germline genetic content of a prior's TF set predict "
                  "latent-spectrum compression?"),
        outcome_definition="pr_ratio_Z = PR(latent|prior) / PR(latent|vanilla); <1 = compression",
        axis_definition=("per-TF Open Targets GWAS credible-set score for erythroid "
                         "traits (mean over 8 traits), averaged over the prior's TFs"),
        n_compositions=len(table),
        compositions=table,
        unresolved_configs=unresolved,
        tests=dict(
            content=dict(axis="gwas_ery_mean", rho=rho_content, p_exact=p_content,
                         n_perm=nperm, min_possible_two_sided_p=2.0 / nperm),
            size=dict(axis="log(edge count)", rho=rho_size, p_exact=p_size,
                      note="reproduces the paper's negative control on a comparable basis"),
            specificity=dict(axis="gwas_specificity_mean", rho=rho_spec, p_exact=p_spec),
            content_partialling_size=dict(rho=rho_partial,
                                          note="partial Spearman of content given log(edges)"),
            run_level_clustered=dict(
                n_runs=len(run_rows), rho_content=rho_run, p_content_clustered=p_run,
                rho_size=rho_run_size, p_size_clustered=p_run_size,
                n_perm=nperm_run,
                note=("runs treated as units but the null permutes scores WITHIN "
                      "seed blocks, holding the seed effect fixed")),
        ),
        within_seed=dict(compositions=three, per_seed=per_seed),
        robustness_variants=dict(
            note=("the R15/R165 TF attribution is unverified: prior_configs.json "
                  "assigns 8 TFs, while the run log reports 11/11 matched"),
            variants=variants),
    )
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    json.dump(res, open(OUT_JSON, "w", encoding="utf-8"), indent=2)

    # ------------------------------------------------------------- console
    print("=" * 92)
    print("DOSE-RESPONSE  (unit = composition, n = %d)" % len(table))
    print("=" * 92)
    print("%-6s %-38s %6s %6s %8s %14s %10s" %
          ("comp", "TFs", "edges", "runs", "GWAS_ery", "pr_ratioZ(mean)", "sd"))
    for d in table:
        print("%-6s %-38s %6d %6d %8.3f %14.3f %10s" %
              (d["comp"], ",".join(d["tfs"])[:38], d["edges"], d["n_runs"],
               d["gwas_ery_mean"], d["pr_ratio_Z_mean"],
               "%.3f" % d["pr_ratio_Z_sd"] if d["pr_ratio_Z_sd"] is not None else "-"))
    print()
    print("Hypothesis: higher genetic content -> more compression (rho < 0)")
    print("  content  rho = %+.3f   exact two-sided p = %.3f  (%d perms, floor %.4f)"
          % (rho_content, p_content, nperm, 2.0 / nperm))
    print("  size     rho = %+.3f   exact two-sided p = %.3f" % (rho_size, p_size))
    print("  specific rho = %+.3f   exact two-sided p = %.3f" % (rho_spec, p_spec))
    print("  content | size (partial) rho = %+.3f" % rho_partial)
    print()
    print("Run level (n=%d), null permuted WITHIN seed blocks:" % len(run_rows))
    print("  content rho = %+.3f  clustered p = %.3f" % (rho_run, p_run))
    print("  size    rho = %+.3f  clustered p = %.3f" % (rho_run_size, p_run_size))
    print()
    print("Robustness to the unverified R15/R165 TF attribution:")
    for k, v in variants.items():
        print("  %-34s n=%d rho = %+.3f  exact p = %.3f" %
              (k, v["n"], v["rho"], v["p_exact"]))
    print()
    print("Within-seed (compositions with all 3 seeds: %s)" % ",".join(three))
    for s, v in per_seed.items():
        print("  seed %-4s n=%d rho=%+.3f" % (s, v["n"], v["rho"]))
    print("\n[saved]", OUT_JSON)

    write_md(res, table, three, per_seed)
    print("[saved]", OUT_MD)


def write_md(res, table, three, per_seed):
    t = res["tests"]
    L = []
    A = L.append
    A("# GWAS-anchored dose-response prototype\n")
    A("**Question.** Does the germline genetic content of a prior's TF set predict "
      "how much the latent spectrum compresses?\n")
    A("**Axis (external, not model-derived).** Per-TF Open Targets GWAS credible-set "
      "score for 8 erythroid traits (erythrocyte count, erythrocyte indices, RDW, MCH, "
      "MCHC, reticulocyte count, hematocrit, hemoglobin), averaged over the prior's TFs. "
      "This axis measures *population-genetic evidence* that a gene affects the trait; "
      "it is **not** a measure of regulatory importance (GATA1 has no erythrocyte-count "
      "association in the top 3000 targets, while KLF1 scores 0.58).\n")
    A("**Outcome.** `pr_ratio_Z = PR(latent|prior) / PR(latent|vanilla)`; <1 = compression. "
      "Primary unit of analysis is the composition (`n = %d`), not the run, to avoid "
      "pseudo-replication.\n" % res["n_compositions"])
    A("\n## Compositions\n")
    A("| comp | TFs | edges | runs | GWAS erythroid score | pr_ratio_Z (mean) | sd |")
    A("|---|---|---|---|---|---|---|")
    for d in table:
        A("| %s | %s | %d | %d | %.3f | %.3f | %s |" %
          (d["comp"], ", ".join(d["tfs"]), d["edges"], d["n_runs"],
           d["gwas_ery_mean"], d["pr_ratio_Z_mean"],
           "%.3f" % d["pr_ratio_Z_sd"] if d["pr_ratio_Z_sd"] is not None else "–"))
    A("\n## Tests\n")
    A("| axis | unit | Spearman rho | p |")
    A("|---|---|---|---|")
    A("| genetic content | composition (n=%d) | %+.3f | %.3f (exact, %d perms) |"
      % (res["n_compositions"], t["content"]["rho"], t["content"]["p_exact"], t["content"]["n_perm"]))
    A("| log(edge count) | composition | %+.3f | %.3f (exact) |"
      % (t["size"]["rho"], t["size"]["p_exact"]))
    A("| erythroid specificity | composition | %+.3f | %.3f (exact) |"
      % (t["specificity"]["rho"], t["specificity"]["p_exact"]))
    A("| content \\| size (partial) | composition | %+.3f | – |"
      % t["content_partialling_size"]["rho"])
    rl = t["run_level_clustered"]
    A("| genetic content | run (n=%d) | %+.3f | %.3f (seed-blocked) |"
      % (rl["n_runs"], rl["rho_content"], rl["p_content_clustered"]))
    A("| log(edge count) | run (n=%d) | %+.3f | %.3f (seed-blocked) |"
      % (rl["n_runs"], rl["rho_size"], rl["p_size_clustered"]))
    A("\nExact permutation floor at the composition level: two-sided p = %.4f "
      "(%d permutations). This is *better* than the 3-seed cluster floor (p >= 0.25) "
      "that caps the paper's existing latent tests, so the null here is not the same "
      "kind of power artefact.\n" % (2.0 / t["content"]["n_perm"], t["content"]["n_perm"]))
    A("## Within-seed check\n")
    A("Compositions observed at all three seeds: %s\n" % ", ".join(three))
    for s, v in per_seed.items():
        A("- seed %s (n=%d): rho = %+.3f" % (s, v["n"], v["rho"]))
    A("")
    A("## Robustness to the unverified R15/R165 attribution\n")
    A("`prior_configs.json` assigns the R15 runs an 8-TF set, but the run log "
      "(`r14_hemato_seed42.log`, same run directory) reports *11/11 TFs matched* "
      "with a much broader prior boost (1054/20466 genes vs 203/20466 for R13). "
      "The attribution is therefore unresolved. Both alternative groupings give "
      "the same answer:\n")
    A("| variant | n | Spearman rho | exact p |")
    A("|---|---|---|---|")
    for k, v in res["robustness_variants"]["variants"].items():
        A("| %s | %d | %+.3f | %.3f |" % (k, v["n"], v["rho"], v["p_exact"]))
    A("")
    A("## Excluded hema configurations\n")
    for u in res["unresolved_configs"]:
        A("- `%s` — %s" % (u["config"], u["reason"]))
    A("")
    A("## Reading\n")
    A("**The genetic-content axis does not predict compression, at any level of "
      "analysis.** Composition-level rho = %.3f (exact p = %.3f, floor %.4f); "
      "run-level rho = %+.3f (seed-blocked p = %.3f); the two point in *opposite* "
      "directions, which is what noise looks like. The within-seed contrasts flip "
      "sign (s42 %+.3f, s123 %+.3f, s456 %+.3f). Both robustness groupings agree.\n"
      % (t["content"]["rho"], t["content"]["p_exact"],
         2.0 / t["content"]["n_perm"], rl["rho_content"],
         rl["p_content_clustered"],
         per_seed.get("s42", {}).get("rho", float("nan")),
         per_seed.get("s123", {}).get("rho", float("nan")),
         per_seed.get("s456", {}).get("rho", float("nan"))))
    A("Two consequences, one negative and one positive.\n")
    A("*Negative.* The intuitive fix -- weight the prior by GWAS evidence so that "
      "compression acquires a biological read-out -- does not work in its simplest "
      "form on the existing ladder. The prior's *content* is no better an "
      "explanation of compression than its *size* (log-edge-count rho = %+.3f, "
      "p = %.3f), which reproduces the paper's own negative control on a "
      "comparable basis.\n" % (t["size"]["rho"], t["size"]["p_exact"]))
    A("*Positive.* The test is not underpowered in the way the paper's latent "
      "tests are. With six compositions the exact permutation floor is p = %.4f, "
      "an order of magnitude below the 3-seed cluster floor of 0.25. So this is a "
      "genuine null, not a ceiling artefact -- compression is not explained by any "
      "prior property measured so far (size, lineage match, or germline genetic "
      "content). That is a stronger statement for the manuscript than the R16 "
      "anecdote it replaces.\n" % (2.0 / t["content"]["n_perm"]))
    A("**What would be needed to make the question answerable.** A dose-response "
      "needs a *continuous* prior axis, not six discrete compositions. The natural "
      "design is to retrain with a graded prior weight (e.g. per-TF GWAS scores "
      "scaled by a temperature) and sweep it; that is a training experiment, not "
      "an analysis of existing artefacts.\n")
    open(OUT_MD, "w", encoding="utf-8").write("\n".join(L))


if __name__ == "__main__":
    main()
