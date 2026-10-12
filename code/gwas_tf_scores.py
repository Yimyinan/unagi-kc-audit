"""
gwas_tf_scores.py
=================
Build an EXTERNAL, genetically anchored relevance score for each of the 11
hematopoietic TFs used by the PG-UNAGI prior ladder.

Why
---
The submission's prior is a binary/unweighted TRRUST mask. Two awkward results
follow: (i) R16 (a wrong-tissue neural prior) compresses as much as R13 (the
matched blood prior), and (ii) edge count does not predict compression
(rho = -0.21, p = 0.70). Both say "prior content is not decoded into the
spectrum". A GWAS-anchored axis lets us ask that question *quantitatively*:
does the genetic evidence each TF carries for erythroid traits track how much
the latent spectrum compresses?

Data source
-----------
Open Targets Platform, GraphQL v4 (https://api.platform.opentargets.org).
We query `disease(efoId).associatedTargets`, i.e. the gene-level aggregation of
GWAS credible-set evidence for that trait. This is the direction that carries
the quantitative blood-cell traits; `target(...).associatedDiseases` does not
surface them.

IMPORTANT naming caveat (carried into the report): this axis measures
POPULATION-GENETIC evidence that a gene affects the trait. It is NOT "regulatory
importance". KLF1 scores highest for erythrocyte count (0.58) while GATA1 --
the erythroid master regulator -- has no erythrocyte-count association in the
top 3000. Reporting the axis as "erythroid master-regulator importance" would
be wrong; the report names it precisely.

Outputs
-------
  <scratch>/opentargets/<EFO>.json      raw cached responses (provenance)
  data/gwas_tf_scores.json              per-TF erythroid / non-erythroid scores
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SCRATCH = os.environ.get(
    "GWAS_SCRATCH",
    "F:/UNAGI_Project/_scratch_gwas")
CACHE = os.path.join(SCRATCH, "opentargets")
OUT = os.path.join(ROOT, "data", "gwas_tf_scores.json")
OT = "https://api.platform.opentargets.org/api/v4/graphql"
PAGE = 3000

# ---- the 11 TFs of the PG-UNAGI hematopoietic prior ladder ------------------
TF_ENSEMBL = {
    "GATA2": "ENSG00000179348",
    "GATA1": "ENSG00000102145",
    "KLF1":  "ENSG00000105610",
    "SPI1":  "ENSG00000066336",
    "CEBPA": "ENSG00000245848",
    "PAX5":  "ENSG00000196092",
    "RUNX1": "ENSG00000159216",
    "TAL1":  "ENSG00000162367",
    "LMO2":  "ENSG00000135363",
    "GFI1":  "ENSG00000162676",
    "GFI1B": "ENSG00000165702",
}

# ---- trait sets ------------------------------------------------------------
# Erythroid: quantitative red-cell traits.
ERYTHROID = {
    "EFO_0004305": "erythrocyte count",
    "EFO_0004306": "erythrocyte indices",
    "EFO_0009188": "red cell distribution width",
    "EFO_0004527": "mean corpuscular hemoglobin",
    "EFO_0004528": "mean corpuscular hemoglobin concentration",
    "EFO_0007986": "reticulocyte count",
    "EFO_0004348": "hematocrit",
    "EFO_0004509": "hemoglobin measurement",
}
# Non-erythroid blood control: same tissue compartment, other lineages.
NON_ERYTHROID = {
    "EFO_0004309": "platelet count",
    "EFO_0004833": "neutrophil count",
    "EFO_0005091": "monocyte count",
    "EFO_0004587": "lymphocyte count",
    "EFO_0004842": "eosinophil count",
    "EFO_0005090": "basophil count",
}


def gql(query, tries=4):
    payload = json.dumps({"query": query})
    last = None
    for k in range(tries):
        p = subprocess.run(
            ["curl", "-s", "-m", "120", "-X", "POST", OT,
             "-H", "Content-Type: application/json", "-d", payload],
            capture_output=True, text=True)
        try:
            d = json.loads(p.stdout)
            if d.get("data"):
                return d
            last = p.stdout[:200]
        except Exception as e:  # noqa: BLE001
            last = f"{type(e).__name__}: {p.stdout[:160]}"
        time.sleep(1.5 * (k + 1))
    raise RuntimeError(f"Open Targets query failed: {last}")


def fetch_trait(efo, name, force=False):
    """associatedTargets for one trait -> {symbol: score}, with raw cache."""
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"{efo}.json")
    if os.path.exists(path) and not force:
        d = json.load(open(path, encoding="utf-8"))
    else:
        q = ('{ disease(efoId:"%s"){ name associatedTargets(page:{index:0,size:%d})'
             '{ count rows{ target{approvedSymbol} score } } } }' % (efo, PAGE))
        d = gql(q)
        json.dump(d, open(path, "w", encoding="utf-8"))
        time.sleep(0.4)
    at = d["data"]["disease"]["associatedTargets"]
    return ({r["target"]["approvedSymbol"]: float(r["score"]) for r in at["rows"]},
            int(at["count"]))


def main():
    scores = {}
    meta = {}
    for efo, name in list(ERYTHROID.items()) + list(NON_ERYTHROID.items()):
        sym, total = fetch_trait(efo, name)
        cls = "erythroid" if efo in ERYTHROID else "non_erythroid"
        print(f"{efo:14s} {name:46s} {cls:14s} n_targets={total:6d}")
        for tf in TF_ENSEMBL:
            scores.setdefault(tf, {})[efo] = sym.get(tf, 0.0)
        meta[efo] = dict(name=name, cls=cls, n_targets=total)

    table = {}
    for tf, per in scores.items():
        ery = [per[e] for e in ERYTHROID]
        non = [per[e] for e in NON_ERYTHROID]
        table[tf] = dict(
            ensembl=TF_ENSEMBL[tf],
            per_trait={k: round(v, 4) for k, v in per.items()},
            ery_max=round(max(ery), 4),
            ery_mean=round(sum(ery) / len(ery), 4),
            ery_n_traits_hit=int(sum(1 for v in ery if v > 0)),
            non_max=round(max(non), 4),
            non_mean=round(sum(non) / len(non), 4),
            non_n_traits_hit=int(sum(1 for v in non if v > 0)),
            # specificity: erythroid evidence above non-erythroid baseline
            erythroid_specificity=round(max(ery) - max(non), 4),
        )

    out = dict(
        source="Open Targets Platform GraphQL v4, disease.associatedTargets",
        retrieved=time.strftime("%Y-%m-%d"),
        axis_definition=("gene-level aggregation of GWAS credible-set evidence "
                         "for the trait; measures population-genetic evidence "
                         "for a gene-trait effect, NOT regulatory importance"),
        page_size=PAGE,
        traits=meta,
        tf_scores=table,
    )
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=2)

    print("\n%-7s %7s %7s %6s | %7s %7s %6s | %8s" %
          ("TF", "ery_max", "ery_mean", "n_ery", "non_max", "non_mean", "n_non", "spec"))
    for tf in TF_ENSEMBL:
        t = table[tf]
        print("%-7s %7.3f %7.3f %6d | %7.3f %7.3f %6d | %+8.3f" %
              (tf, t["ery_max"], t["ery_mean"], t["ery_n_traits_hit"],
               t["non_max"], t["non_mean"], t["non_n_traits_hit"],
               t["erythroid_specificity"]))
    print("\n[saved]", OUT)


if __name__ == "__main__":
    main()
