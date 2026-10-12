"""
prior_injection_chain_audit.py
==============================
Audit the *implementation* of the TF-prior injection path, not just its
statistical footprint.

Motivation
----------
Paper 1 measures what the prior does to the decoder readout spectrum
(kappa-C(W)) and to the latent covariance. It treats "prior injected" as an
experimental condition. That is a statement about the *configuration*, not
about the *computation*. Before proposing new biological knowledge to inject,
we should establish what the existing injection actually touches.

This script traces the injection chain end to end:

    tf_database.build_prior_matrix()          -> gene-level boost vector w_g
      -> CustomUNAGIWrapper.setup_training()  -> self.unagi._tf_prior_weights
        -> ??? (who consumes it?)             -> ???

and checks the recorded training artefacts for the fingerprint of a
per-gene weighting of the likelihood.

Diagnostics
-----------
D1  geneWeight layer occupancy across stagedata/*.h5ad
D2  source-level call graph for _boost_tf_weights / geneWeight
D3  patcher regex collateral damage (syntax validity + hit count)
D4  reachability of the likelihood-weighting branch in customized_elbo.py

Outputs
-------
  outputs/positive_result/INJECTION_CHAIN_AUDIT.md
  outputs/positive_result/injection_chain_audit.json
"""
import json
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path
import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from provenance import attach as _attach_provenance

REPO = Path("F:/UNAGI_Project")
OUT = REPO / "outputs" / "positive_result"
F = Path("F:/UNAGI_Project")

# Candidate source trees, in the order they matter. `b1_expansion` is the tree
# the pg_unagi patcher actually rewrites; `UNAGI/UNAGI_source` is the pristine
# upstream copy.
TREES = {
    "b1_expansion (patched)": F / "b1_expansion/UNAGI_source",
    "UNAGI_source (pristine)": F / "UNAGI/UNAGI_source",
}

STAGEDATA_GLOBS = [
    "UNAGI/benchmark/*/iter_*/stagedata/*.h5ad",
]


# --------------------------------------------------------------------------
# D1: geneWeight occupancy in recorded training artefacts
# --------------------------------------------------------------------------
def d1_geneWeight_occupancy():
    try:
        import anndata as ad
        import numpy as np
    except ImportError:
        return {"status": "skipped", "reason": "anndata unavailable"}

    files = []
    for g in STAGEDATA_GLOBS:
        files += sorted(F.glob(g))
    if not files:
        return {"status": "skipped", "reason": "no stagedata found"}

    rows, nonzero = [], 0
    for f in files:
        try:
            a = ad.read_h5ad(f)
        except Exception as e:  # noqa: BLE001
            rows.append({"file": str(f), "error": str(e)[:120]})
            continue
        rec = {"file": str(f.relative_to(F)), "shape": list(a.shape)}
        if "geneWeight" in a.layers:
            g = a.layers["geneWeight"]
            rec.update(has_layer=True, nnz=int(g.nnz))
            if g.nnz > 0:
                nonzero += 1
                d = np.asarray(g.todense())
                rec["max"] = float(d.max())
                rec["n_cols_touched"] = int((d != 0).any(0).sum())
        else:
            rec.update(has_layer=False, nnz=None)
        rows.append(rec)

    n = len([r for r in rows if r.get("nnz") is not None])
    return {
        "status": "ran",
        "n_files": len(files),
        "n_with_layer": n,
        "n_with_any_nonzero": nonzero,
        "verdict": ("all-zero" if n and nonzero == 0 else
                    "populated" if nonzero else "undetermined"),
        "rows": rows,
    }


# --------------------------------------------------------------------------
# D2: source-level call graph
# --------------------------------------------------------------------------
def d2_call_graph():
    out = {}
    for label, root in TREES.items():
        if not root.exists():
            out[label] = {"status": "missing"}
            continue
        rec = {"status": "ran", "definitions": [], "call_sites": [],
               "geneWeight_uses": []}
        for py in root.rglob("*.py"):
            if "__pycache__" in str(py):
                continue
            try:
                txt = py.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            rel = str(py.relative_to(root))
            for m in re.finditer(r"def\s+(_boost_tf_weights|_apply_tf_prior_weights)", txt):
                rec["definitions"].append({"file": rel, "symbol": m.group(1)})
            # call sites EXCLUDING the definition line
            for m in re.finditer(r"self\.(_boost_tf_weights|_apply_tf_prior_weights)\s*\(", txt):
                line = txt[:m.start()].count("\n") + 1
                src = txt.splitlines()[line - 1].strip()
                if src.startswith("def "):
                    continue
                rec["call_sites"].append({"file": rel, "line": line, "src": src[:100]})
            # where does the likelihood weighting vector surface?
            for m in re.finditer(r"geneWeight", txt):
                line = txt[:m.start()].count("\n") + 1
                src = txt.splitlines()[line - 1].strip()
                rec["geneWeight_uses"].append({"file": rel, "line": line, "src": src[:100]})
        out[label] = rec
    return out


# --------------------------------------------------------------------------
# D3: patcher regex collateral damage
# --------------------------------------------------------------------------
def d3_patcher_damage():
    pat = re.compile(
        r"(#?\s*\[.*[Rr]un.*[iI][dD][Rr][Ee][Mm]|#?\s*[Rr]un.*iDREM|#?\s*[Ii][Dd][Rr][Ee][Mm])"
    )
    out = {}
    for label, root in TREES.items():
        tool = root / "UNAGI" / "UNAGI_tool.py"
        if not tool.exists():
            out[label] = {"status": "missing"}
            continue
        txt = tool.read_text(encoding="utf-8", errors="replace")
        injected = txt.count("_apply_tf_prior_weights()")
        try:
            import ast
            ast.parse(txt)
            syntactically_valid = True
            err = None
        except SyntaxError as e:
            syntactically_valid = False
            err = {"line": e.lineno, "msg": e.msg}
        # how many of the injections landed on something that is clearly
        # NOT an iDREM invocation?
        bad = []
        for i, ln in enumerate(txt.splitlines(), 1):
            if "_apply_tf_prior_weights()" in ln and not ln.strip().startswith("def "):
                # legitimate targets would be bare runIdrem(...) calls
                if not re.search(r"\brunIdrem\s*\(", ln):
                    bad.append({"line": i, "src": ln.strip()[:110]})
        out[label] = {
            "status": "ran",
            "n_injections": injected,
            "n_collateral": len(bad),
            "syntactically_valid": syntactically_valid,
            "syntax_error": err,
            "collateral_samples": bad[:6],
        }
    return out


# --------------------------------------------------------------------------
# D4: is the likelihood-weighting branch reachable?
# --------------------------------------------------------------------------
def d4_elbo_branch():
    out = {}
    for label, root in TREES.items():
        elbo = root / "UNAGI" / "train" / "customized_elbo.py"
        if not elbo.exists():
            out[label] = {"status": "missing"}
            continue
        txt = elbo.read_text(encoding="utf-8", errors="replace")
        inst = [(i, ln.strip()) for i, ln in enumerate(txt.splitlines(), 1)
                if re.search(r"Loss\w*\(\s*geneWeight", ln)]
        out[label] = {
            "status": "ran",
            "n_instantiations": len(inst),
            "instantiations": [{"line": i, "src": s[:110]} for i, s in inst],
        }
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rep = {
        "D1_geneWeight_occupancy": d1_geneWeight_occupancy(),
        "D2_call_graph": d2_call_graph(),
        "D3_patcher_damage": d3_patcher_damage(),
        "D4_elbo_branch": d4_elbo_branch(),
    }
    rep = _attach_provenance(rep, script=__file__,
                            inputs=[],
                            extra={"scanned_globs": STAGEDATA_GLOBS,
                                   "searched_roots": [str(REPO / q) for q in STAGEDATA_GLOBS],
                                   "note": "D1 glob covers UNAGI/benchmark only; the manuscript quotes a 796-file full-tree count"})
    json.dump(rep, open(OUT / "injection_chain_audit.json", "w"), indent=1)

    L = ["# Prior-injection chain audit", "",
         "Measures what the TF-prior injection *touches in code*, as opposed to "
         "what it changes in the trained spectrum.", ""]

    d1 = rep["D1_geneWeight_occupancy"]
    L += ["## D1 - `geneWeight` layer occupancy", ""]
    if d1["status"] == "ran":
        L += [f"- stagedata files inspected: **{d1['n_files']}**",
              f"- carrying a `geneWeight` layer: **{d1['n_with_layer']}**",
              f"- with **any** non-zero entry: **{d1['n_with_any_nonzero']}**",
              f"- verdict: **{d1['verdict']}**", ""]
    else:
        L += [f"- skipped ({d1.get('reason')})", ""]

    for key, title in [("D2_call_graph", "D2 - call graph"),
                       ("D3_patcher_damage", "D3 - patcher collateral damage"),
                       ("D4_elbo_branch", "D4 - likelihood-weighting branch")]:
        L += [f"## {title}", ""]
        for label, rec in rep[key].items():
            L.append(f"**{label}**")
            if rec.get("status") != "ran":
                L += [f"- {rec.get('status')}", ""]
                continue
            if key == "D2_call_graph":
                L += [f"- definitions: {len(rec['definitions'])}, "
                      f"call sites: {len(rec['call_sites'])}, "
                      f"`geneWeight` mentions: {len(rec['geneWeight_uses'])}"]
                for c in rec["call_sites"][:8]:
                    L.append(f"  - `{c['file']}:{c['line']}` {c['src']}")
            elif key == "D3_patcher_damage":
                L += [f"- injections present: **{rec['n_injections']}**; "
                      f"of which collateral (not an iDREM call): **{rec['n_collateral']}**",
                      f"- parses as valid Python: **{rec['syntactically_valid']}**"]
                if rec["syntax_error"]:
                    L.append(f"  - SyntaxError line {rec['syntax_error']['line']}: "
                             f"{rec['syntax_error']['msg']}")
                for c in rec["collateral_samples"][:4]:
                    L.append(f"  - L{c['line']}: `{c['src']}`")
            else:
                L += [f"- `Loss*(geneWeight=...)` instantiations: "
                      f"**{rec['n_instantiations']}**"]
                for c in rec["instantiations"]:
                    L.append(f"  - `{c['line']}`: {c['src']}")
            L.append("")

    open(OUT / "INJECTION_CHAIN_AUDIT.md", "w", encoding="utf-8").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
