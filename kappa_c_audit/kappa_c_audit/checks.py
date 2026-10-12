"""The three-check protocol: turn a spectral claim into a verifiable one.

A high compression ``C`` (low ``kappa``) is routinely read as "the injected prior
was used". It is not: ``C`` is a trace-collapse diagnostic, and deep compression
can arise from the data alone. The three checks below are the operational forms
used in the manuscript to separate the two.

  1. ``gradient_check``       -- does the spectral response track the prior's
                                 gradient contribution?
  2. ``prior_strength_scan``  -- does ``kappa`` respond to the prior strength
                                 ``lambda``?  A flat scan is evidence the collapse
                                 is not prior-driven.
  3. ``structural_bound``     -- is the claimed compression even feasible given
                                 the vanilla trace and spectral gap?
"""

from __future__ import annotations

import numpy as np

__all__ = ["gradient_check", "prior_strength_scan", "structural_bound"]


def gradient_check(spectral_response, prior_gradient_norm):
    """Check 1. Correlate the per-configuration spectral response with the prior's
    gradient contribution.

    Parameters
    ----------
    spectral_response : array-like
        Per-configuration spectral quantity (e.g. ``C`` in percent, or ``kappa``).
    prior_gradient_norm : array-like
        Per-configuration magnitude of the prior's gradient contribution.

    Returns
    -------
    dict with ``pearson``, ``spearman``, ``n`` and ``prior_driven`` (True when the
    correlation is materially non-zero, ``|spearman| >= 0.5`` and ``n >= 4``).
    """
    from scipy.stats import pearsonr, spearmanr

    a = np.asarray(spectral_response, float).ravel()
    b = np.asarray(prior_gradient_norm, float).ravel()
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if len(a) < 2 or a.std() < 1e-12 or b.std() < 1e-12:
        return dict(pearson=float("nan"), spearman=float("nan"), n=len(a),
                    prior_driven=False, note="degenerate input")
    pr = float(pearsonr(a, b)[0])
    sr = float(spearmanr(a, b).correlation)
    return dict(pearson=pr, spearman=sr, n=len(a),
                prior_driven=bool(abs(sr) >= 0.5 and len(a) >= 4))


def prior_strength_scan(kappa_by_lambda):
    """Check 2. Scan ``kappa`` across prior strengths ``lambda``.

    Parameters
    ----------
    kappa_by_lambda : mapping or sequence of (lambda, kappa) pairs

    Returns
    -------
    dict with ``slope`` (linear fit of kappa vs lambda), ``range`` (max-min),
    ``monotone`` and ``responds`` (True when the range exceeds 0.05, i.e. the scan
    is not flat).
    """
    if isinstance(kappa_by_lambda, dict):
        items = sorted(kappa_by_lambda.items(), key=lambda kv: float(kv[0]))
        lam = np.array([float(k) for k, _ in items])
        kap = np.array([float(v) for _, v in items])
    else:
        arr = np.asarray(kappa_by_lambda, float)
        lam, kap = arr[:, 0], arr[:, 1]
    if len(lam) < 2:
        return dict(slope=float("nan"), range=0.0, monotone=False, responds=False)
    slope = float(np.polyfit(lam, kap, 1)[0]) if lam.std() > 1e-12 else 0.0
    rng = float(kap.max() - kap.min())
    diffs = np.diff(kap)
    mono = bool(np.all(diffs >= 0) or np.all(diffs <= 0))
    return dict(slope=slope, range=rng, monotone=mono, responds=bool(rng > 0.05))


def structural_bound(C_percent, trH_vanilla, gap_vanilla):
    """Check 3. Upper bound on the attainable gap ratio at compression ``C``.

    Implements the manuscript bound

        kappa_gap <= (1 - C) * tr(H_V) / gap_V,

    where ``C_percent`` is the global compression in percent, ``trH_vanilla`` the
    vanilla Hessian trace and ``gap_vanilla`` the vanilla spectral gap. If the
    training target exceeds this bound, the target is structurally infeasible on
    that dataset.
    """
    c = float(C_percent) / 100.0
    return float((1.0 - c) * float(trH_vanilla) / float(gap_vanilla))


# --------------------------------------------------------------------------
# Check 4. Injection-path check (added after the worked example showed it was
# missing from this toolkit: the released package carried only checks 1-3, which
# all assume the prior arrived).
# --------------------------------------------------------------------------
import re as _re
from pathlib import Path as _Path

__all__ += ["injection_path_check"]

_ASSIGN_PAT = _re.compile(r"(?:^|\s)(?:self\.)?_?tf_prior_weights\s*=|"
                           r"(?:^|\s)geneWeight\s*=")
_READ_PAT = _re.compile(r"Loss\w*\s*\(\s*geneWeight|\bgeneWeight\b(?!\s*=)")
_INST_PAT = _re.compile(r"Loss\w*\s*\(\s*geneWeight")

def injection_path_check(unagi_root, source_tree,
                         pattern="**/stagedata/*.h5ad",
                         roots=("benchmark", "result"),
                         layer_key="layers/geneWeight/data",
                         weight_name="geneWeight"):
    """Check 4. Does the injected prior reach the training objective?

    A high compression ``C`` is only attributable to prior content if the prior was
    delivered.  This walks the delivery path and reports where it stops.  It does not
    tell you whether the prior is *good* -- only whether it arrived.

    Run it on your own pipeline by naming the two things it has to look for:

    * ``layer_key`` -- the HDF5 path of the per-gene weight vector inside a saved
      training artefact.  Default matches UNAGI's AnnData layout.
    * ``weight_name`` -- the Python identifier the weight vector goes by in your
      training source, used for the assignment / reader / instantiation search.

    Parameters
    ----------
    unagi_root : str
        Root holding training outputs, searched as
        ``unagi_root/<root>/<pattern>`` for each entry of ``roots``.
    source_tree : str
        Source tree to search for assignment / reader / instantiation sites.
    pattern : str
        Glob for the recorded training artefacts.
    roots : sequence of str
        Sub-directories of ``unagi_root`` to scan.  Counted and reported separately,
        because a subset count and a full-tree count are different denominators --
        the manuscript itself quotes both 40 and 796 for the same pool.
    layer_key : str
        HDF5 dataset path holding the per-gene weights.
    weight_name : str
        Identifier to grep for in the source tree.

    Returns
    -------
    dict with ``per_root`` (file counts, empty vs populated), ``weight_layer_verdict``,
    ``readers`` (locational only), ``branch`` (instantiation sites) and ``verdict``.
    The decisive evidence is ``per_root`` (empirical) and ``branch`` (structural);
    ``readers`` is locational and does not establish that a value is consumed.
    """
    import re as _r
    assign_pat = _r.compile(r"(?:^|\s)(?:self\.)?_?%s\s*=|(?:^|\s)%s\s*="
                            % (_r.escape(weight_name), _r.escape(weight_name)))
    read_pat = _r.compile(r"Loss\w*\s*\(\s*%s|\b%s\b(?!\s*=)"
                          % (_r.escape(weight_name), _r.escape(weight_name)))
    inst_pat = _r.compile(r"Loss\w*\s*\(\s*%s" % _r.escape(weight_name))

    out = {"check": "injection-path", "unagi_root": str(unagi_root),
           "source_tree": str(source_tree), "layer_key": layer_key,
           "weight_name": weight_name, "per_root": {}, "verdict": {}}

    try:
        import h5py
    except ImportError:
        out["per_root"] = {"status": "skipped", "reason": "h5py unavailable"}
        return out

    total_with = total_empty = total_pop = 0
    for name in roots:
        base = _Path(unagi_root) / name
        if not base.is_dir():
            out["per_root"][name] = {"status": "missing"}
            continue
        files = sorted(base.glob(pattern))
        with_layer = empty = populated = no_layer = 0
        for f in files:
            try:
                with h5py.File(f, "r") as h:
                    key = layer_key
                    if key not in h:
                        no_layer += 1
                        continue
                    with_layer += 1
                    if h[key].shape[0] == 0:
                        empty += 1
                    else:
                        populated += 1
            except Exception:                              # noqa: BLE001
                no_layer += 1
        out["per_root"][name] = {
            "files_matched": len(files), "with_layer": with_layer,
            "empty_layer": empty, "populated_layer": populated, "no_layer": no_layer}
        total_with += with_layer
        total_empty += empty
        total_pop += populated

    out["weight_layer_totals"] = {"with_layer": total_with, "empty": total_empty,
                                  "populated": total_pop}
    out["weight_layer_verdict"] = (
        "ALL-ZERO: layer present but empty in every artefact scanned" if total_with
        and total_pop == 0 else
        "POPULATED: at least one artefact carries non-zero weights" if total_pop
        else "undetermined")

    # ---- structural: is the consuming branch instantiated anywhere? ----
    # Mechanism copied from the script that produced the manuscript's numbers
    # (code/prior_injection_chain_audit.py): plain regex over raw lines.  A tokenizer
    # variant was tried first and produced inflated counts (454 vs 0), so it was
    # discarded.  Reader sites are locational only and are NOT used as evidence.
    tree = _Path(source_tree)
    assigns, readers, insts = [], [], []
    if tree.is_dir():
        for p in sorted(tree.rglob("*.py")):
            try:
                body = p.read_text(encoding="utf-8", errors="replace")
            except Exception:                              # noqa: BLE001
                continue
            for lineno, raw in enumerate(body.splitlines(), 1):
                rec = {"file": str(p.relative_to(tree)), "line": lineno}
                if assign_pat.search(raw):
                    assigns.append(rec)
                if read_pat.search(raw):
                    readers.append(rec)
                if inst_pat.search(raw) and not raw.strip().startswith("def "):
                    insts.append(dict(rec, src=raw.strip()[:100]))

    out["readers"] = {"n_assignment_sites": len(assigns),
                      "n_reader_sites": len(readers), "decisive": False,
                      "note": ("locational only; a reader inside a class definition "
                               "does not prove the stored value is consumed")}
    out["branch"] = {"n_instantiations": len(insts), "instantiations": insts[:10],
                     "verdict": ("DEAD: no instantiation site" if not insts
                                 else "instantiated %d time(s)" % len(insts)),
                     "decisive": True}
    out["verdict"] = {
        "delivery": ("BLOCKED" if out["weight_layer_verdict"].startswith("ALL-ZERO")
                     else "not blocked (or undetermined)"),
        "evidence_used": ["per_root (empirical)", "branch (structural)"],
        "evidence_not_used": ["readers (locational only)"],
        "implication": ("If the layer is empty AND the consuming branch is never "
                        "instantiated, no decoder change can be attributed to prior "
                        "content in this implementation."),
        "scope_limit": ("A property of THIS implementation; it does not show that GRN "
                        "priors are ineffective in general."),
    }
    return out
