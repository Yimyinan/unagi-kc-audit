# -*- coding: utf-8 -*-
"""Freeze the current manuscript as a reproducible release snapshot.

Creates ``releases/<tag>/`` containing
    source/           the LaTeX sources actually consumed by the build, mirroring
                      the directory layout so the snapshot compiles standalone
    MANIFEST.sha256   sha256 + size of every source AND every data artifact the
                      build depends on (data artifacts are referenced, not copied)
    README.md         what the release is, what changed, how to rebuild, and the
                      headline numbers so the snapshot is self-describing

Usage: python code/freeze_release.py --tag 2026-09-17_s1-corrected
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER1 = os.path.abspath(os.path.join(HERE, ".."))
ROOT = os.path.abspath(os.path.join(PAPER1, "..", ".."))

TOP_TEX = ["main.tex", "supplementary.tex", "star_methods.tex",
           "latent_residence_conclusion.tex", "submission_metadata.tex",
           "supp_addendum.tex"]

SOURCES_FIXED = [
    "kappa_c_audit/kappa_c_audit/core.py", "kappa_c_audit/kappa_c_audit/audit.py",
    "kappa_c_audit/kappa_c_audit/__init__.py", "kappa_c_audit/README.md",
    "kappa_c_audit/tests/test_core.py", "kappa_c_audit/examples/quickstart.py",
]


def discover_scripts():
    """Every analysis/figure/patch script, found by walking the tree rather than by name.

    A hardcoded list has silently dropped newly added scripts three times in this project
    (the reference closure, the archived figures, the manifest), so this is deliberately
    discovery-based: an extra file in a provenance record is harmless, a missing one is not.
    """
    out = []
    for sub in ("code", "figures"):
        d = os.path.join(PAPER1, sub)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.endswith(".py") and not f.startswith("_"):
                out.append(f"{sub}/{f}")
    return out


SOURCES = discover_scripts() + SOURCES_FIXED

DATA = [
    "outputs/positive_result/s1f_0of33_recheck.json",
    "outputs/positive_result/s1g_corrected_summaries.json",
    "outputs/positive_result/s1h_null_bands.json",
    "outputs/positive_result/s1i_crossarch_recheck.json",
    "outputs/positive_result/s1j_holm_test1.json",
    "outputs/positive_result/s1k_gvae_band.json",
    "outputs/positive_result/s1l_pbmc3k_scvi.json",
    "outputs/positive_result/s1_live_path/s1e_reanalysis.json",
]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pdf_pages(p):
    """Page count from the PDF object tree.

    NB: counting form feeds in ``pdftotext`` output is NOT a page count -- layout
    mode emits extra form feeds (it reported 54 for a 37-page document).
    """
    import re
    try:
        with open(p, "rb") as f:
            data = f.read()
    except OSError:
        return None
    return len(re.findall(rb"/Type\s*/Page[^s]", data)) or None


def _targets(tex_text, cmd):
    """Resolve \\includegraphics / \\includepdf / \\input targets from .tex text."""
    import re
    out = []
    for m in re.finditer(re.escape(cmd) + r"\[[^\]]*\]\{([^}]+)\}", tex_text):
        out.append(m.group(1))
    for m in re.finditer(re.escape(cmd) + r"\{([^}]+)\}", tex_text):
        out.append(m.group(1))
    return out


def collect_inputs():
    """The true build closure: every .tex consumed plus every file it includes.

    Derived from the sources rather than assumed from the directory layout -- an
    earlier version listed directories by hand and silently missed
    ``supp_addendum.pdf`` at the project root, which only surfaced when the
    snapshot failed to compile standalone.
    """
    tex_files = []
    for t in TOP_TEX:
        p = os.path.join(PAPER1, t)
        if os.path.exists(p):
            tex_files.append(p)
    sub = os.path.join(PAPER1, "supplementary")
    if os.path.isdir(sub):
        for f in sorted(os.listdir(sub)):
            if f.endswith(".tex"):
                tex_files.append(os.path.join(sub, f))

    inputs = set(tex_files)
    for p in tex_files:
        text = open(p, encoding="utf-8", errors="ignore").read()
        for target in (_targets(text, r"\includegraphics")
                       + _targets(text, r"\includepdf")
                       + _targets(text, r"\input")):
            cand = os.path.normpath(os.path.join(PAPER1, target))
            if not os.path.exists(cand) and not cand.endswith(".tex"):
                cand_tex = cand + ".tex"
                if os.path.exists(cand_tex):
                    cand = cand_tex
            if os.path.exists(cand) and os.path.isfile(cand):
                inputs.add(cand)
    return sorted(inputs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    rel = os.path.join(PAPER1, "releases", args.tag)
    if os.path.exists(rel):
        print(f"refusing: {rel} already exists")
        return 1
    src_dir = os.path.join(rel, "source")
    os.makedirs(src_dir)

    inputs = collect_inputs()
    copied = 0
    for p in inputs:
        rp = os.path.relpath(p, PAPER1)
        if rp.startswith("releases" + os.sep):
            continue
        dst = os.path.join(src_dir, rp)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(p, dst)
        copied += 1

    # the built PDFs belong in the snapshot too
    for name in ("main.pdf", "supplementary.pdf"):
        p = os.path.join(PAPER1, "build", name)
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(rel, name))

    lines = [f"# release {args.tag}  frozen {datetime.now().isoformat(timespec='seconds')}", ""]
    lines.append("## manuscript sources (copied into source/)")
    for p in sorted(inputs):
        rp = os.path.relpath(p, PAPER1)
        if rp.startswith("releases" + os.sep):
            continue
        lines.append(f"{sha(p)}  {os.path.getsize(p):>10}  {rp}")
    lines.append("")
    lines.append("## analysis + figure + patch scripts (referenced)")
    for rp in SOURCES:
        p = os.path.join(PAPER1, rp) if not rp.startswith(("outputs", "kappa")) \
            else os.path.join(PAPER1, rp)
        if os.path.exists(p):
            lines.append(f"{sha(p)}  {os.path.getsize(p):>10}  {rp}")
        else:
            lines.append(f"{'MISSING':<64}  {rp}")
    lines.append("")
    lines.append("## data artifacts (referenced)")
    for rp in DATA:
        p = os.path.join(ROOT, rp)
        if os.path.exists(p):
            lines.append(f"{sha(p)}  {os.path.getsize(p):>10}  {rp}")
        else:
            lines.append(f"{'MISSING':<64}  {rp}")
    with open(os.path.join(rel, "MANIFEST.sha256"), "w", encoding="utf-8",
              newline="\n") as f:
        f.write("\n".join(lines) + "\n")

    mp = os.path.join(rel, "main.pdf")
    sp = os.path.join(rel, "supplementary.pdf")
    head = {}
    for rp in ("outputs/positive_result/s1f_0of33_recheck.json",
               "outputs/positive_result/s1h_null_bands.json"):
        p = os.path.join(ROOT, rp)
        if os.path.exists(p):
            head[os.path.basename(rp)] = json.load(open(p, encoding="utf-8-sig"))

    readme = f"""# Release {args.tag}

Frozen snapshot of the κ-C diagnostic manuscript with the **corrected readout
statistic** fully applied. Self-contained: `source/` compiles standalone with the
same toolchain, and every data artifact used by the figures and tables is
identified by sha256 in `MANIFEST.sha256`.

* `main.pdf` — {pdf_pages(mp) or '?'} pages
* `supplementary.pdf` — {pdf_pages(sp) or '?'} pages
* sources copied: {copied} files
* frozen: {datetime.now().isoformat(timespec='seconds')}

## What this release contains

The audited quantity is the aggregate predictability ratio
`κ-C(W) = Σ_g R²_g(perturbed) / Σ_g R²_g(vanilla)`. The released package version
computed the trace ratio `tr(W'W)` instead, which equals it only for a whitened
latent code; in the 33 audited UNAGI configurations the precondition fails 33/33.

| result | value |
|---|---|
| deep compression κ < 0.5 | **0/33 in both statistics** (band-independent) |
| corrected κ range, 33 configs | **0.873 – 1.003** (trace form 0.518 – 5.78) |
| family-matched corrected band | hematopoietic **[0.971, 1.029]**, neural crest [0.713, 1.298] |
| classification | **31/33 vanilla-equivalent**, 1 sub-vanilla (R13_s123, κ=0.950), 1 unbanded (lung) |
| G-VAE cross-family band, corrected | **1.0005 ± 0.0319 → [0.937, 1.064]** (re-derived; the released 1.006 ± 0.112 is not regenerable) |
| trace-form scatter inflation, family-matched hemo pool | **56×** (0.817 vs 0.015) |
| live-path positive control (Fig. S26) | corrected **t = 8.2**, 3/3 seeds separated; trace form t = 1.4, not separated |

Claims withdrawn in this release: the "positively coupled traces" reading, the
purity paradox, and the tissue gradient / C-percent taxonomy — all trace-form
artefacts. The injection-chain mechanism, the latent-spectrum diagnostic, the
cross-architecture reproduction and the 0/33 headline are unaffected.

## Rebuild

```bash
cd source && cp -r . <workdir>/ && cd <workdir>
# figures and generated table bodies first
python ../code/make_tabS1.py && python ../code/make_tabS6.py && python ../code/make_tabS8.py
python figures/make_fig2_kc_phase.py && python figures/make_figS5_sensitivity.py
python figures/make_figS8_band_calibration.py && python figures/make_figS26_live_path.py
# then the documents (3 passes each)
/d/MiKTeX/miktex/bin/x64/pdflatex.exe -interaction=nonstopmode -halt-on-error \\
    -output-directory=build main.tex
/d/MiKTeX/miktex/bin/x64/pdflatex.exe -interaction=nonstopmode -halt-on-error \\
    -output-directory=build supplementary.tex
```

Note: `code/make_tabS{1,6,8}.py` write into `supplementary/`; the snapshot already
contains their outputs, so a plain recompile needs no regeneration.

## Known open item

Section 2.9's pbmc3k scVI entry is being re-run to obtain its corrected value
(`code/s1l_pbmc3k_scvi_recheck.py`); if that run post-dates this freeze, the
corrected number is not in this snapshot. Every other κ-C(W) in the manuscript is
either corrected or explicitly labelled as the released trace form.
"""
    with open(os.path.join(rel, "README.md"), "w", encoding="utf-8",
              newline="\n") as f:
        f.write(readme)

    print(f"[frozen] {rel}")
    print(f"  sources copied : {copied}")
    print(f"  manifest lines : {len(lines)}")
    print(f"  pages          : main {pdf_pages(mp)}, supplementary {pdf_pages(sp)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
