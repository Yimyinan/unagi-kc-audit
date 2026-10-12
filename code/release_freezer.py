# -*- coding: utf-8 -*-
"""release_freezer.py -- build a self-contained, hash-verified release snapshot.

Discipline this enforces (paper1 project rule): NEVER infer the file set from a
directory listing. Instead resolve the REAL LaTeX include graph:

    entry docs (main.tex, supplementary.tex)
        -> \\input / \\include  (recursively, .tex)
        -> from every resolved .tex: \\includegraphics / \\includepdf (graphics/pdf)
        -> from every \\includepdf{X.pdf} that has an X.tex beside it: X.tex
           (REPAIR 2, r80 -- otherwise a PDF-only include hides its own source,
            which is how supp_addendum.tex and its sunk Notes stayed unfrozen)

then copy that closure into <dest>/source, write MANIFEST.sha256, compile both
documents three times with pdflatex, count pages, and drop the two PDFs at <dest>.

Modes
-----
  --collect            print the resolved source file set (relative paths), no copy
  --verify DIR         compare an existing <DIR>/source against the graph closure
                       (missing / extra / sha256 mismatches)
  --dest DIR           build a new release at DIR (source/ + MANIFEST + PDFs + README)
  --no-compile         skip pdflatex (just copy + manifest) -- for dry validation

pdflatex defaults to the project's MiKTeX path; override with --pdflatex.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))          # paper1_kc_diagnostic/
MAIN = os.path.join(ROOT, "main.tex")
SUPP = os.path.join(ROOT, "supplementary.tex")

INPUT_RE = re.compile(r"\\(?:input|include)\{([^}]+)\}")
GRAPHICS_RE = re.compile(r"\\(?:includegraphics|includepdf)(?:\[[^\]]*\])?\{([^}]+)\}")
# REPAIR 2 (r80): kept separate from GRAPHICS_RE so an \includepdf can be told apart
# from an \includegraphics; only the former drags in a sibling .tex source.
INCLUDEPDF_RE = re.compile(r"\\includepdf(?:\[[^\]]*\])?\{([^}]+)\}")

DEFAULT_PDFLATEX = r"D:\MiKTeX\miktex\bin\x64\pdflatex.exe"
GRAPHIC_EXTS = ("", ".pdf", ".png", ".jpg", ".jpeg", ".eps", ".svg", ".PDF")

# Build artifacts that live inside a release but are NOT part of the source closure.
RELEASE_ARTIFACTS = {"README.md", "MANIFEST.sha256", "main.pdf", "supplementary.pdf"}
SKIP_EXTS = (".aux", ".log", ".out", ".toc", ".bbl", ".blg", ".synctex.gz")


def _norm(p: str) -> str:
    return os.path.normpath(p).replace(os.sep, "/")


def _read(path: str) -> str:
    with open(path, encoding="utf-8", errors="ignore") as fh:
        return fh.read()


def resolve_input(arg: str, including_dir: str) -> str | None:
    base = arg
    if base.endswith(".tex"):
        base = base[:-4]
    cands = [
        os.path.join(including_dir, base + ".tex"),
        os.path.join(ROOT, base + ".tex"),
        os.path.join(including_dir, arg),
        os.path.join(ROOT, arg),
    ]
    for c in cands:
        if os.path.isfile(c):
            return os.path.abspath(c)
    return None


def resolve_graphics(arg: str, including_dir: str) -> str | None:
    for ext in GRAPHIC_EXTS:
        for base_dir in (ROOT, including_dir):
            c = os.path.join(base_dir, arg + ext)
            if os.path.isfile(c):
                return os.path.abspath(c)
    return None


def collect_graph() -> set[str]:
    """Return absolute paths of every source file in the include closure.

    Raises FileNotFoundError if any \\input or \\includegraphics/\\includepdf
    target cannot be resolved -- a silent miss would only surface at compile time.

    REPAIR 2 (r80): a \\includepdf{X.pdf} whose LaTeX source X.tex sits beside the
    PDF also contributes X.tex, and X.tex is then walked like any other .tex in the
    closure (its own \\input and \\includegraphics follow).  supplementary.tex
    reaches supp_addendum.tex only through \\includepdf{supp_addendum.pdf}, so
    before this rule the addendum's source -- and the ~4,700 words of sunk Notes it
    holds -- was present in the live tree but absent from every frozen release.
    """
    tex_files: set[str] = set()
    graphics: set[str] = set()
    unresolved: list[tuple[str, str]] = []
    queue = [MAIN, SUPP]
    while queue:
        entry = queue.pop()
        if entry in tex_files or not os.path.isfile(entry):
            continue
        tex_files.add(os.path.abspath(entry))
        d = os.path.dirname(entry)
        text = _read(entry)
        for m in INPUT_RE.finditer(text):
            r = resolve_input(m.group(1), d)
            if r is None:
                unresolved.append((entry, m.group(1)))
            elif r not in tex_files:
                queue.append(r)
        for m in GRAPHICS_RE.finditer(text):
            r = resolve_graphics(m.group(1), d)
            if r is None:
                unresolved.append((entry, m.group(1)))
                continue
            graphics.add(r)
            # \includepdf{X.pdf} with an X.tex beside it: X.tex is source too.
            for pm in INCLUDEPDF_RE.finditer(text):
                pr = resolve_graphics(pm.group(1), d)
                if pr is None or os.path.abspath(pr) != r:
                    continue
                src_tex = os.path.splitext(pr)[0] + ".tex"
                if os.path.isfile(src_tex) and os.path.abspath(src_tex) not in tex_files:
                    queue.append(os.path.abspath(src_tex))
                break
    if unresolved:
        msg = "; ".join(f"{os.path.relpath(t, ROOT)} -> {a!r}" for t, a in unresolved)
        raise FileNotFoundError(f"unresolved include/graphics targets: {msg}")
    return tex_files | graphics


def rel_paths(abs_paths: set[str]) -> list[str]:
    return sorted(_norm(os.path.relpath(p, ROOT)) for p in abs_paths)


def sha256_of(path: str) -> tuple[str, int]:
    h = hashlib.sha256()
    size = 0
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
            size += len(chunk)
    return h.hexdigest(), size


def cmd_collect() -> int:
    for rp in rel_paths(collect_graph()):
        print(rp)
    print(f"# {len(rel_paths(collect_graph()))} source files in include closure", file=sys.stderr)
    return 0


def cmd_verify(dest_root: str) -> int:
    src_dir = os.path.join(dest_root, "source")
    if not os.path.isdir(src_dir):
        print(f"ERROR: {src_dir} not found", file=sys.stderr)
        return 2
    live = rel_paths(collect_graph())
    live_set = set(live)
    phys: set[str] = set()
    for dirpath, _, files in os.walk(src_dir):
        for f in files:
            rel = _norm(os.path.relpath(os.path.join(dirpath, f), src_dir))
            if rel in RELEASE_ARTIFACTS:
                continue
            if rel.endswith(SKIP_EXTS):
                continue
            phys.add(rel)
    missing = sorted(live_set - phys)      # in graph but absent from release -> incomplete
    extra = sorted(phys - live_set)        # in release but unused -> legacy / over-included
    # sha256 cross-check on the overlap
    mism = []
    for rel in sorted(live_set & phys):
        want, _ = sha256_of(os.path.join(ROOT, rel))
        got, _ = sha256_of(os.path.join(src_dir, rel))
        if want != got:
            mism.append(rel)
    print("== release verification against include-graph closure ==")
    print(f"graph closure : {len(live_set)} files")
    print(f"release source: {len(phys)} files")
    print(f"overlap       : {len(live_set & phys)} files")
    print(f"\nMISSING from release ({len(missing)}) -- release is INCOMPLETE:")
    for r in missing:
        print(f"  - {r}")
    print(f"\nEXTRA in release ({len(extra)}) -- legacy / over-included (not in include graph):")
    for r in extra:
        print(f"  + {r}")
    print(f"\nSHA256 mismatches on overlap ({len(mism)}):")
    for r in mism:
        print(f"  ! {r}")
    ok = not missing and not mism
    print("\nVERDICT:", "OK (complete + intact)" if ok else "DEFECTS FOUND")
    return 0 if ok else 1


def _compile(doc: str, workdir: str, pdflatex: str) -> tuple[bool, str]:
    if not os.path.isfile(pdflatex):
        return False, f"pdflatex not found: {pdflatex}"
    log = []
    for _ in range(3):
        r = subprocess.run(
            [pdflatex, "-interaction=nonstopmode", "-halt-on-error", doc],
            cwd=workdir, capture_output=True, text=True)
        log.append(r.stdout + r.stderr)
        if r.returncode != 0:
            return False, "\n".join(log[-1].splitlines()[-25:])
    return True, "\n".join(log)


def _count_pages(pdf: str) -> int:
    with open(pdf, "rb") as fh:
        data = fh.read()
    return data.count(b"/Type /Page") - data.count(b"/Type /Pages")


def cmd_dest(dest_root: str, pdflatex: str, do_compile: bool) -> int:
    src_dir = os.path.join(dest_root, "source")
    if os.path.exists(src_dir):
        print(f"ERROR: {src_dir} already exists -- refusing to overwrite", file=sys.stderr)
        return 2
    os.makedirs(src_dir)
    files = collect_graph()
    for ab in files:
        rel = _norm(os.path.relpath(ab, ROOT))
        dst = os.path.join(src_dir, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(ab, dst)
    # MANIFEST.sha256:  hash  size  relpath
    manifest = []
    for rel in rel_paths(files):
        h, size = sha256_of(os.path.join(src_dir, rel))
        manifest.append(f"{h}  {size}  {rel}")
    with open(os.path.join(dest_root, "MANIFEST.sha256"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(manifest) + "\n")
    print(f"copied {len(files)} source files -> {src_dir}")
    print(f"wrote MANIFEST.sha256 ({len(manifest)} entries)")

    if do_compile:
        for doc in ("main.tex", "supplementary.tex"):
            ok, msg = _compile(doc, src_dir, pdflatex)
            if not ok:
                print(f"COMPILE FAILED: {doc}\n{msg}", file=sys.stderr)
                return 3
            pdf = os.path.join(src_dir, doc.replace(".tex", ".pdf"))
            pages = _count_pages(pdf)
            shutil.copy2(pdf, os.path.join(dest_root, os.path.basename(pdf)))
            print(f"compiled {doc} -> {os.path.basename(pdf)} ({pages} pages)")
        # README
        with open(os.path.join(dest_root, "README.md"), "w", encoding="utf-8") as fh:
            fh.write(f"# Release snapshot\n\nBuilt by code/release_freezer.py from the live include graph.\n\n"
                     f"- source/: {len(files)} files (tex include-graph closure)\n"
                     f"- MANIFEST.sha256: {len(manifest)} sha256 entries\n"
                     f"- main.pdf / supplementary.pdf: three-pass pdflatex output\n")
        print("wrote README.md")
    else:
        print("(--no-compile: skipped pdflatex)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--collect", action="store_true", help="print resolved file set")
    g.add_argument("--verify", metavar="DIR", help="verify existing <DIR>/source")
    g.add_argument("--dest", metavar="DIR", help="build new release at DIR")
    ap.add_argument("--pdflatex", default=DEFAULT_PDFLATEX, help="pdflatex binary")
    ap.add_argument("--no-compile", action="store_true", help="skip pdflatex")
    args = ap.parse_args(argv)

    if args.collect:
        return cmd_collect()
    if args.verify:
        return cmd_verify(args.verify)
    if args.dest:
        return cmd_dest(args.dest, args.pdflatex, not args.no_compile)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
