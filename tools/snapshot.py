# -*- coding: utf-8 -*-
"""Snapshot the manuscript sources before any bulk edit.

Motivation: a single over-broad string slice once deleted ~14 kB of main.tex
(five figures, the summary table, Methods summary, STAR Methods input and the
whole back-matter). There was no recent backup; recovery only worked because a
week-old copy happened to survive under archive/. Take a snapshot first.

Usage:
    python tools/snapshot.py            # snapshot into snapshots/<timestamp>/
    python tools/snapshot.py --list     # list snapshots
"""
import argparse, datetime, glob, io, os, shutil, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SNAPDIR = os.path.join(ROOT, "snapshots")

SOURCES = ["main.tex", "star_methods.tex", "supplementary.tex",
           "supp_addendum.tex", "latent_residence_conclusion.tex",
           "submission_metadata.tex"]
SOURCES += [os.path.relpath(p, ROOT) for p in glob.glob(os.path.join(ROOT, "supplementary", "*.tex"))]
SOURCES += [os.path.relpath(p, ROOT) for p in glob.glob(os.path.join(ROOT, "*.bib"))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--note", default="")
    a = ap.parse_args()

    if a.list:
        if not os.path.isdir(SNAPDIR):
            print("no snapshots yet")
            return
        for d in sorted(os.listdir(SNAPDIR)):
            n = len(os.listdir(os.path.join(SNAPDIR, d)))
            print("  %s  (%d files)" % (d, n))
        return

    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    dest = os.path.join(SNAPDIR, stamp)
    os.makedirs(dest, exist_ok=True)
    n = 0
    for rel in SOURCES:
        src = os.path.join(ROOT, rel)
        if not os.path.isfile(src):
            continue
        out = os.path.join(dest, rel)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        shutil.copy2(src, out)
        n += 1
    if a.note:
        io.open(os.path.join(dest, "_NOTE.txt"), "w", encoding="utf-8").write(a.note + "\n")
    print("snapshot %s: %d files -> %s" % (stamp, n, dest))


if __name__ == "__main__":
    main()
