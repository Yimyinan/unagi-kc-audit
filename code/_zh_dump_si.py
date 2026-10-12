# -*- coding: utf-8 -*-
"""Dump the prose embedded inside \\includepdf targets to plain-text files so the SI
translation pass can read them directly (their text is NOT in the .tex)."""
import io, json, os, re

BASE = r"F:/UNAGI_Project/paper1/paper1_kc_diagnostic"
MAN = os.path.join(BASE, "zh_assets", "assets_manifest.json")
OUT = os.path.join(BASE, "zh_translation", "si_sources")
os.makedirs(OUT, exist_ok=True)

M = json.load(io.open(MAN, encoding="utf-8"))
PROSE = ("supp_note", "supp_addendum")

written = []
for r in M["includepdf"]:
    tgt = r["target"]
    base = os.path.splitext(os.path.basename(tgt))[0]
    if not any(k in base for k in PROSE):
        continue
    txt = r.get("text", "")
    # tidy: collapse the ragged inter-word spacing PyMuPDF yields from justified text
    txt = re.sub(r"[ \t]+", " ", txt)
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    p = os.path.join(OUT, base + ".txt")
    io.open(p, "w", encoding="utf-8").write(txt)
    written.append((base, len(txt.split()), r["pages"]))

print("=== prose extracted from embedded PDFs ===")
for b, n, pg in written:
    print("  %-34s pages=%-2d words=%5d" % (b, pg, n))
print("  written to:", OUT)

print()
print("=== table PDFs (data-dense; will be embedded as images + translated captions) ===")
for r in M["includepdf"]:
    base = os.path.splitext(os.path.basename(r["target"]))[0]
    if base.startswith("table_"):
        print("  %-40s pages=%d words=%d" % (base, r["pages"], len(r.get("text", "").split())))
