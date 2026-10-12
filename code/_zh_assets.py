# -*- coding: utf-8 -*-
"""paper1 r23 -> Chinese Word: stage 0 (asset extraction).

1. Parse every \\includegraphics / \\includepdf target from the frozen r23 source.
2. Render each PDF target to PNG at 200 dpi (for embedding in the Word document).
3. Dump the text content of every \\includepdf target (their content lives inside the
   PDF, NOT in the .tex -- needed so captions/legends can be translated faithfully).
"""
import io, os, re, json

SRC = r"F:/UNAGI_Project/paper1/paper1_kc_diagnostic/releases/2026-09-19_s1-corrected-r23/source"
OUT = r"F:/UNAGI_Project/paper1/paper1_kc_diagnostic/zh_assets"
os.makedirs(OUT, exist_ok=True)
os.makedirs(os.path.join(OUT, "png"), exist_ok=True)

def rd(f):
    return io.open(os.path.join(SRC, f), encoding="utf-8-sig", errors="replace").read()

def targets(files):
    inc, pdf = [], []
    for f in files:
        t = rd(f)
        for m in re.finditer(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}", t):
            g = m.group(1).strip()
            if not os.path.splitext(g)[1]:
                g += ".pdf"
            inc.append((f, g))
        for m in re.finditer(r"\\includepdf(?:\[[^\]]*\])?\{([^}]*)\}", t):
            g = m.group(1).strip()
            if not os.path.splitext(g)[1]:
                g += ".pdf"
            pdf.append((f, g))
    return inc, pdf

tex = []
for dp, dirs, fs in os.walk(SRC):
    for f in fs:
        if f.endswith(".tex"):
            tex.append(os.path.relpath(os.path.join(dp, f), SRC).replace("\\", "/"))
tex.sort()

inc, pdf = targets(tex)
inc = sorted(set(g for _f, g in inc))
pdf = sorted(set(g for _f, g in pdf))

print("includegraphics targets (%d):" % len(inc))
for g in inc:
    print("   ", g, "EXISTS" if os.path.isfile(os.path.join(SRC, g)) else "*** MISSING ***")
print()
print("includepdf targets (%d):" % len(pdf))
for g in pdf:
    print("   ", g, "EXISTS" if os.path.isfile(os.path.join(SRC, g)) else "*** MISSING ***")

import fitz  # PyMuPDF

manifest = {"includegraphics": [], "includepdf": []}
for kind, items in (("includegraphics", inc), ("includepdf", pdf)):
    for g in items:
        src = os.path.join(SRC, g)
        if not os.path.isfile(src):
            continue
        doc = fitz.open(src)
        rec = {"target": g, "pages": doc.page_count}
        pngs = []
        for i, page in enumerate(doc):
            pix = page.get_pixmap(dpi=200)
            name = "%s__p%d.png" % (os.path.splitext(os.path.basename(g))[0], i + 1)
            pix.save(os.path.join(OUT, "png", name))
            pngs.append(name)
        rec["png"] = pngs
        if kind == "includepdf":
            rec["text"] = "\n".join(page.get_text() for page in doc)[:20000]
        manifest[kind].append(rec)
        doc.close()

json.dump(manifest, io.open(os.path.join(OUT, "assets_manifest.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)

n_png = len(os.listdir(os.path.join(OUT, "png")))
print()
print("rendered PNGs:", n_png)
print("manifest ->", os.path.join(OUT, "assets_manifest.json"))
