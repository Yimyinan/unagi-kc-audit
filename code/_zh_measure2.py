# -*- coding: utf-8 -*-
import io, json, os
M = json.load(io.open(r"F:/UNAGI_Project/paper1/paper1_kc_diagnostic/zh_assets/assets_manifest.json",
                     encoding="utf-8"))
print("=== \\includepdf targets: pages + extracted words ===")
tot = 0
for r in M["includepdf"]:
    t = r.get("text", "")
    n = len(t.split())
    tot += n
    print("  %-46s pages=%-3d words=%6d%s" % (r["target"], r["pages"], n,
          "   (capped at 20k chars)" if len(t) >= 20000 else ""))
print("  %-46s            %6d words from includepdf" % ("TOTAL", tot))
print()
print("=== \\includegraphics: pages/PNG only (captions live in .tex) ===")
print("  targets:", len(M["includegraphics"]), " PNGs:",
      sum(len(r["png"]) for r in M["includegraphics"]))
print()
print("Already counted from .tex: main 14,495 + SI text 6,865 = 21,360 words")
print("GRAND TOTAL (tex + embedded pdf prose) ~= %d words" % (21360 + tot))
