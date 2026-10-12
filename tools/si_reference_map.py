# -*- coding: utf-8 -*-
"""Inventory every SI figure/table/note mention across the paper1 tex files."""
import io, os, re, collections

FILES = ["main.tex", "star_methods.tex", "supplementary.tex", "supp_addendum.tex",
         "latent_residence_conclusion.tex", "geometric_dual_spectrum.tex"]
# ALSO scan everything that is \input/\included from supplementary.tex: the SI
# subfiles carry their own figure captions, and missing them once caused a wrong
# renumbering. Collect the \input graph transitively.
import glob
for _f in glob.glob("supplementary/*.tex"):
    if _f not in FILES:
        FILES.append(_f)

pat_fig = re.compile(r"Supplementary\s+Fig(?:ure)?s?\.?\s*~?\s*S(\d+)")
pat_tab = re.compile(r"Supplementary\s+Tables?\.?\s*~?\s*S(\d+)")
pat_note = re.compile(r"Supplementary\s+Notes?\.?\s*~?\s*(\d+)")
# definitions in supplementary.tex / addendum
def_pdf = re.compile(r"addtotoc=\{1,section,1,(Supplementary (?:Note|Table|Figure) S?\d+[^,]*),")
def_cap = re.compile(r"\\caption\*\{\\textbf\{(Supplementary (?:Note|Table|Figure) S\d+)")

fig_ref = collections.defaultdict(list)       # in-prose references only
fig_cap = collections.defaultdict(list)       # the float's own caption line
tab_ref = collections.defaultdict(list)
tab_cap = collections.defaultdict(list)
note_ref = collections.defaultdict(list)

for f in FILES:
    if not os.path.exists(f):
        continue
    for i, line in enumerate(io.open(f, encoding="utf-8"), 1):
        # A float's own caption mentions it by name.  Counting that as a "reference" made
        # this check vacuous: a figure defined and never cited anywhere scored refs=1 and
        # was reported as referenced.  Split the caption mention out.
        is_caption = ("\\caption" in line)
        for m in pat_fig.finditer(line):
            (fig_cap if is_caption else fig_ref)[int(m.group(1))].append("%s:%d" % (f, i))
        for m in pat_tab.finditer(line):
            (tab_cap if is_caption else tab_ref)[int(m.group(1))].append("%s:%d" % (f, i))
        for m in pat_note.finditer(line):
            note_ref[int(m.group(1))].append("%s:%d" % (f, i))

sup = io.open("supplementary.tex", encoding="utf-8").read()
defined_notes = sorted(set(int(x) for x in re.findall(r"Supplementary (?:Note) (\d+):", sup)))
defined_tabs = sorted(set(int(x) for x in re.findall(r"Supplementary (?:Table) S(\d+):", sup)))

print("=" * 72)
print("FIGURES")
print("=" * 72)
figs = sorted(set(list(fig_ref) + list(fig_cap)))
for n in figs:
    print("  S%-3d prose_refs=%d caption=%-2d %s"
          % (n, len(fig_ref[n]), len(fig_cap[n]), ", ".join(fig_ref[n][:4])))
orphan_figs = [n for n in figs if not fig_ref[n]]
print("\nFIGURES never referenced in prose: %s" % (orphan_figs or "none"))

print()
print("=" * 72)
print("TABLES  (defined in supplementary.tex: %s)" % defined_tabs)
print("=" * 72)
for n in sorted(set(list(tab_ref) + list(tab_cap))):
    print("  S%-3d prose_refs=%d caption=%-2d %s"
          % (n, len(tab_ref[n]), len(tab_cap[n]), ", ".join(tab_ref[n][:4])))
orphan_tabs = [n for n in sorted(set(list(tab_ref) + list(tab_cap))) if not tab_ref[n]]
print("\nTABLES defined but never referenced in prose: %s" % (orphan_tabs or "none"))

print()
print("=" * 72)
print("NOTES  (defined: %s)" % defined_notes)
print("=" * 72)
for n in sorted(note_ref):
    print("  Note %-3d refs=%d  %s" % (n, len(note_ref[n]), ", ".join(note_ref[n][:4])))
print("\nNOTES defined but never referenced:",
      [n for n in defined_notes if not note_ref[n]] or "none")

# machine-readable verdict for run_checks.py
print()
print("ORPHANS: figures=%s tables=%s"
      % (orphan_figs or "none", orphan_tabs or "none"))
