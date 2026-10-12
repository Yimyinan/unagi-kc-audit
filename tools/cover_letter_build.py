# -*- coding: utf-8 -*-
"""Build submission/cover_letter_patterns.pdf from cover_letter_patterns.md.

Ordering is load-bearing: escape LaTeX specials first, then inline emphasis,
and only THEN insert heading commands. If headings are inserted first, their
`*` (in \\subsection*) is swallowed by the italic regex, which then pairs it
with the next heading's `*` and garbles everything in between.
"""
import io, re, subprocess, sys

SRC = "submission/cover_letter_patterns.md"
TEX = "submission/cover_letter_patterns.tex"
OUTDIR = "submission"

md = io.open(SRC, encoding="utf-8").read()
md = md.split("\n---\n")[0].strip()          # drop the notes-to-self block

# --- 1. protect math and inline code ---------------------------------
maths, codes = [], []
md = re.sub(r"\$[^$]*\$", lambda m: (maths.append(m.group(0)),
                                     "\x00M%d\x00" % (len(maths) - 1))[1], md)
md = re.sub(r"`([^`]*)`", lambda m: (codes.append(m.group(1)),
                                     "\x00C%d\x00" % (len(codes) - 1))[1], md)

# --- 2. escape LaTeX specials ----------------------------------------
md = md.replace("\\", r"\textbackslash{}")
for a, b in [("&", r"\&"), ("%", r"\%"), ("#", r"\#"), ("_", r"\_"),
             ("{", r"\{"), ("}", r"\}")]:
    md = md.replace(a, b)

# --- 3. inline emphasis (BEFORE headings; line-bounded) --------------
md = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", md)
md = re.sub(r"(?<![\*\w])\*([^*\n]+?)\*(?!\*)", r"\\emph{\1}", md)

# --- 4. headings and list items --------------------------------------
md = re.sub(r"^\\#\\# (.+)$", r"\\subsection*{\1}", md, flags=re.M)
md = re.sub(r"^\\# (.+)$", r"\\section*{\1}", md, flags=re.M)
md = re.sub(r"^\s*-\s+(.+)$", r"\\item \1", md, flags=re.M)

# --- 5. restore protected fragments ----------------------------------
for i, c in enumerate(codes):
    md = md.replace("\x00C%d\x00" % i, r"\texttt{%s}" % c.replace("_", r"\_"))
for i, m in enumerate(maths):
    md = md.replace("\x00M%d\x00" % i, m)

# --- 6. group \item runs ---------------------------------------------
out, in_list = [], False
for ln in md.split("\n"):
    if ln.startswith(r"\item "):
        if not in_list:
            out.append(r"\begin{itemize}[leftmargin=1.4em]"); in_list = True
        out.append("  " + ln)
    else:
        if in_list:
            out.append(r"\end{itemize}"); in_list = False
        out.append(ln)
if in_list:
    out.append(r"\end{itemize}")

body = "\n".join(out).replace("\n\n", "\n\n\\par\n")
tex = (r"""\documentclass[11pt,a4paper]{article}
\usepackage[margin=2.4cm]{geometry}
\usepackage{amsmath,amssymb}
\usepackage{enumitem}
\usepackage[hidelinks]{hyperref}
\usepackage{parskip}
\setlength{\emergencystretch}{2em}
\pagestyle{empty}
\begin{document}
\small
""" + body + "\n\\end{document}\n")
io.open(TEX, "w", encoding="utf-8").write(tex)

PDF = r"D:/MiKTeX/miktex/bin/x64/pdflatex.exe"
rc = 0
for _ in range(3):
    p = subprocess.run([PDF, "-interaction=nonstopmode",
                        "-output-directory=" + OUTDIR, TEX],
                       capture_output=True, text=True)
    rc = p.returncode
log = io.open(OUTDIR + "/cover_letter_patterns.log", encoding="utf-8",
              errors="replace").read() if rc == 0 else ""
bad = [l for l in log.split("\n") if l.startswith("!")]
print("pdflatex exit:", rc, "| fatal lines:", len(bad))
for l in bad[:5]:
    print("  ", l)
sys.exit(1 if (rc != 0 or bad) else 0)
