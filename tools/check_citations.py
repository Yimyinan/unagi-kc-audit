#!/usr/bin/env python3
"""Dead/broken citation + reference-integrity check for paper1."""
import re, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def read(p):
    try:
        with open(os.path.join(ROOT, p), encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return ''

# NOTE: keep this list in sync with the actual \input graph of main.tex.
# references.tex is gone (bibliography is now inline in main.tex/supplementary.tex);
# star_methods.tex was added when Methods was split into STAR Methods.
FILES = ['main.tex', 'star_methods.tex', 'supplementary.tex', 'supp_addendum.tex',
         'latent_residence_conclusion.tex', 'geometric_dual_spectrum.tex']
text = '\n'.join(read(f) for f in FILES)

cited = set()
for m in re.findall(r'\\cite\{([^}]*)\}', text):
    for k in m.split(','):
        cited.add(k.strip())
bib = set(re.findall(r'\\bibitem\{([^}]*)\}', text))

print('TOTAL bibitems:', len(bib))
print('TOTAL distinct \\cite keys:', len(cited))
print('DEAD (defined, never cited):', sorted(bib - cited))
print('BROKEN (cited, undefined):', sorted(cited - bib))
print('CITED KEYS:', sorted(cited))

# manual section cross-refs written as "Section~N.M"
secs = re.findall(r'Section~\\ref\{([^}]*)\}', text)
print('Section \\ref keys used:', sorted(set(secs)))
labels = set(re.findall(r'\\label\{([^}]*)\}', text))
print('LABELS missing for Section refs:', sorted(set(secs) - labels))

# supplementary S-fig/table mentions
sfig_cited = set(re.findall(r'Supplementary (?:Fig(?:ure)?|Table)\.?~?S(\d+)', text))
print('S-figs/tables mentioned in text:', sorted(sfig_cited, key=int))
