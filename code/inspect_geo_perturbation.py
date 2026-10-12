"""Inspect GEO perturbation-series matrices to decide usability for a
cell-level perturbation benchmark. GEO host is the only reachable one in
this sandbox. Downloads small series_matrix.txt.gz files and reports:
  - number of samples (columns)  -> cell-level if ~10^4..10^6
  - phenotype field names        -> do they encode perturbation/guide?
"""
import urllib.request, gzip, io, sys, re, os

socket_timeout = 60
urllib.request.socket.setdefaulttimeout(socket_timeout)

GEO = "https://ftp.ncbi.nlm.nih.gov/geo/series"
CANDIDATES = {
    "Adamson2016":  f"{GEO}/GSE89nnn/GSE89632/matrix/GSE89632_series_matrix.txt.gz",
    "Norman2019":   f"{GEO}/GSE123nnn/GSE123164/matrix/GSE123164_series_matrix.txt.gz",
}

def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=socket_timeout) as r:
        return r.read()

def inspect(name, url):
    print(f"\n===== {name} =====")
    print("url:", url)
    try:
        raw = fetch(url)
    except Exception as e:
        print("  DOWNLOAD FAIL:", type(e).__name__, str(e)[:100])
        return
    print(f"  downloaded {len(raw)/1e6:.2f} MB")
    text = gzip.decompress(raw).decode("utf-8", "replace")
    lines = text.splitlines()
    # sample count = columns in the data table
    n_samples = None
    sample_fields = []
    titles = []
    chars = []
    for i, ln in enumerate(lines):
        if ln.startswith("!series_matrix_table_begin") or ln.startswith("!sample_table_begin"):
            # next non-! line is the header row of the table
            for j in range(i+1, min(i+3, len(lines))):
                if not lines[j].startswith("!"):
                    n_samples = len(lines[j].split("\t")) - 1  # first col = gene id
                    break
        if ln.startswith("!Sample_title"):
            titles = ln.split("\t")[1:]
        if ln.startswith("!Sample_characteristics"):
            chars = ln.split("\t")[1:]
        if ln.startswith("!Sample_source_name") or ln.startswith("!Sample_description"):
            sample_fields.append(ln.split("\t")[0])
    print(f"  approx samples (columns): {n_samples}")
    print(f"  !Sample_title count: {len(titles)}")
    if titles:
        print("  sample_title examples:", titles[:6])
    if chars:
        print("  sample_characteristics examples:", chars[:6])
    # Look for perturbation/guide keywords in any header
    pert_kw = [l for l in lines if "perturb" in l.lower() or "guide" in l.lower()
               or "crispr" in l.lower() or "knock" in l.lower() or "target" in l.lower()]
    print(f"  perturbation-related header lines: {len(pert_kw)}")
    for l in pert_kw[:8]:
        print("    ", l[:120])

for name, url in CANDIDATES.items():
    inspect(name, url)
print("\nDONE")
