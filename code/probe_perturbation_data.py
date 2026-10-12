"""Probe sandbox egress for downloadable real single-cell perturbation datasets.

Short per-request timeouts so a dead host cannot hang the run.
Reports which hosts / candidate datasets are reachable, plus a tiny real
byte-fetch to confirm data (not just HEAD) is obtainable.
"""
import socket, ssl, urllib.request, sys, time

socket.setdefaulttimeout(20)

HOSTS = [
    ("figshare-cdn", "https://ndownloader.figshare.com/"),
    ("figshare",     "https://figshare.com/"),
    ("geo-ftp",      "https://ftp.ncbi.nlm.nih.gov/geo/"),
    ("zenodo",       "https://zenodo.org/"),
    ("dropbox",      "https://www.dropbox.com/"),
    ("github-raw",   "https://raw.githubusercontent.com/"),
]

# Candidate perturbation datasets (control/perturbed paired, real biology).
CANDIDATES = [
    ("Dixit2016 GEO matrix (Perturb-seq, small)",
     "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE90nnn/GSE90063/matrix/GSE90063_series_matrix.txt.gz"),
    ("Replogle2022 Zenodo (K562 genome-scale Perturb-seq)",
     "https://zenodo.org/record/5836124/files/replogle.h5ad"),
    ("Norman2019 GEARS opendoor (pooled CRISPRi)",
     "https://www.dropbox.com/s/q4sp6h3w1j0zh0b/opendoor.tar.gz?dl=1"),
    ("Adamson2016 GEO matrix (CRISPRi)",
     "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE89nnn/GSE89632/matrix/GSE89632_series_matrix.txt.gz"),
]

def probe(url):
    try:
        t0 = time.time()
        req = urllib.request.Request(url, method="HEAD",
                                     headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=20) as r:
            return f"OK  {r.status}  {time.time()-t0:.1f}s  {r.headers.get('Content-Length','?')}B"
    except Exception as e:
        return f"FAIL {type(e).__name__}: {str(e)[:80]}"

def tiny_fetch(url, n=4096):
    """Confirm real data bytes are obtainable (not just HEAD)."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=25) as r:
            data = r.read(n)
        return f"FETCHED {len(data)} bytes"
    except Exception as e:
        return f"FETCH-FAIL {type(e).__name__}: {str(e)[:80]}"

print("=== HOST REACHABILITY (HEAD) ===")
for name, url in HOSTS:
    print(f"{name:14s}: {probe(url)}")

print("\n=== CANDIDATE PERTURBATION DATASETS (HEAD) ===")
for name, url in CANDIDATES:
    print(f"\n{name}\n  {url}\n  -> {probe(url)}")

# If a candidate host looked alive, try a tiny real fetch on the smallest one (Dixit matrix).
print("\n=== TINY REAL FETCH (Dixit GEO matrix, first 4KB) ===")
print("  ->", tiny_fetch(CANDIDATES[0][1]))

print("\nDONE")
