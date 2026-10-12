"""
verify_loading_duplicates.py
============================
Integrity check on the 30 audit readout matrices in
`UNAGI/decoder_unified/loading/`.

Why this exists
---------------
The audit counts "33 configurations" as independent evidence. Two anomalies
found while building the GWAS dose-response suggest some of those readouts may
not be independent:

  * R14_s*.npy and R14_seed*.npy are byte-identical (two spellings of one file).
  * R15_s42.npy and R14_seed42.npy have identical singular values (rel. 2e-9)
    AND identical row spaces (all principal-angle cosines = 1.0000), i.e. they
    are the same latent->expression map expressed in two different latent bases.
    The same holds for the s123 and s456 pairs.

If the same readout is counted twice, the effective number of independent
configurations is smaller than reported.

Method
------
For each matrix W (n_latent x n_genes) compute the right singular vectors V
(row-space basis, n_genes x n_latent) and compare row spaces pairwise within
each gene set (tissue): cosine of the principal angles is the SVD of Va' Vb,
and the row spaces coincide iff every cosine is 1 to numerical tolerance.
Also record exact byte-level md5 so trivial duplicate files are caught.

Outputs
-------
  data/loading_duplicate_check.json
"""
import glob
import hashlib
import json
import os

import numpy as np

LOAD = os.environ.get("LOADING_DIR",
                      "F:/UNAGI_Project/UNAGI/decoder_unified/loading")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "data", "loading_duplicate_check.json")
TOL = 1e-6


def md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()[:12]


def main():
    files = sorted(glob.glob(os.path.join(LOAD, "*.npy")))
    info = {}
    for p in files:
        name = os.path.basename(p)[:-4]
        W = np.load(p)
        if W.ndim == 1:
            W = W[None, :]
        # row-space basis
        _, sv, Vt = np.linalg.svd(W, full_matrices=False)
        info[name] = dict(shape=list(W.shape), md5=md5(p),
                          tr=float((W ** 2).sum()), sum_sv=float(sv.sum()))
        np.save(os.path.join(os.environ.get("TMP", "/tmp"),
                             f"_Vs_{name}.npy"), Vt[:64])

    # group by gene count (tissue proxy)
    by_g = {}
    for n, d in info.items():
        by_g.setdefault(d["shape"][1], []).append(n)

    dup_files, dup_rowspace = [], []
    for g, names in sorted(by_g.items()):
        Vs = {n: np.load(os.path.join(os.environ.get("TMP", "/tmp"),
                                      f"_Vs_{n}.npy")) for n in names}
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                if info[a]["md5"] == info[b]["md5"]:
                    dup_files.append([a, b])
                    continue
                k = min(Vs[a].shape[0], Vs[b].shape[0])
                cos = np.linalg.svd(Vs[a][:k] @ Vs[b][:k].T, compute_uv=False)
                if cos.min() > 1 - TOL:
                    dup_rowspace.append(dict(a=a, b=b, min_cosine=float(cos.min())))

    res = dict(
        n_files=len(files),
        groups_by_gene_count={str(g): sorted(v) for g, v in by_g.items()},
        byte_identical_files=dup_files,
        identical_row_space_pairs=dup_rowspace,
        note=("identical_row_space means the two readouts are the same "
              "latent->expression map expressed in different latent bases "
              "(same singular values, same row span) -- they are not "
              "independent evidence"),
        matrices={k: {kk: vv for kk, vv in v.items()} for k, v in info.items()},
    )
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w", encoding="utf-8"), indent=2)

    print("files:", len(files))
    for g, names in sorted(by_g.items()):
        print("  G=%-6s n=%d" % (g, len(names)))
    print("\nbyte-identical pairs: %d" % len(dup_files))
    for a, b in dup_files:
        print("   %s == %s" % (a, b))
    print("\nidentical row-space pairs: %d" % len(dup_rowspace))
    for d in dup_rowspace:
        print("   %-14s ~ %-14s  min cos = %.6f" % (d["a"], d["b"], d["min_cosine"]))
    print("\n[saved]", OUT)


if __name__ == "__main__":
    main()
