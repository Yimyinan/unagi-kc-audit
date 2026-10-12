#!/usr/bin/env python3
"""scan_idrem_geodesic.py -- Paper 1 geometric-dual-spectrum, Block A.2 (§5).

Executes the "geodesic / curvature stability of a target program direction under
intervention" protocol for the ERYTHROID-MATURATION program signature s.

Why this is now runnable (and where the hard constraint was)
----------------------------------------------------------
The §5 protocol needs the program signature s expressed in the SAME gene
vocabulary as the PG-UNAGI decoder readouts A = W.T (D x G).  Two pitfalls:
  * the neural PG-UNAGI VAE (R0/R1) is a mouse *neural* VAE -- its gene list is
    neuronal (STMN2, TUBB3, ...) and contains NO erythroid genes, so the
    erythroid direction is physically undefined there;
  * the erythroid-maturation signature shipped with Paper 2 is from the
    human GSE150774 erythroid VAE.
Resolution: the PG-UNAGI *hematopoietic* VAE (R11/R13/R15 family) uses a 20,466
human-style symbol vocabulary, and 4,983 / 5,000 of the Paper-2 signature genes
match it by exact symbol.  So s lives in the hematopoietic latent space with no
cross-species homolog mapping, and the erythroid direction is computed there.

What is computed (honest scope)
--------------------------------
Under the piecewise-linear (ReLU) decoder the readout Jacobian is locally
constant and equal to the Ridge readout W, so the program direction
    v_s = grad_z proj_s(D(z)) = W^T s = A s          (A = W.T, (D,G))
is a CONSTANT per model -- it does not depend on the wild-type code z0.  We
therefore report the cross-intervention invariance of v_s (pairwise cosine) as
the substantive §5 quantity:
  - seed resampling:  R11 KLF1/TAL1 overexpression, 3 seeds
  - training-iteration sweep:  R13 OE_KLF1, L_iter0..4
  - perturbation state:  OE_KLF1 vs KO_KLF1 (same master regulator)
Geodesic curvature is reported as DEGENERATE under the linear readout (the
metric G = A A^T is constant, so latent geodesics are straight and curvature is
0); recovering genuine curvature needs the full nonlinear decoder Jacobian at
z0, which is out of scope of the on-disk readouts.  This is stated explicitly.

Outputs
-------
  paper1/kc_trajectory/geodesic_erythroid.json
"""
from __future__ import annotations
import json, csv, glob, os
import numpy as np

# ----- paths ----------------------------------------------------------------
HEMA_GL   = r"F:/UNAGI_Project/UNAGI/decoder_out/hema_gene_list.json"
SIG_CSV   = r"F:/UNAGI_Project/paper2/positive_result/paper2_casestudy_erythroid_genes.csv"
DECODER_OUT = r"F:/UNAGI_Project/UNAGI/decoder_out"
UNIFIED   = r"F:/UNAGI_Project/UNAGI/decoder_unified/loading"
OUT = r"F:/UNAGI_Project/paper1/kc_trajectory/geodesic_erythroid.json"


def build_signature():
    gl = json.load(open(HEMA_GL))
    idx = {g: i for i, g in enumerate(gl)}
    rows = list(csv.DictReader(open(SIG_CSV)))
    rows = [r for r in rows if r["maturation_sig_z"] not in ("", "nan")]
    sig = {r["gene"].upper(): float(r["maturation_sig_z"]) for r in rows}
    s = np.zeros(len(gl), dtype=np.float64)
    mapped = 0
    for g, v in sig.items():
        j = idx.get(g)
        if j is not None:
            s[j] = v
            mapped += 1
    nz = np.linalg.norm(s)
    if nz > 0:
        s = s / nz
    return s, mapped, len(sig), gl


def load_A(path):
    a = np.load(path).astype(np.float64)
    if a.shape[0] > a.shape[1]:
        a = a.T
    return a  # (D, G)


def vdir(A, s):
    v = A @ s
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def cos_mat(V):
    """V: (n, D) unit rows -> (n, n) cosine matrix."""
    V = np.asarray(V)
    return V @ V.T


def summarize_cos(M, labels):
    n = len(labels)
    tri = []
    for i in range(n):
        for j in range(i + 1, n):
            tri.append(float(M[i, j]))
    tri = np.array(tri) if tri else np.array([np.nan])
    return dict(n=n, min=float(np.nanmin(tri)), median=float(np.nanmedian(tri)),
                max=float(np.nanmax(tri)), mean=float(np.nanmean(tri)),
                labels=labels)


def main():
    s, mapped, total_sig, gl = build_signature()
    print(f"[sig] G={len(gl)}  signature genes={total_sig}  mapped={mapped}  "
          f"coverage={mapped/total_sig:.3f}  ||s||_2=1")

    # ---- collect program directions v_s per intervention ----
    entries = []  # (group, label, v)

    # (1) seed resampling: R11 KLF1/TAL1 overexpression, 3 seeds (unified recompute)
    for p in sorted(glob.glob(os.path.join(UNIFIED, "R11_s*.npy"))):
        seed = os.path.basename(p)[4:-4]
        A = load_A(p)
        entries.append(("seed_R11_KLF1TAL1", f"R11_KLF1/TAL1 s{seed}", vdir(A, s)))

    # (2) iteration sweep: R13 OE_KLF1, L_iter0..4 (decoder_out on-disk trajectory)
    iters = sorted(glob.glob(os.path.join(DECODER_OUT,
                     "R13_OE_KLF1_seed42", "L_iter*.npy")))
    for p in iters:
        it = os.path.basename(p)[2:-4]
        A = load_A(p)
        entries.append(("iter_R13_OE_KLF1", f"R13_OE_KLF1 {it}", vdir(A, s)))

    # (3) perturbation state: OE vs KO of KLF1 (unified recompute)
    for tag, fn in [("OE_KLF1", "R13_OE_KLF1.npy"), ("KO_KLF1", "R13_KO_KLF1.npy")]:
        p = os.path.join(UNIFIED, fn)
        if os.path.exists(p):
            A = load_A(p)
            entries.append(("state", tag, vdir(A, s)))

    # also a vanilla-prior hematopoietic anchor (full hematopoiesis GRN, R15 s42)
    for fn in ["R15_s42.npy"]:
        p = os.path.join(UNIFIED, fn)
        if os.path.exists(p):
            A = load_A(p)
            entries.append(("anchor", "R15 hemo-prior s42", vdir(A, s)))

    groups = {}
    for g, lab, v in entries:
        groups.setdefault(g, []).append((lab, v))

    # ---- per-group cosine summaries ----
    out = dict(
        signature=dict(g_genes=len(gl), n_sig=total_sig, mapped=mapped,
                       coverage=mapped / total_sig,
                       source="paper2/positive_result/paper2_casestudy_erythroid_genes.csv (GSE150774 HBB-anchored maturation)"),
        groups={},
    )
    for g, items in groups.items():
        labels = [lab for lab, _ in items]
        V = np.stack([v for _, v in items])
        M = cos_mat(V)
        out["groups"][g] = summarize_cos(M, labels)
        print(f"[{g}] n={len(labels)} cosine min/med/max = "
              f"{out['groups'][g]['min']:.3f}/{out['groups'][g]['median']:.3f}/{out['groups'][g]['max']:.3f}")

    # ---- targeted contrasts ----
    by_lab = {lab: v for _, lab, v in entries}
    def cos(a, b):
        return float(np.dot(by_lab[a], by_lab[b]))

    contrasts = {}
    if "R13_OE_KLF1" in by_lab and "R13_KO_KLF1" in by_lab:
        contrasts["OE_vs_KO_KLF1"] = cos("R13_OE_KLF1", "R13_KO_KLF1")
    if "R15 hemo-prior s42" in by_lab and "R13_OE_KLF1" in by_lab:
        contrasts["OE_KLF1_vs_hemo_prior"] = cos("R13_OE_KLF1", "R15 hemo-prior s42")
    out["contrasts"] = contrasts
    for k, v in contrasts.items():
        print(f"[contrast] {k} cosine = {v:.3f}")

    # ---- full pairwise matrix (compact) ----
    all_lab = [lab for _, lab, _ in entries]
    all_V = np.stack([v for _, _, v in entries])
    out["full_cosine"] = dict(labels=all_lab, M=cos_mat(all_V).round(4).tolist())

    out["curvature_note"] = (
        "Geodesic curvature along v_s is degenerate under the piecewise-linear "
        "readout: G(z)=A A^T is constant, so the induced latent metric is flat "
        "and geodesics are straight (curvature 0).  The non-degenerate, reported "
        "invariant is the DIRECTION v_s and its cross-intervention cosine; genuine "
        "curvature requires the full nonlinear decoder Jacobian at the wild-type "
        "code z0 (out of scope of the on-disk readouts).")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
