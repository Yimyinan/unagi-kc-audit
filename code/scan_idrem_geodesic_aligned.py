#!/usr/bin/env python3
"""scan_idrem_geodesic_aligned.py -- Paper 1 geometric-dual-spectrum, Block A.2 (§5),
fair latent-basis-aligned upgrade.

The raw cross-seed cosine of the erythroid program direction v_s = A s is NOT a
fair invariance test: a VAE latent basis is defined only up to an orthogonal
rotation, so two seeds can encode the SAME program along rotated axes and still
show a low raw cosine.

We remove this ambiguity WITHOUT re-training or held-out cell encodings, using
only the on-disk decoder readouts A = W.T (D x G):

  1. Build a panel P of M dense random gene-program vectors p_m (G-dim); these
     span the latent readout space (A p_m is a generic latent direction).
  2. For two models A,B, stack panel readouts V_A (M x D), V_B (M x D).
  3. Estimate the latent-basis rotation R (orthogonal Procrustes) aligning
     V_B to V_A:  R = U Vt  from  svd(V_A V_B^T).
  4. Realign the erythroid direction:  v_s^B_aligned = R (A_B s).
  5. Report cosine(v_s^A, v_s^B_aligned)  -- the FAIR test.

If the aligned cosine jumps to ~1, the raw low cosine was pure basis ambiguity
(direction IS invariant); if it stays low, the direction is genuinely unstable.

Reuses build_signature / load_A / vdir from scan_idrem_geodesic.
"""
from __future__ import annotations
import json, os
import numpy as np

from scan_idrem_geodesic import (
    build_signature, load_A, vdir,
    UNIFIED, DECODER_OUT,
)

OUT = r"F:/UNAGI_Project/paper1/kc_trajectory/geodesic_erythroid_aligned.json"
PANEL_M = 600
RNG = np.random.default_rng(20260912)


def load_entries():
    """Return dict group -> list of (label, A). Mirrors scan_idrem_geodesic."""
    s, _, _, _ = build_signature()
    entries = {}
    # (1) seed resampling: R11 KLF1/TAL1, 3 seeds
    for p in sorted(__import__("glob").glob(os.path.join(UNIFIED, "R11_s*.npy"))):
        seed = os.path.basename(p)[4:-4]
        A = load_A(p)
        entries.setdefault("seed_R11_KLF1TAL1", []).append((f"R11 s{seed}", A, vdir(A, s)))
    # (2) iteration sweep: R13 OE_KLF1, L_iter0..4
    for p in sorted(__import__("glob").glob(os.path.join(DECODER_OUT,
                  "R13_OE_KLF1_seed42", "L_iter*.npy"))):
        it = os.path.basename(p)[2:-4]
        A = load_A(p)
        entries.setdefault("iter_R13_OE_KLF1", []).append((f"R13 {it}", A, vdir(A, s)))
    # (3) perturbation state: OE vs KO
    for tag, fn in [("OE_KLF1", "R13_OE_KLF1.npy"), ("KO_KLF1", "R13_KO_KLF1.npy")]:
        p = os.path.join(UNIFIED, fn)
        if os.path.exists(p):
            A = load_A(p)
            entries.setdefault("state", []).append((tag, A, vdir(A, s)))
    return entries, s


_PANEL = None  # shared fixed panel, generated once


def get_panel(G):
    global _PANEL
    if _PANEL is None or _PANEL.shape[1] != G:
        P = RNG.standard_normal((PANEL_M, G))
        P /= np.linalg.norm(P, axis=1, keepdims=True)
        _PANEL = P
    return _PANEL


def panel_matrix(A, G):
    P = get_panel(G)
    return P @ A.T  # (M, D) -- SAME panel P for every model


def procrustes_R(VA, VB):
    """Orthogonal R minimizing ||R VB - VA||_F."""
    H = VA.T @ VB  # (D, D)
    U, _, Vt = np.linalg.svd(H)
    R = U @ Vt
    if np.linalg.det(R) < 0:  # force proper rotation
        D = np.diag([1] * (R.shape[0] - 1) + [-1])
        R = U @ D @ Vt
    return R


def main():
    entries, s = load_entries()
    G = entries["seed_R11_KLF1TAL1"][0][1].shape[1]
    D = entries["seed_R11_KLF1TAL1"][0][1].shape[0]
    print(f"[setup] D={D} G={G} panel M={PANEL_M}")

    results = {}
    # ----- within-group pairwise: raw vs aligned -----
    for grp, items in entries.items():
        if len(items) < 2:
            continue
        pairs = []
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                lab_i, A_i, v_i = items[i]
                lab_j, A_j, v_j = items[j]
                raw = float(np.dot(v_i, v_j))
                # panel readouts
                VA = panel_matrix(A_i, G)
                VB = panel_matrix(A_j, G)
                R = procrustes_R(VA, VB)
                v_j_aligned = R @ v_j
                v_j_aligned /= (np.linalg.norm(v_j_aligned) or 1.0)
                aligned = float(np.dot(v_i, v_j_aligned))
                # self-consistency: panel alignment quality
                panel_cos = float(np.mean(
                    np.sum(VA * (VB @ R.T), axis=1) /
                    (np.linalg.norm(VA, axis=1) * np.linalg.norm(VB @ R.T, axis=1) + 1e-12)))
                pairs.append(dict(pair=f"{lab_i} | {lab_j}", raw=raw, aligned=aligned,
                                  panel_self_cos=panel_cos))
        results[grp] = pairs
        raws = [p["raw"] for p in pairs]
        alis = [p["aligned"] for p in pairs]
        print(f"[{grp}] n_pairs={len(pairs)}  raw med={np.median(raws):.3f}  "
              f"aligned med={np.median(alis):.3f}")

    # ----- raw vs aligned for the headline erythroid direction (R11 seeds & R13 iters) -----
    def aligned_vs_anchor(anchor_lab, grp):
        anchor = next(v for lab, _, v in entries[grp] if lab == anchor_lab)
        out = []
        for lab, A, v in entries[grp]:
            if lab == anchor_lab:
                continue
            raw = float(np.dot(anchor, v))
            VA = panel_matrix(entries[grp][0][1], G)  # use first A as ref panel base
            # build anchor panel from anchor's own A
            A_anchor = next(A_ for lab_, A_, _ in entries[grp] if lab_ == anchor_lab)
            VA = panel_matrix(A_anchor, G)
            VB = panel_matrix(A, G)
            R = procrustes_R(VA, VB)
            v_a = R @ v
            v_a /= (np.linalg.norm(v_a) or 1.0)
            out.append(dict(versus=lab, raw=raw, aligned=float(np.dot(anchor, v_a))))
        return out

    out = dict(
        method=("readout-constrained orthogonal Procrustes on a panel of "
                f"{PANEL_M} random gene-program directions; erythroid v_s "
                "realigned before cosine"),
        panel_M=PANEL_M,
        within_group=results,
    )
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
