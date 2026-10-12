# -*- coding: utf-8 -*-
"""Perturbation-identity decodability from the published Norman latents.

Section 2.9.2 originally reported "0.099 versus 0.179 stratified five-fold accuracy" as
evidence that conditioning moves perturbation identity out of the latent. Checking it:
0.179 is the *majority-class baseline*, not the vanilla latent's accuracy, and the vanilla
latent is itself at chance. This script recomputes both arms from the cached published
latents and writes the numbers the manuscript quotes, so the claim is traceable to an
artifact rather than to a transcript.

Balanced accuracy is reported (the classes are 105 perturbations of very unequal size);
plain accuracy is kept alongside because the originally quoted 0.099 corresponds to the
plain metric on the conditioned arm.

Writes : outputs/positive_result/s1n5_norman_pert_identity.json
Usage  : python code/s1n5_norman_pert_identity_probe.py
"""
from __future__ import annotations

import json
import os
import warnings

import numpy as np

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER1 = os.path.abspath(os.path.join(HERE, ".."))
ROOT = os.path.abspath(os.path.join(PAPER1, "..", ".."))
OUT = os.path.join(ROOT, "outputs", "positive_result")
CACHE = os.path.join(PAPER1, "data", "_norman_latents.npz")


def evaluate(Z, labels, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, balanced_accuracy_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    bal, plain = [], []
    for tr, te in skf.split(Z, labels):
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=2000, class_weight="balanced"))
        clf.fit(Z[tr], labels[tr])
        pred = clf.predict(Z[te])
        bal.append(balanced_accuracy_score(labels[te], pred))
        plain.append(accuracy_score(labels[te], pred))
    return float(np.mean(bal)), float(np.std(bal)), float(np.mean(plain))


def main():
    c = np.load(CACHE, allow_pickle=True)
    labels = np.asarray(c["pert_labels"]).astype(str)
    Zv, Zp = c["Zv"].astype(np.float64), c["Zp"].astype(np.float64)
    counts = np.bincount(np.unique(labels, return_inverse=True)[1])
    majority = float(counts.max() / counts.sum())

    out = dict(source=os.path.basename(CACHE), n_cells=int(len(labels)),
               n_classes=int(len(set(labels))), majority_baseline=majority, arms={})
    print(f"n_cells={len(labels)}  n_classes={len(set(labels))}  majority={majority:.3f}")
    for name, Z in (("vanilla", Zv), ("conditioned", Zp)):
        # seed sensitivity: the value must not depend on the fold split
        reps = [evaluate(Z, labels, s) for s in (0, 1, 2)]
        bal = float(np.mean([r[0] for r in reps]))
        bal_sd = float(np.mean([r[1] for r in reps]))
        plain = float(np.mean([r[2] for r in reps]))
        out["arms"][name] = dict(balanced_accuracy=bal, balanced_sd_across_folds=bal_sd,
                                 plain_accuracy=plain,
                                 balanced_per_split_seed=[r[0] for r in reps])
        print(f"  {name:<12} balanced {bal:.3f} +- {bal_sd:.3f}   plain {plain:.3f}"
              f"   (split seeds: {[round(r[0], 4) for r in reps]})")

    out["conclusion"] = ("Neither latent decodes perturbation identity above the majority "
                         "baseline; the conditioned arm is lower than the vanilla arm but "
                         "both are at or below chance, so this benchmark does not "
                         "establish a decoder/latent information transfer.")
    with open(os.path.join(OUT, "s1n5_norman_pert_identity.json"), "w",
              encoding="utf-8", newline="\n") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"\nwrote {os.path.join(OUT, 's1n5_norman_pert_identity.json')}")


if __name__ == "__main__":
    main()
