# -*- coding: utf-8 -*-
"""Rewrite the 'Known open item' section of a frozen release README with the
cross-architecture null results of the 2026-09-18 line of work.

Usage: python code/update_release_readme.py [tag]
"""
from __future__ import annotations

import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER1 = os.path.abspath(os.path.join(HERE, ".."))

NEW = """## Cross-architecture nulls and paired designs (this release)

Both cross-architecture pairs now carry a family-matched reference measured in the same
configuration, tabulated in Supplementary Table S13 (`code/make_tabS13.py`).

* Norman 2019 scVI -- **paired design** (panel C). The injected arm was refit at four of
  the seeds that already had a protocol-matched baseline fit, so the per-seed ratio
  cancels the baseline scatter and the effect gets its own error bar. Readout
  kappa-C(W) = 0.902 +- 0.013 (t = -15.6 against unity, and the published single run
  sits on that mean, z = +0.3); latent kappa-C(Z) = 0.759 +- 0.069 (t = -7.0); their
  difference D = -0.143 +- 0.059 (t = -4.9). **What does not replicate: the published
  latent effective-rank drop (8.80 -> 6.15).** The paired rank ratio averages 1.02 +-
  0.15 (t = +0.2, per-seed values bracketing unity), so section 2.9.2 no longer reports
  it as a property of the configuration, Supplementary Table S12's rank row is now a
  paired ratio, and panel (b) of Supplementary Figure S21 has been redrawn as the paired
  ratio with seed scatter.
* Norman 2019 scVI -- baseline-only null (panel A) is retained as the weaker test; its
  latent-side verdicts move with the composition of the pool.
* Protocol match matters: the same seed trained as four 50-epoch chunks instead of one
  200-epoch call differs by 10% in readout predictability, 19% in effective rank and
  142% in the latent trace, so a null built from mixed schedules is not a null. One of
  four injected fits diverged near epoch 190 and was excluded (latent not finite).
* Family-4 MLP VAE -- five seeds with both arms retrained. The conditioned decoder
  compresses consistently (0.920 +- 0.017 corrected, 5/5 seeds below unity) but only
  slightly, and the single published run's near-unity effective-rank ratio (0.972) lies
  3.3 sigma above the five-seed estimate (0.868 +- 0.032).

## Known open item

The pbmc3k scVI control in Section 2.9.1 was corrected earlier in this line of releases:
the archived script registered the clusters with scVI's labels_key, which does not
condition the model, so its two arms were bit-identical. Section 2.9.1, Supplementary
Table S11 and the Methods note report the re-derived result. No further open items remain
in the cross-architecture subsections.
"""


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "2026-09-18_s1-corrected-r3"
    p = os.path.join(PAPER1, "releases", tag, "README.md")
    s = io.open(p, encoding="utf-8").read()
    i = s.find("## Known open item")
    assert i > 0, "anchor not found"
    io.open(p, "w", encoding="utf-8", newline="\n").write(s[:i] + NEW)
    print(f"updated {p}")


if __name__ == "__main__":
    main()
