# -*- coding: utf-8 -*-
"""
figure_palette.py -- SINGLE SOURCE OF TRUTH for paper1_kc_diagnostic figure styling.

Imports this module to guarantee that every manuscript figure shares one scientific
colour scheme, one typographic baseline, and one safe-legend helper so that legends
never sit on top of the data they describe.

Usage in a figure script
------------------------
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "code"))
    import figure_palette as fp
    fp.apply_paper_style()          # replaces the per-script rcParams block
    # then use the exported colour constants instead of hardcoded hex:
    ax.plot(x, y, color=fp.C_HEMA)
    fp.safe_legend(ax, loc="upper left")   # transparent, tucked in a corner

Colour semantics (kept identical to the legacy per-script hex values so existing
figures do not change appearance -- only future drift is prevented):
    C_HEMA   blue    hematopoietic tissue / encoder readout
    C_NEURAL orange  neural-crest tissue
    C_LUNG   green   lung-fibrosis tissue / decoder readout
    RED      red     "prior" runs, compression / warning accents
    GRAY     mid-grey  vanilla / control / neutral
    C4       purple  fourth categorical series
The palette is colour-blind-safe in the usual categorical sense (blue/orange/green
are distinguishable for the common deutran/mprotan deficits).
"""
from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# Unified scientific palette (exported names mirror what the scripts already use)
# ----------------------------------------------------------------------------
# Colours follow Okabe & Ito (2008), the colour-blind-safe standard.
C_HEMA = "#0072B2"   # blue   -- hematopoietic / encoder
C_NEURAL = "#D55E00" # orange -- neural crest
C_LUNG = "#009E73"   # green  -- lung / decoder
C_ENC = C_HEMA
C_DEC = C_LUNG
C1, C2, C3, C4 = C_HEMA, C_LUNG, "#C44E52", "#8172B2"
GRAY = "#444444"
GREY = "#6B6B6B"
RED = "#D55E00"
GREEN = "#009E73"
BAND = "#EDF3F8"   # vanilla-equivalence band background (used by S5/S22/S23/concept-latent)
VANILLA = "#444444"
PURPLE = "#CC79A7"

# ----------------------------------------------------------------------------
# SEMANTIC CONTRACT -- which colour may mean what.
#
# The figures had drifted because RED carried two incompatible meanings at once:
# "the prior-injected arm" in the graphical abstract and some scatter panels, and
# "this check FAILED / not prior-driven / withdrawn" in the decision flowchart.
# A reader who learns red = failure from Fig. 1 then meets red = prior in the
# graphical abstract. The contract below removes the overload: identity colours
# never carry a verdict, and verdict colours never encode identity.
#
#   IDENTITY  (categorical, always)   tissue / arm / model
#   VERDICT   (evaluative, reserved)  pass / fail / caution
# ----------------------------------------------------------------------------
# --- verdict semantics: RESERVED. Never use these for a series identity.
FAIL = "#D55E00"      # red    -- a check failed; not prior-driven; withdrawn
PASS = "#009E73"      # green  -- a check passed; prior-active; reproduces
CAUTION = "#E69F00"   # amber  -- marginal: directional corroboration only

# --- arm identity: prior-injected vs control (no verdict implied)
ARM_VANILLA = "#444444"   # dark grey  -- vanilla / prior-free baseline
ARM_PRIOR = "#CC79A7"     # purple     -- prior-injected or ablation arm
ARM_PRIOR_ALT = "#E8C4D8"  # light purple -- second prior variant, same hue family

# --- verdict / band fills (translucent versions of the verdict colours)
FAIL_FILL = "#FBE6D8"
PASS_FILL = "#DDF0E8"
BAND_NULL = "#EDF3F8"     # family-matched null / vanilla-equivalence band
BAND_CROSS = "#CFE0EC"    # cross-family band (darker sibling of BAND_NULL)

INK = "#111111"           # body text
EDGE = "#333333"          # box outlines
MUTED = "#6B6B6B"         # secondary annotation text

TISSUE_COLOR = {"hemo": C_HEMA, "neural": C_NEURAL, "lung": C_LUNG}
TISSUE_LABEL = {
    "hemo": "Hematopoietic",
    "neural": "Neural crest",
    "lung": "Lung fibrosis",
}

# ----------------------------------------------------------------------------
# Typographic / aesthetic baseline -- identical across every figure
# ----------------------------------------------------------------------------
_PAPER_RCPARAMS = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7.5,
    "axes.titlesize": 9.0,
    "axes.labelsize": 8.0,
    "axes.titleweight": "bold",
    "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.0,
    "legend.fontsize": 6.5,
    "xtick.major.size": 2.4,
    "ytick.major.size": 2.4,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "axes.unicode_minus": False,
    "lines.linewidth": 1.2,
    "lines.markersize": 3.4,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    # --- submission-grade export (editable vector text, required by journals) ---
    "pdf.fonttype": 42,        # embed TrueType fonts -> editable text in PDF
    "svg.fonttype": "none",    # editable text in SVG
    "pdf.compression": True,
    # --- clean journal style (Nature/Cell-style defaults) ---
    "legend.frameon": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 1.0,
}


def apply_paper_style() -> None:
    """Install the unified matplotlib rcParams. Call once after `import matplotlib`."""
    plt.rcParams.update(_PAPER_RCPARAMS)


def style_ax(ax) -> None:
    """Despine top/right and use outward ticks (paper convention)."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction="out", length=2.4, width=0.6)


def safe_legend(ax, loc="upper left", bbox_to_anchor=None, ncol=1,
                fontsize=6.5, frameon=False, handlelength=1.1, **kw):
    """Draw a legend that never occludes data.

    Default places a transparent legend in a corner. Pass ``bbox_to_anchor`` to
    push it fully outside the axes (e.g. (1.02, 1.0) to the right, or
    (0.5, -0.20) below). With ``savefig(bbox_inches='tight')`` outside legends are
    captured without cropping.
    """
    if bbox_to_anchor is not None:
        ax.legend(loc=loc, bbox_to_anchor=bbox_to_anchor, ncol=ncol,
                  fontsize=fontsize, frameon=frameon, handlelength=handlelength, **kw)
    else:
        ax.legend(loc=loc, ncol=ncol, fontsize=fontsize,
                  frameon=frameon, handlelength=handlelength, **kw)


def legend_out_right(ax, fontsize=6.5, handlelength=1.1):
    """Convenience: legend to the right of the axes (needs tight-bbox savefig)."""
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0),
              fontsize=fontsize, frameon=False, handlelength=handlelength)


def legend_below(ax, ncol=3, fontsize=6.5, handlelength=1.1):
    """Convenience: legend centred below the axes (needs tight-bbox savefig)."""
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20),
              ncol=ncol, fontsize=fontsize, frameon=False, handlelength=handlelength)


def save_paper(fig, path_pdf, path_png=None, dpi=300):
    """Save a vector PDF and (optionally) a 300-dpi PNG with tight bbox."""
    fig.savefig(path_pdf, dpi=dpi, bbox_inches="tight")
    if path_png is not None:
        fig.savefig(path_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
#  Publication geometry  (added 2026-09-29, r31)
#
#  Problem this fixes: five of the seven main figures were generated far wider
#  than the printed column (up to 14.8 in) and placed at \textwidth = 6.69 in.
#  LaTeX scales the WHOLE figure down, so fonts and line widths shrink with it:
#      fig4_prior_activity   14.75 in -> 6.69 in  (x0.45)  7.5 pt -> 3.4 pt
#      fig2_kc_phase_diagram 13.41 in -> 6.69 in  (x0.50)  7.5 pt -> 3.7 pt
#  i.e. below any journal minimum for figure text and stroke weight.
#
#  fit_to_width() scales every visual element by the same factor the figure will
#  later be shrunk by, so the printed result carries the ORIGINAL design sizes.
#  It also enforces a floor on stroke widths (min_lw) so nothing prints too thin.
# ============================================================================
TEXT_WIDTH_IN = 6.6929          # main.tex: 21 cm - 2 cm - 2 cm = 17 cm


def panel_label(ax, letter, x=-0.015, y=1.02, size=9.0):
    """Bold UPPERCASE panel letter, kept SEPARATE from the panel title."""
    return ax.text(x, y, "(%s)" % letter.upper(), transform=ax.transAxes,
                   fontsize=size, fontweight="bold", va="bottom", ha="left",
                   color=INK, clip_on=False)


def _scale_artist(a, k, min_lw):
    if hasattr(a, "get_linewidth"):
        try:
            lw = a.get_linewidth()
            if isinstance(lw, (int, float)):
                a.set_linewidth(max(lw * k, min_lw))
        except Exception:
            pass
    for g, st in (("get_markersize", "set_markersize"),
                  ("get_markeredgewidth", "set_markeredgewidth")):
        if hasattr(a, g):
            try:
                setattr(a, st, getattr(a, g)() * k)
            except Exception:
                pass
    if hasattr(a, "get_fontsize"):
        try:
            a.set_fontsize(a.get_fontsize() * k)
        except Exception:
            pass


def fit_to_width(fig, gen_width_in=None, final_width_in=TEXT_WIDTH_IN, min_lw=0.5):
    """Scale all visual elements by (generated width / printed width).

    Returns the scale factor k. Idempotent-ish: calling twice would double-scale,
    so call it exactly once, immediately before savefig.
    """
    # gen_width_in: MEASURED printed-at-generated width (pdfinfo). bbox_inches='tight'
    # can make the saved width differ a lot from figsize (e.g. canvas 6.35in -> 10.70in),
    # so the measured value is authoritative when supplied.
    gen = float(gen_width_in) if gen_width_in else fig.get_size_inches()[0]
    k = gen / float(final_width_in)
    if k <= 1.001:
        return 1.0
    for ax in fig.axes:
        for a in list(ax.lines) + list(ax.patches) + list(ax.texts) + list(ax.collections):
            _scale_artist(a, k, min_lw)
        for sp in ax.spines.values():
            try:
                sp.set_linewidth(max(sp.get_linewidth() * k, min_lw))
            except Exception:
                pass
        for axis in (ax.xaxis, ax.yaxis):
            for t in list(axis.get_major_ticks()) + list(axis.get_minor_ticks()):
                for attr in ("label1", "label2"):   # XTick/YTick 用 label1/2，没有 .label
                    lab = getattr(t, attr, None)
                    if lab is not None:
                        lab.set_fontsize(lab.get_fontsize() * k)
                for ln in (t.tick1line, t.tick2line):
                    if ln is not None:
                        ln.set_markersize(ln.get_markersize() * k)
                        ln.set_markeredgewidth(max(ln.get_markeredgewidth() * k, min_lw))
            axis.label.set_fontsize(axis.label.get_fontsize() * k)
        ax.title.set_fontsize(ax.title.get_fontsize() * k)
        lg = ax.get_legend()
        if lg is not None:
            for t in lg.get_texts():
                t.set_fontsize(t.get_fontsize() * k)
            try:
                for ln in lg.get_lines():
                    ln.set_linewidth(max(ln.get_linewidth() * k, min_lw))
            except Exception:
                pass
    for t in fig.texts:
        t.set_fontsize(t.get_fontsize() * k)
    # ★ 放大字号后必须**重新排版**：脚本里的 tight_layout() 是按原字号算的，
    #   放大会让面板间/面板内的文字互相压住（fig4 首版就出现了）。
    try:
        fig.tight_layout()
    except Exception:
        pass
    return k
