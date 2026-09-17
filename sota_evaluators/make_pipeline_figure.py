#!/usr/bin/env python3
"""
Regenerate the manuscript pipeline figure (figs/pipeline_evaluation).

HORIZONTAL (left-to-right) flow, landscape aspect 237::76 mm. The figure is
placed in a full \\textwidth (two-column) \\begin{figure*}[t] in the
manuscript, so it is displayed at ~0.70x native scale -- fonts at 8.5 pt
native stay at ~6 pt display and all label lines are width-checked to fit
their boxes. (The previous vertical version was 23.6 cm tall at full width,
which overfulled the text block and got clipped; the horizontal layout is
~5.3 cm tall at \\textwidth and parks cleanly at the top of a page as a
two-column float.)

Content (current toolkit, 20 techniques), left to right:
  inputs (originals / de-identified) -> identical preprocessing ->
  two frozen probes (SwinFace / TransFace) -> 512-d L2-normalized
  embeddings -> 7 analysis modules (same-condition verification,
  cross-condition linkability, identity-mapping structure,
  per-image displacement, embedding-space projection,
  compactness-separation, attribute utility) -> interpretation/output.

Usage:
    python make_pipeline_figure.py [--out <file stem, no extension>]
"""

import argparse
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# palette: (face, edge)
C_INPUT  = ("#fff4d6", "#d0a83c")
C_PRE    = ("#e8f4fd", "#6fa8cc")
C_PROBE  = ("#e6f4ea", "#5da87a")
C_EMB    = ("#f0e8fa", "#9c7ec9")
C_PANEL  = ("#f7f9fb", "#b8c4d0")
C_MODULE = ("#fdeeee", "#d98a80")
C_OUTPUT = ("#f0f7f4", "#6f9c8a")
ARROW    = "#5a6672"

W, H = 237.0, 76.0   # canvas in mm (1 unit = 1 mm)
L = "\u2192"          # right arrow glyph

# stage x-positions / widths
X1, W1 = 5.0,   28.0   # inputs
X2, W2 = 37.5, 36.0    # preprocessing
X3, W3 = 78.0,  28.0   # probes
X4, W4 = 110.5, 28.0   # embeddings
X5, W5 = 143.0, 56.0   # analysis panels
X6, W6 = 203.5, 28.0   # output

Y0, Y1 = 5.0, 71.0     # full-stage vertical extent
STACK = [(5.0, 27.0), (39.0, 27.0)]  # stacked pair boxes: (ybottom, h)


def rbox(ax, x, y, w, h, title, lines=(), colors=C_MODULE,
         title_fs=9.0, line_fs=8.5, title_color="#1c2733"):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.5,rounding_size=1.6",
                                facecolor=colors[0], edgecolor=colors[1],
                                linewidth=1.2))
    cx = x + w / 2
    lines = list(lines)
    if title:
        if "\n" in title:
            ax.text(cx, y + h - 6.0, title, ha="center", va="center",
                    fontsize=title_fs, fontweight="bold",
                    color=title_color)
            body_top = y + h - 12.0
        else:
            ax.text(cx, y + h - 4.0, title, ha="center", va="center",
                    fontsize=title_fs, fontweight="bold", color=title_color)
            body_top = y + h - 8.0
    else:
        body_top = y + h - 4.0
    for i, ln in enumerate(lines):
        ax.text(cx, body_top - i * 4.4, ln, ha="center", va="center",
                fontsize=line_fs, color="#33404f")


def hrow(ax, x, y, w, text, fs=8.5):
    """Single centered text line inside a slim round row."""
    ax.add_patch(FancyBboxPatch((x, y), w, 6.2,
                                boxstyle="round,pad=0.4,rounding_size=1.2",
                                facecolor=C_MODULE[0], edgecolor=C_MODULE[1],
                                linewidth=1.0))
    ax.text(x + w / 2, y + 3.1, text, ha="center", va="center",
            fontsize=fs, color="#1c2733")


def arrow(ax, x1, y1, x2, y2, lw=1.2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=ARROW,
                                lw=lw, shrinkA=0, shrinkB=0))


def _savefig_safe(fig, path, **kw):
    for attempt in range(3):
        try:
            fig.savefig(path, **kw)
            return
        except PermissionError:
            if attempt == 2:
                print(f"  WARNING: could not save {path} (file locked?)")
                return
            time.sleep(0.5)


def make_pipeline_figure(out_stem):
    XMIN, XMAX = 3.0, W - 3.0
    YMIN, YMAX = 2.5, H - 2.5
    fig = plt.figure(figsize=((XMAX - XMIN) / 25.4, (YMAX - YMIN) / 25.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(XMIN, XMAX)
    ax.set_ylim(YMIN, YMAX)
    ax.axis("off")
    ax.patch.set_visible(False)
    ax.set_frame_on(False)

    yc1, yc2 = STACK[0][0] + 13.5, STACK[1][0] + 13.5   # stacked centers
    yc = (Y0 + Y1) / 2                                  # full-stage center

    # ---- 1. inputs ----
    yb, h = STACK[1]
    rbox(ax, X1, yb, W1, h, "Original faces",
         ["(reference)", "4 datasets"], C_INPUT, line_fs=8.5)
    yb, h = STACK[0]
    rbox(ax, X1, yb, W1, h, "De-identified",
         ["faces", "20 techniques", "4 datasets"], C_INPUT, line_fs=8.5)
    arrow(ax, X1 + W1, yc1, X2, yc1)
    arrow(ax, X1 + W1, yc2, X2, yc2)

    # ---- 2. preprocessing ----
    rbox(ax, X2, Y0, W2, Y1 - Y0, "Identical\npreprocessing",
         [f"BGR {L} RGB,", f"HWC {L} CHW", f"scale {L} crop {L}", "resize",
          f"256 {L} 224 {L} 112", f"[0,255] {L} [\u22121,1]"], C_PRE)
    arrow(ax, X2 + W2, yc1, X3, yc1)
    arrow(ax, X2 + W2, yc2, X3, yc2)

    # ---- 3. probes ----
    yb, h = STACK[1]
    rbox(ax, X3, yb, W3, h, "SwinFace",
         ["frozen Swin", "Transformer,", "identity +", "attributes"],
         C_PROBE, line_fs=8.5)
    yb, h = STACK[0]
    rbox(ax, X3, yb, W3, h, "TransFace",
         ["frozen ViT", "backbone,", "identity only"], C_PROBE, line_fs=8.5)
    arrow(ax, X3 + W3, yc1, X4, yc1)
    arrow(ax, X3 + W3, yc2, X4, yc2)

    # ---- 4. embeddings ----
    rbox(ax, X4, Y0, W4, Y1 - Y0, "Embeddings",
         ["512-d vectors", ".npy per image", "(512,) L2-norm", "per probe"],
         C_EMB)
    arrow(ax, X4 + W4, yc, X5, yc)

    # ---- 5. analysis panels ----
    ax.add_patch(FancyBboxPatch((X5, Y0), W5, Y1 - Y0,
                                boxstyle="round,pad=0.5,rounding_size=2.0",
                                facecolor=C_PANEL[0], edgecolor=C_PANEL[1],
                                linewidth=1.3))
    ax.text(X5 + W5 / 2, Y1 - 4.5, "Analysis modules",
            ha="center", va="center", fontsize=9.0, fontweight="bold",
            color="#1c2733")
    modules = [
        "Same-condition verification",
        "Cross-condition linkability",
        "Identity-mapping structure",
        "Per-image displacement",
        "Embedding-space projection",
        "Compactness\u2013separation",
        "Attribute utility",
    ]
    mh, mgap = 6.2, 2.0
    top = Y1 - 10.0
    for i, m in enumerate(modules):
        hrow(ax, X5 + 2.5, top - i * (mh + mgap) - mh + 0.0, W5 - 5.0, m)

    # ---- 6. output ----
    rbox(ax, X6, Y0, W6, Y1 - Y0, "Interpretation",
         ["displacement", "patterns in", "embedding", "space",
          "\u00b7 1:1 / 1:N / N:1", "regimes", "\u00b7 linkability"],
         C_OUTPUT, line_fs=8.0)
    arrow(ax, X5 + W5, yc, X6, yc)

    _savefig_safe(fig, out_stem + ".png", dpi=300)
    _savefig_safe(fig, out_stem + ".pdf", format="pdf")
    _savefig_safe(fig, out_stem + ".svg", format="svg")
    plt.close(fig)
    print(f"Saved: {out_stem}.png / .pdf / .svg")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=str,
                   default=r"D:\dev\deid-toolkit_additional-materials"
                           r"\deid-toolkit_manuscript_Neurocomputing_Software_track"
                           r"\figs\pipeline_evaluation")
    args = p.parse_args()
    make_pipeline_figure(os.path.abspath(args.out))


if __name__ == "__main__":
    main()
