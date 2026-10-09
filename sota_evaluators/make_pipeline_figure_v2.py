#!/usr/bin/env python3
"""
V2 of the manuscript pipeline figure (figs/pipeline_evaluation_v2).

Changes vs make_pipeline_figure.py (v1):
  * pair/labels inputs made explicit: "one group per person" in both input
    nodes; attribute labels named in the utility axis row;
  * preprocessing box reduced to the protocol claim (identical for both
    conditions, detector-free) -- no formats/resolutions/channels;
  * analysis panel restructured as the FOUR evaluation axes of the paper
    (privacy / embedding structure / attribute utility / naturalness),
    with the joint-fit projection called out in the structure axis;
  * output box reduced to "visualisations & results + strategy family".

Same canvas (237 x 76 mm), same palette, same six-column layout, so the
two-column float geometry in the manuscript does not change. Renders
png/pdf/svg; does NOT touch the v1 files.

Usage:
    python make_pipeline_figure_v2.py [--out <file stem, no extension>]
"""

import argparse
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# palette: (face, edge) -- unchanged from v1
C_INPUT  = ("#fff4d6", "#d0a83c")
C_PRE    = ("#e8f4fd", "#6fa8cc")
C_PROBE  = ("#e6f4ea", "#5da87a")
C_EMB    = ("#f0e8fa", "#9c7ec9")
C_PANEL  = ("#f7f9fb", "#b8c4d0")
C_MODULE = ("#fdeeee", "#d98a80")
C_OUTPUT = ("#f0f7f4", "#6f9c8a")
ARROW    = "#5a6672"

W, H = 237.0, 76.0
L = "\u2192"

# stage x-positions / widths -- unchanged from v1
X1, W1 = 5.0,   28.0   # inputs
X2, W2 = 37.5, 36.0    # preprocessing
X3, W3 = 78.0,  28.0   # probes
X4, W4 = 110.5, 28.0   # embeddings
X5, W5 = 143.0, 56.0   # analysis panels
X6, W6 = 203.5, 28.0   # output

Y0, Y1 = 5.0, 71.0
STACK = [(5.0, 27.0), (39.0, 27.0)]


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
                    fontsize=title_fs, fontweight="bold", color=title_color)
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


def arrow(ax, x1, y1, x2, y2, lw=1.2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=ARROW,
                                lw=lw, shrinkA=0, shrinkB=0))


def elbow_arrow(ax, x1, y1, xc, y2, x2, lw=1.2):
    """Right-angle connector: right to xc, turn, then arrow into (x2, y2)."""
    ax.plot([x1, xc], [y1, y1], color=ARROW, lw=lw, solid_capstyle="round")
    ax.plot([xc, xc], [y1, y2], color=ARROW, lw=lw, solid_capstyle="round")
    arrow(ax, xc, y2, x2, y2, lw=lw)


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


def axis_block(ax, x, y_name, w, name, subs):
    """One evaluation-axis row inside the analysis panel: bold name + content."""
    bottom = y_name - 4.2 * ((len(subs) - 1) + 1) - 3.2
    ax.add_patch(FancyBboxPatch((x, bottom), w, (y_name + 3.4) - bottom,
                                boxstyle="round,pad=0.4,rounding_size=1.2",
                                facecolor=C_MODULE[0], edgecolor=C_MODULE[1],
                                linewidth=1.0, alpha=0.92))
    ax.text(x + w / 2, y_name, name, ha="center", va="center",
            fontsize=8.8, fontweight="bold", color="#1c2733")
    for i, s in enumerate(subs):
        ax.text(x + w / 2, y_name - 4.2 * (i + 1), s, ha="center", va="center",
                fontsize=8.2, color="#33404f")
    return bottom


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
    yc = (Y0 + Y1) / 2

    # ---- 1. inputs (with the per-person grouping made explicit) ----
    yb, h = STACK[1]
    rbox(ax, X1, yb, W1, h, "Original faces",
         ["reference", "4 datasets", "one group", "per person"],
         C_INPUT, line_fs=8.0)
    yb, h = STACK[0]
    rbox(ax, X1, yb, W1, h, "De-identified",
         ["20 techniques", "4 datasets", "one group", "per person"],
         C_INPUT, line_fs=8.0)
    arrow(ax, X1 + W1, yc1, X2, yc1)
    arrow(ax, X1 + W1, yc2, X2, yc2)

    # ---- 2. metadata (top) + preprocessing (bottom) ----
    rbox(ax, X2, 39.0, W2, Y1 - 39.0, "Metadata",
         ["1:1 matched pairs;", "ground-truth labels", "(gender, expression)"],
         C_INPUT)
    rbox(ax, X2, Y0, W2, 34.0, "Preprocessing",
         ["identical for both", "pre-aligned input;", "no re-detection,", "no re-alignment"],
         C_PRE)
    elbow_arrow(ax, X1 + W1, yc2, 35.4, 28.0, X2)
    arrow(ax, X1 + W1, yc1, X2, yc1)

    # ---- 3. probes (independent, frozen) ----
    arrow(ax, X2 + W2, yc1, X3, yc1)
    elbow_arrow(ax, X2 + W2, 28.0, 75.6, yc2, X3)
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
         ["512-d,", "L2-normalized", "per probe"], C_EMB)
    arrow(ax, X4 + W4, yc, X5, yc)

    # ---- 5. analysis panel: the four evaluation axes ----
    ax.add_patch(FancyBboxPatch((X5, Y0), W5, Y1 - Y0,
                                boxstyle="round,pad=0.5,rounding_size=2.0",
                                facecolor=C_PANEL[0], edgecolor=C_PANEL[1],
                                linewidth=1.3))
    ax.text(X5 + W5 / 2, Y1 - 4.5, "Four evaluation axes",
            ha="center", va="center", fontsize=9.5, fontweight="bold",
            color="#1c2733")
    axes_rows = [
        ("Privacy",
         ["same-condition verification", "cross-condition linkability"]),
        ("Embedding structure",
         ["1:1 / 1:N / N:1 \u00b7 displacement", f"collapse \u00b7 UMAP {L} one joint fit"]),
        ("Attribute utility",
         ["gender \u00b7 expression (labels)"]),
        ("Naturalness & image quality",
         ["LPIPS \u00b7 IQA \u00b7 FID"]),
    ]
    ytop = Y1 - 10.0
    for i, (name, subs) in enumerate(axes_rows):
        bottom = axis_block(ax, X5 + 2.5, ytop, W5 - 5.0, name, subs)
        ytop = bottom - 3.5

    # ---- 6. output ----
    rbox(ax, X6, Y0, W6, Y1 - Y0, "Outputs",
         ["visuals and results", "strategy family", "assignment",
          "1:1 / 1:N / N:1", "linkability"], C_OUTPUT, line_fs=8.0)
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
                           r"\figs\pipeline_evaluation_v2")
    args = p.parse_args()
    make_pipeline_figure(os.path.abspath(args.out))


if __name__ == "__main__":
    main()
