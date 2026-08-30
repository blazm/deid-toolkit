#!/usr/bin/env python3
"""
Regenerate the manuscript pipeline figure (figs/pipeline_evaluation).

VERTICAL (top-down) flow, portrait aspect 160::201 mm. The figure is placed
at full \\linewidth (~170 mm wide) in the two-column manuscript, so it is
displayed at ~1.07x native scale -- fonts at 8.5-10.5 pt stay legible and all
label lines are width-checked to fit their boxes.

Content (current toolkit, 16 techniques), top to bottom:
  inputs -> identical preprocessing -> two frozen probes (SwinFace /
  TransFace) -> 512-d L2-normalized embeddings -> 7 analysis modules
  (same-condition verification, cross-condition linkability,
  identity-mapping structure, per-image displacement, embedding-space
  projection, compactness-separation, attribute utility) -> interpretation,
  summary & presentation.

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
C_MODULE = ("#fdeeee", "#d98a80")
C_OUTPUT = ("#f0f7f4", "#6f9c8a")
ARROW    = "#5a6672"
H_CANVAS = 216.0  # canvas height in mm (1 unit = 1 mm)

L = "\u2192"  # right arrow glyph


def rbox(ax, x, ytop, w, h, title, lines=(), colors=C_MODULE,
         title_fs=10.0, line_fs=8.2):
    """Box at ytop (canvas grows upward); title + up to 2-3 centered lines."""
    y = H_CANVAS - ytop - h
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.4,rounding_size=1.4",
                                facecolor=colors[0], edgecolor=colors[1],
                                linewidth=1.2, mutation_aspect=1.0))
    cx = x + w / 2
    flat = h >= 18                      # wide, low boxes: pull text up
    tf = 0.16 if flat else 0.22
    p2 = (0.48, 0.76) if flat else (0.55, 0.82)
    p3 = (0.40, 0.62, 0.84) if flat else (0.42, 0.66, 0.88)
    p1 = (0.58,) if flat else (0.62,)
    ax.text(cx, H_CANVAS - ytop - h * tf, title, ha="center", va="center",
            fontsize=title_fs, fontweight="bold", color="#1c2733")
    if len(lines) == 1:
        ax.text(cx, H_CANVAS - ytop - h * p1[0], lines[0], ha="center",
                va="center", fontsize=line_fs, color="#33404f")
    elif len(lines) == 2:
        ax.text(cx, H_CANVAS - ytop - h * p2[0], lines[0], ha="center",
                va="center", fontsize=line_fs, color="#33404f")
        ax.text(cx, H_CANVAS - ytop - h * p2[1], lines[1], ha="center",
                va="center", fontsize=line_fs, color="#33404f")
    elif len(lines) == 3:
        ax.text(cx, H_CANVAS - ytop - h * p3[0], lines[0], ha="center",
                va="center", fontsize=line_fs, color="#33404f")
        ax.text(cx, H_CANVAS - ytop - h * p3[1], lines[1], ha="center",
                va="center", fontsize=line_fs, color="#33404f")
        ax.text(cx, H_CANVAS - ytop - h * p3[2], lines[2], ha="center",
                va="center", fontsize=line_fs, color="#33404f")


def arrow(ax, x1, y1t, x2, y2t, lw=1.2):
    ax.annotate("", xy=(x2, H_CANVAS - y2t), xytext=(x1, H_CANVAS - y1t),
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
    W, H = 160.0, H_CANVAS        # canvas in mm (1 unit = 1 mm)
    # content spans x 8..152 and y 2..208 on the 216 mm canvas; crop to that extent
    XMIN, XMAX = 7.0, 153.0
    YMIN, YMAX = 7.0, 215.0
    fig = plt.figure(figsize=((XMAX - XMIN) / 25.4, (YMAX - YMIN) / 25.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(XMIN, XMAX)
    ax.set_ylim(YMIN, YMAX)
    ax.axis("off")
    ax.patch.set_visible(False)
    ax.set_frame_on(False)

    # ---- 1. inputs ----
    rbox(ax, 8, 2, 66, 22, "Original faces (reference)",
         ["4 datasets"], C_INPUT)
    rbox(ax, 86, 2, 66, 22, "De-identified faces",
         ["16 techniques", "4 datasets"], C_INPUT)
    arrow(ax, 41, 24.5, 41, 28.5)
    arrow(ax, 119, 24.5, 119, 28.5)

    # ---- 2. preprocessing ----
    rbox(ax, 8, 29, 144, 21, "Identical preprocessing",
         [f"BGR {L} RGB,  HWC {L} CHW",
          f"scale {L} crop {L} resize   (256 {L} 224 {L} 112)",
          f"[0, 255] {L} [\u22121, 1]"], C_PRE)
    arrow(ax, 41, 50.5, 41, 54.5)
    arrow(ax, 119, 50.5, 119, 54.5)

    # ---- 3. probes ----
    rbox(ax, 8, 55, 66, 22, "SwinFace",
         ["frozen Swin Transformer", "recognition + attributes"], C_PROBE)
    rbox(ax, 86, 55, 66, 22, "TransFace",
         ["frozen ViT backbone", "recognition only"], C_PROBE)
    arrow(ax, 41, 77.5, 41, 81.5)
    arrow(ax, 119, 77.5, 119, 81.5)

    # ---- 4. embeddings ----
    rbox(ax, 8, 82, 144, 20, "Embeddings",
         [".npy per image", "shape (512,),  L2-normalized",
          "shared 512-d space"], C_EMB)
    arrow(ax, 16, 102.5, 16, 112.5)
    arrow(ax, 66, 102.5, 66, 112.5)
    arrow(ax, 92, 102.5, 92, 112.5)
    arrow(ax, 146, 102.5, 146, 112.5)
    ax.text(41, H_CANVAS - 106.8, "Identity-level diagnostics", ha="center",
            va="center", fontsize=8.8, fontweight="bold", color="#4a5568")
    ax.text(41, H_CANVAS - 110.2, "(pair / person statistics)", ha="center",
            va="center", fontsize=7.3, color="#4a5568")
    ax.text(119, H_CANVAS - 106.8, "Representation diagnostics", ha="center",
            va="center", fontsize=8.8, fontweight="bold", color="#4a5568")
    ax.text(119, H_CANVAS - 110.2, "(geometry of the 512-d space)", ha="center",
            va="center", fontsize=7.3, color="#4a5568")

    # ---- 5. analysis modules (2-column rows, top-to-bottom) ----
    mh, gap = 15, 4
    rows = [
        [("Same-condition verification",
          ["ROC / AUC / EER (genuine-impostor pairs)"]),
         ("Cross-condition linkability",
          ["asymmetric 1:1 AUC/EER vs original", "open-set 1:N re-ID (rank, R@k)"])],
        [("Identity-mapping structure",
          ["1:1 / 1:N / N:1 drift & collapse", "split \u00b7 mixing \u00b7 K vs. P \u00b7 CMC-1"]),
         ("Per-image displacement",
          ["cosine distance per image", "violin plots / bar charts"])],
        [("Embedding-space projection",
          ["UMAP \u00b7 KDE density", "contour overlay of originals (2D)"]),
         ("Compactness-separation",
          ["intra/inter-person distances", "collapse ratio"])],
    ]
    mtop0 = 113
    for ri, row in enumerate(rows):
        ytop = mtop0 + ri * (mh + gap)
        rbox(ax, 8, ytop, 66, mh, row[0][0], row[0][1], C_MODULE,
             title_fs=9.8, line_fs=8.0)
        rbox(ax, 86, ytop, 66, mh, row[1][0], row[1][1], C_MODULE,
             title_fs=9.8, line_fs=8.0)
        if ri < len(rows) - 1:
            arrow(ax, 41, ytop + mh + 0.5, 41, ytop + mh + gap - 0.5)
            arrow(ax, 119, ytop + mh + 0.5, 119, ytop + mh + gap - 0.5)
    # row 4: attribute utility (full width)
    ay = mtop0 + 3 * (mh + gap)
    arrow(ax, 41, mtop0 + 2 * (mh + gap) + mh + 0.5, 41, ay - 0.5)
    arrow(ax, 119, mtop0 + 2 * (mh + gap) + mh + 0.5, 119, ay - 0.5)
    rbox(ax, 8, ay, 144, mh, "Attribute utility",
         ["gender / expression accuracy   \u00b7   confusion matrices"],
         C_MODULE, title_fs=9.8)

    # ---- 6. interpretation, summary & presentation ----
    iy = ay + mh + 5
    rbox(ax, 8, iy, 144, 18, "Interpretation \u00b7 summary \u00b7 presentation",
         ["behavioral classes in embedding space \u2014 scattering \u00b7 directed migration \u00b7 "
          "representation collapse",
          "identity mapping regimes (1:1 / 1:N / N:1) \u00b7 cross-condition re-identifiability"],
         C_OUTPUT, title_fs=9.8, line_fs=7.8)
    arrow(ax, 41, ay + mh + 0.5, 41, iy - 0.5)
    arrow(ax, 119, ay + mh + 0.5, 119, iy - 0.5)

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
