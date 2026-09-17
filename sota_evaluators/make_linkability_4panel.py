#!/usr/bin/env python3
"""
Double-column 2x2 cross-condition linkability figure for the two-column
manuscript.

Merges the two single-column figures of make_vertical_figures.py
(linkability_transface_2panel_v + linkability_scatter_v) into one wide
2x2 figure with panel letters (a)-(d), intended for a double-column
``figure*`` with one shared LaTeX caption (no per-figure suptitle):

    (a) TransFace 1:1 cross-condition EER -- horizontal bars, strongest
        unlinkability (highest EER) on top; dashed = aligned same-condition
        reference, dotted = chance (50%)
    (b) TransFace open-set 1:N R@1 -- horizontal bars, strongest
        unlinkability (lowest R@1) on top; dotted = chance 1/2824
    (c) 1:1 EER:  SwinFace vs TransFace cross-probe scatter (plain
        in-panel legend, y=x line, Spearman rho as legend title)
    (d) R@1:      SwinFace vs TransFace cross-probe scatter

No per-point scatter labels (crowded regions made them unreadable);
technique identity is carried by the in-panel legend, with the same
colors as the bar panels. Reads the same saved JSON results as
make_vertical_figures.py -- no model inference. Creates NEW output files
only; the existing *_2panel_v / *_scatter_v figures are untouched.

Usage:
    python make_linkability_4panel.py ^
        --link  D:\\dev\\deid-toolkit\\root_dir\\predictions ^
        --output ...\\figs\\linkability_4panel_2col
"""

import argparse
import json
import time

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from make_vertical_figures import compute_spearman


from fig_style import color_for, marker_for


def _savefig_safe(fig, path, **kw):
    for attempt in range(3):
        try:
            fig.savefig(path, bbox_inches="tight", **kw)
            return
        except PermissionError:
            if attempt == 2:
                print(f"  WARNING: could not save {path} (file locked?)")
                return
            time.sleep(0.5)


def export_figure(fig, base):
    _savefig_safe(fig, base + ".png", dpi=300)
    _savefig_safe(fig, base + ".pdf", format="pdf")
    return base + ".png"


def _bar_panel(ax, pairs, letter, title, vmax_line=None, vmax_label=None,
               chance=None, chance_label=None, fmt="{:.1f}"):
    """Horizontal bars, names on the left, value at the bar end.
    ``pairs`` = [(name, value), ...] already sorted so that the most
    notable (strongest unlinkability) entry is first -> drawn on top."""
    names = [n for n, _ in pairs]
    vals = [v for _, v in pairs]
    ys = np.arange(len(names))[::-1]
    ax.barh(ys, vals, color=[color_for(n) for n in names],
            edgecolor="white", height=0.72)
    for y, v in zip(ys, vals):
        ax.annotate(fmt.format(v), (v + 1.5, y), va="center", ha="left",
                    fontsize=4.4)
    ax.set_yticks(ys)
    ax.set_yticklabels(names, fontsize=5.2)
    for lb in ax.get_yticklabels():
        lb.set_ha("right")
    ax.set_ylim(-0.6, len(names) - 0.4)
    xlim = max(vals) * 1.24 + 2
    if vmax_line is not None:
        ax.axvline(vmax_line, color="black", linestyle="--", linewidth=1.0)
        ax.text(vmax_line + 0.8, -0.42, vmax_label, fontsize=4.4,
                va="top", ha="left", color="black")
        xlim = max(xlim, vmax_line * 3)
    if chance is not None:
        ax.axvline(chance, color="gray", linestyle=":", linewidth=1.1)
        ax.text(chance + 0.8, -0.42, chance_label, fontsize=4.4,
                va="top", ha="left", color="dimgray")
        xlim = max(xlim, chance * 1.06)
    ax.set_xlim(0, xlim)
    ax.set_title(f"({letter}) {title}", fontsize=8, fontweight="bold", pad=4)
    ax.tick_params(axis="both", labelsize=6, length=3, width=0.7)
    ax.grid(True, axis="x", alpha=0.3)


def _scatter_panel(ax, pts, letter, title, unit,
                   label_a="SwinFace", label_b="TransFace"):
    """Cross-probe scatter: plain dots + in-panel legend (top-left empty
    zone, clear of y=x and of all points in both panels), Spearman rho as
    legend title. ``pts`` = [(name, x, y), ...] in percent units, already
    in legend order."""
    names = [n for n, _, _ in pts]
    x = np.array([xi for _, xi, _ in pts])
    y = np.array([yi for _, _, yi in pts])
    for n, xi, yi in pts:
        ax.scatter(xi, yi, s=42, color=color_for(n), marker=marker_for(n),
                   edgecolor="k", linewidth=0.4, zorder=3)
    lim = max(x.max(), y.max()) * 1.12 + 2
    ax.plot([0, lim], [0, lim], "k--", linewidth=0.9, alpha=0.55)
    ax.annotate("y = x", (0.70 * lim, 0.70 * lim), xytext=(8, -10),
                textcoords="offset points", fontsize=4.6, color="dimgray")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel(f"{label_a} — {unit}", fontsize=7.0, fontweight="bold")
    ax.set_ylabel(f"{label_b} — {unit}", fontsize=7.0, fontweight="bold")
    ax.set_title(f"({letter}) {title}", fontsize=8, fontweight="bold", pad=4)
    ax.tick_params(axis="both", labelsize=6, length=3, width=0.7)
    ax.grid(True, alpha=0.3)

    rho, p = compute_spearman(x, y)
    ptxt = f"p={p:.2e}" if p < 0.05 else f"p={p:.3f}"
    handles = []
    for n in names:
        handles.append(Line2D([], [], marker=marker_for(n), linestyle="",
                              markerfacecolor=color_for(n), markeredgecolor="k",
                              markeredgewidth=0.5, markersize=5.0, label=n))
    # plain legend across the empty top-left quadrant; Spearman rho spans
    # the top as the legend title
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.02, 0.995),
              ncol=4, fontsize=3.7, title=f"Spearman $\\rho$ = {rho:.3f},  {ptxt}",
              title_fontsize=4.8,               framealpha=0.95, edgecolor="#cccccc",
              handlelength=0.9, handletextpad=0.4, columnspacing=1.0,
              labelspacing=0.4, borderpad=0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--link", required=True,
                    help="Dir containing linkability_<model>_results.json")
    ap.add_argument("--output", required=True,
                    help="Output base (no extension), e.g. .../figs/linkability_4panel_2col")
    args = ap.parse_args()

    A = json.load(open(f"{args.link}/linkability_swinface_results.json"))
    B = json.load(open(f"{args.link}/linkability_transface_results.json"))
    gal = B["gallery_size"]
    ref_eer = B["aligned_ref"]["eer"] * 100

    tech_b = [(n, d) for n, d in B["techniques"].items()]
    eers = [(n, d["eer"] * 100) for n, d in tech_b]
    r1s = [(n, d["r1"] * 100) for n, d in tech_b]
    # strongest unlinkability on top: (a) highest EER first, (b) lowest R@1 first
    eers.sort(key=lambda t: -t[1])
    r1s.sort(key=lambda t: t[1])
    # legend order = matching bar panel order
    eeer_order = {n: i for i, (n, _) in enumerate(eers)}
    r1_order = {n: i for i, (n, _) in enumerate(r1s)}

    common = set(A["techniques"]) & set(B["techniques"])
    pts_eer = [(n, A["techniques"][n]["eer"] * 100,
                B["techniques"][n]["eer"] * 100)
               for n in sorted(common, key=lambda n: eeer_order[n])]
    pts_r1 = [(n, A["techniques"][n]["r1"] * 100,
               B["techniques"][n]["r1"] * 100)
              for n in sorted(common, key=lambda n: r1_order[n])]

    fig, axes = plt.subplots(2, 2, figsize=(6.9, 5.7), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.055, h_pad=0.045)
    _bar_panel(axes[0, 0], eers, "a", "1:1 EER (TransFace)",
               vmax_line=ref_eer, vmax_label="aligned ref",
               chance=50.0, chance_label="chance 50%")
    _bar_panel(axes[0, 1], r1s, "b",
               f"1:N re-ID R@1 (TransFace, {gal}-face gallery)",
               chance=100.0 / gal, chance_label="chance ≈0.04%")
    _scatter_panel(axes[1, 0], pts_eer, "c",
                   "1:1 EER: cross-probe agreement", unit="EER (%)")
    _scatter_panel(axes[1, 1], pts_r1, "d",
                   "1:N re-ID R@1: cross-probe agreement", unit="R@1 (%)")

    out = export_figure(fig, args.output)
    plt.close(fig)
    print(f"  {out}")


if __name__ == "__main__":
    main()
