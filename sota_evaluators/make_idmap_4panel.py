#!/usr/bin/env python3
"""
Double-column 2x2 identity-mapping figure for the two-column manuscript.

Merges the four single-column panels of make_vertical_figures.py into one
wide figure with panel letters (a)-(d), intended for a double-column
``figure*`` with one shared LaTeX caption (no per-figure suptitle):

    (a) SwinFace  -- quadrant: person-split rate (1:N) vs mixing rate (N:1)
    (b) SwinFace  -- CMC-1 cross-identity match bars
    (c) TransFace -- quadrant: person-split rate (1:N) vs mixing rate (N:1)
    (d) TransFace -- CMC-1 cross-identity match bars

Reads the same saved JSON results as make_vertical_figures.py -- no model
inference. Creates NEW output files only; the existing *_2panel_v figures
are untouched.

Usage:
    python make_idmap_4panel.py ^
        --idmap D:\\dev\\deid-toolkit\\root_dir\\predictions ^
        --output ...\\figs\\id_mapping_4panel_2col
"""

import argparse
import json
import os
import time

import numpy as np

import matplotlib
matplotlib.use("Agg")
from matplotlib.lines import Line2D


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


def plot_quadrant(ax, rows, letter, model):
    """One split-vs-mixing quadrant panel. Technique identity is carried by
    a bottom legend spanning the empty band (colors match the bar panels)."""
    tech = [(n, d) for n, d in rows.items() if n != "aligned"]
    ref = rows["aligned"]
    s = {n: d["person_split_rate"] * 100 for n, d in tech}
    m = {n: d["mixing_rate"] * 100 for n, d in tech}
    # same order as the CMC-1 bar panel (by value, strongest on top)
    names = [n for n, _ in sorted(tech, key=lambda x: -x[1]["cmc1_misassignment"])]
    for n in names:
        ax.scatter(s[n], m[n], s=42, color=color_for(n),
                   marker=marker_for(n), edgecolor="k",
                   linewidth=0.4, zorder=3)
    refpt = (ref["person_split_rate"] * 100, ref["mixing_rate"] * 100)
    ax.scatter(*refpt, marker="o", s=70, facecolor="none", edgecolor="black",
               linewidth=1.4, zorder=4)
    ax.axhline(50, color="gray", linestyle=":", linewidth=0.9)
    ax.axvline(50, color="gray", linestyle=":", linewidth=0.9)
    lim = max([max(s.values()), max(m.values()), refpt[0], refpt[1]]) * 1.22 + 6
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    # regime labels pinned to the four corners (axes-fraction coords)
    kw = dict(fontsize=5.5, fontweight="bold", transform=ax.transAxes, va="top")
    ax.text(0.02, 0.97, "1:1 consistent", ha="left", color="#336633", **kw)
    ax.text(0.98, 0.97, "M:N mixed", ha="right", color="#663366", **kw)
    ax.text(0.02, 0.30, "N:1 collapse", ha="left", va="bottom",
            transform=ax.transAxes, fontsize=5.5, fontweight="bold", color="#993333")
    ax.text(0.98, 0.30, "1:N drift", ha="right", va="bottom",
            transform=ax.transAxes, fontsize=5.5, fontweight="bold", color="#996633")
    ax.set_xlabel("Person-split rate (%)", fontsize=7.5, fontweight="bold")
    ax.set_ylabel("Cluster mixing rate (%)", fontsize=7.5, fontweight="bold")
    ax.set_title(f"({letter}) {model}: mapping structure",
                 fontsize=8, fontweight="bold", pad=4)
    ax.tick_params(axis="both", labelsize=6, length=3, width=0.7)
    ax.grid(True, alpha=0.3)
    # legend across the empty bottom band; open circle = Reference
    handles = [Line2D([], [], marker="o", linestyle="", markerfacecolor="none",
                      markeredgecolor="k", markeredgewidth=1.4, markersize=5.5,
                      label="Reference")]
    for n in names:
        handles.append(Line2D([], [], marker=marker_for(n), linestyle="",
                              markerfacecolor=color_for(n), markeredgecolor="k",
                              markeredgewidth=0.5, markersize=5.5, label=n))
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.005),
              ncol=4, fontsize=4.3, framealpha=0.95, edgecolor="#cccccc",
              handlelength=1.0, handletextpad=0.45, columnspacing=1.1,
              labelspacing=0.4, borderpad=0.4)


def plot_cmc1(ax, rows, letter, model):
    """One CMC-1 misassignment panel -- horizontal bars, names on the left."""
    names = [n for n in rows if n != "aligned"]
    vals = [rows[n]["cmc1_misassignment"] * 100 for n in names]
    # order by value, strongest cross-identity link on top
    order = sorted(range(len(vals)), key=lambda i: -vals[i])
    names = [names[i] for i in order]
    vals = [vals[i] for i in order]
    ys = np.arange(len(names))[::-1]          # first name on top
    ax.barh(ys, vals, color=[color_for(n) for n in names],
            edgecolor="white", height=0.72)
    for y, v in zip(ys, vals):
        ax.annotate(f"{v:.1f}", (v + 1.2, y), va="center", ha="left",
                    fontsize=4.4)
    ax.set_yticks(ys)
    ax.set_yticklabels(names, fontsize=5.2)
    for lb in ax.get_yticklabels():
        lb.set_ha("right")
    ax.set_ylim(-0.6, len(names) - 0.4)
    ax.set_xlim(0, max(vals) * 1.22 + 2)
    ax.set_xlabel("CMC-1 cross-identity match (%)", fontsize=7.5, fontweight="bold")
    ax.set_title(f"({letter}) {model}: CMC-1 cross-identity match rate",
                 fontsize=8, fontweight="bold", pad=4)
    ax.tick_params(axis="both", labelsize=5.5, length=3, width=0.7)
    ax.grid(True, axis="x", alpha=0.3)


def main():
    import matplotlib.pyplot as plt

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--idmap", required=True,
                   help="Dir with id_mapping_{swinface,transface}_results.json")
    p.add_argument("--output", required=True,
                   help="Output base name (without extension)")
    args = p.parse_args()

    data = {}
    for model in ("swinface", "transface"):
        jp = os.path.join(args.idmap, f"id_mapping_{model}_results.json")
        if not os.path.exists(jp):
            raise SystemExit(f"missing {jp}")
        data[model] = json.load(open(jp))

    # double-column width ~17.5 cm; 2x2 panels, tightly packed
    fig, axs = plt.subplots(2, 2, figsize=(6.9, 5.7), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.055, h_pad=0.045)
    plot_quadrant(axs[0, 0], data["swinface"]["rows"], "a", "SwinFace")
    plot_cmc1(axs[0, 1], data["swinface"]["rows"], "b", "SwinFace")
    plot_quadrant(axs[1, 0], data["transface"]["rows"], "c", "TransFace")
    plot_cmc1(axs[1, 1], data["transface"]["rows"], "d", "TransFace")

    out = export_figure(fig, args.output)
    plt.close(fig)
    print(f"  {out}")


if __name__ == "__main__":
    main()
