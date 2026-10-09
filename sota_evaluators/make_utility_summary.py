#!/usr/bin/env python3
"""
Efficiency-oriented summary of attribute-utility preservation — sorted dot matrix.

Alternative to the per-technique radar grid: ONE panel showing every technique
(20 rows) against every attribute-utility axis (7 for KDEF/RaFD/MUG, 9 for
celeba-test where the naturalness pillar is added), so the two questions a
reader actually asks are both answerable at a glance:

  1. RANKING  — rows are sorted by overall preservation (top = preserves
     utility = weaker privacy; bottom = erases attributes = stronger privacy)
  2. AGREEMENT — each row reads left-to-right across evaluators; tightly
     bunched dots = all models agree, a wide spread = model-dependent verdict

Layout: row = technique, column = (attribute, model) pair, cell = dot coloured
by preservation % (RdYlGn, 25-100) + numeric label.  Columns are grouped
Expression | Gender | (naturalness: LPIPS/IQA on celeba-test).  A final
"overall" column (mean) with tier-coloured row labels
(reddish < 50, amberish 50-70, greenish >= 70).

NO GPU inference — reads only root_dir/predictions/<dataset>/ CSVs
(same sources as make_utility_radar.py; data loader is reused from there).

Outputs: utility_summary_<dataset>.pdf/.png + *_data.csv

Usage:
    python make_utility_summary.py --dataset kdef
"""

import argparse
import csv
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
import fig_style  # noqa: E402
from make_utility_radar import load_tech_rates  # noqa: E402  (shared data layer)

DISPLAY = {
    "expr_swinface": "SwinFace", "expr_deepface": "DeepFace", "expr_uniface": "UniFace",
    "gender_swinface": "SwinFace", "gender_facexformer": "FaceXFormer",
    "gender_deepface": "DeepFace", "gender_uniface": "UniFace",
    "lpips_util": "LPIPS", "iqa_ret": "eDIF-IIQA",
}
ATTR = {
    "expr_swinface": "EX", "expr_deepface": "EX", "expr_uniface": "EX",
    "gender_swinface": "GD", "gender_facexformer": "GD", "gender_deepface": "GD",
    "gender_uniface": "GD", "lpips_util": "NAT", "iqa_ret": "NAT",
}


def tier_color(overall):
    if overall < 50:
        return "#c0392b"
    if overall < 70:
        return "#d68910"
    return "#1e8449"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="kdef")
    parser.add_argument("--output", default=SCRIPT_DIR)
    args = parser.parse_args()

    data, axes = load_tech_rates(args.dataset)
    if not data or not axes:
        raise SystemExit(f"No usable data for '{args.dataset}'.")

    # rows sorted ASC by overall so the strongest erasure sits at the BOTTOM
    def overall(t):
        return float(np.mean(list(data[t].values())))
    techs = sorted(data, key=overall)
    n = len(techs)
    m = len(axes)

    # data CSV
    os.makedirs(args.output, exist_ok=True)
    data_csv = Path(args.output) / f"utility_summary_{args.dataset}_data.csv"
    with open(data_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["technique"] + axes + ["overall"])
        for t in techs:
            w.writerow([t] + [f"{data[t][a]:.1f}" if a in data[t] else "" for a in axes]
                       + [f"{overall(t):.1f}"])
    print(f"Data: {data_csv}")

    cmap = matplotlib.colormaps["RdYlGn"]
    norm = Normalize(vmin=25, vmax=100)

    fig, ax = plt.subplots(figsize=(8.2, 0.42 * n + 2.6), dpi=150)
    xcols = np.arange(m + 1)  # last column = overall

    # column groups
    groups = {}
    for i, a in enumerate(axes):
        groups.setdefault(ATTR[a], []).append(i)
    group_x = {g: np.mean(x) + 0.0 for g, x in groups.items()}

    for tech, row in zip(techs, range(n)):
        # row base
        y = row
        for i, a in enumerate(axes):
            v = data[tech].get(a)
            if v is None:
                continue
            c = cmap(norm(v))
            ax.scatter(xcols[i], y, s=120, color=c, edgecolor="white",
                       linewidth=0.8, zorder=3)
            # label colour from actual fill luminance
            L = 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
            lbl_color = "#ffffff" if L < 0.52 else "#222222"
            ax.text(xcols[i], y, f"{v:.0f}", ha="center", va="center",
                    fontsize=7.6, fontweight="bold", color=lbl_color, zorder=4)
        ov = overall(tech)
        ax.scatter(xcols[m], y, s=200, color=tier_color(ov), marker="D",
                   edgecolor="white", linewidth=0.8, zorder=3)
        ax.text(xcols[m], y, f"{ov:.0f}", ha="center", va="center",
                fontsize=8, fontweight="bold", color="white", zorder=4)

    # row labels (tier colour = privacy tier)
    ax.set_yticks(range(n))
    ax.set_yticklabels(
        [f"{t}" for t in techs], fontsize=10, fontweight="bold")
    for tick, t in zip(ax.get_yticklabels(), techs):
        tick.set_color(tier_color(overall(t)))

    # column labels: model name (line 1) + attribute (line 2)
    col_labels = [f"{DISPLAY[a]}\n{ATTR[a]}" for a in axes] + ["Overall"]
    ax.set_xticks(xcols)
    ax.set_xticklabels(col_labels, fontsize=9.5)
    ax.get_xticklabels()[-1].set_fontweight("bold")

    # group header annotations (Expression | Gender | Naturalness)
    if groups:
        ymax = n - 1
        for gi, (g, xs) in enumerate(groups.items()):
            x0 = min(xs) - 0.42
            x1 = max(xs) + 0.42
            ax.plot([x0, x1], [ymax + 0.62, ymax + 0.62],
                    color="#666666", lw=1.2, solid_capstyle="round", clip_on=False)
            names = {"EX": "Expression preservation",
                     "GD": "Gender preservation",
                     "NAT": "Naturalness (celeba only)"}
            ax.text((x0 + x1) / 2, ymax + 0.95, names.get(g, g),
                    ha="center", va="bottom", fontsize=10.5,
                    color="#444444", fontweight="bold", clip_on=False)
        # separator between groups
        gkeys = list(groups)
        for a, b in zip(gkeys, gkeys[1:]):
            xs, xb = groups[a], groups[b]
            sep = (max(xcols[xs]) + min(xcols[xb])) / 2
            ax.axvline(sep, color="#bbbbbb", lw=1.4, zorder=1)
        ax.axvline(xcols[m] - 0.5, color="#bbbbbb", lw=1.4, zorder=1)

    # tier info is carried by row-label colours + suptitle; no in-plot annotations

    ax.set_xlim(-0.75, m + 0.55)
    ax.set_ylim(-0.62, n - 1 + 1.15)
    ax.set_frame_on(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax.grid(axis="y", color="#e8e8e8", lw=0.6, zorder=0)

    # colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=ax, fraction=0.028, pad=0.012)
    cbar.set_label("preservation rate (de-ID / original, %)", fontsize=9.5)
    cbar.ax.tick_params(labelsize=8.5)

    fig.suptitle(
        f"Attribute utility preservation across evaluators — {args.dataset}\n"
        "(rows sorted top = utility preserved / weaker privacy, bottom = strong "
        "erasure / stronger privacy; diamonds = overall mean; row colour = privacy tier)",
        fontsize=12.5)
    fig.tight_layout(rect=[0.02, 0.045, 1, 0.955])

    base = Path(args.output) / f"utility_summary_{args.dataset}"
    fig.savefig(f"{base}.png", dpi=300, bbox_inches="tight", pad_inches=0.12)
    fig.savefig(f"{base}.pdf", bbox_inches="tight", pad_inches=0.12)
    print(f"Figures: {base}.png / {base}.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
