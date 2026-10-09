#!/usr/bin/env python3
"""
Utility-preservation radar charts — per-technique profiles across models.

One radar per DeID technique (5x4 grid for the 20 baselines). Axes are
PERSERVATION RATES (de-identified accuracy / aligned accuracy, 0-100%)
computed by the attribute-evaluation models:

  Expression preservation:  SwinFace, DeepFace, UniFace
  Gender preservation:      SwinFace, FaceXFormer, DeepFace, UniFace
  (celeba-test only)        LPIPS utility, IQA retention (tradeoff summary)

FaceXFormer has no expression task, hence 3 expression axes / 4 gender axes.
Axes whose source file is absent for the chosen dataset are auto-skipped, so
the script renders now (KDEF: 7 axes) and later (celeba-test: 9 axes) without
code changes.  NO GPU inference — reads only root_dir/predictions/<dataset>/
caches (the same per-image/aggregate CSVs every other figure uses).

Rendering follows deid/explore/radar_charts.py (spider layout with 25-degree
first-axis offset); colors/markers come from fig_style.py so encodings match
all other manuscript figures.

Outputs (in --output dir, default sota_evaluators/):
  utility_radar_<dataset>.pdf / .png   (grid)
  utility_radar_<dataset>_data.csv     (per-technique axis values, for re-plots/tables)

Usage:
    python make_utility_radar.py --dataset kdef
    python make_utility_radar.py --dataset celeba-test --sort alpha
"""

import argparse
import csv
import os
from math import pi
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import numpy as np


def fill_alpha_for(hex_color):
    """Luminance-compensated fill alpha: fig_style pairs dark+light tones per
    hue family, so a fixed alpha reads much paler on light tones. Making the
    fill opacity grow with luminance equalizes perceived panel density."""
    r, g, b = to_rgb(hex_color)
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return min(0.55, max(0.28, 0.18 + 0.5 * lum))

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
import fig_style  # noqa: E402  (paper color/marker encoding)

PRED = Path(SCRIPT_DIR).parent / "root_dir" / "predictions"
TRADEOFF = Path(SCRIPT_DIR).parent / "root_dir" / "predictions" / "tradeoff"


# ── Data loading ──────────────────────────────────────────────────────────

def _read_rate(csv_path, tech, column):
    """Read a rate/accuracy column for a technique row; None if missing/NA."""
    if not csv_path.is_file():
        return None
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("Technique", "").strip() == tech:
                val = (row.get(column, "") or "").strip()
                if val in ("", "—", "-", "N/A", "NA"):
                    return None
                try:
                    return float(val)
                except ValueError:
                    return None
    return None


def load_tech_rates(dataset):
    """Returns dict: {technique: {axis_key: value(0-100)}}.

    Axis order (top, clockwise): expression x3, gender x4, then naturalness
    (LPIPS util, IQA retention) if the tradeoff summary exists for the dataset.
    Each axis is (key, display_label, source_file, column).
    """
    ds_dir = PRED / dataset
    axes = [
        ("expr_swinface",  "Expr\nSwinFace",    ("expression_results.csv", "Expr_Preservation_Rate")),
        ("expr_deepface",  "Expr\nDeepFace",    ("expression_results_deepface.csv", "Expr_Preservation_Rate")),
        ("expr_uniface",   "Expr\nUniFace",     ("expression_results_uniface.csv", "Expr_Preservation_Rate")),
        ("gender_swinface",   "Gender\nSwinFace",    ("gender_results.csv", "Gender_Preservation_Rate")),
        ("gender_facexformer", "Gender\nFaceXFormer", ("gender_results_facexformer.csv", "Gender_Preservation_Rate")),
        ("gender_deepface",  "Gender\nDeepFace",    ("gender_results_deepface.csv", "Gender_Preservation_Rate")),
        ("gender_uniface",   "Gender\nUniFace",     ("gender_results_uniface.csv", "Gender_Preservation_Rate")),
    ]
    # naturalness pillar — celeba-test only
    tradeoff_csv = TRADEOFF / dataset / "tradeoff_summary.csv"
    if tradeoff_csv.is_file():
        axes.extend([
            ("lpips_util", "LPIPS\nutility", ("__tradeoff__", "lpips_util")),
            ("iqa_ret",    "IQA\nretention", ("__tradeoff__", "iqa_ret")),
        ])

    # technique list = union of all technique dirs for this dataset (minus meta)
    reserved = ("aligned", "labels", "original", "pairs", "deidentified", "_bak", "_bak_gendrelabel_20260908")
    techs = sorted(
        p.name for p in (Path(SCRIPT_DIR).parent / "root_dir" / "datasets").iterdir()
        if p.is_dir() and p.name not in reserved and (p / dataset).is_dir()
    )

    tradeoff_rows = {}
    if tradeoff_csv.is_file():
        with open(tradeoff_csv, newline="") as f:
            for row in csv.DictReader(f):
                tradeoff_rows[row["technique"]] = row

    data = {}
    for tech in techs:
        vals = {}
        for key, _disp, (src, col) in axes:
            if src == "__tradeoff__":
                row = tradeoff_rows.get(tech)
                if row and row.get(col, "").strip() not in ("", "nan"):
                    try:
                        vals[key] = float(row[col]) * 100.0
                    except ValueError:
                        pass
            else:
                v = _read_rate(ds_dir / src, tech, col)
                if v is not None:
                    vals[key] = v
        if vals:
            data[tech] = vals
    present = [k for k, _d, _s in axes if any(k in d for d in data.values())]
    return data, [k for k in present]


# ── Rendering (spider layout from deid/explore/radar_charts.py) ───────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="kdef")
    parser.add_argument("--output", default=SCRIPT_DIR)
    parser.add_argument("--sort", choices=["overall", "alpha"], default="overall")
    parser.add_argument("--cols", type=int, default=5)
    args_g = parser.parse_args()
    args = args_g

    data, axes = load_tech_rates(args.dataset)
    display = {
        "expr_swinface": "SwinFace\nEX", "expr_deepface": "DeepFace\nEX",
        "expr_uniface": "UniFace\nEX",
        "gender_swinface": "SwinFace\nGD", "gender_facexformer": "FaceXFormer\nGD",
        "gender_deepface": "DeepFace\nGD", "gender_uniface": "UniFace\nGD",
        "lpips_util": "LPIPS\nutil", "iqa_ret": "IQA\nret",
    }
    if not data or not axes:
        raise SystemExit(f"No usable data for dataset '{args.dataset}' — "
                         "check root_dir/predictions/<dataset> CSVs.")
    print(f"Dataset: {args.dataset} | techniques: {len(data)} | axes: {len(axes)}")
    print("Axes (top, clockwise):", " | ".join(axes))

    # sort techniques: overall = mean of available axes (desc), or alphabetical
    def overall(t):
        return np.mean(list(data[t].values()))
    techs = sorted(data, key=overall, reverse=True) if args.sort == "overall" else sorted(data)

    # data CSV (for offline re-plots / tables)
    os.makedirs(args.output, exist_ok=True)
    data_csv = Path(args.output) / f"utility_radar_{args.dataset}_data.csv"
    with open(data_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["technique"] + axes)
        for t in techs:
            w.writerow([t] + [f"{data[t][a]:.1f}" if a in data[t] else "" for a in axes])
    print(f"Data: {data_csv}")

    N = len(axes)
    nrows = int(np.ceil(len(techs) / args.cols))
    # bigger cells + constrained layout: minimal wasted padding around every polar panel
    fig, axes_arr = plt.subplots(nrows, args.cols,
                                 figsize=(4.4 * args.cols, 4.15 * nrows),
                                 subplot_kw={"polar": True}, dpi=150,
                                 constrained_layout=True)
    axes_arr = np.atleast_2d(axes_arr)

    display_labels = [display.get(a, a) for a in axes]
    # start first axis 25deg right of top (same trick as the reference renderer)
    angles = [25.0 * pi / 180.0 + n / N * 2 * pi for n in range(N)]
    angles += angles[:1]

    for idx, tech in enumerate(techs):
        row, col = divmod(idx, args.cols)
        ax = axes_arr[row, col]
        ax.set_theta_offset(pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(display_labels, color="#444444", size=12.5)
        ax.set_rlabel_position(0)
        ax.set_yticks([25, 50, 75, 100])
        ax.set_yticklabels(["25", "50", "75", "100"], color="#999999", size=8.5)
        ax.set_ylim(0, 100)
        ax.tick_params(axis="x", pad=5)

        color, marker = fig_style.style_for(tech)
        values = [data[tech][a] if a in data[tech] else 0.0 for a in axes]
        values += values[:1]
        ax.plot(angles, values, color=color, linewidth=1.4)
        ax.fill(angles, values, color=color, alpha=fill_alpha_for(color))
        # missing axes: open vertex marks absence
        if any(a not in data[tech] for a in axes):
            ax.plot(angles[:-1], values[:-1].tolist(),
                    marker=marker, markersize=3.2, markerfacecolor="none",
                    linestyle="none", color=color)
        ax.set_title(f"{tech}  ({overall(tech):.0f}%)", size=16, fontweight="bold",
                     color=color, loc="center", y=1.085)
        ax.grid(color="#d5d5d5", linewidth=0.5)

    for i in range(len(techs), nrows * args.cols):
        axes_arr[i // args.cols, i % args.cols].set_visible(False)

    fig.suptitle(
        f"Utility preservation after de-identification — {args.dataset} "
        f"(axis = de-ID accuracy / original accuracy, %)",
        fontsize=18)

    base = Path(args.output) / f"utility_radar_{args.dataset}"
    fig.savefig(f"{base}.png", dpi=300, bbox_inches="tight", pad_inches=0.12)
    fig.savefig(f"{base}.pdf", bbox_inches="tight", pad_inches=0.12)
    print(f"Figures: {base}.png / {base}.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
