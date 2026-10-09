#!/usr/bin/env python3
"""
Privacy <-> attribute-utility trade-off scatter — 4 points per technique.

Each technique is drawn ONCE PER DATASET:

  x  = 1:1 cross-condition identity EER for that dataset, mean of the
       SwinFace and TransFace backbones (from per-dataset
       linkability_<Model>_<dataset>_results.json; celeba-test reuses the
       original linkability_{swinface,transface}_results.json)
  y  = attribute-utility preservation for the SAME dataset (mean of the
       expr_*/gender_* evaluator axes for that dataset; celeba-test =
       gender only -- no expression labels in CelebA)

so both axes of every point come from the same dataset (balanced
comparison; no cross-dataset mixing). Faint lines connect a technique's
four points so per-baseline clusters are traceable. A tiny dataset tag
sits next to each point.

House style (make_linkability_4panel.py): fig_style (color, marker)
identity, s=42 black-edged points, in-panel legend, black dashed / gray
dotted reference lines, no per-figure suptitle.

EER note: values above 0.5 (chance) mean the verifier scores the
de-id-vs-aligned pair BELOW impostor pairs (anti-verification; seen for
RP / NullFace on the non-celeba sets).

NO GPU inference — reads only existing JSON/CSV results.
Outputs: privacy_utility_scatter.png/.pdf + _data.csv

Usage:
    python make_privacy_utility_scatter.py
"""

import csv
import json
import os
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
import fig_style  # noqa: E402
from make_utility_radar import load_tech_rates  # noqa: E402

PRED = Path(SCRIPT_DIR).parent / "root_dir" / "predictions"
DATASETS = ["kdef", "rafd", "mug-still", "celeba-test"]
DATASET_TAGS = {"kdef": "kdef", "rafd": "RAFD", "mug-still": "MUG",
                "celeba-test": "CELEBA"}
# dataset -> categorical marker (shape channel = dataset in this figure;
# technique identity stays with fig_style COLOUR)
DATASET_MARK = {"kdef": (".", 90), "rafd": ("^", 52),
                "mug-still": ("*", 46), "celeba-test": ("s", 34)}
REF_X, REF_Y, CHANCE = 0.20, 90.0, 0.50

# per-dataset tag offsets (points) to spread the 4 labels around a point
TAG_OFF = {
    "kdef": (5, 6, "left"),
    "rafd": (5, -9, "left"),
    "mug-still": (-5, 6, "right"),
    "celeba-test": (-5, -9, "right"),
}


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


def eer_per_dataset():
    """tech -> {dataset -> mean(EER_swin, EER_trans)}"""
    files = {
        "celeba-test": (PRED / "linkability_swinface_results.json",
                        PRED / "linkability_transface_results.json"),
        "kdef": (PRED / "linkability_SwinFace_kdef_results.json",
                 PRED / "linkability_TransFace_kdef_results.json"),
        "rafd": (PRED / "linkability_SwinFace_rafd_results.json",
                 PRED / "linkability_TransFace_rafd_results.json"),
        "mug-still": (PRED / "linkability_SwinFace_mug-still_results.json",
                      PRED / "linkability_TransFace_mug-still_results.json"),
    }
    out = {}
    for ds, (fa, fb) in files.items():
        A, B = json.load(open(fa)), json.load(open(fb))
        techs = set(A["techniques"]) & set(B["techniques"])
        for t in techs:
            out.setdefault(t, {})[ds] = (A["techniques"][t]["eer"]
                                         + B["techniques"][t]["eer"]) / 2
    return out


def attr_y_per_dataset(dataset):
    """tech -> mean preservation rate of expr_*/gender_* axes (or None)."""
    data, axes = load_tech_rates(dataset)
    out = {}
    for tech, vals in (data or {}).items():
        sel = [v for a, v in vals.items()
               if a.startswith(("expr_", "gender_"))]
        if sel:
            out[tech] = float(np.mean(sel))
    return out


def main():
    out_dir = SCRIPT_DIR
    os.makedirs(out_dir, exist_ok=True)

    eer = eer_per_dataset()
    per_ds = {ds: attr_y_per_dataset(ds) for ds in DATASETS}
    techs = sorted(set(eer) & set.intersection(*[set(v) for v in per_ds.values()]))

    # data CSV
    data_csv = Path(out_dir) / "privacy_utility_scatter_data.csv"
    with open(data_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["technique", "dataset", "eer", "attr_utility"])
        for t in techs:
            for ds in DATASETS:
                w.writerow([t, ds, f"{eer[t][ds]:.4f}",
                            f"{per_ds[ds][t]:.1f}"])
    print(f"Data: {data_csv}")

    fig, ax = plt.subplots(figsize=(6.9, 4.9), constrained_layout=True)

    # per-technique trace + points (house style)
    for t in techs:
        pts = np.array([[eer[t][ds], per_ds[ds][t]] for ds in DATASETS])
        ax.plot(pts[:, 0], pts[:, 1], color="dimgray", linewidth=0.35,
                alpha=0.28, zorder=2)
        for i, ds in enumerate(DATASETS):
            x, y = eer[t][ds], per_ds[ds][t]
            mk, ms = DATASET_MARK[ds]
            ax.scatter(x, y, s=ms, color=fig_style.color_for(t),
                       marker=mk, edgecolor="k", linewidth=0.35, zorder=3)
            dx, dy, ha = TAG_OFF[ds]
            ax.annotate(DATASET_TAGS[ds], (x, y), textcoords="offset points",
                        xytext=(dx, dy), ha=ha, va="center", fontsize=3.4,
                        color="#777777", zorder=2)

    xs_all = np.array([eer[t][ds] for t in techs for ds in DATASETS])
    ys_all = np.array([per_ds[ds][t] for t in techs for ds in DATASETS])
    xlim = (float(xs_all.min()) - 0.03, max(float(xs_all.max()), CHANCE) + 0.05)
    ylim = (float(ys_all.min()) - 3.0, float(ys_all.max()) + 3.0)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    # reference lines: chance gray dotted, metric black dashed (house style)
    ax.axvline(CHANCE, color="gray", linestyle=":", linewidth=1.1)
    ax.axvline(REF_X, color="black", linestyle="--", linewidth=1.0)
    ax.axhline(REF_Y, color="black", linestyle="--", linewidth=1.0)
    ax.text(CHANCE + 0.004, ylim[0] + 0.9, "chance 0.5", fontsize=4.4,
            va="bottom", ha="left", color="dimgray")
    ax.text(REF_X + 0.004, REF_Y + 0.6, "EER 0.20", fontsize=4.4,
            va="bottom", ha="left", color="black")
    ax.text(REF_X - 0.004, REF_Y + 0.6, "90%", fontsize=4.4,
            va="bottom", ha="right", color="black")

    ax.text(xlim[1] - 0.004, ylim[1] - 0.9, "unlinkable + preserved (ideal)",
            ha="right", va="top", fontsize=4.5, color="dimgray", style="italic")
    ax.text(xlim[0] + 0.004, ylim[0] + 10.0, "linkable + distorted",
            ha="left", va="bottom", fontsize=4.5, color="dimgray", style="italic")
    ax.text(xlim[1] - 0.004, ylim[0] + 1.0, "EER > 0.5: de-id scores below impostors "
            "(anti-verification)", ha="right", va="bottom",
            fontsize=4.3, color="#999999", style="italic")

    ax.set_xlabel("identity EER (1:1 cross-condition, de-id vs aligned; "
                  "SwinFace + TransFace mean)", fontsize=7.0, fontweight="bold")
    ax.set_ylabel("attribute utility preservation, per-dataset mean (%)",
                  fontsize=7.0, fontweight="bold")
    ax.tick_params(axis="both", labelsize=6, length=3, width=0.7)
    ax.grid(True, alpha=0.3)

    # in-panel technique legend, linkability-plot handle style
    # technique = COLOUR in this figure (shape channel belongs to dataset);
    # legend swatches are therefore neutral uniform dots
    handles = [Line2D([], [], marker="o", linestyle="",
                      markerfacecolor=fig_style.color_for(n),
                      markeredgecolor="k", markeredgewidth=0.5,
                      markersize=5.0, label=n)
               for n in sorted(techs,
                               key=lambda x: -np.mean([eer[x][ds]
                                                       for ds in DATASETS]))]
    leg = ax.legend(handles=handles, loc="center left", bbox_to_anchor=(0.015, 0.55),
                    ncol=4, fontsize=3.5, framealpha=0.95, edgecolor="#cccccc",
                    handlelength=0.9, handletextpad=0.4, columnspacing=0.7,
                    labelspacing=0.35, borderpad=0.5)
    # dataset-shape legend (small, top-left empty zone)
    ds_handles = [Line2D([], [], marker=DATASET_MARK[ds][0], color="none",
                         markerfacecolor="#666666", markeredgecolor="k",
                         markersize={("."): 9, ("^"): 5.5, ("*"): 5.5,
                                      ("s"): 5}[DATASET_MARK[ds][0]],
                         label=DATASET_TAGS[ds])
                 for ds in DATASETS]
    ds_leg = ax.legend(handles=ds_handles, loc="upper left",
                       bbox_to_anchor=(0.015, 0.99), ncol=4, fontsize=4.0,
                       framealpha=0.95, edgecolor="#cccccc", title="dataset",
                       title_fontsize=4.6, columnspacing=0.9,
                       handletextpad=0.3, borderpad=0.4)
    ds_leg.get_title().set_fontweight("bold")
    # a second ax.legend() call removes the first one — restore it
    ax.add_artist(leg)

    base = Path(out_dir) / "privacy_utility_scatter"
    _savefig_safe(fig, f"{base}.png", dpi=300)
    _savefig_safe(fig, f"{base}.pdf", format="pdf")
    print(f"Figures: {base}.png / {base}.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
