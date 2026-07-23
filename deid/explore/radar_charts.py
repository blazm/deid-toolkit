"""Radar charts — compare techniques across multiple evaluation axes.

Ported from raw_visualisations_to_port/radar_charts_comparison.py.
Supports both static benchmark data and per-dataset/technique scores loaded
from saved evaluation CSVs.
"""
from __future__ import annotations

import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from math import pi


# ── Static benchmark data (from published papers) ───────────────────────
STATIC_BENCHMARKS: dict[str, dict[str, float]] = {
    "Ours (pp=0.0)": {
        "privacy\nprotection": 1.0,
        "image\nquality": 1.0,
        "data\nutility": 0.67,
        "context\npreservation": 1.0,
    },
    "Ours (pp=0.5)": {
        "privacy\nprotection": 0.67,
        "image\nquality": 1.0,
        "data\nutility": 1.0,
        "context\npreservation": 1.0,
    },
    "Ours (pp=0.9)": {
        "privacy\nprotection": 0.33,
        "image\nquality": 1.0,
        "data\nutility": 1.0,
        "context\npreservation": 1.0,
    },
    "Ren et al.": {
        "privacy\nprotection": 0.33,
        "image\nquality": 0.67,
        "data\nutility": 0.33,
        "context\npreservation": 0.67,
    },
    "DeepPrivacy": {
        "privacy\nprotection": 0.67,
        "image\nquality": 0.67,
        "data\nutility": 0.67,
        "context\npreservation": 0.67,
    },
    "CIAGAN": {
        "privacy\nprotection": 0.33,
        "image\nquality": 0.67,
        "data\nutility": 0.67,
        "context\npreservation": 0.67,
    },
    "CLEANIR": {
        "privacy\nprotection": 0.67,
        "image\nquality": 0.67,
        "data\nutility": 0.67,
        "context\npreservation": 0.67,
    },
    "AMT-GAN": {
        "privacy\nprotection": 0.33,
        "image\nquality": 0.67,
        "data\nutility": 0.67,
        "context\npreservation": 0.67,
    },
    "k-Same-Net": {
        "privacy\nprotection": 1.0,
        "image\nquality": 0.33,
        "data\nutility": 0.67,
        "context\npreservation": 0.33,
    },
    "Croft et al.": {
        "privacy\nprotection": 1.0,
        "image\nquality": 0.33,
        "data\nutility": 0.67,
        "context\npreservation": 0.33,
    },
}


# ── Axes categories that can be computed from evaluation results ─────────
# Each key: display label (with newlines for multi-line rendering)
# Each value: list of (column_substring, dataset_suffix, invert) tuples
#   invert=True means lower raw score is better (e.g. 1-MSE)
AXIS_SPECS: dict[str, list[tuple[str, str, bool]]] = {
    "DEID": [
        ("deid", "CelebA", False),
        ("deid", "XM2VTS", False),
        ("deid", "RaFD", False),
    ],
    "DIV": [
        ("div", "CelebA", False),
        ("div", "XM2VTS", False),
        ("div", "RaFD", False),
    ],
    "1-MSE": [
        ("1-mse", "CelebA", False),
        ("1-mse", "XM2VTS", False),
        ("1-mse", "RaFD", False),
    ],
    "EX": [
        ("ex", "RaFD", False),
        ("ex", "AffectNet", False),
    ],
    "GD": [
        ("gd", "RaFD", False),
        ("gd", "CelebA", False),
    ],
}


def render_static_benchmark() -> tuple[plt.Figure, io.BytesIO, io.BytesIO]:
    """Radar chart comparing techniques against static benchmark values.

    Returns (figure, pdf_buffer, svg_buffer).
    """
    methods = list(STATIC_BENCHMARKS.keys())
    categories = list(next(iter(STATIC_BENCHMARKS.values())).keys())
    N = len(categories)

    fig, axes = plt.subplots(2, 5, figsize=(20, 8), subplot_kw={"polar": True}, dpi=96)
    axes = axes.flatten()

    my_palette = plt.cm.tab10(len(methods))

    for idx, method in enumerate(methods):
        data = STATIC_BENCHMARKS[method]
        angles = [25.0 * pi / 180.0 + n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]

        ax = axes[idx]
        ax.set_theta_offset(pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, color="grey", size=6)

        ax.set_rlabel_position(0)
        plt.yticks([], [], color="grey", size=6)
        plt.ylim(0, 1.0)

        values = [data[c] for c in categories] + [data[categories[0]]]
        ax.plot(angles, values, color=my_palette[idx], linewidth=1.2, linestyle="solid")
        ax.fill(angles, values, color=my_palette[idx], alpha=0.25)

        ax.set_title(method, size=9, color=my_palette[idx], loc="center", y=1.15)

    for i in range(len(methods), 10):
        axes[i].set_visible(False)

    fig.suptitle("Technique Comparison — Static Benchmark", fontsize=13, y=1.02)
    fig.tight_layout()

    # Save to both buffers before closing
    pdf_buf = io.BytesIO()
    fig.savefig(pdf_buf, format="pdf", bbox_inches="tight")
    pdf_buf.seek(0)

    svg_buf = io.BytesIO()
    fig.savefig(svg_buf, format="svg", bbox_inches="tight", dpi=150)
    svg_buf.seek(0)

    return fig, pdf_buf, svg_buf


def _compute_axis_value(axis_name: str, results: dict) -> float:
    """Average the scores for an axis category across its datasets."""
    specs = AXIS_SPECS.get(axis_name, [])
    if not specs:
        return 0.5

    vals = []
    for col_substr, ds_suffix, invert in specs:
        for tech_name, tech_results in results.items():
            for ev_name, csv_path in tech_results.items():
                if col_substr in ev_name.lower() and ds_suffix in csv_path:
                    try:
                        df = pd.read_csv(csv_path)
                        scores = df.iloc[:, 1] if len(df.columns) > 1 else df.iloc[:, 0]
                        mean_score = float(scores.mean())
                        if invert:
                            mean_score = 1.0 - mean_score
                        vals.append(np.clip(mean_score, 0.0, 1.0))
                    except Exception:
                        pass

    return float(np.mean(vals)) if vals else 0.5


def render_loaded_charts(
    results: dict[str, dict[str, dict[str, Path]]],
    selected_ds: str,
    selected_techniques: list[str] | None = None,
) -> tuple[plt.Figure, io.BytesIO, io.BytesIO]:
    """Radar chart where each axis is an aggregate metric from evaluation CSVs.

    results: {dataset: {technique: {eval_name: csv_path}}}
    selected_ds: which dataset group to pull from
    Returns (figure, pdf_buffer, svg_buffer).
    """
    if selected_techniques is None:
        selected_techniques = list(results.get(selected_ds, {}).keys())

    if not selected_techniques:
        empty_fig, _ = plt.subplots(figsize=(8, 6)), io.BytesIO()
        plt.close(empty_fig)
        return empty_fig, io.BytesIO(), io.BytesIO()

    axes_list = list(AXIS_SPECS.keys())
    N = len(axes_list)

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True}, dpi=96)
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks([25.0 * pi / 180.0 + n / float(N) * 2 * pi for n in range(N)])
    ax.set_xticklabels(axes_list, color="grey", size=9)
    ax.set_rlabel_position(0)
    ax.set_yticks([0.33, 0.67, 1.0])
    ax.set_yticklabels(["low", "med", "high"], color="grey", size=7)
    ax.set_ylim(0, 1.0)

    my_palette = plt.cm.tab10(len(selected_techniques))

    for idx, technique in enumerate(selected_techniques):
        tech_results = results.get(selected_ds, {}).get(technique, {})
        values = [
            _compute_axis_value(axis, results)
            for axis in axes_list
        ]

        angles = [25.0 * pi / 180.0 + n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]
        vals = values + values[:1]

        ax.plot(angles, vals, color=my_palette[idx], linewidth=1.5, label=technique)
        ax.fill(angles, vals, color=my_palette[idx], alpha=0.2)

    ax.legend(loc="center right", bbox_to_anchor=(1.3, 0.5), fontsize=8)
    fig.suptitle(
        f"Radar Chart — Aggregate Metrics  |  Dataset: {selected_ds}",
        fontsize=12, y=0.98,
    )
    fig.tight_layout()

    pdf_buf = io.BytesIO()
    fig.savefig(pdf_buf, format="pdf", bbox_inches="tight")
    pdf_buf.seek(0)

    svg_buf = io.BytesIO()
    fig.savefig(svg_buf, format="svg", bbox_inches="tight", dpi=150)
    svg_buf.seek(0)

    return fig, pdf_buf, svg_buf


def to_svg_buffer(fig: plt.Figure) -> io.BytesIO:
    """Render a matplotlib figure to an SVG string buffer."""
    buf = io.BytesIO()
    fig.savefig(buf, format="svg", bbox_inches="tight", dpi=150)
    buf.seek(0)
    return buf
