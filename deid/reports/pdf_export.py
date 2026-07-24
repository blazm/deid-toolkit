"""PDF export module for pipeline results.

After the pipeline completes, this module generates PDF reports from
the CSV results:

- **ROC curves** for identity verification metrics
- **CMC curves** for identification metrics
- **Distribution plots** for image quality metrics
- **Confusion matrices** for classification metrics
- **Summary tables** of all evaluation results

Each report is saved as a PDF alongside the CSV in the results directory.
"""
from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


def export_results_to_pdf(results_dir: Path, output_dir: Optional[Path] = None) -> list[Path]:
    """Export all results in ``results_dir`` to PDF reports.

    Scans for CSV files and generates appropriate plots based on the
    evaluation type.

    Returns list of generated PDF paths.
    """
    if output_dir is None:
        output_dir = results_dir / "pdf_reports"
    output_dir.mkdir(parents=True, exist_ok=True)

    generated = []

    # Scan for CSV files
    for csv_file in results_dir.glob("**/*.csv"):
        if csv_file.parent == results_dir and csv_file.name == "summary.csv":
            continue  # skip summary

        try:
            df = pd.read_csv(csv_file)
        except Exception:
            continue

        # Determine evaluation type from content (columns) first, then filename
        name = csv_file.stem.lower()
        pdf_path = output_dir / f"{csv_file.stem}.pdf"

        if "ground_truth" in df.columns or "roc" in name:
            generated.append(_plot_roc(df, pdf_path))
        elif "rank1_correct" in df.columns or "cmc" in name or "identification" in name:
            generated.append(_plot_cmccurve(df, pdf_path))
        elif "fid" in name or "lpips" in name or "mse" in name or "ssim" in name:
            generated.append(_plot_distribution(df, pdf_path))
        elif "confusion" in name or "accuracy" in name:
            generated.append(_plot_confusion_matrix(df, pdf_path))
        else:
            # Generic table export
            generated.append(_export_table_to_pdf(df, pdf_path))

    return generated


def _plot_roc(df: pd.DataFrame, pdf_path: Path) -> Path:
    """Plot ROC curve from evaluation results."""
    fig, ax = plt.subplots(figsize=(8, 6))
    x, y = None, None

    if "FPR" in df.columns and "TPR" in df.columns:
        # Already has computed rates — plot directly
        x, y = df["FPR"].values, df["TPR"].values
    elif "ground_truth" in df.columns:
        # Verification CSV (image,cossim,img_b,ground_truth) — compute ROC from scores
        score_col = None
        for c in ["cossim", "score", "similarity"]:
            if c in df.columns:
                score_col = c
                break
        if score_col is not None:
            x, y = _compute_roc_points(df[score_col].values, df["ground_truth"].values)

    if x is None or y is None:
        # No suitable data found
        ax.text(0.5, 0.5, "No ROC-compatible data\n(need FPR/TPR columns or ground_truth + score)",
                transform=ax.transAxes, ha="center", va="center")
        fig.savefig(pdf_path, dpi=150, format="pdf")
        plt.close()
        return pdf_path

    ax.plot(x, y, "b-", linewidth=2)
    ax.set_title("ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.plot([0, 1], [0, 1], "r--", alpha=0.5)  # random baseline

    auc_val = float(-(x[1:] - x[:-1]) * ((y[1:] + y[:-1]) / 2).sum()) if len(x) > 1 else 0.0
    ax.set_title(f"ROC Curve\nAUC: {auc_val:.4f}")

    plt.tight_layout()
    fig.savefig(pdf_path, dpi=150, format="pdf")
    plt.close()
    return pdf_path


def _compute_roc_points(scores, labels):
    """Compute ROC curve points (FPR, TPR) from scores and binary labels.

    Higher score = more similar (genuine). Sorts by descending score,
    computes TPR/FPR at each threshold.
    """
    import numpy as np

    desc_indices = np.argsort(scores)[::-1]
    sorted_labels = labels[desc_indices].astype(int)
    total_pos = sorted_labels.sum()
    total_neg = len(sorted_labels) - total_pos

    if total_pos == 0 or total_neg == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])

    tpr_prev, fpr_prev = 0.0, 0.0
    tprs, fprs = [0.0], [0.0]

    for i in range(len(sorted_labels)):
        if sorted_labels[i] == 1:
            tpr_prev += 1.0 / total_pos
        else:
            fpr_prev += 1.0 / total_neg
        tprs.append(tpr_prev)
        fprs.append(fpr_prev)

    return np.array(fprs), np.array(tprs)


def _plot_cmccurve(df: pd.DataFrame, pdf_path: Path) -> Path:
    """Plot CMC curve from identification results.

    Expects columns: rank1_correct, rank2_correct, ..., rankK_correct
    (or raw rank column — will be converted to cumulative CMC).
    """
    import numpy as np

    fig, ax = plt.subplots(figsize=(8, 6))
    total = len(df)

    if "rank" not in df.columns or total == 0:
        ax.text(0.5, 0.5, "No CMC data\n(need 'rank' column or rank-k_correct columns)",
                transform=ax.transAxes, ha="center", va="center")
        fig.savefig(pdf_path, dpi=150, format="pdf")
        plt.close()
        return pdf_path

    # Find rank-k columns: rank1_correct, rank2_correct, ..., rankN_correct
    rank_cols = sorted(
        [int(c.split("_correct")[0].replace("rank", ""))
         for c in df.columns if c.endswith("_correct") and c.startswith("rank")],
    )

    if rank_cols:
        # Cumulative CMC: fraction of probes matched at each rank
        ranks = np.array(rank_cols)
        cumulative_rate = np.array([df[f"rank{k}_correct"].sum() / total for k in rank_cols])
    else:
        # Fallback: compute from raw 'rank' column
        max_k = min(df["rank"].max(), 20)
        ranks = np.arange(1, int(max_k) + 1)
        cumulative_rate = np.array([df["rank"].le(k).sum() / total for k in ranks])

    ax.plot(ranks, cumulative_rate, "b-", linewidth=2, marker="o", markersize=4)
    ax.set_xlabel("Rank", fontsize=12)
    ax.set_ylabel("Cumulative Match Rate", fontsize=12)
    r5_idx = min(4, len(cumulative_rate) - 1)
    ax.set_title(f"CMC Curve\nRank@1: {cumulative_rate[0]:.1%}, Rank@{ranks[r5_idx]}: {cumulative_rate[r5_idx]:.1%}")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0, top=1.05)
    ax.grid(True, alpha=0.3)

    # Annotate rank-1 and rank-5 points
    for k_idx in [0, min(4, len(ranks) - 1)]:
        if k_idx < len(ranks):
            ax.annotate(f"{cumulative_rate[k_idx]:.1%}",
                        xy=(ranks[k_idx], cumulative_rate[k_idx]),
                        textcoords="offset points", xytext=(5, 8),
                        fontsize=9, color="red")

    plt.tight_layout()
    fig.savefig(pdf_path, dpi=150, format="pdf")
    plt.close()
    return pdf_path


def _plot_distribution(df: pd.DataFrame, pdf_path: Path) -> Path:
    """Plot distribution of image quality metrics."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Get the metric column (usually the last one)
    numeric_cols = df.select_dtypes(include="number").columns
    if len(numeric_cols) == 0:
        plt.text(0.5, 0.5, "No numeric data", transform=ax.transAxes, ha="center")
    else:
        metric_col = numeric_cols[-1]
        values = df[metric_col].dropna()
        ax.hist(values, bins=50, alpha=0.7, edgecolor="black")
        ax.set_title(f"Distribution of {metric_col}")
        ax.set_xlabel(metric_col)
        ax.set_ylabel("Count")

        # Add summary statistics
        mean_val = values.mean()
        std_val = values.std()
        ax.axvline(mean_val, color="red", linestyle="--", label=f"Mean: {mean_val:.4f}")
        ax.legend()

    plt.tight_layout()
    fig.savefig(pdf_path, dpi=150, format="pdf")
    plt.close()
    return pdf_path


def _plot_confusion_matrix(df: pd.DataFrame, pdf_path: Path) -> Path:
    """Plot confusion matrix from classification results."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Try to extract confusion matrix from DataFrame
    if "confusion" in df.columns or "predicted" in df.columns:
        # Assuming columns: actual, predicted, count
        if "predicted" in df.columns and "actual" in df.columns:
            cm = df.pivot_table(index="actual", columns="predicted", values="count", aggfunc="sum")
            im = ax.imshow(cm.values, cmap="Blues")
            ax.set_xticks(range(len(cm.columns)))
            ax.set_yticks(range(len(cm.index)))
            ax.set_xticklabels(cm.columns)
            ax.set_yticklabels(cm.index)
            ax.set_title("Confusion Matrix")
            plt.colorbar(im)
        else:
            ax.text(0.5, 0.5, "Confusion matrix data format not recognized", transform=ax.transAxes, ha="center")
    else:
        ax.text(0.5, 0.5, "No confusion matrix data", transform=ax.transAxes, ha="center")

    plt.tight_layout()
    fig.savefig(pdf_path, dpi=150, format="pdf")
    plt.close()
    return pdf_path


def _export_table_to_pdf(df: pd.DataFrame, pdf_path: Path) -> Path:
    """Export a DataFrame as a PDF table."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis("tight")
    ax.axis("off")

    # Convert DataFrame to table
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # Style header
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor("#4472C4")
        table[(0, i)].set_text_props(color="white")

    ax.set_title("Evaluation Results Table")
    plt.tight_layout()
    fig.savefig(pdf_path, dpi=150, format="pdf")
    plt.close()
    return pdf_path


def generate_summary_report(results_dir: Path, output_path: Optional[Path] = None) -> Path:
    """Generate a comprehensive summary report for all results in ``results_dir``.

    Returns the path to the generated PDF.
    """
    if output_path is None:
        output_path = results_dir / "summary_report.pdf"

    with PdfPages(output_path) as pdf:
        # Title page
        fig, ax = plt.subplots(figsize=(8, 10))
        ax.text(0.5, 0.8, "DEID Toolkit — Results Summary", transform=ax.transAxes,
                ha="center", va="center", fontsize=24, fontweight="bold")
        ax.text(0.5, 0.6, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                transform=ax.transAxes, ha="center", va="center", fontsize=12)
        ax.text(0.5, 0.5, f"Results directory: {results_dir}",
                transform=ax.transAxes, ha="center", va="center", fontsize=10)

        csv_files = list(results_dir.glob("**/*.csv"))
        ax.text(0.5, 0.3, f"Total CSV files: {len(csv_files)}",
                transform=ax.transAxes, ha="center", va="center", fontsize=10)
        pdf.savefig(fig)
        plt.close()

        # Summary of each CSV
        for csv_file in sorted(csv_files):
            try:
                df = pd.read_csv(csv_file)
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.axis("tight")
                ax.axis("off")
                ax.set_title(f"{csv_file.name}", fontsize=14, fontweight="bold")

                # Display summary stats
                numeric_cols = df.select_dtypes(include="number").columns
                if len(numeric_cols) > 0:
                    summary = df[numeric_cols].describe()
                    table = ax.table(
                        cellText=summary.values,
                        colLabels=summary.columns,
                        cellLoc="center",
                        loc="center",
                        bbox=[0.05, 0.1, 0.9, 0.8],
                    )
                    table.auto_set_font_size(False)
                    table.set_fontsize(9)
                    table.scale(1, 1.2)
                else:
                    ax.text(0.5, 0.5, str(df.head()), transform=ax.transAxes, ha="center", va="center")

                pdf.savefig(fig)
                plt.close()
            except Exception:
                pass

    return output_path
