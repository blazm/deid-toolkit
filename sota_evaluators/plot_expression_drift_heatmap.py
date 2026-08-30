#!/usr/bin/env python3
"""
Expression drift heatmap — dense matrix form.

Single compact heat grid replacing stacked confusion matrices.

Rows = anonymization techniques (sorted by overall accuracy, descending).
Columns = true expression classes (Angry … Neutral).
Each cell = % of that class predicted correctly by that technique (0–100%).
Green = high accuracy, Red = low accuracy.

Two-panel figure:
  Top  (70%): Accuracy matrix heatmap — per-class correct-prediction rate
  Bot  (30%): Drift-out matrix   — % of images misclassified into *other* classes

Usage (KDEF):
    conda run -n swinface python plot_expression_drift_heatmap.py \\
        --output-dir D:/dev/deid-toolkit/root_dir/predictions/kdef \\
        --dataset kdef \\
        --output expression_drift_heatmap

Usage (RaFD, with custom techniques):
    conda run -n swinface python plot_expression_drift_heatmap.py \\
        --output-dir D:/dev/deid-toolkit/root_dir/predictions/rafd \\
        --dataset rafd \\
        --techniques AIDPro AMT-GAN DeepPrivacy DeepPrivacy2 G2Face GANonymization IPFA RiDDLE CLEANIR NullFace FADM FAMS \\
        --output expression_drift_heatmap

Outputs:
  - PNG + PDF at output_dir/output.png / output.pdf
"""

import argparse
import csv
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Reuse model ────────────────────────────────────────────────────────────

EXTRACT_ATTRS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EXTRACT_ATTRS_DIR)

from extract_attributes_swinface import (
    preprocess_image,
    build_swinface_model,
    EXPRESSION_LABELS,  # ["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"]
)

EXPRESSION_LABELS = list(EXPRESSION_LABELS)


# ── Helpers ────────────────────────────────────────────────────────────────

def _savefig_safe(fig, base_path, dpi=300):
    """Save figure to PNG and PDF with retry on locked files."""
    import time as _time
    for fmt in ("png", "pdf"):
        fpath = os.path.splitext(base_path)[0] + "." + fmt
        retries = 0
        while True:
            try:
                fig.savefig(fpath, dpi=dpi, bbox_inches="tight")
                break
            except PermissionError:
                retries += 1
                if retries > 5:
                    print(f"  WARN: could not save {fpath}")
                    break
                _time.sleep(0.3)


def find_image_path(directory, stem):
    d = Path(directory)
    for ext in (".png", ".PNG", ".jpg", ".jpeg", ".bmp"):
        c = d / (stem + ext)
        if c.exists():
            return c
    return None


def predict_images(image_dir, stems, model, device, batch_size=32):
    """Run inference and return {stem: pred_expr_idx}."""
    paths = []
    for stem in stems:
        img_path = find_image_path(image_dir, stem)
        if img_path is not None:
            try:
                t = preprocess_image(img_path)
                paths.append((stem, t))
            except (ValueError, OSError):
                pass

    preds = {}
    for start in range(0, len(paths), batch_size):
        end = min(start + batch_size, len(paths))
        batch = torch.stack([p[1] for p in paths[start:end]]).to(device)
        with torch.no_grad():
            outputs = model(batch)
        expr_preds = outputs["Expression"].float().argmax(dim=1).cpu().numpy()
        for i, (stem, _) in enumerate(paths[start:end]):
            preds[stem] = int(expr_preds[i])

    return preds


# ── GT loaders ─────────────────────────────────────────────────────────────

def load_kdef_gt(labels_dir):
    gt = {}
    path = os.path.join(labels_dir, "kdef_labels.csv")
    if not os.path.exists(path):
        print(f"  WARNING: Labels not found: {path}")
        return gt
    with open(path) as f:
        reader = csv.DictReader(f)
        code_to_expr = {0: 6, 1: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5}
        for row in reader:
            name = row.get("Name", "").strip()
            if not name:
                continue
            stem = os.path.splitext(name)[0]
            if row.get("Scream") == "1" or row.get("Contempt") == "1":
                continue
            emo = row.get("Emotion_code", "").strip()
            expr = None
            if emo:
                try:
                    expr = code_to_expr.get(int(emo))
                except ValueError:
                    pass
            gt[stem] = {"expression": expr}
    return gt


def load_rafd_gt(labels_dir):
    gt = {}
    path = os.path.join(labels_dir, "rafd-frontal_aligned_labels.csv")
    if not os.path.exists(path):
        print(f"  WARNING: Labels not found: {path}")
        return gt
    expr_map = {l: i for i, l in enumerate(EXPRESSION_LABELS)}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("Name", "").strip()
            if not name:
                continue
            stem = os.path.splitext(name)[0]
            gc = row.get("Gender_code", "").strip()
            if gc not in ("1", "-1"):
                continue
            expr = None
            for label in EXPRESSION_LABELS:
                col = label if label != "Sad" else "Sadness"
                if row.get(col, "").strip() == "1":
                    expr = expr_map[label]
                    break
            gt[stem] = {"expression": expr}
    return gt


# ── Drift heatmap plotter — matrix form ───────────────────────────────────

def plot_drift_heatmap(technique_cm_data, technique_summary, baseline_acc, output_path, dataset_name):
    """Generate a compact matrix-style drift heatmap.

    Design: two panels (both matrix heat grids)
      Top  (70%): Accuracy — % correct per class × technique  (RdYlGn reversed)
      Bottom (30%): Drift-out — % misclassified into *any other* class  (YlOrRd)

    Parameters
    ----------
    technique_cm_data : dict {tech_name: [(gt_idx, pred_idx), ...]}
    technique_summary : dict {tech_name: {"deid_acc": float, "preservation_rate": float}}
    baseline_acc : float
    output_path : str (PNG; PDF auto-generated)
    dataset_name : str
    """
    n_classes = len(EXPRESSION_LABELS)

    # Compute per-technique per-class accuracy (row-normalized from CM data)
    sorted_techs = sorted(
        technique_summary.keys(),
        key=lambda t: technique_summary[t]["deid_acc"],
        reverse=True,
    )

    tech_names = []
    acc_matrix = np.zeros((len(sorted_techs), n_classes), dtype=np.float64)

    for ti, tech in enumerate(sorted_techs):
        cm_data = technique_cm_data.get(tech, [])
        if not cm_data:
            continue
        row_counts = np.zeros(n_classes, dtype=np.float64)
        correct_per_class = np.zeros(n_classes, dtype=np.float64)
        for gt_idx, pred_idx in cm_data:
            row_counts[gt_idx] += 1.0
            if gt_idx == pred_idx:
                correct_per_class[gt_idx] += 1.0
        with np.errstate(divide="ignore", invalid="ignore"):
            class_acc = np.divide(
                correct_per_class, row_counts,
                out=np.zeros(n_classes), where=(row_counts > 0),
            ) * 100.0
        tech_names.append(tech)
        acc_matrix[ti] = class_acc

    if not tech_names:
        print("ERROR: No confusion data available.")
        return

    # Shrink to actually-loaded techniques
    n_techs = len(tech_names)
    acc_matrix = acc_matrix[:n_techs]

    # Drift-out: 100 - accuracy → % of that class misclassified into *any* other class
    drift_out = 100.0 - acc_matrix  # shape (n_techs, n_classes)

    # ── Layout ───────────────────────────────────────────────────────────
    fig_width = 4.5          # single-column journal width
    fig_height = 7.0
    fig_height_bot = 2.8     # height for bottom panel (inches)

    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = plt.GridSpec(
        2, 1,
        figure=fig,
        height_ratios=[n_techs * 0.75 + 1.5, fig_height_bot],
        hspace=0.35,
    )

    # ── Top panel: accuracy heatmap matrix ───────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    im1 = ax1.imshow(
        acc_matrix, aspect="auto",
        cmap="RdYlGn_r", vmin=0, vmax=100,
    )

    # Tick marks and labels
    x_ticks = list(range(n_classes))
    y_ticks = list(range(n_techs))
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels(EXPRESSION_LABELS, rotation=35, ha="right", fontsize=8)
    ax1.set_yticks(y_ticks)

    # Shorten technique names for readability
    short = [t[:16].ljust(16) if len(t) > 18 else f"{t:<18}" for t in tech_names]
    ax1.set_yticklabels(short, fontsize=7)

    # Cell annotations — numeric accuracy %
    for ti in range(n_techs):
        for ci in range(n_classes):
            val = acc_matrix[ti, ci]
            txt_color = "white" if val < 50 else "black"
            ax1.text(ci, ti, f"{val:.0f}", ha="center", va="center",
                     fontsize=7, color=txt_color)

    # Subtle cell grid lines
    ax1.set_xticks([x - 0.5 for x in x_ticks], minor=True)
    ax1.set_yticks([y - 0.5 for y in y_ticks], minor=True)
    ax1.grid(which="minor", color="#666", linewidth=0.3, alpha=0.6)

    # Colorbar
    cbar = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.02)
    cbar.set_label("Correct Pred. (%)", fontsize=9)

    ax1.set_xlabel("True Expression Class", fontsize=10, fontweight="bold")
    ax1.set_ylabel("", fontsize=10)  # technique implied by row order
    ds_display = {"rafd": "RaFD", "mug-still": "MUG-Still", "kdef": "KDEF"}.get(
        dataset_name, dataset_name)
    ax1.set_title(f"Per-Class Accuracy — {ds_display}", fontsize=12, fontweight="bold")

    # ── Bottom panel: drift-out heatmap matrix ───────────────────────────
    ax2 = fig.add_subplot(gs[1])
    im2 = ax2.imshow(
        drift_out[:n_techs], aspect="auto",
        cmap="YlOrRd", vmin=0, vmax=60,  # cap at 60% for readability
    )

    x_ticks2 = list(range(n_classes))
    ax2.set_xticks(x_ticks2)
    ax2.set_xticklabels(EXPRESSION_LABELS, rotation=35, ha="right", fontsize=8)

    # Only show technique labels for first 2 and last row to save space
    shown_names = []
    for i in range(n_techs):
        if i < 2 or i == n_techs - 1:
            shown_names.append(short[i])
        elif i == 2:
            shown_names.append("...")
        else:
            shown_names.append("")
    ax2.set_yticks(list(range(n_techs)))
    ax2.set_yticklabels(shown_names, fontsize=7)

    # Cell annotations — drift-out %
    for ti in range(n_techs):
        for ci in range(n_classes):
            val = drift_out[ti, ci]
            txt_color = "white" if val > 30 else "black"
            ax2.text(ci, ti, f"{val:.0f}", ha="center", va="center",
                     fontsize=7, color=txt_color)

    # Cell grid lines
    ax2.set_xticks([x - 0.5 for x in x_ticks2], minor=True)
    ax2.set_yticks([y - 0.5 for y in range(n_techs)], minor=True)
    ax2.grid(which="minor", color="#666", linewidth=0.3, alpha=0.6)

    # Colorbar
    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.02)
    cbar2.set_label("Misclass. Rate (%)", fontsize=9)

    ax2.set_xlabel("True Expression Class", fontsize=10, fontweight="bold")
    ax2.set_title(
        "Drift-Out: % Misclassified into Any Other Class — Average Across Techniques",
        fontsize=10, fontweight="bold",
    )

    # Divider line between panels
    divider_y = 0.49
    ax1.axhline(y=n_techs - 0.5 + 0.5, color="#333", linewidth=1.2, xmin=0, xmax=1)

    fig.tight_layout()
    _savefig_safe(fig, output_path, dpi=300)
    plt.close(fig)
    print(f"Drift heatmap: {output_path}.png  |  {output_path}.pdf")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Expression drift heatmap (matrix form)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset", required=True, choices=["rafd", "kdef"])
    parser.add_argument("--techniques", nargs="+", default=None)
    parser.add_argument("--output", type=str, default="expression_drift_heatmap")
    parser.add_argument("--weight", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=32)

    args = parser.parse_args()

    print("=" * 60)
    print(f"Expression Drift Heatmap (Matrix) — {args.dataset.upper()}")
    print("=" * 60)

    # Load results CSV for technique summary
    csv_path = os.path.join(args.output_dir, "expression_results.csv")
    tech_summary = {}
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                tech = row["Technique"].strip()
                if tech == "aligned (Validation)":
                    continue
                try:
                    tech_summary[tech] = {
                        "deid_acc": float(row["DeID_Expr_Accuracy"]),
                        "preservation_rate": float(row["Expr_Preservation_Rate"]),
                    }
                except (ValueError, KeyError):
                    pass

    if not tech_summary:
        print("ERROR: No technique results found.")
        sys.exit(1)

    # Filter to requested techniques
    available = list(tech_summary.keys())
    if args.techniques:
        available = [t for t in args.techniques if t in tech_summary]

    print(f"Techniques: {len(available)}")

    # Load model
    weight_path = args.weight or os.path.join(EXTRACT_ATTRS_DIR, "models", "swinface",
                                              "checkpoint_step_79999_gpu_0.pt")
    if not os.path.exists(weight_path):
        print(f"ERROR: Checkpoint not found: {weight_path}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_swinface_model(weight_path, device)
    model.eval()
    print(f"Model loaded on {device}")

    # Load GT + valid stems
    datasets_base = Path(EXTRACT_ATTRS_DIR).parent / "deid-toolkit" / "root_dir" / "datasets"
    labels_dir = datasets_base / "labels"

    if args.dataset == "kdef":
        gt = load_kdef_gt(str(labels_dir))
    else:
        gt = load_rafd_gt(str(labels_dir))

    valid_stems = sorted([s for s in gt if gt[s]["expression"] is not None])
    print(f"GT stems: {len(valid_stems)}")

    # Predict on aligned (baseline)
    aligned_dir = str(datasets_base / "aligned" / args.dataset)
    pred_aligned = predict_images(aligned_dir, valid_stems, model, device, args.batch_size)
    n_labeled = len(valid_stems)
    baseline_correct = sum(1 for s in valid_stems if gt[s]["expression"] is not None and
                           pred_aligned.get(s) == gt[s]["expression"])
    baseline_acc = baseline_correct / n_labeled * 100

    # Predict per technique
    all_cm_data = {}
    for tech in available:
        tech_dir = str(datasets_base / tech / args.dataset)
        if not os.path.isdir(tech_dir):
            print(f"  SKIP — no dir: {tech_dir}")
            continue

        preds = predict_images(tech_dir, valid_stems, model, device, args.batch_size)
        cm_data = []
        for stem in valid_stems:
            gt_expr = gt[stem]["expression"]
            pred_expr = preds.get(stem)
            if gt_expr is not None and pred_expr is not None:
                cm_data.append((gt_expr, pred_expr))
        all_cm_data[tech] = cm_data

    # Compute baseline accuracy from confusion data
    aligned_cm = [(gt[s]["expression"], pred_aligned.get(s)) for s in valid_stems
                  if gt[s]["expression"] is not None and pred_aligned.get(s) is not None]
    baseline_correct = sum(1 for g, p in aligned_cm if g == p)
    n_labeled = len(aligned_cm)
    baseline_acc = baseline_correct / n_labeled * 100 if n_labeled > 0 else 0.0

    print(f"Baseline: {baseline_acc:.0f}% ({baseline_correct}/{n_labeled})")
    print(f"CM data collected for {len(all_cm_data)} technique(s).")

    # Generate heatmap
    output_path = os.path.join(args.output_dir, f"{args.output}")
    plot_drift_heatmap(all_cm_data, tech_summary, baseline_acc, output_path, args.dataset)
    print("Done.")


if __name__ == "__main__":
    main()
