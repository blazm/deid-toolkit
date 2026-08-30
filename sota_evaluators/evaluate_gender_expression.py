#!/usr/bin/env python3
"""
SwinFace Gender & Expression Evaluation on DEID Baselines.

Runs SwinFace gender and expression predictions on aligned (original) faces
and their anonymized versions across multiple DEID techniques, then compares
against ground-truth labels to produce:

  1. Side-by-side tabular results per technique
     - `gender_results.csv`  — aligned accuracy, de-identified accuracy, preservation rate
     - `expression_results.csv` — aligned accuracy, de-identified accuracy, preservation rate
  2. Confusion matrix visualization
     - `expression_confusion_matrix.png` — per-technique heatmaps of predictions vs ground truth

Datasets supported: RaFD, MUG, KDEF (extensible via dataset-specific loaders).

Usage:
    conda run -n swinface python evaluate_gender_expression.py ^
        --dataset rafd ^
        --techniques AIDPro AMT-GAN DeepPrivacy DeepPrivacy2 G2Face ^
            GANonymization IPFA RiDDLE CLEANIR NullFace FADM FAMS ^
        --output D:\\dev\\deid-toolkit\\root_dir\\predictions\\rafd

    conda run -n swinface python evaluate_gender_expression.py ^
        --dataset mug-still ^
        --techniques AIDPro AMT-GAN DeepPrivacy G2Face CLEANIR RiDDLE ^
        --output D:\\dev\\deid-toolkit\\root_dir\\predictions\\mug-still

    conda run -n swinface python evaluate_gender_expression.py ^
        --dataset kdef ^
        --techniques AIDPro AMT-GAN DeepPrivacy G2Face CLEANIR RiDDLE ^
        --output D:\\dev\\deid-toolkit\\root_dir\\predictions\\kdef
"""

import argparse
import csv
import os
import sys
from pathlib import Path

# Fix OpenMP duplicate lib conflict on Windows (torch + cv2 each load their own libiomp5md.dll)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Reuse from existing attribute extractor ────────────────────────────────

EXTRACT_ATTRS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EXTRACT_ATTRS_DIR)

from extract_attributes_swinface import (
    preprocess_image,
    build_swinface_model,
    EXPRESSION_LABELS,  # ["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"]
)


# ── Dataset-specific label loaders ────────────────────────────────────────

EXPRESSION_LABELS = list(EXPRESSION_LABELS)  # ["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"]
EXPR_IDX_MAP = {label: i for i, label in enumerate(EXPRESSION_LABELS)}

# MUG Emotion_code → SwinFace expression index
MUG_EXPR_CODE_TO_IDX = {
    0: 6,   # Neutral
    1: 0,   # Anger -> Angry
    4: 1,   # Disgust
    5: 2,   # Fear
    6: 3,   # Happy
    7: 4,   # Sadness -> Sad
    8: 5,   # Surprise
}

# KDEF Emotion_code → SwinFace expression index
# Note: Scream(2) and Contempt(3) do NOT map to any SwinFace class; those images are excluded.
KDEF_EXPR_CODE_TO_IDX = {
    0: 6,   # Neutral
    1: 0,   # Anger -> Angry
    4: 1,   # Disgust
    5: 2,   # Fear
    6: 3,   # Happy
    7: 4,   # Sadness -> Sad
    8: 5,   # Surprise
}

# Valid KDEF emotion codes that map to SwinFace classes
VALID_KDEF_EXPR_CODES = set(KDEF_EXPR_CODE_TO_IDX.keys())


def load_rafd_labels(labels_path):
    """Load RaFD ground truth labels.

    Returns dict: {filename_stem: {"gender": "Male"/"Female", "expression": EXPRESSION_LABELS index}}
    """
    gt = {}
    if not os.path.exists(labels_path):
        print(f"  WARNING: Labels file not found: {labels_path}")
        return gt

    with open(labels_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("Name", "").strip()
            if not name:
                continue
            stem = os.path.splitext(name)[0]

            # Gender: Gender_code -1=Female, 1=Male
            gender_code = row.get("Gender_code", "").strip()
            if gender_code == "1":
                gender = "Male"
            elif gender_code == "-1":
                gender = "Female"
            else:
                continue  # skip unlabeled

            # Expression: find the active binary column
            expr_idx = None
            for label in EXPRESSION_LABELS:
                col_name = label if label != "Sad" else "Sadness"
                val = row.get(col_name, "").strip()
                if val == "1":
                    expr_idx = EXPR_IDX_MAP[label]
                    break

            gt[stem] = {"gender": gender, "expression": expr_idx}

    return gt


def load_mug_labels(labels_path):
    """Load MUG-Still ground truth labels.

    Returns dict: {filename_stem: {"gender": "Male"/"Female"/None, "expression": EXPRESSION_LABELS index}}
    (Gender assigned per subject, author-verified; see mug_still_subject_gender.csv.)
    """
    gt = {}
    if not os.path.exists(labels_path):
        print(f"  WARNING: Labels file not found: {labels_path}")
        return gt

    with open(labels_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("Name", "").strip()
            if not name:
                continue
            stem = os.path.splitext(name)[0]

            # Gender: filled from verified per-subject mapping ("M"/"F"), see
            # mug_still_subject_gender.csv (30 M / 22 F across 52 subjects)
            g = row.get("Gender", "").strip()
            if g == "M":
                gender = "Male"
            elif g == "F":
                gender = "Female"
            else:
                gender = None

            # Expression from Emotion_code column
            emo_code_str = row.get("Emotion_code", "").strip()
            expr_idx = None
            if emo_code_str:
                try:
                    emo_code = int(emo_code_str)
                    expr_idx = MUG_EXPR_CODE_TO_IDX.get(emo_code)
                except ValueError:
                    pass

            gt[stem] = {"gender": gender, "expression": expr_idx}

    return gt


def load_kdef_labels(labels_path):
    """Load KDEF ground truth labels.

    Returns dict: {filename_stem: {"gender": None, "expression": EXPRESSION_LABELS index}}
    Gender is not available; images with Scream/Contempt labels are excluded.
    """
    gt = {}
    if not os.path.exists(labels_path):
        print(f"  WARNING: Labels file not found: {labels_path}")
        return gt

    with open(labels_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("Name", "").strip()
            if not name:
                continue
            stem = os.path.splitext(name)[0]

            # Gender: filled from verified per-actor mapping ("M"/"F")
            g = row.get("Gender", "").strip()
            if g == "M":
                gender = "Male"
            elif g == "F":
                gender = "Female"
            else:
                gender = None

            # Check for Scream or Contempt (no SwinFace equivalent) — skip those images
            scream_val = row.get("Scream", "").strip()
            contempt_val = row.get("Contempt", "").strip()
            if scream_val == "1" or contempt_val == "1":
                continue

            # Expression from Emotion_code column (same schema as MUG)
            emo_code_str = row.get("Emotion_code", "").strip()
            expr_idx = None
            if emo_code_str:
                try:
                    emo_code = int(emo_code_str)
                    expr_idx = KDEF_EXPR_CODE_TO_IDX.get(emo_code)
                except ValueError:
                    pass

            gt[stem] = {"gender": gender, "expression": expr_idx}

    return gt


# Map dataset name -> loader function + labels filename pattern
def load_celeba_labels(labels_path):
    """Load CelebA ground truth labels (gender only, no expression data).

    Returns dict: {filename_stem: {"gender": "Male"/"Female", "expression": None}}
    """
    gt = {}
    if not os.path.exists(labels_path):
        print(f"  WARNING: Labels file not found: {labels_path}")
        return gt

    with open(labels_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("Name", "").strip()
            if not name:
                continue
            stem = os.path.splitext(name)[0]

            # Gender: Gender_code -1=Female, 1=Male
            gender_code = row.get("Gender_code", "").strip()
            if gender_code == "1":
                gender = "Male"
            elif gender_code == "-1":
                gender = "Female"
            else:
                continue

            gt[stem] = {"gender": gender, "expression": None}

    return gt


DATASET_LOADERS = {
    "rafd":   (load_rafd_labels,   "rafd-frontal_aligned_labels.csv"),
    "mug-still": (load_mug_labels, "mug-still_labels.csv"),
    "kdef":   (load_kdef_labels,   "kdef_labels.csv"),
    "celeba-test": (load_celeba_labels, "celeba-test_labels.csv"),
}


# ── Discovery helpers ─────────────────────────────────────────────────────

def find_image_path(directory, stem):
    """Find image file for a given stem, trying common extensions."""
    d = Path(directory)
    for ext in (".png", ".PNG", ".jpg", ".jpeg", ".bmp"):
        candidate = d / (stem + ext)
        if candidate.exists():
            return candidate
    # Fallback: any extension
    for p in d.iterdir():
        if p.stem == stem and p.is_file():
            return p
    return None


def get_image_stems(directory):
    """Return sorted stems from a directory."""
    stems = set()
    for ext in (".png", ".jpg", ".jpeg", ".bmp"):
        for p in Path(directory).glob(f"*{ext}"):
            stems.add(p.stem)
    return sorted(stems)


def discover_technique_dirs(technique_names, dataset_name):
    """Discover de-identified image directories for the given techniques.

    Returns list of (technique_name, technique_image_dir) sorted alphabetically.
    Skips techniques whose directory doesn't contain images for this dataset.
    """
    datasets_base = Path(EXTRACT_ATTRS_DIR).parent / "deid-toolkit" / "root_dir" / "datasets"
    results = []

    for tech in (technique_names or sorted(set(p.name for p in datasets_base.iterdir() if p.is_dir() and p.name not in ("aligned", "labels", "original", "pairs", "deidentified")))):
        tech_dir = datasets_base / tech / dataset_name
        if tech_dir.is_dir():
            images = list(tech_dir.glob("*.jpg")) + list(tech_dir.glob("*.png"))
            if images:
                results.append((tech, str(tech_dir)))

    # Sort alphabetically (skip "aligned" as it's the baseline, not a technique)
    results.sort(key=lambda x: x[0])
    return results


# ── Prediction helpers ────────────────────────────────────────────────────

def predict_batch(image_stems, image_dir, model, device, batch_size=32):
    """Run SwinFace inference on a directory of images.

    Returns:
        dict: {stem: {"gender": "Male"/"Female", "expression_idx": int}}
              None entries for stems where the image couldn't be loaded.
    """
    results = {}

    # Build path list
    paths = []
    valid_stems = []
    for stem in image_stems:
        img_path = find_image_path(image_dir, stem)
        if img_path is not None:
            paths.append((stem, img_path))
            valid_stems.append(stem)

    # Process in batches
    for start in range(0, len(paths), batch_size):
        end = min(start + batch_size, len(paths))
        batch_tensors = []
        batch_indices = []

        for i in range(start, end):
            stem, img_path = paths[i]
            try:
                t = preprocess_image(img_path)
                batch_tensors.append(t)
                batch_indices.append(i)
            except (ValueError, OSError):
                pass

        if not batch_tensors:
            continue

        batch = torch.stack(batch_tensors).to(device)
        with torch.no_grad():
            outputs = model(batch)

        # Gender: argmax on logits → 0=Female, 1=Male
        gender_logits = outputs["Gender"].float()
        gender_preds = gender_logits.argmax(dim=1).cpu().numpy()

        # Expression: argmax on logits → index into EXPRESSION_LABELS
        expression_logits = outputs["Expression"].float()
        expression_preds = expression_logits.argmax(dim=1).cpu().numpy()

        for j, batch_idx in enumerate(batch_indices):
            stem, _ = paths[batch_idx]
            results[stem] = {
                "gender": "Male" if gender_preds[j] == 1 else "Female",
                "expression_idx": int(expression_preds[j]),
            }

    return results


# ── Evaluation & reporting ────────────────────────────────────────────────

def evaluate_technique(tech_name, tech_dir, aligned_results, deid_results, gt):
    """Evaluate a single technique: compute metrics vs ground truth.

    Returns dict with per-technique metrics.
    """
    results = {"name": tech_name}

    # Find common stems that have both GT and predictions from both aligned and deid
    aligned_set = set(aligned_results.keys())
    deid_set = set(deid_results.keys())
    gt_set = set(gt.keys())
    common = sorted(aligned_set & deid_set & gt_set)

    results["n_matched"] = len(common)

    # Gender evaluation (only if GT has gender)
    has_gender_gt = any(v.get("gender") is not None for v in gt.values())
    if has_gender_gt:
        aligned_correct = sum(
            1 for s in common
            if aligned_results[s]["gender"] == gt[s]["gender"]
        )
        deid_correct = sum(
            1 for s in common
            if deid_results[s]["gender"] == gt[s]["gender"]
        )
        gender_preserved = sum(
            1 for s in common
            if aligned_results[s]["gender"] == deid_results[s]["gender"]
        )

        results["aligned_gender_acc"] = aligned_correct / len(common) * 100 if common else 0.0
        results["deid_gender_acc"] = deid_correct / len(common) * 100 if common else 0.0
        results["gender_preservation_rate"] = gender_preserved / len(common) * 100 if common else 0.0
    else:
        results["aligned_gender_acc"] = None
        results["deid_gender_acc"] = None
        results["gender_preservation_rate"] = None

    # Expression evaluation — only if GT has expression labels for any common stems
    expr_labeled = sum(1 for s in common if gt[s]["expression"] is not None)
    if expr_labeled > 0:
        expr_aligned_correct = sum(
            1 for s in common
            if aligned_results[s]["expression_idx"] == gt[s]["expression"]
        )
        expr_deid_correct = sum(
            1 for s in common
            if deid_results[s]["expression_idx"] == gt[s]["expression"]
        )
        expr_preserved = sum(
            1 for s in common
            if aligned_results[s]["expression_idx"] == deid_results[s]["expression_idx"]
        )

        results["aligned_expr_acc"] = expr_aligned_correct / expr_labeled * 100
        results["deid_expr_acc"] = expr_deid_correct / expr_labeled * 100
        results["expr_preservation_rate"] = expr_preserved / len(common) * 100 if common else 0.0
    else:
        results["aligned_expr_acc"] = None
        results["deid_expr_acc"] = None
        results["expr_preservation_rate"] = None

    # Collect confusion matrix data (GT vs deid predictions) for expression
    cm_data = []  # list of (gt_label, pred_label) tuples
    for s in common:
        gt_expr = gt[s]["expression"]
        pred_expr = deid_results[s]["expression_idx"]
        if gt_expr is not None and pred_expr is not None:
            cm_data.append((gt_expr, pred_expr))

    results["cm_data"] = cm_data

    return results


def _fmt_pct(val, has_gender_gt):
    """Format a percentage value, returning 'N/A' or '—' as appropriate."""
    if not has_gender_gt:
        return "N/A"
    if val is None:
        return "—"
    return f"{val:.1f}"


def generate_gender_csv(results, output_path):
    """Write gender_results.csv — one section for baseline, one for each technique."""
    has_gender_gt = results[0]["aligned_gender_acc"] is not None

    with open(output_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Technique", "Aligned_Gender_Accuracy", "DeID_Gender_Accuracy",
                         "Gender_Preservation_Rate", "N_Matched_GT"])

        # Baseline row
        baseline = results[0]
        writer.writerow([
            "aligned (Validation)",
            _fmt_pct(baseline["aligned_gender_acc"], has_gender_gt),
            "—",
            "—",
            baseline["n_matched"],
        ])

        # Technique rows
        for r in results:
            if r["name"] == "aligned":
                continue
            writer.writerow([
                r["name"],
                _fmt_pct(r["aligned_gender_acc"], has_gender_gt),
                _fmt_pct(r["deid_gender_acc"], True),
                _fmt_pct(r["gender_preservation_rate"], True),
                r["n_matched"],
            ])


def generate_expression_csv(results, output_path):
    """Write expression_results.csv — one section for baseline, one for each technique."""
    has_expr_gt = results[0]["aligned_expr_acc"] is not None

    with open(output_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Technique", "Aligned_Expr_Accuracy", "DeID_Expr_Accuracy",
                         "Expr_Preservation_Rate", "N_Matched_GT"])

        # Baseline row
        baseline = results[0]
        writer.writerow([
            "aligned (Validation)",
            f"{baseline['aligned_expr_acc']:.1f}" if has_expr_gt else "N/A",
            "—",
            "—",
            baseline["n_matched"],
        ])

        # Technique rows
        for r in results:
            if r["name"] == "aligned":
                continue
            writer.writerow([
                r["name"],
                f"{r['aligned_expr_acc']:.1f}" if has_expr_gt else "N/A",
                f"{r['deid_expr_acc']:.1f}" if has_expr_gt else "N/A",
                f"{r['expr_preservation_rate']:.1f}" if has_expr_gt else "N/A",
                r["n_matched"],
            ])


def _render_cm_grid(axes, technique_results, nrows, ncols):
    """Render per-technique confusion matrix subplots into a pre-sized axes grid.

    Tight layout: percentage-only annotations, no per-subplot colorbars,
    small fonts, minimal spacing. Designed for dense grids (4x4 up to 8x2).

    Returns list of (name, deid_expr_acc, expr_preservation_rate, n_matched) tuples
    and the baseline accuracy for caption generation.
    """
    # Per-technique accuracy stats for caption
    tech_accuracies = []
    for r in technique_results:
        expr_acc = r["deid_expr_acc"]
        expr_pres = r["expr_preservation_rate"]
        n = r["n_matched"]
        tech_accuracies.append((r["name"], expr_acc, expr_pres, n))

    # Find best and worst techniques for caption highlights
    ranked = sorted(tech_accuracies, key=lambda x: x[1])
    worst_name, worst_acc, _, _ = ranked[0] if ranked else ("—", 0.0, 0.0, 0)
    best_name, best_acc, _, _ = ranked[-1] if ranked else ("—", 0.0, 0.0, 0)

    # Shared colorbar axis at bottom-right of the whole figure
    has_data = any(r.get("cm_data") for r in technique_results)

    for i, r in enumerate(technique_results):
        row, col = divmod(i, ncols)
        ax = axes[row, col]

        cm_data = r.get("cm_data", [])
        if not cm_data:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_title(r["name"], fontsize=9, fontweight="bold")
            ax.axis("off")
            continue

        # Build confusion matrix (always 7 classes, fixed order)
        cm = np.zeros((7, 7), dtype=np.float64)
        for gt_val, pred_val in cm_data:
            cm[gt_val, pred_val] += 1.0

        # Normalize by row (ground truth) to get %
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_norm = cm / row_sums * 100.0

        im = ax.imshow(cm_norm, cmap="YlOrRd", vmin=0, vmax=100, aspect="auto")

        # Title: short technique name + accuracy
        acc_val = r["deid_expr_acc"]
        short_name = r["name"][:12] if len(r["name"]) > 14 else r["name"]
        ax.set_title(f"{short_name} ({acc_val:.0f}%)", fontsize=7.5, fontweight="bold")

        # Minimal ticks — only show expression labels on diagonal for readability
        # X axis (predicted) on bottom row only
        if row == nrows - 1:
            ax.set_xticks(range(7))
            ax.set_xticklabels(EXPRESSION_LABELS, rotation=25, ha="right", fontsize=6.5)
        else:
            ax.set_xticks([])

        # Y axis (true) on left column only
        if col == 0:
            ax.set_yticks(range(7))
            ax.set_yticklabels(EXPRESSION_LABELS, fontsize=6.5)
        else:
            ax.set_yticks([])

        # Cell annotations: percentage only, small font
        for gi in range(7):
            for pi in range(7):
                pct = cm_norm[gi, pi]
                if pct > 0.5:
                    txt_color = "white" if pct > 70 else "black"
                    ax.text(pi, gi, f"{pct:.0f}", ha="center", va="center",
                            fontsize=6, color=txt_color)

        # Thin grid lines for cell boundaries
        ax.set_xticks(np.arange(-0.5, 7, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 7, 1), minor=True)
        ax.grid(which="minor", color="#ccc", linewidth=0.3)

    # Hide unused subplots
    for row_idx in range(nrows):
        for col_idx in range(ncols):
            cell = row_idx * ncols + col_idx
            if cell >= len(technique_results):
                axes[row_idx, col_idx].set_visible(False)

    return tech_accuracies, ranked


def generate_confusion_matrix(results, output_path, dataset_name, caption_txt=None, layouts=None):
    """Generate expression confusion matrix heatmap — one subplot per technique.

    Produces tightly cropped PNG + PDF with no title (caption in separate .txt file).
    When expression GT is unavailable, writes a placeholder caption and skips plotting.

    Parameters
    ----------
    layouts : list of (nrows, ncols) tuples, or None for auto-layout.
        Auto produces: default 4×4 grid + optional landscape/portrait variants
        when there are >9 techniques. Example: [(2,6)] for single 2x6 layout.
    """
    n_techs = len([r for r in results if r["name"] != "aligned"])
    if n_techs == 0:
        print("No techniques to plot.")
        return

    # Check if any technique has expression GT data
    has_expr_gt = any(r["aligned_expr_acc"] is not None for r in results)
    if not has_expr_gt:
        # No expression GT — write placeholder caption, skip plot
        if caption_txt:
            ds_display = {"rafd": "RaFD", "mug-still": "MUG-Still", "kdef": "KDEF"}.get(dataset_name, dataset_name)
            with open(caption_txt, "w", encoding="utf-8") as f:
                f.write(f"Expression classification confusion matrices — ground-truth expression labels are not available for the {ds_display} dataset. No plot generated.")
            print(f"Caption saved (no expr GT): {caption_txt}")
        print("Confusion matrix skipped: no expression ground truth for this dataset.")
        return

    technique_results = [r for r in results if r["name"] != "aligned"]
    technique_results.sort(key=lambda x: x["name"])

    # Compute layouts to render
    if layouts is None:
        # Auto-detect: start with default 4×4, add variants when >9 techniques
        layouts = [(4, 4)]
        if n_techs > 9:
            landscape_cols = int(np.ceil(n_techs / 2.0))
            portrait_rows = int(np.ceil(n_techs / 2.0))
            layouts.append((2, landscape_cols))   # landscape
            layouts.append((portrait_rows, 2))     # portrait
        layout_names = ["", f"_{layouts[1][0]}x{layouts[1][1]}", f"_{layouts[2][0]}x{layouts[2][1]}"]
    else:
        layout_names = [f"_{n}x{c}" for n, c in layouts]

    baseline_expr_acc = results[0]["aligned_expr_acc"] if len(results) > 0 else 0.0

    # Dataset display name
    DATASET_DISPLAY = {"rafd": "RaFD", "mug-still": "MUG-Still", "kdef": "KDEF"}
    ds_display = DATASET_DISPLAY.get(dataset_name, dataset_name)

    tech_accuracies = []
    ranked = []

    for idx, (nrows, ncols) in enumerate(layouts):
        fig_width = 4.2 * ncols
        fig_height = 4.2 * nrows
        fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height),
                                 gridspec_kw={"wspace": 0.15, "hspace": 0.18})
        axes = np.atleast_2d(axes)

        tech_accuracies, ranked = _render_cm_grid(axes, technique_results, nrows, ncols)

        # Tightly cropped layout — no suptitle, minimal margins
        fig.subplots_adjust(left=0.12, right=0.96, top=0.95, bottom=0.10)

        # Save PNG + PDF
        base = os.path.splitext(output_path)[0] if output_path.endswith(".png") else output_path
        png_path = f"{base}{layout_names[idx]}.png"
        pdf_path = f"{base}{layout_names[idx]}.pdf"

        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"Confusion matrix ({nrows}x{ncols}): {png_path}  |  {pdf_path}")

    # ── Generate LaTeX caption (shared across variants) ────────────────
    if caption_txt is None:
        caption_txt = output_path + ".txt"

    best_name, best_acc, _, _ = ranked[-1] if ranked else ("—", 0.0, 0.0, 0)
    worst_name, worst_acc, _, _ = ranked[0] if ranked else ("—", 0.0, 0.0, 0)

    best_expr_pres, worst_expr_pres = max(tech_accuracies, key=lambda x: x[2])[2], min(tech_accuracies, key=lambda x: x[2])[2]
    best_expr_pres_name = max(tech_accuracies, key=lambda x: x[2])[0]
    worst_expr_pres_name = min(tech_accuracies, key=lambda x: x[2])[0]

    expr_pres_min = min(a[2] for a in tech_accuracies)
    expr_pres_max = max(a[2] for a in tech_accuracies)
    expr_pres_min_name = min(tech_accuracies, key=lambda x: x[2])[0]
    expr_pres_max_name = max(tech_accuracies, key=lambda x: x[2])[0]

    caption_lines = []
    caption_lines.append(f"Expression classification confusion matrices for each anonymization technique on the {ds_display} dataset.")
    caption_lines.append(f"SwinFace was trained to recognize 7 expression classes (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral).")
    caption_lines.append(f"The aligned (original) faces serve as a baseline: SwinFace achieves {baseline_expr_acc:.0f}% top-1 accuracy on these ground-truth labels.")
    caption_lines.append(f"After anonymization, expression prediction accuracy drops significantly across all techniques ({best_name}: {best_acc:.0f}% to {worst_name}: {worst_acc:.0f}%), indicating that facial expression information is largely erased.")
    caption_lines.append(f"Expression preservation rates range from {expr_pres_min:.0f}% ({expr_pres_min_name}) to {expr_pres_max:.0f}% ({expr_pres_max_name}),")
    caption_lines.append(f"meaning that between {expr_pres_min:.0f}\\% and {expr_pres_max:.0f}\\% of images retain the same predicted expression after anonymization.")
    caption_lines.append(f"Diagonal elements show correct predictions; off-diagonal elements reveal systematic misclassification patterns.")

    with open(caption_txt, "w", encoding="utf-8") as f:
        f.write(" ".join(caption_lines))
    print(f"Caption saved: {caption_txt}")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SwinFace gender & expression evaluation across DEID baselines")
    parser.add_argument("--dataset", required=True, choices=list(DATASET_LOADERS.keys()),
                        help="Dataset to evaluate: rafd, mug-still, or kdef")
    parser.add_argument("--techniques", nargs="+", default=None,
                        help="Technique names to evaluate (default: all available)")
    parser.add_argument("--output", required=True,
                        help="Output directory for results")
    parser.add_argument("--weight", type=str, default=None,
                        help="SwinFace checkpoint path")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for inference (default: 32)")
    parser.add_argument("--layouts", nargs="+", metavar="NROWSxCOLS",
                        help="Grid layouts to produce, e.g. '2x6 8x2' (auto when omitted and >9 techs)")

    args = parser.parse_args()

    # Parse --layouts into list of (nrows, ncols) tuples
    cm_layouts = None
    if args.layouts:
        cm_layouts = []
        for spec in args.layouts:
            parts = spec.split("x")
            if len(parts) != 2 or not parts[0].isdigit() or not parts[1].isdigit():
                print(f"ERROR: Invalid layout spec '{spec}'. Use format N x M, e.g. '8x2'.")
                sys.exit(1)
            cm_layouts.append((int(parts[0]), int(parts[1])))

    # Resolve weight path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    weight_path = args.weight or os.path.join(
        script_dir, "models", "swinface", "checkpoint_step_79999_gpu_0.pt")
    if not os.path.exists(weight_path):
        print(f"ERROR: Checkpoint not found: {weight_path}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_name = args.dataset

    print("=" * 70)
    print(f"SwinFace Gender & Expression Evaluation — {dataset_name.upper()}")
    print("=" * 70)
    print(f"Device:           {device}")
    print(f"Checkpoint:       {weight_path}")
    print()

    # ── Load ground truth ────────────────────────────────────────────────
    loader_func, labels_filename = DATASET_LOADERS[dataset_name]
    labels_dir = Path(EXTRACT_ATTRS_DIR).parent / "deid-toolkit" / "root_dir" / "datasets" / "labels"
    labels_path = labels_dir / labels_filename

    print("Loading ground truth labels...")
    gt = loader_func(str(labels_path))
    print(f"  Loaded {len(gt)} labeled image(s) from {labels_path}")

    if not gt:
        print("ERROR: No ground truth labels loaded. Check labels file path.")
        sys.exit(1)

    # ── Load model ───────────────────────────────────────────────────────
    print(f"\nLoading SwinFace model on {device}...")
    model = build_swinface_model(weight_path, device)
    model.eval()
    print("Model loaded.")

    # ── Discover aligned images ──────────────────────────────────────────
    datasets_base = Path(EXTRACT_ATTRS_DIR).parent / "deid-toolkit" / "root_dir" / "datasets"
    aligned_dir = str(datasets_base / "aligned" / dataset_name)
    if not os.path.isdir(aligned_dir):
        print(f"ERROR: Aligned images directory not found: {aligned_dir}")
        sys.exit(1)

    # Filter stems to only those that have both GT and images in the aligned folder
    gt_stems = sorted(gt.keys())
    all_aligned_stems = set(get_image_stems(aligned_dir))
    eval_stems = [s for s in gt_stems if s in all_aligned_stems]

    if not eval_stems:
        print(f"ERROR: No matching aligned images found in {aligned_dir}")
        sys.exit(1)

    print(f"\nEvaluating on {len(eval_stems)} image(s) with both GT and images.")

    # ── Predict on aligned originals ─────────────────────────────────────
    print("\nPredicting on aligned originals...")
    aligned_results = predict_batch(eval_stems, aligned_dir, model, device, args.batch_size)
    aligned_matching = sum(1 for s in eval_stems if s in aligned_results)
    print(f"  Aligned predictions: {aligned_matching}/{len(eval_stems)} images processed.")

    # ── Discover and process techniques ──────────────────────────────────
    tech_dirs = discover_technique_dirs(args.techniques, dataset_name)
    if not tech_dirs:
        print("\nERROR: No technique directories found with images for this dataset.")
        sys.exit(1)

    print(f"\nFound {len(tech_dirs)} technique(s): {', '.join(t[0] for t in tech_dirs)}")
    print()

    results = []  # list of result dicts (aligned first, then techniques)

    # ── Baseline: aligned vs GT ──────────────────────────────────────────
    print(f"--- Baseline: Aligned vs Ground Truth ---")
    baseline_result = evaluate_technique("aligned", None, aligned_results, aligned_results, gt)
    results.append(baseline_result)

    # Build gender info string (handle missing GT gracefully)
    if baseline_result["aligned_gender_acc"] is not None:
        gender_str = f"Gender Acc={baseline_result['aligned_gender_acc']:.1f}%"
    else:
        gender_str = "Gender Acc=N/A"

    # Build expression info string
    if baseline_result["aligned_expr_acc"] is not None:
        expr_str = f"Expr Acc={baseline_result['aligned_expr_acc']:.1f}%"
    else:
        expr_str = "Expr Acc=N/A"

    print(f"  {baseline_result['name']}: {gender_str}, {expr_str}, "
          f"N={baseline_result['n_matched']}")

    # ── Per-technique evaluation ─────────────────────────────────────────
    print()
    for tech_name, tech_dir in tech_dirs:
        print(f"--- {tech_name} ---")

        deid_stems = sorted(set(get_image_stems(tech_dir)) & set(eval_stems))
        if not deid_stems:
            print(f"  SKIP — no matching images for this dataset.")
            continue

        # Predict on de-identified images
        deid_results = predict_batch(deid_stems, tech_dir, model, device, args.batch_size)
        deid_matching = sum(1 for s in deid_stems if s in deid_results)
        print(f"  De-identified predictions: {deid_matching}/{len(deid_stems)} images processed.")

        # Evaluate
        tech_result = evaluate_technique(tech_name, tech_dir, aligned_results, deid_results, gt)
        results.append(tech_result)

        gender_str = ""
        if tech_result["aligned_gender_acc"] is not None:
            gender_str = f" Gender Acc={tech_result['deid_gender_acc']:.1f}%, "

        # Build expression string (handle missing GT gracefully)
        if tech_result["aligned_expr_acc"] is not None:
            expr_str = f" Expr Acc={tech_result['deid_expr_acc']:.1f}%"
        else:
            expr_str = " Expr Acc=N/A"

        # Build preservation info string (handle missing GT gracefully)
        if tech_result["gender_preservation_rate"] is not None and tech_result["expr_preservation_rate"] is not None:
            preserve_str = f"preservation: Gender={tech_result['gender_preservation_rate']:.1f}%, Expr={tech_result['expr_preservation_rate']:.1f}%"
        elif tech_result["expr_preservation_rate"] is not None:
            preserve_str = f"preservation: Expr={tech_result['expr_preservation_rate']:.1f}% (no gender GT)"
        else:
            preserve_str = "preservation: N/A (no expression GT)"

        print(f"  {gender_str}{expr_str} "
              f"(Aligned pred vs DeID pred: {tech_result['name']} "
              f"{preserve_str}), "
              f"N={tech_result['n_matched']}")
        print()

    # ── Write outputs ────────────────────────────────────────────────────
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save ground truth for reference
    gt_path = output_dir / "ground_truth.csv"
    with open(gt_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Stem", "Gender_GT", "Expression_GT_idx"])
        for stem in sorted(gt.keys()):
            entry = gt[stem]
            writer.writerow([stem, entry["gender"] if entry["gender"] else "N/A",
                             str(entry["expression"]) if entry["expression"] is not None else "N/A"])
    print(f"Ground truth saved: {gt_path}")

    # Save results CSVs
    gender_csv = output_dir / "gender_results.csv"
    expression_csv = output_dir / "expression_results.csv"
    generate_gender_csv(results, str(gender_csv))
    generate_expression_csv(results, str(expression_csv))
    print(f"Gender results:     {gender_csv}")
    print(f"Expression results: {expression_csv}")

    # Generate confusion matrix plot
    cm_path = output_dir / "expression_confusion_matrix.png"
    caption_txt = output_dir / "expression_confusion_matrix_caption.txt"
    generate_confusion_matrix(results, str(cm_path), dataset_name, str(caption_txt), layouts=cm_layouts)

    print()
    print("=" * 70)
    print(f"Evaluation complete. Results saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
