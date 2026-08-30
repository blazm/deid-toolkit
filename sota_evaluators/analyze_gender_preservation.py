#!/usr/bin/env python3
"""
Gender Preservation Analysis — De-identification leakage quantification.

For each baseline, predicts gender on aligned (original) faces and their anonymized versions,
then computes:
  - Gender match rate: % of faces where original & de-identified get the SAME prediction.
    Lower = better privacy (gender erased), but too low (<50%) means destructive DEID.
  - Ground-truth validation accuracy: how well SwinFace's gender predictions align with
    CelebA labels on the aligned originals (sanity check).

Output: horizontal bar chart — manuscript-ready PNG + SVG + PDF.

Usage:
    python analyze_gender_preservation.py ^
        --aligned-dir "D:/dev/deid-toolkit/root_dir/datasets/aligned/celeba-test" ^
        --baselines   "D:/dev/deid-toolkit/root_dir/embeddings/SwinFace/datasets" ^
        --labels      "D:/dev/deid-toolkit/root_dir/datasets/labels/celeba-test_labels.csv" ^
        --output      gender_preservation.pdf
"""

import argparse
import csv
import os
import sys
from pathlib import Path

# Fix OpenMP duplicate lib conflict on Windows (torch + cv2 each load their own libiomp5md.dll)
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Import from existing attribute extractor ──────────────────────────────
# Reuse the SwinFace model building and inference code — no duplication.

EXTRACT_ATTRS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EXTRACT_ATTRS_DIR)

from extract_attributes_swinface import (
    preprocess_image,
    build_swinface_model,
    EXPRESSION_LABELS,
)


# ── Label loading ────────────────────────────────────────────────────────

def load_celeba_labels(labels_path):
    """Load CelebA ground truth labels. Returns dict: {image_stem: 'Male'|'Female'}."""
    labels = {}
    if not os.path.exists(labels_path):
        return labels
    with open(labels_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_name = row.get("Image_Name", "") or row.get("Name", "")
            gender_code = row.get("Gender_code", "").strip()
            if not img_name or not gender_code:
                continue
            stem = os.path.splitext(img_name)[0]
            labels[stem] = "Male" if gender_code == "1" else "Female"
    return labels


# ── Discovery ────────────────────────────────────────────────────────────

def discover_baselines(baselines_dir):
    """Find technique directories containing celeba-test/*.jpg or *.png."""
    bd = Path(baselines_dir)
    techniques = []
    for d in sorted(bd.iterdir()):
        if not d.is_dir():
            continue
        ceiba = d / "celeba-test"
        jpgs = list(ceiba.glob("*.jpg")) + list(ceiba.glob("*.png"))
        if jpgs:
            techniques.append((d.name, ceiba))
    return sorted(techniques, key=lambda x: x[0])


def get_image_stems(directory):
    """Return sorted stems from a directory."""
    stems = set()
    for ext in (".jpg", ".png"):
        for p in Path(directory).glob(f"*{ext}"):
            stems.add(p.stem)
    return sorted(stems)


def find_image_path(directory, stem):
    """Find image file for a given stem, trying .png then .jpg."""
    d = Path(directory)
    for ext in (".png", ".PNG", ".jpg", ".jpeg"):
        candidate = d / (stem + ext)
        if candidate.exists():
            return candidate
    # Fallback: any extension
    for p in d.iterdir():
        if p.stem == stem and p.is_file():
            return p
    return None


# ── Gender prediction (batch, reuses extract_attributes pipeline) ───────

def predict_gender_batch(image_stems, aligned_dir, technique_dir, model, device, batch_size=32):
    """Predict gender for pairs of aligned + technique images.

    Returns: list of (aligned_pred_label, deid_pred_label) tuples.
    """
    import torch

    # Resolve actual paths
    paths = []
    for stem in image_stems:
        ap = find_image_path(aligned_dir, stem)
        dp = find_image_path(technique_dir, stem)
        if ap and dp:
            paths.append((ap, dp))

    n_pairs = len(paths)
    aligned_preds = []
    deid_preds = []

    # Process aligned originals in batches
    for start in range(0, n_pairs, batch_size):
        end = min(start + batch_size, n_pairs)
        batch_tensors = []
        indices = []
        for i in range(start, end):
            try:
                t = preprocess_image(paths[i][0])
                batch_tensors.append(t)
                indices.append(i)
            except (ValueError, OSError):
                pass

        if not batch_tensors:
            continue

        batch = torch.stack(batch_tensors).to(device)
        with torch.no_grad():
            outputs = model(batch)

        gender_logits = outputs["Gender"].float()
        preds = gender_logits.argmax(dim=1).cpu().numpy()
        for idx, pred in zip(indices, preds):
            aligned_preds.append((idx, "Male" if pred == 1 else "Female"))

    # Process de-identified versions in batches
    for start in range(0, n_pairs, batch_size):
        end = min(start + batch_size, n_pairs)
        batch_tensors = []
        indices = []
        for i in range(start, end):
            try:
                t = preprocess_image(paths[i][1])
                batch_tensors.append(t)
                indices.append(i)
            except (ValueError, OSError):
                pass

        if not batch_tensors:
            continue

        batch = torch.stack(batch_tensors).to(device)
        with torch.no_grad():
            outputs = model(batch)

        gender_logits = outputs["Gender"].float()
        preds = gender_logits.argmax(dim=1).cpu().numpy()
        for idx, pred in zip(indices, preds):
            deid_preds.append((idx, "Male" if pred == 1 else "Female"))

    # Pair up by index
    aligned_dict = dict(aligned_preds)
    deid_dict = dict(deid_preds)
    results = []
    for i in range(n_pairs):
        al_lbl = aligned_dict.get(i, None)
        di_lbl = deid_dict.get(i, None)
        if al_lbl and di_lbl:
            results.append((al_lbl, di_lbl))

    return results


# ── Visualization ───────────────────────────────────────────────────────

def generate_figure(baseline_matches, validation_accuracy, output_path):
    """Minimalistic horizontal bar chart — technique names and match rates only."""
    n = len(baseline_matches)
    if n == 0:
        print("No baselines to plot.")
        return

    # Sort: Validation first (if present), then alphabetical by match rate ascending
    tech_names = list(baseline_matches.keys())
    if "aligned" in tech_names:
        tech_names.remove("aligned")
    tech_names.sort()
    display_order = ["aligned"] + tech_names
    display_names = ["Validation" if name == "aligned" else name for name in display_order]

    fig, ax = plt.subplots(figsize=(9, max(3.2, n * 0.22)))

    # Uniform gray bars (no color coding)
    y_pos = np.arange(n)[::-1]
    rates = [baseline_matches[name][0] for name in display_order]

    bars = ax.barh(y_pos, rates, height=1.0, color="#333333", edgecolor="none")

    # Y axis labels — left-aligned, no collision with bars
    ax.set_yticks(y_pos)
    ax.set_yticklabels(display_names, fontsize=11)
    ax.yaxis.tick_left()
    ax.yaxis.set_label_position("left")

    # X axis — enough room for even 100% + label text
    max_rate = max(rates)
    ax.set_xlim(0, min(max_rate + 5, 105))
    ax.tick_params(axis="x", labelsize=11, length=3, direction="inout")
    ax.xaxis.grid(False)

    # Minimal title
    ax.set_title("Gender Match Rate (%)", fontsize=14, fontweight="bold", pad=10)
    ax.xaxis.set_visible(True)
    ax.yaxis.set_visible(True)

    # Value labels — white text inside bars (right-aligned), no collision possible
    for bar_i in range(n):
        label_text = f"{rates[bar_i]:.1f}%"
        ax.text(rates[bar_i] - 2, y_pos[bar_i], label_text,
                va="center", ha="right", fontsize=10, fontweight="bold", color="white")

    ax.grid(True, axis="x", alpha=0.15)
    fig.tight_layout()

    # Export — 300 DPI PNG + SVG vector + PDF
    base = os.path.splitext(output_path)[0] if output_path else "gender_preservation"
    png_path = f"{base}.png"
    svg_path = f"{base}.svg"
    pdf_path = f"{base}.pdf"

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {png_path} | {svg_path} | {pdf_path}")


# ── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Gender preservation analysis across DEID baselines")
    parser.add_argument("--aligned-dir", required=True,
                        help="Aligned originals directory (e.g., .../datasets/aligned/celeba-test)")
    parser.add_argument("--baselines", required=True,
                        help="Parent dir with per-technique subfolders (e.g., .../embeddings/SwinFace/datasets)")
    parser.add_argument("--labels", required=False, default=None,
                        help="CelebA labels CSV for ground-truth validation")
    parser.add_argument("--output", default="gender_preservation.pdf",
                        help="Output path (PNG + SVG + PDF generated)")
    parser.add_argument("--weight", type=str, default=None,
                        help="SwinFace checkpoint. Default: models/swinface/checkpoint_step_79999_gpu_0.pt")
    parser.add_argument("--techniques", nargs="+", default=None,
                        help="Limit to specific technique names (e.g., CPP-DeID). All are processed if omitted.")

    args = parser.parse_args()

    # Resolve weight path
    weight_path = args.weight or os.path.join(
        EXTRACT_ATTRS_DIR, "models", "swinface", "checkpoint_step_79999_gpu_0.pt")
    if not os.path.exists(weight_path):
        print(f"ERROR: Checkpoint not found: {weight_path}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"SwinFace Gender Preservation Analysis")
    print(f"=====================================")
    print(f"Device: {device}")
    print()

    # Load model (reuse from extract_attributes_swinface)
    print("Loading SwinFace model...")
    model = build_swinface_model(weight_path, device)
    model.eval()
    print("Model loaded.")
    print()

    # Discover baselines
    techniques = discover_baselines(args.baselines)
    if not techniques:
        print(f"ERROR: No technique folders with images found in {args.baselines}")
        sys.exit(1)

    # Get image stems from aligned originals
    aligned_stems = get_image_stems(args.aligned_dir)
    if not aligned_stems:
        print(f"ERROR: No images found in {args.aligned_dir}")
        sys.exit(1)
    print(f"Reference stems: {len(aligned_stems)}")

    # Load ground-truth labels (for validation accuracy)
    gt_labels = load_celeba_labels(args.labels) if args.labels else {}

    baseline_matches = {}  # {technique_name: (match_rate, n_matched)}

    for tech_name, tech_dir in techniques:
        stems = get_image_stems(tech_dir)
        common = sorted(set(aligned_stems) & set(stems))
        if not common:
            continue

        pairs = predict_gender_batch(common, args.aligned_dir, tech_dir, model, device)
        n_pairs = len(pairs)
        if n_pairs == 0:
            print(f"SKIP {tech_name}: no valid pairs")
            continue

        matches = sum(1 for (al_lbl, di_lbl) in pairs if al_lbl == di_lbl)
        match_rate = matches / n_pairs * 100.0
        baseline_matches[tech_name] = (match_rate, n_pairs, matches)

        print(f"{tech_name:<20s}: {matches}/{n_pairs} ({match_rate:.1f}%) same gender prediction")

    # Validation accuracy: SwinFace vs ground truth on aligned originals
    validation_accuracy = 0.0
    if gt_labels:
        # Use ALL aligned stems that have ground truth labels (not just last technique's common set)
        labeled_stems = sorted(s for s in aligned_stems if s in gt_labels)
        if labeled_stems:
            import cv2

            # Predict gender on labeled original images directly
            aligned_gt_paths = [find_image_path(args.aligned_dir, s) for s in labeled_stems]
            valid_paths = [p for p in aligned_gt_paths if p is not None]

            batch_tensors = []
            indices = []
            for i, p in enumerate(valid_paths):
                try:
                    t = preprocess_image(p)
                    batch_tensors.append(t)
                    indices.append(i)
                except (ValueError, OSError):
                    pass

            if batch_tensors:
                batch = torch.stack(batch_tensors).to(device)
                with torch.no_grad():
                    outputs = model(batch)
                gender_logits = outputs["Gender"].float()
                preds = gender_logits.argmax(dim=1).cpu().numpy()
                correct = sum(1 for idx, pred in zip(indices, preds)
                              if ("Male" if pred == 1 else "Female") == gt_labels[labeled_stems[idx]])
                validation_accuracy = correct / len(valid_paths)

    print()
    if validation_accuracy > 0:
        n_labeled = sum(1 for s in aligned_stems if s in gt_labels)
        print(f"Validation accuracy on originals: {validation_accuracy*100:.1f}% ({n_labeled} labeled pairs)")

    # Write CSV with raw numbers
    csv_path = os.path.splitext(args.output)[0] + "_gender_results.csv"
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Technique", "MatchCount", "TotalPairs", "MatchRatePct", "N_Labeled_GT"])
        for tech_name, (rate, n_pairs, matches) in sorted(baseline_matches.items(), key=lambda x: x[1][0]):
            display = "Validation" if tech_name == "aligned" else tech_name
            writer.writerow([display, matches, n_pairs, f"{rate:.1f}", ""])
        writer.writerow(["---", "---", "---", "---", "---"])
        writer.writerow([f"Validation (GT)", "", "", f"{validation_accuracy*100:.1f}", sum(1 for s in aligned_stems if s in gt_labels)])

    print(f"CSV results:      {csv_path}")

    # Generate figure
    generate_figure(baseline_matches, validation_accuracy, args.output)


if __name__ == "__main__":
    main()
