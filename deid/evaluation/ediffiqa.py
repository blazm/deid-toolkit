"""eDifFIQA — Face Image Quality Assessment evaluation metric.

Scores each image independently (no pairwise comparison needed).
Outputs per-image quality scores for both aligned and deidentified images.

Usage (via pipeline):
    python ediffiqa.py <aligned_path> <deid_path> --dataset_name X --technique_name Y ...

Variant selection: --variant T|S|M|L (default: S)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

import utils as util


def score_images(image_dir: str, model, transform, device: str):
    """Score all images in a directory. Returns dict of {filename: quality_score}."""
    files = sorted(f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")))
    scores = {}

    # Batch processing for speed
    batch_size = 32
    for i in range(0, len(files), batch_size):
        batch_files = files[i : i + batch_size]
        batch_tensors = []
        valid_indices = []

        for idx, fname in enumerate(batch_files):
            fpath = os.path.join(image_dir, fname)
            try:
                img = Image.open(fpath).convert("RGB")
                tensor = transform(img)  # (C, H, W) in [-1, 1]
                batch_tensors.append(tensor)
                valid_indices.append(idx)
            except Exception:
                continue

        if not batch_tensors:
            continue

        batch = torch.stack(batch_tensors).to(device)
        with torch.no_grad():
            preds = model(batch).cpu().numpy().flatten()

        for vi, pred in zip(valid_indices, preds):
            scores[batch_files[vi]] = float(pred)

    return scores


def main():
    args = util.read_args()

    # Parse variant (add to existing parser via re-parsing)
    variant = "S"  # default: ResNet-18 backbone
    for i, arg in enumerate(sys.argv):
        if arg == "--variant":
            variant = sys.argv[i + 1].upper().replace("EDIFFIQA", "")
            break

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Force CPU for sm_120 GPUs — PyTorch doesn't support them yet
    if os.environ.get("DEID_FORCE_CPU", "0") == "1":
        device = "cpu"

    # Locate weights directory (sibling of eval scripts)
    eval_package_dir = Path(args.eval_package_dir)
    weights_root = eval_package_dir / "weights"
    variant_dir = weights_root / f"ediffiqa{variant}"

    if not variant_dir.exists():
        print(f"[ediffiqa] Variant dir not found: {variant_dir}")
        print(f"[ediffiqa] Available variants: {[d.name for d in weights_root.iterdir() if d.is_dir()]}")
        sys.exit(1)

    # Load model
    from deid.evaluation.weights.model.ediffiqa import load_ediffiqa
    model, transform = load_ediffiqa(f"ediffiqa{variant}", variant_dir, device)
    print(f"[ediffiqa] Loaded ediffiqa{variant} on {device}")

    # Score aligned images (baseline quality)
    print(f"[ediffiqa] Scoring aligned: {args.aligned_path}")
    aligned_scores = score_images(args.aligned_path, model, transform, device)

    # Score deidentified images (post-deid quality)
    print(f"[ediffiqa] Scoring deidentified: {args.deidentified_path}")
    deid_scores = score_images(args.deidentified_path, model, transform, device)

    # Common image names (only images present in both sets)
    common = sorted(set(aligned_scores.keys()) & set(deid_scores.keys()))

    # Build metrics CSVs
    # 1. Aligned baseline
    aligned_df = util.Metrics(name_score="quality_aligned")
    for fname in common:
        aligned_df.add_score(img=fname, metric_result=aligned_scores[fname])

    # 2. Deidentified quality
    deid_df = util.Metrics(name_score="quality_deid")
    for fname in common:
        deid_df.add_score(img=fname, metric_result=deid_scores[fname])

    # 3. Delta (how much quality changed after deid — lower is more degradation)
    delta_df = util.Metrics(name_score="quality_delta")
    for fname in common:
        delta = deid_scores[fname] - aligned_scores[fname]
        delta_df.add_score(img=fname, metric_result=delta)

    # Save
    save_dir = os.path.dirname(args.save_path) or "."
    os.makedirs(save_dir, exist_ok=True)

    aligned_csv = args.save_path.replace(".csv", "") + "_aligned.csv"
    deid_csv = args.save_path.replace(".csv", "") + "_deid.csv"
    delta_csv = args.save_path.replace(".csv", "") + "_delta.csv"

    aligned_df.save_to_csv(aligned_csv)
    deid_df.save_to_csv(deid_csv)
    delta_df.save_to_csv(delta_csv)

    # Print summary stats
    def summary(label, scores):
        arr = np.array(list(scores.values()))
        print(f"  {label}: mean={arr.mean():.4f} std={arr.std():.4f} min={arr.min():.4f} max={arr.max():.4f}")

    print(f"\n[ediffiqa] Summary (common images: {len(common)}):")
    summary("Aligned", {k: aligned_scores[k] for k in common})
    summary("Deidentified", {k: deid_scores[k] for k in common})
    summary("Delta (deid - aligned)", {k: deid_scores[k] - aligned_scores[k] for k in common})

    print(f"\n[ediffiqa] Results saved:")
    print(f"  Aligned:   {aligned_csv}")
    print(f"  Deidentified: {deid_csv}")
    print(f"  Delta:     {delta_csv}")


if __name__ == "__main__":
    main()
