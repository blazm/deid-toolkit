#!/usr/bin/env python3
"""
FaceXFormer Gender Evaluation on DEID Baselines.

Runs FaceXFormer (ICCV 2025, Narayan et al.) gender prediction on aligned
(original) faces and their anonymized versions across multiple DEID
techniques, then compares against ground-truth labels.

FaceXFormer: single unified transformer (Swin-B encoder + FaceX decoder),
gender = task token 4 of 6 released tasks.  Class mapping (verified against
KDEF actor-provenance labels): 0 = Male, 1 = Female.

Preprocessing (Option A — fair comparison with SwinFace):
  MTCNN detect-and-crop is BYPASSED.  Aligned images (already centered/cropped
  by the dataset pipeline) are fed directly:
  PIL open -> Resize(224,224, BICUBIC) -> ToTensor -> ImageNet normalize.
  This matches the SwinFace attribute-protocoll on the SAME images.

Outputs (same layout as SwinFace's gender_results.csv so the two are directly
comparable, just suffixed):
  - `gender_results_facexformer.csv` — aligned accuracy, de-ID accuracy, preservation
  - `predictions_per_image_facexformer.csv` — per-image prediction cache

Ground-truth labels and image discovery are REUSED from evaluate_gender_expression.py
(same loaders, same filters — e.g. KDEF Scream/Contempt images are excluded in both
scripts, so N_Matched sets are identical and results are like-for-like).

Usage:
    conda run -n facexformer python evaluate_gender_facexformer.py ^
        --dataset kdef ^
        --output D:\\dev\\deid-toolkit\\root_dir\\predictions\\kdef

    (dataset: rafd | mug-still | kdef | celeba-test)
"""

import argparse
import csv
import os
import sys
from pathlib import Path

# Fix OpenMP duplicate lib conflict on Windows (torch + cv2 each load their own libiomp5md.dll)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
from PIL import Image
import torchvision

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Repo-local weight cache (swin_b ImageNet base + any torch.hub checkpoints)
# so no weights are kept in the user home / C: cache.
os.environ.setdefault("TORCH_HOME", os.path.join(SCRIPT_DIR, "models", "facexformer"))
sys.path.insert(0, SCRIPT_DIR)

from evaluate_gender_expression import (  # noqa: E402  (reuse GT + discovery protocol)
    DATASET_LOADERS,
    find_image_path,
    get_image_stems,
)


def discover_technique_dirs(technique_names, dataset_name):
    """Same logic as evaluate_gender_expression.discover_technique_dirs, but with the
    datasets base resolved relative to THIS repo (script lives in deid-toolkit/sota_evaluators;
    the imported version assumes the script sits in a dir whose parent contains deid-toolkit,
    which is only true for the old flat deid-toolkit_evaluators/ layout)."""
    datasets_base = Path(SCRIPT_DIR).parent / "root_dir" / "datasets"
    results = []
    reserved = ("aligned", "labels", "original", "pairs", "deidentified")

    for tech in (technique_names or sorted(
            set(p.name for p in datasets_base.iterdir()
                if p.is_dir() and p.name not in reserved))):
        tech_dir = datasets_base / tech / dataset_name
        if tech_dir.is_dir():
            images = list(tech_dir.glob("*.jpg")) + list(tech_dir.glob("*.png"))
            if images:
                results.append((tech, str(tech_dir)))

    results.sort(key=lambda x: x[0])
    return results

# ── FaceXFormer imports ─────────────────────────────────────────────────────

FACEXFORMER_DIR = os.path.join(SCRIPT_DIR, "FaceXFormer")
sys.path.insert(0, FACEXFORMER_DIR)

from network import FaceXFormer  # noqa: E402

# Gender task token (see FaceXFormer/inference.py docstring: task 4 = age/gender/race)
GENDER_TASK_ID = 4
GENDER_LABELS = {0: "Male", 1: "Female"}  # VERIFIED against KDEF provenance labels

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ── Model & preprocessing ──────────────────────────────────────────────────

def build_facexformer_model(weight_path, device):
    """Build FaceXFormer and load the unified checkpoint (state_dict_backbone)."""
    model = FaceXFormer().to(device)
    checkpoint = torch.load(weight_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict_backbone"])
    model.eval()
    return model


TRANSFORM = torchvision.transforms.Compose([
    torchvision.transforms.Resize(
        size=(224, 224),
        interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
    ),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def preprocess_image(image_path):
    """PIL open -> 224x224 bicubic resize -> tensor (MTCNN bypassed; Option A)."""
    image = Image.open(image_path).convert("RGB")
    return TRANSFORM(image)


def _forward_gender(model, batch, device):
    """Run task-4 (age/gender/race) forward pass on a preprocessed batch."""
    images = batch if batch.dim() == 4 else batch.unsqueeze(0)  # -> (B,3,224,224)
    images = images.to(device)
    batch_size = images.size(0)
    tasks = torch.full((batch_size,), GENDER_TASK_ID, dtype=torch.long, device=device)
    labels = {
        "segmentation": torch.zeros(batch_size, 224, 224, device=device),
        "lnm_seg": torch.zeros(batch_size, 5, 2, device=device),
        "landmark": torch.zeros(batch_size, 68, 2, device=device),
        "headpose": torch.zeros(batch_size, 3, device=device),
        "attribute": torch.zeros(batch_size, 40, device=device),
        "a_g_e": torch.zeros(batch_size, 3, device=device),
        "visibility": torch.zeros(batch_size, 29, device=device),
    }
    with torch.no_grad():
        (
            _lm, _pose, _attr, _vis,
            _age, gender_output, _race, _seg,
        ) = model(images, labels, tasks)
    return gender_output.float().argmax(dim=1).cpu().numpy()


def predict_batch(image_stems, image_dir, model, device, batch_size=32):
    """Run FaceXFormer gender inference on a directory of images.

    Returns dict: {stem: "Male"/"Female"}  (stems that fail to load are skipped).
    """
    results = {}

    paths = []
    for stem in image_stems:
        img_path = find_image_path(image_dir, stem)
        if img_path is not None:
            paths.append((stem, img_path))

    for start in range(0, len(paths), batch_size):
        batch_tensors = []
        batch_stems = []
        for stem, img_path in paths[start:start + batch_size]:
            try:
                batch_tensors.append(preprocess_image(img_path))
                batch_stems.append(stem)
            except (ValueError, OSError):
                pass

        if not batch_tensors:
            continue

        preds = _forward_gender(model, torch.stack(batch_tensors), device)
        for stem, p in zip(batch_stems, preds):
            results[stem] = GENDER_LABELS.get(int(p), "Unknown")

    return results


# ── Evaluation & reporting ─────────────────────────────────────────────────

def evaluate_technique(tech_name, aligned_results, deid_results, gt):
    """Gender-only metrics vs ground truth (mirrors SwinFace evaluate_technique)."""
    common = sorted(set(aligned_results) & set(deid_results) & set(gt))
    # Restrict to stems with a GT gender label (loaders store None when unknown)
    common = [s for s in common if gt[s].get("gender") is not None]

    aligned_correct = sum(1 for s in common if aligned_results[s] == gt[s]["gender"])
    deid_correct = sum(1 for s in common if deid_results[s] == gt[s]["gender"])
    preserved = sum(1 for s in common if aligned_results[s] == deid_results[s])

    return {
        "name": tech_name,
        "n_matched": len(common),
        "aligned_gender_acc": aligned_correct / len(common) * 100 if common else 0.0,
        "deid_gender_acc": deid_correct / len(common) * 100 if common else 0.0,
        "gender_preservation_rate": preserved / len(common) * 100 if common else 0.0,
    }


def generate_gender_csv(results, output_path):
    with open(output_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Technique", "Aligned_Gender_Accuracy", "DeID_Gender_Accuracy",
                         "Gender_Preservation_Rate", "N_Matched_GT"])
        baseline = results[0]
        writer.writerow([
            "aligned (Validation)",
            f"{baseline['aligned_gender_acc']:.1f}",
            "—", "—", baseline["n_matched"],
        ])
        for r in results[1:]:
            writer.writerow([
                r["name"],
                f"{r['aligned_gender_acc']:.1f}",
                f"{r['deid_gender_acc']:.1f}",
                f"{r['gender_preservation_rate']:.1f}",
                r["n_matched"],
            ])


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="FaceXFormer gender evaluation across DEID baselines")
    parser.add_argument("--dataset", required=True, choices=list(DATASET_LOADERS.keys()))
    parser.add_argument("--techniques", nargs="+", default=None,
                        help="Technique names to evaluate (default: all available)")
    parser.add_argument("--output", required=True, help="Output directory for results")
    parser.add_argument("--weight", type=str, default=None,
                        help="FaceXFormer checkpoint (default: models/facexformer/ckpts/model.pt)")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    script_dir = SCRIPT_DIR
    weight_path = args.weight or os.path.join(
        script_dir, "models", "facexformer", "ckpts", "model.pt")
    if not os.path.exists(weight_path):
        print(f"ERROR: Checkpoint not found: {weight_path}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_name = args.dataset

    print("=" * 70)
    print(f"FaceXFormer Gender Evaluation — {dataset_name.upper()}")
    print("=" * 70)
    print(f"Device:           {device}")
    print(f"Checkpoint:       {weight_path}")
    print(f"Preprocessing:    Option A (aligned passthrough, no MTCNN)")
    print()

    # ── Load ground truth (same loaders as SwinFace evaluation) ──────────
    loader_func, labels_filename = DATASET_LOADERS[dataset_name]
    labels_dir = Path(SCRIPT_DIR).parent / "root_dir" / "datasets" / "labels"
    labels_path = labels_dir / labels_filename

    print("Loading ground truth labels...")
    gt = loader_func(str(labels_path))
    print(f"  Loaded {len(gt)} labeled image(s) from {labels_path}")
    n_gender_gt = sum(1 for v in gt.values() if v.get("gender"))
    print(f"  With gender GT: {n_gender_gt}")
    if n_gender_gt == 0:
        print("ERROR: No gender ground truth available for this dataset.")
        sys.exit(1)

    # ── Load model ───────────────────────────────────────────────────────
    print(f"\nLoading FaceXFormer on {device}...")
    model = build_facexformer_model(weight_path, device)
    print("Model loaded.")

    # ── Discover aligned images (same dirs as SwinFace evaluation) ───────
    datasets_base = Path(SCRIPT_DIR).parent / "root_dir" / "datasets"
    aligned_dir = str(datasets_base / "aligned" / dataset_name)
    if not os.path.isdir(aligned_dir):
        print(f"ERROR: Aligned images directory not found: {aligned_dir}")
        sys.exit(1)

    eval_stems = [s for s in sorted(gt.keys()) if s in set(get_image_stems(aligned_dir))]
    if not eval_stems:
        print(f"ERROR: No matching aligned images found in {aligned_dir}")
        sys.exit(1)
    print(f"\nEvaluating on {len(eval_stems)} image(s) with both GT and images.")

    # ── Predict on aligned originals ─────────────────────────────────────
    print("\nPredicting on aligned originals...")
    aligned_results = predict_batch(eval_stems, aligned_dir, model, device, args.batch_size)
    print(f"  Aligned predictions: {len(aligned_results)}/{len(eval_stems)} images processed.")

    # ── Discover and process techniques ──────────────────────────────────
    tech_dirs = discover_technique_dirs(args.techniques, dataset_name)
    if not tech_dirs:
        print("\nERROR: No technique directories found with images for this dataset.")
        sys.exit(1)
    print(f"\nFound {len(tech_dirs)} technique(s): {', '.join(t[0] for t in tech_dirs)}\n")

    results = [evaluate_technique("aligned", aligned_results, aligned_results, gt)]
    b = results[0]
    print(f"--- Baseline: Aligned vs Ground Truth ---")
    print(f"  Gender Acc={b['aligned_gender_acc']:.1f}%, N={b['n_matched']}")
    print()

    all_preds = {"aligned": aligned_results}
    for tech_name, tech_dir in tech_dirs:
        print(f"--- {tech_name} ---")
        deid_stems = sorted(set(get_image_stems(tech_dir)) & set(eval_stems))
        if not deid_stems:
            print("  SKIP — no matching images for this dataset.")
            continue
        deid_results = predict_batch(deid_stems, tech_dir, model, device, args.batch_size)
        print(f"  De-identified predictions: {len(deid_results)}/{len(deid_stems)} images processed.")
        all_preds[tech_name] = deid_results

        r = evaluate_technique(tech_name, aligned_results, deid_results, gt)
        results.append(r)
        print(f"  Aligned Acc={r['aligned_gender_acc']:.1f}%, "
              f"DeID Acc={r['deid_gender_acc']:.1f}%, "
              f"Preservation={r['gender_preservation_rate']:.1f}%, "
              f"N={r['n_matched']}")
        print()

    # ── Write outputs ────────────────────────────────────────────────────
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    pred_path = output_dir / "predictions_per_image_facexformer.csv"
    with open(pred_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Technique", "Stem", "Gender"])
        for tech_key in sorted(all_preds):
            for stem in sorted(all_preds[tech_key]):
                writer.writerow([tech_key, stem, all_preds[tech_key][stem]])
    print(f"Per-image prediction cache: {pred_path}")

    gender_csv = output_dir / "gender_results_facexformer.csv"
    generate_gender_csv(results, str(gender_csv))
    print(f"Gender results: {gender_csv}")

    print()
    print("=" * 70)
    print(f"Evaluation complete. Results saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
