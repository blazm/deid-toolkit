#!/usr/bin/env python3
"""
Extract facial attributes (age, expression, gender) using SwinFace.

Usage:
    python extract_attributes_swinface.py \
        --input <folder_with_face_images> \
        --output <csv_output_file_or_folder> \
        --weight <path_to_checkpoint>.pt

Pretrained Model:
    Same checkpoint as the embedding script:
    models/swinface/checkpoint_step_79999_gpu_0.pt

Conda Environment:
    conda activate swinface

Output:
    A CSV file with columns: filename, age, gender, gender_label, expression, expression_label

SwinFace predicts:
  - Age: continuous value (regression)
  - Gender: binary classification (0 = Female, 1 = Male)
  - Expression: 7-class classification (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral)
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

# Add SwinFace project directory to path for imports
SWINFACE_PROJECT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "SwinFace", "swinface_project")
sys.path.insert(0, SWINFACE_PROJECT_DIR)


EXPRESSION_LABELS = [
    "Angry",    # 0
    "Disgust",  # 1
    "Fear",     # 2
    "Happy",    # 3
    "Sad",      # 4
    "Surprise", # 5
    "Neutral",  # 6
]


def preprocess_image(img_path, target_size=(112, 112)):
    """Load and preprocess a single image for SwinFace.

    Preprocessing pipeline:
    1. Load image (expects ~256x256 facial images)
    2. If larger than 256px in any dimension, scale down proportionally
    3. Center crop to 224x224
    4. Resize to 112x112
    5. Normalize: map [0, 255] to [-1, 1]

    Returns:
        torch.Tensor of shape (1, 3, 112, 112) on CPU
    """
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Failed to read image: {img_path}")

    h, w = img.shape[:2]

    # If image is larger than 256x256, scale down proportionally first
    max_dim = max(h, w)
    if max_dim > 256:
        scale = 256.0 / max_dim
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]

    # Center crop to 224x224 from input
    crop_size = min(224, min(h, w))
    y_start = max(0, (h - crop_size) // 2)
    x_start = max(0, (w - crop_size) // 2)
    img = img[y_start:y_start + crop_size, x_start:x_start + crop_size]

    # Resize to model's expected input size (112x112)
    img = cv2.resize(img, target_size)

    # BGR -> RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # HWC -> CHW
    img = np.transpose(img, (2, 0, 1))
    # To tensor and normalize to [-1, 1] — do NOT add batch dim here (torch.stack adds it)
    img = torch.from_numpy(img).float()
    img.div_(255).sub_(0.5).div_(0.5)
    return img  # shape: (3, 112, 112)


@torch.no_grad()
def extract_attributes(image_paths, model, device, batch_size=32):
    """Extract age, gender, expression predictions for a list of images.

    Returns:
        list of dicts with keys: filename, age, gender, gender_label,
                                  expression, expression_label
        None is returned for images that failed to load.
    """
    all_results = []
    total = len(image_paths)

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_tensors = []
        batch_valid = []

        for i in range(start, end):
            try:
                tensor = preprocess_image(image_paths[i])
                batch_tensors.append(tensor)
                batch_valid.append(True)
            except ValueError:
                batch_valid.append(False)
                batch_tensors.append(None)

        valid_tensors = [t for t in batch_tensors if t is not None]
        if not valid_tensors:
            all_results.extend([None] * len(batch_tensors))
            continue

        batch = torch.stack(valid_tensors).to(device)
        outputs = model(batch)  # dict of outputs

        # Extract attributes from the output dict
        # Gender: binary logits, shape (batch, 2) -> argmax gives 0=Female, 1=Male
        gender_logits = outputs["Gender"].float()
        gender_preds = gender_logits.argmax(dim=1).cpu().numpy()

        # Age: regression output, shape (batch, 1) -> squeeze to get scalar age
        age_vals = outputs["Age"].float().cpu().numpy().flatten()

        # Expression: 7-class logits, shape (batch, 7) -> argmax for predicted class
        expression_logits = outputs["Expression"].float()
        expression_preds = expression_logits.argmax(dim=1).cpu().numpy()

        # Build results mapped back to batch positions
        result_idx = 0
        for valid in batch_valid:
            if valid:
                result = {
                    "age": round(float(age_vals[result_idx]), 1),
                    "gender": int(gender_preds[result_idx]),
                    "gender_label": "Male" if gender_preds[result_idx] == 1 else "Female",
                    "expression": int(expression_preds[result_idx]),
                    "expression_label": EXPRESSION_LABELS[expression_preds[result_idx]],
                }
                all_results.append(result)
                result_idx += 1
            else:
                all_results.append(None)

        print(f"\rProgress: {end}/{total} images processed", end="", flush=True)

    print()
    return all_results


def get_image_paths(input_dir, extensions=(".jpg", ".jpeg", ".png", ".bmp", ".webp")):
    """Recursively find all image files in a directory, sorted by filename."""
    input_path = Path(input_dir)
    if not input_path.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {input_dir}")

    paths = []
    for ext in extensions:
        paths.extend(input_path.rglob(f"*{ext}"))
        paths.extend(input_path.rglob(f"*{ext.upper()}"))

    paths = sorted(set(paths))
    return paths


def build_swinface_model(weight_path, device):
    """Build and load the full SwinFace model from checkpoint.

    Loads analysis submodules directly via importlib to avoid
    analysis/__init__.py which pulls in mxnet (not needed for inference).
    """
    import torch.nn as nn
    import importlib.util

    # Create a fake "analysis" namespace package so relative imports work
    if "analysis" not in sys.modules:
        class _FakeAnalysisPkg:
            __path__ = [SWINFACE_PROJECT_DIR + "/analysis"]
            __package__ = "analysis"
        sys.modules["analysis"] = _FakeAnalysisPkg()

    # Load cbam.py as analysis.cbam
    if "analysis.cbam" not in sys.modules:
        cbam_path = os.path.join(SWINFACE_PROJECT_DIR, "analysis", "cbam.py")
        spec_cbam = importlib.util.spec_from_file_location("analysis.cbam", cbam_path)
        cbam_mod = importlib.util.module_from_spec(spec_cbam)
        sys.modules["analysis.cbam"] = cbam_mod
        spec_cbam.loader.exec_module(cbam_mod)

    # Load subnets.py as analysis.subnets (imports from .cbam now resolves)
    if "analysis.subnets" not in sys.modules:
        subnets_path = os.path.join(SWINFACE_PROJECT_DIR, "analysis", "subnets.py")
        spec_subnets = importlib.util.spec_from_file_location("analysis.subnets", subnets_path)
        subnets_mod = importlib.util.module_from_spec(spec_subnets)
        sys.modules["analysis.subnets"] = subnets_mod
        spec_subnets.loader.exec_module(subnets_mod)

    class SwinFaceCfg:
        network = "swin_t"
        fam_kernel_size = 3
        fam_in_chans = 2112
        fam_conv_shared = False
        fam_conv_mode = "split"
        fam_channel_attention = "CBAM"
        fam_spatial_attention = None
        fam_pooling = "max"
        fam_la_num_list = [2 for _ in range(11)]
        fam_feature = "all"
        embedding_size = 512

    cfg = SwinFaceCfg()
    from backbones import get_model
    backbone = get_model(cfg.network, num_features=cfg.embedding_size)

    fam = subnets_mod.FeatureAttentionModule(
        in_chans=cfg.fam_in_chans, kernel_size=cfg.fam_kernel_size,
        conv_shared=cfg.fam_conv_shared, conv_mode=cfg.fam_conv_mode,
        channel_attention=cfg.fam_channel_attention, spatial_attention=cfg.fam_spatial_attention,
        pooling=cfg.fam_pooling, la_num_list=cfg.fam_la_num_list)
    tss = subnets_mod.TaskSpecificSubnets()
    om = subnets_mod.OutputModule()

    checkpoint = torch.load(weight_path, map_location=device)
    backbone.load_state_dict(checkpoint["state_dict_backbone"])
    fam.load_state_dict(checkpoint["state_dict_fam"])
    tss.load_state_dict(checkpoint["state_dict_tss"])
    om.load_state_dict(checkpoint["state_dict_om"])

    model = subnets_mod.ModelBox(
        backbone=backbone, fam=fam, tss=tss, om=om, feature=cfg.fam_feature)

    model = model.to(device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Extract facial attributes (age, expression, gender) using SwinFace"
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Path to folder containing facial images (scans recursively)"
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output CSV file path (e.g., attributes.csv) or output folder"
    )
    parser.add_argument(
        "--weight", type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "models", "swinface", "checkpoint_step_79999_gpu_0.pt"),
        help="Path to SwinFace checkpoint (.pt file)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Batch size for inference (default: 32)"
    )
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (default: auto-detect)"
    )

    args = parser.parse_args()

    # Resolve paths
    input_dir = os.path.abspath(args.input)
    weight_path = os.path.abspath(args.weight)

    # Determine output CSV path
    if args.output.endswith(".csv"):
        csv_path = os.path.abspath(args.output)
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    else:
        out_dir = os.path.abspath(args.output)
        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(out_dir, "attributes.csv")

    print(f"SwinFace Attribute Extractor")
    print(f"=============================")
    print(f"Input directory: {input_dir}")
    print(f"Output CSV:      {csv_path}")
    print(f"Model checkpoint:{weight_path}")
    print()

    # Validate inputs
    if not os.path.exists(weight_path):
        print(f"ERROR: Checkpoint not found: {weight_path}")
        print("Download from:")
        print("  https://drive.google.com/drive/folders/1NjVN3Kp_Tmwt17hWCIWgHpuWzkHYaman?usp=sharing")
        sys.exit(1)

    # Get image paths
    image_paths = get_image_paths(input_dir)
    if not image_paths:
        print(f"ERROR: No images found in {input_dir}")
        sys.exit(1)

    print(f"Found {len(image_paths)} image(s)")
    print()

    # Build model
    device = torch.device(args.device)
    print(f"Loading model on {device}...")
    model = build_swinface_model(weight_path, device)
    print("Model loaded successfully.")
    print()

    # Extract attributes
    image_path_list = list(image_paths)
    print("Extracting attributes...")
    results = extract_attributes(image_path_list, model, device, args.batch_size)

    # Write CSV
    fieldnames = ["filename", "age", "gender", "gender_label", "expression", "expression_label"]
    saved_count = 0
    failed_count = 0

    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for img_path, result in zip(image_path_list, results):
            if result is not None:
                rel_name = os.path.relpath(str(img_path), input_dir)
                writer.writerow({"filename": rel_name, **result})
                saved_count += 1
            else:
                failed_count += 1

    print()
    print(f"Done! Attributes for {saved_count} image(s) saved to: {csv_path}")
    if failed_count > 0:
        print(f"Failed: {failed_count} image(s) could not be processed.")
    print()
    print("Columns:")
    print("  filename         - relative path from input folder")
    print("  age              - predicted age (continuous value)")
    print("  gender           - 0 = Female, 1 = Male")
    print("  gender_label     - Female / Male")
    print("  expression       - class index (0-6)")
    print(f"  expression_label - {', '.join(EXPRESSION_LABELS)}")


if __name__ == "__main__":
    main()
