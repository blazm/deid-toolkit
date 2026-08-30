#!/usr/bin/env python3
"""
Generate face embeddings using SwinFace.

Usage:
    python generate_embeddings_swinface.py \
        --input <folder_with_face_images> \
        --output <folder_for_embeddings> \
        --weight <path_to_checkpoint>.pt

Pretrained Model:
    Download from: https://drive.google.com/drive/folders/1NjVN3Kp_Tmwt17hWCIWgHpuWzkHYaman?usp=sharing
    Place in: models/swinface/checkpoint_step_79999_gpu_0.pt

Conda Environment:
    conda env create -f environment_swinface.yml
    conda activate swinface

Example:
    python generate_embeddings_swinface.py \
        --input ./face_images/ \
        --output ./swinface_embeddings/ \
        --weight models/swinface/checkpoint_step_79999_gpu_0.pt
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

# Add SwinFace project directory to path for imports
SWINFACE_PROJECT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "SwinFace", "swinface_project")
sys.path.insert(0, SWINFACE_PROJECT_DIR)

from backbones import get_model


def preprocess_image(img_path, target_size=(112, 112)):
    """Load and preprocess a single image for SwinFace.

    Preprocessing pipeline:
    1. Load image (expects ~256x256 facial images)
    2. Center crop to 224x224 (focuses on face, trims edge content)
    3. Resize to 112x112 (model input size)
    4. Normalize: pixel / 255 -> (x - 0.5) / 0.5  (map [0, 255] to [-1, 1])

    Returns:
        torch.Tensor of shape (1, 3, 112, 112) on CPU
    """
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Failed to read image: {img_path}")

    h, w = img.shape[:2]

    # If image is larger than 256x256, scale it down proportionally first
    max_dim = max(h, w)
    if max_dim > 256:
        scale = 256.0 / max_dim
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]

    # Center crop to 224x224 from input (works well for 256x256 images)
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
def extract_embeddings(image_paths, model, device, batch_size=32):
    """Extract embeddings for a list of image paths in batches.

    Returns:
        list of numpy arrays (each 512-d), aligned with image_paths order.
        None is returned for images that failed to load.
    """
    all_embeddings = []
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

        # Stack valid tensors
        valid_tensors = [t for t in batch_tensors if t is not None]
        if not valid_tensors:
            all_embeddings.extend([None] * len(batch_tensors))
            continue

        batch = torch.stack(valid_tensors).to(device)
        outputs = model(batch)

        # SwinFace ModelBox returns a dict; the 'Recognition' key holds the embedding
        # If running backbone-only, outputs is a tuple (local, global, embedding)
        if isinstance(outputs, dict):
            emb = outputs["Recognition"]
        else:
            emb = outputs[2]  # (local_features, global_features, x)

        # L2 normalize embeddings
        emb = emb.float()
        norm = torch.norm(emb, p=2, dim=1, keepdim=True)
        emb = emb / norm.clamp(min=1e-5)

        # Distribute back to batch positions (preserve None for invalid images)
        emb_list = emb.cpu().numpy().tolist()
        batch_embeddings = []
        emb_idx = 0
        for valid in batch_valid:
            if valid:
                batch_embeddings.append(np.array(emb_list[emb_idx], dtype=np.float32))
                emb_idx += 1
            else:
                batch_embeddings.append(None)

        all_embeddings.extend(batch_embeddings)
        print(f"\rProgress: {end}/{total} images processed", end="", flush=True)

    print()  # newline after progress
    return all_embeddings


def get_image_paths(input_dir, extensions=(".jpg", ".jpeg", ".png", ".bmp", ".webp")):
    """Recursively find all image files in a directory, sorted by filename."""
    input_path = Path(input_dir)
    if not input_path.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {input_dir}")

    paths = []
    for ext in extensions:
        paths.extend(input_path.rglob(f"*{ext}"))
        paths.extend(input_path.rglob(f"*{ext.upper()}"))

    # Sort by filename for deterministic order
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
    backbone = get_model(cfg.network, num_features=cfg.embedding_size)

    fam = subnets_mod.FeatureAttentionModule(
        in_chans=cfg.fam_in_chans, kernel_size=cfg.fam_kernel_size,
        conv_shared=cfg.fam_conv_shared, conv_mode=cfg.fam_conv_mode,
        channel_attention=cfg.fam_channel_attention, spatial_attention=cfg.fam_spatial_attention,
        pooling=cfg.fam_pooling, la_num_list=cfg.fam_la_num_list)
    tss = subnets_mod.TaskSpecificSubnets()
    om = subnets_mod.OutputModule()

    # Load checkpoint
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
        description="Generate face embeddings using SwinFace"
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Path to folder containing facial images (scans recursively)"
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Path to output folder for embeddings (.npy files)"
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
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (default: auto-detect)"
    )

    args = parser.parse_args()

    # Resolve paths — preserve last folder levels from input in output
    # e.g. input=datasets/aligned/celeba-test, output=embeddings/SwinFace
    #      → actual output = embeddings/SwinFace/aligned/celeba-test/
    input_dir = os.path.abspath(args.input)
    output_root = os.path.abspath(args.output)
    weight_path = os.path.abspath(args.weight)

    # Take last 2 components of input path (e.g. "aligned/celeba-test")
    input_parts = Path(input_dir).parts
    if len(input_parts) >= 2:
        sub_folder = str(Path(input_parts[-2], input_parts[-1]))
    elif len(input_parts) == 1:
        sub_folder = input_parts[0]
    else:
        sub_folder = "."

    output_dir = os.path.join(output_root, sub_folder)

    print(f"SwinFace Embedding Generator")
    print(f"=============================")
    print(f"Input directory:  {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Model checkpoint: {weight_path}")
    print()

    # Validate inputs
    if not os.path.exists(weight_path):
        print(f"ERROR: Checkpoint not found: {weight_path}")
        print()
        print("Download the pretrained model from:")
        print("  https://drive.google.com/drive/folders/1NjVN3Kp_Tmwt17hWCIWgHpuWzkHYaman?usp=sharing")
        print()
        print(f"Place it here: {weight_path}")
        sys.exit(1)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

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

    # Extract embeddings
    image_path_list = list(image_paths)
    print("Extracting embeddings...")
    embeddings = extract_embeddings(image_path_list, model, device, args.batch_size)

    # Save embeddings
    saved_count = 0
    failed_count = 0
    for img_path, emb in zip(image_path_list, embeddings):
        if emb is not None:
            # Use relative path from input dir as filename to preserve structure
            rel_path = os.path.relpath(str(img_path), input_dir)
            out_name = os.path.splitext(rel_path)[0] + ".npy"
            out_path = os.path.join(output_dir, out_name)

            # Create subdirectory if needed (preserve folder structure)
            out_subdir = os.path.dirname(out_path)
            if out_subdir:
                os.makedirs(out_subdir, exist_ok=True)

            np.save(out_path, emb)
            saved_count += 1
        else:
            failed_count += 1
            print(f"  WARNING: Skipped (failed to load): {img_path}")

    print()
    print(f"Done! Saved {saved_count} embedding(s).")
    if failed_count > 0:
        print(f"Failed: {failed_count} image(s) could not be processed.")
    print(f"Embedding dimension: 512")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
