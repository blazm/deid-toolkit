"""Batch de-identification using AMT-GAN (adversarial makeup transfer).

AMT-GAN transfers makeup from reference images onto source faces,
creating adversarial examples that fool face recognition systems.

For images where dlib face detection fails, falls back to using the
full image as the face bounding box for landmark prediction. This is
useful for pre-cropped face datasets (e.g., CelebA-test aligned) where
the whole image IS the face but ROI detection may miss it.

Images larger than 256px are resized to match the model input size.

Usage:
    python deidentify_batch.py --input <input_dir> --output <output_dir>
        [--ref_index 0] [--device cuda]
"""
import argparse
import os
import sys
from pathlib import Path

# Ensure repo root is on path so local modules resolve
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
import dlib
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from backbone import Inference, PostProcess
from backbone.config import get_config
from backbone.preprocess import to_var
import faceutils as futils


# Normalize transform matching preprocess.py
TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])


def preprocess_bypass(preprocess_obj, image: Image, device):
    """Preprocess using full-image bbox for landmarks (bypass dlib detection).

    Used when dlib detects no face but the image is a known face crop.
    """
    img_size = preprocess_obj.img_size

    w, h = image.size
    face_on_image = dlib.rectangle(0, 0, w, h)

    image_cropped, face_in_crop, crop_face = futils.dlib.crop(
        image, face_on_image,
        preprocess_obj.up_ratio, preprocess_obj.down_ratio, preprocess_obj.width_ratio
    )

    np_image = np.array(image_cropped)
    mask = preprocess_obj.face_parse.parse(cv2.resize(np_image, (512, 512)))
    mask = F.interpolate(
        mask.view(1, 1, 512, 512),
        (img_size, img_size),
        mode="nearest"
    )
    mask = mask.type(torch.uint8)
    mask = to_var(mask, requires_grad=False).to(device)

    lms = futils.dlib.landmarks(image_cropped, face_in_crop) * img_size / image_cropped.width
    lms = lms.round()

    mask_aug, diff_re = preprocess_obj.process(mask, lms, device=device)

    image_tensor = TRANSFORM(image_cropped.resize((img_size, img_size), Image.LANCZOS))
    real = to_var(image_tensor.unsqueeze(0))

    return [real, mask_aug, diff_re], face_on_image, crop_face


def get_image_paths(input_dir):
    """Recursively find all images in input_dir."""
    image_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    paths = []
    for root, _dirs, files in os.walk(input_dir):
        for f in sorted(files):
            if Path(f).suffix.lower() in image_suffixes:
                paths.append(Path(root) / f)
    return paths


def get_reference_paths():
    """Load reference (makeup) images from assets/datasets/reference/."""
    ref_dir = REPO_ROOT / "assets" / "datasets" / "reference"
    if not ref_dir.is_dir():
        print(f"ERROR: Reference directory not found: {ref_dir}")
        sys.exit(1)
    refs = []
    for f in sorted(ref_dir.iterdir()):
        if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
            refs.append(f)
    return refs


def main():
    parser = argparse.ArgumentParser(
        description="Batch anonymize faces with AMT-GAN (adversarial makeup transfer)"
    )
    parser.add_argument(
        "--input", required=True,
        help="Input directory containing images to anonymize"
    )
    parser.add_argument(
        "--output", required=True,
        help="Output directory for anonymized images (mirrors input structure)"
    )
    parser.add_argument(
        "--ref_index", type=int, default=0,
        help="Index of reference makeup image to use (0-based). Default=0"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device to run on. Default=cuda"
    )
    args = parser.parse_args()

    input_dir = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()

    if not input_dir.is_dir():
        print(f"ERROR: Input directory does not exist: {input_dir}")
        sys.exit(1)

    # Load reference images
    ref_paths = get_reference_paths()
    if not ref_paths:
        print("ERROR: No reference images found!")
        sys.exit(1)
    print(f"Found {len(ref_paths)} reference images, using index {args.ref_index}: {ref_paths[args.ref_index].name}")

    # Setup config and load model
    config = get_config()
    config.merge_from_file("configs.yaml")
    config.freeze()

    print("Loading AMT-GAN generator (checkpoints/G.pth) ...")
    inference = Inference(config, args.device, "checkpoints/G.pth")
    postprocess = PostProcess(config)
    print("Model loaded.")

    # Load reference image once
    reference = Image.open(ref_paths[args.ref_index]).convert("RGB")

    # Collect images
    image_paths = get_image_paths(input_dir)
    if not image_paths:
        print(f"No images found in {input_dir}")
        sys.exit(0)

    print(f"\nProcessing {len(image_paths)} images from {input_dir}")
    print(f"Output: {output_dir}\n")

    # Preprocess reference once (cached for all source images)
    ref_input, _, _ = inference.preprocess(reference)
    if ref_input is None:
        print("ERROR: Reference image failed preprocessing (no face detected in reference)!")
        sys.exit(1)
    ref_input_cached = [r.to(args.device) for r in ref_input]

    skipped = 0
    fallback_count = 0
    for idx, img_path in enumerate(image_paths, 1):
        # Compute relative path to mirror directory structure
        rel = img_path.relative_to(input_dir)
        out_path = output_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Read image
        source = Image.open(str(img_path)).convert("RGB")

        # Try normal transfer first (dlib face detection + landmark-based processing)
        result = inference.transfer(source, reference)
        used_fallback = False

        if result is None:
            # Fallback: preprocess with full-image bbox for landmark prediction.
            # Useful for pre-aligned face datasets (RAF-DB, KDEF) where dlib may
            # fail to detect a face that is clearly present in the crop.
            src_input, _, _ = preprocess_bypass(
                inference.preprocess, source, device=args.device
            )

            if src_input is None:
                skipped += 1
                print(f"  [{idx}/{len(image_paths)}] SKIP (no face detected): {img_path.name}")
                continue

            for i in range(len(src_input)):
                src_input[i] = src_input[i].to(args.device)

            result = inference.solver.test(*src_input, *ref_input_cached)
            used_fallback = True
            fallback_count += 1

        if result is None:
            skipped += 1
            print(f"  [{idx}/{len(image_paths)}] SKIP (no face detected): {img_path.name}")
            continue

        # Save as PNG to preserve quality
        result.save(str(out_path.with_suffix(".png")))
        label = "FALLBACK" if used_fallback else "OK"
        print(f"  [{idx}/{len(image_paths)}] {label}: {img_path.name} -> {out_path.with_suffix('.png').name} ({result.size})")

    print(f"\nDone. Processed {len(image_paths) - skipped}/{len(image_paths)} images.")
    if skipped:
        print(f"Skipped {skipped} (no face detected).")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
