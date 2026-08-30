"""Batch de-identification of images using DeepPrivacy (fully local models).

Usage:
    python deidentify_batch.py --input <input_dir> --output <output_dir>
        [--truncation 0.5] [--confidence 0.3]
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

from deep_privacy.build import build_anonymizer


def get_image_paths(input_dir):
    """Recursively find all images in input_dir."""
    image_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    paths = []
    for root, _dirs, files in os.walk(input_dir):
        for f in sorted(files):
            if Path(f).suffix.lower() in image_suffixes:
                paths.append(Path(root) / f)
    return paths


def main():
    parser = argparse.ArgumentParser(
        description="Batch anonymize images with DeepPrivacy"
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
        "--truncation", type=float, default=0.5,
        help="GAN truncation level (0=deterministic, 5=max diversity). Default=0.5"
    )
    parser.add_argument(
        "--confidence", type=float, default=0.3,
        help="Face detection confidence threshold. Default=0.3"
    )
    parser.add_argument(
        "--batch-size", type=int, default=1,
        help="Inference batch size. Default=1"
    )
    parser.add_argument(
        "--resize", type=int, default=None,
        help="Resize input images to NxN before processing (e.g. 256 for speed). Output matches this size. Default=keep original size"
    )
    args = parser.parse_args()

    input_dir = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()

    if not input_dir.is_dir():
        print(f"ERROR: Input directory does not exist: {input_dir}")
        sys.exit(1)

    # Build anonymizer from local config + weights
    # NOTE: save_debug=False — only produces the anonymized image,
    # NOT the debug detection visualization that the original CLI writes.
    # Use relative path so Config.output_dir resolves correctly (outputs/fdf/512_dsfd).
    REPO_ABS = REPO_ROOT.resolve()
    CWD = Path.cwd().resolve()
    try:
        config_path = str((REPO_ABS / "configs" / "fdf" / "512_dsfd.py").relative_to(CWD))
    except ValueError:
        # Fallback: change to repo dir so relative paths work regardless of caller CWD.
        os.chdir(str(REPO_ABS))
        config_path = "configs/fdf/512_dsfd.py"
    print(f"Loading DeepPrivacy model (config={config_path}) ...")
    anon, cfg = build_anonymizer(
        config_path=config_path,
        truncation_level=args.truncation,
        batch_size=args.batch_size,
        detection_threshold=args.confidence,
        return_cfg=True,
    )
    print(f"Model loaded. Truncation={args.truncation}")

    # Collect images
    image_paths = get_image_paths(input_dir)
    if not image_paths:
        print(f"No images found in {input_dir}")
        sys.exit(0)

    print(f"\nProcessing {len(image_paths)} images from {input_dir}")
    print(f"Output: {output_dir}\n")

    for idx, img_path in enumerate(image_paths, 1):
        # Compute relative path to mirror directory structure
        rel = img_path.relative_to(input_dir)
        out_path = output_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Read image (OpenCV BGR)
        im_bgr = cv2.imread(str(img_path))
        if im_bgr is None:
            print(f"  [{idx}/{len(image_paths)}] SKIP (unreadable): {img_path.name}")
            continue

        # Optional resize for faster processing
        if args.resize is not None:
            im_bgr = cv2.resize(im_bgr, (args.resize, args.resize),
                                interpolation=cv2.INTER_LANCZOS4)

        # DeepPrivacy expects RGB
        im_rgb = im_bgr[:, :, ::-1].copy()

        # Anonymize — returns [0,1] float32 RGB
        anon_rgb = anon.anonymize(im_rgb)

        # Convert to BGR for OpenCV (keep as float32, cv2.imwrite handles it)
        anon_bgr = anon_rgb[:, :, ::-1].copy()
        cv2.imwrite(str(out_path), anon_bgr)

        print(f"  [{idx}/{len(image_paths)}] OK: {img_path.name} -> {out_path.name}")

    print(f"\nDone. Processed {len(image_paths)} images.")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
