#!/usr/bin/env python
"""Batch de-identification using CLEANIR.

Usage:
    python deidentify_batch.py --input <source_folder> --output <dest_folder> [options]

Options:
    --model_path   Path to model directory containing encoder.h5, decoder.h5, dn.h5.
                   Default: ./model/
    --degree       Degrees of identity manipulation (0–90).
                   0 = same face, 90 = maximum change. Default: 45
    --dsize        Cropped face size (width, height). Default: (64, 64)
    --paste-back   Paste de-identified face back onto original image.
                   If not set, outputs cropped anonymized faces only (64×64).
    --resize       Resize all images to this size before processing (default: 256x256).
                   Speeds up face detection significantly. Output is always this size.

The script detects faces using the face_recognition library (CNN model),
crops each face to dsize, encodes it into latent vectors separating identity
from attributes, rotates the identity vector by the given degrees, and
decodes back to an image.

When --paste-back is used: images are first resized to --resize dimensions,
the 64×64 anonymized face is pasted back onto the resized canvas with
feathered blending. Output is always --resize dimensions (default 256×256).

If an image contains multiple faces, the largest one is processed.
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import cv2

# Silence TF warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
tf.get_logger().setLevel("ERROR")

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)

from cleanir.cleanir import Cleanir
from cleanir.tools.crop_face import crop_face_from_image


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


def get_image_files(input_dir: str) -> list[str]:
    """Return sorted list of image file paths in input_dir."""
    files = []
    for ext in SUPPORTED_EXTENSIONS:
        files.extend(Path(input_dir).glob(f"*{ext}"))
        files.extend(Path(input_dir).glob(f"*{ext.upper()}"))
    return sorted(set(files))


def init_model(model_path: str, dsize=(64, 64)):
    """Load the CLEANIR models (encoder, decoder, disentangle network)."""
    cleanir = Cleanir(dsize=dsize)

    if not os.path.isdir(model_path):
        raise FileNotFoundError(
            f"Model directory not found: {model_path}\n"
            "Run python model_download.py first to download pretrained models."
        )

    print(f"Loading CLEANIR models from: {model_path} ...")
    success = cleanir.load_models(model_path)
    if not success:
        raise RuntimeError("Failed to load one or more CLEANIR models.")
    print("CLEANIR models loaded.")
    return cleanir


def paste_face_back(original_img: np.ndarray, cropped_face: np.ndarray,
                    box: tuple, feather_px: int = 10) -> np.ndarray:
    """Paste a de-identified face crop back onto the original image.

    Parameters
    ----------
    original_img : np.ndarray [H, W, 3] RGB
        Original full-resolution image.
    cropped_face : np.ndarray [h, w, 3] RGB
        De-identified face (may be different size from original crop).
    box : tuple (top, right, bottom, left) in original image coords
        Bounding box of the detected face.
    feather_px : int
        Number of pixels for smooth blending at the edge.

    Returns
    -------
    np.ndarray [H, W, 3] RGB with face pasted back.
    """
    top, right, bottom, left = box
    target_h, target_w = bottom - top, right - left

    # Resize anonymized face to match original bounding box
    resized_face = cv2.resize(cropped_face, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

    # Create a copy of the original
    result = original_img.copy()

    # Build a feathered mask for smooth blending
    mask = np.ones((target_h, target_w), dtype=np.float32)
    for c in range(feather_px):
        alpha = (c + 1) / (feather_px + 1)
        if c < target_h // 2 and c < target_w // 2:
            mask[c:target_h - c, c:target_w - c] = min(
                mask[c:target_h - c, c:target_w - c].max(), 1.0
            )

    # Normalize mask to [0, 1] with smooth edge
    mask = np.minimum(mask, 1.0)

    # Expand dims for RGB channels
    mask = np.stack([mask] * 3, axis=-1)

    # Blend
    region = result[top:bottom, left:right]
    blended = (resized_face * mask + region * (1 - mask)).astype(np.uint8)

    result[top:bottom, left:right] = blended
    return result


def detect_largest_face_box(img_rgb: np.ndarray):
    """Detect the largest face and return bounding box.

    Returns
    -------
    (box, cropped_face) or (None, None)
        box : tuple (top, right, bottom, left) in original image coords
        cropped_face : np.ndarray — the 64×64 crop from CLEANIR
    """
    import face_recognition

    faces = face_recognition.face_locations(img_rgb, model="cnn")

    if len(faces) > 0:
        t, r, b, l = max(faces,
                        key=lambda face: (face[2] - face[0]) * (face[1] - face[3]))
        return (t, r, b, l)
    else:
        return None


def deidentify_image(pil_img: Image.Image, cleanir: Cleanir, degree: int,
                     dsize=(64, 64), paste_back: bool = False):
    """De-identify a single image.

    Returns
    -------
    (result_pil, did_find_face)
        result_pil : PIL.Image  — either full-image with pasted face, or cropped face only
        did_find_face : bool   — whether a face was detected
    """
    # Convert PIL → RGB numpy array
    img_np = np.array(pil_img.convert("RGB"))

    if paste_back:
        # Detect face to get bounding box for pasting back
        box = detect_largest_face_box(img_np)
        if box is None:
            return pil_img, False

        top, right, bottom, left = box

        # Crop largest face to dsize using CLEANIR's crop function
        cropped = crop_face_from_image(img_np, dsize)

        if cropped is None or cropped.size == 0:
            return pil_img, False

        # Run CLEANIR deidentification
        anon_np = cleanir.deidentify(cropped, degree)

        if anon_np is None or anon_np.size == 0:
            return pil_img, False

        # Ensure uint8 range
        anon_np = np.clip(anon_np, 0, 255).astype(np.uint8)

        # Paste back onto original
        result_rgb = paste_face_back(img_np, anon_np, box, feather_px=15)
        result_pil = Image.fromarray(result_rgb, mode="RGB")
        return result_pil, True

    else:
        # Crop-only mode (original behavior)
        cropped = crop_face_from_image(img_np, dsize)

        if cropped is None or cropped.size == 0:
            return None, False

        # Run CLEANIR deidentification
        anon_np = cleanir.deidentify(cropped, degree)

        if anon_np is None or anon_np.size == 0:
            return None, False

        # Ensure uint8 range
        anon_np = np.clip(anon_np, 0, 255).astype(np.uint8)

        # Convert to PIL Image
        pil_anon = Image.fromarray(anon_np, mode="RGB")
        return pil_anon, True


def run(args):
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_files = get_image_files(str(input_dir))
    if not image_files:
        print(f"No supported images found in {input_dir}")
        return

    # Parse dsize tuple
    dsize = tuple(int(x) for x in args.dsize.split(","))
    if len(dsize) != 2:
        raise ValueError("--dsize must be two comma-separated integers, e.g. 64,64")

    # Parse resize tuple
    resize_wh = tuple(int(x) for x in args.resize.split(","))
    if len(resize_wh) != 2:
        raise ValueError("--resize must be two comma-separated integers, e.g. 256,256")

    cleanir = init_model(args.model_path, dsize=dsize)

    processed = 0
    skipped = 0

    for img_path in tqdm(image_files, desc="De-identifying"):
        try:
            pil_img = Image.open(img_path).convert("RGB")

            # Resize to target size for faster processing
            if resize_wh != (pil_img.width, pil_img.height):
                pil_img = pil_img.resize(resize_wh, Image.LANCZOS)

            result_pil, found_face = deidentify_image(
                pil_img, cleanir, args.degree, dsize, args.paste_back
            )

            if not found_face:
                # No faces detected — save copy of original (resized)
                print(f"\n⚠ No face detected in {img_path.name}, saving resized original")
                skipped += 1

            stem = img_path.stem
            out_path = output_dir / f"{stem}.png"
            result_pil.save(out_path, "PNG")
            processed += 1

        except Exception as e:
            print(f"\n✗ Failed on {img_path.name}: {e}")

    print(f"\nDone. Processed: {processed}, Skipped (no face): {skipped}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch face de-identification using CLEANIR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="Input folder with images")
    parser.add_argument("--output", required=True, help="Output folder for anonymized images")
    default_model = os.path.join(REPO_ROOT, "model")
    parser.add_argument("--model_path", default=default_model,
                        help=f"Path to model directory (encoder.h5, decoder.h5, dn.h5). Default: {default_model}")
    parser.add_argument("--degree", type=int, default=45,
                        help="Degrees of identity manipulation (0–90). 0 = same face, 90 = maximum change. Default: 45")
    parser.add_argument("--dsize", default="64,64",
                        help="Cropped face size (width,height). Default: 64,64")
    parser.add_argument("--paste-back", action="store_true",
                        help="Paste de-identified face back onto original image (preserves context)")
    parser.add_argument("--resize", default="256,256",
                        help="Resize input to W,H before processing for speed. Output is this size. Default: 256,256")

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
