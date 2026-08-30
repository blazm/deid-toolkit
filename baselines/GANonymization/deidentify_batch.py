#!/usr/bin/env python
"""Batch de-identification using GANonymization.

Usage:
    python deidentify_batch.py --input <source_folder> --output <dest_folder> [options]

Options:
    --model_file   Path to checkpoint (.ckpt). Default: models/GANonymization_25.ckpt
    --img_size     Processing image size for GAN. Default: 512
    --align        Align faces based on orientation. Default: True
    --device       GPU device ID or -1 for CPU. Default: -1 (CPU)
    --paste-back   Paste de-identified face back onto original image.
                   If not set, outputs cropped anonymized faces only.
    --resize       Resize all images to this size before processing (default: 256x256).
                   Speeds up face detection significantly. Output is always this size.

The script detects faces using RetinaFace, creates facial landmark
visualizations, and generates new face images with the same expression/pose
but different identity via a pix2pix GAN.

When --paste-back is used: images are first resized to --resize dimensions,
the anonymized face is pasted back onto the resized canvas with feathered blending.
Output is always --resize dimensions (default 256×256).
"""

import os
import sys
import argparse
import numpy as np
import cv2
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from torchvision.transforms import transforms

# Add repo root to path so we can import GANonymization modules
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)

from lib.transform.face_crop_transformer import FaceCrop
from lib.transform.face_segmentation_transformer import FaceSegmentation
from lib.transform.facial_landmarks_478_transformer import FacialLandmarks478
from lib.transform.pix2pix_transformer import Pix2PixTransformer


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


def get_image_files(input_dir: str) -> list[str]:
    """Return sorted list of image file paths in input_dir."""
    files = []
    for ext in SUPPORTED_EXTENSIONS:
        files.extend(Path(input_dir).glob(f"*{ext}"))
        files.extend(Path(input_dir).glob(f"*{ext.upper()}"))
    return sorted(set(files))


def init_model(model_file: str, img_size: int, device_id: int) -> Pix2PixTransformer:
    """Load the GANonymization pix2pix model."""
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model not found: {model_file}")

    print(f"Loading model from: {model_file} ...")
    transformer = Pix2PixTransformer(model_file, img_size, device_id)
    print("Model loaded.")
    return transformer


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
        mask[c:target_h - c, c:target_w - c] = min(mask[c:target_h - c, c:target_w - c].max(), 1.0)

    # Normalize mask to [0, 1] with smooth edge
    mask = np.minimum(mask, 1.0)

    # Expand dims for RGB channels
    mask = np.stack([mask] * 3, axis=-1)

    # Blend
    region = result[top:bottom, left:right]
    blended = (resized_face * mask + region * (1 - mask)).astype(np.uint8)

    result[top:bottom, left:right] = blended
    return result


def detect_largest_face_cv2(img_bgr: np.ndarray) -> tuple | None:
    """Detect the largest face in a BGR image using RetinaFace.
    Returns (top, right, bottom, left) box or None if no face found."""

    from retinaface import RetinaFace
    faces = RetinaFace.detect_faces(img_bgr, threshold=0.6)

    if not faces:
        return None

    # Find largest face by area
    best_face = max(faces.values(), key=lambda f: (f["facial_area"][2] - f["facial_area"][0]) * (f["facial_area"][3] - f["facial_area"][1]))
    bbox = best_face["facial_area"]  # [left, top, right, bottom]

    left, top, right_b, bottom_b = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    return (top, right_b, bottom_b, left)


def zero_padding_resize(pic: np.ndarray, size: int):
    """Zero-pad and center an image to a square canvas of given size.
    Returns (padded_img, h_start, w_start, crop_h, crop_w) so we can reverse the padding later."""
    from PIL import Image as PIL_Image
    h_in, w_in = pic.shape[:2]
    ratio = min(size / w_in, size / h_in)
    new_w = int(ratio * w_in)
    new_h = int(ratio * h_in)
    resized = cv2.resize(pic, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    h_start = (size - new_h) // 2
    w_start = (size - new_w) // 2
    canvas[h_start:h_start + new_h, w_start:w_start + new_w] = resized
    return canvas, h_start, w_start, new_h, new_w


def crop_back_from_padded(padded_img, h_start, w_start, crop_h, crop_w):
    """Extract the face region from a padded image, removing padding."""
    cropped = padded_img[h_start:h_start + crop_h, w_start:w_start + crop_w]
    return cropped


def deidentify_image(pil_img: Image.Image, transformer: Pix2PixTransformer,
                     img_size: int, align: bool, device_id: int, paste_back: bool):
    """De-identify a single image.

    Returns
    -------
    (result_pil, did_find_face)
        result_pil : PIL.Image  — either full-image with pasted face, or cropped face only
        did_find_face : bool   — whether a face was detected
    """
    # Convert PIL → BGR numpy (cv2 format)
    img_cv_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    img_rgb = cv2.cvtColor(img_cv_bgr, cv2.COLOR_BGR2RGB)

    # Detect face bounding box for paste-back
    if paste_back:
        box = detect_largest_face_cv2(img_cv_bgr)
        if box is None:
            return pil_img, False
        top, right_b, bottom_b, left = box

    # Run GANonymization pipeline
    face_cropper = FaceCrop(align=align)
    try:
        cropped_faces = face_cropper(img_cv_bgr)
    except Exception as e:
        print(f"\nRetinaFace error: {e}")
        return pil_img if paste_back else None, False

    if not cropped_faces:
        return pil_img if paste_back else None, False

    # Take largest face crop from RetinaFace
    largest = max(cropped_faces, key=lambda f: f.shape[0] * f.shape[1])
    orig_crop_h, orig_crop_w = largest.shape[:2]

    # Zero-pad to square img_size canvas (tracking padding for reversal)
    resized, pad_h, pad_w, face_h, face_w = zero_padding_resize(largest, img_size)

    # Generate landmarks + mask from FaceMesh
    flm_result = FacialLandmarks478()(resized)
    if isinstance(flm_result, tuple):
        landmarks_img, mask_at_full = flm_result  # (wireframe, binary_mask) at img_size
    else:
        landmarks_img = flm_result
        mask_at_full = np.full((img_size, img_size), 255, dtype=np.uint8)

    anon_tensor = transformer(landmarks_img)  # Returns tensor in [-1, 1] range, shape [1,3,img_size,img_size]

    # Convert back to RGB numpy array
    anon_np = torch.squeeze(anon_tensor).cpu()
    anon_np = (anon_np + 1) / 2  # [-1,1] → [0,1]
    anon_rgb = transforms.ToPILImage()(anon_np)
    anon_full = np.array(anon_rgb)  # shape: [img_size, img_size, 3]

    if not paste_back:
        return Image.fromarray(anon_full), True

    # --- Paste-back with proper padding reversal + mask blending ---

    # 1. Crop the GAN output back (remove padding) → face-sized image
    anon_face = crop_back_from_padded(anon_full, pad_h, pad_w, face_h, face_w)

    # 2. Resize anonymized face to match the original detected bbox size
    target_h = bottom_b - top
    target_w = right_b - left
    anon_target = cv2.resize(anon_face, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

    # 3. Create mask at full resolution, then crop/resize to match the bbox
    mask_cropped = crop_back_from_padded(mask_at_full, pad_h, pad_w, face_h, face_w)
    mask_target = cv2.resize(mask_cropped, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

    # 4. Feather the mask for smooth blending at edges
    feather_px = max(5, min(target_w, target_h) // 30)
    mask_target = cv2.GaussianBlur(mask_target, (feather_px * 2 + 1, feather_px * 2 + 1), feather_px / 2)
    mask_3ch = np.stack([mask_target.astype(np.float32) / 255.0] * 3, axis=-1)

    # 5. Blend into original image at the face bbox location
    result_rgb = img_rgb.copy()
    orig_region = result_rgb[top:bottom_b, left:right_b].astype(np.float32)
    anon_region = anon_target.astype(np.float32)
    blended = anon_region * mask_3ch + orig_region * (1.0 - mask_3ch)
    result_rgb[top:bottom_b, left:right_b] = np.clip(blended, 0, 255).astype(np.uint8)

    return Image.fromarray(result_rgb), True


def run(args):
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_files = get_image_files(str(input_dir))
    if not image_files:
        print(f"No supported images found in {input_dir}")
        return

    device_id = args.device
    if device_id == -1:
        device = "cpu"
        print("Using CPU (no GPU). Processing will be slower.")
    else:
        device = f"cuda:{device_id}"
        print(f"Using GPU device {device_id}.")

    # Parse resize tuple
    resize_wh = tuple(int(x) for x in args.resize.split(","))
    if len(resize_wh) != 2:
        raise ValueError("--resize must be two comma-separated integers, e.g. 256,256")

    transformer = init_model(args.model_file, args.img_size, device_id)

    processed = 0
    skipped = 0

    for img_path in tqdm(image_files, desc="De-identifying"):
        try:
            pil_img = Image.open(img_path).convert("RGB")

            # Resize to target size for faster processing
            if resize_wh != (pil_img.width, pil_img.height):
                pil_img = pil_img.resize(resize_wh, Image.LANCZOS)

            result_pil, found_face = deidentify_image(
                pil_img, transformer, args.img_size, args.align, device_id, args.paste_back
            )

            if not found_face:
                # No faces detected — save copy of resized original
                print(f"\n[WARN] No face detected in {img_path.name}, saving resized original")
                skipped += 1

            stem = img_path.stem
            out_path = output_dir / f"{stem}.png"
            result_pil.save(out_path, "PNG")
            processed += 1

        except Exception as e:
            print(f"\n[FAIL] Failed on {img_path.name}: {e}")

    print(f"\nDone. Processed: {processed}, Skipped (no face): {skipped}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch face de-identification using GANonymization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="Input folder with images")
    parser.add_argument("--output", required=True, help="Output folder for anonymized images")
    default_model = os.path.join(REPO_ROOT, "models", "GANonymization_25.ckpt")
    parser.add_argument("--model_file", default=default_model,
                        help=f"Path to checkpoint (.ckpt). Default: {default_model}")
    parser.add_argument("--img_size", type=int, default=512, help="Processing image size. Default: 512")
    parser.add_argument("--align", action="store_true", default=True,
                        help="Align faces based on orientation (default: True)")
    parser.add_argument("--no-align", action="store_false", dest="align",
                        help="Disable face alignment")
    parser.add_argument("--device", type=int, default=-1,
                        help="GPU device ID or -1 for CPU. Default: -1 (CPU)")
    parser.add_argument("--paste-back", action="store_true",
                        help="Paste de-identified face back onto original image (preserves context)")
    parser.add_argument("--resize", default="256,256",
                        help="Resize input to W,H before processing for speed. Output is this size. Default: 256,256")

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
