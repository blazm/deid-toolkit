#!/usr/bin/env python
"""Batch face de-identification using NullFace.

NullFace is a training-free method that uses diffusion inversion to anonymize faces
by projecting them into the null space of an identity encoder. It does NOT modify
the Stable Diffusion model — it only needs SD 1.5 + InsightFace for identity extraction.

Usage:
    python deidentify_batch.py --input <source_folder> --output <dest_folder> [options]

Options:
    --guidance_scale   Classifier-free guidance scale. Default: 10.0
    --steps            Number of diffusion steps. Default: 100
    --skip             Steps to skip in reverse process. Default: 70
    --seed             Random seed for reproducibility. Default: 0

Models auto-download from Hugging Face on first run:
    - stable-diffusion-v1-5/stable-diffusion-v1-5 (SD pipeline)

Output: Saves PNG files with the same basename as input images.
Note: Requires GPU (CUDA). Works best on aligned face images.
"""

import argparse
import os
import sys
import torch
import logging
from pathlib import Path
from PIL import Image
from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


def get_image_files(input_dir: str) -> list:
    files = []
    for ext in SUPPORTED_EXTENSIONS:
        files.extend(Path(input_dir).glob(f"*{ext}"))
        files.extend(Path(input_dir).glob(f"*{ext.upper()}"))
    return sorted(set(files))


def init_model(args):
    """Load NullFace dependencies (SD pipeline + InsightFace)."""
    if not torch.cuda.is_available():
        raise RuntimeError("NullFace requires GPU (CUDA).")

    print("NullFace ready (models download from HF on first run).")
    return {
        'device_num': 0,
        'guidance_scale': args.guidance_scale,
        'steps': args.steps,
        'skip': args.skip,
        'seed': args.seed
    }


def deidentify(img_path, out_path, model_state):
    """De-identify a single face image using NullFace."""
    import contextlib
    import io

    from anonymize_face import anonymize_face as nullface_anonymize

    # Suppress noisy output from insightface and per-step diffusion progress bars
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        result = nullface_anonymize(
            image_path=str(img_path),
            mask_image_path=None,  # Will use white mask if not found
            device_num=model_state['device_num'],
            guidance_scale=model_state['guidance_scale'],
            num_diffusion_steps=model_state['steps'],
            skip=model_state['skip'],
            seed=model_state['seed'],
            output_log_file=os.devnull,  # Suppress log file writes
        )

    if result is not None:
        result.save(out_path, "PNG")
        return True
    return False


def run(args):
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_files = get_image_files(str(input_dir))
    if not image_files:
        print(f"No supported images found in {input_dir}")
        return

    model_state = init_model(args)

    processed = 0
    failed = 0

    skipped = 0
    for img_path in tqdm(image_files, desc="De-identifying"):
        try:
            stem = img_path.stem
            out_path = output_dir / f"{stem}.png"

            # Skip already processed images
            if out_path.exists():
                tqdm.write(f"  ⏭ Skipping {img_path.name} (already exists)")
                skipped += 1
                continue

            ok = deidentify(img_path, out_path, model_state)
            if ok:
                processed += 1
            else:
                failed += 1

        except Exception as e:
            print(f"\n✗ Failed on {img_path.name}: {e}")
            failed += 1

    print(f"\nDone. Processed: {processed}, Skipped (exists): {skipped}, Failed: {failed}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch face de-identification using NullFace (training-free)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="Input folder with face images")
    parser.add_argument("--output", required=True, help="Output folder for anonymized images (PNG)")
    parser.add_argument("--guidance_scale", type=float, default=10.0,
                        help="Guidance scale for diffusion. Default: 10.0")
    parser.add_argument("--steps", type=int, default=100,
                        help="Number of diffusion steps. Default: 100 (must be > skip). Paper uses 100.")
    parser.add_argument("--skip", type=int, default=70,
                        help="Steps to skip in reverse process. Default: 70 (paper default). Higher = more realistic face structure preserved.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for reproducibility. Default: 0")

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
