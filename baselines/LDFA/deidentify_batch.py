#!/usr/bin/env python
"""Batch face de-identification using LDFA (Latent Diffusion Face Anonymization).

Klemp et al., CVPRW 2023. Uses Stable Diffusion 1.5 inpainting to generate a
completely different face within the masked region. NOT REVERSIBLE.

Pipeline per image:
  Aligned face (256x256) -> full white mask -> SD 1.5 inpainting -> new face PNG

Model requirements:
  Download any SD 1.5-compatible .safetensors checkpoint (base model, not LoRA).
  Original LDFA paper used RealisticVision V6.0 B1:
    https://huggingface.co/SG161222/Realistic_Vision_V6.0_B1_noVAE/blob/main/realisticVisionV60B1_v60B1VAE.safetensors

  Place the .safetensors file in LDFA/models/ (or specify --model-dir).

Usage:
    python deidentify_batch.py --input <source_folder> --output <dest_folder> [options]

Options:
    --model-dir        Local path to SD 1.5 model (optional). Falls back to HF download.
    --steps            Inference steps. Default: 40 (matches original LDFA).
    --denoising        Denoising strength for inpainting. Default: 0.75.
    --seed             Random seed for reproducibility. Default: 42.

Notes:
    For aligned face datasets (FRI): uses full white mask → inpaints entire image
    with a generated face. Output preserves original dimensions.

    Original LDFA used A1111 Docker container. This script replaces that with
    diffusers StableDiffusionInpaintPipeline (no Docker, no API calls).
"""

import argparse
import os
import sys

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Stub cv2 — not needed for inference
class _FakeCv2:
    COLOR_RGB2BGR = 0; INTER_AREA = 1; INTER_LINEAR = 1
    @staticmethod
    def imread(*a, **k): return None
    @staticmethod
    def imwrite(*a, **k): return True
    @staticmethod
    def resize(*a, **k): return None
    @staticmethod
    def cvtColor(*a, **k): return None
sys.modules['cv2'] = _FakeCv2()

from pathlib import Path
from PIL import Image
from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


def get_image_files(input_dir: str) -> list:
    files = []
    for ext in SUPPORTED_EXTENSIONS:
        files.extend(Path(input_dir).glob(f"*{ext}"))
        files.extend(Path(input_dir).glob(f"*{ext.upper()}"))
    return sorted(set(files))


def run(args):
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_files = get_image_files(str(input_dir))
    if not image_files:
        print(f"No supported images found in {input_dir}")
        return

    # Lazy import + load model on first use
    from ldfa_diffusers import LdfaAnonymizer

    model_file = args.model_dir if args.model_dir else os.path.join(REPO_ROOT, "models", "realisticVisionV60B1_v60B1VAE.safetensors")

    if not os.path.isfile(model_file):
        print(f"Error: Model file not found: {model_file}")
        print("Download realisticVisionV60B1_v60B1VAE.safetensors to LDFA/models/")
        return

    anon = LdfaAnonymizer(model_dir=model_file, seed=args.seed)

    processed = 0
    skipped = 0
    failed = 0
    for img_path in tqdm(image_files, desc="De-identifying"):
        stem = img_path.stem
        out_path = output_dir / f"{stem}.png"

        # Skip already-processed images
        if out_path.exists():
            skipped += 1
            continue

        try:
            pil_img = Image.open(img_path).convert("RGB")

            anon_pil = anon.anonymize(
                pil_img,
                bbox=None,  # full white mask (entire aligned face)
                steps=args.steps,
                denoising_strength=args.denoising,
            )

            # Resize output to 256x256 for consistent size across all baselines
            if anon_pil.size != (256, 256):
                anon_pil = anon_pil.resize((256, 256), Image.LANCZOS)

            anon_pil.save(out_path, "PNG")
            processed += 1

        except Exception as e:
            import traceback
            print(f"\nFailed on {img_path.name}: {e}")
            traceback.print_exc()
            failed += 1

    print(f"\nDone. Processed: {processed}, Skipped: {skipped}, Failed: {failed}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch face de-identification using LDFA (Latent Diffusion Face Anonymization)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="Input folder with aligned face images")
    parser.add_argument("--output", required=True, help="Output folder for anonymized images (PNG)")
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Path to .safetensors checkpoint file. Default: models/realisticVisionV60B1_v60B1VAE.safetensors",
    )
    parser.add_argument(
        "--steps", type=int, default=40, help="Inference steps. Default: 40"
    )
    parser.add_argument(
        "--denoising", type=float, default=0.75, help="Denoising strength. Default: 0.75"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility. Default: 42"
    )

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
