#!/usr/bin/env python
"""Batch face de-identification using FAMS (Face Anonymization Made Simple).

FAMS uses a ReferenceNet-based Stable Diffusion pipeline to generate anonymized faces.
The method is controllable via `anonymization_degree` (0 = face swap, 1.25 = full deid).

Usage:
    python deidentify_batch.py --input <source_folder> --output <dest_folder> [options]

Options:
    --degree           Anonymization degree (0-2.0). Default: 1.25
                       0.0 = face swap, 1.25 = recommended deid, higher = more distortion
    --steps            Number of diffusion inference steps. Default: 50
    --guidance_scale   Classifier-free guidance scale. Default: 4.0
    --seed             Random seed for reproducibility. Default: 42

Models auto-download from Hugging Face on first run:
    - hkung/face-anon-simple (ReferenceNet, UNet)
    - stabilityai/stable-diffusion-2-1 (VAE, scheduler)
    - openai/clip-vit-large-patch14 (image encoder)

Output: Saves PNG files resized to `--size` (default: 256x256). Diffusion runs at native 512x512 for quality.
"""

import argparse
import os
import sys
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from diffusers import AutoencoderKL, DDPMScheduler
from transformers import CLIPImageProcessor, CLIPVisionModel
from diffusers.utils import load_image

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
    """Load all FAMS models from Hugging Face."""
    print("Loading FAMS models (downloads from HF on first run) ...")

    face_model_id = "hkung/face-anon-simple"
    clip_model_id = "openai/clip-vit-large-patch14"
    sd_model_id = "stabilityai/stable-diffusion-2-1"

    # Import FAMS pipeline (custom ReferenceNet code in src/)
    from src.diffusers.models.referencenet.referencenet_unet_2d_condition import (
        ReferenceNetModel,
    )
    from src.diffusers.models.referencenet.unet_2d_condition import UNet2DConditionModel
    from src.diffusers.pipelines.referencenet.pipeline_referencenet import (
        StableDiffusionReferenceNetPipeline,
    )

    unet = UNet2DConditionModel.from_pretrained(
        face_model_id, subfolder="unet", use_safetensors=True
    )
    referencenet = ReferenceNetModel.from_pretrained(
        face_model_id, subfolder="referencenet", use_safetensors=True
    )
    conditioning_referencenet = ReferenceNetModel.from_pretrained(
        face_model_id, subfolder="conditioning_referencenet", use_safetensors=True
    )
    # Priority: --vae_path arg → local weights/ subfolder → HF (gated, requires login)
    auto_vae = os.path.join(REPO_ROOT, "weights", "vae")
    auto_sched = os.path.join(REPO_ROOT, "weights", "scheduler")
    vae_path = getattr(args, 'vae_path', None) or (auto_vae if os.path.isdir(auto_vae) else sd_model_id)
    scheduler_path = getattr(args, 'scheduler_path', None) or (auto_sched if os.path.isdir(auto_sched) else sd_model_id)
    vae_subfolder = None if vae_path != sd_model_id else "vae"
    scheduler_subfolder = None if scheduler_path != sd_model_id else "scheduler"
    vae = AutoencoderKL.from_pretrained(vae_path, subfolder=vae_subfolder, use_safetensors=True)
    scheduler = DDPMScheduler.from_pretrained(
        scheduler_path, subfolder=scheduler_subfolder, use_safetensors=True
    )
    feature_extractor = CLIPImageProcessor.from_pretrained(clip_model_id, use_safetensors=True)
    image_encoder = CLIPVisionModel.from_pretrained(clip_model_id, use_safetensors=True)

    pipe = StableDiffusionReferenceNetPipeline(
        unet=unet,
        referencenet=referencenet,
        conditioning_referencenet=conditioning_referencenet,
        vae=vae,
        feature_extractor=feature_extractor,
        image_encoder=image_encoder,
        scheduler=scheduler,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device)
    generator = torch.Generator(device=device).manual_seed(args.seed)

    print(f"FAMS loaded on {device}.")
    return {
        'pipe': pipe,
        'generator': generator,
        'device': device
    }


def deidentify(pil_img, model_state, args):
    """De-identify a single aligned face image using FAMS."""
    pipe = model_state['pipe']
    generator = model_state['generator']

    result_pil = pipe(
        source_image=pil_img,
        conditioning_image=pil_img,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
        anonymization_degree=args.degree,
        width=512,
        height=512,
    ).images[0]

    # Resize output to target size
    out_wh = tuple(int(x) for x in args.size.split(","))
    if (result_pil.width, result_pil.height) != out_wh:
        result_pil = result_pil.resize(out_wh, Image.LANCZOS)

    return result_pil


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
        stem = img_path.stem
        out_path = output_dir / f"{stem}.png"
        if out_path.exists():
            skipped += 1
            continue

        try:
            pil_img = Image.open(img_path).convert("RGB")

            result_pil = deidentify(pil_img, model_state, args)

            result_pil.save(out_path, "PNG")
            processed += 1

        except Exception as e:
            print(f"\n✗ Failed on {img_path.name}: {e}")
            failed += 1

    print(f"\nDone. Processed: {processed}, Skipped: {skipped}, Failed: {failed}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch face de-identification using FAMS (Face Anonymization Made Simple)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="Input folder with face images")
    parser.add_argument("--output", required=True, help="Output folder for anonymized images (PNG)")
    parser.add_argument("--degree", type=float, default=1.25,
                        help="Anonymization degree (0=swap, 1.25=deid). Default: 1.25")
    parser.add_argument("--steps", type=int, default=50,
                        help="Diffusion inference steps. Default: 50")
    parser.add_argument("--guidance_scale", type=float, default=4.0,
                        help="Classifier-free guidance scale. Default: 4.0")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility. Default: 42")
    parser.add_argument("--vae_path", default=None,
                        help="Local path to SD VAE (avoids HF gated download). "
                             "Download vae/ from stabilityai/stable-diffusion-2-1 (~167MB FP16 safetensors)")
    parser.add_argument("--scheduler_path", default=None,
                        help="Local path to SD scheduler config. Tiny (~1KB), only needed if VAE is local too.")
    parser.add_argument("--size", default="256,256",
                        help="Resize output to W,H. Diffusion runs at native 512x512. Default: 256,256")
    parser.add_argument("--resize", dest="size",
                        help="Alias for --size (deprecated)")

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
