"""Re-process NullFace failures by bypassing face detection with random identity embeddings.

NullFace uses InsightFace to detect faces and extract identity embeddings.
When detection fails, we can skip it entirely by generating a random 512-d
embedding — the diffusion inpainting still produces anonymized output.

Usage:
    python fix_nullface_failures.py --input <celeba-test_input_dir> --output <celeba-test_output_dir>
        [--missing_ids 00002 02086 ...]  # auto-detected if omitted
"""
import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from PIL import Image
from diffusers import DDIMScheduler, StableDiffusionInpaintPipeline
from diffusers.utils import load_image
from torch import autocast, inference_mode

# Import NullFace internals
from ddm_inversion.inversion_utils import (
    inversion_forward_process,
    inversion_reverse_process,
)
from ddm_inversion.utils import image_grid


def random_identity_embedding(max_angle_deg=0.0, seed=None):
    """Generate a random 512-d identity embedding (InsightFace buffalo_l dimension).

    Instead of detecting a face and extracting its embedding, we sample
    a purely random normalized vector. The IP-Adapter-FaceID will still
    use it as conditioning, producing an anonymized result."""
    if seed is not None:
        np.random.seed(seed)
    # Random point on the unit sphere in 512 dimensions
    vec = np.random.randn(512)
    vec /= np.linalg.norm(vec)
    return vec


def build_ip_adapter_embeds(ref_embedding, dtype=torch.float32, device="cuda", scale_factor=1.0):
    """Build ip_adapter_image_embeds tensors for IP-Adapter-FaceID from a raw embedding.

    Mirrors the logic in FaceEmbeddingExtractor.get_face_embeddings() but without detection."""
    ref_images_embeds = [ref_embedding * scale_factor]
    ref_images_embeds = torch.stack(ref_images_embeds, dim=0).unsqueeze(0)  # (1, 1, 1, 512)
    neg_ref_images_embeds = torch.zeros_like(ref_images_embeds)

    concat_emb_inv = torch.cat([neg_ref_images_embeds, neg_ref_images_embeds]).to(dtype=dtype, device=device)
    concat_emb = torch.cat([neg_ref_images_embeds, ref_images_embeds]).to(dtype=dtype, device=device)
    return concat_emb_inv, concat_emb


def anonymize_face_no_detect(
    image_path: str,
    sd_model_path: str = "stable-diffusion-v1-5/stable-diffusion-v1-5",
    device_num: int = 0,
    guidance_scale: float = 10.0,
    num_diffusion_steps: int = 100,
    eta: float = 1.0,
    skip: int = 70,
    ip_adapter_scale: float = 1.0,
    id_emb_scale: float = 1.0,
    seed: int = 0,
    mask_delay_steps: int = 10,
):
    """Anonymize a face image without face detection — use random identity embedding."""
    device = f"cuda:{device_num}"

    # Load model
    ldm_stable = StableDiffusionInpaintPipeline.from_pretrained(
        sd_model_path, torch_dtype=torch.float16
    ).to(device)
    ldm_stable.load_ip_adapter(
        "h94/IP-Adapter-FaceID",
        subfolder=None,
        weight_name="ip-adapter-faceid_sd15.bin",
        image_encoder_folder=None,
    )
    ldm_stable.set_ip_adapter_scale(ip_adapter_scale)
    dtype = ldm_stable.dtype

    # Use a random identity embedding instead of detecting a face
    ref_emb = torch.from_numpy(random_identity_embedding(seed=seed)).to(dtype=torch.float32)
    id_embs_inv, id_embs = build_ip_adapter_embeds(
        ref_emb, dtype=dtype, device=device, scale_factor=id_emb_scale
    )

    ldm_stable.scheduler = DDIMScheduler.from_config(sd_model_path, subfolder="scheduler")
    ldm_stable.scheduler.set_timesteps(num_diffusion_steps)

    # Load image — load_512 crops to 512x512
    from prompt_to_prompt.ptp_classes import load_512
    offsets = (0, 0, 0, 0)
    x0 = load_512(image_path, *offsets, device).to(dtype=dtype)

    # White mask (full inpainting)
    height, width = x0.shape[-2:]
    mask_image = Image.new("RGB", (width, height), "white")

    # VAE encode
    with autocast("cuda"), inference_mode():
        w0 = (ldm_stable.vae.encode(x0).latent_dist.mode() * 0.18215).to(dtype=dtype)

    # Forward process
    wt, zs, wts = inversion_forward_process(
        ldm_stable,
        w0,
        etas=eta,
        prompt="",
        cfg_scale=guidance_scale,
        prog_bar=True,
        num_inference_steps=num_diffusion_steps,
        ip_adapter_image_embeds=[id_embs_inv],
    )

    generator = torch.manual_seed(seed)

    # Reverse process with random identity embedding
    w0, _ = inversion_reverse_process(
        ldm_stable,
        xT=wts[num_diffusion_steps - skip],
        etas=eta,
        prompts=[""],
        cfg_scales=[guidance_scale],
        prog_bar=True,
        zs=zs[: (num_diffusion_steps - skip)],
        controller=None,
        ip_adapter_image_embeds=[id_embs],
        init_image=x0,
        mask_image=mask_image,
        generator=generator,
        mask_delay_steps=mask_delay_steps,
    )

    # VAE decode
    with autocast("cuda"), inference_mode():
        x0_dec = ldm_stable.vae.decode(1 / 0.18215 * w0).sample
    if x0_dec.dim() < 4:
        x0_dec = x0_dec[None, :, :, :]
    img = image_grid(x0_dec)
    return img


def get_missing_images(input_dir, output_dir):
    """Find images in input that have no corresponding output (extension-agnostic)."""
    input_files = set()
    for f in Path(input_dir).iterdir():
        if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}:
            input_files.add(f.stem)

    output_files = set()
    for f in Path(output_dir).iterdir():
        if f.is_file():
            output_files.add(f.stem)

    return sorted(input_files - output_files)


def main():
    parser = argparse.ArgumentParser(
        description="Re-process NullFace failures using random identity embeddings (no face detection)"
    )
    parser.add_argument("--input", required=True, help="Input celeba-test directory")
    parser.add_argument("--output", required=True, help="Output celeba-test directory")
    parser.add_argument("--guidance_scale", type=float, default=10.0)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--skip", type=int, default=70)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    input_dir = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find missing images
    missing_ids = get_missing_images(input_dir, output_dir)
    if not missing_ids:
        print("No missing images found — all processed!")
        return

    print(f"Found {len(missing_ids)} missing images:")
    print(" ".join(missing_ids))

    # Find image files by stem
    from difflib import get_close_matches
    input_map = {}
    for f in input_dir.iterdir():
        if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}:
            input_map[f.stem] = f

    image_paths = [(sid, input_map[sid]) for sid in missing_ids if sid in input_map]
    if not image_paths:
        print("ERROR: Could not find input images.")
        return

    print(f"\nProcessing {len(image_paths)} images with random identity embeddings...")
    processed = 0
    failed = 0

    for idx, (stem, img_path) in enumerate(image_paths, 1):
        out_path = output_dir / f"{stem}.png"
        print(f"  [{idx}/{len(image_paths)}] Processing {img_path.name} ...")
        try:
            result = anonymize_face_no_detect(
                image_path=str(img_path),
                device_num=0,
                guidance_scale=args.guidance_scale,
                num_diffusion_steps=args.steps,
                skip=args.skip,
                seed=args.seed,
                mask_delay_steps=10,
            )
            result.save(str(out_path), "PNG")
            processed += 1
            print(f"    ✓ Saved {out_path.name}")
        except Exception as e:
            failed += 1
            print(f"    ✗ Failed: {e}")

    print(f"\nDone. Processed: {processed}, Failed: {failed}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
