#!/usr/bin/env python
"""Batch face de-identification using RiDDLE (Reversible De-identification in Latent Space).

For pre-aligned face images (256x256). Uses StyleGAN2 latent manipulation.

Pipeline:
  Image -> e4e encoder -> w latent codes -> mapper(w + password_w) -> encrypted w -> StyleGAN2 decoder -> deidentified face

Reversibility:
  encrypted_w + same password_w -> mapper -> original w -> StyleGAN2 decoder -> recovered face

Usage:
    python deidentify_batch.py --input <source_folder> --output <dest_folder> [options]

Options:
    --reverse          Also save recovered (reversed) images to prove reversibility
    --seed             Random seed for password generation. Default: 42
"""

import argparse
import os
import sys
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torchvision.transforms.functional as F
import torchvision.transforms as transforms

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "mapper"))

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


def get_image_files(input_dir: str) -> list:
    files = []
    for ext in SUPPORTED_EXTENSIONS:
        files.extend(Path(input_dir).glob(f"*{ext}"))
        files.extend(Path(input_dir).glob(f"*{ext.upper()}"))
    return sorted(set(files))


def init_model(args):
    """Load RiDDLE pipeline: e4e encoder + transformer mapper + StyleGAN2 decoder.

    Uses pSp wrapper (same as coach_test.py) for correct e4e loading.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load StyleGAN2 generator (decoder) — shared by all components
    from models.stylegan2.model import Generator
    decoder = Generator(args.size, 512, 8).to(device)
    ckpt_g = torch.load(args.stylegan_weights, map_location="cpu")
    decoder.load_state_dict(ckpt_g["g_ema"], strict=False)
    decoder.eval()

    # 2. Load e4e encoder using pSp (handles checkpoint loading correctly — same as coach_test.py)
    from argparse import Namespace
    from models.psp import pSp, get_keys
    ckpt_e4e = torch.load(args.e4e_weights, map_location="cpu")
    e4e_opts = ckpt_e4e["opts"]
    # Point at local weights so pSp loads encoder+decoder properly
    e4e_opts["checkpoint_path"] = args.e4e_weights
    e4e_opts["stylegan_weights"] = args.stylegan_weights
    e4e_opts["device"] = str(device)

    # Build pSp — this internally loads encoder weights via get_keys(ckpt, 'encoder')
    # The pSp decoder is NOT used (we use our own decoder for StyleGAN2 generation)
    e4e_net = pSp(Namespace(**e4e_opts))
    e4e_net.eval().to(device)

    # 3. Load transformer mapper (the RiDDLE de-id model)
    from mapper.latent_id_mappers import TransformerMapperSplit
    mapper = TransformerMapperSplit(
        split_list=args.split_list,
        normalize_type="layernorm",
        add_linear=True,
        add_pos_embedding=True,
    ).to(device)
    ckpt_mapper = torch.load(args.mapper_weights, map_location="cpu")
    mapper.load_state_dict(ckpt_mapper["mapper_state_dict"])
    mapper.eval()

    # Get style_count from encoder
    style_count = e4e_net.encoder.style_count if hasattr(e4e_net.encoder, 'style_count') else 14

    print(f"RiDDLE loaded on {device}")
    print(f"  Encoder type: {e4e_opts.get('encoder_type')}")
    print(f"  Style count: {style_count}")
    return {"e4e": e4e_net, "decoder": decoder, "mapper": mapper,
            "latent_avg": None, "device": device}


def deidentify(pil_img, model_state, password_w=None):
    """De-identify a single face image.

    Returns:
        (deid_pil, original_latent, password_latent, encrypted_latent)
    """
    e4e = model_state["e4e"]
    decoder = model_state["decoder"]
    mapper = model_state["mapper"]
    device = model_state["device"]
    style_count = e4e.encoder.style_count  # Typically 14 for FFHQ e4e

    # Prepare image: resize -> normalize to [-1, 1] (matches e4e training)
    size = 256
    img = pil_img.resize((size, size), Image.LANCZOS)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    img_t = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        # Encode image -> w latent codes using pSp encoder directly
        # pSp.forward() calls decoder too, so we bypass and use encoder directly
        w_code = e4e.encoder(img_t)

        # Apply latent_avg centering (pSp does this in forward() when start_from_latent_avg=True)
        if e4e.latent_avg is not None:
            w_code = w_code + e4e.latent_avg.repeat(w_code.shape[0], 1, 1)

        # If no password provided, generate one from random z -> style -> w
        if password_w is None:
            z_pwd = torch.randn(1, 512, device=device)
            w_pwd = decoder.style(z_pwd).unsqueeze(1).repeat(1, style_count, 1)
        else:
            w_pwd = password_w.clone()

        # Encrypt: mapper([w || w_pwd]) -> w_enc
        # Input: [batch, n_latent, 1024] where last dim is concatenated w + w_pwd
        w_enc = mapper(torch.cat([w_code, w_pwd], dim=-1))

        # Decode: w_enc -> deidentified face
        deid_img, _ = decoder([w_enc], input_is_latent=True, randomize_noise=False)

    # Denormalize [-1, 1] -> [0, 255] -> PIL
    deid_pil = F.to_pil_image(
        (deid_img[0].clamp(-1, 1).cpu() + 1).div(2).mul(255).byte()
    )

    return deid_pil, w_code, w_pwd, w_enc


def reverse(model_state, encrypted_w, password_w):
    """Reverse de-identification to recover the original face."""
    decoder = model_state["decoder"]
    mapper = model_state["mapper"]
    device = model_state["device"]

    with torch.no_grad():
        # Reverse: mapper([w_enc || w_pwd]) -> recovered w
        w_rec = mapper(torch.cat([encrypted_w, password_w], dim=-1))

        # Decode: recovered w -> face
        rec_img, _ = decoder([w_rec], input_is_latent=True, randomize_noise=False)

    # Denormalize [-1, 1] -> [0, 255] -> PIL
    rec_pil = F.to_pil_image(
        (rec_img[0].clamp(-1, 1).cpu() + 1).div(2).mul(255).byte()
    )

    return rec_pil


def run(args):
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.reverse:
        reverse_dir = output_dir.parent / (output_dir.name + "_reversed")
        reverse_dir.mkdir(parents=True, exist_ok=True)
        print(f"Reverse output will be saved to: {reverse_dir}")

    image_files = get_image_files(str(input_dir))
    if not image_files:
        print(f"No supported images found in {input_dir}")
        return

    model_state = init_model(args)

    # Generate one fixed password per run (seeded) for consistency across all images
    state = np.random.RandomState(seed=args.seed)
    z_pwd = torch.from_numpy(state.normal(size=(1, 512)).astype(np.float32))
    device = model_state["device"]
    decoder = model_state["decoder"]
    e4e = model_state["e4e"]
    style_count = e4e.encoder.style_count if hasattr(e4e.encoder, 'style_count') else 14
    z_pwd = z_pwd.to(device)
    w_pwd = decoder.style(z_pwd).unsqueeze(1).repeat(1, style_count, 1)
    print(f"Using password seed={args.seed}, w_pwd shape: {w_pwd.shape}")

    processed = 0
    failed = 0

    for img_path in tqdm(image_files, desc="De-identifying"):
        try:
            pil_img = Image.open(img_path).convert("RGB")
            deid_pil, _, _, w_enc = deidentify(
                pil_img, model_state, password_w=w_pwd
            )

            stem = img_path.stem
            out_path = output_dir / f"{stem}.png"
            deid_pil.save(out_path, "PNG")

            if args.reverse:
                rec_pil = reverse(model_state, w_enc, w_pwd)
                rev_path = reverse_dir / f"{stem}.png"
                rec_pil.save(rev_path, "PNG")

            processed += 1

        except Exception as e:
            import traceback
            print(f"\nFailed on {img_path.name}: {e}")
            traceback.print_exc()
            failed += 1

    print(f"\nDone. Processed: {processed}, Failed: {failed}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch face de-identification using RiDDLE (Reversible)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="Input folder with aligned face images")
    parser.add_argument("--output", required=True, help="Output folder for anonymized images (PNG)")
    parser.add_argument("--reverse", action="store_true",
                        help="Also save recovered (reversed) images in output/reversed/")

    # Model paths
    parser.add_argument("--e4e-weights", default=None,
                        help="Path to e4e encoder checkpoint")
    parser.add_argument("--stylegan-weights", default=None,
                        help="Path to StyleGAN2 FFHQ checkpoint")
    parser.add_argument("--mapper-weights", default=None,
                        help="Path to RiDDLE mapper (iteration_90000.pt)")

    # Architecture
    parser.add_argument("--size", type=int, default=256, help="StyleGAN image size. Default: 256")
    parser.add_argument("--split-list", type=int, nargs="+", default=[4, 4, 6],
                        help="Transformer split list. Default: 4 4 6")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for password. Default: 42")

    args = parser.parse_args()

    # Auto-resolve model paths from weights/RiDDLE-assets/
    assets_dir = Path(REPO_ROOT) / "weights" / "RiDDLE-assets"
    if assets_dir.exists():
        if args.e4e_weights is None:
            args.e4e_weights = str(assets_dir / "e4e_ffhq_encode_256.pt")
        if args.stylegan_weights is None:
            args.stylegan_weights = str(assets_dir / "stylegan2-ffhq-256.pt")
        if args.mapper_weights is None:
            args.mapper_weights = str(assets_dir / "iteration_90000.pt")

    # Validate required paths exist
    for name, path in [("e4e", args.e4e_weights), ("stylegan", args.stylegan_weights),
                        ("mapper", args.mapper_weights)]:
        p = Path(path)
        if not p.exists():
            print(f"Error: {name} weights not found at {path}")
            sys.exit(1)

    run(args)


if __name__ == "__main__":
    main()
