#!/usr/bin/env python
"""Batch face de-identification using FALCO (Latent Code Optimization).

Simplified pipeline for pre-aligned face images (no fake dataset / NN search needed).

Pipeline per image:
  Image -> resize to 1024 -> e4e encoder -> W+ latent codes
  -> Optimize layers 3-7 (ArcFace id_loss + FaRL attr_loss)
  -> StyleGAN2 decode -> center-crop face -> resize to 256 -> deidentified PNG

Usage:
    python deidentify_batch.py --input <source_folder> --output <dest_folder> [options]

Options:
    --id-margin          Identity loss margin (default: 0.5). Higher = stronger anonymization.
    --epochs             Optimization steps per image (default: 50).
    --lr                 Learning rate (default: 0.01).
    --lambda-id          Identity loss weight (default: 10.0).
    --lambda-attr        Attribute preservation weight (default: 0.1).
"""

import argparse
import os
import sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # Avoid OMP conflict with MKL/OpenCV

# Stub cv2 — not needed for inference (only SFD face detection needs it)
class _FakeCv2:
    COLOR_RGB2BGR = 0
    INTER_AREA = 1
    INTER_LINEAR = 1
    INTER_CUBIC = 2
    BORDER_REFLECT = 4
    @staticmethod
    def imread(*a, **k): return None
    @staticmethod
    def imwrite(*a, **k): return True
    @staticmethod
    def resize(*a, **k): return None
    @staticmethod
    def cvtColor(*a, **k): return None
    @staticmethod
    def detectMultiScale(*a, **k): return []
    class dnn:
        @staticmethod
        def NMSBoxes(*a, **k): return []
sys.modules['cv2'] = _FakeCv2()

import numpy as np
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torchvision.transforms.functional as F
import torchvision.transforms as transforms

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
    """Load StyleGAN2 (GenForce), e4e encoder (pSp), IDLoss (ArcFace), AttrLoss (FaRL)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. StyleGAN2 generator (GenForce — pure PyTorch, no CUDA ops)
    from models.load_generator import load_generator
    G = load_generator(model_name='stylegan2_ffhq1024', latent_is_w=True, verbose=True).eval().to(device)

    # 2. e4e encoder via pSp (same pattern as invert.py)
    from argparse import Namespace
    from models.psp import pSp
    e4e_ckpt_path = os.path.join(REPO_ROOT, 'models', 'pretrained', 'e4e', 'e4e_ffhq_encode.pt')
    ckpt_e4e = torch.load(e4e_ckpt_path, map_location='cpu')
    e4e_opts = ckpt_e4e['opts']
    e4e_opts['checkpoint_path'] = e4e_ckpt_path
    e4e_opts['device'] = str(device)
    e4e = pSp(Namespace(**e4e_opts)).eval().to(device)

    # 3. IDLoss (ArcFace backbone — uses IR-SE50, same as e4e backbone)
    from lib.id_loss import IDLoss
    id_criterion = IDLoss(id_margin=args.id_margin).eval().to(device)

    # 4. AttrLoss (FaRL — CLIP ViT-B/16 fine-tuned on faces)
    from lib.attr_loss import AttrLoss
    attr_criterion = AttrLoss(feat_ext='farl', use_cuda=(device.type == 'cuda')).eval().to(device)

    print(f"\nFALCO loaded on {device}")
    print(f"  StyleGAN2: FFHQ-1024 (GenForce, pure PyTorch)")
    print(f"  e4e: Encoder4Editing, start_from_latent_avg={e4e_opts.get('start_from_latent_avg')}")
    return {
        "G": G, "e4e": e4e, "id_loss": id_criterion, "attr_loss": attr_criterion,
        "device": device, "latent_avg": e4e.latent_avg
    }


def tensor2pil(img_tensor):
    """Convert [-1,1] range StyleGAN2 output tensor to PIL image."""
    x = ((img_tensor.cpu() + 1) / 2).clamp(0, 1).squeeze(0)
    return F.to_pil_image((x.mul(255).byte()))


def anonymize_face(pil_img, model_state, args):
    """De-identify a single face via latent code optimization.

    Steps:
      1. Resize to 1024 -> e4e encode -> W+ codes (18 layers)
      2. Optimize layers 3-7 with ArcFace + FaRL losses
      3. StyleGAN2 decode -> center-crop -> resize to 256
    """
    device = model_state["device"]
    G = model_state["G"]
    e4e = model_state["e4e"]
    id_criterion = model_state["id_loss"]
    attr_criterion = model_state["attr_loss"]
    latent_avg = model_state["latent_avg"]

    # --- Step 1: Invert via e4e (at 256x256, matching training) ---
    img_256 = pil_img.resize((256, 256), Image.LANCZOS)
    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    img_t = to_tensor(img_256).unsqueeze(0).to(device)

    with torch.no_grad():
        w_code = e4e.encoder(img_t)
        if latent_avg is not None:
            w_code = w_code + latent_avg.repeat(w_code.shape[0], 1, 1)
        # Also get a reconstruction at 256 for identity reference
        recon_full = G(w_code, randomize_noise=False)  # [1, 3, 1024, 1024]
        # No center-crop — full 1024 image IS the face (FFHQ). Loss functions have built-in transforms.

    # --- Step 2: Setup latent code optimization ---
    # FALCO approach: freeze layers 0-2 and 8-end, optimize layers 3-7
    layer_start = 3
    layer_end = 8

    fixed_start = w_code[:, :layer_start, :]  # Identity/structure
    trainable = w_code[:, layer_start:layer_end, :].clone().requires_grad_(True)
    fixed_end = w_code[:, layer_end:, :]  # Fine details

    optimizer = torch.optim.Adam([trainable], lr=args.lr)

    # --- Step 3: Optimization loop ---
    for epoch in range(args.epochs):
        wp = torch.cat([fixed_start, trainable, fixed_end], dim=1)

        # Generate anonymized face at 1024
        anon_full = G(wp, randomize_noise=False)  # [1, 3, 1024, 1024]
        # Pass full 1024 images — IDLoss has built-in resize+crop, AttrLoss has its own transform

        # Compute losses
        id_loss = id_criterion(anon_full, recon_full)
        attr_loss = attr_criterion(anon_full, recon_full)
        loss = args.lambda_id * id_loss + args.lambda_attr * attr_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # --- Step 4: Final decode + resize to 256 ---
    with torch.no_grad():
        wp_final = torch.cat([fixed_start, trainable.detach(), fixed_end], dim=1)
        anon_final = G(wp_final, randomize_noise=False)  # [1, 3, 1024, 1024]

    # Resize full face from 1024 -> 256 (no crop — FFHQ generates full-face at 1024)
    import torch.nn.functional as nnF
    anon_resized = nnF.interpolate(anon_final, size=256, mode='bilinear', align_corners=False)
    return tensor2pil(anon_resized)


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
    for img_path in tqdm(image_files, desc="Anonymizing"):
        stem = img_path.stem
        out_path = output_dir / f"{stem}.png"
        if out_path.exists():
            skipped += 1
            continue

        try:
            pil_img = Image.open(img_path).convert("RGB")
            anon_pil = anonymize_face(pil_img, model_state, args)

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
        description="Batch face de-identification using FALCO (Latent Code Optimization)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="Input folder with aligned face images")
    parser.add_argument("--output", required=True, help="Output folder for anonymized images (PNG)")

    # FALCO parameters
    parser.add_argument("--id-margin", type=float, default=0.5,
                        help="Identity loss margin (0.0-1.0). Higher = stronger anonymization. Default: 0.5")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Optimization steps per image. Default: 50")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="Learning rate for latent optimization. Default: 0.01")
    parser.add_argument("--lambda-id", type=float, default=10.0,
                        help="Identity loss weight. Default: 10.0")
    parser.add_argument("--lambda-attr", type=float, default=0.1,
                        help="Attribute preservation weight. Default: 0.1")

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
