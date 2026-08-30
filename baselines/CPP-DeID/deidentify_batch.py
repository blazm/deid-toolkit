#!/usr/bin/env python
"""Batch DeID runner for CPP-DeID (StyleCLIP / latent optimization).

Uses GradualStyleBlock encoder + 1024px StyleGAN2 generator.
Optimizes latents in W+ space to reduce face ID similarity at multiple thresholds.
Matches deid-cpp's sampling process: raw dot-product ID loss, no gender/expression losses.

All models (facial_recognition, stylegan2, encoders) merged into CPP-DeID/models/
so the `models` namespace is a single package -- no import conflicts.
"""

import argparse
import math
import os
import sys

import torch
import torchvision
import torch.nn.functional as F
from PIL import Image
from torch import optim
from torch.nn import MSELoss
from torchvision import transforms as T
from tqdm import tqdm


# ============================================================================
# Paths
# ============================================================================
BASE = os.path.abspath(os.path.dirname(__file__))  # CPP-DeID/ root

# Only CPP-DeID root on sys.path. All `models.*` resolve to CPP-DeID/models/
# which now contains facial_recognition/, stylegan2/, encoders/ in one place.
sys.path.insert(0, BASE)


def _load_module(name, path):
    """Load a .py file as a module."""
    import importlib.util as iu
    spec = iu.spec_from_file_location(name, path)
    mod = iu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ============================================================================
# Load encoder FIRST so `models` namespace binds to CPP-DeID/models/
# HyperStyle's generator also uses `from models.stylegan2.op`, which will
# now correctly resolve to our merged directory.
# ============================================================================
from models.encoders.psp_encoders import Encoder4Editing
from criteria.id_loss import IDLoss

_gen_mod = _load_module('cpp_deid_gen', os.path.join(BASE, 'hyperstyle_local/models/stylegan2/model.py'))
Generator = _gen_mod.Generator


# ============================================================================
# Encoder loading -- GradualStyleBlock (IR-SE50 backbone)
# ============================================================================

def _load_encoder(checkpoint_path, device='cuda'):
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    opts = argparse.Namespace(**ckpt['opts'])

    encoder = Encoder4Editing(50, 'ir_se', opts)
    enc_dict = {k.replace('encoder.', ''): v for k, v in ckpt['state_dict'].items() if k.startswith('encoder.')}
    encoder.load_state_dict(enc_dict)
    encoder.eval().to(device)

    latent_avg = ckpt['latent_avg'].to(device)
    def add_latent_avg(model, inputs, outputs):
        return outputs + latent_avg.repeat(outputs.shape[0], 1, 1)
    encoder.register_forward_hook(add_latent_avg)
    return encoder


E4E_CKPTS = [
    os.path.join(BASE, "e4e_local", "pretrained_models", "e4e_ffhq_encode.pt"),
    os.path.join(BASE, "content", "encoder4editing", "pretrained_models", "e4e_ffhq_encode.pt"),
]


# ---------------------------------------------------------------------------
# Step 1 -- Encoding
# ---------------------------------------------------------------------------

def init_encoder():
    ckpt_path = None
    for p in E4E_CKPTS:
        if os.path.exists(p):
            ckpt_path = p
            break
    if ckpt_path is None:
        print("ERROR: encoder checkpoint not found.")
        sys.exit(1)
    return _load_encoder(ckpt_path, device="cuda")


def encode_images(images_dir, latents_dir, encoder_net, generator):
    os.makedirs(latents_dir, exist_ok=True)
    gen_n_styles = generator.n_latent
    print(f"  Generator n_styles: {gen_n_styles}")

    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = sorted(f for f in os.listdir(images_dir) if f.lower().endswith(valid_exts))
    if not image_files:
        print(f"  ERROR: no images in {images_dir}")
        return []

    transform = T.Compose([T.Resize((256, 256)), T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    print(f"  Found {len(image_files)} images, encoding ...")
    encoded = []
    with torch.no_grad():
        for img_path in image_files:
            stem = os.path.splitext(img_path)[0]
            latent_path = os.path.join(latents_dir, f"{stem}.pt")

            if os.path.exists(latent_path):
                print(f"  {img_path} -> latent exists, skipping encode")
                encoded.append((img_path, latent_path))
                continue

            im = Image.open(os.path.join(images_dir, img_path)).convert("RGB")
            tensor = transform(im).unsqueeze(0).cuda()
            code = encoder_net(tensor)  # [1, 18, 512]

            if code.shape[1] != gen_n_styles:
                print(f"  Truncating latents from {code.shape[1]} -> {gen_n_styles} styles")
                code = code[:, :gen_n_styles, :]

            torch.save(code.cpu(), latent_path)
            encoded.append((img_path, latent_path))
            print(f"  Encoded {img_path}")

    return encoded


# ---------------------------------------------------------------------------
# Step 2 -- Optimization (matches deid-cpp: ID + L2 only)
# ---------------------------------------------------------------------------

def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)
    return initial_lr * lr_ramp


def load_image(path):
    tr = T.Compose([T.Resize((256, 256)), T.ToTensor(), T.Normalize(mean=[0.5], std=[0.5])])
    return tr(Image.open(path).convert("RGB")).unsqueeze(0)


def deidentify(full_image_path, latent_path, g_ema, id_loss_fn, output_dir, thresholds, step=30, lr=0.1,
               id_lambda=2.48, l2_lambda=0.016):

    def ensure_dir(d):
        if not os.path.exists(d):
            os.makedirs(d)

    img_orig = load_image(full_image_path).cuda()
    latent_code_init = torch.load(latent_path, map_location="cuda", weights_only=False)

    # If no thresholds specified, default to single run
    if not thresholds:
        thresholds = [0.5]

    # Single threshold: save directly under output_dir/
    # Multiple thresholds: save to subdirs {tsim}/img.png for backward compat
    use_subdirs = len(thresholds) > 1
    for tsim in thresholds:
        latent = latent_code_init.detach().clone()
        latent.requires_grad = True
        optimizer = optim.Adam([latent], lr=lr)

        for i in range(step):
            t = i / step
            optimizer.param_groups[0]["lr"] = get_lr(t, lr)

            img_gen, _ = g_ema([latent], input_is_latent=True, randomize_noise=False)

            # Reconstruction MSE at 256px (matching original image resolution)
            img_gen_256 = F.interpolate(img_gen, size=(256, 256), mode='bilinear', align_corners=False)
            c_loss = MSELoss()(img_gen_256, img_orig)

            # ID loss at native generator resolution (matches deid-cpp -- no interpolation blur)
            i_loss_raw = id_loss_fn(img_gen, img_orig)[0]
            i_loss = ((i_loss_raw - tsim) ** 2).sum()

            # L2 latent constraint (matches deid-cpp's default l2_lambda=0.016)
            l2_loss = ((latent_code_init - latent) ** 2).sum()

            loss = c_loss + l2_lambda * l2_loss + id_lambda * i_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            img_gen, _ = g_ema([latent], input_is_latent=True, randomize_noise=False)

        stem = os.path.splitext(os.path.basename(full_image_path))[0]

        if use_subdirs:
            # Multiple thresholds — create subdir per threshold (backward compat)
            base_path = os.path.join(output_dir, str(tsim))
            ensure_dir(base_path)
            out_path = os.path.join(base_path, f"{stem}.png")
        else:
            # Single threshold — save directly under output_dir/
            out_path = os.path.join(output_dir, f"{stem}.png")

        # 1024px output -> save at 256px for visibility
        img_256 = F.interpolate(img_gen, size=(256, 256), mode='bilinear', align_corners=False)
        torchvision.utils.save_image(img_256, out_path, normalize=True, value_range=(-1, 1))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _check_skip(img_name, output_dir, thresholds, use_subdirs):
    """Check if this image's expected output file(s) already exist."""
    stem = os.path.splitext(img_name)[0]
    if not use_subdirs:
        # Single threshold — one expected output per image
        return os.path.exists(os.path.join(output_dir, f"{stem}.png"))

    # Multiple thresholds — check if at least one version exists so we can skip the expensive loop entirely
    for tsim in thresholds:
        out_path = os.path.join(output_dir, str(tsim), f"{stem}.png")
        if os.path.exists(out_path):
            return True
    return False


def main(args):
    print("=" * 60)
    print("CPP-DeID batch DeID (GradualStyleBlock + 1024px StyleGAN2)")
    print("=" * 60)

    args.input_dir = os.path.abspath(args.input_dir)
    args.latents_dir = os.path.abspath(args.latents_dir)
    args.output_dir = os.path.abspath(args.output_dir)
    if not os.path.isabs(args.ckpt):
        args.ckpt = os.path.normpath(os.path.join(BASE, args.ckpt))

    print(f"\n[Loading] Encoder ...")
    encoder_net = init_encoder()

    print(f"[Loading] 1024px StyleGAN2 generator from {args.ckpt} ...")
    g_ema = Generator(1024, 512, 8)
    ckpt_data = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    g_ema.load_state_dict(ckpt_data["g_ema"], strict=False)
    g_ema.eval().cuda()

    print(f"\n[Step 1] Encoding images -> W+ latents ...")
    pairs = encode_images(args.input_dir, args.latents_dir, encoder_net, g_ema)
    if not pairs:
        print("No images to process. Exiting.")
        sys.exit(1)

    # Resolve thresholds (same logic as deidentify())
    if not args.threshold:
        effective_thresholds = [0.5]
    else:
        effective_thresholds = args.threshold
    use_subdirs = len(effective_thresholds) > 1

    print(f"\n[Step 2] Optimizing {len(pairs)} images ...")
    print(f"  Threshold(s): {effective_thresholds}")

    # Only ID loss (no gender/expression -- matches deid-cpp)
    class _Opts:
        ir_se50_weights = args.ir_se50_weights or os.path.join(BASE, "pretrained_models", "model_ir_se50.pth")
    id_loss_fn = IDLoss(_Opts()).cuda()

    skipped = 0
    for img_name, latent_path in tqdm(pairs, desc="De-identifying"):
        # Skip if all expected output files already exist
        if args.skip_existing and _check_skip(img_name, args.output_dir, effective_thresholds, use_subdirs):
            skipped += 1
            continue

        deidentify(
            full_image_path=os.path.join(args.input_dir, img_name),
            latent_path=latent_path,
            g_ema=g_ema,
            id_loss_fn=id_loss_fn,
            output_dir=args.output_dir,
            thresholds=args.threshold,
            step=args.step,
            lr=args.lr,
            id_lambda=args.id_lambda,
            l2_lambda=args.l2_lambda,
        )

    if skipped:
        print(f"\nSkipped {skipped} images (output already exists).")
    print("\nDone. Results saved to", args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CPP-DeID batch de-identification (GradualStyleBlock + 1024px)")
    parser.add_argument("--input_dir", required=True, help="Aligned face images")
    parser.add_argument("--latents_dir", required=True, help="Directory for .pt latent codes")
    parser.add_argument("--output_dir", required=True, help="Where anonymized results go")
    parser.add_argument("--ckpt", default=os.path.join(BASE, "pretrained_models", "stylegan2-ffhq-config-f.pt"),
                        help="StyleGAN2 checkpoint path (1024px)")
    parser.add_argument("--step", type=int, default=30, help="Optimization steps per threshold")
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--threshold", nargs="+", type=float, default=None,
                        help="Target similarity threshold(s). Default: all 0.1..0.9. Use e.g. --threshold 0.5 for single run")
    # Loss weights matching deid-cpp defaults (run_deid.py edit mode)
    parser.add_argument("--id_lambda", type=float, default=2.48, help="ID loss weight (deid-cpp run_deid default: 2.48; folder: 5.5)")
    parser.add_argument("--l2_lambda", type=float, default=0.016, help="L2 latent constraint (deid-cpp default: 0.016)")
    parser.add_argument("--ir_se50_weights", default=None)
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip images whose output files already exist in the output directory")

    args = parser.parse_args()
    main(args)
