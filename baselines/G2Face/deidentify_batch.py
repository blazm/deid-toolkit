#!/usr/bin/env python
"""Batch face de-identification using G²Face (steganographic face anonymization).

G²Face hides the original identity in steganographic bits while replacing the visible face
with an anonymous one. The result is a single image that:
  - Looks like a different person (anonymous face)
  - Contains the original identity hidden in noise-level perturbations
  - Can be recovered back to the original face using the correct latent code

Usage:
    python deidentify_batch.py --input <source_folder> --output <dest_folder> [options]

Options:
    --model_path     Path to G2Face.pth checkpoint. Default: ./weights/G2Face.pth
    --arcface_path   Path to ArcFace backbone weights. Default: ./pretrain/ms1mv3_arcface_r50.pth
    --recon_path     Path to face reconstruction model weights. Default: ./weights/epoch_20.pth
    --device         CUDA device (default: cuda) or cpu for CPU-only mode.
                     Note: StyleGAN2 ops use PyTorch fallback, so CPU/GPU both work.
    --resize         Resize all images to W,H before processing. Output is this size. Default: 256x256

Output: Saves PNG files with the same basename as input images.
Each image is de-identified (anonymous face with hidden steganographic data).

Note: G²Face works on already-aligned/cropped face images (256x256 recommended).
For full-image processing with face detection + paste-back, consider preprocessing
the images first or using a wrapper script.
"""

import argparse
import os
import sys
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from torchvision.transforms import transforms
from torchvision.utils import save_image

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


def get_image_files(input_dir: str) -> list:
    """Return sorted list of image file paths in input_dir."""
    files = []
    for ext in SUPPORTED_EXTENSIONS:
        files.extend(Path(input_dir).glob(f"*{ext}"))
        files.extend(Path(input_dir).glob(f"*{ext.upper()}"))
    return sorted(set(files))


def image_transform():
    """Transform matching the one used in test.py."""
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])


def init_model(args):
    """Load all G²Face models needed for inference."""
    from model import HidingExtractor, Generator, Map2ID, MLP
    from model.d3dfr.arcface_torch.backbones import iresnet50

    device = torch.device(args.device)

    # Load checkpoint
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"G²Face model not found: {args.model_path}")

    print(f"Loading G²Face from: {args.model_path} ...")
    checkpoint = torch.load(args.model_path, map_location='cpu')

    # Anonymous face Generator (StyleGAN2-based)
    anonymous_net = Generator(size=256, style_dim=512).to(device)
    anonymous_net.load_state_dict(checkpoint['anonymous_net'])
    anonymous_net.eval()

    # ArcFace for identity extraction
    arc_face = iresnet50().to(device)
    if not os.path.exists(args.arcface_path):
        raise FileNotFoundError(f"ArcFace model not found: {args.arcface_path}")
    arc_face.load_state_dict(torch.load(args.arcface_path, map_location='cpu'))
    arc_face.eval()

    # Map2ID: maps random vector → identity latent
    map_2_id = Map2ID().to(device)
    map_2_id.load_state_dict(checkpoint['map_2_id'])
    map_2_id.eval()

    # Style MLP: controls the style based on ID + 3D params
    style_mlp = MLP(latent_dim=659, style_dim=512, n_mlp=4).to(device)
    style_mlp.load_state_dict(checkpoint['style_mlp'])
    style_mlp.eval()

    # Hiding extractor: recovers hidden bits from anonymized image
    hiding_extractor = HidingExtractor().to(device)
    hiding_extractor.load_state_dict(checkpoint['hiding_extractor'])
    hiding_extractor.eval()

    print("All G²Face models loaded.")
    return {
        'anonymous_net': anonymous_net,
        'arc_face': arc_face,
        'map_2_id': map_2_id,
        'style_mlp': style_mlp,
        'hiding_extractor': hiding_extractor,
        'device': device,
        'transform': image_transform()
    }


def deidentify(pil_img, model_state):
    """De-identify a single PIL image using G²Face.

    Returns the anonymized face (with steganographic identity hidden in it).
    """
    from utils.binary_converter import float2bit

    device = model_state['device']
    transform = model_state['transform']
    anonymous_net = model_state['anonymous_net']
    arc_face = model_state['arc_face']
    map_2_id = model_state['map_2_id']
    style_mlp = model_state['style_mlp']

    # Preprocess: PIL → normalized tensor [0.5, 0.5]
    img_tensor = transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        # Extract original face identity
        latent_face = arc_face(img_tensor)

        # Generate random anonymous identity
        batch_size = img_tensor.size(0)
        rand_z = torch.randn([batch_size, 512]).to(device)
        rand_id = map_2_id(rand_z)

        # Encode original identity as bits (steganographic hiding)
        latent_hiding = float2bit(latent_face)

        # Generate style control from random ID
        # Note: full pipeline uses 3D face params, but for simplicity we skip them
        # and use zero-initialized shape/expr (face is already aligned)
        latent_3d_placeholder = torch.zeros(batch_size, 147).to(device)  # 144 + 3 placeholder
        latent_control = style_mlp(rand_id, latent_3d_placeholder)

        # Generate anonymous image with hidden identity
        anonymous_image = anonymous_net(img_tensor, latent_control, latent_hiding)

    # Denormalize: [-1, 1] → [0, 1], tensor → PIL
    denorm = transforms.Lambda(lambda x: torch.clamp((x + 1) / 2, 0, 1))
    result_tensor = denorm(anonymous_image.squeeze(0).cpu())
    result_pil = transforms.ToPILImage()(result_tensor)

    return result_pil


def run(args):
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_files = get_image_files(str(input_dir))
    if not image_files:
        print(f"No supported images found in {input_dir}")
        return

    # Parse resize tuple
    resize_wh = tuple(int(x) for x in args.resize.split(","))

    model_state = init_model(args)

    processed = 0
    failed = 0

    for img_path in tqdm(image_files, desc="De-identifying"):
        try:
            pil_img = Image.open(img_path).convert("RGB")

            # Resize to target size for faster processing
            if resize_wh != (pil_img.width, pil_img.height):
                pil_img = pil_img.resize(resize_wh, Image.LANCZOS)

            result_pil = deidentify(pil_img, model_state)

            stem = img_path.stem
            out_path = output_dir / f"{stem}.png"
            result_pil.save(out_path, "PNG")
            processed += 1

        except Exception as e:
            print(f"\n✗ Failed on {img_path.name}: {e}")
            failed += 1

    print(f"\nDone. Processed: {processed}, Failed: {failed}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch face de-identification using G²Face (steganographic)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="Input folder with face images")
    parser.add_argument("--output", required=True, help="Output folder for anonymized images (PNG)")
    parser.add_argument("--model_path", default=os.path.join(REPO_ROOT, "weights", "G2Face.pth"),
                        help="Path to G2Face checkpoint. Default: ./weights/G2Face.pth")
    parser.add_argument("--arcface_path", default=os.path.join(REPO_ROOT, "pretrain", "ms1mv3_arcface_r50.pth"),
                        help="Path to ArcFace backbone. Default: ./pretrain/ms1mv3_arcface_r50.pth")
    parser.add_argument("--resize", default="256,256",
                        help="Resize input to W,H before processing. Output is this size. Default: 256,256")
    parser.add_argument("--device", default="cuda",
                        help="Device to use (cuda or cpu). Default: cuda")

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
