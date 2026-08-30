#!/usr/bin/env python
"""Batch face de-identification using AIDPro.

AIDPro (Wang et al., IEEE TIFS 2025): AdaIN-based face protection with
watermark embedding. Generates a different person's face while embedding an
extractable watermark for image authentication. NOT REVERSIBLE.

Pipeline per image:
  Image -> center-crop to 224x224 (model input) -> ArcFace identity embed ->
  random watermark concat -> WatermarkInsert transform -> FFM/AdaIN generator
  (SimSwap-based) -> de-identified 224x224 face -> paste back directly
  onto resized image (256x256) at center crop location (no feathering)

Model requirements (place in specified paths or override with --model-dir):
  models/arcface_checkpoint.tar        — ArcFace iResNet50 identity encoder
  pretrained_models/simswap_90000.pth  — SimSwap FFM generator (AdaIN-based)
  pretrained_models/aidpro_wiwd.pt     — AIDPro WI+WD weights

Usage:
    python deidentify_batch.py --input <source_folder> --output <dest_folder> [options]

Options:
    --model-dir        Directory containing models/ and pretrained_models/.
                       Default: script directory (AIDPro/).
    --len-watermark    Watermark bit length. Default: 30.
    --crop-size        Center crop size before processing. Default: 224 (matches model training).

Notes:
    For aligned face datasets (FRI): resizes input to 256x256, crops center
    224x224, processes through AIDPro, pastes de-identified face back directly
    (no feathering). Output is always 256x256 PNG.
"""

import argparse
import os
import sys

sys.dont_write_bytecode = True
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Stub cv2 — not needed for inference (ArcFace only, no face detection)
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

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms
import torchvision.transforms.functional as F_torch

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


def get_image_files(input_dir: str) -> list:
    files = []
    for ext in SUPPORTED_EXTENSIONS:
        files.extend(Path(input_dir).glob(f"*{ext}"))
        files.extend(Path(input_dir).glob(f"*{ext.upper()}"))
    return sorted(set(files))


def init_model(args):
    """Load ArcFace, FFM generator (SimSwap), and WI+WD networks."""
    model_dir = Path(args.model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. ArcFace identity encoder (loaded as nn.Module, not state_dict)
    print(f"Loading ArcFace from {model_dir / 'models' / 'arcface_checkpoint.tar'} ...")
    net_arc = torch.load(
        str(model_dir / "models" / "arcface_checkpoint.tar"),
        map_location=torch.device("cpu"),
        weights_only=False,
    ).eval().to(device)

    # 2. FFM Generator (SimSwap-based AdaIN face swap)
    print(f"Loading FFM generator from {model_dir / 'pretrained_models' / '90000_net_G.pth'} ...")
    from fs_networks_fix import Generator_Adain_Upsample

    net_g = Generator_Adain_Upsample(
        input_nc=3, output_nc=3, latent_size=512, n_blocks=9
    ).eval().to(device)
    net_g.load_state_dict(
        torch.load(str(model_dir / "pretrained_models" / "90000_net_G.pth"), map_location="cpu", weights_only=False)
    )

    # 3. Watermark Insert + Watermark Decoder networks
    print(f"Loading WI+WD from {model_dir / 'pretrained_models' / 'LPIPS_L1(0.2)_all_loss_id_5_rec_5_wa_10_step_60000.pt'} ...")
    from AIDPro import Watermark_insert, Watermark_decoder

    wi = Watermark_insert(
        input_size=args.len_watermark + 512, hidden_size=512, output_size=512
    ).eval().to(device)
    wd = Watermark_decoder(
        output_size=args.len_watermark, input_size=512, hidden_size=512
    ).eval().to(device)

    wiwd_state = torch.load(
        str(model_dir / "pretrained_models" / "LPIPS_L1(0.2)_all_loss_id_5_rec_5_wa_10_step_60000.pt"), map_location="cpu", weights_only=False
    )
    wi.load_state_dict(wiwd_state["WI"])
    wd.load_state_dict(wiwd_state["WD"])

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    print(f"\nAIDPro loaded on {device}")
    return {
        "net_arc": net_arc,
        "net_g": net_g,
        "wi": wi,
        "wd": wd,
        "device": device,
        "to_tensor": to_tensor,
    }


def deidentify_face(pil_img_crop, model_state, args):
    """De-identify a single 224x224 cropped face via AIDPro.

    Returns PIL.Image at 224x224 with de-identified face.
    """
    device = model_state["device"]
    net_arc = model_state["net_arc"]
    net_g = model_state["net_g"]
    wi = model_state["wi"]
    to_tensor = model_state["to_tensor"]

    img_t = to_tensor(pil_img_crop).unsqueeze(0).to(device)

    with torch.no_grad():
        emb_img = net_arc(F.interpolate(img_t, (112, 112), mode="bicubic"))
        original_id = F.normalize(emb_img, p=2, dim=1)

        # Random watermark (deterministic seed for reproducibility)
        torch.manual_seed(0)
        watermark = torch.zeros(1, args.len_watermark).to(device)

        concatenated_matrix = torch.cat((original_id, watermark), dim=1)
        concatenated_id = wi(concatenated_matrix)
        concatenated_id = F.normalize(concatenated_id, p=2, dim=1)

        img_fake = net_g(img_t, concatenated_id)

    # Denormalize from ImageNet-normalized space → pixel space:
    # output * ImageNet_std + ImageNet_mean (matches original mynorm())
    imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
    imagenet_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)
    img_out = (img_fake * imagenet_std + imagenet_mean).squeeze(0).cpu().clamp(0, 1)
    return F_torch.to_pil_image((img_out.mul(255).byte()))


def paste_face_back(original_img_np, deid_face_np, crop_box):
    """Paste de-identified face back onto original image (no feathering).

    Parameters
    ----------
    original_img_np : np.ndarray [H, W, 3] RGB at working resolution
    deid_face_np : np.ndarray [crop_size, crop_size, 3] RGB de-identified face
    crop_box : tuple (top, left, crop_w, crop_h) — center crop region in original image

    Returns
    -------
    np.ndarray [H, W, 3] RGB with face pasted back
    """
    top, left, crop_w, crop_h = crop_box

    # Resize de-identified face to match crop region if needed
    if deid_face_np.shape[:2] != (crop_h, crop_w):
        deid_pil = Image.fromarray(deid_face_np).resize((crop_w, crop_h), Image.LANCZOS)
        deid_face_np = np.array(deid_pil)

    result = original_img_np.copy()
    result[top:top + crop_h, left:left + crop_w] = deid_face_np
    return result


def run(args):
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_files = get_image_files(str(input_dir))
    if not image_files:
        print(f"No supported images found in {input_dir}")
        return

    model_state = init_model(args)
    crop_size = args.crop_size

    processed = 0
    failed = 0
    for img_path in tqdm(image_files, desc="De-identifying"):
        try:
            pil_img = Image.open(img_path).convert("RGB")

            # Resize to 256x256 so center crop covers the full face
            if pil_img.size != (256, 256):
                pil_img = pil_img.resize((256, 256), Image.LANCZOS)

            orig_w, orig_h = pil_img.size

            # Center crop to model input size (16px margin on each side)
            left = (orig_w - crop_size) // 2
            top = (orig_h - crop_size) // 2
            img_crop = pil_img.crop((left, top, left + crop_size, top + crop_size))

            # De-identify the cropped face
            deid_pil = deidentify_face(img_crop, model_state, args)

            # Paste back onto resized image (no feathering)
            orig_np = np.array(pil_img)
            deid_np = np.array(deid_pil)
            result_np = paste_face_back(orig_np, deid_np, (top, left, crop_size, crop_size))

            stem = img_path.stem
            out_path = output_dir / f"{stem}.png"
            Image.fromarray(result_np, mode="RGB").save(out_path, "PNG")
            processed += 1

        except Exception as e:
            import traceback
            print(f"\nFailed on {img_path.name}: {e}")
            traceback.print_exc()
            failed += 1

    print(f"\nDone. Processed: {processed}, Failed: {failed}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch face de-identification using AIDPro (AdaIN Protection)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="Input folder with aligned face images")
    parser.add_argument("--output", required=True, help="Output folder for de-identified images (PNG)")
    parser.add_argument(
        "--model-dir",
        default=REPO_ROOT,
        help=f"Directory containing models/ and pretrained_models/. Default: {REPO_ROOT}",
    )
    parser.add_argument(
        "--len-watermark",
        type=int,
        default=30,
        help="Watermark bit length. Default: 30 (matches pretrained model)",
    )
    parser.add_argument(
        "--crop-size",
        type=int,
        default=224,
        help="Center crop size for model input. Default: 224 (matches training resolution)",
    )
    parser.add_argument(
        "--feather",
        type=int,
        default=15,
        help=argparse.SUPPRESS,  # Kept for backward compat, no longer used (no feathering)
    )

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
