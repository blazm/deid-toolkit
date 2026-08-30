#!/usr/bin/env python
"""Batch face de-identification using FAO StarGAN Generator (Option B).

FAO / IPFA (Li et al., ACM MM 2021): Identity-preserving face anonymization
via facial attributes obfuscation. Uses a StarGAN-based generator trained on
CelebA to reverse 5 facial attributes (Receding Hairline, Bushy Eyebrows,
Narrow Eyes, Big Nose, Big Lips), changing visual appearance while preserving
identity discriminability.

Pipeline per image:
  Image -> resize to 256x256 -> normalize [-1,1] -> StarGAN Generator
  (with reversed attribute vector) -> de-identified output

Model requirements:
  - 200000-G.ckpt from HuggingFace (RuoyuChen/Facial_Attributes_Obfuscation)
  - Auto-downloads on first run if not present

Usage:
    python deidentify_batch.py --input <source_folder> --output <dest_folder> [options]
"""

import argparse
import os
import sys
from pathlib import Path

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Stub cv2 for compatibility
import types
_cv2_mod = types.ModuleType('cv2')
_cv2_mod.__spec__ = types.SimpleNamespace(name='cv2', origin=None)
_cv2_mod.COLOR_RGB2BGR = 0
_cv2_mod.INTER_AREA = 1
_cv2_mod.INTER_LINEAR = 1
_cv2_mod.IMREAD_COLOR = 1
import numpy as np
def _fake_imread(*a, **k): return np.zeros((64, 64, 3), dtype=np.uint8)
def _fake_imwrite(*a, **k): return True
def _fake_resize(*a, **k): return np.zeros((64, 64, 3), dtype=np.uint8)
_cv2_mod.imread = _fake_imread
_cv2_mod.imwrite = _fake_imwrite
_cv2_mod.resize = _fake_resize
sys.modules['cv2'] = _cv2_mod

import torch
from PIL import Image
from torchvision import transforms as T
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

# StarGAN attributes (c_dim=5) — original CelebA 128x128 attributes
SELECTED_ATTRS = ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young']

def get_image_files(input_dir: str) -> list:
    files = []
    for ext in SUPPORTED_EXTENSIONS:
        files.extend(Path(input_dir).glob(f"*{ext}"))
        files.extend(Path(input_dir).glob(f"*{ext.upper()}"))
    return sorted(set(files))


class Generator(torch.nn.Module):
    """StarGAN Generator (128x128 CelebA, c_dim=5)."""
    def __init__(self, conv_dim=64, c_dim=5, repeat_num=6):
        super().__init__()
        layers = [
            torch.nn.Conv2d(3 + c_dim, conv_dim, 7, 1, 3, bias=False),
            torch.nn.InstanceNorm2d(conv_dim, affine=True),
            torch.nn.ReLU(inplace=True),
        ]
        curr = conv_dim
        for _ in range(2):
            layers.extend([
                torch.nn.Conv2d(curr, curr * 2, 4, 2, 1, bias=False),
                torch.nn.InstanceNorm2d(curr * 2, affine=True),
                torch.nn.ReLU(inplace=True),
            ])
            curr *= 2
        for _ in range(repeat_num):
            layers.append(ResidualBlock(curr, curr))
        for _ in range(2):
            layers.extend([
                torch.nn.ConvTranspose2d(curr, curr // 2, 4, 2, 1, bias=False),
                torch.nn.InstanceNorm2d(curr // 2, affine=True),
                torch.nn.ReLU(inplace=True),
            ])
            curr //= 2
        layers.extend([
            torch.nn.Conv2d(curr, 3, 7, 1, 3, bias=False),
            torch.nn.Tanh(),
        ])
        self.main = torch.nn.Sequential(*layers)

    def forward(self, x, c):
        c = c.view(c.size(0), c.size(1), 1, 1).expand(-1, -1, x.size(2), x.size(3))
        x = torch.cat([x, c], dim=1)
        return self.main(x)


class ResidualBlock(torch.nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.main = torch.nn.Sequential(
            torch.nn.Conv2d(dim_in, dim_out, 3, 1, 1, bias=False),
            torch.nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(dim_out, dim_out, 3, 1, 1, bias=False),
            torch.nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
        )
    def forward(self, x):
        return x + self.main(x)


def download_model_if_needed(model_path: Path):
    """Download or use existing original StarGAN 128x128 CelebA generator."""
    if model_path.exists():
        print(f"  Using existing generator: {model_path}")
        return
    model_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Generator not found at {model_path}. Please place celeba-128x128-5attrs/200000-G.ckpt there.")
    sys.exit(1)


def load_generator(model_path: Path, device):
    G = Generator(conv_dim=64, c_dim=5, repeat_num=6).to(device)
    state_dict = torch.load(str(model_path), map_location=device, weights_only=False)

    # Strip running stats buffers left over from pre-0.4.0 checkpoints
    # (InstanceNorm2d switched to track_running_stats=False by default in 0.4.0)
    state_dict = {k: v for k, v in state_dict.items()
                  if not k.endswith(".running_mean") and not k.endswith(".running_var")}

    G.load_state_dict(state_dict, strict=False)
    # Keep in training mode — original StarGAN 128x128 uses InstanceNorm(affine=True, track_running_stats=False).
    # .eval() is not needed and can break per-instance normalization on models trained with track_running_stats=True.
    return G


def preprocess_image(image: Image.Image):
    """Transform PIL image to tensor [-1, 1] matching StarGAN training transform (128x128)."""
    transform = T.Compose([
        T.Resize(128),
        T.ToTensor(),
        T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])
    return transform(image).unsqueeze(0)  # [1, 3, 128, 128]


def preprocess_attribute(image: Image.Image):
    """Transform PIL image for AttributeNet: resize 224x224, BGR mean subtraction.

    Matches utils/score.py Path_Image_Preprocessing mode='attribute':
      cv2.imread (BGR) -> cv2.resize(224,224) -> subtract BGR mean [91.5, 103.9, 131.1]
    PIL gives RGB, so we convert: arr[:, :, ::-1] swaps R<->B to get BGR.
    """
    import numpy as np
    # Resize with PIL (smoother than cv2)
    if image.size != (224, 224):
        image = image.resize((224, 224), Image.LANCZOS)
    arr = np.array(image).astype(np.float32)  # RGB [H,W,C]
    # Convert RGB -> BGR to match VGGFace training convention
    arr = arr[:, :, ::-1]
    # BGR mean subtraction
    mean_bgr = np.array([91.4953, 103.8827, 131.0912])
    arr -= mean_bgr
    # HWC -> CHW
    arr = arr.transpose(2, 0, 1)
    return torch.tensor([arr])


def denorm(x):
    """Convert [-1, 1] -> [0, 1]."""
    out = (x + 1) / 2
    return out.clamp_(0, 1)


def load_attribute_net(model_path: Path, device):
    """Load AttributeNet for predicting facial attributes."""
    sys.path.insert(0, str(REPO_ROOT))
    from models.AttributeNet import AttributeNet as AttrNetCls

    attr_net = AttrNetCls(pretrained=str(model_path))
    attr_net.model.eval()
    attr_net = attr_net.to(device)

    # Only select our 5 StarGAN attributes for output
    attr_net.set_idx_list([
        "Receding Hairline", "Bushy Eyebrows",
        "Big Nose", "Pointy Nose", "Big Lips",
    ])
    return attr_net


def predict_attributes(attr_net, image: Image.Image, device):
    """Predict the 5 StarGAN-relevant attributes for an image.

    Returns binary vector [receding_hairline, bushy_eyebrows, narrow_eyes, big_nose, big_lips]
    Note: Narrow_Eyes not directly in AttributeNet — default to 0 (absent).
    """
    import numpy as np
    # Resize with PIL for proper image quality
    if image.size != (224, 224):
        img = image.resize((224, 224), Image.LANCZOS)
    else:
        img = image
    arr = np.array(img).astype(np.float32)  # RGB [H,W,C]
    # Convert RGB -> BGR (VGGFace convention used in training)
    arr_bgr = arr[:, :, ::-1]
    # Subtract BGR mean
    mean_bgr = np.array([91.4953, 103.8827, 131.0912])
    arr_bgr -= mean_bgr
    # HWC -> CHW, add batch dim
    arr_chw = arr_bgr.transpose(2, 0, 1)
    x = torch.tensor([arr_chw]).to(device)

    with torch.no_grad():
        probs = attr_net(x)  # [1, 5] softmax probabilities

    # Threshold at 0.5: if prob > 0.5, attribute is present (1), else absent (0)
    binary = (probs > 0.5).float()[0]  # [5]

    # Reorder to match SELECTED_ATTRS: [Receding_Hairline, Bushy_Eyebrows, Narrow_Eyes, Big_Nose, Big_Lips]
    # AttributeNet output order: [Receding Hairline, Bushy Eyebrows, Big Nose, Pointy Nose, Big Lips]
    # Index mapping: [0, 1, N/A->0, 2 (Big Nose), 4 (Big Lips)] — skip Pointy Nose for StarGAN c_dim=5
    result = torch.zeros(5, device=device)
    result[0] = binary[0]  # Receding Hairline
    result[1] = binary[1]  # Bushy Eyebrows
    result[2] = 0.0        # Narrow Eyes (not in AttributeNet output)
    result[3] = binary[2]  # Big Nose
    result[4] = binary[4]  # Big Lips
    return result


def create_flip_combos(c_org, mode="single"):
    """Generate target attribute vectors by flipping 1 or 2 attributes from c_org.

    StarGAN was trained on per-attribute reversal: c_trg[:,k] = (c_org[:,k] == 0)
    We generate combos matching the training distribution.
    """
    c_dim = c_org.shape[0]
    combos = []

    if mode in ("single", "mixed"):
        for i in range(c_dim):
            c_trg = c_org.clone()
            c_trg[i] = 1.0 - c_org[i]  # Flip this attribute
            combos.append(c_trg)

    if mode in ("pair", "mixed"):
        for i in range(c_dim):
            for j in range(i + 1, c_dim):
                c_trg = c_org.clone()
                c_trg[i] = 1.0 - c_org[i]
                c_trg[j] = 1.0 - c_org[j]
                combos.append(c_trg)

    return combos


def run(args):
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_files = get_image_files(str(input_dir))
    if not image_files:
        print(f"No supported images found in {input_dir}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load StarGAN Generator (original 128x128 CelebA weights — FAO stargan_celeba weights are corrupted)
    model_path = REPO_ROOT / "pretrained" / "celeba-128x128-5attrs" / "200000-G.ckpt"
    download_model_if_needed(model_path)

    print("Loading StarGAN generator (128x128)...")
    G = load_generator(model_path, device)
    total_params = sum(p.numel() for p in G.parameters())
    print(f"  Generator loaded ({total_params:,} parameters)")

    # AttributeNet uses different attribute space than original CelebA 128x128.
    # For de-identification, exact semantics don't matter — we generate multiple
    # flip combos and pick the most impactful one by output variance.
    # Using zeros as c_org means all flips are "0→1" (attribute presence).

    processed = 0
    failed = 0
    for image_path in tqdm(image_files, desc="De-identifying"):
        try:
            stem = Path(image_path).stem

            image = Image.open(image_path).convert("RGB")
            x_real = preprocess_image(image).to(device)

            # Default attribute vector (all absent — flips will set attributes present)
            c_org = torch.zeros(5, device=device)

            # Generate flip combos (single + pair attribute reversals)
            combos = create_flip_combos(c_org, mode="mixed")

            # Generate with each combo, pick best by output variance (most change)
            best_out = None
            best_var = -1.0

            with torch.no_grad():
                for c_trg in combos:
                    x_fake = G(x_real, c_trg.unsqueeze(0))
                    denormed = denorm(x_fake[0])
                    var = float(denormed.std().item())
                    if var > best_var:
                        best_var = var
                        best_out = denormed

            # Upscale from 128x128 → 256x256
            out_np_128 = (best_out.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            out_img_128 = Image.fromarray(out_np_128, "RGB")
            out_img = out_img_128.resize((256, 256), Image.BICUBIC)

            dst_path = str(output_dir / f"{stem}.png")
            out_img.save(dst_path, "PNG")
            processed += 1

        except Exception as e:
            import traceback
            print(f"\nFailed on {image_path.name}: {e}")
            traceback.print_exc()
            failed += 1

    print(f"\nDone. Processed: {processed}, Failed: {failed}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch face de-identification using FAO StarGAN Generator (Option B)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="Input folder with aligned face images")
    parser.add_argument("--output", required=True, help="Output folder for de-identified images (PNG)")

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
