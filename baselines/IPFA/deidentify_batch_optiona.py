#!/usr/bin/env python
"""Batch face de-identification using IPFA full pipeline (Option A).

IPFA (Li et al., ACM MM 2021): Identity-preserving face anonymization via
facial attributes obfuscation. Full two-stage pipeline following utils/score.py:

  Stage 1 — Identity-aware region discovery:
    ArcFace-CelebA-R50-10177 → Grad-CAM salience map → FaceParser segmentation
    → per-face-part importance scores (mouth/eyebrows/eyes/hair/nose/skin)

  Stage 2 — Adaptive face obfuscation:
    AttributeNet predicts present attributes → smart selection flips only those
    in least identity-critical regions → StarGAN Generator produces output

Pipeline per image:
  Image → Grad-CAM salience + FaceParser segmentation → per-part scores
  → AttributeNet prediction → select safe attributes to flip
  → StarGAN G(real, selected_target) → de-identified output (256x256 PNG)

Model requirements (all in pretrained/):
  - ArcFace-CelebA-R50-10177.pth  (identity network for Grad-CAM)
  - FaceParser.ckpt                (face part segmentation)
  - Face-Attributes2.pth           (attribute prediction)
  - 200000-G.ckpt                  (StarGAN generator)

Usage:
    python deidentify_batch_optiona.py --input <folder> --output <folder>
"""

import argparse
import os
import sys
from pathlib import Path
from collections import OrderedDict

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

# ─── Paths ────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

# ─── StarGAN attributes (c_dim=5) ────────────────────────────────
SELECTED_ATTRS = ['Receding_Hairline', 'Bushy_Eyebrows', 'Narrow_Eyes',
                   'Big_Nose', 'Big_Lips']

# ─── IPFA anonymized attribute definitions (from score.py) ───────
anonymized_hair = ["Receding Hairline"]
anonymized_eyebrows = ["Bushy Eyebrows", "Arched Eyebrows"]
anonymized_eye = ["Brown Eyes"]
anonymized_nose = ["Big Nose", "Pointy Nose"]
anonymized_lips = ["Big Lips"]
anonymized_attribute = (anonymized_hair + anonymized_eyebrows +
                        anonymized_eye + anonymized_nose + anonymized_lips)

# Map from anonymized attribute name → StarGAN c_dim index
ATTR_TO_STARGAN_IDX = {
    "Receding Hairline": 0,
    "Bushy Eyebrows": 1,
    "Arched Eyebrows": 1,   # Both eyebrow attrs → index 1 (Bushy_Eyebrows)
    "Brown Eyes": 2,         # Eye attribute → index 2 (Narrow_Eyes)
    "Big Nose": 3,
    "Pointy Nose": 3,        # Both nose attrs → index 3 (Big_Nose)
    "Big Lips": 4,
}

# ─── Utility functions (from score.py) ──────────────────────────

def mkdir(name):
    """Create folder if not exists."""
    if not os.path.exists(name):
        os.makedirs(name)


def get_last_conv_name(net):
    """Get the name of last convolutional layer."""
    layer_name = None
    for name, m in net.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            layer_name = name
    return layer_name


def path_image_preprocessing(image_path, mode="arcface"):
    """Preprocessing method — from original score.py.

    Args:
        image_path: Image path
        mode: "arcface" (112x112 normalize 0.5/0.5) or
              "attribute" (224x224 BGR mean subtraction)
    """
    if mode == "arcface":
        data = Image.open(image_path).convert("RGB")
        data = arcface_transform(data)
        data = torch.unsqueeze(data, 0)
    elif mode == "attribute":
        mean_bgr = np.array([91.4953, 103.8827, 131.0912])
        image = cv2.imread(image_path)
        assert image is not None, f"Cannot read {image_path}"
        image = cv2.resize(image, (224, 224))
        image = image.astype(np.float32)
        image -= mean_bgr
        image = image.transpose(2, 0, 1)
        data = torch.tensor(image[np.newaxis, ...], dtype=torch.float32)
    return data


def calculate_score(salience_map, segmentation):
    """The average response values of different parts — from score.py."""
    score = (segmentation * salience_map).sum() / (segmentation.sum() + 1e-10)
    return score


def sort_face_part_score(face_part_score):
    """Sort parts by score descending — from score.py."""
    tmp = sorted(face_part_score.items(),
                 key=lambda kv: (kv[1], kv[0]), reverse=True)
    ordered = {}
    part_sort = []
    for item in tmp:
        ordered[item[0]] = item[1]
        part_sort.append(item[0])
    return ordered, part_sort


def return_list(part):
    """Map part name → anonymized attributes — from score.py."""
    mapping = {
        "hair": anonymized_hair,
        "eyebrows": anonymized_eyebrows,
        "eyes": anonymized_eye,
        "nose": anonymized_nose,
        "lips": anonymized_lips,
    }
    return mapping.get(part, [])


def check_activated(face_part_score, part_sort, activated_attr_name):
    """Assign each activated attr its part's score — from score.py."""
    activated_attribute = {}
    for part in part_sort:
        part_list = return_list(part)
        for attr_name in activated_attr_name:
            if attr_name in part_list:
                activated_attribute[attr_name] = face_part_score[part]
    return activated_attribute


# ─── ArcFace transform ──────────────────────────────────────────
arcface_transform = T.Compose([
    T.Resize((112, 112)),
    T.ToTensor(),
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# ─── StarGAN Generator (from model.py) ──────────────────────────

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


class Generator(torch.nn.Module):
    """StarGAN Generator from model.py."""
    def __init__(self, conv_dim=64, c_dim=5, repeat_num=6):
        super().__init__()
        layers = [
            torch.nn.Conv2d(3 + c_dim, conv_dim, 7, 1, 3, bias=False),
            torch.nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True),
            torch.nn.ReLU(inplace=True),
        ]
        curr = conv_dim
        for _ in range(2):
            layers.extend([
                torch.nn.Conv2d(curr, curr * 2, 4, 2, 1, bias=False),
                torch.nn.InstanceNorm2d(curr * 2, affine=True, track_running_stats=True),
                torch.nn.ReLU(inplace=True),
            ])
            curr *= 2
        for _ in range(repeat_num):
            layers.append(ResidualBlock(curr, curr))
        for _ in range(2):
            layers.extend([
                torch.nn.ConvTranspose2d(curr, curr // 2, 4, 2, 1, bias=False),
                torch.nn.InstanceNorm2d(curr // 2, affine=True, track_running_stats=True),
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


# ─── Model loading ──────────────────────────────────────────────

def load_arcface(pretrained_path, device):
    """Load ArcFace-CelebA-R50-10177 identity network."""
    sys.path.insert(0, str(REPO_ROOT))
    from models.iresnet import iresnet50

    net = iresnet50(people_num=10177)
    model_dict = net.state_dict()
    pretrained_raw = torch.load(pretrained_path, map_location="cpu",
                                weights_only=False)
    if hasattr(pretrained_raw, 'state_dict'):
        pretrained = pretrained_raw.state_dict()
    else:
        pretrained = pretrained_raw

    new_state_dict = OrderedDict()
    for k, v in pretrained.items():
        if k in model_dict:
            new_state_dict[k] = v
        elif k[7:] in model_dict:
            new_state_dict[k[7:]] = v
    model_dict.update(new_state_dict)
    net.load_state_dict(model_dict)
    net.to(device)
    net.eval()
    return net


def load_faceparser(pretrained_path, device):
    """Load FaceParser segmentation network."""
    sys.path.insert(0, str(REPO_ROOT))
    from models.face_parser import FaceParser, read_img

    net = FaceParser(num_classes=9, model_path=pretrained_path)
    net.to(device)
    net.eval()
    # Expose read_img for use in the pipeline
    global _fp_read_img
    _fp_read_img = read_img
    return net


def load_attributenet(pretrained_path, device):
    """Load AttributeNet with anonymized attributes."""
    sys.path.insert(0, str(REPO_ROOT))
    from models.AttributeNet import AttributeNet

    net = AttributeNet(pretrained=pretrained_path)
    net.set_idx_list(attribute=anonymized_attribute)
    net.model.eval()
    net.to(device)
    return net


def load_staragan(pretrained_path, device):
    """Load StarGAN generator."""
    G = Generator(conv_dim=64, c_dim=5, repeat_num=6).to(device)
    state_dict = torch.load(pretrained_path, map_location=device,
                            weights_only=False)
    G.load_state_dict(state_dict)
    G.eval()
    return G


# ─── Grad-CAM (from interpretability/grad_cam.py) ───────────────

class GradCAM:
    """GradCAM for single-output networks — from original grad_cam.py."""
    def __init__(self, net, layer_name):
        self.net = net
        self.net.eval()
        self.layer_name = layer_name
        self.feature = None
        self.gradient = None

    def _get_features_hook(self, module, input, output):
        self.feature = output

    def _get_grads_hook(self, module, input_grad, output_grad):
        self.gradient = output_grad[0]

    def _register_hook(self):
        self.handlers = []
        for name, module in self.net.named_modules():
            if name == self.layer_name:
                self.handlers.append(
                    module.register_forward_hook(self._get_features_hook))
                self.handlers.append(
                    module.register_full_backward_hook(self._get_grads_hook))

    def remove_handlers(self):
        for handle in self.handlers:
            handle.remove()

    def __call__(self, inputs, index=None):
        """Return (cam[H,W], class_id, score)."""
        return self.get_heatmap(inputs, index)

    def get_heatmap(self, inputs, index=None):
        shape = inputs.shape[-2:]
        self._register_hook()
        self.net.zero_grad()

        output = self.net(inputs)  # [1, 10177] cosine logits
        if index is None:
            index = torch.argmax(output)

        scores = torch.softmax(output, dim=1)[0, index].item()
        class_id = index.item()

        target = output[0, index]
        target.backward()

        gradient = self.gradient[0]  # [C, H, W]
        weight = torch.mean(gradient, axis=(1, 2))  # [C]
        feature = self.feature[0]  # [C, H, W]

        cam = feature * weight[:, np.newaxis, np.newaxis]  # [C,H,W]
        cam = torch.sum(cam, axis=0)  # [H,W]
        cam = torch.relu(cam)

        # Normalization
        cam -= torch.min(cam)
        if torch.max(cam) != 0:
            cam /= torch.max(cam)

        # Resize to input shape
        cam = cv2.resize(cam.cpu().data.numpy(), shape)

        self.remove_handlers()
        return cam, class_id, scores


# ─── StarGAN preprocessing (256x256 input) ──────────────────────

stargan_transform_crop = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
])


def denorm(x):
    """Convert [-1, 1] → [0, 1]."""
    out = (x + 1) / 2
    return out.clamp_(0, 1)


# ─── AIDPro-style center-crop / paste-back ──────────────────────
# StarGAN was trained on tight CelebA face crops. When given images with
# background (FRI faces downsized to 256x256 have dark margins), it distorts
# colors in non-face regions → yellow/blue tint. Solution: only modify the
# central face region, preserve original border pixels exactly as-is.

FACE_CROP_SIZE = 224   # Central crop for StarGAN (matches AIDPro)


def prepare_stargan_input(pil_img_256):
    """Resize to 256x256, center-crop FACE_CROP_SIZE region.

    Returns (crop_pil, canvas_np, crop_pos).
    - crop_pil: FACE_CROP_SIZE×FACE_CROP_SIZE PIL image for StarGAN
    - canvas_np: 256×256 numpy array of the resized original (for paste-back)
    - crop_pos: (top, left) of the crop region
    """
    w, h = pil_img_256.size
    left = (w - FACE_CROP_SIZE) // 2
    top = (h - FACE_CROP_SIZE) // 2
    crop = pil_img_256.crop((left, top, left + FACE_CROP_SIZE,
                              top + FACE_CROP_SIZE))
    canvas = np.array(pil_img_256.convert("RGB"))
    return crop, canvas, (top, left)


def paste_face_back(canvas_np, face_out_np, face_size, crop_pos):
    """Paste processed face back onto 256x256 canvas."""
    top, left = crop_pos
    result = canvas_np.copy()

    if face_out_np.shape[:2] != (face_size, face_size):
        # Resize output to match crop region
        import cv2
        face_out_np = cv2.resize(face_out_np.astype(np.float32),
                                 (face_size, face_size)).astype(np.uint8)

    result[top:top+face_size, left:left+face_size] = face_out_np
    return result


# ─── AIDPro-style crop/paste helpers ─────────────────────────────

def resize_to_256(img: Image.Image) -> Image.Image:
    """Resize any image to 256x256 (matching AIDPro approach)."""
    if img.size != (256, 256):
        return img.resize((256, 256), Image.LANCZOS)
    return img


def center_crop(img: Image.Image, crop_size: int) -> tuple:
    """Center-crop from a 256x256 image.

    Returns (cropped_img, (top, left)) for paste-back.
    """
    w, h = img.size
    left = (w - crop_size) // 2
    top = (h - crop_size) // 2
    return img.crop((left, top, left + crop_size, top + crop_size)), (top, left)


def paste_back(result_256: np.ndarray, processed_face: np.ndarray,
               face_size: int, crop_pos: tuple) -> np.ndarray:
    """Paste processed face back into 256x256 canvas at crop_pos."""
    top, left = crop_pos
    result_256[top:top+face_size, left:left+face_size] = processed_face
    return result_256


def visualize_gradcam(image_path, salience_map):
    """Create Grad-CAM heatmap overlay (from score.py)."""
    img_cv = cv2.imread(image_path)
    img_cv = cv2.resize(img_cv, (salience_map.shape[1], salience_map.shape[0]))

    mask = salience_map.copy()
    mask -= np.max(np.min(mask), 0)
    mask /= np.max(mask)
    mask *= 255.
    masks = np.uint8(mask)
    heatmap = cv2.applyColorMap(masks, cv2.COLORMAP_JET)
    heatmap = cv2.resize(np.float32(heatmap), (img_cv.shape[1], img_cv.shape[0]))
    return (0.4 * heatmap + 0.6 * np.float32(img_cv)).astype(np.uint8)


def visualize_parsedmask(seg_image, image_path):
    """Create colored segmentation mask visualization."""
    # FaceParser class colors: bg=black, mouth=red, eyebrows=green,
    # eyes=blue, hair=yellow, nose=cyan, skin=magenta, ears=white, belowface=gray
    colors = [
        (0, 0, 0),         # 0: background
        (255, 0, 0),       # 1: mouth
        (0, 255, 0),       # 2: eyebrows
        (0, 0, 255),       # 3: eyes
        (255, 255, 0),     # 4: hair
        (0, 255, 255),     # 5: nose
        (255, 0, 255),     # 6: skin
        (255, 255, 255),   # 7: ears
        (128, 128, 128),   # 8: belowface
    ]
    h, w = seg_image.shape[1], seg_image[0].shape[0]
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    for idx in range(seg_image.shape[0]):
        mask = (seg_image[idx] > 0.5).astype(np.uint8)
        if mask.sum() > 0:
            vis[mask > 0] = colors[idx][:3]
    return vis


# ─── Main pipeline ──────────────────────────────────────────────

def get_image_files(input_dir):
    files = []
    for ext in SUPPORTED_EXTENSIONS:
        files.extend(Path(input_dir).glob(f"*{ext}"))
        files.extend(Path(input_dir).glob(f"*{ext.upper()}"))
    return sorted(set(files))


def compute_smart_target(attr_net, seg_net, cam_wrapper, image_path, device,
                          img_256=None):
    """Stage 1 + smart selection for one image.

    Args:
        attr_net: AttributeNet model
        seg_net: FaceParser model
        cam_wrapper: GradCAM wrapper
        image_path: path to original image (for cv2.imread / read_img)
        device: torch device
        img_256: pre-loaded 256x256 PIL image for Grad-CAM / AttributeNet

    Returns:
        c_org, c_trg, info dict with scores + debug images
    """
    # --- Grad-CAM salience map (from score.py) ---
    x_arcface = path_image_preprocessing(image_path, "arcface").to(device)
    salience_map, class_id, scores = cam_wrapper(x_arcface)

    # Resize to segmentation size (512x512) — matches score.py line 433
    salience_resized = cv2.resize(salience_map, (512, 512))

    # --- FaceParser segmentation (using original read_img → 1024x1024) ---
    x_seg = _fp_read_img(image_path).to(device)
    with torch.no_grad():
        parsed_face = seg_net(x_seg)

    seg_image = parsed_face.cpu().numpy()[0]
    seg_image = (seg_image > 0.5).astype(np.uint8)

    mouth = seg_image[1]; eyebrows = seg_image[2]; eyes = seg_image[3]
    hair = seg_image[4]; nose = seg_image[5]

    # --- Per-part importance scores ---
    mouth_score = calculate_score(salience_resized, mouth)
    eyebrows_score = calculate_score(salience_resized, eyebrows)
    eyes_score = calculate_score(salience_resized, eyes)
    hair_score = calculate_score(salience_resized, hair)
    nose_score = calculate_score(salience_resized, nose)

    face_part_score = {
        "lips": mouth_score,      # Original maps mouth → lips
        "eyebrows": eyebrows_score,
        "eyes": eyes_score,
        "hair": hair_score,
        "nose": nose_score,
    }
    face_part_score, part_sort = sort_face_part_score(face_part_score)

    # --- AttributeNet prediction (from score.py) ---
    x_attr = path_image_preprocessing(image_path, "attribute").to(device)
    with torch.no_grad():
        predicted_attr = attr_net(x_attr)
    selected_attr_ = (predicted_attr[0] > 0.5).cpu().numpy()

    # Build c_org for StarGAN c_dim=5
    # AttributeNet output order: [Receding Hairline, Bushy Eyebrows, Arched
    #   Eyebrows, Brown Eyes, Big Nose, Pointy Nose, Big Lips]
    # StarGAN c_dim:             [Receding_Hairline, Bushy_Eyebrows, Narrow_
    #   Eyes, Big_Nose, Big_Lips]
    c_org = torch.zeros(5, device=device)
    for attr_name, idx in ATTR_TO_STARGAN_IDX.items():
        attr_idx_in_list = anonymized_attribute.index(attr_name)
        if selected_attr_[attr_idx_in_list]:
            c_org[idx] = 1.0

    # --- Smart selection: activated attrs → part scores ────
    activated_names = np.array(anonymized_attribute)[
        selected_attr_.reshape(len(selected_attr_))]

    info = {
        "part_scores": dict(face_part_score),
        "part_sort": list(part_sort),
        "predicted_attrs": {},
        "activated_with_scores": {},
        "salience_map": salience_map,          # For debug visualization
        "seg_image": seg_image,                # For debug visualization
    }

    for i, name in enumerate(anonymized_attribute):
        info["predicted_attrs"][name] = float(
            predicted_attr[0][i].cpu().item())

    if len(activated_names) > 0:
        activated_scores = check_activated(face_part_score, part_sort,
                                           activated_names)
        info["activated_with_scores"] = activated_scores

        # IPFA strategy: flip attributes in LEAST important regions
        safe_to_modify = sorted(activated_scores.items(), key=lambda x: x[1])

        flip_indices = set()
        for attr_name, _ in safe_to_modify:
            if attr_name in ATTR_TO_STARGAN_IDX:
                flip_indices.add(ATTR_TO_STARGAN_IDX[attr_name])

        c_trg = c_org.clone()
        for idx in flip_indices:
            c_trg[idx] = 1.0 - c_org[idx]

        info["flip_attributes"] = [a for a, _ in safe_to_modify]
        info["flip_indices"] = sorted(flip_indices)
    else:
        # Fallback: no attrs detected → try all single flips downstream
        c_trg = None
        info["flip_attributes"] = []
        info["flip_indices"] = []

    return c_org, c_trg, info


def main():
    parser = argparse.ArgumentParser(
        description="IPFA Option A: Full pipeline de-identification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", required=True,
                        help="Input folder with aligned face images")
    parser.add_argument("--output", required=True,
                        help="Output folder for de-identified images (PNG)")
    parser.add_argument("--debug", action="store_true",
                        help="Save Grad-CAM and FaceParser debug visualizations")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_files = get_image_files(str(input_dir))
    if not image_files:
        print(f"No supported images found in {input_dir}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Images: {len(image_files)}")

    # ── Load models ────────────────────────────────────────────
    print("\nLoading ArcFace-CelebA-R50-10177...")
    recognition_net = load_arcface(
        REPO_ROOT / "pretrained" / "ArcFace-CelebA-R50-10177.pth", device)
    layer_name = get_last_conv_name(recognition_net)
    print(f"  Last conv: {layer_name}")

    print("Loading FaceParser...")
    seg_net = load_faceparser(
        REPO_ROOT / "pretrained" / "FaceParser.ckpt", device)

    print("Loading AttributeNet...")
    attr_net = load_attributenet(
        REPO_ROOT / "pretrained" / "Face-Attributes2.pth", device)

    print("Loading StarGAN Generator...")
    G = load_staragan(REPO_ROOT / "pretrained" / "200000-G.ckpt", device)
    print(f"  Params: {sum(p.numel() for p in G.parameters()):,}")

    # Wrap Grad-CAM
    cam = GradCAM(recognition_net, layer_name)

    # ── Process images ─────────────────────────────────────────
    processed = 0
    failed = 0

    for image_path in tqdm(image_files, desc="De-identifying"):
        try:
            stem = Path(image_path).stem
            orig_image = Image.open(image_path).convert("RGB")

            # Stage 1 + smart selection
            c_org, c_trg, info = compute_smart_target(
                attr_net, seg_net, cam, str(image_path), device)

            # ── Debug: save Grad-CAM overlay & parsed mask ──
            if args.debug:
                orig_cv = cv2.imread(str(image_path))
                sal_map = info["salience_map"]
                sal_resized_for_orig = cv2.resize(sal_map,
                    (orig_cv.shape[1], orig_cv.shape[0]))
                gradcam_vis = visualize_gradcam(str(image_path),
                    sal_resized_for_orig)
                cv2.imwrite(str(output_dir / f"{stem}_gradcam.jpg"),
                            cv2.cvtColor(gradcam_vis, cv2.COLOR_BGR2RGB))

                seg_vis = visualize_parsedmask(info["seg_image"],
                    str(image_path))
                # Resize seg vis to match original for saving
                h_orig, w_orig = orig_cv.shape[0], orig_cv.shape[1]
                seg_vis_resized = cv2.resize(seg_vis, (w_orig, h_orig))
                cv2.imwrite(str(output_dir / f"{stem}_parsedmask.jpg"),
                            cv2.cvtColor(seg_vis_resized, cv2.COLOR_RGB2BGR))

            # Stage 2: StarGAN generation (center-crop/paste-back)
            img_256 = resize_to_256(orig_image)
            crop_pil, canvas_np, crop_pos = prepare_stargan_input(img_256)

            x_real = stargan_transform_crop(crop_pil).unsqueeze(0).to(device)

            with torch.no_grad():
                if c_trg is not None:
                    x_fake = G(x_real, c_trg.unsqueeze(0))  # [1,3,FACE,H,W]
                else:
                    # Fallback: try all single-attribute flips, pick best diff
                    best_out = None
                    best_diff = -1.0
                    for i in range(5):
                        c_try = c_org.clone()
                        c_try[i] = 1.0 - c_org[i]
                        x_try = G(x_real, c_try.unsqueeze(0))
                        diff = float(torch.abs(x_real - x_try).mean().item())
                        if diff > best_diff:
                            best_diff = diff
                            best_out = x_try
                    x_fake = best_out  # Already [1,3,H,W] from G()

                denormed = denorm(x_fake.squeeze(0))  # [3,H,W]

            # Safety check: ensure output is [3, H, W]
            if denormed.dim() != 3 or denormed.shape[0] != 3:
                raise ValueError(f"Unexpected StarGAN output shape: "
                                 f"{x_fake.shape} (after squeeze: {denormed.shape}) for {stem}")

            face_out_np = (denormed.cpu().numpy()
                           .transpose(1, 2, 0) * 255).astype(np.uint8)

            # Paste processed face back onto original canvas
            result_np = paste_face_back(canvas_np, face_out_np,
                                         FACE_CROP_SIZE, crop_pos)
            out_img = Image.fromarray(result_np, "RGB")

            dst_path = output_dir / f"{stem}.png"
            out_img.save(str(dst_path), "PNG")
            processed += 1

        except Exception as e:
            import traceback
            print(f"\nFailed on {image_path.name}: {e}")
            traceback.print_exc()
            failed += 1

    print(f"\nDone. Processed: {processed}, Failed: {failed}")


if __name__ == "__main__":
    main()
