"""Batch face de-identification using iFADIT (Invertible Face Anonymization via
Disentangled Identity Transformation, Pattern Recognition 2025).

NOTE ON THE RELEASED CODE: the upstream test.py is a partially broken dump
(hardcoded /data/lk/... paths, never-loaded FCN weights, a StyleGANv2 loader that
ignores its argument). This wrapper implements the paper's anonymization flow with
all weights loaded from ./pretrained_drive (Google Drive checkpoint folder):

    model_ir_se50.pth        IResNet50 (ArcFace-style) identity encoder
    attr_best.pth            pSp attribute encoder (encoder + 256 decoder + mapping)
    G_net_ffhq15INN*.pth     secure identity-transform INN (the anonymization core)
    mlp_best.pth             latent MLP (14x1024 -> w++)
    fuse_mlp_*_INN*.pth      FusionMapper10 (backward / recovery path only)
    550000.pt                StyleGAN2 (128 px) generator used both for passwords
                             and final synthesis (g_ema); its loader in
                             models/deghosting/stylegan2.py is patched to this path
    deghosting_weight_*.pth  deghosting FCN (128 -> 256 super-resolution)

Protocol (paper, "with passwords"):
    img256 [0,1] -> pSp W + ArcFace id -> INN(id, c=password) -> concat with W
    -> MLP -> w++ -> StyleGAN2(128) -> deghosting(256) -> masked composite back onto
    the original 256 crop -> output (single save, --out-size 256 by default).

Usage (env `ifadit`):
    python deidentify_batch_ifadit.py --input <aligned_face_dir> --output <out_dir>
        [--batch-size 4] [--seed 0]
"""
import argparse
import contextlib
import io
import math
import os
import sys
import time
import types
from argparse import Namespace
from pathlib import Path

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "models", "deghosting"))  # for `from stylegan2 import ...`
sys.path.insert(0, os.path.join(REPO, "models", "attr_encoder"))  # psp.py does `from train_options import ...`

DRIVE = os.path.join(REPO, "pretrained_drive")
W = {
    "id": os.path.join(DRIVE, "model_ir_se50.pth"),
    "psp": os.path.join(DRIVE, "attr_best.pth"),
    "inn": os.path.join(DRIVE, "G_net_ffhq15INN607_20_202528.pth"),
    "mlp": os.path.join(DRIVE, "mlp_best.pth"),
    "icl": os.path.join(DRIVE, "fuse_mlp_ffhq15INN607_20_202528.pth"),
    "stylegan128": os.path.join(DRIVE, "550000.pt"),
    "deghost": os.path.join(DRIVE, "deghosting_weight_1_140.pth"),
}
for k, v in W.items():
    if not os.path.isfile(v):
        raise FileNotFoundError(f"missing iFADIT weight: {v}")

# ── mmcv stub: only import-time symbols + trivial helpers are used by the vendored
#    deghosting code (the inference-relevant FCN is pure Conv2d). ──
try:
    import mmcv  # noqa: F401
except ImportError:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    def _xavier_init(module, *a, **k):
        for p in module.parameters():
            if p.dim() >= 2:
                nn.init.xavier_uniform_(p)
        return module

    def _normal_init(module_or_tensor, mean=0.0, std=1.0, a=0.0, b=1.0, mode='fan_in'):
        if isinstance(module_or_tensor, nn.Module):
            for p in module_or_tensor.parameters():
                nn.init.normal_(p, mean, std)
        else:
            nn.init.normal_(module_or_tensor, mean, std)

    def _is_seq_of(seq, type_or_classes, seq_type=None):
        return (isinstance(seq, (list, tuple))
                and len(seq) > 0
                and all(isinstance(s, type_or_classes) for s in seq))

    def _build_activation_layer(act_cfg):
        if act_cfg is None:
            return nn.Identity()
        t = act_cfg.get("type") if isinstance(act_cfg, dict) else act_cfg
        table = {"ReLU": lambda: nn.ReLU(inplace=True),
                 "ReLU6": lambda: nn.ReLU6(inplace=True),
                 "LeakyReLU": lambda: nn.LeakyReLU(0.1, inplace=True),
                 "PReLU": lambda: nn.PReLU()}
        if isinstance(t, str) and t in table:
            return table[t]()
        if isinstance(act_cfg, dict) and "place_holder" in act_cfg:
            return nn.Identity()
        return act_cfg

    class _ConvModule(nn.Module):
        """Minimal mmcv.cnn.ConvModule: conv (+ optional norm via dict) (+ optional act)."""

        def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                     padding=0, dilation=1, groups=1, norm_cfg=None, act_cfg=None,
                     bias=True, order=("conv", "norm", "act")):
            super().__init__()
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                                  padding, dilation, groups=groups, bias=bias)
            self.with_norm = norm_cfg is not None
            self.with_act = act_cfg is not None
            self.norm = None
            self.act = None
            if self.with_norm:
                if isinstance(norm_cfg, dict):
                    t = norm_cfg["type"]
                    if t.startswith("GN"):
                        self.norm = nn.GroupNorm(norm_cfg.get("num_groups", 4), out_channels)
                    elif t in ("BN", "SyncBN"):
                        self.norm = nn.BatchNorm2d(out_channels)
                    elif t == "IN":
                        self.norm = nn.GroupNorm(1, out_channels)
                else:
                    self.norm = norm_cfg
            if self.with_act:
                self.act = _build_activation_layer(act_cfg)
            self._forward_parts = list(order)

        def forward(self, x):
            for name in self._forward_parts:
                if name == "conv":
                    x = self.conv(x)
                elif name == "norm" and self.with_norm and self.norm is not None:
                    x = self.norm(x)
                elif name == "act" and self.with_act and self.act is not None:
                    x = self.act(x)
            return x

    class FusedBiasLeakyReLU(nn.GroupNorm):
        """Plain-GroupNorm stand-in (the fused CUDA op is not on the inference path)."""

        def __init__(self, num_channels, num_groups=32, eps=1e-3, alpha=0.01,
                     activation=None, **kwargs):
            super().__init__(min(num_groups, num_channels), num_channels, eps=eps)
            self.alpha = alpha if activation is None else activation
            self.activation = activation

        def forward(self, x):
            return F.leaky_relu(super().forward(x), self.alpha)

    def fused_bias_leakyrelu(input, bias, act):
        return F.leaky_relu((input + bias.view(1, -1, 1, 1)).float(), act)

    def upfirdn2d(input, kernel, up_x=1, up_y=1, down_x=1, down_y=1,
                  pad_x0=0, pad_x1=0, pad_y0=0, pad_y1=0):
        # crude fallback (used only by the unused equalized-convs path)
        out = F.interpolate(input, scale_factor=(up_y, up_x), mode="bilinear", align_corners=False)
        if (down_x, down_y) != (1, 1):
            out = F.interpolate(out, scale_factor=(1 / down_y, 1 / down_x),
                                mode="area")
        return out

    def _load_checkpoint_with_prefix(prefix, ckpt_path, map_location="cpu", strict=True):
        ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=False)
        return {k[len(prefix) + 1:]: v for k, v in ckpt.items() if k.startswith(prefix)}

    def _make(mod_name, **attrs):
        m = types.ModuleType(mod_name)
        for a, b in attrs.items():
            setattr(m, a, b)
        sys.modules[mod_name] = m
        return m

    mmcv = _make("mmcv", is_seq_of=_is_seq_of)
    _nn_mod = _make("mmcv.nn")
    cnn = _make("mmcv.cnn", xavier_init=_xavier_init, ConvModule=_ConvModule)
    bricks = _make("mmcv.cnn.bricks")
    _make("mmcv.cnn.bricks.activation", build_activation_layer=_build_activation_layer)
    _make("mmcv.cnn.utils", normal_init=_normal_init)
    ops = _make("mmcv.ops")
    _make("mmcv.ops.fused_bias_leakyrelu", FusedBiasLeakyReLU=FusedBiasLeakyReLU,
          fused_bias_leakyrelu=fused_bias_leakyrelu)
    _make("mmcv.ops.upfirdn2d", upfirdn2d=upfirdn2d)
    _make("mmcv.runner", _load_checkpoint_with_prefix=_load_checkpoint_with_prefix)
    mmcv.cnn = cnn
    cnn.bricks = bricks
    bricks.activation = sys.modules["mmcv.cnn.bricks.activation"]
    cnn.utils = sys.modules["mmcv.cnn.utils"]
    mmcv.ops = ops
    mmcv.runner = sys.modules["mmcv.runner"]
    mmcv.nn = _nn_mod

import cv2
import numpy as np
import torch
from PIL import Image

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


def get_image_files(input_dir: str):
    d = Path(input_dir)
    files = []
    for ext in SUPPORTED_EXTENSIONS:
        files.extend(d.glob(f"*{ext}"))
        files.extend(d.glob(f"*{ext.upper()}"))
    return sorted(set(files))


def elliptical_face_mask(size=256, center=(128, 122), radii=(92, 112)):
    """Aligned-crop face mask (FFHQ-style ellipse), 0/1 uint8 at `size`."""
    y, x = np.ogrid[:size, :size]
    m = (((x - center[0]) / radii[0]) ** 2 + ((y - center[1]) / radii[1]) ** 2) <= 1.0
    return (m.astype(np.uint8))


def build_models(device):
    from models.id_encoder import id_loss
    from models.attr_encoder.psp import pSp
    from models.id_transformer.inn import bulid_models
    from models.mlp.LatentMapper import LatentMapper, FusionMapper10
    from models.attr_encoder.StyleGan2.model import Generator
    from models.deghosting.deghosting import Deghosting
    from utils.utils import loading_pretrianed

    # The vendored deghosting StyleGANv2 generator hardcodes
    # torch.load('/data/lk/ID-disen4/pretrained_models/550000.pt') inside its
    # _load_pretrained_model — rebind it to the local checkpoint.
    import models.deghosting.stylegan2 as _sg2

    def _lpm_local(self, ckpt_path=None, prefix="generator_ema", map_location="cpu", strict=True):
        state = torch.load(W["stylegan128"], map_location="cpu", weights_only=False)["g_ema"]
        self.load_state_dict(state, strict=False)

    _sg2.StyleGANv2Generator._load_pretrained_model = _lpm_local

    print("iFADIT: loading id encoder (IResNet50)...", flush=True)
    id_encoder = id_loss.IDLoss(W["id"]).to(device).eval()

    print("iFADIT: loading pSp attribute encoder...", flush=True)
    psp_opts = Namespace(
        device=device,
        image_size=256, output_size=256, stylegan_size=256, latent=512,
        input_nc=3, label_nc=0,
        image_range=(0.0, 1.0),
        checkpoint_path=W["psp"],
        stylegan_ckpt=W["stylegan128"],
        stylegan_weights=W["stylegan128"],
        encoder_type="GradualStyleEncoder",
        mapping_network="88", mapping_fullyConnected=False, mapping_fp16=False,
        start_from_latent_avg=False, learn_in_w=False,
        exp_dir=os.path.join(REPO, "_tmp_exp"), eval_mode=True,
    )
    attr_encoder = pSp(psp_opts).to(device).eval()

    print("iFADIT: loading INN identity transformer + mappers...", flush=True)
    id_transformer = bulid_models().to(device).eval()
    loading_pretrianed(torch.load(W["inn"], map_location="cpu", weights_only=False), id_transformer)

    mlp = LatentMapper().to(device).eval()
    loading_pretrianed(torch.load(W["mlp"], map_location="cpu", weights_only=False), mlp)

    icl = FusionMapper10().to(device).eval()
    loading_pretrianed(torch.load(W["icl"], map_location="cpu", weights_only=False), icl)

    print("iFADIT: loading StyleGAN2 (256 px, g_ema from 550000.pt)...", flush=True)
    generator = Generator(256, 512, 8).to(device).eval()
    missing, unexpected = generator.load_state_dict(
        torch.load(W["stylegan128"], map_location="cpu", weights_only=False)["g_ema"], strict=False)
    # the pSp-style generator registers random `noises.*` buffers not stored in the ckpt
    real_missing = [k for k in missing if not k.startswith("noises.")]
    print(f"   g_ema load: missing={len(real_missing)} {real_missing[:4]} unexpected={len(unexpected)} "
          f"{unexpected[:4]}", flush=True)
    return dict(device=device, id_encoder=id_encoder, attr_encoder=attr_encoder,
                id_transformer=id_transformer, mlp=mlp, icl=icl,
                generator=generator,
                psp_opts=psp_opts)


def get_concat_vec(id_images, attr_images, M, mode, passwords):
    """Verbatim logic of upstream utils.get_concat_vec (forward = anonymize)."""
    from utils.utils import get_concat_vec as _gcv
    with torch.no_grad():
        return _gcv(id_images, attr_images, M["id_encoder"], M["attr_encoder"],
                    M["id_transformer"], M["icl"], passwords, mode=mode)


def main():
    parser = argparse.ArgumentParser(description="Batch iFADIT de-identification")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-size", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--verify-roundtrip", action="store_true",
                        help="also run the password recovery pass and save side-by-side (smoke tests)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    M = build_models(f"cuda:0")
    device = M["device"]
    gen = M["generator"]
    mlp = M["mlp"]

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    failures_path = output_dir.parent / (Path(args.input).name + "_detection_failures.txt")
    files = get_image_files(args.input)
    if not files:
        sys.exit(f"No supported images found in {args.input}")
    mask256 = elliptical_face_mask(256)
    mask_t = torch.from_numpy(mask256).float().unsqueeze(0).unsqueeze(0)  # [1,1,256,256] in {0,1}

    def img2t(p):
        im = Image.open(p).convert("RGB").resize((256, 256), Image.LANCZOS)
        return torch.from_numpy(np.asarray(im, dtype=np.float32)).permute(2, 0, 1).div_(255.0)

    processed = skipped = failed = 0
    t0 = time.time()
    for start in range(0, len(files), args.batch_size):
        batch = files[start:start + args.batch_size]
        keep = []
        for f in batch:
            if args.limit and processed >= args.limit:
                break
            if not (output_dir / f"{f.stem}.png").exists():
                keep.append(f)
            else:
                skipped += 1
        if not keep:
            continue

        imgs = torch.stack([img2t(f) for f in keep]).to(device)          # [B,3,256,256] in [0,1]
        masks = mask_t.expand(len(keep), -1, -1, -1).to(device)          # [B,1,256,256]
        masks3 = masks.expand(-1, 3, -1, -1).contiguous()
        b = imgs.shape[0]
        try:
            with torch.no_grad():
                # 512-bit password (secret key) for this batch
                passwords = gen.style((torch.rand((b, 512), device=device) + 1.) / 2.)

                concat, _idf, _av, _idb = get_concat_vec(imgs, imgs, M, "forward", passwords)
                wpp = mlp(concat)
                anon, _ = gen([wpp], input_is_latent=True, return_latents=False, randomize_noise=False)
                anon = anon.clamp(-1, 1)

                from utils.utils_data import get_masked_imgs
                anon = get_masked_imgs((anon + 1) / 2, imgs, masks3).detach().cpu()
            for i, f in enumerate(keep):
                out = (anon[i].permute(1, 2, 0).clamp(0, 1) * 255).to(torch.uint8).numpy()
                Image.fromarray(out).resize((args.out_size, args.out_size), Image.LANCZOS) \
                    .save(str(output_dir / f"{f.stem}.png"), "PNG")
                processed += 1
                if args.verify_roundtrip and processed <= 2:
                    with torch.no_grad():
                        anon_dev = ((anon[i:i + 1] + 1) / 2).contiguous().to(device)
                        conc_rec, *_ = get_concat_vec(anon_dev, anon_dev,
                                                     M, "backward", passwords[i:i + 1])
                        rec, _ = gen([mlp(conc_rec)], input_is_latent=True,
                                     return_latents=False, randomize_noise=False)
                    rec = rec.clamp(-1, 1)[0] if rec.dim() == 4 else rec.clamp(-1, 1)
                    row = torch.cat([imgs[i].cpu(), (anon[i] + 1) / 2, ((rec + 1) / 2).cpu()], dim=2)
                    row = row.permute(1, 2, 0).clamp(0, 1)
                    Image.fromarray((row.cpu().numpy() * 255).astype(np.uint8)) \
                        .save(str(output_dir / f"{f.stem}_roundtrip.jpg"), quality=92)
        except Exception as e:
            import traceback
            traceback.print_exc()
            failed += len(keep)
            for f in keep:
                with open(failures_path, "a") as fh:
                    fh.write(f"{f.name}\n")
            print(f"  batch at {os.path.basename(keep[0])} FAILED: {e}", flush=True)

        done = min(start + args.batch_size, len(files))
        rate = processed / max(time.time() - t0, 1e-6)
        eta = max(len(files) - done, 0) / max(rate, 1e-6) / 60
        print(f"  [{done}/{len(files)}] done={processed} skipped={skipped} fails={failed} "
              f"{rate:.3f} img/s ETA {eta:.1f} min", flush=True)

    print(f"\nDone. Processed: {processed}, Skipped (exists): {skipped}, Failed: {failed}")
    if failed:
        print(f"Failures logged to: {failures_path}")


if __name__ == "__main__":
    main()
