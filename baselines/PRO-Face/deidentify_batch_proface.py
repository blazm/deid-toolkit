"""
PRO-Face (ACM MM 2022) batch de-identification for the deid-toolkit.

Two obfuscation configurations (each with its released restore checkpoint family):

  --obf blur_31_2_8  (default, original setting)
    crop [0,1] -> input_trans (Resize 112, ToTensor, Normalize(0.5,0.5) -> [-1,1])
      -> GaussianBlur(kernel 31, sigma ~ U[2,8])  (verbatim released Blur)
      -> ProFaceEmbedder(orig, obfs)  (SiameDenseUNet, in-repo checkpoint)
      -> clamp[-1,1] -> (x+1)/2 -> 256 -> PNG

  --obf simswap
    Same input_trans for the source; target identity = random aligned face from
    the INPUT folder (batch-level target, like the released evaluate.py target_set).
      -> SimSwap(target 112, source 224)  (verbatim released SimSwap.forward,
         local weights in SimSwap/checkpoints/people + arcface_checkpoint.tar)
      -> resize obfs to 224 -> ProFaceEmbedder(orig, obfs) (simswap ckpt family)
      -> clamp[0,1] -> 256 -> PNG     (Note: simswap outputs [0,1] range per released code)

The released utils/image_processing.py is NOT imported: it hard-imports the
FaceShifter stack and relies on a hardcoded PROJECT_DIR in config/config.py.
The Blur, SimSwap-forward, and embedder logic below are copied verbatim from
the released sources. The recognizer is only for the paper's accuracy metrics,
never for image generation, so no recognition checkpoint is required.
"""
import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as F
from tqdm import tqdm

REPO = os.path.dirname(os.path.abspath(__file__))
for p in (REPO,):
    if p not in sys.path:
        sys.path.insert(0, p)

from embedder import ProFaceEmbedder  # noqa: E402  (repo-local, dependency-free)


class Blur(torch.nn.Module):  # verbatim from utils/image_processing.py
    def __init__(self, kernel_size, sigma_min, sigma_max):
        super().__init__()
        self.random = True
        self.kernel_size = kernel_size
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_avg = (sigma_min + sigma_max) / 2

    def forward(self, img):
        sigma = random.uniform(self.sigma_min, self.sigma_max) if self.random else self.sigma_avg
        img_blurred = F.gaussian_blur(img, self.kernel_size, [sigma, sigma])
        return img_blurred


def build_simswap():  # mirrors SimSwap class in utils/image_processing.py (no hardcoded paths)
    from config import config as c  # import first: SimSwap's simswap_config reads c.device at import time
    if str(c.device).startswith('cuda'):
        c.device = 'cuda:0'  # config.py hardcodes cuda:1; use the first visible GPU
    del c
    _ss_dir = os.path.join(REPO, 'SimSwap')
    for _p in (_ss_dir,):
        if _p not in sys.path:
            sys.path.insert(0, _p)  # arcface tar pickles reference top-level `models.` classes
    from SimSwap.options.test_options import TestOptions
    from SimSwap.models.models import create_model

    _saved_argv = sys.argv
    _orig_torch_load = torch.load
    sys.argv = [sys.argv[0]]  # TestOptions parses sys.argv; our CLI args must not leak in
    torch.load = lambda *a, **k: _orig_torch_load(*a, **{**k, 'weights_only': False})  # SimSwap ckpts are full pickles
    try:
        opt = TestOptions().parse()
        opt.checkpoints_dir = os.path.join(REPO, 'SimSwap', 'models', 'checkpoints')
        opt.name = 'people'
        opt.Arc_path = os.path.join(REPO, 'SimSwap', 'arcface_model', 'arcface_checkpoint.tar')
        swapper = create_model(opt)
    finally:
        sys.argv = _saved_argv
        torch.load = _orig_torch_load
    swapper.eval()

    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])

    def _swap(x, target_image):  # verbatim released SimSwap.forward
        x_resize = Fn.interpolate(x.mul(0.5).add(0.5), [224, 224], mode='bicubic', align_corners=False)
        target_image_resize = Fn.interpolate(target_image, size=[112, 112], mode='bilinear', align_corners=False)
        latent_id = swapper.netArc(target_image_resize)
        latent_id = latent_id.detach().to('cpu')
        latent_id = latent_id / np.linalg.norm(latent_id, axis=1, keepdims=True)
        latent_id = latent_id.to(next(swapper.netG.parameters()).device)
        x_swap = swapper(target_image, x_resize, latent_id, latent_id, True)
        x_swap = Fn.interpolate(x_swap.mul(2.0).sub(1.0), [112, 112], mode='bicubic', align_corners=False)
        return x_swap

    # released targ_img_trans = ToTensor + ImageNet-Normalize (targets were pre-cropped at 224)
    def target_prep(img_pil):
        t = transforms.functional.to_tensor(img_pil)
        return transforms.functional.normalize(t, mean.tolist(), std.tolist())

    return _swap, target_prep


def find_restore_ckpt(obf_name: str, rec_name: str) -> str:
    import glob
    pattern = os.path.join(REPO, "checkpoints", f"{obf_name}_{rec_name}_ep*_BEST.pth")
    hits = sorted(glob.glob(pattern))
    if len(hits) != 1:
        raise FileNotFoundError(f"expected exactly 1 match for {pattern}, got {hits}")
    return hits[0]


def main():
    ap = argparse.ArgumentParser(description='PRO-Face (blur/simswap + restore) batch de-identification')
    ap.add_argument('--input', required=True, help='folder of aligned single-face images')
    ap.add_argument('--output', required=True, help='output folder (skip-existing enabled)')
    ap.add_argument('--rec-name', default='IResNet50',
                    choices=['IResNet50', 'IResNet100', 'SEResNet50', 'MobileFaceNet', 'InceptionResNet'])
    ap.add_argument('--obf', default='blur_31_2_8', choices=['blur_31_2_8', 'simswap'])
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out-size', type=int, default=256)
    ap.add_argument('--limit', type=int, default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    ckpt_path = find_restore_ckpt(args.obf, args.rec_name)
    print(f"Obfuscation: {args.obf} | restore checkpoint: {ckpt_path}")
    model = ProFaceEmbedder()
    sd = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"ckpt load: {len(missing)} missing / {len(unexpected)} unexpected")
    if missing or unexpected:
        print("  missing:", list(missing)[:5])
        print("  unexpected:", list(unexpected)[:5])
    model.to(device).eval()

    if args.obf == 'simswap':
        blur = None
        print("Building SimSwap swapper (local checkpoints)...")
        swap_fn, target_prep = build_simswap()
    else:
        swap_fn = target_prep = None
        blur = Blur(31, 2., 8.).to(device)  # params from obf name blur_31_2_8

    if args.obf == 'simswap':
        # released main(): trans = input_trans_simswap (Resize 224, no normalize -> [0,1])
        input_trans = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])
    else:
        input_trans = transforms.Compose([
            transforms.Resize(112),
            transforms.ToTensor(),
            transforms.Normalize(mean=0.5, std=0.5),   # -> [-1, 1] (released input_trans)
        ])

    imagedir = Path(args.input)
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    files = [f for f in sorted(imagedir.iterdir()) if f.suffix.lower() in exts]
    if args.limit:
        files = files[:args.limit]

    # SimSwap target pool = the dataset's aligned faces (like the released target_set)
    target_pool = None
    if swap_fn is not None:
        target_pool = [target_prep(Image.open(f).convert('RGB')) for f in files]
        print(f"SimSwap target pool: {len(target_pool)} faces")

    def pending(files):
        return [f for f in files if not (outdir / f'{f.stem}.png').exists()]

    to_do = pending(files)
    print(f"files found={len(files)} pending={len(to_do)} (skip-existing on; re-run to fill gaps)")
    skipped = len(files) - len(to_do)
    fail_path = outdir / f"{args.obf}_{args.rec_name}_failures.txt"
    done = 0
    lower = 0 if args.obf == 'simswap' else -1

    with torch.no_grad():
        for s in range(0, len(to_do), args.batch_size):
            chunk = to_do[s:s + args.batch_size]
            bs = len(chunk)
            imgs = torch.stack([input_trans(Image.open(f).convert('RGB')) for f in chunk])
            x = imgs.to(device)
            try:
                if blur is not None:
                    xb_obfs = blur(x)
                    xb_proc = model(x, xb_obfs)
                    xb_proc = torch.clamp(xb_proc, -1, 1)
                    outputs = (xb_proc + 1) / 2
                else:
                    targ = target_pool[random.randrange(len(target_pool))]
                    targ_batch = targ.repeat(bs, 1, 1, 1).to(device)
                    xb_obfs = swap_fn(x, targ_batch)
                    xb_obfs = Fn.interpolate(xb_obfs, [224, 224], mode='bicubic', align_corners=False)
                    xb_proc = model(x, xb_obfs)
                    xb_proc = torch.clamp(xb_proc, 0, 1)
                    outputs = xb_proc
                for f, out in zip(chunk, outputs):
                    outp = transforms.Resize((args.out_size, args.out_size),
                                             transforms.InterpolationMode.BICUBIC)(out.cpu())
                    arr = (outp.clamp(0, 1).numpy().transpose(1, 2, 0) * 255).round().astype(np.uint8)
                    Image.fromarray(arr).save(str(outdir / f'{f.stem}.png'), 'PNG')
            except Exception as e:
                import traceback
                traceback.print_exc()
                with open(fail_path, 'a') as fh:
                    for f in chunk:
                        fh.write(f"{f.name}\t{e}\n")
                continue
            done += len(chunk)
            if (s // args.batch_size) % 10 == 0:
                print(f"  processed {done}/{len(to_do)}")

    print(f"\nDone. Processed this run: {done}, Skipped (exists): {skipped}")
    if os.path.exists(fail_path):
        print(f"Failures logged to: {fail_path}")


if __name__ == '__main__':
    main()
