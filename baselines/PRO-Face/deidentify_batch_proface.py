"""
PRO-Face (ACM MM 2022) batch de-identification for the deid-toolkit.

Pipeline (matches the released evaluate.py for the `blur_31_2_8` configuration):
    aligned crop [0,1] -> input_trans (Resize 112, ToTensor, Normalize(0.5,0.5) -> [-1,1])
    -> obfuscate: GaussianBlur(kernel 31, sigma ~ U[2,8])   (exact released Blur forward)
    -> restore: ProFaceEmbedder(orig, obfs)  (Siamese DenseUNet, checkpoint in-repo)
    -> clamp[-1,1] -> /2+... -> [0,1] -> resize to out-size (256) -> single PNG save.

The released image_processing.py is NOT imported: it hard-imports the FaceShifter and
SimSwap stacks (only needed by the faceshifter/simswap obfuscators). The Blur and
embedder logic below are copied verbatim from the released sources.
The face recognizer itself is only used for the paper's accuracy metrics, never for
image generation, so no recognition checkpoint is required.
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


def find_restore_ckpt(obf_name: str, rec_name: str) -> str:
    import glob
    pattern = os.path.join(REPO, "checkpoints", f"{obf_name}_{rec_name}_ep*_BEST.pth")
    hits = sorted(glob.glob(pattern))
    if len(hits) != 1:
        raise FileNotFoundError(f"expected exactly 1 match for {pattern}, got {hits}")
    return hits[0]


def main():
    ap = argparse.ArgumentParser(description='PRO-Face (blur+restore) batch de-identification')
    ap.add_argument('--input', required=True, help='folder of aligned single-face images')
    ap.add_argument('--output', required=True, help='output folder (skip-existing enabled)')
    ap.add_argument('--rec-name', default='IResNet50',
                    choices=['IResNet50', 'IResNet100', 'SEResNet50', 'MobileFaceNet', 'InceptionResNet'])
    ap.add_argument('--obf', default='blur_31_2_8')
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
    print(f"Restoration checkpoint: {ckpt_path}")
    model = ProFaceEmbedder()
    sd = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"ckpt load: {len(missing)} missing / {len(unexpected)} unexpected")
    if missing or unexpected:
        print("  missing:", list(missing)[:5])
        print("  unexpected:", list(unexpected)[:5])
    model.to(device).eval()

    blur = Blur(31, 2., 8.).to(device)  # params from obf name blur_31_2_8

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

    def pending(files):
        return [f for f in files if not (outdir / f'{f.stem}.png').exists()]

    to_do = pending(files)
    print(f"files found={len(files)} pending={len(to_do)} (skip-existing on; re-run to fill gaps)")
    skipped = len(files) - len(to_do)
    fail_path = outdir / f"{args.obf}_{args.rec_name}_failures.txt"
    done = 0

    with torch.no_grad():
        for s in range(0, len(to_do), args.batch_size):
            chunk = to_do[s:s + args.batch_size]
            imgs = torch.stack([input_trans(Image.open(f).convert('RGB')) for f in chunk])
            x = imgs.to(device)
            try:
                xb_obfs = blur(x)
                xb_proc = model(x, xb_obfs)
                xb_proc = torch.clamp(xb_proc, -1, 1)
                for f, out in zip(chunk, (xb_proc + 1) / 2):
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
