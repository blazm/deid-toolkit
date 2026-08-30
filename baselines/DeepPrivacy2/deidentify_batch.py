#!/usr/bin/env python
"""Batch face de-ID using DeepPrivacy2 with DSFD detection.

Matches the exact pipeline from anonymizer.py::anonymize_detections().
Detects faces → crops/expands → generates → pastes back with mask blending.
"""
import argparse, os, sys, warnings; warnings.filterwarnings('ignore')
import numpy as np, torch, tops
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torchvision.transforms.functional as F_tv

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--truncation", type=float, default=0.5)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_files = []
    for ext in {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}:
        image_files.extend(Path(args.input).glob(f"*{ext}"))
        image_files.extend(Path(args.input).glob(f"*{ext.upper()}"))
    image_files = sorted(set(image_files))

    # Build generator
    from dp2.generator.stylegan_unet import StyleGANUnet
    from dp2.infer import load_generator_state
    G = StyleGANUnet(scale_grad=True, im_channels=3, min_fmap_resolution=8, imsize=[256, 256], cnum=64, max_cnum_mul=8, mask_output=True, conv_clamp=256, input_cse=False, cse_nc=None, n_middle_blocks=0, input_keypoints=False, n_keypoints=0, input_keypoint_indices=[], fix_errors=True, z_channels=512, w_dim=512)
    ckpt = torch.load(str(Path(REPO_ROOT) / "checkpoints" / "89660f04-5c11-4dbf-adac-cbe2f11b0aeea25cbf78-7558-475a-b3c7-03f5c10b7934646b0720-ca0a-4d53-aded-daddbfa45c9e"), map_location="cpu")
    load_generator_state(ckpt, G)
    G.cuda().eval()

    # Build detector
    from dp2.detection.face_detector import FaceDetector
    detector = FaceDetector(face_detector_cfg=dict(name="DSFDDetector", clip_boxes=True), face_post_process_cfg=dict(target_imsize=(256, 256), fdf128_expand=False), score_threshold=0.3, cache_directory=str(Path(REPO_ROOT) / "face_cache"))

    processed = 0
    for img_path in tqdm(image_files, desc="De-ID"):
        try:
            pil_img = Image.open(img_path).convert("RGB")
            im_np = np.array(pil_img)
            C, H, W = 3, *im_np.shape[:2]

            # Detection expects uint8
            im_uint8 = tops.im2torch(im_np, to_float=False)[0].byte().cuda()

            # Detect faces — matches anonymizer.py exactly
            all_detections = detector(im_uint8)

            # Convert to float32 for processing (avoids uint8 arithmetic overflow)
            im_tensor = im_uint8.float()

            for det in all_detections:
                if len(det) == 0:
                    continue
                C_, H_, W_ = im_tensor.shape  # current image dims
                for idx in range(len(det)):
                    batch = det.get_crop(idx, im_tensor)
                    x0, y0, x1, y1 = [int(v) for v in batch.pop("boxes")[0]]
                    batch = {k: tops.to_cuda(v) for k, v in batch.items()}

                    # Normalize — matches forward_G exactly
                    batch["img"] = F_tv.normalize(batch["img"].float(), [127.5]*3, [127.5]*3)
                    batch["condition"] = batch["mask"].float() * batch["img"]

                    # Random latent + truncation — matches forward_G
                    state = np.random.RandomState(seed=np.random.randint(0, 2**31, dtype=np.int64))
                    z = torch.from_numpy(state.normal(size=(1, G.z_channels)).astype(np.float32)).cuda()
                    w = G.style_net.get_truncated(args.truncation, condition=batch["condition"], z=z)

                    # Generate with AMP — matches forward_G
                    with torch.cuda.amp.autocast(True), torch.no_grad():
                        anonymized_im = G(**batch, w=w)["img"]

                    # Resize + paste back — EXACTLY as anonymize_detections in anonymizer.py
                    gim = F_tv.resize(anonymized_im[0], (y1-y0, x1-x0), interpolation=F_tv.InterpolationMode.BICUBIC, antialias=True)
                    mask = F_tv.resize(batch["mask"][0], (y1-y0, x1-x0), interpolation=F_tv.InterpolationMode.NEAREST).squeeze(0)

                    # Remove padding (handles expanded bbox going outside image)
                    pad = [max(-x0, 0), max(-y0, 0), max(x1-W_, 0), max(y1-H_, 0)]
                    gh, gw = gim.shape[-2], gim.shape[-1]
                    mh, mw = mask.shape[0], mask.shape[1]
                    gim = gim[:, pad[1]:gh-pad[3], pad[0]:gw-pad[2]]
                    mask = mask[pad[1]:mh-pad[3], pad[0]:mw-pad[2]]

                    x0, y0 = max(x0, 0), max(y0, 0)
                    x1, y1 = min(x1, W_), min(y1, H_)

                    # Denormalize from [-1,1] to [0,255] — standard formula (no inversion needed)
                    gim_255 = torch.nan_to_num((gim + 1.0).div(2.0).clamp(0.0, 1.0).mul(255.0), nan=0.0)

                    # Paste back with mask blending — matches anonymizer.py exactly
                    face_mask = (mask < 0.5).bool()[None].repeat(3, 1, 1)
                    im_tensor[:, y0:y1, x0:x1][face_mask] = gim_255[face_mask].round()

            # Convert float32 [0,255] → uint8 for numpy/PIL
            out_np = im_tensor.cpu().clamp(0, 255).round().byte().numpy()
            if out_np.shape[0] == 3:  # [C, H, W] → [H, W, C]
                out_np = np.transpose(out_np, (1, 2, 0))
            Image.fromarray(out_np).save(output_dir / f"{img_path.stem}.png", "PNG")
            processed += 1

        except Exception as e:
            print(f"\n[FAIL] {img_path.name}: {e}")

    print(f"\nDone. {processed}/{len(image_files)}")


if __name__ == "__main__":
    main()
