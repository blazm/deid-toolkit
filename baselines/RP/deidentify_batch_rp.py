"""Batch face de-identification using RP (Reverse Personalization, WACV 2026).

All models load from local folders under this directory (no HF/OS cache):
  pretrained_models/stable-diffusion-xl-base-1.0
  pretrained_models/IP-Adapter (CLIP image encoder subfolder)
  pretrained_models/IP-Adapter-FaceID (ip-adapter-faceid_sdxl.bin)
  insightface_root/models/buffalo_l (InsightFace detectors/analysis, ONNX)

Usage (run inside the `rp` conda env):
    python deidentify_batch_rp.py --input <aligned_face_dir> --output <out_dir> \
        [--steps 100] [--skip 0.7] [--guidance_scale -10.0] [--seed 0]

Conventions match the other baselines (e.g. NullFace/deidentify_batch.py):
- input images are pre-aligned single-face crops (protocol: enable_face_detection=False);
- outputs are PNGs with the same basename, resized in memory to --out-size (default 256x256)
  BEFORE saving, so each image hits disk exactly once; existing outputs are skipped;
- images where InsightFace detects no face are NOT written to the output dir and are
  appended to <out_dir>/../<input name>_detection_failures.txt for follow-up.
"""
import argparse
import os
import sys
import time
import contextlib
import io
from pathlib import Path
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


def get_image_files(input_dir: str):
    files = []
    d = Path(input_dir)
    for ext in SUPPORTED_EXTENSIONS:
        files.extend(d.glob(f"*{ext}"))
        files.extend(d.glob(f"*{ext.upper()}"))
    return sorted(set(files))


def main():
    parser = argparse.ArgumentParser(description="Batch RP (Reverse Personalization) de-identification")
    parser.add_argument("--input", required=True, help="Input folder with aligned face images")
    parser.add_argument("--output", required=True, help="Output folder for de-identified images (PNG)")
    parser.add_argument("--steps", type=int, default=100, help="Inversion/denoising steps (default 100)")
    parser.add_argument("--skip", type=float, default=0.7, help="Fraction of steps skipped in reverse (default 0.7)")
    parser.add_argument("--guidance_scale", type=float, default=-10.0, help="CFG scale (default -10.0)")
    parser.add_argument("--id_emb_scale", type=float, default=1.0)
    parser.add_argument("--ip_adapter_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None, help="Process at most N images (smoke tests)")
    parser.add_argument("--out-size", type=int, default=256,
                        help="Resize de-identified output to NxN in memory before the single save "
                             "(default 256; 0 = keep pipeline-native 1024x1024)")
    args = parser.parse_args()

    from anonymize_local import init_pipeline, anonymize_one, FaceDetectionError

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    failures_path = output_dir.parent / (Path(args.input).name + "_detection_failures.txt")

    image_files = get_image_files(args.input)
    if not image_files:
        sys.exit(f"No supported images found in {args.input}")

    init_pipeline()
    processed = skipped = failed = 0
    t0 = time.time()

    for idx, img_path in enumerate(image_files, 1):
        out_path = output_dir / f"{img_path.stem}.png"
        if out_path.exists():
            skipped += 1
            continue
        if args.limit and processed >= args.limit:
            break
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                result = anonymize_one(
                    image_path=str(img_path),
                    skip=args.skip,
                    id_emb_scale=args.id_emb_scale,
                    guidance_scale=args.guidance_scale,
                    num_inversion_steps=args.steps,
                    ip_adapter_scale=args.ip_adapter_scale,
                    seed=args.seed,
                )
            if args.out_size and (result.width, result.height) != (args.out_size, args.out_size):
                result = result.resize((args.out_size, args.out_size), Image.LANCZOS)
            result.save(str(out_path), "PNG")  # single save, at final resolution
            processed += 1
        except FaceDetectionError as e:
            failed += 1
            with open(failures_path, "a") as f:
                f.write(f"{img_path.name}\n")
            print(f"  [{idx}/{len(image_files)}] DETECTION FAIL {img_path.name}")
        except Exception as e:
            failed += 1
            print(f"  [{idx}/{len(image_files)}] FAILED {img_path.name}: {e}")
        if idx % 5 == 0 or idx == len(image_files):
            rate = processed / max(time.time() - t0, 1e-6)
            eta = (len(image_files) - idx) / max(rate, 1e-6) / 60
            print(f"  [{idx}/{len(image_files)}] done={processed} skipped={skipped} fails={failed} "
                  f"{rate:.3f} img/s  ETA {eta:.1f} min", flush=True)

    print(f"\nDone. Processed: {processed}, Skipped (exists): {skipped}, Failed: {failed}")
    if failed:
        print(f"Detection failures logged to: {failures_path}")


if __name__ == "__main__":
    main()
