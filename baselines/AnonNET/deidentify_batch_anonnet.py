"""Batch face de-identification using AnonNET (ICCVW CV4BIOM 2025), image protocol.

All models load from LOCAL paths under this repository (no external caches at run time):
  AnonHead/models/local/Realistic_Vision_V5.0_noVAE  (SD1.5 base)
  AnonHead/models/cache_model                        (3 SD1.5 ControlNets, HF cache layout)
  AnonHead/models/local/sd-vae-ft-mse-original       (SV2 VAE)
  AnonHead/models/local/Annotators                   (lineart + openpose .pth files)
  AnonHead/face_parsing/weights/resnet18.pt          (BiSeNet, in-repo)
  deepface_home/.deepface/weights                    (TF RetinaFace/age/gender/race/emotion)

Usage (run inside the `anonnet` conda env):
    python deidentify_batch_anonnet.py --input <aligned_face_dir> --output <out_dir>
        [--steps 35] [--seed 0] [--strength 0.9 0.4 0.3]

Conventions (match other baselines, e.g. RP/deidentify_batch_rp.py):
- input images are pre-aligned single-face crops;
- prompt attributes come from DeepFace analysis of the input (paper protocol);
- outputs are PNGs with the same basename, resized in memory to --out-size
  (default 256x256) BEFORE the single save; existing outputs are skipped;
- images where no face is detected are NOT written and are appended to
  <out_dir>/../<input name>_detection_failures.txt
"""
import argparse
import contextlib
import gc
import io
import os
import sys
import time
from pathlib import Path

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)

LOCAL = os.path.join(REPO_ROOT, "AnonHead", "models", "local")
VAE_PATH = os.path.join(LOCAL, "sd-vae-ft-mse-original", "vae-ft-mse-840000-ema-pruned.safetensors")
ANNOTATORS_DIR = os.path.join(LOCAL, "Annotators")
CACHE_MODEL = os.path.join(REPO_ROOT, "AnonHead", "models", "cache_model")

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

# ENVIRONMENT: TF must be imported before onnx (proto DType symbol collision)
os.environ["DEEPFACE_HOME"] = os.path.join(REPO_ROOT, "deepface_home")
import tensorflow  # noqa: F401  (must precede onnx/deepface/controlnet_aux imports)


def patch_local_models():
    """Redirect the remaining HF-cache lookups to local files."""
    from diffusers import AutoencoderKL

    _orig_ksf = AutoencoderKL.from_single_file

    @classmethod
    def _ksf_patched(cls, pretrained_model_name_or_path, *args, **kwargs):
        kwargs.pop("cache_dir", None)
        # The VAE URL in AnonHead/predict_multiple.py is replaced by the local file
        if "huggingface.co" in str(pretrained_model_name_or_path):
            pretrained_model_name_or_path = VAE_PATH
        return _orig_ksf(pretrained_model_name_or_path, *args, **kwargs)

    AutoencoderKL.from_single_file = _ksf_patched

    import controlnet_aux.lineart as ca_lineart
    import controlnet_aux.open_pose as ca_openpose
    import controlnet_aux.hed as ca_hed
    import controlnet_aux.canny as ca_canny

    def _local_hub_download(repo_id, filename=None, *args, **kwargs):
        if "lllyasviel/Annotators" in str(repo_id):
            p = os.path.join(ANNOTATORS_DIR, filename)
            if os.path.isfile(p):
                return p
        raise FileNotFoundError(f"AnonNET wrapper: no local file for {repo_id}/{filename} — "
                                "run model pre-download step first")

    for mod in (ca_lineart, ca_openpose, ca_hed, ca_canny):
        if hasattr(mod, "hf_hub_download"):
            mod.hf_hub_download = _local_hub_download


def get_image_files(input_dir: str):
    d = Path(input_dir)
    files = []
    for ext in SUPPORTED_EXTENSIONS:
        files.extend(d.glob(f"*{ext}"))
        files.extend(d.glob(f"*{ext.upper()}"))
    return sorted(set(files))


def main():
    parser = argparse.ArgumentParser(description="Batch AnonNET de-identification (image protocol)")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--segment", default="face", choices=["face", "head"])
    parser.add_argument("--margin", type=float, default=1)
    parser.add_argument("--steps", type=int, default=35)
    parser.add_argument("--strength", type=float, nargs="+", default=[0.9, 0.4, 0.3])
    parser.add_argument("--guidance_scale", type=float, default=8.0)
    parser.add_argument("--max_height", type=int, default=612)
    parser.add_argument("--max_width", type=int, default=612)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-size", type=int, default=256,
                        help="Resize to NxN in memory before the single save (0 = keep native)")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if len(args.strength) not in (1, 3):
        sys.exit("--strength needs 1 or 3 values")

    patch_local_models()

    import AnonHead.face_parsing.models.bisenet as _bisenet_mod
    # The custom BiSeNet backbone resnet18() defaults to torchvision weights download,
    # which load_face_seg() immediately overwrites with the local full state dict
    # (AnonHead/face_parsing/weights/resnet18.pt). Disable the pointless download.
    for _name in ("resnet18", "resnet34"):
        _orig = getattr(_bisenet_mod, _name)
        def _noload(_orig=_orig, **_):
            return _orig(weights=None)
        setattr(_bisenet_mod, _name, _noload)

    from PIL import Image
    from AnonHead.predict_multiple import Predictor
    from AnonHead.segment_multiple import Segment

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    failures_path = output_dir.parent / (Path(args.input).name + "_detection_failures.txt")
    image_files = get_image_files(args.input)
    if not image_files:
        sys.exit(f"No supported images found in {args.input}")

    print("Loading AnonNET Predictor (SD1.5 + 3 ControlNets, local weights)...", flush=True)
    predictor = Predictor()
    print("Loading Segment (BiSeNet + TF RetinaFace, local weights)...", flush=True)
    segment = Segment(load_det=False, load_seg=False)
    print("Models ready.", flush=True)

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
                outputs = segment.retinaface_detect_and_annotate(
                    img=str(img_path), margin=args.margin, method=args.segment)
                if not outputs:
                    raise RuntimeError("no face detected")
                image_pil = Image.open(img_path)
                generated = []
                for (mask_img, seg_img, bbox) in outputs:
                    crop, _dist, _attrs = predictor.anonymize(
                        image=seg_img, prompt="",
                        mask=mask_img, negative_prompt="",
                        strength=args.strength,
                        max_height=args.max_height, max_width=args.max_width,
                        steps=args.steps, seed=args.seed,
                        guidance_scale=args.guidance_scale, im_path=str(img_path))
                    if crop is None:
                        continue
                    generated.append((mask_img.resize(seg_img.size, Image.LANCZOS),
                                      crop.resize(seg_img.size, Image.LANCZOS), bbox))
                if not generated:
                    raise RuntimeError("anonymization returned no crop")
                out_image = segment.merge_crops(image_pil, generated)
                out_image = out_image.resize(image_pil.size, Image.LANCZOS)
            if args.out_size and out_image.size != (args.out_size, args.out_size):
                out_image = out_image.resize((args.out_size, args.out_size), Image.LANCZOS)
            out_image.convert("RGB").save(str(out_path), "PNG")
            processed += 1
        except Exception as e:
            failed += 1
            if "no face detected" in str(e):
                with open(failures_path, "a") as f:
                    f.write(f"{img_path.name}\n")
            print(f"  [{idx}/{len(image_files)}] FAILED {img_path.name}: {e}", flush=True)
        if idx % 10 == 0 or idx == len(image_files):
            rate = processed / max(time.time() - t0, 1e-6)
            eta = (len(image_files) - idx) / max(rate, 1e-6) / 60
            print(f"  [{idx}/{len(image_files)}] done={processed} skipped={skipped} fails={failed} "
                  f"{rate:.3f} img/s  ETA {eta:.1f} min", flush=True)
        gc.collect()

    print(f"\nDone. Processed: {processed}, Skipped (exists): {skipped}, Failed: {failed}")
    if failed:
        print(f"Detection failures logged to: {failures_path}")


if __name__ == "__main__":
    main()
