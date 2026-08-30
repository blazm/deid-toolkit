#!/usr/bin/env python
"""
Batch face de-identification using DiffPrivate Perturb (diffusion-based).
Loads config.yaml directly — no Hydra dependency needed.

Usage:
    python deidentify_batch.py --input <aligned_faces_dir> --output <output_dir>
"""
import sys
import os
import argparse
import glob
import yaml


class AttrDict(dict):
    """Allow dot-notation access to dict keys."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for k, v in self.items():
            if isinstance(v, dict):
                self[k] = AttrDict(v)

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' has no '{key}'")

    __setattr__ = dict.__setitem__


def load_config():
    """Load the default DiffPrivate config from YAML."""
    cfg_path = os.path.join(os.path.dirname(__file__), "configs", "config.yaml")
    with open(cfg_path, "r") as f:
        data = yaml.safe_load(f)

    # Convert all nested dicts to AttrDict for dot-notation access
    def _to_attr(d):
        if isinstance(d, dict):
            return AttrDict({k: _to_attr(v) for k, v in d.items()})
        elif isinstance(d, list):
            return [_to_attr(x) for x in d]
        return d

    return _to_attr(data)


def main():
    parser = argparse.ArgumentParser(
        description="Batch face de-identification using DiffPrivate Perturb"
    )
    parser.add_argument("--input", required=True, help="Input folder with aligned face images")
    parser.add_argument("--output", required=True, help="Output folder for protected images")
    parser.add_argument("--debug", action="store_true", help="Save all intermediate images and logs (default: only adversarial image)")
    args = parser.parse_args()

    # Validate input directory
    if not os.path.isdir(args.input):
        print(f"Error: Input directory does not exist: {args.input}")
        sys.exit(1)

    # Load default config
    cfg = load_config()

    # Override paths from CLI args — use absolute paths everywhere
    cfg.paths.images_root = os.path.abspath(args.input)
    cfg.paths.save_dir = os.path.abspath(args.output)

    # Count images
    image_paths = [
        p for p in glob.glob(os.path.join(args.input, "*"))
        if p.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    if not image_paths:
        print(f"Error: No images found in {args.input}")
        sys.exit(1)

    print(f"Found {len(image_paths)} images in {args.input}")
    print(f"Saving to {args.output}")

    # Load diffusion model
    # Use locally downloaded weights (same as config.yaml pretrained_diffusion_path)
    local_model_dir = os.path.join(os.path.dirname(__file__), "models", "stable-diffusion-2-base")
    if os.path.isdir(local_model_dir):
        model_path = local_model_dir
    else:
        model_path = getattr(cfg.paths, 'pretrained_diffusion_path', 'stabilityai/stable-diffusion-2-base')
    print(f"Loading Stable Diffusion 2-base model from '{model_path}'...")

    # Prevent ALL network downloads — serve from local files instead.
    # Covers: (1) PyTorch hub AlexNet weights, (2) LPIPS linear mapping weights (alex.pth),
    # and any other hub.load_state_dict_from_url calls by torch/hub.
    MODELS_DIR = os.path.join(os.path.dirname(__file__), "model-weights")

    # Tell criteria library to load face recognition models from local model-weights/
    # instead of downloading from Google Drive during blackbox evaluation.
    os.environ["CRITERIA_WEIGHTS_DIR"] = os.path.abspath(MODELS_DIR)

    # Mapping of known download URLs to local files (exact filename -> path)
    _URL_TO_LOCAL = {
        "alex.pth": os.path.join(MODELS_DIR, "alex.pth"),
        "alexnet-owt-7be5e715.pth": os.path.join(MODELS_DIR, "alexnet-owt-7be5e715.pth"),
        "alexnet-owt-7be5be79.pth": os.path.join(MODELS_DIR, "alexnet-owt-7be5be79.pth"),
    }

    def _mock_load_state_dict_from_url(url, progress=True, map_location=None, **kwargs):
        """Intercept torch.hub.load_state_dict_from_url — serve local weights.

        **kwargs absorbs extra parameters (e.g. check_hash) added by newer torchvision versions.
        """
        filename = os.path.basename(url)
        candidate = _URL_TO_LOCAL.get(filename)  # exact match first
        if candidate and os.path.isfile(candidate):
            print(f"  [offline] Served '{filename}' from '{candidate}'")
            return torch.load(candidate, map_location=map_location or "cpu")
        raise RuntimeError(f"No local weights for URL: {url} (requested filename='{filename}')")

    import torch
    torch.hub.load_state_dict_from_url = _mock_load_state_dict_from_url  # noqa: F811

    # Patch criteria.models.irse.Flatten to use .reshape() instead of .view().
    # Older criteria versions use .view() which fails on non-contiguous tensors.
    import torch.nn as nn
    class _Flatten(nn.Module):
        def forward(self, input):
            return input.reshape(input.size(0), -1)

    import criteria.models.irse as _irse_mod  # noqa: F811
    _irse_mod.Flatten = _Flatten

    import torch.cuda

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    from diffusers import StableDiffusionPipeline, DDIMScheduler

    ldm_stable = StableDiffusionPipeline.from_pretrained(model_path).to(device)
    ldm_stable.scheduler = DDIMScheduler.from_config(ldm_stable.scheduler.config)

    # Ensure CLIP model is in its cache location. clip._download checks SHA256 and only uses
    # local file if hash matches — otherwise it would try to re-download over HTTPS (fails).
    import shutil
    clip_cache = os.path.expanduser("~/.cache/clip")
    os.makedirs(clip_cache, exist_ok=True)
    clip_target = os.path.join(clip_cache, "ViT-B-32.pt")
    local_clip = os.path.join(MODELS_DIR, "ViT-B-32.pt")
    if os.path.isfile(local_clip):
        if not os.path.isfile(clip_target):
            shutil.copy2(local_clip, clip_target)
            print(f"  Copied CLIP model to '{clip_cache}'")
        elif os.path.getmtime(local_clip) > os.path.getmtime(clip_target):
            shutil.copy2(local_clip, clip_target)
            print(f"  Updated CLIP model in cache")

    # Patch LPIPS device mismatch: criteria LPiPS hardcodes .to("cuda") per-submodule,
    # leaving BaseNet mean/std buffers on CPU. Fix by wrapping the class init.
    import torch.nn as nn
    from criteria.lpips import lpips as _orig_lpips_mod
    from criteria.lpips.networks import get_network, LinLayers

    class _OfflineLPIPS(nn.Module):
        """Drop-in replacement for LPIPS that ensures ALL tensors on CUDA."""
        def __init__(self, net_type="alex", version="0.1"):
            super().__init__()
            self.net = get_network(net_type)
            self.pool = nn.AdaptiveAvgPool2d((256, 256))
            self.lin = LinLayers(self.net.n_channels_list)
            self.lin.load_state_dict(_orig_lpips_mod.get_state_dict(net_type, version))
            # Move EVERYTHING to CUDA in one call — this moves all registered buffers too
            self.to("cuda")

        def forward(self, x, y, normalize=True):
            if x.shape[2] != 256:
                x = self.pool(x)
            if y.shape[2] != 256:
                y = self.pool(y)
            x = x[:, :, 35:223, 32:220]
            y = y[:, :, 35:223, 32:220]
            feat_x, feat_y = self.net(x), self.net(y)
            diff = [(fx - fy) ** 2 for fx, fy in zip(feat_x, feat_y)]
            res = [l(d).mean((2, 3), True) for d, l in zip(diff, self.lin)]
            if normalize:
                return torch.sum(torch.cat(res, 0)) / x.shape[0]
            return torch.cat(res, 0)

    _orig_lpips_mod.LPIPS = _OfflineLPIPS

    # Import DiffPrivate modules
    from src.attCtr import AttentionControlEdit
    import src.diffprivate_pert as dp_pert
    from PIL import Image
    from natsort import natsorted

    image_paths = natsorted(image_paths)
    os.makedirs(args.output, exist_ok=True)

    skipped = 0
    for i, image_path in enumerate(image_paths):
        image_basename = os.path.splitext(os.path.basename(image_path))[0]

        # Skip images whose output already exists (resume support)
        if os.path.isfile(os.path.join(args.output, image_basename + ".png")):
            skipped += 1
            print(f"[{i+1}/{len(image_paths)}] Skipping {image_basename} (output already exists)")
            continue

        print(f"\n[{i+1}/{len(image_paths)}] Processing {image_basename}...")

        # Set up attention controller (self_replace_steps from config)
        self_replace = cfg.diffusion.self_replace_steps
        if isinstance(self_replace, float):
            self_replace = (0, self_replace)

        controller = AttentionControlEdit(
            cfg.diffusion.diffusion_steps,
            self_replace,
            cfg.diffusion.res,
        )

        save_path = os.path.join(args.output, image_basename)
        result, distance_dict, recognition_dict = dp_pert.protect(
            model=ldm_stable,
            controller=controller,
            args=cfg,
            image_path=image_path,
            save_path=save_path,
            debug=args.debug,
        )

        if not args.debug:
            # Minimal mode: only the adversarial image is already written by dp_pert as <basename>_adv_image.png
            # Rename it to match the original filename with .png extension.
            adv_path = save_path + "_adv_image.png"
            if os.path.isfile(adv_path):
                out_name = os.path.splitext(os.path.basename(image_path))[0] + ".png"
                final_path = os.path.join(args.output, out_name)
                # If output file already exists (same name as input in a different dir), skip
                if not os.path.exists(final_path):
                    os.replace(adv_path, final_path)
                else:
                    # Backup — don't overwrite existing file
                    backup = final_path
                    idx = 1
                    while os.path.exists(f"{backup}_{idx}.png"):
                        idx += 1
                    final_path = f"{backup}_{idx}.png"
                    os.replace(adv_path, final_path)

        print(f"  Protected against blackbox models:")
        for name, recognized in recognition_dict.items():
            status = "RECOGNIZED" if recognized else "DE-ID'd"
            print(f"    {name}: {status}")

    print(f"\nDone! {len(image_paths) - skipped} images processed, {skipped} skipped (already existed).")
    output_mode = "debug (all files saved)" if args.debug else "minimal (only adversarial image)"
    print(f"Mode: {output_mode}")
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
