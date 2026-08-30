"""Blur a directory of face images using Gaussian blur.

Basic built-in technique (also the default toy example in the docs).

CLI (batch contract, shared with the external de-identification runners in
`baselines/`):

    python blur.py --input <aligned_dir> --output <deid_dir>

CLI (legacy pipeline interface -- positional, used by `deid run`):

    python blur.py <dataset_path> <dataset_save>
                   [--dataset_filetype jpg] [--dataset_newtype jpg]
"""
import cv2
import numpy as np
import argparse
from pathlib import Path


def blur_image(img, kernel_size=31, sigma=2):
    """Apply Gaussian blur to an image to protect the face."""
    blurred_img = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)
    return blurred_img


def process_files(filepaths, save_dir, source_extension='jpg', destination_extension='png'):
    """Read images, apply blur, and save them."""
    for file_path in filepaths:
        if Path(file_path).suffix.lower() not in ('.jpg', '.jpeg', '.png', '.bmp', '.webp'):
            continue
        img = cv2.imread(file_path)
        if img is not None:
            blurred_img = blur_image(img)
            try:
                cv2.imwrite(str(Path(save_dir) / (file_path.name.rsplit('.', 1)[0] + '.' + destination_extension)), blurred_img)
            except RuntimeError:
                pass


def main(dir_path, save_dir, source_extension='jpg', destination_extension='jpg'):
    """Process all images in a directory, apply blurring, and save the result."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    filepaths = list(Path(dir_path).glob(f'*.{source_extension}'))
    process_files(filepaths, save_dir, source_extension, destination_extension)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description="Blur images (basic built-in technique). "
                    "Batch contract: --input/--output; legacy pipeline: two positional args.")
    # Batch-contract interface (same flags as the external baseline runners)
    ap.add_argument('--input', help='Directory of aligned face images (batch contract)')
    ap.add_argument('--output', help='Output directory (batch contract)')
    ap.add_argument('--batch-size', type=int, default=None, help='Accepted for CLI compatibility (no effect)')
    ap.add_argument('--seed', type=int, default=None, help='Accepted for CLI compatibility (no effect)')
    ap.add_argument('--out-size', type=int, default=None, help='Accepted for CLI compatibility (no effect)')
    # Legacy pipeline interface (positional + filetypes)
    ap.add_argument('dataset_path', nargs='?', type=str)
    ap.add_argument('dataset_save', nargs='?', type=str)
    ap.add_argument('--dataset_filetype', type=str, default='jpg')
    ap.add_argument('--dataset_newtype', type=str, default='jpg')
    args, _ = ap.parse_known_args()

    if args.input and args.output:
        main(args.input, args.output)
    elif args.dataset_path and args.dataset_save:
        main(args.dataset_path, args.dataset_save,
             args.dataset_filetype, args.dataset_newtype)
    else:
        ap.print_help()
