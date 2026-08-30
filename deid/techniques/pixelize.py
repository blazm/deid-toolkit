"""Pixelize a directory of face images (mosaic / block averaging).

Basic built-in technique (also the default toy example in the docs).

CLI (batch contract, shared with the external de-identification runners in
`baselines/`):

    python pixelize.py --input <aligned_dir> --output <deid_dir>

CLI (legacy pipeline interface -- positional, used by `deid run`):

    python pixelize.py <dataset_path> <dataset_save>
                       [--dataset_filetype jpg] [--dataset_newtype jpg]
"""
import cv2
import numpy as np
import argparse
from pathlib import Path


def pixelize_image(img, block_size=10):
    """Pixelize an image to protect the face."""
    h, w = img.shape[:2]
    # Resize the image to a smaller size and then resize it back to the original size
    small_img = cv2.resize(img, (w // block_size, h // block_size), interpolation=cv2.INTER_AREA)
    pixelized_img = cv2.resize(small_img, (w, h), interpolation=cv2.INTER_NEAREST)
    return pixelized_img


def process_files(filepaths, save_dir, source_extension='jpg', destination_extension='png'):
    """Read images, apply pixelization, and save them."""
    for file_path in filepaths:
        if Path(file_path).suffix.lower() not in ('.jpg', '.jpeg', '.png', '.bmp', '.webp'):
            continue
        img = cv2.imread(file_path)
        if img is not None:
            pixelized_img = pixelize_image(img)
            try:
                cv2.imwrite(str(Path(save_dir) / (file_path.name.rsplit('.', 1)[0] + '.' + destination_extension)), pixelized_img)
            except RuntimeError:
                pass


def main(dir_path, save_dir, source_extension='jpg', destination_extension='jpg'):
    """Process all images in a directory, apply pixelization, and save the result."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    filepaths = list(Path(dir_path).glob(f'*.{source_extension}'))
    process_files(filepaths, save_dir, source_extension, destination_extension)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description="Pixelize images (basic built-in technique). "
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
