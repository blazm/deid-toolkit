#!/usr/bin/env python3
"""
Batch-run SwinFace embedding generation across multiple dataset folders.

Scans a parent directory for subfolders (excluding specified ones),
and generates embeddings for each one.

Usage:
    python batch_run_swinface.py ^
        --datasets-dir D:\dev\deid-toolkit\root_dir\datasets ^
        --output-root  D:\dev\deid-toolkit\root_dir\embeddings\SwinFace

Excludes: aligned, pairs, labels, original, deidentified (customizable via --exclude)
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def discover_dataset_folders(datasets_dir, exclude=None):
    """Find all subfolders in datasets_dir, excluding specified names."""
    if exclude is None:
        exclude = {"aligned", "pairs", "labels", "original", "deidentified"}

    ds_path = Path(datasets_dir)
    if not ds_path.is_dir():
        print(f"ERROR: Datasets directory not found: {datasets_dir}")
        sys.exit(1)

    folders = []
    for entry in sorted(ds_path.iterdir()):
        if entry.is_dir() and entry.name.lower() not in exclude:
            folders.append(entry)

    return folders


def main():
    parser = argparse.ArgumentParser(description="Batch SwinFace embedding generation")
    parser.add_argument(
        "--datasets-dir", type=str, required=True,
        help="Parent directory containing dataset subfolders"
    )
    parser.add_argument(
        "--output-root", type=str, required=True,
        help="Root output directory (script appends each dataset name)"
    )
    parser.add_argument(
        "--weight", type=str, default=None,
        help="SwinFace checkpoint path (default: models/swinface/checkpoint_step_79999_gpu_0.pt)"
    )
    parser.add_argument(
        "--exclude", type=str, nargs="+", default=None,
        help="Folder names to exclude (default: aligned pairs labels original deidentified)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Batch size for inference"
    )

    args = parser.parse_args()

    # Resolve paths
    datasets_dir = os.path.abspath(args.datasets_dir)
    output_root = os.path.abspath(args.output_root)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    embed_script = os.path.join(script_dir, "generate_embeddings_swinface.py")

    if args.weight is None:
        args.weight = os.path.join(script_dir, "models", "swinface", "checkpoint_step_79999_gpu_0.pt")

    exclude = {e.lower() for e in args.exclude} if args.exclude else {
        "aligned", "pairs", "labels", "original", "deidentified"
    }

    # Discover folders
    folders = discover_dataset_folders(datasets_dir, exclude)

    if not folders:
        print("No dataset folders found (after exclusions).")
        sys.exit(0)

    print(f"SwinFace Batch Embedding Generation")
    print(f"=====================================")
    print(f"Datasets dir: {datasets_dir}")
    print(f"Output root:  {output_root}")
    print(f"Exclude:      {', '.join(sorted(exclude))}")
    print(f"Found {len(folders)} folder(s): {', '.join(f.name for f in folders)}")
    print()

    # Process each folder — output_root is passed as-is; the script auto-appends input folder levels
    script_dir_abs = os.path.abspath(script_dir)
    failed = []
    for idx, ds_folder in enumerate(folders, 1):
        ds_name = ds_folder.name

        # The single-folder script appends last 2 input path components to output.
        # So with --output {output_root}, input=datasets/X → output=output_root/datasets/X
        print(f"[{idx}/{len(folders)}] Processing: {ds_name}")
        print(f"  → {output_root}/... (auto-appended from input path)")

        cmd = [
            sys.executable, embed_script,
            "--input", str(ds_folder),
            "--output", output_root,
            "--weight", args.weight,
            "--batch-size", str(args.batch_size),
        ]

        result = subprocess.run(cmd, cwd=script_dir_abs)
        if result.returncode != 0:
            failed.append(ds_name)
            print(f"  ERROR: Failed (exit code {result.returncode})")
        else:
            print(f"  OK")
        print()

    # Summary
    print(f"Batch complete. Processed {len(folders)} folder(s).")
    if failed:
        print(f"Failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
