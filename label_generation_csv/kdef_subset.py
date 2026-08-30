"""Create a representative subset of the KDEF dataset for faster baseline runs.

Strategy: one random shot per expression, from every subject (140 subjects × 7 expr = ~980 images).
Keeps full coverage of all expressions and identities while cutting runtime roughly in half.

Usage:
    python kdef_subset.py [--seed 42]
"""
import argparse
import os
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Create a representative KDEF subset (1 shot/expr, all subjects)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--input", required=True, help="Path to aligned/kdef/")
    parser.add_argument("--output", required=True, help="Output directory for the subset")
    args = parser.parse_args()

    np.random.seed(args.seed)
    input_dir = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Group files: subject -> expression -> [filenames]
    groups = defaultdict(lambda: defaultdict(list))
    for f in sorted(input_dir.iterdir()):
        if not f.is_file():
            continue
        parts = os.path.splitext(f.name)[0].split('_')
        if len(parts) != 3:
            continue
        expr, subj, shot = parts
        groups[subj][expr].append(f.name)

    total_selected = 0
    total_skipped = 0
    skipped_subjects = []

    for subj in sorted(groups.keys()):
        selected_subj_dir = output_dir / subj
        selected_subj_dir.mkdir(parents=True, exist_ok=True)
        subj_kept = 0
        subj_dropped = 0

        for expr in sorted(groups[subj].keys()):
            files = groups[subj][expr]
            if len(files) == 1:
                selected = files[0]
            else:
                selected = np.random.choice(files, size=1)[0]
            shutil.copy2(input_dir / selected, selected_subj_dir / selected)
            subj_kept += 1

        for expr in groups[subj]:
            total_skipped += len(groups[subj][expr]) - (1 if expr in locals() and selected else 0)

        total_selected += subj_kept
        skipped = sum(len(files) - 1 for files in groups[subj].values())
        total_skipped += skipped

    print(f"Subset created: {output_dir}")
    print(f"Selected: {total_selected} images (all subjects, 1 shot/expr)")
    print(f"Dropped:  {total_skipped} images")


if __name__ == "__main__":
    main()
