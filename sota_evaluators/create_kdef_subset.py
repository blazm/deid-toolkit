"""Create a representative KDEF subset for faster baseline runs.

From all 140 subjects, randomly select one shot per expression (7 expressions).
Total: ~980 images instead of 2,934 (~67% reduction).

Usage:
    python create_kdef_subset.py [--seed 42]
"""
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path


def main():
    seed = 42
    random.seed(seed)

    aligned_root = Path(r"D:\dev\deid-toolkit\root_dir\datasets\aligned")
    kdef_src = aligned_root / "kdef"
    kdef_subset = aligned_root / "kdef_subset"

    if not kdef_src.is_dir():
        print(f"ERROR: {kdef_src} does not exist")
        return

    # Group files by (subject, expression) -> list of filenames
    groups = defaultdict(list)  # (subj, expr) -> [filename]
    for f in sorted(kdef_src.iterdir()):
        if not f.is_file() or not f.suffix.lower() == ".jpg":
            continue
        parts = f.name.rsplit(".", 1)[0].split("_")
        if len(parts) != 3:
            continue
        expr, subj, shot = parts
        groups[(subj, expr)].append(f.name)

    print(f"Found {sum(len(v) for v in groups.values())} images from {len(set(s[0] for s in groups))} subjects")
    print(f"Expressions: {', '.join(sorted(set(e for _, e in groups)))}")

    # Create output directory
    kdef_subset.mkdir(parents=True, exist_ok=True)

    selected_count = 0
    dropped_count = 0

    for key in sorted(groups.keys()):
        files = groups[key]
        if len(files) == 1:
            selected = files[0]
        else:
            selected = random.choice(files)
        shutil.copy2(kdef_src / selected, kdef_subset / selected)
        selected_count += 1
        dropped_count += len(files) - 1

    print(f"\nSubset created: {kdef_subset}")
    print(f"Selected: {selected_count} images (1 shot/expr from each subject)")
    if selected_count + dropped_count > 0:
        print(f"Dropped:  {dropped_count} images ({dropped_count/(selected_count+dropped_count)*100:.0f}% reduction)")


if __name__ == "__main__":
    main()
