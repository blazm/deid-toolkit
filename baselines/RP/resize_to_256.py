"""One-off utility: in-place LANCZOS resize of any RP output directory tree to 256x256.

Usage:
    python resize_to_256.py <root_dir>     # e.g. "D:\dev\deid-toolkit\root_dir\datasets\RP"
Resizes every PNG under <root_dir> (recursively) that is not already 256x256.
Safe to re-run (already-resized files are skipped).
"""
import sys
from pathlib import Path
from PIL import Image

def main():
    if len(sys.argv) != 2:
        sys.exit(f"Usage: python {sys.argv[0]} <root_dir>")
    root = Path(sys.argv[1])
    if not root.is_dir():
        sys.exit(f"not a directory: {root}")

    done = skipped = 0
    for p in sorted(root.rglob("*.png")):
        im = Image.open(p)
        if im.size == (256, 256):
            skipped += 1
            continue
        im.convert("RGB").resize((256, 256), Image.LANCZOS).save(p, "PNG")
        done += 1
        print(f"resized {p.relative_to(root)} ({im.size} -> 256x256)")
    print(f"\nDone: {done} resized, {skipped} already 256x256.")

if __name__ == "__main__":
    main()
