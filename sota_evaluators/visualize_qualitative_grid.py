#!/usr/bin/env python3
"""
Qualitative Grid Visualization — Compare baseline outputs against reference images.

Reads a set of subject images from an aligned (reference) dataset folder, then looks
for matching anonymized versions under each baseline's {dataset}/ subfolder.  Assembles
a labelled grid suitable for manuscript inclusion (PNG + PDF).

Usage:
    python visualize_qualitative_grid.py ^
        --ref-dir     D:\dev\deid-toolkit\root_dir\datasets\aligned\fri ^
        --baselines   D:\dev\deid-toolkit\root_dir\datasets ^
        --subjects    AjdaLampe AlesJaklic BlazMeden ^
        --orientation horizontal  | vertical ^
        --output      qualitative_comparison.pdf

Orientation:
    horizontal — baseline names on top, subjects down rows (wide landscape)
    vertical   — baseline names along side, subjects across columns (tall portrait)
"""

import argparse
import os
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


# ── Image helpers ───────────────────────────────────────────────────────

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

# Priority order for extension matching (lossless first)
EXT_PRIORITY = [".png", ".PNG", ".tiff", ".tif", ".bmp", ".jpg", ".jpeg"]


def discover_extensions(directory):
    """Return sorted list of unique file extensions in a directory, prioritised lossless first."""
    exts = set()
    for p in Path(directory).iterdir():
        if p.is_file():
            _, ext = os.path.splitext(p.name)
            if ext:
                exts.add(ext.lower())
    return sorted(exts)


def find_image(directory, stem):
    """Find the first image file matching *stem* (filename without extension).

    Tries extensions in quality-priority order: .png → .tiff → .bmp → .jpg.
    Returns Path or None.
    """
    d = Path(directory)
    if not d.is_dir():
        return None
    # Try priority order first (user might want .png over .jpg for BlazMeden)
    for ext in EXT_PRIORITY:
        ext_lower = ext.lower()
        candidate = d / (stem + ext_lower)
        if candidate.exists():
            return candidate
    # Fall back to whatever is in the directory
    for ext in discover_extensions(d):
        candidate = d / (stem + ext)
        if candidate.exists():
            return candidate
        if candidate.exists():
            return candidate
    return None


def get_subject_stems(ref_dir):
    """Return sorted list of stems (basename without extension) from the reference directory."""
    stems = []
    for p in Path(ref_dir).iterdir():
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
            stems.append(p.stem)
    return sorted(stems)


def load_resized(path, target_size):
    """Load an image as RGB, resized to (width, height)."""
    return Image.open(path).convert("RGB").resize(target_size, Image.LANCZOS)


# ── Grid assembly ───────────────────────────────────────────────────────

def _load_font(size=13, bold=False):
    """Try to load a TrueType font for labels."""
    candidates = [
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    bold_candidates = [
        "C:/Windows/Fonts/arialbd.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]
    font_list = bold_candidates if bold else candidates
    for path in font_list + candidates:  # try bold first, fall back to regular
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def assemble_grid(images, row_stems, col_names, horizontal=True, label_align="right"):
    """Assemble images into a single canvas with labels.

    Horizontal layout:  baseline names BELOW grid (row_labels unused)
    Vertical layout:    baseline names on LEFT side (col_labels unused)

    Parameters
    ----------
    images : list[list[PIL.Image]]
        images[row][col] — all already resized to the same dimensions.
    row_stems : list[str]
        For horizontal: subject names (unused — no row labels).
        For vertical: baseline names shown as LEFT row labels.
    col_names : list[str]
        For horizontal: baseline names shown below grid.
        For vertical: subject names (unused — no top/bottom column labels).
    labels_on_top : bool
        True  → horizontal layout, col_names shown below grid
        False → vertical layout, row_stems shown as LEFT labels

    Returns
    -------
    PIL.Image RGB canvas
    """
    n_rows = len(row_stems)
    n_cols = len(col_names)

    # Cell size from first image
    cell_w, cell_h = images[0][0].size

    border = 16   # outer padding around entire grid
    pad = 8       # gap between cells

    label_color = "#222222"

    # Font sizing — large enough to read at manuscript resolution
    label_font_size = max(32, min(48, int(cell_w / 8)))
    font_normal = _load_font(label_font_size)
    font_bold = _load_font(label_font_size, bold=True)

    # Measure label text on dummy canvas for sizing (using normal font for height reference)
    _dummy = Image.new("RGB", (1, 1), "white")
    _d = ImageDraw.Draw(_dummy)
    bbox_m = _d.textbbox((0, 0), "M", font=font_normal)
    _dh = bbox_m[3] - bbox_m[1] + 8

    canvas_h = border * 2 + n_rows * (cell_h + pad) - pad  # base height without labels
    canvas_w = border * 2 + n_cols * (cell_w + pad) - pad   # base width without labels

    if horizontal:
        # Horizontal: extra space below for baseline column labels
        bottom_label_h = _dh + 28
        canvas_h += bottom_label_h
    else:
        # Vertical: measure all row label widths to find the widest, then set left column width
        bbox_max = (0, 0, 0, 0)
        for name in row_stems:
            bb = _d.textbbox((0, 0), name, font=font_bold)
            if bb[2] - bb[0] > bbox_max[2] - bbox_max[0]:
                bbox_max = bb
        label_text_w = bbox_max[2] - bbox_max[0]
        # Left column: text + generous padding so labels don't clip into image area
        side_label_w = max(140, label_text_w + 32)
        canvas_w += side_label_w

    del _dummy, _d

    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)

    if horizontal:
        # ── Horizontal layout (labels at BOTTOM) ─────────────────
        y0 = border          # first image row starts at top border
        for r in range(n_rows):
            for c in range(n_cols):
                img = images[r][c]
                x = border + c * (cell_w + pad)
                y = y0 + r * (cell_h + pad)

                canvas.paste(img, (x, y))

                draw.rectangle(
                    [x, y, x + cell_w - 1, y + cell_h - 1],
                    outline="#cccccc", width=1,
                )

        # Column labels below grid
        for c, name in enumerate(col_names):
            cx = border + c * (cell_w + pad) + cell_w // 2
            bbox = draw.textbbox((0, 0), name, font=font_normal)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            text_y = canvas_h - (th + 8)
            draw.text((cx - tw // 2, text_y), name, fill=label_color, font=font_normal)

    else:
        # ── Vertical layout (row labels on LEFT side, horizontal) ───────
        x0 = border + side_label_w   # first image column starts after left labels

        for r in range(n_rows):
            row_name = row_stems[r]
            y_img = border + r * (cell_h + pad)

            # Measure label text to center vertically within the cell
            bbox = draw.textbbox((0, 0), row_name, font=font_bold)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]

            ly = y_img + cell_h // 2 - th // 2
            if label_align == "right":
                lx = border + side_label_w - tw - 4  # flush right, close to images
            else:
                lx = border  # flush left
            draw.text((lx, ly), row_name, fill=label_color, font=font_bold)

            for c in range(n_cols):
                img = images[r][c]
                x = x0 + c * (cell_w + pad)
                y = y_img

                canvas.paste(img, (x, y))

                draw.rectangle(
                    [x, y, x + cell_w - 1, y + cell_h - 1],
                    outline="#cccccc", width=1,
                )

    return canvas


# ── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Qualitative grid: reference vs. baseline anonymized images")
    parser.add_argument("--ref-dir", required=True,
                        help="Reference (aligned) directory, e.g. …/datasets/aligned/fri")
    parser.add_argument("--baselines", required=True,
                        help="Parent directory with per-baseline subfolders (e.g. …/datasets)")
    parser.add_argument("--subjects", nargs="+", default=None,
                        help="Subject stems to include. Default: all reference files.")
    parser.add_argument("--orientation", choices=["horizontal", "vertical"], default="horizontal")
    parser.add_argument("--cell-size", type=int, nargs=2, default=[256, 256],
                        metavar=("W", "H"), help="Cell dimensions WxH (default: 256 256, square)")
    parser.add_argument("--label-align", choices=["left", "right"], default="right",
                        help="Vertical mode: horizontal text alignment for left-side labels (default: right)")
    parser.add_argument("--output", default="qualitative_comparison.pdf")

    args = parser.parse_args()

    ref_dir = Path(args.ref_dir)
    baselines_dir = Path(args.baselines)

    if not ref_dir.is_dir():
        print(f"ERROR: Reference directory not found: {ref_dir}")
        sys.exit(1)

    # ── Subjects ────────────────────────────────────────────────────
    all_stems = get_subject_stems(ref_dir)
    if not all_stems:
        print(f"ERROR: No images in {ref_dir} (expected .jpg/.png)")
        sys.exit(1)

    if args.subjects:
        available = set(all_stems)
        for s in args.subjects:
            if s not in available:
                print(f"WARNING: Subject '{s}' not found in reference dir, skipping.")
        subject_stems = [s for s in args.subjects if s in available]
    else:
        subject_stems = all_stems

    if not subject_stems:
        print("ERROR: No valid subjects."); sys.exit(1)

    # ── Baselines ───────────────────────────────────────────────────
    baseline_names = []
    for d in sorted(baselines_dir.iterdir()):
        if not d.is_dir():
            continue
        bname = d.name
        # Skip non-anonymization dirs (reference, originals, metadata)
        if bname in ("aligned", "original", "labels", "pairs", "deidentified"):
            continue
        fri = d / "fri"
        if not fri.is_dir():
            continue
        # Only include if at least one matching subject exists
        if any(find_image(fri, s) is not None for s in subject_stems):
            baseline_names.append(bname)

    # Always put "aligned" first (shown as "Reference")
    baseline_names = ["aligned"] + [b for b in baseline_names if b != "aligned"]
    col_names = ["Reference"] + [b for b in baseline_names if b != "aligned"]

    print(f"Subjects:      {len(subject_stems)}")
    print(f"Baselines:     {len(col_names) - 1} anonymized + Reference")
    print(f"Orientation:   {args.orientation}")
    print(f"Cell size:     {args.cell_size[0]}x{args.cell_size[1]}")

    # ── Load & assemble ─────────────────────────────────────────────
    target_size = (args.cell_size[0], args.cell_size[1])

    if args.orientation == "horizontal":
        # Reference on left, subjects as rows, baselines as columns
        images_grid = []
        for stem in subject_stems:
            row_imgs = []
            ref_path = find_image(ref_dir, stem)
            if ref_path:
                row_imgs.append(load_resized(ref_path, target_size))
            else:
                row_imgs.append(Image.new("RGB", target_size, "#f0f0f0"))
            for bname in baseline_names:
                if bname == "aligned":
                    continue
                fri = baselines_dir / bname / "fri"
                img_path = find_image(fri, stem)
                if img_path:
                    row_imgs.append(load_resized(img_path, target_size))
                else:
                    row_imgs.append(Image.new("RGB", target_size, "#f0f0f0"))
            images_grid.append(row_imgs)

        # col_names = ["Reference", "AIDPro", "AMT-GAN", ...]
        canvas = assemble_grid(images_grid, subject_stems, col_names, horizontal=True)

    else:
        # Vertical: baselines as ROWS (labeled on left), subjects as COLUMNS (labeled on top)
        # Structure: each row = one baseline, showing all 7 subjects side by side
        images_grid_v = []

        for bname in baseline_names:
            row_imgs = []
            for stem in subject_stems:
                if bname == "aligned":
                    img_path = find_image(ref_dir, stem)
                else:
                    fri = baselines_dir / bname / "fri"
                    img_path = find_image(fri, stem)

                if img_path:
                    row_imgs.append(load_resized(img_path, target_size))
                else:
                    row_imgs.append(Image.new("RGB", target_size, "#f0f0f0"))

            images_grid_v.append(row_imgs)

        # Row labels = baseline names, top labels = ["Reference"] + subjects
        col_labels_v = ["Reference"] + list(subject_stems)
        canvas = assemble_grid(images_grid_v, baseline_names, subject_stems,
                               horizontal=False, label_align=args.label_align)

    # ── Save ────────────────────────────────────────────────────────
    out_path = Path(args.output)
    ext = out_path.suffix.lower()

    if ext == ".png":
        canvas.save(out_path, "PNG", optimize=True)
    elif ext == ".pdf":
        rgb = canvas.convert("RGB")
        rgb.save(out_path, "PDF", resolution=150)
    else:
        # Fallback — try whatever PIL supports
        canvas.save(out_path)

    print(f"\nSaved: {os.path.abspath(out_path)}  ({canvas.size[0]}x{canvas.size[1]} px)")


if __name__ == "__main__":
    main()
