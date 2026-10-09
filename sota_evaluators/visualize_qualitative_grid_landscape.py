#!/usr/bin/env python3
"""
Landscape double-column qualitative grid for the two-column manuscript.

Splits the 20 techniques into two halves and renders them side by side as a
single wide canvas (intended for a double-column ``figure*``):

    left  half:  Reference + techniques 1-10   (11 rows x 7 subjects)
    right half:  Reference + techniques 11-20  (11 rows x 7 subjects)

Same visual language as visualize_qualitative_grid.py (vertical):
techniques = rows with bold left labels, subjects = columns.

Creates NEW output files only -- the original vertical grid is untouched.

Usage:
    python visualize_qualitative_grid_landscape.py ^
        --ref-dir     D:\\dev\\deid-toolkit\\root_dir\\datasets\\aligned\\fri ^
        --baselines   D:\\dev\\deid-toolkit\\root_dir\\datasets ^
        --subjects    BlazMeden JernejSabadin ZigaEmersic MatejVitek PeterPeer VitomirStruc2 KristijanLenac SandiLjubic ^
        --cell-size   256 256 ^
        --output      qualitative_comparison_landscape_2col
"""

import argparse
import os
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


# ── shared helpers (mirrors of visualize_qualitative_grid.py) ───────────────

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
EXT_PRIORITY = [".png", ".PNG", ".tiff", ".tif", ".bmp", ".jpg", ".jpeg"]


def discover_extensions(directory):
    exts = set()
    for p in Path(directory).iterdir():
        if p.is_file():
            _, ext = os.path.splitext(p.name)
            if ext:
                exts.add(ext.lower())
    return sorted(exts)


def find_image(directory, stem):
    d = Path(directory)
    if not d.is_dir():
        return None
    for ext in EXT_PRIORITY:
        candidate = d / (stem + ext.lower())
        if candidate.exists():
            return candidate
    for ext in discover_extensions(d):
        candidate = d / (stem + ext)
        if candidate.exists():
            return candidate
    return None


def get_subject_stems(ref_dir):
    stems = []
    for p in Path(ref_dir).iterdir():
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
            stems.append(p.stem)
    return sorted(stems)


def load_resized(path, target_size):
    return Image.open(path).convert("RGB").resize(target_size, Image.LANCZOS)


def _load_font(size=13, bold=False):
    candidates = [
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    bold_candidates = [
        "C:/Windows/Fonts/arialbd.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]
    for path in (bold_candidates if bold else candidates) + candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


# ── half grid assembly ──────────────────────────────────────────────────────

def draw_half(canvas, draw, x_off, row_labels, grid, cell_w, cell_h, font_bold,
              border, pad, label_w):
    """Draw one half: bold row labels in a left gutter, then the image grid.

    Returns the half's total width (gutter + grid + outer border).
    """
    n_rows, n_cols = len(row_labels), len(grid[0])
    half_w = border + label_w + n_cols * (cell_w + pad) - pad
    # x_off positions the half so it starts at the canvas left edge (or after
    # a previous half + gap); the border here is the inter-half margin.
    x0 = x_off + label_w  # first image column

    for r, row_name in enumerate(row_labels):
        y_img = border + r * (cell_h + pad)
        bbox = draw.textbbox((0, 0), row_name, font=font_bold)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        lx = x_off + label_w - tw - 4          # flush right, close to images
        ly = y_img + cell_h // 2 - th // 2
        draw.text((lx, ly), row_name, fill="#222222", font=font_bold)
        for c in range(n_cols):
            x = x0 + c * (cell_w + pad)
            canvas.paste(grid[r][c], (x, y_img))
            draw.rectangle([x, y_img, x + cell_w - 1, y_img + cell_h - 1],
                           outline="#cccccc", width=1)
    return half_w


def label_gutter_width(names, font_bold):
    dummy = Image.new("RGB", (1, 1), "white")
    d = ImageDraw.Draw(dummy)
    wmax = max(d.textbbox((0, 0), n, font=font_bold)[2]
               for n in names)
    return max(140, wmax + 32)


# ── main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ref-dir", required=True)
    ap.add_argument("--baselines", required=True)
    ap.add_argument("--subjects", nargs="+", required=True)
    ap.add_argument("--cell-size", type=int, nargs=2, default=[256, 256])
    ap.add_argument("--output", default="qualitative_comparison_landscape_2col",
                    help="Output base name (without extension); saves .pdf/.png")
    args = ap.parse_args()

    ref_dir, baselines_dir = Path(args.ref_dir), Path(args.baselines)
    if not ref_dir.is_dir():
        sys.exit(f"ERROR: reference dir not found: {ref_dir}")

    # ── subjects ──
    all_stems = set(get_subject_stems(ref_dir))
    subject_stems = [s for s in args.subjects if s in all_stems]
    for s in args.subjects:
        if s not in all_stems:
            print(f"WARNING: subject '{s}' not in reference dir, skipped.")
    if not subject_stems:
        sys.exit("ERROR: no valid subjects")

    # ── baselines ── same discovery order as the vertical generator ──
    baseline_names = []
    for d in sorted(baselines_dir.iterdir()):
        if not d.is_dir() or d.name in ("aligned", "original", "labels",
                                        "pairs", "deidentified"):
            continue
        fri = d / "fri"
        if not fri.is_dir():
            continue
        if any(find_image(fri, s) is not None for s in subject_stems):
            baseline_names.append(d.name)
    baseline_names = [b for b in baseline_names if b != "aligned"]

    n = len(baseline_names)
    if n % 2 != 0:
        print(f"WARNING: {n} techniques (odd) -- halves split {n // 2}/{n - n // 2}")
    left_names, right_names = baseline_names[:n // 2], baseline_names[n // 2:]
    print(f"Subjects (cols):  {len(subject_stems)}")
    print(f"Left  half rows:  Reference + {len(left_names)}  -> {left_names}")
    print(f"Right half rows:  Reference + {len(right_names)} -> {right_names}")

    cell_w, cell_h = args.cell_size
    border, pad = 16, 8
    gap = 24  # margin between the two halves
    target = (cell_w, cell_h)

    # row count = max of the two halves so both are the same height
    n_rows = max(len(left_names), len(right_names)) + 1
    canvas_h = border * 2 + n_rows * (cell_h + pad) - pad
    cell_cols = len(subject_stems)
    grid_w = cell_cols * (cell_w + pad) - pad

    font_bold = _load_font(max(32, min(48, int(cell_w / 8))), bold=True)
    lw_left = label_gutter_width(["reference"] + left_names, font_bold)
    lw_right = label_gutter_width(["reference"] + right_names, font_bold)
    canvas_w = (border + lw_left + grid_w) + gap + (lw_right + grid_w + border)

    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)

    def load_row(names):
        rows = []
        for b in ["Reference"] + names:
            row = []
            for s in subject_stems:
                src = ref_dir if b == "Reference" else baselines_dir / b / "fri"
                p = find_image(src, s)
                row.append(load_resized(p, target) if p
                           else Image.new("RGB", target, "#f0f0f0"))
            rows.append(row)
        return rows

    draw_half(canvas, draw, 0, ["reference"] + left_names, load_row(left_names),
              cell_w, cell_h, font_bold, border, pad, lw_left)
    draw_half(canvas, draw, border + lw_left + grid_w + gap,
              ["reference"] + right_names, load_row(right_names),
              cell_w, cell_h, font_bold, 0, pad, lw_right)

    out_base = Path(args.output)
    if out_base.suffix:
        png_path = out_base.with_suffix(".png")
        pdf_path = out_base.with_suffix(".pdf")
    else:
        png_path = out_base.with_name(out_base.name + ".png")
        pdf_path = out_base.with_name(out_base.name + ".pdf")
    canvas.save(png_path, "PNG", optimize=True)
    canvas.convert("RGB").save(pdf_path, "PDF", resolution=150)
    print(f"Saved: {png_path}  and  {pdf_path}  ({canvas.size[0]}x{canvas.size[1]} px)")


if __name__ == "__main__":
    main()
