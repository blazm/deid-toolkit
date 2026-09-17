#!/usr/bin/env python3
"""
Batch verification — ROC/AUC for all anonymization techniques in one report.

Scans an embeddings root directory (e.g., .../embeddings/SwinFace), finds each
technique's celeba-test embeddings, runs verification against celeba-test pairs,
and produces a combined HTML report with all ROC curves on a single plot.

Only processes folders named 'celeba-test' (skips fri, mug-still, *_reversed).

Usage:
    python batch_verify_all.py ^
        --embeddings-root D:\dev\deid-toolkit\root_dir\embeddings\SwinFace ^
        --pairs           D:\dev\deid-toolkit\root_dir\datasets\pairs ^
        --output          verification_swinface_all.html
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np


def load_celeba_embeddings(celeba_test_dir):
    """Load all .npy files from a celeba-test embedding folder.

    Returns dict: {image_name_without_ext: embedding_array}
    e.g. {"00002": array([...], dtype=float32)}
    """
    cdir = Path(celeba_test_dir)
    if not cdir.is_dir():
        return {}

    embeddings = {}
    for npy_path in sorted(cdir.glob("*.npy")):
        name = npy_path.stem  # e.g. "00002"
        emb = np.load(npy_path)
        embeddings[name] = emb

    return embeddings


def parse_pairs_file(pairs_path):
    """Parse a pairs file. Format per line: <id1> <image1.jpg> <id2> <image2.jpg>"""
    pairs = []
    with open(pairs_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                img1 = os.path.splitext(parts[1])[0]
                img2 = os.path.splitext(parts[3])[0]
                pairs.append((img1, img2))
    return pairs


def compute_scores(embeddings, genuine_pairs, impostor_pairs):
    """Compute cosine similarity for L2-normalized embeddings."""
    g_scores, i_scores = [], []
    g_missing, i_missing = 0, 0

    for img1, img2 in genuine_pairs:
        if img1 in embeddings and img2 in embeddings:
            g_scores.append(float(np.dot(embeddings[img1].flatten(), embeddings[img2].flatten())))
        else:
            g_missing += 1

    for img1, img2 in impostor_pairs:
        if img1 in embeddings and img2 in embeddings:
            i_scores.append(float(np.dot(embeddings[img1].flatten(), embeddings[img2].flatten())))
        else:
            i_missing += 1

    return np.array(g_scores), np.array(i_scores), g_missing, i_missing


def compute_roc(genuine_scores, impostor_scores):
    """Compute ROC curve and AUC."""
    labels = np.concatenate([np.ones(len(genuine_scores)), np.zeros(len(impostor_scores))])
    scores = np.concatenate([genuine_scores, impostor_scores])

    order = np.argsort(-scores)
    sorted_labels = labels[order]

    total_pos = sorted_labels.sum()
    total_neg = len(sorted_labels) - total_pos

    tp = np.cumsum(sorted_labels)
    fp = np.cumsum(1 - sorted_labels)

    tpr = np.concatenate([[0.0], tp / total_pos])
    fpr = np.concatenate([[0.0], fp / total_neg])

    auc = float(np.trapezoid(tpr, fpr))
    return fpr, tpr, auc


def compute_eer(genuine_scores, impostor_scores):
    """Compute EER and accuracy at EER threshold."""
    labels = np.concatenate([np.ones(len(genuine_scores)), np.zeros(len(impostor_scores))])
    scores = np.concatenate([genuine_scores, impostor_scores])

    order = np.argsort(-scores)
    sorted_labels = labels[order]

    total_pos = sorted_labels.sum()
    total_neg = len(sorted_labels) - total_pos

    tp = np.cumsum(sorted_labels)
    fp = np.cumsum(1 - sorted_labels)

    far = fp / total_neg
    frr = 1.0 - tp / total_pos

    diff = np.abs(far - frr)
    eer_idx = int(np.argmin(diff))
    eer = float((far[eer_idx] + frr[eer_idx]) / 2)

    threshold = float(scores[order][eer_idx])

    correct = sum(1 for s, l in zip(scores, labels) if (1 if s >= threshold else 0) == int(l))
    accuracy = correct / len(labels)

    return eer, accuracy, threshold


def discover_techniques(embeddings_root):
    """Find all technique directories containing a celeba-test/ subfolder with .npy files.

    Structure examples:
        root/aligned/celeba-test/*.npy           -> technique = "aligned"
        root/datasets/AIDPro/celeba-test/*.npy   -> technique = "AIDPro"

    Skips any folder whose name contains '_reversed'.

    Returns list of (technique_name, celeba_test_path) sorted: aligned first, then alpha.
    """
    root = Path(embeddings_root)
    if not root.is_dir():
        raise NotADirectoryError(f"Embeddings root not found: {embeddings_root}")

    # Find all folders named exactly "celeba-test" that contain .npy files
    technique_map = {}  # technique_name -> celeba_test_path

    for npy in root.rglob("*.npy"):
        parent = npy.parent
        if parent.name != "celeba-test":
            continue
        # Skip _reversed variants (e.g. RiDDLE/celeba-test_reversed won't match anyway, but be safe)
        # Technique name is the grandparent of celeba-test (skip intermediate "datasets" if present)
        technique_dir = parent.parent  # e.g. "aligned" or "AIDPro" or "datasets"
        technique_name = technique_dir.name

        # If we found root/datasets/AIDPro/celeba-test, technique is AIDPro
        # If we found root/aligned/celeba-test, technique is aligned
        celeba_path = str(parent)

        if technique_name not in technique_map:
            technique_map[technique_name] = celeba_path

    if not technique_map:
        return []

    results = [(name, path) for name, path in technique_map.items()]
    # Sort: aligned first, then alphabetical
    results.sort(key=lambda x: (0 if x[0].lower() == "aligned" else 1, x[0]))
    return results


def generate_html_report(results, output_path):
    """Generate HTML report with combined ROC curve."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Colors: shared (color, marker) encoding from fig_style (aligned = black)
    from fig_style import color_for

    # Display name mapping — "aligned" → "Validation" for manuscript readability
    DISPLAY_NAME = {"aligned": "Validation"}

    def get_display_name(name):
        return DISPLAY_NAME.get(name, name)

    def get_color(name):
        return color_for(name)

    def display_label(r):
        """Human-friendly label used in legend, table, plot."""
        return get_display_name(r["name"])

    # Print table
    print()
    print(f"  {'Technique':<20} {'Genuine':>8} {'Impostor':>9} {'AUC':>8} {'EER':>8} {'Acc@EER':>8}")
    print(f"  {'-'*68}")

    sorted_results = sorted(results, key=lambda r: (0 if r["name"].lower() == "aligned" else 1, get_display_name(r["name"])))

    for r in sorted_results:
        print(f"  {display_label(r):<20} {r['n_genuine']:>8,d} {r['n_impostor']:>9,d} "
              f"{r['auc']:>8.4f} {r['eer']:>8.4f} {r['accuracy']:>8.4f}")

    # ROC plot — manuscript-ready: large fonts, high DPI, clean styling
    fig, ax = plt.subplots(figsize=(12, 9))

    for r in sorted_results:
        if r.get("fpr") is not None and len(r["fpr"]) > 1:
            name = r["name"]
            label = display_label(r)
            color = get_color(name)
            style = "--" if name.lower() == "aligned" else "-"
            lw = 3.0 if name.lower() == "aligned" else 2.5
            ax.plot(r["fpr"] * 100, r["tpr"] * 100, linewidth=lw, color=color, linestyle=style,
                    label=f"{label} (AUC={r['auc']:.4f})")

    ax.plot([0, 100], [0, 100], "k:", linewidth=0.8, alpha=0.4)
    ax.set_xlabel("False Positive Rate (%)", fontsize=16, fontweight="bold")
    ax.set_ylabel("True Positive Rate (%)", fontsize=16, fontweight="bold")
    ax.set_title("Face Verification — ROC Curves (All Techniques)", fontsize=18, fontweight="bold", pad=20)

    # Major ticks and labels at larger size
    ax.tick_params(axis="both", which="major", labelsize=14, length=6, direction="inout")
    ax.set_xlim(-1, 101)
    ax.set_ylim(-1, 101)
    ax.grid(True, alpha=0.25, linestyle="-", linewidth=0.8)

    # Legend — spans the whole lower-right quadrant: all ROC curves run above the
    # diagonal, so that quadrant stays empty. Box is anchored at the corner and
    # sized so the box approaches a square in display units: with an anchored
    # corner at the axes' lower-right, a square legend fills that quadrant.
    # Legend: larger fonts (13 pt, 2 cols, with per-technique AUC). The box is
    # anchored at the lower-right corner and extends left past mid-x (all ROC
    # curves run above the diagonal, so the region below the diagonal is empty).
    ax.legend(loc="lower right", fontsize=13, ncol=2, framealpha=0.95,
              edgecolor="#888888", handlelength=1.5, handletextpad=0.6,
              columnspacing=1.0, borderpad=0.8, labelspacing=0.3)
    fig.tight_layout()
    # Margin guard: verify no curve passes below the legend top within the
    # legend's x-range (should hold: all curves are above the diagonal there).
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    lb = ax.get_legend().get_window_extent(renderer)
    axb = ax.get_window_extent(renderer)
    span = (axb.width, axb.height)
    to_x = lambda px: -1 + (px - axb.x0) / span[0] * 102
    to_y = lambda py: -1 + (py - axb.y0) / span[1] * 102
    lx0, ly1 = to_x(lb.x0), to_y(lb.y1)
    worst = 101.0
    for r in sorted_results:
        if r.get("fpr") is None or len(r["fpr"]) < 2:
            continue
        xs = r["fpr"] * 100
        mask = xs >= lx0
        if mask.any():
            worst = min(worst, float(np.min(r["tpr"][mask]) * 100))
    print(f"  [legend] top edge y={ly1:.1f}%, lowest curve over x>={lx0:.1f}% is at {worst:.1f}% "
          f"(clearance {worst - ly1:.1f} percentage points)")
    assert worst - ly1 > 1.0, "legend would overlap a ROC curve - shrink font/spacing"

    # High-DPI export for manuscript (PNG + PDF + SVG vector + PNG thumbnail)
    base = os.path.splitext(output_path)[0] + "_roc"
    png_path = base + ".png"
    svg_path = base + ".svg"
    pdf_path = base + ".pdf"

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close(fig)

    # HTML report — use SVG for the ROC curve (vector, scales well), table with display names
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Batch Verification Results</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
       max-width: 1200px; margin: 2em auto; padding: 0 1em; }}
h1 {{ border-bottom: 2px solid #333; padding-bottom: 0.5em; }}
table {{ border-collapse: collapse; width: 100%; margin: 1em 0; }}
th, td {{ border: 1px solid #ddd; padding: 6px 10px; text-align: right; }}
th {{ background: #f5f5f5; text-align: center; }}
svg, img {{ max-width: 100%; height: auto; border: 1px solid #eee; border-radius: 4px; }}
</style></head><body>
<h1>Face Verification — Combined ROC / AUC Report</h1>
<table><tr><th>Technique</th><th>Genuine pairs</th><th>Impostor pairs</th><th>AUC</th><th>EER</th><th>Accuracy@EER</th></tr>"""

    for r in sorted_results:
        html += f"<tr><td style='text-align:left'>{display_label(r)}</td>"
        html += f"<td>{r['n_genuine']:,}</td><td>{r['n_impostor']:,}</td>"
        html += f"<td>{r['auc']:.4f}</td><td>{r['eer']:.4f}</td><td>{r['accuracy']:.4f}</td></tr>\n"

    # Embed SVG inline for the interactive HTML report
    try:
        with open(svg_path, "r", encoding="utf-8") as _sf:
            svg_content = _sf.read()
        svg_header = '<svg xmlns="http://www.w3.org/2000/svg"'
        if svg_content.startswith(svg_header):
            html += "\n" + svg_content
        else:
            html += f'\n<img src="{os.path.basename(png_path)}" alt="ROC Curves">'
    except Exception:
        html += f'\n<img src="{os.path.basename(png_path)}" alt="ROC Curves">'

    html += "\n</body></html>"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\nReport saved: {output_path}")
    print(f"ROC plot:     {png_path}  |  {svg_path}  |  {pdf_path}")


def main():
    parser = argparse.ArgumentParser(description="Batch verification — combined ROC/AUC report")
    parser.add_argument("--embeddings-root", type=str, required=True,
                        help="Root embeddings directory (e.g., .../embeddings/SwinFace)")
    parser.add_argument("--pairs", type=str, required=True,
                        help="Pairs directory (contains celeba-test_*_pairs.txt)")
    parser.add_argument("--output", type=str, default="verification_all.html",
                        help="Output HTML report path")

    args = parser.parse_args()
    embeddings_root = os.path.abspath(args.embeddings_root)
    pairs_dir = os.path.abspath(args.pairs)
    output_path = os.path.abspath(args.output)

    print("=" * 60)
    print("Batch Verification — Combined ROC / AUC")
    print("=" * 60)
    print(f"Embeddings root: {embeddings_root}")
    print(f"Pairs dir:       {pairs_dir}")
    print(f"Output:          {output_path}")

    # Load celeba-test pairs only
    genuine_path = os.path.join(pairs_dir, "celeba-test_genuine_pairs.txt")
    impostor_path = os.path.join(pairs_dir, "celeba-test_impostor_pairs.txt")

    if not os.path.exists(genuine_path) or not os.path.exists(impostor_path):
        print("ERROR: celeba-test pair files not found in pairs directory.")
        sys.exit(1)

    genuine_pairs = parse_pairs_file(genuine_path)
    impostor_pairs = parse_pairs_file(impostor_path)
    print(f"CelebA-test pairs: {len(genuine_pairs)} genuine, {len(impostor_pairs)} impostor\n")

    # Discover techniques
    techniques = discover_techniques(embeddings_root)
    if not techniques:
        print("ERROR: No technique folders found.")
        sys.exit(1)

    print(f"Found {len(techniques)} technique(s):")
    for name, _ in techniques:
        print(f"  - {name}")
    print()

    results = []

    for tech_name, celeba_path in techniques:
        print(f"Loading embeddings: {tech_name} ({celeba_path})")

        embeddings = load_celeba_embeddings(celeba_path)
        if not embeddings:
            print(f"  SKIP — no .npy files found")
            continue

        print(f"  Loaded {len(embeddings)} embedding(s)")

        g_scores, i_scores, g_miss, i_miss = compute_scores(
            embeddings, genuine_pairs, impostor_pairs)

        if len(g_scores) == 0 or len(i_scores) == 0:
            print(f"  SKIP — no matching pairs (g_missing={g_miss}, i_missing={i_miss})")
            continue

        if g_miss > 0 or i_miss > 0:
            print(f"  Note: {g_miss} genuine + {i_miss} impostor pairs skipped (missing embeddings)")

        fpr, tpr, auc = compute_roc(g_scores, i_scores)
        eer, accuracy, threshold = compute_eer(g_scores, i_scores)

        results.append({
            "name": tech_name,
            "fpr": fpr, "tpr": tpr,
            "auc": auc,
            "eer": eer,
            "accuracy": accuracy,
            "n_genuine": len(g_scores),
            "n_impostor": len(i_scores),
        })

    if not results:
        print("\nNo results to report.")
        sys.exit(1)

    generate_html_report(results, output_path)


if __name__ == "__main__":
    main()
