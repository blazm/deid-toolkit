#!/usr/bin/env python3
"""
Face Verification — ROC / AUC from precomputed embeddings.

Reads genuine + impostor pair files, looks up embeddings (.npy),
computes cosine similarity, and produces ROC curves with AUC scores.

Usage:
    python verify_embeddings.py ^
        --embeddings D:\dev\deid-toolkit\root_dir\embeddings\SwinFace ^
        --pairs      D:\dev\deid-toolkit\root_dir\datasets\pairs ^
        --output     verification_results.html

    python verify_embeddings.py ^
        --embeddings D:\dev\deid-toolkit\root_dir\embeddings\TransFace ^
        --pairs      D:\dev\deid-toolkit\root_dir\datasets\pairs ^
        --output     verification_results.html

Works with either SwinFace or TransFace embeddings (both are 512-d, L2-normalized).
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np


def load_all_embeddings(embeddings_dir):
    """Load all .npy files from a directory (recursive).

    Returns dict: {relative_name_without_ext: embedding_array}
    e.g. {"000001": array([0.01, ...], dtype=float32)}
    """
    emb_dir = Path(embeddings_dir)
    if not emb_dir.is_dir():
        raise NotADirectoryError(f"Embeddings directory not found: {embeddings_dir}")

    embeddings = {}
    for npy_path in sorted(emb_dir.rglob("*.npy")):
        rel = npy_path.relative_to(emb_dir)
        key = str(rel).replace(os.sep, "/")  # normalize separators
        name_without_ext = os.path.splitext(key)[0]
        emb = np.load(npy_path)
        embeddings[name_without_ext] = emb

    return embeddings


def parse_pairs_file(pairs_path):
    """Parse a pairs file.

    Format per line: <id1> <image1.jpg> <id2> <image2.jpg>
    (fields separated by whitespace; id is ignored, images are filenames)

    Returns list of (image1_name, image2_name) tuples.
    """
    pairs = []
    with open(pairs_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                img1 = os.path.splitext(parts[1])[0]  # strip .jpg/.png
                img2 = os.path.splitext(parts[3])[0]
                pairs.append((img1, img2))
    return pairs


def cosine_similarity(emb1, emb2):
    """Cosine similarity between two vectors (works for L2-normalized embeddings)."""
    return np.dot(emb1.flatten(), emb2.flatten())


def compute_scores(embeddings, genuine_pairs, impostor_pairs):
    """Compute similarity scores for genuine and impostor pairs.

    Returns (genuine_scores, impostor_scores) as numpy arrays.
    Pairs where an embedding is missing are skipped.
    """
    g_scores = []
    g_missing = 0
    for img1, img2 in genuine_pairs:
        if img1 in embeddings and img2 in embeddings:
            score = cosine_similarity(embeddings[img1], embeddings[img2])
            g_scores.append(float(score))
        else:
            g_missing += 1

    i_scores = []
    i_missing = 0
    for img1, img2 in impostor_pairs:
        if img1 in embeddings and img2 in embeddings:
            score = cosine_similarity(embeddings[img1], embeddings[img2])
            i_scores.append(float(score))
        else:
            i_missing += 1

    if g_missing:
        print(f"  WARNING: {g_missing} genuine pair(s) missing embeddings, skipped.")
    if i_missing:
        print(f"  WARNING: {i_missing} impostor pair(s) missing embeddings, skipped.")

    return np.array(g_scores), np.array(i_scores)


def compute_roc(genuine_scores, impostor_scores):
    """Compute ROC curve points (FPR, TPR) and AUC using the trapezoidal rule.

    Returns (fpr, tpr, auc).
    """
    # Combine all scores with labels: 1 = genuine (positive), 0 = impostor (negative)
    labels = np.concatenate([np.ones(len(genuine_scores)), np.zeros(len(impostor_scores))])
    scores = np.concatenate([genuine_scores, impostor_scores])

    # Sort by score descending (high similarity → genuine)
    order = np.argsort(-scores)
    sorted_labels = labels[order]

    # Compute cumulative TPR and FPR
    total_pos = sorted_labels.sum()
    total_neg = len(sorted_labels) - total_pos

    tp = np.cumsum(sorted_labels)
    fp = np.cumsum(1 - sorted_labels)

    tpr = tp / total_pos
    fpr = fp / total_neg

    # Prepend origin
    fpr = np.concatenate([[0.0], fpr])
    tpr = np.concatenate([[0.0], tpr])

    # AUC via trapezoidal rule
    auc = float(np.trapezoid(tpr, fpr))

    return fpr, tpr, auc


def compute_metrics(genuine_scores, impostor_scores):
    """Compute verification metrics: AUC, EER, accuracy @ threshold."""
    # AUC
    _, _, auc = compute_roc(genuine_scores, impostor_scores)

    # EER (Equal Error Rate) — find threshold where FPR ≈ TPR
    labels = np.concatenate([np.ones(len(genuine_scores)), np.zeros(len(impostor_scores))])
    scores = np.concatenate([genuine_scores, impostor_scores])
    order = np.argsort(-scores)
    sorted_labels = labels[order]

    total_pos = sorted_labels.sum()
    total_neg = len(sorted_labels) - total_pos
    tp = np.cumsum(sorted_labels)
    fp = np.cumsum(1 - sorted_labels)

    tpr_arr = tp / total_pos
    fpr_arr = fp / total_pos  # note: use total_pos for FAR to find crossing

    # EER: where FPR crosses TPR (actually FAR = FPR_neg = fp/total_neg)
    far_arr = fp / total_neg  # false accept rate
    # Find the point where |far - frr| is minimized
    frr_arr = 1.0 - tpr_arr  # false reject rate
    diff = np.abs(far_arr - frr_arr)
    eer_idx = np.argmin(diff)
    eer = float((far_arr[eer_idx] + frr_arr[eer_idx]) / 2)

    # Verification accuracy at EER threshold
    scores_sorted = scores[order]
    eer_threshold = scores_sorted[eer_idx]
    correct = 0
    total = len(labels)
    for i in range(total):
        pred = 1 if scores[i] >= eer_threshold else 0
        if pred == labels[i]:
            correct += 1
    accuracy = correct / total

    return {
        "auc": auc,
        "eer": eer,
        "eer_accuracy": accuracy,
        "eer_threshold": eer_threshold,
    }


def compute_far_tpr(genuine_scores, impostor_scores, far_levels=None):
    """Compute TPR at specific FAR levels (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6)."""
    if far_levels is None:
        far_levels = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]

    total_neg = len(impostor_scores)
    results = {}

    for far_target in far_levels:
        # Number of false positives allowed at this FAR
        n_fp_allowed = int(np.ceil(far_target * total_neg))
        # Threshold: the n_fp_allowed-th highest impostor score
        if n_fp_allowed >= len(impostor_scores):
            threshold = impostor_scores.min()
        else:
            sorted_impostor = np.sort(-impostor_scores)
            threshold = sorted_impostor[n_fp_allowed - 1] if n_fp_allowed > 0 else sorted_impostor[0]

        # TPR at this threshold
        tpr_at_far = np.mean(genuine_scores >= threshold)
        results[f"FAR@{far_target:.0e}"] = float(tpr_at_far)

    return results


def generate_html_report(dataset_results, output_path):
    """Generate a standalone HTML report with ROC curves (no JS dependencies)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Combine all datasets for "Overall" curve
    all_genuine = []
    all_impostor = []
    for ds in dataset_results:
        all_genuine.extend(ds["genuine_scores"].tolist())
        all_impostor.extend(ds["impostor_scores"].tolist())

    if all_genuine and all_impostor:
        dataset_results.append({
            "name": "Overall (Combined)",
            "genuine_scores": np.array(all_genuine),
            "impostor_scores": np.array(all_impostor),
        })

    # Build report lines
    report_lines = []
    for ds in dataset_results:
        name = ds["name"]
        g = ds["genuine_scores"]
        i = ds["impostor_scores"]

        if len(g) == 0 or len(i) == 0:
            report_lines.append(f"  {name}: insufficient data\n")
            continue

        fpr, tpr, auc = compute_roc(g, i)
        metrics = compute_metrics(g, i)
        far_tpr = compute_far_tpr(g, i)

        report_lines.append(f"  {'=' * 60}")
        report_lines.append(f"  {name}")
        report_lines.append(f"  {'=' * 60}")
        report_lines.append(f"  Genuine pairs:     {len(g):>7,d}")
        report_lines.append(f"  Impostor pairs:    {len(i):>7,d}")
        report_lines.append(f"")
        report_lines.append(f"  AUC:               {metrics['auc']:.4f}  ({metrics['auc']*100:.2f}%)")
        report_lines.append(f"  EER:               {metrics['eer']:.4f}  ({metrics['eer']*100:.2f}%)")
        report_lines.append(f"  Accuracy @ EER:    {metrics['eer_accuracy']:.4f}  ({metrics['eer_accuracy']*100:.2f}%)")
        report_lines.append(f"  EER threshold:     {metrics['eer_threshold']:.4f}")
        report_lines.append(f"")

        # FAR@TPR table
        report_lines.append(f"  TPR at FAR levels:")
        for far_key, tpr_val in far_tpr.items():
            report_lines.append(f"    {far_key:>12s}:  {tpr_val:.4f}  ({tpr_val*100:.2f}%)")

        report_lines.append(f"")

        # ROC plot
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(fpr * 100, tpr * 100, linewidth=2, label=f"{name} (AUC={auc:.4f})")
        ax.plot([0, 100], [0, 100], "k--", linewidth=0.8, alpha=0.5, label="Random")
        ax.set_xlabel("False Positive Rate (%)")
        ax.set_ylabel("True Positive Rate (%)")
        ax.set_title(f"ROC — {name}")
        ax.legend(loc="lower right", fontsize=10)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        plot_path = output_path.replace(".html", f"_{name.replace(' ', '_').replace('(', '').replace(')', '')}.png")
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        fig.savefig(os.path.splitext(plot_path)[0] + ".pdf", bbox_inches="tight")
        plt.close(fig)
        ds["plot_path"] = plot_path

    # Print summary
    for line in report_lines:
        print(line)

    # Generate HTML
    html_parts = []
    html_parts.append("""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Verification Results</title>
<style>
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 960px; margin: 2em auto; padding: 0 1em; background: #fff; color: #222; }
h1 { border-bottom: 2px solid #333; padding-bottom: 0.5em; }
table { border-collapse: collapse; width: 100%; margin: 1em 0; }
th, td { border: 1px solid #ddd; padding: 8px 12px; text-align: right; }
th { background: #f5f5f5; text-align: center; font-weight: 600; }
tr:hover { background: #fafafa; }
img { max-width: 100%; height: auto; margin: 1em 0; border: 1px solid #eee; border-radius: 4px; }
.metric { font-size: 1.1em; }
</style></head><body>
""")
    html_parts.append("<h1>Face Verification — ROC &amp; AUC</h1>\n")

    for ds in dataset_results:
        name = ds.get("name", "Unknown")
        g = ds["genuine_scores"]
        i = ds["impostor_scores"]
        if len(g) == 0 or len(i) == 0:
            continue

        metrics = compute_metrics(g, i)
        far_tpr = compute_far_tpr(g, i)

        html_parts.append(f"<h2>{name}</h2>\n")
        html_parts.append("<table>\n<tr><th>Metric</th><th>Value</th></tr>\n")
        html_parts.append(f"<tr><td>Genuine pairs</td><td>{len(g):,}</td></tr>\n")
        html_parts.append(f"<tr><td>Impostor pairs</td><td>{len(i):,}</td></tr>\n")
        html_parts.append(f'<tr><td class="metric">AUC</td><td class="metric">{metrics["auc"]:.4f} ({metrics["auc"]*100:.2f}%)</td></tr>\n')
        html_parts.append(f'<tr><td class="metric">EER</td><td class="metric">{metrics["eer"]:.4f} ({metrics["eer"]*100:.2f}%)</td></tr>\n')
        html_parts.append(f'<tr><td>Accuracy @ EER</td><td>{metrics["eer_accuracy"]:.4f} ({metrics["eer_accuracy"]*100:.2f}%)</td></tr>\n')
        html_parts.append(f'<tr><td>EER threshold</td><td>{metrics["eer_threshold"]:.4f}</td></tr>\n')

        for far_key, tpr_val in far_tpr.items():
            label = far_key.replace("FAR@", "TPR @ FAR=").replace("e-0", "e-")
            html_parts.append(f"<tr><td>{label}</td><td>{tpr_val:.4f} ({tpr_val*100:.2f}%)</td></tr>\n")

        html_parts.append("</table>\n")

        if "plot_path" in ds:
            # Make path relative for HTML embedding; copy to same dir as HTML
            rel_plot = os.path.basename(ds["plot_path"])
            html_parts.append(f'<img src="{rel_plot}" alt="ROC — {name}">\n')

    html_parts.append("</body></html>")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("".join(html_parts))

    print(f"\nHTML report saved to: {output_path}")


def discover_pairs(pairs_dir):
    """Auto-discover dataset pairs from the pairs directory.

    Looks for *_genuine_pairs.txt and *_impostor_pairs.txt files.
    Returns list of (dataset_name, genuine_path, impostor_path).
    """
    pairs_path = Path(pairs_dir)
    if not pairs_path.is_dir():
        raise NotADirectoryError(f"Pairs directory not found: {pairs_dir}")

    genuine_files = sorted(pairs_path.glob("*_genuine_pairs.txt"))
    results = []

    for gf in genuine_files:
        # Derive dataset name: strip "_genuine_pairs.txt"
        ds_name = gf.stem.replace("_genuine_pairs", "")
        impostor_file = gf.parent / gf.name.replace("_genuine_", "_impostor_")

        if impostor_file.exists():
            results.append((ds_name, str(gf), str(impostor_file)))

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Face verification: ROC/AUC from precomputed embeddings"
    )
    parser.add_argument(
        "--embeddings", type=str, required=True,
        help="Path to embeddings folder (e.g., .../embeddings/SwinFace or .../embeddings/TransFace)"
    )
    parser.add_argument(
        "--pairs", type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "..", "deid-toolkit", "root_dir", "datasets", "pairs"),
        help="Path to pairs directory (contains *_genuine_pairs.txt and *_impostor_pairs.txt)"
    )
    parser.add_argument(
        "--output", type=str, default="verification_results.html",
        help="Output HTML report path (default: verification_results.html)"
    )

    args = parser.parse_args()

    embeddings_dir = os.path.abspath(args.embeddings)
    pairs_dir = os.path.abspath(args.pairs)
    output_path = os.path.abspath(args.output)

    print("=" * 60)
    print("Face Verification — ROC / AUC")
    print("=" * 60)
    print(f"Embeddings: {embeddings_dir}")
    print(f"Pairs dir:  {pairs_dir}")
    print(f"Output:     {output_path}")
    print()

    # Load embeddings
    print("Loading embeddings...")
    embeddings = load_all_embeddings(embeddings_dir)
    print(f"Loaded {len(embeddings)} embedding(s)")
    print()

    # Discover and process pairs datasets
    datasets = discover_pairs(pairs_dir)
    if not datasets:
        print("ERROR: No pair files found. Expected *_genuine_pairs.txt and *_impostor_pairs.txt")
        sys.exit(1)

    print(f"Found {len(datasets)} dataset(s): {', '.join(ds[0] for ds in datasets)}")
    print()

    all_results = []

    for ds_name, genuine_path, impostor_path in datasets:
        print(f"Processing: {ds_name}")
        genuine_pairs = parse_pairs_file(genuine_path)
        impostor_pairs = parse_pairs_file(impostor_path)
        print(f"  Genuine pairs:   {len(genuine_pairs):>7,d}")
        print(f"  Impostor pairs:  {len(impostor_pairs):>7,d}")

        g_scores, i_scores = compute_scores(embeddings, genuine_pairs, impostor_pairs)

        all_results.append({
            "name": ds_name,
            "genuine_scores": g_scores,
            "impostor_scores": i_scores,
        })

    # Generate report (includes combined "Overall" curve)
    print()
    generate_html_report(all_results, output_path)


if __name__ == "__main__":
    main()
