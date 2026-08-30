#!/usr/bin/env python3
"""
Retrieval Accuracy Analysis — How well can you still identify people from anonymized embeddings?

For each technique, computes k-NN retrieval accuracy: given an anonymized embedding,
how often does the top-k nearest neighbor in the aligned space contain the true identity?

Also computes mAP (mean Average Precision) and recall@k curves.

Usage:
    python analyze_retrieval.py \\
        --aligned D:/dev/deid-toolkit/root_dir/embeddings/SwinFace/aligned/celeba-test \\
        --techniques-dir D:/dev/deid-toolkit/root_dir/embeddings/SwinFace/datasets \\
        --output retrieval_analysis.html \\
        --sample 1000
"""

import argparse
import os
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np


def load_embeddings(folder):
    """Load .npy files from a folder. Returns {stem: array}."""
    p = Path(folder)
    if not p.is_dir():
        return {}
    embs = {}
    for npy in sorted(p.glob("*.npy")):
        embs[npy.stem] = np.load(npy)
    return embs


def compute_recall_at_k(query_embs, gallery_embs, gallery_ids, query_ids, k_values=None):
    """Compute recall@k for each k in k_values.

    query_embs: (N, D) anonymized embeddings (queries)
    gallery_embs: (M, D) aligned embeddings (gallery)
    gallery_ids: list of M image names in gallery
    query_ids: list of N image names (should match gallery entries)
    k_values: list of k values to compute recall at

    Recall@k: fraction of queries where the true image appears in top-k gallery results.
    Since each query IS a gallery entry, this measures how retrievable each identity is.
    """
    if k_values is None:
        k_values = [1, 5, 10, 20, 50, 100]

    # Build gallery index
    gallery_matrix = gallery_embs.T  # (D, M) for fast dot product
    query_matrix = query_embs  # (N, D)

    # Compute similarities: (N, M)
    sims = query_matrix @ gallery_matrix

    recalls = {}
    for k in k_values:
        top_k = np.argsort(-sims)[:, :k]  # (N, k)
        # For each query, check if its own index is in top-k
        correct = 0
        for i in range(len(query_ids)):
            if i in top_k[i]:
                correct += 1
        recalls[k] = correct / len(query_ids)

    return recalls


def compute_mean_drift_rank(query_embs, gallery_embs, query_ids):
    """For each query, find the rank of its own aligned embedding in the retrieval results.

    Returns distribution of ranks.
    """
    n = len(query_ids)
    gallery_matrix = gallery_embs.T  # (D, M)

    sims = query_embs @ gallery_matrix  # (N, M)

    # For each query, its own index should ideally be rank 0 (highest similarity)
    own_sims = np.diag(sims)  # similarity to self

    # Rank: how many gallery items have higher similarity than self
    ranks = np.zeros(n, dtype=int)
    for i in range(n):
        ranks[i] = np.sum(sims[i] > own_sims[i])

    return ranks


def compute_mAP(query_embs, gallery_embs, query_ids):
    """Compute mean Average Precision.

    For each query, AP = (1/num_relevant) * sum(rank_i * precision_at_rank_i)
    Since there's exactly one relevant item (the query's own aligned embedding),
    AP = 1 / rank_of_self (where rank starts at 1).
    """
    n = len(query_ids)
    gallery_matrix = gallery_embs.T

    sims = query_embs @ gallery_matrix  # (N, M)
    own_sims = np.diag(sims)

    aps = []
    for i in range(n):
        # Sort gallery by similarity descending
        order = np.argsort(-sims[i])
        # Find position of self (index i in gallery)
        self_pos = np.where(order == i)[0][0]
        ap = 1.0 / (self_pos + 1)  # rank starts at 1
        aps.append(ap)

    return float(np.mean(aps)), aps


def generate_plots(results, output_path):
    """Generate publication-ready retrieval analysis plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 12,
        "axes.linewidth": 1.2,
        "grid.alpha": 0.2,
    })

    # Sort: aligned first, then by mAP ascending (worse techniques first)
    sorted_results = sorted(results, key=lambda r: (0 if r["name"].lower() == "aligned" else 1, r["map"]))

    COLORS = ["#000000"] + [
        "#e74c3c", "#2980b9", "#27ae60", "#8e44ad", "#d35400",
        "#16a085", "#c0392b", "#2c3e50", "#f39c12", "#1abc9c",
        "#e84393", "#6c5ce7", "#fdcb6e", "#00b894", "#d63031",
    ]

    def get_color(idx, name):
        if name.lower() == "aligned":
            return "#000000"
        return COLORS[idx % len(COLORS)]

    # Print table
    print()
    print(f"  {'Technique':<22} {'mAP':>8} {'R@1':>8} {'R@5':>8} {'R@10':>8} {'R@50':>8} {'R@100':>8} {'Mean Rank':>10}")
    print(f"  {'-'*90}")
    for r in sorted_results:
        recalls = r["recalls"]
        print(f"  {r['name']:<22} {r['map']:>8.4f} {recalls.get(1, 0):>8.4f} {recalls.get(5, 0):>8.4f} "
              f"{recalls.get(10, 0):>8.4f} {recalls.get(50, 0):>8.4f} {recalls.get(100, 0):>8.4f} "
              f"{r['mean_rank']:>10.1f}")

    # === Figure 1: Recall@k curve ===
    fig1, ax1 = plt.subplots(figsize=(9, 6))

    k_values_sorted = sorted(sorted_results[0]["recalls"].keys()) if sorted_results else []

    for i, r in enumerate(sorted_results):
        ks = k_values_sorted
        vs = [r["recalls"].get(k, 0) for k in ks]
        name = r["name"]
        color = get_color(i, name)
        style = "--" if name.lower() == "aligned" else "-"
        lw = 2.5 if name.lower() == "aligned" else 1.5
        marker = "o" if name.lower() == "aligned" else None
        ax1.plot(ks, vs, linestyle=style, linewidth=lw, color=color,
                label=f"{name} (mAP={r['map']:.4f})", marker=marker, markersize=6)

    ax1.set_xlabel("k (Number of Retrieval Results)", fontsize=13)
    ax1.set_ylabel("Recall@k", fontsize=13)
    ax1.set_title("Retrieval Recall — Can You Still Identify People?", fontsize=14, fontweight="bold")
    ax1.legend(loc="lower right", fontsize=8, ncol=2, framealpha=0.9)
    ax1.grid(True, alpha=0.2)
    ax1.set_ylim(0, 1.05)
    ax1.set_yticks(np.arange(0, 1.1, 0.1))

    # Shade regions
    ax1.axhspan(0.8, 1.0, facecolor="#e74c3c", alpha=0.1, label="High identifiability")
    ax1.axhspan(0.0, 0.2, facecolor="#2ecc71", alpha=0.1, label="Low identifiability")

    fig1.tight_layout()
    recall_path = os.path.splitext(output_path)[0] + "_recall.png"
    fig1.savefig(recall_path, dpi=200, bbox_inches="tight")
    fig1.savefig(os.path.splitext(recall_path)[0] + ".pdf", bbox_inches="tight")
    plt.close(fig1)
    print(f"\nRecall plot: {recall_path}")

    # === Figure 2: mAP bar chart ===
    fig2, ax2 = plt.subplots(figsize=(max(9, len(sorted_results) * 0.6), 5))

    x_pos = np.arange(len(sorted_results))
    map_vals = [r["map"] for r in sorted_results]
    bar_colors = [get_color(i, r["name"]) for i, r in enumerate(sorted_results)]

    bars = ax2.barh(x_pos, map_vals, color=bar_colors, alpha=0.8, edgecolor="#333", linewidth=0.5)
    ax2.set_yticks(x_pos)
    ax2.set_yticklabels([r["name"] for r in sorted_results], fontsize=max(7, 10 - len(sorted_results) // 4))
    ax2.set_xlabel("mean Average Precision (mAP)", fontsize=13)
    ax2.set_title("Retrieval mAP — Identity Identifiability", fontsize=14, fontweight="bold")
    ax2.grid(True, axis="x", alpha=0.2)

    # Value labels
    for i, v in enumerate(map_vals):
        ax2.text(v + 0.02, i, f"{v:.4f}", ha="left", va="center", fontsize=9)

    fig2.tight_layout()
    map_path = os.path.splitext(output_path)[0] + "_map.png"
    fig2.savefig(map_path, dpi=200, bbox_inches="tight")
    fig2.savefig(os.path.splitext(map_path)[0] + ".pdf", bbox_inches="tight")
    plt.close(fig2)
    print(f"mAP chart:   {map_path}")

    # === Figure 3: Rank distribution (CDF) ===
    fig3, ax3 = plt.subplots(figsize=(9, 5))

    for i, r in enumerate(sorted_results):
        ranks = np.array(r["ranks"])
        if len(ranks) == 0:
            continue
        sorted_ranks = np.sort(ranks)
        cdf = np.arange(1, len(sorted_ranks) + 1) / len(sorted_ranks)
        name = r["name"]
        color = get_color(i, name)
        style = "--" if name.lower() == "aligned" else "-"
        lw = 2.5 if name.lower() == "aligned" else 1.2
        # Cap x axis at reasonable value
        max_rank_display = min(sorted_ranks[-1], 200)
        mask = sorted_ranks <= max_rank_display
        ax3.plot(sorted_ranks[mask], cdf[:sum(mask)], linestyle=style, linewidth=lw,
                color=color, label=f"{name} (median={np.median(ranks):.0f})")

    ax3.set_xlabel("Rank of True Identity in Retrieval Results", fontsize=13)
    ax3.set_ylabel("Cumulative Fraction", fontsize=13)
    ax3.set_title("Retrieval Rank Distribution — Lower is Better for Privacy", fontsize=14, fontweight="bold")
    ax3.legend(loc="lower right", fontsize=7.5, ncol=2, framealpha=0.9)
    ax3.grid(True, alpha=0.2)

    fig3.tight_layout()
    rank_path = os.path.splitext(output_path)[0] + "_rank_cdf.png"
    fig3.savefig(rank_path, dpi=200, bbox_inches="tight")
    fig3.savefig(os.path.splitext(rank_path)[0] + ".pdf", bbox_inches="tight")
    plt.close(fig3)
    print(f"Rank CDF:    {rank_path}")

    # === HTML ===
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Retrieval Accuracy Analysis</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
       max-width: 1200px; margin: 2em auto; padding: 0 1em; }}
h1 {{ border-bottom: 2px solid #333; padding-bottom: 0.5em; }}
img {{ max-width: 100%; height: auto; border: 1px solid #eee; border-radius: 4px; margin: 1em 0; }}
.note {{ color: #666; font-style: italic; }}
table {{ border-collapse: collapse; width: 100%; margin: 1em 0; }}
th, td {{ border: 1px solid #ddd; padding: 6px 10px; text-align: right; }}
th {{ background: #f5f5f5; text-align: center; font-weight: 600; }}
</style></head><body>
<h1>Retrieval Accuracy Analysis</h1>
<p class="note">For each anonymized embedding, we search the aligned (original) gallery and measure
how easily the true identity can be retrieved. Lower recall/mAP = better privacy protection.</p>

<h2>Summary Table</h2>
<table>
<tr><th style="text-align:left">Technique</th><th>mAP</th><th>R@1</th><th>R@5</th>
<th>R@10</th><th>R@50</th><th>R@100</th><th>Mean Rank</th></tr>"""

    for r in sorted_results:
        recalls = r["recalls"]
        html += (f"<tr><td style='text-align:left'>{r['name']}</td>"
                 f"<td>{r['map']:.4f}</td><td>{recalls.get(1, 0):.4f}</td>"
                 f"<td>{recalls.get(5, 0):.4f}</td><td>{recalls.get(10, 0):.4f}</td>"
                 f"<td>{recalls.get(50, 0):.4f}</td><td>{recalls.get(100, 0):.4f}</td>"
                 f"<td>{r['mean_rank']:.1f}</td></tr>\n")

    html += """</table>

<h2>Recall@k Curve</h2>
<img src=""" + f'\"{os.path.basename(recall_path)}\"' + """ alt="Recall@k">

<h2>mAP per Technique</h2>
<img src=""" + f'\"{os.path.basename(map_path)}\"' + """ alt="mAP bars">

<h2>Rank Distribution (CDF)</h2>
<img src=""" + f'\"{os.path.basename(rank_path)}\"' + """ alt="Rank CDF">

<p class="note"><strong>Interpretation:</strong> Recall@1 = probability the anonymized face matches
its own original as the top result. mAP aggregates this across all k-values. A technique with
R@1 ≈ 0 and low mAP effectively prevents identity retrieval, while high values indicate the
anonymized embedding still preserves identifiable features.</p>
</body></html>"""

    # Ensure UTF-8 output on all platforms (cp1252 on Windows can't encode ≈, etc.)
    import io
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\nHTML report: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Retrieval accuracy analysis")
    parser.add_argument("--aligned", type=str, required=True,
                        help="Path to aligned embeddings celeba-test folder")
    parser.add_argument("--techniques-dir", type=str, required=True,
                        help="Path to datasets root with technique/celeba-test folders")
    parser.add_argument("--output", type=str, default="retrieval_analysis.html",
                        help="Output HTML report path")
    parser.add_argument("--sample", type=int, default=1000,
                        help="Maximum images to sample (default: 1000)")

    args = parser.parse_args()

    print("=" * 60)
    print("Retrieval Accuracy Analysis")
    print("=" * 60)

    # Load aligned (gallery)
    print(f"Loading aligned: {args.aligned}")
    aligned = load_embeddings(args.aligned)
    if not aligned:
        print("ERROR: No aligned embeddings found.")
        sys.exit(1)
    print(f"  Loaded {len(aligned)} embedding(s)")

    # Sample
    all_names = sorted(aligned.keys())
    if len(all_names) > args.sample:
        rng = np.random.RandomState(42)
        sample_names = sorted(rng.choice(all_names, size=args.sample, replace=False).tolist())
        print(f"  Sampling {args.sample} images")
    else:
        sample_names = all_names

    # Build gallery matrix from aligned
    gallery_embs = np.array([aligned[n] for n in sample_names], dtype=np.float32)
    gallery_ids = sample_names
    print(f"  Gallery shape: {gallery_embs.shape}")

    results = []

    # Aligned as baseline
    print("\nComputing aligned baseline retrieval...")
    q_embs = gallery_embs.copy()
    recalls = compute_recall_at_k(q_embs, gallery_embs, gallery_ids, gallery_ids)
    mAP_val, aps = compute_mAP(q_embs, gallery_embs, gallery_ids)
    ranks = compute_mean_drift_rank(q_embs, gallery_embs, gallery_ids)

    results.append({
        "name": "aligned",
        "recalls": recalls,
        "map": mAP_val,
        "mean_rank": float(np.mean(ranks)),
        "ranks": ranks.tolist(),
    })
    print(f"  Aligned: mAP={mAP_val:.4f}, R@1={recalls.get(1, 0):.4f}")

    # Load techniques
    tech_dir = Path(args.techniques_dir)
    if not tech_dir.is_dir():
        print("ERROR: Techniques directory not found.")
        sys.exit(1)

    tech_folders = sorted([d for d in tech_dir.iterdir()
                           if d.is_dir() and not d.name.endswith("_reversed")],
                          key=lambda x: x.name)

    for tech_folder in tech_folders:
        celeba_path = tech_folder / "celeba-test"
        if not celeba_path.is_dir():
            continue

        name = tech_folder.name
        print(f"\nProcessing: {name}")
        anon = load_embeddings(celeba_path)

        # Build query matrix from anonymized, in gallery order
        query_embs_list = []
        query_ids = []
        for n in sample_names:
            if n in anon:
                query_embs_list.append(anon[n])
                query_ids.append(n)

        if not query_embs_list:
            print(f"  SKIP — no matching embeddings")
            continue

        q_matrix = np.array(query_embs_list, dtype=np.float32)
        print(f"  Query shape: {q_matrix.shape}")

        # For retrieval, we use the full gallery but only queries that match
        # We need to compute rank within the full gallery, so map query_ids back to gallery positions
        # Since gallery = aligned[sample_names], and each query IS one of those,
        # the "correct" answer for query i is the gallery entry with the same image name.

        recalls = compute_recall_at_k(q_matrix, gallery_embs, gallery_ids, query_ids)
        mAP_val, aps = compute_mAP(q_matrix, gallery_embs, query_ids)

        # For rank computation, we need queries aligned with gallery positions
        # Build full query matrix in gallery order
        full_q = np.zeros_like(gallery_embs)
        valid_mask = np.zeros(len(gallery_ids), dtype=bool)
        for i, gid in enumerate(gallery_ids):
            if gid in anon:
                full_q[i] = anon[gid]
                valid_mask[i] = True

        ranks = compute_mean_drift_rank(full_q[valid_mask], gallery_embs[valid_mask],
                                       [gallery_ids[i] for i in range(len(gallery_ids)) if valid_mask[i]])

        results.append({
            "name": name,
            "recalls": recalls,
            "map": mAP_val,
            "mean_rank": float(np.mean(ranks)),
            "ranks": ranks.tolist(),
        })
        print(f"  mAP={mAP_val:.4f}, R@1={recalls.get(1, 0):.4f}, Mean Rank={np.mean(ranks):.1f}")

    if not results:
        print("ERROR: No results.")
        sys.exit(1)

    generate_plots(results, args.output)


if __name__ == "__main__":
    main()
