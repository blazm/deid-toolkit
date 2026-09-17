#!/usr/bin/env python3
"""
Identity Drift Analysis — How much does each anonymization technique shift facial embeddings?

For each technique, computes cosine distance between aligned (original) and anonymized embeddings
per image. Produces:
  1. Summary table: mean drift, median drift, std, max drift per technique
  2. Violin plot: per-image drift distribution across techniques
  3. Histogram: drift density overlay

Usage:
    python analyze_identity_drift.py ^
        --aligned D:\dev\deid-toolkit\root_dir\embeddings\SwinFace\aligned\celeba-test ^
        --techniques-dir D:\dev\deid-toolkit\root_dir\embeddings\SwinFace\datasets ^
        --output drift_analysis.html
"""

import argparse
import os
import sys
from pathlib import Path

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


def compute_drift(aligned, anonymized):
    """Compute cosine distance (1 - similarity) for matched image pairs.

    Returns list of (filename, drift) tuples.
    """
    drifts = []
    for name in aligned:
        if name in anonymized:
            a = aligned[name].flatten()
            b = anonymized[name].flatten()
            sim = float(np.dot(a, b))  # L2-normalized → dot = cosine similarity
            drift = 1.0 - sim
            drifts.append((name, drift))
    return drifts


def generate_plots(drift_data, output_path, portrait=False):
    """Generate publication-ready drift visualization.

    drift_data: dict {technique_name: list of (filename, drift)}
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    # Publication style
    plt.rcParams.update({
        "font.size": 12,
        "font.family": "sans-serif",
        "axes.linewidth": 1.2,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "grid.alpha": 0.2,
    })

    techniques = sorted(drift_data.keys(), key=lambda n: (n.lower() != "aligned", n))

    # Compute stats
    stats = []
    for name in techniques:
        drfts = [d for _, d in drift_data[name]]
        arr = np.array(drfts)
        stats.append({
            "name": name,
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "std": float(np.std(arr)),
            "max": float(np.max(arr)),
            "q25": float(np.percentile(arr, 25)),
            "q75": float(np.percentile(arr, 75)),
            "count": len(arr),
        })

    # Sort by mean drift for display
    stats.sort(key=lambda s: (0 if s["name"].lower() == "aligned" else 1, s["mean"]))

    # Print table
    print()
    print(f"  {'Technique':<22} {'Count':>6} {'Mean':>8} {'Median':>8} {'Std':>8} {'Q25':>8} {'Q75':>8} {'Max':>8}")
    print(f"  {'-'*80}")
    for s in stats:
        print(f"  {s['name']:<22} {s['count']:>6,d} {s['mean']:>8.4f} {s['median']:>8.4f} "
              f"{s['std']:>8.4f} {s['q25']:>8.4f} {s['q75']:>8.4f} {s['max']:>8.4f}")

    # Colors: shared (color, marker) encoding from fig_style (aligned = black)
    from fig_style import color_for

    def get_color(idx):
        return color_for(techniques[idx])

    # === Figure 1: Violin plot of drift distribution ===
    # 2 x 10 stacked rows: the manuscript figure is single-column, so a single
    # 20-violin axis shrinks every label below legibility. Two rows of 10 keep
    # the same data/colors/order and give ~13 pt labels at column display size.
    # Portrait (--portrait): 4 x 5 stacked rows, taller plate, horizontal labels --
    # for single-column figure[t] floats where the 2 x 10 plate shrinks too far.
    n = len(techniques)
    nrows, per = (7, 3) if portrait else (2, 11)
    figsize = (7.8, 15.8) if portrait else (11, 10.4)
    hspace = 0.55 if portrait else 0.45
    fig1, axes1 = plt.subplots(nrows, 1, figsize=figsize, sharey=True,
                               gridspec_kw={"hspace": hspace})

    for r, ax in enumerate(axes1):
        lo, hi = r * per, min((r + 1) * per, n)
        row_names = [s["name"] for s in stats[lo:hi]]
        # Display names: baseline "aligned" -> "Reference" (same convention as
        # the UMAP plate, qualitative grid and idmap figure)
        row_labels = ["Reference" if name.lower() == "aligned" else name
                      for name in row_names]
        row_data = [np.array([d for _, d in drift_data[name]]) for name in row_names]
        positions = list(range(len(row_data)))
        widths = 0.82 if portrait else 0.62
        parts = ax.violinplot(row_data, positions=positions, widths=widths,
                              showmeans=False, showmedians=True)
        for pc_idx, body in enumerate(parts["bodies"]):
            c = get_color(lo + pc_idx)
            body.set_facecolor(c)
            body.set_alpha(0.6)
            body.set_edgecolor("#333")
        parts["cmedians"].set_color("white")
        parts["cmedians"].set_linewidth(2)

        # Mean markers
        row_means = [stats[lo + i]["mean"] for i in range(len(row_data))]
        ax.scatter(positions, row_means, marker="o", color="white", s=50, zorder=5,
                   edgecolors="#333", linewidths=1.5)

        ax.set_xticks(positions)
        if portrait:
            ax.set_xticklabels(row_labels, rotation=0, fontsize=12)
            ax.set_xlim(-0.9, max(len(row_data), per) + 0.1)
        else:
            ax.set_xticklabels(row_labels, rotation=45, ha="right", fontsize=13)
        ax.grid(True, axis="y", alpha=0.2)

    # Shaded region: low vs high drift (drawn on both rows; legend on top row)
    for r, ax in enumerate(axes1):
        ax.axhspan(0, 0.1, facecolor="#2ecc71", alpha=0.15,
                   label="Minimal drift (< 0.1)" if r == 0 else None)
        ax.axhspan(0.1, 0.3, facecolor="#f39c12", alpha=0.12)
        ax.axhspan(0.3, 1.0, facecolor="#e74c3c", alpha=0.12,
                   label="Severe drift (> 0.3)" if r == 0 else None)
        if r == 0:
            legend_patches = [
                mpatches.Patch(color="#2ecc71", alpha=0.2, label="Minimal (< 0.1)"),
                mpatches.Patch(color="#e74c3c", alpha=0.2, label="Severe (> 0.3)"),
            ]
            ax.legend(handles=legend_patches, loc="upper left",
                      fontsize=10 if not portrait else 9, framealpha=0.9)

    if not portrait:
        axes1[-1].set_xlim(axes1[0].get_xlim())
    axes1[0].set_ylabel("Cosine Distance from Original", fontsize=13)

    fig1.tight_layout()
    suffix = "_portrait" if portrait else ""
    violin_path = os.path.splitext(output_path)[0] + "_violin" + suffix + ".png"
    fig1.savefig(violin_path, dpi=200, bbox_inches="tight")
    fig1.savefig(os.path.splitext(violin_path)[0] + ".pdf", bbox_inches="tight")
    plt.close(fig1)
    print(f"\nViolin plot: {violin_path}")

    # === Figure 2: Bar chart with mean drift + error bars (95% CI) ===
    fig2, ax2 = plt.subplots(figsize=(max(10, len(techniques) * 0.7), 5))

    x_pos = np.arange(len(stats))
    bar_means = [s["mean"] for s in stats]
    bar_errs = [s["std"] / np.sqrt(s["count"]) for s in stats]  # standard error

    bar_colors = [get_color(i) if s["name"].lower() != "aligned" else "#000000"
                  for i, s in enumerate(stats)]

    bars = ax2.bar(x_pos, bar_means, yerr=bar_errs, capsize=3, color=bar_colors, alpha=0.8, edgecolor="#333", linewidth=0.5)

    # Value labels on top of bars
    for i, (m, s_obj) in enumerate(zip(bar_means, stats)):
        ax2.text(i, m + bar_errs[i] + 0.01, f"{m:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([s["name"] for s in stats], rotation=45, ha="right", fontsize=max(7, 10 - len(stats) // 3))
    ax2.set_ylabel("Mean Cosine Distance", fontsize=13)
    ax2.set_title("Identity Drift — Mean ± SE per Technique", fontsize=14, fontweight="bold")
    ax2.grid(True, axis="y", alpha=0.2)
    ax2.set_ylim(0, max(bar_means) * 1.3 if bar_means else 0.5)

    fig2.tight_layout()
    bar_path = os.path.splitext(output_path)[0] + "_bar.png"
    fig2.savefig(bar_path, dpi=200, bbox_inches="tight")
    fig2.savefig(os.path.splitext(bar_path)[0] + ".pdf", bbox_inches="tight")
    plt.close(fig2)
    print(f"Bar chart:   {bar_path}")

    # === Figure 3: Cumulative distribution overlay ===
    fig3, ax3 = plt.subplots(figsize=(8, 5))

    for i, s_obj in enumerate(stats):
        name = s_obj["name"]
        drfts = sorted([d for _, d in drift_data[name]])
        cdf = np.arange(1, len(drfts) + 1) / len(drfts)
        style = "--" if name.lower() == "aligned" else "-"
        lw = 2.5 if name.lower() == "aligned" else 1.5
        color = "#000000" if name.lower() == "aligned" else get_color(i)
        ax3.plot(drfts, cdf, linestyle=style, linewidth=lw, color=color, label=f"{name} ({s_obj['mean']:.3f})")

    ax3.set_xlabel("Cosine Distance from Original", fontsize=13)
    ax3.set_ylabel("Cumulative Fraction", fontsize=13)
    ax3.set_title("Identity Drift — Cumulative Distribution Function", fontsize=14, fontweight="bold")
    ax3.legend(loc="lower right", fontsize=7.5, ncol=2, framealpha=0.9)
    ax3.grid(True, alpha=0.2)

    fig3.tight_layout()
    cdf_path = os.path.splitext(output_path)[0] + "_cdf.png"
    fig3.savefig(cdf_path, dpi=200, bbox_inches="tight")
    fig3.savefig(os.path.splitext(cdf_path)[0] + ".pdf", bbox_inches="tight")
    plt.close(fig3)
    print(f"CDF plot:    {cdf_path}")

    # === HTML report ===
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Identity Drift Analysis</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
       max-width: 1200px; margin: 2em auto; padding: 0 1em; }}
h1 {{ border-bottom: 2px solid #333; padding-bottom: 0.5em; }}
h2 {{ color: #444; margin-top: 2em; }}
table {{ border-collapse: collapse; width: 100%; margin: 1em 0; }}
th, td {{ border: 1px solid #ddd; padding: 6px 10px; text-align: right; }}
th {{ background: #f5f5f5; text-align: center; font-weight: 600; }}
img {{ max-width: 100%; height: auto; border: 1px solid #eee; border-radius: 4px; margin: 1em 0; }}
.note {{ color: #666; font-style: italic; margin: 0.5em 0; }}
</style></head><body>
<h1>Identity Drift Analysis</h1>
<p class="note">Cosine distance between aligned (original) and anonymized embeddings per image.
Lower values = identity better preserved. Higher values = more identity drift.</p>

<h2>Summary Statistics</h2>
<table>
<tr><th style="text-align:left">Technique</th><th>Images</th><th>Mean Drift</th>
<th>Median</th><th>Std Dev</th><th>Q25</th><th>Q75</th><th>Max</th></tr>"""

    for s in stats:
        html += (f"<tr><td style='text-align:left'>{s['name']}</td>"
                 f"<td>{s['count']:,}</td><td>{s['mean']:.4f}</td>"
                 f"<td>{s['median']:.4f}</td><td>{s['std']:.4f}</td>"
                 f"<td>{s['q25']:.4f}</td><td>{s['q75']:.4f}</td>"
                 f"<td>{s['max']:.4f}</td></tr>\n")

    html += """</table>

<h2>Drift Distribution (Violin Plot)</h2>
<img src=""" + f'\"{os.path.basename(violin_path)}\"' + """ alt="Violin plot">

<h2>Mean Drift with Standard Error</h2>
<img src=""" + f'\"{os.path.basename(bar_path)}\"' + """ alt="Bar chart">

<h2>Cumulative Distribution Function</h2>
<img src=""" + f'\"{os.path.basename(cdf_path)}\"' + """ alt="CDF plot">

<p class="note"><strong>Interpretation:</strong> A technique with high mean drift and low variance
consistently transforms all faces equally. High variance suggests the technique affects some
identities more than others — potentially revealing bias or uneven anonymization.</p>
</body></html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\nHTML report: {output_path}")



def _kde_curve(x, grid, bw=None):
    """Gaussian kernel density of x evaluated on grid (numpy only)."""
    if bw is None:
        bw = max(1.06 * float(np.std(x)), 1e-3) * max(len(x), 1) ** (-1.0 / 5.0)
    z = (grid[:, None] - x[None, :]) / bw
    return np.exp(-0.5 * z * z).mean(axis=1) / (bw * (2.0 * np.pi) ** 0.5)

def generate_horizontal_violin(drift_data, output_path):
    """Single-axis laid-down violin plate for narrow (single-column) layout.

    y = one row per technique (Reference on top, strongest drift at bottom),
    x = cosine distance from original. Labels stay full-size at column width.
    Saves <output>_violin_h.png and .pdf alongside the requested output.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    from fig_style import color_for

    plt.rcParams.update({
        "font.size": 12,
        "font.family": "sans-serif",
        "axes.linewidth": 1.2,
    })

    # The aligned baseline is a constant-zero line; it portrays nothing -- omit it.
    techniques = sorted((n_ for n_ in drift_data.keys() if n_.lower() != "aligned"),
                        key=lambda n_: n_)
    stats = []
    for name in techniques:
        arr = np.array([d for _, d in drift_data[name]])
        stats.append({"name": name, "mean": float(np.mean(arr)),
                      "median": float(np.median(arr))})
    stats.sort(key=lambda s_2: s_2["mean"])

    labels = ["Reference" if s_2["name"].lower() == "aligned" else s_2["name"]
              for s_2 in stats]
    series = [np.array([d for _, d in drift_data[s_2["name"]]]) for s_2 in stats]
    n = len(stats)
    y = list(range(n))

    grid_lo, grid_hi = 0.0, 1.70
    grid = np.linspace(grid_lo, grid_hi, 600)
    half_w_max = 0.36   # in y-units (21 rows -> row pitch 1.0)

    fig, ax = plt.subplots(figsize=(7.6, 10.2))
    for i in range(n):
        dens = _kde_curve(series[i], grid)
        if series[i].std() < 1e-9:     # baseline / degenerate: thin line
            ax.axhline(y[i], xmin=0.0, xmax=1.0, color="#333", lw=0.8,
                       alpha=0.5, zorder=2)
            continue
        dens = dens / dens.max()
        ytop = np.clip(y[i] + dens * half_w_max, y[i] - 0.46, y[i] + 0.46)
        ybot = np.clip(y[i] - dens * half_w_max, y[i] - 0.46, y[i] + 0.46)
        name = stats[i]["name"]
        face = "#000000" if name.lower() == "aligned" else color_for(name)
        ax.fill_between(grid, ybot, ytop,
                        facecolor=face, alpha=0.55, edgecolor="#333",
                        linewidth=0.9, zorder=3)
        # whisker from min to max
        ax.plot([series[i].min(), series[i].max()], [y[i], y[i]],
                color="#2a5d8f", lw=1.3, zorder=2, solid_capstyle="round")
        # median (white bar) and mean (white dot)
        ax.plot([stats[i]["median"], stats[i]["median"]],
                [y[i] - 0.10, y[i] + 0.10], color="white", lw=2.2, zorder=5,
                solid_capstyle="round")
        ax.scatter([stats[i]["mean"]], [y[i]], marker="o", s=34,
                   facecolor="white", edgecolor="#333", linewidths=1.3,
                   zorder=6)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10.5)
    ax.set_ylim(n - 0.35, -0.65)   # y=0 (Reference) at the top
    ax.set_xlim(grid_lo - 0.05, 1.74)
    ax.set_xlabel("Cosine Distance from Original", fontsize=13)
    ax.grid(True, axis="x", alpha=0.2, zorder=1)

    ax.axvspan(0.0, 0.1, facecolor="#2ecc71", alpha=0.15, zorder=0)
    ax.axvspan(0.1, 0.3, facecolor="#f39c12", alpha=0.12, zorder=0)
    ax.axvspan(0.3, 1.74, facecolor="#e74c3c", alpha=0.10, zorder=0)
    legend_patches = [
        mpatches.Patch(color="#2ecc71", alpha=0.2, label="Minimal (< 0.1)"),
        mpatches.Patch(color="#f39c12", alpha=0.35, label="Moderate (0.1-0.3)"),
        mpatches.Patch(color="#e74c3c", alpha=0.2, label="Severe (> 0.3)"),
    ]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=9.5,
              framealpha=0.9)

    fig.tight_layout()
    h_png = os.path.splitext(output_path)[0] + "_violin_h.png"
    fig.savefig(h_png, dpi=200, bbox_inches="tight")
    fig.savefig(os.path.splitext(h_png)[0] + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print("Horizontal violin plot: " + h_png)


def main():
    parser = argparse.ArgumentParser(description="Identity drift analysis from facial embeddings")
    parser.add_argument("--aligned", type=str, required=True,
                        help="Path to aligned (original) embeddings celeba-test folder")
    parser.add_argument("--techniques-dir", type=str, required=True,
                        help="Path to datasets root containing technique/celeba-test folders")
    parser.add_argument("--output", type=str, default="drift_analysis.html",
                        help="Output HTML report path")
    parser.add_argument("--portrait", action="store_true",
                        help="Save the violin as a 4x5 portrait plate "
                             "(_violin_portrait.png/.pdf) instead of 2x10")
    parser.add_argument("--horizontal", action="store_true",
                        help="Save a laid-down single-axis violin plate "
                             "(_violin_h.png/.pdf); y = technique, x = drift")

    args = parser.parse_args()

    print("=" * 60)
    print("Identity Drift Analysis")
    print("=" * 60)

    # Load aligned embeddings
    print(f"Loading aligned embeddings: {args.aligned}")
    aligned = load_embeddings(args.aligned)
    if not aligned:
        print("ERROR: No aligned embeddings found.")
        sys.exit(1)
    print(f"  Loaded {len(aligned)} embedding(s)")

    # Discover technique folders
    tech_dir = Path(args.techniques_dir)
    if not tech_dir.is_dir():
        print(f"ERROR: Techniques directory not found: {args.techniques_dir}")
        sys.exit(1)

    drift_data = {}

    # Add aligned as baseline (drift should be ~0)
    drift_data["aligned"] = [(name, 0.0) for name in aligned]

    # Process each technique with a celeba-test subfolder
    techniques_found = sorted([d for d in tech_dir.iterdir()
                               if d.is_dir() and not d.name.endswith("_reversed")],
                              key=lambda x: x.name)

    print(f"\nScanning {len(techniques_found)} technique folder(s)...")

    for tech_folder in techniques_found:
        celeba_path = tech_folder / "celeba-test"
        if not celeba_path.is_dir():
            continue

        name = tech_folder.name
        print(f"\nProcessing: {name}")
        anon = load_embeddings(celeba_path)

        if not anon:
            print(f"  SKIP — no embeddings")
            continue

        matched = sum(1 for n in aligned if n in anon)
        print(f"  Loaded {len(anon)} embedding(s), {matched} matched with aligned")

        drifts = compute_drift(aligned, anon)
        drift_data[name] = drifts
        mean_d = np.mean([d for _, d in drifts]) if drifts else 0
        print(f"  Mean drift: {mean_d:.4f}")

    if len(drift_data) < 2:
        print("ERROR: Need at least aligned + one technique.")
        sys.exit(1)

    if args.horizontal:
        generate_horizontal_violin(drift_data, args.output)
        return
    generate_plots(drift_data, args.output, portrait=args.portrait)


if __name__ == "__main__":
    main()
