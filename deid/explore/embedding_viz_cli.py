"""CLI entry point for embedding space visualization generation.

Generates PDF + PNG figures for manuscript use.

Usage:
    python -m deid.explore.embedding_viz_cli \
        --dataset mug-still --model swinface \
        --techniques blur pixelize --output root_dir/results/viz/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Generate embedding space analysis visualizations.",
    )
    parser.add_argument("--dataset", "-d", default="mug-still",
                        help="Dataset name (default: mug-still)")
    parser.add_argument("--model", "-m", default="swinface",
                        choices=["adaface", "swinface", "deepface_vggface"],
                        help="Embedding model cache to use (default: swinface)")
    parser.add_argument("--techniques", "-t", nargs="+", default=None,
                        help="Technique names. If omitted, auto-discovers from cache.")
    parser.add_argument("--output", "-o", default=None,
                        help="Output directory (default: root_dir/results/viz/{dataset}_{model}/)")
    parser.add_argument("--root-dir", "-r", default="root_dir",
                        help="Root directory (default: root_dir)")
    parser.add_argument("--method", choices=["umap", "tsne", "pca"], default="umap",
                        help="Projection method (default: umap). PCA gives interpretable axes.")

    args = parser.parse_args()

    # Force non-interactive matplotlib backend
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = args.root_dir
    ds = args.dataset
    model = args.model
    output = Path(args.output) if args.output else Path(root) / "results" / "viz" / f"{ds}_{model}"
    output.mkdir(parents=True, exist_ok=True)

    print(f"Dataset:   {ds}")
    print(f"Model:     {model}")
    print(f"Root dir:  {root}")
    print(f"Output:    {output}")
    print()

    # Discover techniques from cache if not specified
    techniques = args.techniques
    if not techniques:
        techniques = []
        temp_dir = Path(root) / "preprocess" / "temp" / model
        if temp_dir.exists():
            ds_key = ds.replace("-", "_")  # normalize for dir matching
            for d in sorted(temp_dir.iterdir()):
                if not (d.is_dir() and (d / "original").exists() and (d / "deid").exists()):
                    continue
                # Dir name is {dataset}_{technique} — strip dataset prefix
                name = d.name.replace("-", "_")
                for suffix_sep in ("_",):  # separator between ds and tech
                    if name.startswith(ds_key + suffix_sep):
                        tech_name = name[len(ds_key) + len(suffix_sep):]
                        # Convert back to original naming (preserve hyphens)
                        tech_name = tech_name.replace("_", "-")
                        if tech_name not in techniques:
                            techniques.append(tech_name)
                    elif name == ds_key:  # edge case: dir is just dataset name
                        if d.name not in techniques:
                            techniques.append(d.name)
                        break
        if not techniques:
            print(f"ERROR: Cache directory not found: {temp_dir}")
            sys.exit(1)

    if not techniques:
        print("ERROR: No techniques with cached embeddings found.")
        sys.exit(1)

    print(f"Techniques: {techniques}")
    print()

    # Load labels
    from deid.explore.embedding_analysis import (
        load_labels, load_paired_embeddings,
        prepare_displacement_data, prepare_collapse_data,
        prepare_comparison_data, compute_technique_summary,
        EmbeddingCacheNotFoundError,
    )
    from deid.explore.viz import (
        plot_embedding_displacement,
        plot_identity_dispersion,
        plot_technique_comparison,
    )

    labels_df = load_labels(root, ds)
    valid_techniques = []

    # --- Per-technique visualizations ---
    for tech in techniques:
        print(f"\n=== {tech} ===")
        try:
            recs = load_paired_embeddings(root, model, ds, tech, labels_df)
            if not recs:
                print(f"  No paired embeddings for {tech}, skipping.")
                continue

            valid_techniques.append(tech)
            print(f"  Loaded {len(recs)} paired embeddings (dim={recs[0].original.shape[0]}).")

            # 1. Displacement plot
            print("  [1/2] Generating displacement plot...")
            data = prepare_displacement_data(root, model, ds, tech, labels_df, args.method)
            if "error" not in data:
                fig = plot_embedding_displacement(
                    data["orig_xy"], data["deid_xy"], data["magnitudes"],
                    ds, tech, data.get("image_names"), data.get("identities"),
                    raw_euclidean_displacement=data.get("raw_euclidean_displacement"),
                    cosine_similarity=data.get("cosine_similarity"),
                    projection_method=args.method,
                )
                p = output / f"displacement_{model}_{ds}_{tech}"
                fig.savefig(p.with_suffix(".pdf"), dpi=150, format="pdf")
                fig.savefig(p.with_suffix(".png"), dpi=150)
                plt.close(fig)

                # Save per-image displacement CSV for interactive viewer
                disp_csv = p.with_name(p.name + "_data").with_suffix(".csv")
                import pandas as pd
                disp_rows = []
                for idx in range(len(data["image_names"])):
                    disp_rows.append({
                        "image": data["image_names"][idx],
                        "identity": data.get("identities", [""] * len(data["image_names"]))[idx] if data.get("identities") else "",
                        "orig_x": float(data["orig_xy"][idx, 0]),
                        "orig_y": float(data["orig_xy"][idx, 1]),
                        "deid_x": float(data["deid_xy"][idx, 0]),
                        "deid_y": float(data["deid_xy"][idx, 1]),
                        "disp_2d": float(data["magnitudes"][idx]),
                    })
                    if raw_eucl is not None:
                        disp_rows[-1]["euclidean_distance"] = round(float(raw_eucl[idx]), 4)
                    if cos_sim is not None:
                        disp_rows[-1]["cosine_similarity"] = round(float(cos_sim[idx]), 4)
                pd.DataFrame(disp_rows).to_csv(disp_csv, index=False)

                raw_eucl = data.get("raw_euclidean_displacement")
                cos_sim  = data.get("cosine_similarity")
                print(f"    Saved: {p.with_suffix('.pdf')}, {p.with_suffix('.png')}")
                print(f"    Data CSV: {disp_csv}")
                if raw_eucl is not None and cos_sim is not None:
                    print(f"    Embedding stats: N={len(raw_eucl)}, CosSim={cos_sim.mean():.4f}±{cos_sim.std():.4f}, EuclDist={raw_eucl.mean():.2f}±{raw_eucl.std():.2f}")
                else:
                    mags = data["magnitudes"]
                    print(f"    2D stats: N={len(mags)}, mean_disp={mags.mean():.4f}, std={mags.std():.4f}")

            # 2. Collapse analysis
            print("  [2/2] Generating collapse analysis...")
            cdata = prepare_collapse_data(root, model, ds, tech, labels_df)
            if "error" not in cdata and not cdata["dispersion_df"].empty:
                fig, sdf = plot_identity_dispersion(
                    cdata["dispersion_df"], ds, tech, top_n=20,
                )
                p = output / f"collapse_{model}_{ds}_{tech}"
                fig.savefig(p.with_suffix(".pdf"), dpi=150, format="pdf")
                fig.savefig(p.with_suffix(".png"), dpi=150)
                plt.close(fig)
                csv_path = p.with_name(p.name + "_metrics").with_suffix(".csv")
                sdf.to_csv(csv_path, index=False)
                print(f"    Saved: {p.with_suffix('.pdf')}, {p.with_suffix('.png')}")
                print(f"    Metrics: {csv_path}")
                summary = cdata.get("summary", {})
                if summary:
                    print(f"    Cosine sim: mean={summary.get('mean_cosine_similarity', 0):.4f}")

        except EmbeddingCacheNotFoundError as e:
            print(f"  Cache not found: {e}")
        except Exception as exc:
            print(f"  Error: {exc}")
            import traceback
            traceback.print_exc()

    # --- Multi-technique comparison (if 2+ techniques) ---
    if len(valid_techniques) >= 2:
        print(f"\n=== Multi-technique comparison ({len(valid_techniques)} techniques) ===")
        try:
            cmp = prepare_comparison_data(root, model, ds, valid_techniques, labels_df, args.method)
            if "error" not in cmp:
                fig = plot_technique_comparison(
                    cmp["orig_xy"],
                    {t: cmp["deid_xys"][f"deid_{t}"] for t in cmp["techniques"]},
                    cmp["magnitudes_by_tech"],
                    ds,
                )
                p = output / f"comparison_{model}_{ds}_multi"
                fig.savefig(p.with_suffix(".pdf"), dpi=150, format="pdf")
                fig.savefig(p.with_suffix(".png"), dpi=150)
                plt.close(fig)

                # Save comparison data CSV for interactive viewer
                import pandas as pd
                cmp_csv = p.with_name(p.name + "_data").with_suffix(".csv")
                cmp_rows = []
                n_images = len(cmp["image_names"])
                for idx in range(n_images):
                    row = {
                        "image": cmp["image_names"][idx],
                        "orig_x": float(cmp["orig_xy"][idx, 0]),
                        "orig_y": float(cmp["orig_xy"][idx, 1]),
                    }
                    for t in cmp["techniques"]:
                        dxy = cmp["deid_xys"][f"deid_{t}"]
                        row[f"{t}_x"] = round(float(dxy[idx, 0]), 4)
                        row[f"{t}_y"] = round(float(dxy[idx, 1]), 4)
                        mag_tech = cmp["magnitudes_by_tech"].get(t)
                        if mag_tech is not None:
                            row[f"{t}_disp_2d"] = round(float(mag_tech[idx]), 4)
                    cmp_rows.append(row)
                pd.DataFrame(cmp_rows).to_csv(cmp_csv, index=False)

                print(f"  Saved: {p.with_suffix('.pdf')}, {p.with_suffix('.png')}")
                print(f"  Data CSV: {cmp_csv}")

                # Print summary table
                summaries = cmp.get("technique_summaries", {})
                if summaries:
                    print(f"\n  Summary Table:")
                    print(f"  {'Technique':<20} {'Images':>6} {'Mean CosSim':>12} {'Mean Disp':>12}")
                    print(f"  {'-'*50}")
                    for t, s in sorted(summaries.items()):
                        print(f"  {t:<20} {s.get('n_images', 0):>6} "
                              f"{s.get('mean_cosine_similarity', 0):>12.4f} "
                              f"{s.get('mean_euclidean_displacement', 0):>12.4f}")

        except EmbeddingCacheNotFoundError as e:
            print(f"  Cache not found: {e}")
        except Exception as exc:
            print(f"  Error: {exc}")
            import traceback
            traceback.print_exc()

    # --- List output files ---
    print(f"\nOutput directory: {output}")
    if output.exists():
        for f in sorted(output.iterdir()):
            print(f"  {f.name}")
    print("\nDone.")


if __name__ == "__main__":
    main()
