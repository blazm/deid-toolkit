"""Quick test script for the new radar + attack surface visualizations."""
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from deid.explore.viz import plot_radar_with_attributes, plot_attack_surface_map

# Build results dict from existing CSVs on disk
RESULTS_DIR = Path("root_dir/results")

# {technique: {dataset: {eval_name: csv_path}}}
results = {}
for technique_dir in RESULTS_DIR.iterdir():
    if not technique_dir.is_dir():
        continue
    tech_name = technique_dir.name
    results[tech_name] = {}
    for dataset_dir in technique_dir.iterdir():
        if not dataset_dir.is_dir():
            continue
        ds_name = dataset_dir.name
        results[tech_name][ds_name] = {}
        for csv_file in dataset_dir.glob("*.csv"):
            eval_name = csv_file.stem
            results[tech_name][ds_name][eval_name] = csv_file

print("Techniques:", list(results.keys()))
for tech, ds_dict in results.items():
    print(f"  {tech}:")
    for ds, evals in ds_dict.items():
        print(f"    {ds}: {list(evals.keys())}")

# Generate Attribute Radar for each dataset
for ds in sorted(set(ds for t in results.values() for ds in t.keys())):
    techs = sorted(t for t, d in results.items() if ds in d)
    if not techs:
        continue

    print(f"\nGenerating attribute radar for {ds}...")
    fig1 = plot_radar_with_attributes(results, ds, techs)
    out1 = f"sample_attribute_radar_{ds}.pdf"
    fig1.savefig(out1, format="pdf", bbox_inches="tight", dpi=120)
    print(f"  -> {out1}")
    plt.close(fig1)

    # Also save PNG for quick inspection
    out1_png = f"sample_attribute_radar_{ds}.png"
    fig1 = plot_radar_with_attributes(results, ds, techs)
    fig1.savefig(out1_png, format="png", bbox_inches="tight", dpi=150)
    print(f"  -> {out1_png}")
    plt.close(fig1)

# Generate Attack Surface Map for each dataset
for ds in sorted(set(ds for t in results.values() for ds in t.keys())):
    techs = sorted(t for t, d in results.items() if ds in d)
    if not techs:
        continue

    print(f"\nGenerating attack surface map for {ds}...")
    fig2 = plot_attack_surface_map(results, ds, techs, "privacy", "ssim", "quality")
    out2 = f"sample_attack_surface_{ds}.pdf"
    fig2.savefig(out2, format="pdf", bbox_inches="tight", dpi=120)
    print(f"  -> {out2}")
    plt.close(fig2)

    out2_png = f"sample_attack_surface_{ds}.png"
    fig2 = plot_attack_surface_map(results, ds, techs, "privacy", "ssim", "quality")
    fig2.savefig(out2_png, format="png", bbox_inches="tight", dpi=150)
    print(f"  -> {out2_png}")
    plt.close(fig2)

print("\nDone! Check the sample_*.png files.")
