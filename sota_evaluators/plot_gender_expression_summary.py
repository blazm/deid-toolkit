#!/usr/bin/env python3
"""
Summarize gender and expression results across datasets.

Reads the per-dataset CSV outputs from evaluate_gender_expression.py and produces:

  1. Grouped bar chart — De-identified gender accuracy on RaFD + CelebA, side by side.
     Also shows aligned baseline accuracy as a dashed reference line.
  2. Expression table — CSV with aligned baseline deid accuracy across RaFD, MUG, KDEF.

Usage:
    conda run -n swinface python plot_gender_expression_summary.py ^
        --rafd-dir   D:\dev\deid-toolkit\root_dir\predictions\rafd ^
        --celeba-dir D:\dev\deid-toolkit\root_dir\predictions\celeba-test ^
        --mug-dir    D:\dev\deid-toolkit\root_dir\predictions\mug-still ^
        --kdef-dir   D:\dev\deid-toolkit\root_dir\predictions\kdef ^
        --output     gender_expression_summary
"""

import argparse
import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_gender_csv(path):
    """Read gender_results.csv and return list of (technique, aligned_acc, deid_acc, preservation) dicts.

    Skips the baseline row for technique-specific rows; handles N/A values.
    """
    results = []
    if not os.path.exists(path):
        print(f"  WARNING: File not found: {path}")
        return None  # no data available

    with open(path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return None

    results = []
    baseline_row = None

    for row in rows:
        name = row["Technique"].strip()
        aligned_str = row["Aligned_Gender_Accuracy"].strip()
        deid_str = row["DeID_Gender_Accuracy"].strip()
        pres_str = row["Gender_Preservation_Rate"].strip()

        if name == "aligned (Validation)":
            baseline_row = {
                "name": name,
                "aligned_acc": float(aligned_str) if aligned_str != "N/A" else None,
            }
        elif aligned_str not in ("—", ""):
            results.append({
                "name": name,
                "aligned_acc": float(aligned_str),
                "deid_acc": float(deid_str) if deid_str != "—" and deid_str != "N/A" else None,
                "preservation_rate": float(pres_str) if pres_str != "—" and pres_str != "N/A" else None,
            })

    return baseline_row, results


def read_expression_csv(path):
    """Read expression_results.csv and return technique results with preservation rates.

    Returns (baseline_row, technique_rows) where technique_rows include
    aligned_acc, deid_acc, and preservation_rate per technique.
    baseline_row is {"aligned_acc": float} for the Validation row.
    """
    results = []
    if not os.path.exists(path):
        print(f"  WARNING: File not found: {path}")
        return None, None

    with open(path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return None, None

    results = []
    baseline_row = None

    for row in rows:
        name = row["Technique"].strip()
        aligned_str = row["Aligned_Expr_Accuracy"].strip()
        deid_str = row["DeID_Expr_Accuracy"].strip()
        pres_str = row.get("Expr_Preservation_Rate", "—").strip()

        if name == "aligned (Validation)":
            baseline_row = {
                "name": name,
                "aligned_acc": float(aligned_str) if aligned_str != "N/A" else None,
            }
        elif aligned_str not in ("—", ""):
            results.append({
                "name": name,
                "aligned_acc": float(aligned_str),
                "deid_acc": float(deid_str) if deid_str != "—" and deid_str != "N/A" else None,
                "preservation_rate": float(pres_str) if pres_str not in ("—", "", "N/A") else None,
            })

    return baseline_row, results


def plot_gender_grouped(results_rafd, baseline_rafd, results_celeba, baseline_celeba, output_prefix):
    """Grouped bar chart — RaFD vs CelebA gender accuracy per technique.

    Only plots techniques present in BOTH datasets; reports missing ones.
    """
    names_rafd = set(r["name"] for r in results_rafd)
    names_celeba = set(r["name"] for r in results_celeba)
    shared = sorted(names_rafd & names_celeba)

    if not shared:
        print("No common techniques between RaFD and CelebA to plot.")
        return

    missing_rafd = names_celeba - names_rafd
    missing_celeba = names_rafd - names_celeba
    if missing_rafd:
        print(f"  Technique(s) only in CelebA (skipped from bar chart): {sorted(missing_rafd)}")
    if missing_celeba:
        print(f"  Technique(s) only in RaFD (skipped from bar chart): {sorted(missing_celeba)}")

    # Build lookup maps for fast access
    map_rafd = {r["name"]: r["deid_acc"] for r in results_rafd}
    map_celeba = {r["name"]: r["deid_acc"] for r in results_celeba}

    techniques = shared

    deid_acc_rafd = [map_rafd[t] for t in techniques]
    deid_acc_celeba = [map_celeba[t] for t in techniques]

    x = np.arange(len(techniques))
    width = 0.35

    fig, ax = plt.subplots(figsize=(len(techniques) * 1.2, 5),
                           gridspec_kw={"hspace": 0.3})

    bars_rafd = ax.bar(x - width / 2, deid_acc_rafd, width, label="RaFD", color="#4e79a7", edgecolor="#333")
    bars_celeba = ax.bar(x + width / 2, deid_acc_celeba, width, label="CelebA (test)", color="#e15759", edgecolor="#333")

    # Baseline reference lines
    baseline_rafd_val = baseline_rafd["aligned_acc"] if baseline_rafd and baseline_rafd.get("aligned_acc") else None
    baseline_celeba_val = baseline_celeba["aligned_acc"] if baseline_celeba and baseline_celeba.get("aligned_acc") else None

    ax.axhline(y=baseline_rafd_val, color="#4e79a7", linestyle="--", linewidth=1.5, alpha=0.6)
    ax.axhline(y=baseline_celeba_val, color="#e15759", linestyle="--", linewidth=1.5, alpha=0.6)

    ax.set_xlabel("")  # no label — caption covers it
    ax.set_ylabel("De-identified Gender Accuracy (%)", fontsize=12, fontweight="bold")
    # No suptitle — caption describes the figure instead
    ax.set_xticks(x)
    ax.set_xticklabels(techniques, rotation=45, ha="right", fontsize=9)
    ax.legend(loc="lower right", fontsize=10)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.2)

    # Value labels on bars (only for non-None values)
    for bar in bars_rafd:
        if bar.get_height() is not None and bar.get_height() > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{bar.get_height():.0f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

    for bar in bars_celeba:
        if bar.get_height() is not None and bar.get_height() > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{bar.get_height():.0f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

    fig.subplots_adjust(top=0.92, right=0.96, bottom=0.12)

    # Save PNG + PDF
    fig.savefig(f"{output_prefix}_gender_accuracy.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{output_prefix}_gender_accuracy.pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Gender chart: {output_prefix}_gender_accuracy.png | .pdf")

    # ── Caption ───────────────────────────────────────────────────────
    caption_lines = []
    caption_lines.append("De-identified gender classification accuracy for each anonymization technique on the RaFD and CelebA (test) datasets.")
    caption_lines.append(f"SwinFace achieves {baseline_rafd_val:.0f}% aligned accuracy on RaFD")
    caption_lines.append(f"and {baseline_celeba_val:.0f}% on CelebA (test), serving as baseline reference levels shown as dashed lines in the plot.")

    # Stats — use the map dicts directly for correct technique-name lookup
    ra_fd_vals = [(t, v) for t, v in zip(techniques, deid_acc_rafd)]
    cel_vals = [(t, v) for t, v in zip(techniques, deid_acc_celeba)]

    if ra_fd_vals:
        worst_rf = min(ra_fd_vals, key=lambda x: x[1])
        best_rf = max(ra_fd_vals, key=lambda x: x[1])
    else:
        worst_rf = best_rf = (None, None)

    if cel_vals:
        cel_min = min(v for _, v in cel_vals)
        cel_max = max(v for _, v in cel_vals)
    else:
        cel_min = cel_max = None

    if worst_rf[0]:
        caption_lines.append(f"Post-anonymization accuracy ranges from {worst_rf[0]} ({worst_rf[1]:.0f}%) to {best_rf[0]} ({best_rf[1]:.0f}%) on RaFD")
    if cel_min is not None:
        caption_lines.append(f"and from {min(v for v in deid_acc_celeba if v is not None):.0f}% to {max(v for v in deid_acc_celeba if v is not None):.0f}% on CelebA.")
    caption_lines.append(f"Dashed lines indicate the aligned (original) baseline accuracy for each dataset.")

    with open(f"{output_prefix}_gender_accuracy_caption.txt", "w", encoding="utf-8") as f:
        f.write(" ".join(caption_lines))
    print(f"Caption saved: {output_prefix}_gender_accuracy_caption.txt")


def write_expression_table(results_dict, output_file):
    """Write a compact CSV table with expression deid accuracy across RaFD, MUG, KDEF.

    results_dict: {technique_name: {dataset_name: aligned_acc, deid_acc}}
    """
    techniques = sorted(results_dict.keys())
    datasets = ["rafd", "mug-still", "kdef"]
    ds_display = {"rafd": "RaFD", "mug-still": "MUG-Still", "kdef": "KDEF"}

    with open(output_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        header = ["Technique"]
        for ds in datasets:
            header.append(f"{ds_display[ds]}_Aligned_Accuracy")
            header.append(f"{ds_display[ds]}_DeID_Accuracy")
        writer.writerow(header)

        for tech in techniques:
            row = [tech]
            for ds in datasets:
                d = results_dict.get(tech, {}).get(ds, {})
                aligned = f"{d['aligned_acc']:.1f}" if d and d.get("aligned_acc") is not None else "—"
                deid = f"{d['deid_acc']:.1f}" if d and d.get("deid_acc") is not None else "—"
                row.extend([aligned, deid])
            writer.writerow(row)

    print(f"Expression table saved: {output_file}")


def write_gender_tex(results_rafd, results_celeba, baseline_rafd_val, baseline_celeba_val, output_file):
    """Write LaTeX table of gender de-identification accuracy."""
    techniques = [r["name"] for r in results_rafd]

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{De-identified gender classification accuracy on RaFD and CelebA (test).}")
    lines.append(r"\begin{tabular}{lcc|cc}")
    lines.append(r"\toprule")
    lines.append(r"Technique & \multicolumn{2}{c|}{RaFD} & \multicolumn{2}{c}{CelebA} \\")
    lines.append(r" & Aligned & DeID & Aligned & DeID \\")
    lines.append(r"\midrule")

    # Baseline row
    lines.append(f"{baseline_rafd_val:.0f}\\% & --- & {baseline_rafd_val:.0f}\\% & {baseline_celeba_val:.0f}\\% & --- \\")
    lines.append(r"& \multicolumn{1}{l}{} & (baseline) & \multicolumn{1}{l}{} & (baseline) \\")

    for r in results_rafd:
        name = r["name"]
        a_rafd = f"{r['aligned_acc']:.0f}" if r.get("aligned_acc") is not None else "—"
        d_rafd = f"{r['deid_acc']:.0f}" if r.get("deid_acc") is not None else "—"
        celeba = next((c for c in results_celeba if c["name"] == name), None)
        a_celeba = f"{celeba['aligned_acc']:.0f}" if celeba and celeba.get("aligned_acc") is not None else "—"
        d_celeba = f"{celeba['deid_acc']:.0f}" if celeba and celeba.get("deid_acc") is not None else "—"
        lines.append(f"{name} & {a_rafd}\\% & {d_rafd}\\% & {a_celeba}\\% & {d_celeba}\\% \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"LaTeX gender table saved: {output_file}")


def write_expression_tex(results_dict, baselines, output_file):
    """Write merged LaTeX table of expression results across RaFD, MUG-Still, KDEF.

    Format mirrors gender table 8 (Table~\\ref{tab:gender}): sorted by preservation rate,
    one row per technique, compact columns with paired DeID Acc / Preservation Rate.
    Baseline accuracy shown in caption; aligned originals as Validation row.

    Parameters
    ----------
    results_dict : dict {technique: {dataset: {"aligned_acc", "deid_acc"}}}
    baselines : dict {dataset: float}  -- aligned baseline accuracies
    output_file : str
    """
    techniques = sorted(
        results_dict.keys(),
        key=lambda t: results_dict[t].get("kdef", {}).get("preservation_rate"),
        reverse=True,
    )
    datasets = ["rafd", "mug-still", "kdef"]
    ds_display = {"rafd": "RaFD", "mug-still": "MUG-Still", "kdef": "KDEF"}

    # Compute baseline accuracy for caption (average across available datasets)
    bvals = [baselines.get(ds, {}).get("aligned_acc") for ds in datasets]
    bstr = " + ".join(f"{b:.0f}%" for b in bvals if b is not None)

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    cap = (
        r"Expression classification results on de-identified faces across three datasets. "
        r"SwinFace aligned baseline accuracy: " + bstr + r". "
        r"Lower DeID accuracy indicates stronger expression erasure; "
        r"higher preservation rate means fewer prediction changes after anonymization."
    )
    lines.append(f"\caption{{{cap}}}")
    lines.append(r"\resizebox{\columnwidth}{!}{%")
    # Layout: Technique | RaFD(DeID,Pres) | MUG-Still(DeID,Pres) | KDEF(DeID,Pres)
    lines.append(r"\begin{tabular}{@{}lcc|cc|cc@{}}")
    lines.append(r"\toprule")
    lines.append(
        r"Technique & \multicolumn{2}{c|}{RaFD} & "
        r"\multicolumn{2}{c|}{MUG-Still} & "
        r"\multicolumn{2}{c@{}}{KDEF} \\")
    lines.append(r" & DeID Acc. & Pres. Rate & DeID Acc. & Pres. Rate & DeID Acc. & Pres. Rate \\")
    lines.append(r"\midrule")

    # Validation row (aligned originals)
    vcells = []
    for ds in datasets:
        b = baselines.get(ds, {}).get("aligned_acc")
        if b is not None:
            vcells.extend([f"{b:.0f}\\%", r"\textemdash{}"])
        else:
            vcells.extend([r"\textemdash{}", r"\textemdash{}"])
    lines.append(f"Validation   & {' & '.join(vcells)} \\")

    for tech in techniques:
        name = tech
        cells = []
        has_any = False
        for ds in datasets:
            d = results_dict.get(tech, {}).get(ds, {})
            deid = f"{d['deid_acc']:.0f}\\%" if d and d.get("deid_acc") is not None else r"\textemdash{}"
            pres = ""
            pr = d.get("preservation_rate")
            if pr is not None:
                pres = f"{pr:.0f}"
                has_any = True
            else:
                pres = r"\textemdash{}"
            cells.extend([deid, pres])

        if not has_any:
            continue

        lines.append(f"{name} & {' & '.join(cells)} \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append("}")  # end \resizebox
    lines.append(r"\end{table}")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"LaTeX expression table saved: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Summarize gender & expression results across datasets")
    parser.add_argument("--rafd-dir", required=True, help="RaFD predictions directory")
    parser.add_argument("--celeba-dir", required=True, help="CelebA test predictions directory")
    parser.add_argument("--mug-dir", required=True, help="MUG-Still predictions directory")
    parser.add_argument("--kdef-dir", required=True, help="KDEF predictions directory")
    parser.add_argument("--output", default="gender_expression_summary",
                        help="Output prefix (PNG/PDF/TXT files written with this prefix)")

    args = parser.parse_args()

    print("=" * 60)
    print("Gender & Expression Summary — Plots + Tables")
    print("=" * 60)

    # ── Read CSVs ─────────────────────────────────────────────────────
    print("\nRaFD gender results...")
    baseline_rafd, tech_rafd = read_gender_csv(os.path.join(args.rafd_dir, "gender_results.csv"))
    print(f"  {len(tech_rafd)} technique(s), aligned baseline: {baseline_rafd['aligned_acc'] if baseline_rafd and baseline_rafd.get('aligned_acc') else 'N/A'}%")

    print("CelebA test gender results...")
    baseline_celeba, tech_celeba = read_gender_csv(os.path.join(args.celeba_dir, "gender_results.csv"))
    print(f"  {len(tech_celeba)} technique(s), aligned baseline: {baseline_celeba['aligned_acc'] if baseline_celeba and baseline_celeba.get('aligned_acc') else 'N/A'}%")

    # ── Gender grouped bar chart ──────────────────────────────────────
    print("\nGenerating gender grouped bar chart...")
    plot_gender_grouped(tech_rafd, baseline_rafd, tech_celeba, baseline_celeba, args.output)

    # ── Expression accuracy table ─────────────────────────────────────
    print("\nBuilding expression accuracy table across RaFD/MUG/KDEF...")
    expr_results = {}  # {technique: {dataset: {"aligned_acc", "deid_acc", "preservation_rate"}}}
    expr_baselines = {}  # {dataset: {"aligned_acc": float}}

    for ds_name, ds_dir in [("rafd", args.rafd_dir), ("mug-still", args.mug_dir), ("kdef", args.kdef_dir)]:
        baseline_exp, tech_exp = read_expression_csv(os.path.join(ds_dir, "expression_results.csv"))
        if baseline_exp is None:
            print(f"  {ds_name}: no expression data")
            continue

        expr_baselines[ds_name] = {
            "aligned_acc": baseline_exp.get("aligned_acc"),
        }

        for t in tech_exp:
            if t["name"] not in expr_results:
                expr_results[t["name"]] = {}
            expr_results[t["name"]][ds_name] = {
                "aligned_acc": t["aligned_acc"],
                "deid_acc": t["deid_acc"],
                "preservation_rate": t.get("preservation_rate"),
            }

    output_expr_table = args.output + "_expression_accuracy.csv"
    write_expression_table(expr_results, output_expr_table)

    # ── LaTeX tables ────────────────────────────────────────────────
    baseline_rafd_val = baseline_rafd["aligned_acc"] if baseline_rafd and baseline_rafd.get("aligned_acc") else None
    baseline_celeba_val = baseline_celeba["aligned_acc"] if baseline_celeba and baseline_celeba.get("aligned_acc") else None

    output_gender_tex = args.output + "_gender_accuracy.tex"
    write_gender_tex(tech_rafd, tech_celeba, baseline_rafd_val, baseline_celeba_val, output_gender_tex)

    output_expr_tex = args.output + "_expression_accuracy.tex"
    write_expression_tex(expr_results, expr_baselines, output_expr_tex)

    print()
    print("=" * 60)
    print(f"All outputs saved with prefix: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
