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


def _read_csv_rows(path):
    """Read CSV lines with an explicit encoding fallback (utf-8 then cp1252)."""
    data = open(path, "rb").read()
    text = None
    for enc in ("utf-8", "cp1252"):
        try:
            text = data.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        text = data.decode("utf-8", errors="replace")
    return list(csv.DictReader(text.splitlines()))


def read_gender_csv(path):
    """Read gender_results.csv and return list of (technique, aligned_acc, deid_acc, preservation) dicts.

    Skips the baseline row for technique-specific rows; handles N/A values.
    """
    results = []
    if not os.path.exists(path):
        print(f"  WARNING: File not found: {path}")
        return None  # no data available

    rows = _read_csv_rows(path)

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

    rows = _read_csv_rows(path)

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


def plot_gender_grouped(dataset_list, output_prefix):
    """Grouped bar chart — de-identified gender accuracy per technique, one bar per dataset.

    dataset_list: [(label, results, baseline_row), ...] in display order
    (e.g. RaFD, MUG, KDEF, CelebA-test). Plots techniques present in ALL datasets.
    """
    DS_COLORS = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759"]

    name_sets = [set(r["name"] for r in res) for _, res, _ in dataset_list]
    shared = sorted(set.intersection(*name_sets)) if name_sets else []
    if not shared:
        print("No common techniques between datasets to plot.")
        return
    for (label, res, _), ns in zip(dataset_list, name_sets):
        missing = ns - set(shared)
        if missing:
            print(f"  Technique(s) without {label} (skipped): {sorted(missing)}")

    maps = {label: {r["name"]: r["deid_acc"] for r in res}
            for label, res, _ in dataset_list}
    n_ds = len(dataset_list)
    techniques = shared

    # Wide two-column layout: single row, ordered by the first dataset's
    # de-identified accuracy (matches the tab:gender_all_datasets row order)
    wide = len(techniques) > 12
    if wide:
        first = maps[dataset_list[0][0]]
        order = sorted(range(len(techniques)), key=lambda i: first[techniques[i]], reverse=True)
        techniques = [techniques[i] for i in order]

    x = np.arange(len(techniques))
    width = 0.8 / n_ds
    fig, ax_row = plt.subplots(1, 1, figsize=(16.8 if wide else 11, 5.2))

    def bar_values(label):
        return [maps[label].get(t) for t in techniques]

    for k, (label, res, baseline_row) in enumerate(dataset_list):
        vals = bar_values(label)
        pos = x + (k - (n_ds - 1) / 2.0) * width
        ax_row.bar(pos, vals, width, label=label,
                   color=DS_COLORS[k % len(DS_COLORS)], edgecolor="#333")
        bv = baseline_row.get("aligned_acc") if baseline_row else None
        if bv:
            ax_row.axhline(y=bv, color=DS_COLORS[k % len(DS_COLORS)],
                           linestyle="--", linewidth=1.2, alpha=0.55)

    # Value labels, staggered vertically per bar so adjacent bars never collide
    for k in range(n_ds):
        for i, v in enumerate(bar_values(dataset_list[k][0])):
            if v is None:
                continue
            offset = 1.0 if k % 2 == 0 else 3.6
            ax_row.text(x[i] + (k - (n_ds - 1) / 2.0) * width, v + offset,
                        f"{v:.0f}%", ha="center", va="bottom",
                        fontsize=7.5, fontweight="bold")

    ax_row.set_xlabel("")  # no label -- caption covers it
    ax_row.set_xticks(x)
    ax_row.set_xticklabels(techniques, rotation=45, ha="right", fontsize=9.5)
    ax_row.set_xlim(-0.5, len(techniques) - 0.5)  # bars flush with both axes edges (no inner gap)
    ax_row.set_ylim(0, 110)
    ax_row.grid(axis="y", alpha=0.2)
    ax_row.set_ylabel("De-identified Gender Accuracy (%)", fontsize=12, fontweight="bold")
    # legend: upper-right is empty for the wide plot (tallest bars are mid-plot)
    ax_row.legend(loc="upper right" if wide else "lower right",
                  fontsize=9, ncol=(n_ds if wide else 1))

    fig.subplots_adjust(top=0.93, left=0.045, right=0.975, bottom=0.10)

    # Save PNG + PDF
    fig.savefig(f"{output_prefix}_gender_accuracy.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{output_prefix}_gender_accuracy.pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Gender chart: {output_prefix}_gender_accuracy.png | .pdf")

    # ── Caption ───────────────────────────────────────────────────────
    caption_lines = ["De-identified gender classification accuracy for each anonymization technique:"]
    parts = []
    for k, (label, res, baseline_row) in enumerate(dataset_list):
        vals = [v for v in bar_values(label) if v is not None]
        bv = baseline_row.get("aligned_acc") if baseline_row else None
        if vals and bv:
            parts.append(f"{label} {min(vals):.0f}–{max(vals):.0f}% (aligned {bv:.0f}%)")
    caption_lines.append("; ".join(parts) + ".")
    caption_lines.append("Dashed lines indicate the aligned (original) baseline accuracy for each dataset.")

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

    print("MUG gender results...")
    baseline_mug, tech_mug = read_gender_csv(os.path.join(args.mug_dir, "gender_results.csv"))
    print(f"  {len(tech_mug)} technique(s), aligned baseline: {baseline_mug['aligned_acc'] if baseline_mug and baseline_mug.get('aligned_acc') else 'N/A'}%")

    print("KDEF gender results...")
    baseline_kdef, tech_kdef = read_gender_csv(os.path.join(args.kdef_dir, "gender_results.csv"))
    print(f"  {len(tech_kdef)} technique(s), aligned baseline: {baseline_kdef['aligned_acc'] if baseline_kdef and baseline_kdef.get('aligned_acc') else 'N/A'}%")

    # ── Gender grouped bar chart ──────────────────────────────────────
    print("\nGenerating gender grouped bar chart...")
    plot_gender_grouped([("RaFD", tech_rafd, baseline_rafd),
                         ("MUG", tech_mug, baseline_mug),
                         ("KDEF", tech_kdef, baseline_kdef),
                         ("CelebA (test)", tech_celeba, baseline_celeba)],
                        args.output)

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
