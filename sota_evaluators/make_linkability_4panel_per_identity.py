"""Regenerate the 2x2 cross-condition linkability figure with PER-IDENTITY
1:N metrics (C028 follow-up): R@1 = top-1 contains ANY aligned image of the
query's CelebA identity; gallery spans 604 persons; chance R@1 = 0.3096%.
1:1 EER panel is unchanged (already person-level)."""
import argparse
import csv
import json
import sys

import numpy as np
from pathlib import Path

sys.path.insert(0, ".")
from make_linkability_4panel import _bar_panel, _scatter_panel, export_figure  # noqa: E402
from make_vertical_figures import compute_spearman  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_per_identity(p):
    out = {}
    for r in csv.DictReader(open(p)):
        out[r["technique"].strip("'")] = {
            "r1": float(r["r1_id"]), "r5": float(r["r5_id"])}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--link", required=True)
    ap.add_argument("--pi-a", required=True, help="per-identity CSV swinface")
    ap.add_argument("--pi-b", required=True, help="per-identity CSV transface")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    B = json.load(open(f"{args.link}/linkability_transface_results.json"))
    A = json.load(open(f"{args.link}/linkability_swinface_results.json"))
    ref_eer = B["aligned_ref"]["eer"] * 100
    PiA = load_per_identity(args.pi_a)
    PiB = load_per_identity(args.pi_b)
    n_id = 604
    chance = 0.3096  # group-size-weighted null R@1

    eers = [(n, d["eer"] * 100) for n, d in B["techniques"].items()]
    r1s = [(n, PiB[n]["r1"] * 100) for n in PiB]
    eers.sort(key=lambda t: -t[1])
    r1s.sort(key=lambda t: t[1])
    eeer_order = {n: i for i, (n, _) in enumerate(eers)}
    r1_order = {n: i for i, (n, _) in enumerate(r1s)}

    common = set(A["techniques"]) & set(B["techniques"])
    pts_eer = [(n, A["techniques"][n]["eer"] * 100,
                B["techniques"][n]["eer"] * 100)
               for n in sorted(common, key=lambda n: eeer_order[n])]
    pts_r1 = [(n, PiA[n]["r1"] * 100, PiB[n]["r1"] * 100)
              for n in sorted(common, key=lambda n: r1_order[n])]

    fig, axes = plt.subplots(2, 2, figsize=(6.9, 5.7), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.055, h_pad=0.045)
    _bar_panel(axes[0, 0], eers, "a", "1:1 EER (TransFace)",
               vmax_line=ref_eer, vmax_label="aligned ref",
               chance=50.0, chance_label="chance 50%")
    _bar_panel(axes[0, 1],
               r1s, "b",
               f"1:N re-ID R@1 (TransFace, {n_id}-identity gallery)",
               chance=chance, chance_label="chance \u22480.31%")
    _scatter_panel(axes[1, 0], pts_eer, "c",
                   "1:1 EER: cross-probe agreement", unit="EER (%)")
    _scatter_panel(axes[1, 1], pts_r1, "d",
                   "1:N re-ID R@1: cross-probe agreement", unit="R@1 (%)")

    out = export_figure(fig, args.output)
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
