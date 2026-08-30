#!/usr/bin/env python3
"""
Render VERTICAL (stacked) versions of the two-panel figures for the
two-column manuscript, reading from the saved JSON results (no model
inference, no re-analysis):

  linkability <model> 2-panel   -> linkability_<model>_2panel_v.{png,pdf,svg}
  linkability scatter           -> linkability_scatter_v.{png,pdf,svg}
  id_mapping  <model> 2-panel   -> id_mapping_<model>_2panel_v.{png,pdf,svg}

Each output goes next to the source JSON file.

Usage:
    python make_vertical_figures.py \
        --link  D:\dev\deid-toolkit\root_dir\predictions \
        --idmap D:\dev\deid-toolkit\root_dir\predictions
"""

import argparse
import json
import os
import time

import numpy as np


TECHNIQUE_COLORS = [
    "#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f", "#edc948",
    "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac", "#af7aa1", "#86bcb6",
    "#cdc500", "#d37295", "#fac864", "#a6d854",
]


def color_for(name):
    if name == "aligned":
        return "#000000"
    return TECHNIQUE_COLORS[sum(ord(c) for c in name) % len(TECHNIQUE_COLORS)]


def _savefig_safe(fig, path, **kw):
    for attempt in range(3):
        try:
            fig.savefig(path, bbox_inches="tight", **kw)
            return
        except PermissionError:
            if attempt == 2:
                print(f"  WARNING: could not save {path} (file locked?)")
                return
            time.sleep(0.5)


def export_figure(fig, base):
    _savefig_safe(fig, base + ".png", dpi=300)
    _savefig_safe(fig, base + ".pdf", format="pdf")
    _savefig_safe(fig, base + ".svg", format="svg")
    return base + ".png"


def compute_spearman(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(x)
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    mx, my = rx.mean(), ry.mean()
    den = np.sqrt(((rx - mx) ** 2).sum() * ((ry - my) ** 2).sum())
    rho = float(((rx - mx) * (ry - my)).sum() / den) if den > 0 else float("nan")
    from math import sqrt
    from analyze_linkability import _erf
    if n > 2 and abs(rho) < 1:
        t = rho * sqrt((n - 2) / (1 - rho ** 2))
        try:
            from scipy import stats
            pval = float(2 * stats.t.sf(abs(t), n - 2))
        except Exception:
            pval = float(2 * (1 - 0.5 * (1 + np.sign(t) * _erf(abs(t) / np.sqrt(2)))))
    else:
        pval = 1.0 if abs(rho) >= 1 else 0.0
    return rho, pval


# --------------------------------------------------------------------------

def plot_linkability_v(data, base):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model = data["model"]
    n_gal = data["gallery_size"]
    ref = data["aligned_ref"]
    techs = list(data["techniques"].items())
    rs = sorted(techs, key=lambda x: x[1]["eer"])
    names = [n for n, _ in rs]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 13.5))

    # (a) 1:1 EER, sorted ascending
    eers = [r["eer"] * 100 for _, r in rs]
    ax1.bar(range(len(rs)), eers, color=[color_for(n) for n in names], edgecolor="white")
    ax1.axhline(50, color="gray", linestyle=":", linewidth=1.2, alpha=0.7,
                label="Chance (50%)")
    ax1.axhline(ref["eer"] * 100, color="black", linestyle="--", linewidth=1.6,
                label=f"Aligned same-condition EER = {ref['eer']*100:.2f}% (ref.)")
    for b, v in zip(ax1.patches, eers):
        ax1.annotate(f"{v:.1f}", (b.get_x() + b.get_width() / 2, v),
                     ha="center", va="bottom", fontsize=10)
    ax1.set_xticks(range(len(rs)))
    ax1.set_xticklabels(names, rotation=45, ha="right", fontsize=11)
    ax1.set_ylim(0, max(66.0, max(eers) * 1.12, ref["eer"] * 100 * 1.15))
    ax1.set_ylabel("1:1 Linkability EER (%)", fontsize=15, fontweight="bold")
    ax1.set_title("(a) Cross-condition 1:1 verification — de-identified probe vs. aligned gallery",
                  fontsize=14, fontweight="bold", pad=12)
    ax1.tick_params(axis="both", labelsize=12, length=6, direction="inout")
    ax1.grid(True, axis="y", alpha=0.3)
    ax1.legend(loc="upper right", fontsize=11, framealpha=0.95)

    # (b) 1:N R@1, sorted ascending
    rv = sorted(techs, key=lambda x: x[1]["r1"])
    r1s = [r["r1"] * 100 for _, r in rv]
    ax2.bar(range(len(rv)), r1s, color=[color_for(n) for n, _ in rv], edgecolor="white")
    ax2.axhline(100.0 / n_gal, color="gray", linestyle=":", linewidth=1.2, alpha=0.7,
                label=f"Chance = 1/{n_gal} ≈ {100.0/n_gal:.3f}%")
    for b, v in zip(ax2.patches, r1s):
        ax2.annotate(f"{v:.1f}", (b.get_x() + b.get_width() / 2, v),
                     ha="center", va="bottom", fontsize=10)
    ax2.set_xticks(range(len(rv)))
    ax2.set_xticklabels([n for n, _ in rv], rotation=45, ha="right", fontsize=11)
    ax2.set_ylim(0, max(r1s) * 1.18 + 0.5)
    ax2.set_xlabel("Anonymization Technique", fontsize=15, fontweight="bold")
    ax2.set_ylabel("Open-set 1:N re-ID rate R@1 (%)", fontsize=15, fontweight="bold")
    ax2.set_title(f"(b) 1:N re-identification attack — de-identified probe vs. {n_gal}-face aligned gallery",
                  fontsize=14, fontweight="bold", pad=12)
    ax2.tick_params(axis="both", labelsize=12, length=6, direction="inout")
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.legend(loc="upper right", fontsize=11, framealpha=0.95)

    fig.suptitle(f"Cross-Condition Linkability — {model}", fontsize=18, fontweight="bold", y=0.995)
    fig.tight_layout()
    out = export_figure(fig, base)
    plt.close(fig)
    print(f"  {out}")


def plot_linkability_scatter_v(path_a, path_b, base):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    A = json.load(open(path_a))
    B = json.load(open(path_b))
    label_a, label_b = A["model"], B["model"]
    names = sorted(set(A["techniques"]) & set(B["techniques"]))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 12.5))
    for ax, key, unit, title in (
        (ax1, "eer", "1:1 Linkability EER (%)", "(a) 1:1 linkability EER"),
        (ax2, "r1", "Open-set 1:N R@1 (%)", "(b) 1:N re-identification R@1"),
    ):
        x = np.array([A["techniques"][n][key] for n in names]) * 100
        y = np.array([B["techniques"][n][key] for n in names]) * 100
        ax.scatter(x, y, s=100, color=[color_for(n) for n in names],
                   edgecolor="k", linewidth=0.5, zorder=3)
        lim = [min(x.min(), y.min()) * 0.95, max(x.max(), y.max()) * 1.08]
        if lim[0] < 0:
            lim[0] = 0
        ax.plot(lim, lim, "k--", linewidth=1.2, alpha=0.6, label="y = x")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        for n, xi, yi in zip(names, x, y):
            ax.annotate(n, (xi, yi), textcoords="offset points", xytext=(7, 5), fontsize=9)
        rho, p = compute_spearman(x, y)
        ptxt = f"p={p:.2e}" if p < 0.05 else f"p={p:.3f}"
        ax.text(0.03, 0.97, f"Spearman $\\rho$ = {rho:.3f}\n{ptxt}",
                transform=ax.transAxes, va="top", fontsize=13,
                bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow", ec="gray"))
        ax.set_title(title, fontsize=15, fontweight="bold", pad=12)
        ax.set_xlabel(f"{label_a} — {unit}", fontsize=14, fontweight="bold")
        ax.set_ylabel(f"{label_b} — {unit}", fontsize=14, fontweight="bold")
        ax.tick_params(axis="both", labelsize=12, length=6, direction="inout")
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Cross-Probe Agreement of Cross-Condition Linkability — {label_a} vs. {label_b}",
                 fontsize=17, fontweight="bold", y=0.995)
    fig.tight_layout()
    out = export_figure(fig, base)
    plt.close(fig)
    print(f"  {out}")


def plot_idmap_v(data, base):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model = data["model"]
    rows = data["rows"]
    tech = [(n, d) for n, d in rows.items() if n != "aligned"]
    ref = rows["aligned"]
    names = [n for n, _ in sorted(tech, key=lambda x: x[0])]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 13.0))

    # (a) quadrant: split (x) vs mixing (y)
    s = {n: d["person_split_rate"] * 100 for n, d in tech}
    m = {n: d["mixing_rate"] * 100 for n, d in tech}
    for n in names:
        ax1.scatter(s[n], m[n], s=110, color=color_for(n), edgecolor="k",
                    linewidth=0.5, zorder=3)
        ax1.annotate(n, (s[n], m[n]), textcoords="offset points",
                     xytext=(7, 5), fontsize=9)
    ax1.scatter(ref["person_split_rate"] * 100, ref["mixing_rate"] * 100,
                marker="o", s=140, facecolor="none", edgecolor="black",
                linewidth=2.0, zorder=4, label="Aligned (ref.)")
    ax1.axhline(50, color="gray", linestyle=":", linewidth=1)
    ax1.axvline(50, color="gray", linestyle=":", linewidth=1)
    lim = max([max(s.values()), max(m.values()),
               ref["person_split_rate"] * 100, ref["mixing_rate"] * 100]) * 1.15 + 5
    ax1.set_xlim(0, lim)
    ax1.set_ylim(0, lim)
    ax1.text(lim * 0.2, 97, "1:1 consistent", ha="center", va="top", fontsize=11,
             color="#336633", fontweight="bold")
    ax1.text(lim * 0.78, 97, "M:N mixed", ha="center", va="top", fontsize=11,
             color="#663366", fontweight="bold")
    ax1.text(lim * 0.2, 3, "N:1 collapse", ha="center", va="bottom", fontsize=11,
             color="#993333", fontweight="bold")
    ax1.text(lim * 0.78, 3, "1:N drift", ha="center", va="bottom", fontsize=11,
             color="#996633", fontweight="bold")
    ax1.set_xlabel("Person-split rate (%) — 1:N identity drift", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Cluster mixing rate (%) — N:1 identity collapse", fontsize=14, fontweight="bold")
    ax1.set_title("(a) Identity mapping structure — split vs. mixing "
                  "(cluster threshold from aligned EER)",
                  fontsize=14, fontweight="bold", pad=12)
    ax1.tick_params(axis="both", labelsize=12, length=6, direction="inout")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left", fontsize=11)

    # (b) CMC-1 misassignment bars
    order = sorted(names)
    vals = [rows[n]["cmc1_misassignment"] * 100 for n in order]
    ax2.bar(range(len(order)), vals, color=[color_for(n) for n in order], edgecolor="white")
    refv = ref["cmc1_misassignment"] * 100
    ax2.axhline(refv, color="black", linestyle="--", linewidth=1.5,
                label=f"Aligned (ref.) = {refv:.2f}%")
    for b, v in zip(ax2.patches, vals):
        ax2.annotate(f"{v:.1f}", (b.get_x() + b.get_width() / 2, v),
                     ha="center", va="bottom", fontsize=10)
    ax2.set_xticks(range(len(order)))
    ax2.set_xticklabels(order, rotation=45, ha="right", fontsize=11)
    ax2.set_ylim(0, max(max(vals), refv) * 1.18 + 2)
    ax2.set_xlabel("Anonymization Technique", fontsize=15, fontweight="bold")
    ax2.set_ylabel("CMC-1 misassignment rate (%)", fontsize=14, fontweight="bold")
    ax2.set_title("(b) CMC-1 cross-identity match — nearest aligned face belongs to a different person",
                  fontsize=14, fontweight="bold", pad=12)
    ax2.tick_params(axis="both", labelsize=12, length=6, direction="inout")
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.legend(loc="upper right", fontsize=11)

    fig.suptitle(f"Identity Mapping Structure — {model}", fontsize=18, fontweight="bold", y=0.995)
    fig.tight_layout()
    out = export_figure(fig, base)
    plt.close(fig)
    print(f"  {out}")


# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--link", type=str, default=None,
                   help="Dir containing linkability_<model>_results.json files")
    p.add_argument("--idmap", type=str, default=None,
                   help="Dir containing id_mapping_<model>_results.json files")
    args = p.parse_args()

    if args.link:
        print("Linkability (vertical):")
        for model in ("swinface", "transface"):
            jp = os.path.join(args.link, f"linkability_{model}_results.json")
            if not os.path.exists(jp):
                print(f"  SKIP {jp} (not found)")
                continue
            plot_linkability_v(json.load(open(jp)),
                               os.path.join(args.link, f"linkability_{model}_2panel_v"))
        pa = os.path.join(args.link, "linkability_swinface_results.json")
        pb = os.path.join(args.link, "linkability_transface_results.json")
        if os.path.exists(pa) and os.path.exists(pb):
            plot_linkability_scatter_v(pa, pb, os.path.join(args.link, "linkability_scatter_v"))

    if args.idmap:
        print("Identity mapping (vertical):")
        for model in ("swinface", "transface"):
            jp = os.path.join(args.idmap, f"id_mapping_{model}_results.json")
            if not os.path.exists(jp):
                print(f"  SKIP {jp} (not found)")
                continue
            plot_idmap_v(json.load(open(jp)),
                         os.path.join(args.idmap, f"id_mapping_{model}_2panel_v"))


if __name__ == "__main__":
    main()
