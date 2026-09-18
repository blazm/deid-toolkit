#!/usr/bin/env python
"""
Double-column 2x2 naturalness--privacy--utility trade-off figure (v2).

Encodings (no number grids; all colors/markers per fig_style, as in the
rest of the paper):
  (a) SwinFace   scatter: x = P (privacy), y = N (naturalness),
      bubble size = U in THREE quantized tiers (small/med/large by U tertile);
      (color, marker) = technique; dashed = Pareto front; top-3 frontier
      labelled; black open circle = Original.
  (b) TransFace: same.
  (c) Utility bars (rows sorted by overall-SwinFace desc): per technique,
      dark bar = mean gender retention (4 datasets), light bar = mean
      expression retention (3 datasets) -> the gender-vs-expression gap is
      the length difference in every row.
  (d) Overall bars (same row order): dark = overall (SwinFace),
      light = overall (TransFace); right-edge strip = technique family.

Reads ONLY per-technique tradeoff JSON + _meta.json (no inference).
Emits PNG (300 dpi) + PDF + SVG.
"""
import argparse, json, os, sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fig_style

GEN = ["AIDPro", "AMT-GAN", "AnonNET", "CLEANIR", "CPP-DeID", "DeepPrivacy",
       "DeepPrivacy2", "DiffPrivate", "FADM", "FALCO", "FAMS", "G2Face",
       "GANonymization", "IPFA", "LDFA", "NullFace", "PRO-Face", "RP", "RiDDLE",
       "iFADIT"]
# three size tiers for U (pt^2): keeps U visible, avoids giant overlapping discs
TIER_SIZES = [55, 150, 300]


def u_tier(u, lo, hi):
    if u < lo:
        return TIER_SIZES[0]
    if u < hi:
        return TIER_SIZES[1]
    return TIER_SIZES[2]


def load(root):
    d = os.path.join(root, "root_dir", "predictions", "tradeoff", "celeba-test")
    data = {}
    for t in GEN:
        data[t] = json.load(open(os.path.join(d, t + ".json"), encoding="ascii"))
    meta = json.load(open(os.path.join(d, "_meta.json"), encoding="ascii"))
    return data, meta


def panel_scatter(ax, data, meta, bk, u_lo, u_hi):
    for t in GEN:
        r = data[t]["backbones"][bk]
        u = data[t]["U"]["u"]
        if r["p"] is None or u is None:
            continue
        c, m = fig_style.style_for(t)
        ax.scatter([r["p"]], [data[t]["N"]["n"]], s=u_tier(u, u_lo, u_hi),
                   color=c, marker=m, edgecolor="k", linewidth=0.35, zorder=3,
                   alpha=0.95)
    ax.scatter([0.05], [0.97], s=200, facecolor="none", edgecolor="k",
               linewidth=1.1, marker="o", zorder=4)
    ax.text(0.115, 0.955, "Original", fontsize=4.8, va="center", zorder=4)
    front = [t for t in meta["pareto_fronts"][bk]]
    fp = sorted([(data[t]["backbones"][bk]["p"], data[t]["N"]["n"]) for t in front])
    if len(fp) >= 2:
        ax.plot([x for x, _ in fp], [y for _, y in fp], linestyle="--",
                color="0.35", linewidth=0.8, zorder=2)
        ax.text(0.985, 0.035, "Pareto front", transform=ax.transAxes,
                ha="right", fontsize=4.4, color="0.35")
    OFFS = {"G2Face": (-6, 9, "right"), "RP": (5, 9, "left"),
            "AIDPro": (5, -13, "left"), "CPP-DeID": (6, -12, "left")}
    for t in sorted(front, key=lambda x: -data[x]["overall"][bk])[:3]:
        ox, oy, ha = OFFS.get(t, (5, 5, "left"))
        ax.annotate(t, (data[t]["backbones"][bk]["p"], data[t]["N"]["n"]),
                    xytext=(ox, oy), textcoords="offset points", fontsize=4.8,
                    fontweight="bold", ha=ha, zorder=5)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Privacy score P (0 = fully linkable, 1 = at chance)",
                  fontsize=5.2, labelpad=2)
    ax.set_ylim(0, 1.05)
    ax.tick_params(labelsize=4.6)
    ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.6)
    ax.set_aspect("equal")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get("DEID_ROOT", "D:/dev/deid-toolkit"))
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    data, meta = load(args.root)
    out = args.output or "tradeoff_4panel_2col"

    us = np.array([data[t]["U"]["u"] for t in GEN])
    u_lo, u_hi = np.percentile(us, [33.3, 66.7])

    order = sorted(GEN, key=lambda t: -data[t]["overall"].get("swinface", 0))
    y = np.arange(len(order))[::-1]  # top row = rank 1

    fig, axs = plt.subplots(2, 2, figsize=(7.0, 7.6), constrained_layout=True)
    axa, axb, axc, axd = axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]

    panel_scatter(axa, data, meta, "swinface", u_lo, u_hi)
    panel_scatter(axb, data, meta, "transface", u_lo, u_hi)
    axa.set_title(" (a) SwinFace", fontsize=6.0, loc="left", fontweight="bold")
    axb.set_title(" (b) TransFace", fontsize=6.0, loc="left", fontweight="bold")
    axa.set_ylabel("Naturalness score N", fontsize=5.2, labelpad=2)
    axb.text(0.01, 0.03, "3 bubble sizes = Utility U (low / mid / high)",
             transform=axb.transAxes, ha="left", fontsize=4.3)

    # (c) utility: gender (dark) vs expression (light) bars
    for yi, t in zip(y, order):
        dk, lt = fig_style.pair_for(t)
        g = data[t]["U"]["gender"]
        e = data[t]["U"]["expression"]
        gv = float(np.mean(list(g.values()))) if g else 0.0
        ev = float(np.mean(list(e.values()))) if e else 0.0
        axc.barh(yi + 0.19, gv, height=0.34, color=dk, zorder=3)
        axc.barh(yi - 0.19, ev, height=0.34, color=lt, zorder=3)
    axc.set_yticks(y)
    axc.set_yticklabels(order, fontsize=4.4)
    axc.set_xlim(0, 1)
    axc.set_ylim(-0.75, len(order) - 0.25)
    axc.set_xlabel("Baseline-relative retention (0 = lost, 100% = fully kept)",
                   fontsize=5.2, labelpad=2)
    axc.xaxis.set_major_formatter(lambda v, p: f"{v*100:.0f}%")
    axc.tick_params(labelsize=4.6)
    axc.grid(True, axis="x", linestyle=":", linewidth=0.4, alpha=0.6)
    axc.set_title(" (c) Utility: gender (dark bar) vs. expression (light bar)",
                  fontsize=6.0, loc="left", fontweight="bold")

    # (d) overall per backbone
    for yi, t in zip(y, order):
        dk, lt = fig_style.pair_for(t)
        axd.barh(yi + 0.19, data[t]["overall"]["swinface"], height=0.34,
                 color=dk, zorder=3)
        axd.barh(yi - 0.19, data[t]["overall"]["transface"], height=0.34,
                 color=lt, zorder=3)
        axd.add_patch(plt.Rectangle((1.003, yi - 0.5), 0.012, 1.0,
                       transform=axd.transData, color=dk, clip_on=False))
    axd.set_yticks([])
    axd.set_xlim(0, 1)
    axd.set_ylim(-0.75, len(order) - 0.25)
    axd.set_xlabel("Overall score  (P + N + U) / 3", fontsize=5.2, labelpad=2)
    axd.tick_params(labelsize=4.6)
    axd.grid(True, axis="x", linestyle=":", linewidth=0.4, alpha=0.6)
    axd.set_title(" (d) Overall: SwinFace (dark bar) vs. TransFace (light bar)",
                  fontsize=6.0, loc="left", fontweight="bold")

    for ext in ("png", "pdf", "svg"):
        p = f"{out}.{ext}"
        fig.savefig(p, dpi=300 if ext == "png" else None)
        print("saved", p)


if __name__ == "__main__":
    main()
