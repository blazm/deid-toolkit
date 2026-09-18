#!/usr/bin/env python
"""
Double-column 2x2 naturalness--utility--privacy trade-off figure.

Panels:
  (a) Pareto scatter, SwinFace:  x = P (privacy, SwinFace), y = N (naturalness),
      bubble size = U (utility), (color, marker) = technique via fig_style;
      original (aligned) face = black open circle (top-left corner anchor);
      dashed line = Pareto front; top-overall frontier techniques labelled.
  (b) same, TransFace backbone.
  (c) utility retention heatmap 20 x 7 (gender: CelebA RaFD MUG KDEF |
      expression: RaFD MUG KDEF); rows sorted by overall (SwinFace) desc;
      right-edge strip = technique family color.
  (d) scorecard heatmap 20 x 8: P components (SwinFace), N components,
      overall per backbone; cells = normalized score, annotated with raw values.

Reads ONLY the per-technique tradeoff JSON + _meta.json (no inference).
Emits PNG (300 dpi) + PDF + SVG.

Usage:
  python make_tradeoff_figure.py --output <.../figs/tradeoff_4panel_2col>
  (--root D:/dev/deid-toolkit for the predictions location)
"""
import argparse, json, os, sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import Normalize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fig_style

GEN = ["AIDPro", "AMT-GAN", "AnonNET", "CLEANIR", "CPP-DeID", "DeepPrivacy",
       "DeepPrivacy2", "DiffPrivate", "FADM", "FALCO", "FAMS", "G2Face",
       "GANonymization", "IPFA", "LDFA", "NullFace", "PRO-Face", "RP", "RiDDLE",
       "iFADIT"]
G_COLS = ["celeba-test", "rafd", "mug-still", "kdef"]
E_COLS = ["rafd", "mug-still", "kdef"]
COL_SHORT = {"celeba-test": "CelebA", "rafd": "RaFD", "mug-still": "MUG", "kdef": "KDEF"}
CMAP = "YlGnBu"  # dark = high score/good


def load(root):
    d = os.path.join(root, "root_dir", "predictions", "tradeoff", "celeba-test")
    data = {}
    for t in GEN:
        data[t] = json.load(open(os.path.join(d, t + ".json"), encoding="ascii"))
    meta = json.load(open(os.path.join(d, "_meta.json"), encoding="ascii"))
    return data, meta, d


def panel_scatter(ax, data, meta, bk):
    uvals = np.array([v["U"]["u"] for v in data.values()])
    usmin, usmax = uvals.min(), uvals.max()
    def bsize(u):
        return 60 + 800 * (u - usmin) / max(1e-9, usmax - usmin)
    for t in GEN:
        r = data[t]["backbones"][bk]
        if r["p"] is None or data[t]["U"]["u"] is None:
            continue
        c, m = fig_style.style_for(t)
        ax.scatter([r["p"]], [data[t]["N"]["n"]], s=bsize(data[t]["U"]["u"]),
                   color=c, marker=m, edgecolor="k", linewidth=0.35, zorder=3)
    # original (aligned) anchor: P ~ 0, N = 1, U = 1
    ax.scatter([0.05], [0.97], s=240, facecolor="none", edgecolor="k",
               linewidth=1.1, marker="o", zorder=4)
    ax.text(0.11, 0.965, "Original", fontsize=4.6, va="center", zorder=4)
    # pareto front
    front = [t for t in meta["pareto_fronts"][bk]]
    fp = sorted([(data[t]["backbones"][bk]["p"], data[t]["N"]["n"]) for t in front])
    if len(fp) >= 2:
        fs = [f"{x:.3f}" for x, _ in fp]
        ax.plot([x for x, _ in fp], [y for _, y in fp], linestyle="--",
                color="0.35", linewidth=0.8, zorder=2)
        ax.text(0.985, 0.04, "Pareto front", transform=ax.transAxes,
                ha="right", fontsize=4.4, color="0.35")
    # label top frontier by overall
    OFFS = {"G2Face": (-6, 9, "right"), "RP": (5, 9, "left"),
            "AIDPro": (5, -13, "left"), "CPP-DeID": (6, -12, "left")}
    for t in sorted(front, key=lambda x: -data[x]["overall"][bk])[:3]:
        ox, oy, ha = OFFS.get(t, (5, 5, "left"))
        ax.annotate(t, (data[t]["backbones"][bk]["p"], data[t]["N"]["n"]),
                    xytext=(ox, oy), textcoords="offset points", fontsize=4.6,
                    fontweight="bold", ha=ha, zorder=5)
    for label, on in (("x", True), ("y", True)):
        ax.set_xlim(0, 1)
    ax.set_xlabel(("Privacy P" if True else ""), fontsize=5.4)
    ax.set_ylim(0, 1.06)
    ax.tick_params(labelsize=4.6)
    ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.6)
    ax.set_aspect("equal")


def family_strip(ax, order):
    for i, t in enumerate(order):
        d, l = fig_style.pair_for(t)
        _, marker = fig_style.style_for(t)
        if marker == "o":
            ax.add_patch(plt.Rectangle((0.985, 1 - (i + 1) / len(order)),
                                       0.015, 1 / len(order),
                                       transform=ax.transAxes,
                                       color=d, clip_on=False))
        else:
            ax.add_patch(plt.Rectangle((0.985, 1 - (i + 1) / len(order)),
                                       0.015, 1 / len(order),
                                       transform=ax.transAxes,
                                       color=l, clip_on=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get("DEID_ROOT", "D:/dev/deid-toolkit"))
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    data, meta, _ = load(args.root)
    out = args.output or "tradeoff_4panel_2col"

    # row order: overall (swinface) desc
    order = sorted(GEN, key=lambda t: -data[t]["overall"].get("swinface", 0))
    labels = order.copy()

    fig, axs = plt.subplots(2, 2, figsize=(7.0, 7.9), constrained_layout=True)
    axa, axb, axc, axd = axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]

    bknames = {"swinface": "SwinFace", "transface": "TransFace"}
    panel_scatter(axa, data, meta, "swinface")
    panel_scatter(axb, data, meta, "transface")
    axa.set_title(" (a) SwinFace", fontsize=5.6, loc="left", fontweight="bold")
    axb.set_title(" (b) TransFace", fontsize=5.6, loc="left", fontweight="bold")
    for ax in (axa, axb):
        ax.set_xlabel("Privacy score  P (0 = fully linkable, 1 = at chance)",
                      fontsize=5.0, labelpad=2)
    axa.set_ylabel("Naturalness score  N", fontsize=5.0, labelpad=2)
    axb.text(0.01, 0.02, "bubble size = Utility U", transform=axb.transAxes,
            ha="left", fontsize=4.4)

    # (c) utility retention heatmap
    uc = G_COLS + E_COLS
    M = np.array([[data[t]["U"]["gender"].get(d, np.nan) if d in G_COLS
                   else data[t]["U"]["expression"].get(d, np.nan) for d in uc]
                  for t in order])
    im = axc.imshow(M, aspect="auto", cmap=CMAP, norm=Normalize(vmin=0, vmax=1))
    axc.set_xticks(range(len(uc)))
    axc.set_xticklabels([COL_SHORT[d] for d in uc], fontsize=4.4)
    axc.set_yticks(range(len(order)))
    axc.set_yticklabels(labels, fontsize=4.2)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if not np.isnan(v):
                axc.text(j, i, f"{v*100:.0f}%", ha="center", va="center",
                         fontsize=3.7, color="w" if v > 0.5 else "k")
    # group divider + labels
    axc.axvline(len(G_COLS) - 0.5, color="k", linewidth=0.9)
    axc.text(len(G_COLS) / 2 - 0.5, -0.16, "GENDER",
             transform=axc.get_xaxis_transform(), ha="center", fontsize=4.3,
             fontweight="bold")
    axc.text((len(G_COLS) + len(E_COLS)) / 2 - 0.5, -0.16, "EXPRESSION",
             transform=axc.get_xaxis_transform(), ha="center", fontsize=4.3,
             fontweight="bold")
    axc.set_title(" (c) Utility retention by dataset (rows: overall rank, SwinFace)",
                  fontsize=5.6, loc="left", fontweight="bold", pad=8)
    cbar = fig.colorbar(im, ax=axc, fraction=0.045, pad=0.10)
    cbar.ax.tick_params(labelsize=4.0)
    family_strip(axc, order)

    # (d) scorecard
    cols = [
        ("EER$_{11}$", lambda r, b: f'{r["backbones"][b]["eer_same"]*100:.0f}%',
         lambda r, b: (r["backbones"][b]["p1_eer_sc"] or np.nan), "swinface"),
        ("$R@1_{1N}$", lambda r, b: f'{r["backbones"][b]["r1_reid"]*100:.1f}%',
         lambda r, b: (r["backbones"][b]["p2_r1"] or np.nan), "swinface"),
        ("EER$_{xc}$", lambda r, b: f'{r["backbones"][b]["eer_xc"]*100:.0f}%',
         lambda r, b: (r["backbones"][b]["p3_eer_xc"] or np.nan), "swinface"),
        ("LPIPS", lambda r, b: f'{r["N"]["lpips"]:.2f}',
         lambda r, b: r["N"]["lpips_util"], "swinface"),
        ("IQA-ret", lambda r, b: f'{r["N"]["iqa_ret_raw"]:.2f}',
         lambda r, b: r["N"]["iqa_retention"], "swinface"),
        ("Overall$_S$", lambda r, b: f'{r["overall"]["swinface"]:.2f}',
         lambda r, b: r["overall"].get("swinface", np.nan), "swinface"),
        ("Overall$_T$", lambda r, b: f'{r["overall"]["transface"]:.2f}',
         lambda r, b: r["overall"].get("transface", np.nan), "transface"),
    ]
    Md = np.empty((len(order), len(cols)))
    for i, t in enumerate(order):
        for j, (_, _, ff, b) in enumerate(cols):
            v = ff(data[t], b)
            Md[i, j] = v if (v is not None and v == v) else np.nan
    imd = axd.imshow(Md, aspect="auto", cmap=CMAP, norm=Normalize(vmin=0, vmax=1))
    axd.set_xticks(range(len(cols)))
    axd.set_xticklabels([c[0] for c in cols], fontsize=4.2)
    axd.set_yticks([])
    for i, t in enumerate(order):
        for j, (name, raw, _, b) in enumerate(cols):
            v = Md[i, j]
            if not np.isnan(v):
                axd.text(j, i, raw(data[t], b), ha="center", va="center",
                         fontsize=3.6, color="w" if v > 0.5 else "k")
    axd.axvline(2.5, color="k", linewidth=0.7)
    axd.axvline(4.5, color="k", linewidth=0.7)
    axd.text(1, -0.16, "P components (SwinFace)", transform=axd.get_xaxis_transform(),
             ha="center", fontsize=4.3, fontweight="bold")
    axd.text(3.5, -0.16, "N components", transform=axd.get_xaxis_transform(),
             ha="center", fontsize=4.3, fontweight="bold")
    axd.text(5.5, -0.16, "Overall", transform=axd.get_xaxis_transform(),
             ha="center", fontsize=4.3, fontweight="bold")
    axd.set_title(" (d) Scorecard (raw values; cell color = normalized score)",
                  fontsize=5.6, loc="left", fontweight="bold", pad=8)
    dbar = fig.colorbar(imd, ax=axd, fraction=0.045, pad=0.10)
    dbar.ax.tick_params(labelsize=4.0)

    for ext in ("png", "pdf", "svg"):
        p = f"{out}.{ext}"
        fig.savefig(p, dpi=300 if ext == "png" else None)
        print("saved", p)


if __name__ == "__main__":
    main()
