#!/rusr/bin/env python3
"""
Copy candidates for Jernej #12 (direction of displacement), v2 batch.

A. COMPASS (swinface): one radial tick per technique from the center along
   its mean 2-D drift direction; radial length = tanh(mean 512-D cosine drift
   / D_SAT); light wedge = angular dispersion (circular std of per-image
   directions) -- short+faint ticks with wide wedges = isotropic scatter,
   long+directional ticks = directed migration.

B. DENSITY+ARROWS (transface, 4x4): exact reproduction of the manuscript's
   embedding_projection_transface_density style (KDE imshow + black 6-level
   contours, 12-level gray dashed aligned background, clean, colored name
   labels), PLUS one white mean-displacement arrow per panel (aligned 2-D
   centroid -> technique 2-D centroid, dark under-outline for contrast).

COPIES ONLY: output names carry _v1; nothing existing is overwritten.

Usage:
  python drift_direction_v2.py --probe swinface     # compass
  python drift_direction_v2.py --probe transface    # 4x4 density + white arrows
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

ROOT = Path("D:/dev/deid-toolkit/root_dir/embeddings")
EVAL = Path("D:/dev/deid-toolkit_evaluators")
MANU_FIGS = Path(
    "D:/dev/deid-toolkit_additional-materials/"
    "deid-toolkit_manuscript_Neurocomputing_Software_track/figs")

D_SAT = 0.6
SEED = 42

PALETTE = [
    "#e74c3c", "#2980b9", "#27ae60", "#8e44ad", "#d35400",
    "#16a085", "#c0392b", "#2c3e50", "#f39c12", "#1abc9c",
    "#e84393", "#6c5ce7", "#fdcb6e", "#00b894", "#d63031",
]
HEXMAP = {
    "#777777": "Greys",  "#e74c3c": "Reds",   "#2980b9": "Blues",
    "#27ae60": "Greens", "#8e44ad": "PuRd",    "#d35400": "Oranges",
    "#16a085": "GnBu",   "#c0392b": "Reds",   "#2c3e50": "Greys",
    "#f39c12": "YlOrBr", "#1abc9c": "turbo",  "#e84393": "Purples",
    "#6c5ce7": "BuPu",   "#fdcb6e": "YlGn",   "#00b894": "Greens",
    "#d63031": "Reds",   "#17becf": "Blues",
}


def load_dir(folder):
    out = {}
    for p in sorted(Path(folder).glob("*.npy")):
        out[p.stem] = np.load(p)
    return out


def setup(probe):
    """Load aligned+technique embeddings for probe; fit Fig-3 UMAP space."""
    e = ROOT / probe
    aligned = load_dir(e / "aligned" / "celeba-test")
    stems = sorted(aligned.keys())
    A = np.stack([aligned[s] for s in stems])
    import umap
    reducer = umap.UMAP(n_components=2, random_state=SEED, metric="cosine",
                        n_neighbors=30, min_dist=0.1)
    aligned_2d = reducer.fit_transform(A)

    techs = sorted(d.name for d in (e / "datasets").iterdir()
                   if d.is_dir() and d.name != "celeba-test")
    colors = {}
    for i, t in enumerate(techs):
        # original figure's mapping: aligned occupied palette slot 0,
        # so technique i (alphabetical, 0-based) falls on PALETTE[(i+1) % 15]
        colors[t] = PALETTE[(i + 1) % len(PALETTE)]
    colors["FALCO"] = "#17becf"  # explicit override (same as original run)

    tech = {}
    for t in techs:
        folder = e / "datasets" / t / "celeba-test"
        if not folder.is_dir():
            continue
        deid = load_dir(folder)
        common = [s for s in stems if s in deid]
        if not common:
            continue
        D = np.stack([deid[s] for s in common])
        idx = [stems.index(s) for s in common]
        A_c = A[idx]
        D2 = reducer.transform(D)
        v2 = D2 - aligned_2d[idx]               # per-image 2-D drift vectors
        d = 1.0 - np.clip((A_c * D).sum(1) /
                          (np.linalg.norm(A_c, axis=1) *
                           np.linalg.norm(D, axis=1) + 1e-12), -1, 1)
        tech[t] = dict(d2=D2, v2=v2, d=d, color=colors[t])
        print(f"    {t}: {len(common)}")
    return A, stems, aligned_2d, tech, colors


def circular_stats(v2):
    ang = np.arctan2(v2[:, 1], v2[:, 0])
    mean_ang = np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())
    R1 = np.sqrt(np.cos(ang).mean()**2 + np.sin(ang).mean()**2)
    circ_std = np.sqrt(-2.0 * np.log(max(R1, 1e-12)))
    return mean_ang, circ_std


def compass(probe):
    A, stems, aligned_2d, tech, colors = setup(probe)
    cx, cy = aligned_2d[:, 0].mean(), aligned_2d[:, 1].mean()

    D_S = 0.8  # shared tanh scale for BOTH rings and arrows
    RMAX = 0.95
    def _r(dmean):
        return RMAX * float(np.tanh(dmean / D_S))
    fig = plt.figure(figsize=(8.8, 5.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.0, 1.0],
                          left=0.02, right=0.98, top=0.80, bottom=0.05)
    ax = fig.add_subplot(gs[0, 0])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.12, 1.18)
    ax.axis("off")

    # ring grid + ring labels (staggered angles, no crowding)
    for m in (0.3, 0.5, 0.75, 1.0, 1.5):
        ax.add_patch(plt.Circle((0, 0), _r(m), fill=False,
                               ls="--", color="#cccccc", lw=0.7, alpha=0.9))
    for lab, deg in [("0.3", 80), ("0.5", 35), ("0.75", 215),
                     ("1.0", 300), ("1.5", 150)]:
        rr = _r(float(lab))
        th = np.deg2rad(deg)
        ax.text(rr * np.cos(th) + 0.008, rr * np.sin(th) + 0.035, lab,
                fontsize=7, color="#999999", ha="center")
    ax.text(0, 1.10, "512-D mean cosine drift (ring scale)", fontsize=7.5,
            color="#999999", ha="center")

    # technique arrows (center -> mean drift direction, tanh-capped length)
    for t, td in tech.items():
        mean_ang, circ_std = circular_stats(td["v2"])
        L = _r(td["d"].mean())
        x0, y0 = 0.045 * np.cos(mean_ang), 0.045 * np.sin(mean_ang)
        x1, y1 = L * np.cos(mean_ang), L * np.sin(mean_ang)
        col = td["color"]
        # dispersion: two hairline sector borders (no fill)
        w = min(circ_std, np.pi)
        ax.plot([x0, L * np.cos(mean_ang - w)],
                [y0, L * np.sin(mean_ang - w)],
                color=col, alpha=0.30, lw=0.7, zorder=2)
        ax.plot([x0, L * np.cos(mean_ang + w)],
                [y0, L * np.sin(mean_ang + w)],
                color=col, alpha=0.30, lw=0.7, zorder=2)
        # dark under-outline arrow + main colored arrow
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color="#1a1a1a",
                                    lw=3.6, alpha=0.55, mutation_scale=15,
                                    shrinkA=0, shrinkB=0), zorder=3)
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=1.9,
                                    alpha=0.95, mutation_scale=12,
                                    shrinkA=0, shrinkB=0), zorder=4)

    ax.plot(0, 0, "o", ms=4, c="#555555", zorder=5)

    handles = [plt.Line2D([], [], color=td["color"], lw=2.5)
               for td in tech.values()]
    axleg = fig.add_subplot(gs[0, 1]); axleg.axis("off")
    axleg.legend(handles, list(tech.keys()), loc="center left",
                 fontsize=7.5, frameon=False, handlelength=1.6)
    fig.suptitle("Mean drift direction per technique (2-D projection) -- "
                 f"{probe}, CelebA-test. Arrow length: 512-D mean cosine drift "
                 "(ring scale 0--1.0+); hairlines: angular dispersion of per-image directions",
                 fontsize=9.5, fontweight="bold", y=0.97)

    out = EVAL / f"drift_compass_{probe}_v3"
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    for suf in (".png", ".pdf"):
        (MANU_FIGS / f"drift_compass_{probe}_v3{suf}").write_bytes(
            (out.with_suffix(suf)).read_bytes())
    print(f"Compass v3 saved: {out}.png/.pdf (+ manuscript figs/ copy)")


def density_arrows(probe):
    A, stems, aligned_2d, tech, colors = setup(probe)

    # Global fixed bounds like the original 4x4 (aligned cloud + margin,
    # same for all panels)
    xlo, xhi = aligned_2d[:, 0].min() - 0.5, aligned_2d[:, 0].max() + 0.5
    ylo, yhi = aligned_2d[:, 1].min() - 0.5, aligned_2d[:, 1].max() + 0.5

    fig, axes = plt.subplots(4, 4, figsize=(2.6 * 4, 2.4 * 4))
    gx = np.linspace(xlo, xhi, 160)
    gy = np.linspace(ylo, yhi, 160)
    GX, GY = np.meshgrid(gx, gy)
    Zref = gaussian_kde(np.vstack([aligned_2d[:, 0], aligned_2d[:, 1]]),
                        bw_method=0.3)(
        np.vstack([GX.ravel(), GY.ravel()])).reshape(GX.shape)

    acx, acy = aligned_2d[:, 0].mean(), aligned_2d[:, 1].mean()
    extent = (xlo, xhi, ylo, yhi)

    flat = 0
    for t, td in tech.items():
        ax = axes[(flat // 4), flat % 4]
        px, py = td["d2"][:, 0], td["d2"][:, 1]
        Zt = gaussian_kde(np.vstack([px, py]), bw_method=0.3)(
            np.vstack([GX.ravel(), GY.ravel()])).reshape(GX.shape)
        Znorm = (Zt - Zt.min()) / (Zt.max() - Zt.min() + 1e-12)
        cmap = HEXMAP.get(td["color"], "Greys")
        if cmap == "GnBu":
            cmap = "Greens"  # GnBu renders two hues (green->blue); keep mono
        ax.imshow(Znorm, extent=extent, origin="lower", cmap=cmap,
                  aspect="equal", alpha=0.95)
        # black frame per panel, no ticks, no UMAP axis labels
        ax.tick_params(left=False, bottom=False, labelleft=False,
                       labelbottom=False)
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_color("black")
            sp.set_linewidth(1.0)
        try:
            ax.contour(GX, GY, Zt, levels=6, colors="black", alpha=0.25,
                       linewidths=0.5)
        except Exception:
            pass
        ax.contour(GX, GY, Zref, levels=12, colors="#555555", alpha=0.6,
                   linewidths=1.0, linestyles="--")
        ax.text(0.02, 0.96, t, transform=ax.transAxes, fontsize=10,
                fontweight="bold", va="top", ha="left", color=td["color"])

        # white mean-displacement arrow:
        #   direction = true mean displacement vector (de-id centroid - aligned centroid)
        #   length    = tanh(mean per-image 512-D drift / 0.5) * 0.35 * span (common scale)
        # open circle marks the de-identified centroid's true (unscaled) position.
        tcx, tcy = px.mean(), py.mean()
        span = max(xhi - xlo, yhi - ylo)
        dmean = td["d"].mean()
        dx, dy = tcx - acx, tcy - acy
        nrm = (dx**2 + dy**2) ** 0.5
        if nrm > 1e-6:
            ux, uy = dx / nrm, dy / nrm
            L = 0.35 * span * float(np.tanh(dmean / 0.5))
            head_w = 0.030 * span
            ax.arrow(acx, acy, ux * L, uy * L, head_width=head_w * 1.6,
                     head_length=head_w * 1.35, length_includes_head=True,
                     color="#1a1a1a", alpha=0.70, lw=3.6, zorder=6)
            ax.arrow(acx, acy, ux * L, uy * L, head_width=head_w,
                     head_length=head_w * 1.05, length_includes_head=True,
                     color="white", alpha=0.97, lw=2.0, zorder=7)
        # true de-identified centroid: dark under-ring + white ring
        ax.add_patch(plt.Circle((tcx, tcy), 0.011 * span, fill=False,
                                ec="#1a1a1a", lw=2.6, alpha=0.70, zorder=8))
        ax.add_patch(plt.Circle((tcx, tcy), 0.011 * span, fill=False,
                                ec="white", lw=1.3, zorder=9))
        ax.scatter([acx], [acy], s=9, c="#1a1a1a", zorder=6)
        # raw mean drift value (512-D cosine), lower-right, white backing for
        # readability on dark panels
        ax.text(0.965, 0.045, f"mean drift: {dmean:.2f}",
                transform=ax.transAxes, fontsize=8.5, color="#222222",
                ha="right", va="bottom", zorder=10,
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="none",
                          pad=1.5))
        flat += 1

    fig.tight_layout()
    base = f"embedding_projection_{probe}_density_arrows_v3"
    out = EVAL / base
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    for suf in (".png", ".pdf"):
        (MANU_FIGS / f"{base}{suf}").write_bytes(
            (out.with_suffix(suf)).read_bytes())
    print(f"Density+arrows saved: {out}.png/.pdf (+ manuscript figs/ copy)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", required=True, choices=["swinface", "transface"])
    ap.add_argument("--what", required=True, choices=["compass", "density"])
    a = ap.parse_args()
    if a.what == "compass":
        compass(a.probe)
    else:
        density_arrows(a.probe)
