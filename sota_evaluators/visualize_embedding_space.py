#!/usr/bin/env python3
"""
Embedding Space Projection — UMAP 2D density visualization of identity drift.

Projects aligned and anonymized embeddings into 2D using UMAP, rendering each
technique's embedding distribution as a 2D kernel-density heatmap (hexbin + KDE contour),
with overlay arrows showing per-image displacement vectors.

Uses ALL embeddings by default -- no sampling required.
For 2824 images, UMAP fitting takes ~15-30s; full pipeline ~2-4 minutes.

Usage:
    python visualize_embedding_space.py \\
        --aligned D:/dev/deid-toolkit/root_dir/embeddings/SwinFace/aligned/celeba-test \\
        --techniques-dir D:/dev/deid-toolkit/root_dir/embeddings/SwinFace/datasets \\
        --output embedding_projection.html
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np


def _savefig_safe(fig, path, dpi=200):
    """Save figure to PNG and PDF, with retry on locked files."""
    import time as _time
    for fmt in ("png", "pdf"):
        fpath = os.path.splitext(path)[0] + "." + fmt
        retries = 0
        while True:
            try:
                fig.savefig(fpath, dpi=dpi, bbox_inches="tight")
                break
            except PermissionError:
                retries += 1
                if retries > 5:
                    print(f"  WARN: could not save {fpath} (file locked)")
                    break
                _time.sleep(0.3)


def load_embeddings(folder):
    """Load .npy files from a folder. Returns {stem: array}."""
    p = Path(folder)
    if not p.is_dir():
        return {}
    embs = {}
    for npy in sorted(p.glob("*.npy")):
        embs[npy.stem] = np.load(npy)
    return embs


def _density_kde(ax, x, y, cmap_name, title="", xlabel="UMAP 1", ylabel="UMAP 2",
                 x_bounds=None, y_bounds=None, show_labels=True):
    """Render a continuous 2D density map via Gaussian KDE surface (imshow).

    If x_bounds/y_bounds are given, the KDE grid spans those fixed coordinates
    so the rendered image matches any enforced axis limits.
    """
    try:
        from scipy.stats import gaussian_kde
        if len(x) < 10:
            raise ValueError("too few points")
    except Exception:
        # Fallback to simple scatter if KDE impossible
        ax.scatter(x, y, s=1, alpha=0.15, c="#333")
        ax.set_title(title, fontsize=10, fontweight="bold") if title else None
        return

    grid = 160  # resolution of the density grid

    # Use fixed bounds for grid → extent matches enforced axis limits
    x_lo, x_hi = x_bounds if x_bounds else (x.min(), x.max())
    y_lo, y_hi = y_bounds if y_bounds else (y.min(), y.max())

    xs = np.linspace(x_lo, x_hi, grid)
    ys = np.linspace(y_lo, y_hi, grid)
    X, Y = np.meshgrid(xs, ys)
    # vstack gives shape (2, N) -- scipy gaussian_kde expects 2D data of form (dim, n_points)
    Z = gaussian_kde(np.vstack([x, y]), bw_method=0.3)(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)

    # Normalize to [0,1] for colormap
    if Z.max() > Z.min():
        Znorm = (Z - Z.min()) / (Z.max() - Z.min())
    else:
        Znorm = np.zeros_like(Z)

    ax.imshow(Znorm, extent=[x_lo, x_hi, y_lo, y_hi],
              origin="lower", cmap=cmap_name, aspect="equal", alpha=0.95)

    # Contour lines for shape definition
    try:
        cs = ax.contour(X, Y, Z, levels=10, colors="black", alpha=0.25, linewidths=0.5)
    except Exception:
        pass

    if title:
        ax.set_title(title, fontsize=10, fontweight="bold")
    if show_labels:
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)




def generate_plots(all_data, sample_names, techniques, output_path, clean=False,
                  use_cache=False, aligned_panel=False):
    """Generate UMAP projection density plots using ALL embeddings.

    Parameters
    ----------
    clean : bool
        If True, strip all ticks/labels/grids and tighten layout for manuscript plates.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    try:
        import umap
    except ImportError:
        print("\nERROR: umap-learn not installed. Install with: pip install umap-learn")
        sys.exit(1)

    print("[1/9] Matplotlib initialized, UMAP available")
    plt.rcParams.update({"font.size": 11, "axes.linewidth": 1.0})

    # ── Build name-to-embedding maps ─────────────────────────────────
    tech_order = [t for t in techniques]
    n_imgs = len(sample_names)
    print(f"[2/9] Built embeddings: {n_imgs} images × {len(tech_order)} technique(s)")

    name_to_emb = {}
    for tech, data_list in all_data.items():
        name_to_emb[tech] = {n: e for n, e in data_list}

    # Build (N_valid, 512) matrices + aligned-valid mask per technique
    # aligned_indices: indices into sample_names where this image exists in ALL techniques
    matrices = {}  # {tech: (mat[N,D], global_indices)}
    for tech in tech_order:
        emb_dict = name_to_emb.get(tech, {})
        valid_idx = []
        vecs = []
        for i, nm in enumerate(sample_names):
            if nm in emb_dict:
                valid_idx.append(i)
                vecs.append(emb_dict[nm])
        matrices[tech] = (np.array(vecs, dtype=np.float32), np.array(valid_idx, dtype=int))

    aligned_mat, aligned_indices = matrices["aligned"]

    # Mean per-image cosine drift d = 1 - cos(e(x), e(x_hat)) in full
    # embedding space, one value per technique (annotated per UMAP panel).
    MEAN_DRIFT = {}
    for tech in tech_order[1:]:
        mat_t, idxs = matrices[tech]
        if len(mat_t) == 0:
            continue
        A = aligned_mat[idxs].astype(np.float64)
        B = mat_t.astype(np.float64)
        A = A / np.linalg.norm(A, axis=1, keepdims=True)
        B = B / np.linalg.norm(B, axis=1, keepdims=True)
        MEAN_DRIFT[tech] = float(1.0 - np.sum(A * B, axis=1).mean())
    if len(aligned_indices) == 0:
        print("ERROR: No valid aligned embeddings."); sys.exit(1)

    n_aligned = len(aligned_indices)
    set_aligned = set(int(i) for i in aligned_indices)

    # ── UMAP fit on ALL aligned embeddings ────────────────────────────
    print(f"[3/9] Fitting UMAP on {n_aligned} aligned embeddings (n_neighbors=30)...")
    import time as _t0
    cache_path = Path(os.path.splitext(str(output_path))[0] + ".umap_cache.pkl")
    t_umap = _t0.time()
    cached_projs, cached_drift = None, None
    if use_cache and cache_path.exists():
        import pickle
        with open(cache_path, "rb") as _cf:
            _c = pickle.load(_cf)
        reducer, aligned_2d = _c["reducer"], _c["aligned_2d"]
        cached_projs = _c.get("projections")
        cached_drift = _c.get("mean_drift")
        if cached_drift:
            MEAN_DRIFT.update(cached_drift)
        print(f"      UMAP loaded from cache in {_t0.time()-t_umap:.1f}s ({cache_path.name}); "
              f"remove the .umap_cache.pkl to refit")
    else:
        reducer = umap.UMAP(n_components=2, random_state=42, metric="cosine",
                            n_neighbors=30, min_dist=0.1)
        aligned_2d = reducer.fit_transform(aligned_mat)  # (N_aligned, 2)
        print(f"      UMAP done in {_t0.time()-t_umap:.1f}s")
        import pickle
        with open(cache_path, "wb") as _cf:
            pickle.dump({"reducer": reducer, "aligned_2d": aligned_2d}, _cf)
        print(f"      UMAP cache saved: {cache_path.name}")

    # Global axis limits from the aligned embedding space (+ 10% margin)
    x_margin = (aligned_2d[:, 0].max() - aligned_2d[:, 0].min()) * 0.10 + 1e-6
    y_margin = (aligned_2d[:, 1].max() - aligned_2d[:, 1].min()) * 0.10 + 1e-6
    GLOBAL_XLIM = (aligned_2d[:, 0].min() - x_margin, aligned_2d[:, 0].max() + x_margin)
    GLOBAL_YLIM = (aligned_2d[:, 1].min() - y_margin, aligned_2d[:, 1].max() + y_margin)

    # ── Project each technique through fitted UMAP ────────────────────
    print(f"[4/9] Projecting {len(tech_order)-1} technique(s) through fitted UMAP...")
    t_proj = _t0.time()
    projections = {}  # {tech: (x_arr, y_arr)} -- x/y indexed by tech's valid points order
    if use_cache and cached_projs is not None:
        projections = cached_projs
        print(f"      Projections loaded from cache in {_t0.time()-t_proj:.1f}s "
              f"(skipping UMAP transform)")
    else:
        for tech in tech_order[1:]:
            mat_t, _ = matrices[tech]
            if len(mat_t) == 0:
                projections[tech] = (np.zeros((0,)), np.zeros((0,)))
                continue
            proj = reducer.transform(mat_t)
            projections[tech] = (proj[:, 0], proj[:, 1])
        # enrich the cache with the 2D projections (+ drift) so that later
        # figure tweaks with --use-cache skip the ~4-min UMAP transform
        import pickle
        with open(cache_path, "wb") as _cf:
            pickle.dump({"reducer": reducer, "aligned_2d": aligned_2d,
                         "projections": projections, "mean_drift": MEAN_DRIFT}, _cf)
        print(f"      Projection done in {_t0.time()-t_proj:.1f}s; projections "
              f"saved to cache")

    # Build a fast lookup: for each technique, map global image index -> projection position
    tech_proj_map = {}  # {tech: {global_idx: (px, py)}}
    for tech in tech_order[1:]:
        _, idxs = matrices[tech]
        px, py = projections.get(tech, (np.zeros((0,)), np.zeros((0,))))
        d = {}
        for p_idx in range(len(px)):
            global_i = int(idxs[p_idx])
            if global_i in set_aligned:  # only aligned positions matter
                d[global_i] = (px[p_idx], py[p_idx])
        tech_proj_map[tech] = d

    # ── Colors & colormaps ────────────────────────────────────────────
    # Single source of truth: the shared (color, marker) encoding from
    # fig_style (10 hue families; dark = circle / light = square in the
    # 4-panel figures). For the UMAP density ramps, each family reuses the
    # matplotlib built-in sequential map of its closest hue (the same class
    # of well-tuned Lightmap ramps the original v4 plate used: near-white
    # -> pale -> saturated -> deep core, with full coverage of the tails).
    # Gray ("Greys") stays reserved for the aligned Original.
    from matplotlib.lines import Line2D
    from fig_style import color_for, pair_for

    _FAMILY_MAP = {
        "#1f77b4": "Blues",    # blue
        "#ff7f0e": "Oranges",  # orange
        "#2ca02c": "Greens",   # green
        "#d62728": "Reds",     # red
        "#9467bd": "Purples",  # purple
        "#8c564b": "OrRd",     # brown
        "#e377c2": "RdPu",     # magenta
        "#c9a227": "YlOrBr",   # gold
        "#bcbd22": "YlGn",     # olive
        "#17becf": "GnBu",     # cyan
    }

    TECH_COLORS = {"aligned": "#777777"}
    TECH_CMAP = {"aligned": "Greys"}
    for i, t in enumerate(tech_order):
        if t == "aligned":
            continue
        TECH_COLORS[t] = color_for(t)
        try:
            dark, light = pair_for(t)
        except KeyError:
            continue
        TECH_CMAP[t] = _FAMILY_MAP.get(dark, "Greys")

    n_techs = len([t for t in tech_order if t != "aligned"])

    # Display name: show "Original" instead of "aligned" in titles / labels
    def display_name(tech):
        return "Original" if tech == "aligned" else tech

    # ===================================================================
    # Density panel renderer — shared code for all grid variants
    # ===================================================================
    from scipy.stats import gaussian_kde as _gkde_ref
    try:
        _REF_KDE = _gkde_ref(np.vstack([aligned_2d[:, 0], aligned_2d[:, 1]]),
                             bw_method=0.3)
    except Exception:
        _REF_KDE = None

    # Data-centred per-panel framing (same rule as the arrows plate): keep the
    # global span for a comparable scale, but center each panel window on the
    # midpoint of the aligned centroid and the technique's cloud centroid.
    AC_X = float(aligned_2d[:, 0].mean()); AC_Y = float(aligned_2d[:, 1].mean())
    GX_HALF = (GLOBAL_XLIM[1] - GLOBAL_XLIM[0]) / 2.0
    GY_HALF = (GLOBAL_YLIM[1] - GLOBAL_YLIM[0]) / 2.0

    def _panel_bounds(px, py):
        """Union-fit frame: fully contains BOTH the aligned cloud (global
        extent) and the technique cloud (dashed reference never clipped),
        centred on their combined centre, never smaller than the global span
        (comparable scale across panels; square window)."""
        u0x, u1x = GLOBAL_XLIM
        u0y, u1y = GLOBAL_YLIM
        if len(px):
            u0x = min(u0x, float(np.min(px)) - 0.5)
            u1x = max(u1x, float(np.max(px)) + 0.5)
            u0y = min(u0y, float(np.min(py)) - 0.5)
            u1y = max(u1y, float(np.max(py)) + 0.5)
        cx = 0.5 * (u0x + u1x)
        cy = 0.5 * (u0y + u1y)
        # square window at the global scale: same data scale on all panels
        # (still directly comparable) with a square frame
        s = max(GLOBAL_XLIM[1] - GLOBAL_XLIM[0], GLOBAL_YLIM[1] - GLOBAL_YLIM[0],
                u1x - u0x, u1y - u0y)
        return ((cx - s / 2.0, cx + s / 2.0), (cy - s / 2.0, cy + s / 2.0))

    def _ref_on_grid(bx, by):
        """Aligned-reference KDE contours re-gridded for a panel's bounds."""
        if _REF_KDE is None:
            return None, None, None
        xs = np.linspace(bx[0], bx[1], 150)
        ys = np.linspace(by[0], by[1], 150)
        X, Y = np.meshgrid(xs, ys)
        Z = _REF_KDE(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
        return X, Y, Z

    anon_techs_only = [t for t in tech_order if t != "aligned"]
    # Panel order: optionally the aligned distribution as its own first panel
    panel_order = (["aligned"] if aligned_panel else []) + anon_techs_only

    def _render_density_panels(fig, axes, nrows, ncols, title_text=None):
        """Render per-technique density panels into a pre-sized (nrows×ncols) figure.

        Returns the fig for further saving/closing by caller.
        """
        if nrows == 1 and ncols == 1:
            axes = np.array([[axes]])
        elif nrows == 1:
            axes = axes.reshape(1, -1)
        elif ncols == 1:
            axes = axes.reshape(-1, 1)

        flat_idx = 0
        for tech in panel_order:
            if flat_idx >= nrows * ncols:
                break
            row, col = divmod(flat_idx, ncols)
            ax = axes[row, col]

            if tech == "aligned":
                # Standalone aligned-distribution panel (reference)
                bx, by = _panel_bounds(aligned_2d[:, 0], aligned_2d[:, 1])
                _density_kde(ax, aligned_2d[:, 0], aligned_2d[:, 1], "Greys", title="",
                             x_bounds=bx,
                             y_bounds=by,
                             show_labels=not clean)
                ax.set_aspect("equal", adjustable="box")
                if clean:
                    ax.set_xticks([]); ax.set_yticks([])
                    for sp in ax.spines.values():
                        sp.set_visible(True)
                        sp.set_color("black")
                        sp.set_linewidth(1.0)
                else:
                    ax.grid(True, alpha=0.15)
                ax.text(0.02, 0.96, "Reference", transform=ax.transAxes,
                        fontsize=10, fontweight="bold", va="top", ha="left", color="#777777")
                # Filled dot = aligned centroid (the single reference marker
                # in the whole figure).
                ax.scatter([AC_X], [AC_Y], s=30, facecolor="black", edgecolor="k",
                           linewidth=0.9, zorder=6)
                flat_idx += 1
                continue

            px, py = projections.get(tech, (np.zeros((0,)), np.zeros((0,))))

            cmap_name = TECH_CMAP.get(tech, "Greys")
            bx, by = _panel_bounds(px, py)

            # Original dashed contour background (re-gridded for the panel)
            ref_X, ref_Y, Zref = _ref_on_grid(bx, by)
            if Zref is not None:
                ax.contour(ref_X, ref_Y, Zref, levels=10, colors="#555555", alpha=0.6,
                           linewidths=1.0, linestyles="--")

            _density_kde(ax, px, py, cmap_name, title="",
                         x_bounds=bx,
                         y_bounds=by,
                         show_labels=not clean)

            # Centroid: open circle = this technique's mean de-identified
            # centroid. The aligned reference centroid is shown exactly once
            # (filled dot in the Reference panel); no displacement arrow: the
            # 2D UMAP direction carries no intrinsic meaning.
            cx_t, cy_t = float(px.mean()), float(py.mean())
            ax.scatter([cx_t], [cy_t], s=30, facecolor="none", edgecolor="k",
                       linewidth=1.0, zorder=6)
            if tech in MEAN_DRIFT:
                ax.text(0.98, 0.03, f"mean drift: {MEAN_DRIFT[tech]:.2f}",
                        transform=ax.transAxes, ha="right", va="bottom",
                        fontsize=7.5, color="#222222", zorder=7,
                        bbox=dict(boxstyle="round,pad=0.25", fc="white",
                                  ec="none", alpha=0.7))

            ax.set_aspect("equal", adjustable="box")
            if clean:
                ax.set_xticks([])
                ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(True)
                    sp.set_color("black")
                    sp.set_linewidth(1.0)
            else:
                ax.grid(True, alpha=0.15)

            dname = display_name(tech)
            # title in the family's DARK shade (light shades are unreadable
            # on the pale end of the density ramp)
            try:
                color_for_label = pair_for(tech)[0]
            except KeyError:
                color_for_label = TECH_COLORS[tech]
            ax.text(0.02, 0.96, dname, transform=ax.transAxes, fontsize=10,
                    fontweight="bold", va="top", ha="left", color=color_for_label)
            flat_idx += 1

        # Hide unused subplots
        for r in range(nrows):
            for c in range(ncols):
                cell = r * ncols + c
                if cell >= len(panel_order):
                    axes[r, c].set_visible(False)

        if title_text:
            fig.suptitle(title_text, fontsize=14, fontweight="bold", y=1.01)
            fig.tight_layout(rect=[0, 0, 1, 0.96])
        else:
            # No bottom legend row: each panel's open centroid dot is
            # self-explanatory and the reference (filled) dot appears once,
            # in the Reference panel. Explicit margins (not tight_layout,
            # which over-reserves vertical space for tick areas that clean
            # mode removed) keep the rows packed edge-to-edge.
            if clean:
                fig.subplots_adjust(left=0.002, right=0.998, top=0.985,
                                    bottom=0.002, wspace=0.04, hspace=0.05)
            else:
                fig.tight_layout(pad=0.4, h_pad=0.5, w_pad=0.3)
        return fig

    # ===================================================================
    # Fig 1 -- Per-baseline density panels (4x4 default + optional 2x8 landscape + 8x2 portrait)
    # ===================================================================
    density_path = os.path.splitext(output_path)[0] + "_density.png"

    if aligned_panel:
        # 1 (aligned) + n_techs technique panels
        n_panels = 1 + n_techs
        density_layouts = [(7, 3), (3, 7)]   # portrait 7r×3c + landscape 3r×7c
        if n_panels > 21:
            density_layouts = [(4, 6), (6, 4)]
    elif n_techs <= 16:
        density_layouts = [(4, 4)]
        # Add fixed 2×8 (landscape) and 8×2 (portrait) variants when there are >9 techniques
        if n_techs > 9:
            density_layouts.append((2, 8))   # landscape: 2 rows × 8 cols
            density_layouts.append((8, 2))   # portrait:  8 rows × 2 cols
    else:
        # >16 techniques: full-grid landscape/portrait variants (no truncation)
        density_layouts = [(4, 5), (5, 4)]

    print(f"[5/9] Rendering density panels ({len(density_layouts)} layout variant(s))...")

    if clean:
        caption_path = os.path.splitext(density_path)[0] + "_caption.txt"
        caption = (
            f"Fig. X — Per-baseline UMAP embedding density maps (cosine metric, "
            f"n_neighbors=30). Each panel shows one of {n_techs} anonymization "
            f"techniques applied to the CelebA test set ({n_aligned} identities × 1 image each = "
            f"{n_aligned} embeddings). Original face embeddings serve as a faint dashed contour in "
            f"the background for reference. Technique name is labeled in the upper-left corner of "
            f"each panel (colors correspond to Fig. Y overlay). Darker shades indicate higher "
            f"embedding density, revealing how each DEID method perturbs identity information in "
            f"the UMAP 2D projection."
        )
        with open(caption_path, "w", encoding="utf-8") as _cf:
            _cf.write(caption)
        print(f"\nCaption: {caption_path}")

    for idx, (rows, cols) in enumerate(density_layouts):
        if clean:
            # Square cells (the data window is square): cell = 2.6" wide x
            # (2.55" high, i.e. rows pack edge-to-edge like the v4 plate) so
            # the aspect-forced square axes box fills each cell vertically.
            fig1, axes1 = plt.subplots(rows, cols, figsize=(2.6 * cols, 2.55 * rows))
        else:
            fig1, axes1 = plt.subplots(rows, cols, figsize=(3.4 * cols, 3.0 * rows))

        _render_density_panels(fig1, axes1, rows, cols)

        # Always suffix with layout dimensions when multiple variants exist
        if len(density_layouts) > 1:
            suffix = f"_{rows}x{cols}"
        else:
            suffix = ""

        if idx == 0:
            out_path = density_path   # default name for first variant
        else:
            out_path = os.path.splitext(density_path)[0] + suffix + ".png"

        _savefig_safe(fig1, out_path, dpi=200)
        plt.close(fig1)
        print(f"Density panels ({rows}x{cols}): {out_path}")

    # ===================================================================
    # Fig 1b -- All techniques overlaid as KDE contour lines (no fill)
    #   Filled hexbin with 16 × 2824 points is unreadable; instead use
    #   smooth KDE contours so each technique's shape is visible.
    # ===================================================================
    print("[6/9] Rendering overlay contour plot...")
    fig1b, ax1b = plt.subplots(figsize=(12, 9))

    from scipy.stats import gaussian_kde as _gkde

    def _contour_layer(ax, x, y, color, levels=6, lw=0.9, alpha_fill=0.015):
        """Draw a single KDE: filled (very transparent) + colored contour lines."""
        xs = np.linspace(x.min(), x.max(), 150)
        ys = np.linspace(y.min(), y.max(), 150)
        X, Y = np.meshgrid(xs, ys)
        Z = _gkde(np.vstack([x, y]), bw_method=0.3)(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
        # Very light fill so many layers don't opaque up
        ax.contourf(X, Y, Z, levels=20, colors=color, alpha=alpha_fill)
        # Solid contour lines for shape visibility
        cs = ax.contour(X, Y, Z, levels=levels, colors=color, alpha=0.75, linewidths=lw)

    # Background aligned (gray, light fill + thin contours)
    _contour_layer(ax1b, aligned_2d[:, 0], aligned_2d[:, 1], "#999999",
                   levels=3, lw=0.6, alpha_fill=0.04)

    anon_techs = [t for t in tech_order if t != "aligned"]
    for idx, tech in enumerate(anon_techs):
        px, py = projections.get(tech, (np.zeros((0,)), np.zeros((0,))))
        if len(px) > 0:
            _contour_layer(ax1b, px, py, TECH_COLORS[tech])

    legend_patches = [Patch(color=color_for(t), alpha=0.3, label=t)
                      for t in anon_techs]
    # Add aligned patch
    legend_patches.insert(0, Patch(color="#999999", alpha=0.3, label="Original"))
    ax1b.legend(handles=legend_patches, loc="best", fontsize=8.5, ncol=4, framealpha=0.9)
    ax1b.set_title("All Techniques -- KDE Contour Overlay", fontsize=14, fontweight="bold")
    ax1b.set_xlabel("UMAP 1", fontsize=12)
    ax1b.set_ylabel("UMAP 2", fontsize=12)
    ax1b.set_aspect("equal", adjustable="box")

    fig1b.tight_layout()
    overlay_path = os.path.splitext(output_path)[0] + "_overlay.png"
    _savefig_safe(fig1b, overlay_path, dpi=200)
    plt.close(fig1b)
    print(f"Overlay (contour-only): {overlay_path}")

    # ===================================================================
    # Fig 2 -- Drift arrows (4 techniques shown; Original as background)
    # ===================================================================
    print("[7/9] Rendering drift vector plot...")
    rng_a = np.random.RandomState(42)
    max_arrows = min(80, n_aligned // 8)
    draw_indices = rng_a.choice(n_aligned, size=max_arrows, replace=False)

    anon_techs_list = [t for t in tech_order if t != "aligned"]

    fig2, axes2 = plt.subplots(1, min(n_techs, 4), figsize=(3.8 * min(n_techs, 4), 4.0))
    if n_techs == 1:
        axes2 = np.array([axes2])

    for i, tech in enumerate(anon_techs_list[:4]):
        ax = axes2[i]

        # Original contour in background (per-panel, data-centred bounds)
        px, py = projections.get(tech, (np.zeros((0,)), np.zeros((0,))))
        bx, by = _panel_bounds(px, py)
        ref_X, ref_Y, Zref = _ref_on_grid(bx, by)
        if Zref is not None:
            ax.contour(ref_X, ref_Y, Zref, levels=3,
                       colors="#aaaaaa", alpha=0.25, linewidths=0.7, linestyles="--")

        # Technique density + drift arrows
        if len(px) > 0:
            cmap_name = TECH_CMAP.get(tech, "Greys")
            _density_kde(ax, px, py, cmap_name, title="", x_bounds=bx, y_bounds=by)

        # Build global_idx -> projection mapping for this technique
        tech_proj_map_local = {}
        _, idxs = matrices[tech]
        for p_idx in range(len(px)):
            global_i = int(idxs[p_idx])
            if global_i in set_aligned:
                tech_proj_map_local[global_i] = (px[p_idx], py[p_idx])

        # Draw arrows from aligned -> anonymized positions
        for ai in draw_indices:
            global_i = int(aligned_indices[ai])
            proj_xy = tech_proj_map_local.get(global_i)
            if proj_xy is not None:
                ax.arrow(aligned_2d[ai, 0], aligned_2d[ai, 1],
                         proj_xy[0] - aligned_2d[ai, 0],
                         proj_xy[1] - aligned_2d[ai, 1],
                         head_width=0, length_includes_head=True,
                         color=TECH_COLORS[tech], alpha=0.25, linewidth=0.6)

        ax.set_xlim(*GLOBAL_XLIM)
        ax.set_ylim(*GLOBAL_YLIM)
        dname = display_name(tech)
        ax.text(0.02, 0.97, dname + " -- Drift", transform=ax.transAxes, fontsize=10,
                fontweight="bold", va="top", ha="left", color=TECH_COLORS[tech])
        ax.set_xlabel("UMAP 1" if i == 0 else "")
        ax.set_ylabel("UMAP 2" if i == 0 else "")
        ax.grid(True, alpha=0.1)
        ax.set_aspect("equal", adjustable="box")

    fig2.suptitle("Drift Vectors -- Density with Per-Image Displacement (sampled)",
                  fontsize=13, fontweight="bold", y=1.01)
    fig2.tight_layout(rect=[0, 0, 1, 0.96])

    arrows_path = os.path.splitext(output_path)[0] + "_arrows.png"
    _savefig_safe(fig2, arrows_path, dpi=200)
    plt.close(fig2)
    print(f"Drift plot:   {arrows_path}")

    # ===================================================================
    # Fig 3 -- Compactness bar chart (dispersion from centroid in UMAP 2D)
    # ===================================================================
    print("[8/9] Rendering compactness bar chart...")
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    compactness = []
    for tech in tech_order:
        _, idxs = matrices[tech]
        if tech == "aligned" and n_aligned > 0:
            centroid = aligned_2d.mean(axis=0)
            dists = np.linalg.norm(aligned_2d - centroid, axis=1)
        else:
            px, py = projections.get(tech, (np.zeros((0,)), np.zeros((0,))))
            if len(px) < 2:
                continue
            centroid_c = np.mean([px, py], axis=1)
            dists = np.sqrt((px - centroid_c[0])**2 + (py - centroid_c[1])**2)
        compactness.append({
            "name": tech,
            "mean_d": float(np.mean(dists)),
            "std_d":  float(np.std(dists)),
            "max_d":  float(np.max(dists)),
        })

    x_pos = np.arange(len(compactness))
    means_c = [c["mean_d"] for c in compactness]
    stds_c  = [c["std_d"] for c in compactness]
    bars_colors = [TECH_COLORS[c["name"]] for c in compactness]

    ax3.bar(x_pos, means_c, yerr=stds_c, capsize=3, color=bars_colors, alpha=0.7,
            edgecolor="#333", linewidth=0.5)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([display_name(c["name"]) for c in compactness], rotation=45, ha="right", fontsize=9)
    ax3.set_ylabel("Mean Distance from Centroid (UMAP 2D)", fontsize=12)
    ax3.set_title("Embedding Compactness -- Identity Dispersion After DEID",
                  fontsize=13, fontweight="bold")
    ax3.grid(True, axis="y", alpha=0.2)

    fig3.tight_layout()
    if clean:
        ax3.set_ylabel("")
        ax3.set_title("Compactness -- Dispersion After DEID",
                      fontsize=12, fontweight="bold")
        ax3.yaxis.set_visible(False)
    else:
        ax3.set_ylabel("Mean Distance from Centroid (UMAP 2D)", fontsize=12)
        ax3.grid(True, axis="y", alpha=0.2)

    fig3.tight_layout()
    compact_path = os.path.splitext(output_path)[0] + "_compactness.png"
    _savefig_safe(fig3, compact_path, dpi=200)
    plt.close(fig3)
    print(f"Compactness:  {compact_path}")

    # ===================================================================
    # HTML report
    # ===================================================================
    html = (f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Embedding Space Projection</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
       max-width: 1400px; margin: 2em auto; padding: 0 1em; }}
h1 {{ border-bottom: 2px solid #333; padding-bottom: 0.5em; }}
img {{ max-width: 100%; height: auto; border: 1px solid #eee; border-radius: 4px; margin: 1em 0; }}
.note {{ color: #666; font-style: italic; }}
table {{ border-collapse: collapse; width: 50%; margin: 1em 0; }}
th, td {{ border: 1px solid #ddd; padding: 6px 10px; text-align: right; }}
th {{ background: #f5f5f5; text-align: center; }}
</style></head><body>
<h1>Embedding Space Projection -- UMAP 2D Density Maps</h1>
<p class="note">UMAP (cosine, n_neighbors=30) projects {n_aligned} embeddings into 2D.
Each panel shows a continuous Gaussian KDE density map: darker regions = higher embedding concentration.
Original is embedded as a faint dashed contour in every panel for comparison.</p>

<h2>A. Per-Baseline Density (4x4 grid, Original contour overlaid)</h2>
<img src="{os.path.basename(density_path)}" alt="Density subplots">

<h2>B. All Techniques Overlaid</h2>
<p>KDE contour overlay in a single UMAP space. Different spread shapes reveal distinct identity perturbation strategies.</p>
<img src="{os.path.basename(overlay_path)}" alt="Overlay density">

<h2>C. Drift Vectors -- Density with Per-Image Displacement Arrows</h2>
<p>Each arrow connects an original face's UMAP position to its anonymized counterpart (sampled subset for clarity).</p>
<img src="{os.path.basename(arrows_path)}" alt="Drift vectors">

<h2>D. Compactness -- Dispersion After DEID</h2>
<img src="{os.path.basename(compact_path)}" alt="Compactness">

<table><tr><th style='text-align:left'>Technique</th><th>Mean Distance</th><th>Std Dev</th><th>Max</th></tr>""")

    for c in compactness:
        dname = display_name(c["name"])
        html += (f"<tr><td style='text-align:left'>{dname}</td>"
                 f"<td>{c['mean_d']:.4f}</td><td>{c['std_d']:.4f}</td>"
                 f"<td>{c['max_d']:.4f}</td></tr>\n")

    html += ("""</table>
<p class="note"><strong>Interpretation:</strong> Wide spread = the technique preserves inter-person variation in embedding space.
Tight clusters (low compactness) indicate identity collapse -- distinct faces converge to similar representations, which means
the DEID method erases all individuality along with identity information. The Original distribution provides the baseline spread.</p>
</body></html>""")

    # ===================================================================
    print("[9/9] Writing HTML report...")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\nHTML report: {output_path}")
    print("\nDone!")


def main():
    parser = argparse.ArgumentParser(description="UMAP embedding space visualization")
    parser.add_argument("--aligned", type=str, required=True,
                        help="Path to aligned embeddings celeba-test folder")
    parser.add_argument("--techniques-dir", type=str, required=True,
                        help="Path to datasets root with technique/celeba-test folders")
    parser.add_argument("--output", type=str, default="embedding_projection.html",
                        help="Output HTML report path")
    parser.add_argument("--use-cache", action="store_true", default=False,
                        help="Reuse a cached fitted UMAP (faster figures-only re-render; "
                             "coordinates identical since random_state is fixed)")
    parser.add_argument("--aligned-panel", action="store_true", default=False,
                        help="Render the aligned distribution as its own first panel "
                             "(grid becomes 7x3 + 3x7)")
    parser.add_argument("--clean", action="store_true", default=False,
                        help="Tightest layout: no ticks/labels/grid, minimal spacing (for manuscript plates)")

    args = parser.parse_args()

    print("=" * 60)
    print("Embedding Space Projection -- UMAP 2D (all embeddings)")
    print("=" * 60)

    # Load aligned
    print(f"Loading aligned: {args.aligned}")
    aligned = load_embeddings(args.aligned)
    if not aligned:
        print("ERROR: No aligned embeddings found."); sys.exit(1)
    print(f"  Loaded {len(aligned)} embedding(s)")

    sample_names = sorted(aligned.keys())
    print(f"  Using ALL {len(sample_names)} images (no sampling)")

    # Load techniques
    tech_dir = Path(args.techniques_dir)
    if not tech_dir.is_dir():
        print("ERROR: Techniques directory not found."); sys.exit(1)

    all_data = {}
    techniques = []

    aligned_sampled = [(n, aligned[n]) for n in sample_names if n in aligned]
    all_data["aligned"] = aligned_sampled
    techniques.append("aligned")

    tech_folders = sorted([d for d in tech_dir.iterdir()
                           if d.is_dir() and not d.name.endswith("_reversed")],
                          key=lambda x: x.name)

    for tech_folder in tech_folders:
        celeba_path = tech_folder / "celeba-test"
        if not celeba_path.is_dir():
            continue

        name = tech_folder.name
        print(f"\nLoading {name}...")
        anon = load_embeddings(celeba_path)
        matched = [(n, anon[n]) for n in sample_names if n in anon]
        all_data[name] = matched
        techniques.append(name)
        print(f"  Loaded {len(matched)} matching embedding(s)")

    if len(techniques) < 2:
        print("ERROR: Need at least aligned + one technique."); sys.exit(1)

    generate_plots(all_data, sample_names, techniques, args.output, clean=args.clean,
                  use_cache=args.use_cache, aligned_panel=args.aligned_panel)


if __name__ == "__main__":
    main()
