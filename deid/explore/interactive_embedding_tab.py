"""Streamlit tab: Interactive Embedding Space Viewer.

Reads pre-computed CSV data files from embedding_viz_cli and renders interactive Plotly charts
with hover tooltips, filtering, and technique switching. Also exports vector-quality PDFs for
manuscripts via matplotlib (same rendering as CLI).

Requires: plotly (pip install plotly)
"""
from __future__ import annotations

import glob
import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from deid.explore.data_loader import get_loader


def _export_pdf(fig) -> bytes:
    """Export a matplotlib Figure to PDF bytes."""
    buf = io.BytesIO()
    fig.savefig(buf, format="pdf", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


def render() -> None:
    st.title("Interactive Embedding Viewer")
    st.caption(
        "Browse pre-computed embedding analysis results. "
        "Run the CLI first to generate CSVs:\n"
        "`python -m deid.explore.embedding_viz_cli --dataset X --techniques Y --model swinface`"
    )

    root_dir = get_loader().settings.root_dir
    viz_dir = Path(root_dir) / "results" / "viz"

    # Discover available CSV data files
    displacement_csvs = sorted(glob.glob(str(viz_dir / "*_swinface/displacement_*_data.csv")))
    if not displacement_csvs:
        # Try broader search
        displacement_csvs = sorted(glob.glob(str(viz_dir / "**/displacement_*_data.csv"), recursive=True))

    comparison_csvs = sorted(glob.glob(str(viz_dir / "**/comparison_*_data.csv"), recursive=True))
    collapse_csvs = sorted(glob.glob(str(viz_dir / "**/collapse_*_metrics.csv"), recursive=True))

    available = []
    for c in displacement_csvs:
        available.append(("Displacement", Path(c)))
    for c in comparison_csvs:
        available.append(("Comparison", Path(c)))
    for c in collapse_csvs:
        available.append(("Collapse Metrics", Path(c)))

    if not available:
        st.warning(
            "No visualization data files found.\n\n"
            "Generate them with:\n"
            "`python -m deid.explore.embedding_viz_cli --dataset celeba-test_aligned --model swinface --techniques blur pixelize`"
        )
        return

    # Sidebar: select a CSV file
    st.sidebar.header("Data Source")
    file_labels = [f"{t}: {p.name}" for t, p in available]
    selected_idx = st.sidebar.selectbox("CSV File", range(len(file_labels)), format_func=lambda i: file_labels[i])
    _, selected_path = available[selected_idx]

    # Load and parse
    df = pd.read_csv(selected_path)
    st.sidebar.markdown(f"**Rows:** {len(df):,}  |  **Columns:** {len(df.columns)}")

    tab_type, _ = available[selected_idx]

    if tab_type == "Displacement":
        _render_displacement(df, selected_path.parent)
    elif tab_type == "Comparison":
        _render_comparison(df)
    elif tab_type == "Collapse Metrics":
        _render_collapse_table(df)


def _render_displacement(df: pd.DataFrame, viz_dir: Path):
    """Interactive displacement plot with Plotly scatter + hover."""
    st.subheader("Per-Image Displacement")

    try:
        import plotly.express as px
    except ImportError:
        st.error("Plotly not installed. Run: pip install plotly")
        return

    # Determine available color columns
    color_options = {}
    for col in ["cosine_similarity", "euclidean_distance", "disp_2d"]:
        if col in df.columns:
            label_map = {
                "cosine_similarity": "Cosine Similarity",
                "euclidean_distance": "Euclidean Distance",
                "disp_2d": "2D Displacement",
            }
            color_options[label_map[col]] = col

    if not color_options:
        st.warning("No color column found in data.")
        return

    default_color = list(color_options.keys())[0]
    color_col_name = st.selectbox("Color by", list(color_options.keys()), index=0)
    color_col = color_options[color_col_name]

    # Identity filter
    identity_filter = None
    if "identity" in df.columns and df["identity"].notna().any():
        unique_ids = sorted(df["identity"].dropna().unique())
        if len(unique_ids) > 1:
            identity_filter = st.multiselect(
                "Filter by Identity (select any to show only those)",
                options=unique_ids[:50],  # limit for performance
                default=[],
                help=f"Showing first 50 of {len(unique_ids)} identities. Use text search below."
            )

    if identity_filter:
        df = df[df["identity"].isin(identity_filter)]

    # Text search
    search = st.text_input("Search image filename", "")
    if search:
        df = df[df["image"].str.contains(search, case=False, na=False)]

    # Arrow subsampling for performance (Plotly can handle ~1000 points but 2000+ is slow)
    max_points = 800
    if len(df) > max_points:
        st.caption(f"Subsampled to {max_points} points for interactivity (N={len(df)})")
        df_plot = df.sample(n=max_points, random_state=42)
    else:
        df_plot = df

    # Determine reverse color for cosine similarity (higher = less perturbed)
    reverse_color = color_col == "cosine_similarity"

    fig = px.scatter(
        df_plot,
        x="orig_x", y="orig_y",
        hover_data={
            "image": True,
            "identity": True,
            "cosine_similarity": True if "cosine_similarity" in df_plot.columns else False,
            "euclidean_distance": True if "euclidean_distance" in df_plot.columns else False,
            "disp_2d": True if "disp_2d" in df_plot.columns else False,
        },
        color=color_col,
        color_continuous_scale="RdYlBu_r" if reverse_color else "Viridis",
        opacity=0.75,
        title=f"Displacement — Hover for details  |  Points={len(df_plot)}",
    )

    # Add arrows (only for subsampled set or small N)
    if len(df_plot) <= 500:
        # Compute displacement vectors
        dx = df_plot["deid_x"] - df_plot["orig_x"]
        dy = df_plot["deid_y"] - df_plot["orig_y"]

        fig.add_trace(px.scatter(
            df_plot, x="orig_x", y="orig_y",
            custom_data=[dx, dy],
            color=color_col,
            color_continuous_scale=fig.data[0].colorscale,
            hoverinfo="skip",
            mode="markers+text",
            text=["→"] * len(df_plot),
            text_position="bottom right",
        ).data[-1])

    fig.update_layout(
        xaxis_title="Component 1",
        yaxis_title="Component 2",
        hovermode="closest",
        height=600,
    )

    st.plotly_chart(fig, use_container_width=True)

    # Stats summary
    st.markdown("### Statistics")
    cols = st.columns(4)
    if "cosine_similarity" in df.columns:
        cols[0].metric("Mean Cosine Sim", f"{df['cosine_similarity'].mean():.3f}")
        cols[1].metric("Std", f"{df['cosine_similarity'].std():.3f}")
    if "euclidean_distance" in df.columns:
        cols[2].metric("Mean Eucl Dist", f"{df['euclidean_distance'].mean():.2f}")
    cols[3].metric("Total Images", len(df))

    # ── Export ───────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Export for Manuscript")
    export_cols = st.columns(3)

    with export_cols[0]:
        pdf_bytes = _generate_displacement_pdf(
            orig_xy=df[["orig_x", "orig_y"]].to_numpy(),
            deid_xy=df[["deid_x", "deid_y"]].to_numpy(),
            magnitudes=df.get("euclidean_distance", pd.Series()).to_numpy() if "euclidean_distance" in df.columns else df.get("disp_2d", pd.Series()).to_numpy(),
            dataset=selected_path.stem,
            technique="interactive",
            projection_method="umap",
        )
        st.download_button(
            "Download Displacement PDF (Vector)",
            data=pdf_bytes,
            file_name=f"displacement_{selected_path.stem}.pdf",
            mime="application/pdf",
            help="High-resolution vector PDF for manuscript use",
        )

    with export_cols[1]:
        st.download_button(
            "Download Full Data CSV",
            data=df.to_csv(index=False).encode(),
            file_name=f"displacement_data_{Path(viz_dir).name}.csv",
            mime="text/csv",
        )

    with export_cols[2]:
        png_bytes = _fig_to_png(_generate_displacement_png(df))
        st.download_button(
            "Download Displacement PNG (High-Res)",
            data=png_bytes,
            file_name=f"displacement_{selected_path.stem}_highres.png",
            mime="image/png",
        )


def _generate_displacement_pdf(orig_xy, deid_xy, magnitudes, dataset, technique, projection_method="umap"):
    """Generate manuscript-quality displacement PDF from data arrays."""
    from deid.explore.viz import plot_embedding_displacement
    fig = plot_embedding_displacement(
        orig_xy=orig_xy, deid_xy=deid_xy, magnitudes=magnitudes,
        dataset=dataset, technique=technique,
        raw_euclidean_displacement=magnitudes if len(magnitudes) > 0 else None,
        projection_method=projection_method,
    )
    pdf_bytes = _export_pdf(fig)
    plt.close(fig)
    return pdf_bytes


def _fig_to_png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


def _generate_displacement_png(df):
    """Quick PNG for download button."""
    from deid.explore.viz import plot_embedding_displacement
    fig = plot_embedding_displacement(
        orig_xy=df[["orig_x", "orig_y"]].to_numpy(),
        deid_xy=df[["deid_x", "deid_y"]].to_numpy(),
        magnitudes=df.get("euclidean_distance", pd.Series()).to_numpy() if "euclidean_distance" in df.columns else df.get("disp_2d", pd.Series()).to_numpy(),
        dataset="interactive", technique="viewer", projection_method="umap",
    )
    return fig


def _render_comparison(df: pd.DataFrame):
    """Interactive multi-technique comparison with technique toggle."""
    st.subheader("Multi-Technique Comparison")

    try:
        import plotly.graph_objects as go
    except ImportError:
        st.error("Plotly not installed. Run: pip install plotly")
        return

    # Discover technique columns (pattern: {tech}_x, {tech}_y)
    techniques = set()
    for col in df.columns:
        if col.endswith("_x"):
            tech = col[:-2]
            if f"{tech}_y" in df.columns and tech != "orig":
                techniques.add(tech)

    techniques = sorted(techniques)
    selected_techniques = st.multiselect("Show Techniques", techniques, default=techniques[:3])

    # Search filter
    search = st.text_input("Search image filename", "")
    if search:
        df = df[df["image"].str.contains(search, case=False, na=False)]

    max_points = 500
    if len(df) > max_points:
        st.caption(f"Subsampled to {max_points} points (N={len(df)})")
        df_plot = df.sample(n=max_points, random_state=42)
    else:
        df_plot = df

    fig = go.Figure()

    # Original points (gray)
    fig.add_trace(go.Scattergl(
        x=df_plot["orig_x"], y=df_plot["orig_y"],
        mode="markers", marker=dict(size=4, color="lightgray", opacity=0.4),
        name="Original", showlegend=True,
    ))

    colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628"]
    for i, tech in enumerate(selected_techniques):
        color = colors[i % len(colors)]
        tx, ty = f"{tech}_x", f"{tech}_y"

        # De-identified points (faint)
        fig.add_trace(go.Scattergl(
            x=df_plot[tx], y=df_plot[ty],
            mode="markers", marker=dict(size=3, color=color, opacity=0.2),
            name=f"{tech} (deid)", showlegend=False,
        ))

        # Arrows from orig to deid
        dx = df_plot[tx] - df_plot["orig_x"]
        dy = df_plot[ty] - df_plot["orig_y"]
        fig.add_trace(go.Cone(
            x=df_plot["orig_x"], y=df_plot["orig_y"], z=[0] * len(df_plot),
            u=dx, v=dy, w=[0] * len(df_plot),
            colorscale=[[0, color], [1, color]],
            showscale=False,
            name=f"{tech} (displacement)",
            opacity=0.5,
        ))

    fig.update_layout(
        xaxis_title="Component 1",
        yaxis_title="Component 2",
        hovermode="closest",
        height=650,
        showlegend=True,
    )

    st.plotly_chart(fig, use_container_width=True)

    # ── Export ───────────────────────────────────────────────
    st.markdown("---")
    export_cols = st.columns(3)

    with export_cols[0]:
        techs = [t for t in techniques if t in selected_techniques]
        if techs:
            cmp_fig = _generate_comparison_pdf(df, techs)
            st.download_button(
                "Download Comparison PDF (Vector)",
                data=_export_pdf(cmp_fig),
                file_name=f"comparison_{Path(viz_dir).name}.pdf",
                mime="application/pdf",
            )

    with export_cols[1]:
        st.download_button("Download Data CSV", data=df.to_csv(index=False).encode(), file_name=selected_path.name, mime="text/csv")


def _generate_comparison_pdf(df: pd.DataFrame, techniques: list[str]):
    """Generate manuscript-quality comparison PDF."""
    from deid.explore.viz import plot_technique_comparison

    orig_xy = df[["orig_x", "orig_y"]].to_numpy()
    deid_xys = {}
    magnitudes = {}
    for tech in techniques:
        tx, ty = f"{tech}_x", f"{tech}_y"
        deid_xys[tech] = df[[tx, ty]].to_numpy()
        mag_col = f"{tech}_disp_2d"
        magnitudes[tech] = df[mag_col].to_numpy() if mag_col in df.columns else np.ones(len(df))

    fig = plot_technique_comparison(orig_xy, deid_xys, magnitudes, dataset="comparison")
    return fig


def _render_collapse_table(df: pd.DataFrame):
    """Render collapse metrics as interactive sortable table."""
    st.subheader("Identity Collapse Metrics")

    try:
        import plotly.express as px
    except ImportError:
        st.error("Plotly not installed. Run: pip install plotly")
        return

    # Bar chart
    sorted_df = df.sort_values("collapse_ratio").head(30)
    colors = ["#d73027" if r < 0.5 else "#fee090" if r < 1.0 else "#31a354"
              for r in sorted_df["collapse_ratio"]]

    fig = px.bar(
        sorted_df, x="collapse_ratio", y="identity", orientation="h",
        color="collapse_ratio",
        color_continuous_scale=[
            (0, "#d73027"), (0.5, "#fee090"), (1.0, "#fee090"), (1.0, "#31a354")
        ],
        labels={"identity": "Identity", "collapse_ratio": "Collapse Ratio"},
        hover_data=["n_images", "dispersion_before", "dispersion_after", "mean_displacement_norm"],
        title="Per-Identity Collapse Ratio (top 30 most collapsed)",
    )

    fig.add_vline(x=1.0, line_dash="dash", line_color="black", annotation_text="Neutral")
    st.plotly_chart(fig, use_container_width=True)

    # Full sortable table
    st.dataframe(df.sort_values("collapse_ratio"), use_container_width=True)

    # Stats
    cols = st.columns(4)
    cols[0].metric("Mean Collapse Ratio", f"{df['collapse_ratio'].mean():.3f}")
    cols[1].metric("Collapsed (<1.0)", len(df[df["collapse_ratio"] < 1.0]))
    cols[2].metric("Preserved (>1.0)", len(df[df["collapse_ratio"] > 1.0]))
    cols[3].metric("Total Identities", len(df))

    # ── Export ───────────────────────────────────────────────
    st.markdown("---")
    export_cols = st.columns(2)

    with export_cols[0]:
        from deid.explore.viz import plot_identity_dispersion
        collapse_fig, _ = plot_identity_dispersion(df.sort_values("collapse_ratio"), dataset="interactive", technique="viewer", top_n=30)
        st.download_button(
            "Download Collapse PDF (Vector)",
            data=_export_pdf(collapse_fig),
            file_name=f"collapse_{Path(viz_dir).name}.pdf",
            mime="application/pdf",
        )
        plt.close(collapse_fig)

    with export_cols[1]:
        st.download_button("Download Metrics CSV", data=df.to_csv(index=False).encode(), file_name=selected_path.name, mime="text/csv")
