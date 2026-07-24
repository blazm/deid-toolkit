"""Streamlit tab: Embedding Space Analysis.

Provides interactive visualizations for embedding displacement, identity collapse,
and multi-technique comparison in a shared projection space.
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import streamlit as st

import pandas as pd

from deid.explore.data_loader import get_loader, list_datasets, list_techniques


def render() -> None:
    st.title("Embedding Space Analysis")
    st.caption(
        "Visualize how de-identification techniques manipulate identity embeddings. "
        "Uses cached .pkl files from evaluation runs."
    )

    # ── Sidebar controls ────────────────────────────────────────
    st.sidebar.header("Controls")

    datasets = [d[0] for d in list_datasets()] if list_datasets() else ["(none)"]
    dataset = st.sidebar.selectbox("Dataset", datasets, index=0)
    if dataset == "(none)":
        st.warning("No datasets found.")
        return

    # Discover available embedding models from cache
    root_dir = get_loader().settings.root_dir
    from deid.explore.embedding_analysis import discover_embedding_models
    available_models = discover_embedding_models(root_dir)
    if not available_models:
        st.error(
            "No cached embeddings found. Run the evaluation pipeline first:\n\n"
            "`deid run evaluation`  (with arcface, adaface_optimized, or swinface selected)"
        )
        return

    model = st.sidebar.selectbox("Embedding Model", available_models, index=0)

    proj_method = st.sidebar.radio("Projection Method", ["umap", "pca", "tsne"], index=0)

    # Technique selection (multi-select for comparison, single for displacement/collapse)
    techniques = list_techniques()
    if techniques:
        sel_techniques = st.sidebar.multiselect("Technique(s)", techniques, default=[techniques[0]] if techniques else [])
    else:
        st.warning("No techniques found.")
        return

    # ── Tabs ────────────────────────────────────────────────────
    tabs = st.tabs(["Displacement", "Collapse Analysis", "Technique Comparison"])

    with tabs[0]:  # Displacement
        _render_displacement(dataset, model, proj_method, sel_techniques)

    with tabs[1]:  # Collapse
        _render_collapse(dataset, model, sel_techniques)

    with tabs[2]:  # Comparison
        _render_comparison(dataset, model, proj_method, sel_techniques)


def _export_pdf(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="pdf", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


def _render_displacement(dataset, model, proj_method, techniques):
    st.subheader("Per-Image Displacement Field")
    st.caption("Arrows show how each image's embedding moved from original to de-identified.")

    if not techniques:
        st.warning("Select at least one technique in the sidebar.")
        return

    tech = st.selectbox("Technique", techniques, key="disp_tech")

    if st.button("Generate Displacement Plot"):
        try:
            from deid.explore.embedding_analysis import (
                load_labels, prepare_displacement_data,
                EmbeddingCacheNotFoundError,
            )
            from deid.explore.viz import plot_embedding_displacement

            labels_df = load_labels(get_loader().settings.root_dir, dataset)
            data = prepare_displacement_data(
                get_loader().settings.root_dir, model, dataset, tech,
                labels_df=labels_df, projection_method=proj_method,
            )

            if "error" in data:
                st.error(data["error"])
                return

            fig = plot_embedding_displacement(
                orig_xy=data["orig_xy"],
                deid_xy=data["deid_xy"],
                magnitudes=data["magnitudes"],
                dataset=dataset,
                technique=tech,
                image_names=data.get("image_names"),
                identities=data.get("identities"),
                raw_euclidean_displacement=data.get("raw_euclidean_displacement"),
                cosine_similarity=data.get("cosine_similarity"),
                projection_method=proj_method,
            )

            st.pyplot(fig)

            # Stats — prefer raw embedding-space metrics
            raw_eucl = data.get("raw_euclidean_displacement")
            cos_sim  = data.get("cosine_similarity")
            cols = st.columns(4)
            cols[0].metric("Images", len(data["magnitudes"]))
            cols[1].metric("Mean Cosine Sim.", f"{float(cos_sim.mean()):.3f}" if cos_sim is not None else "/")
            cols[2].metric("Euclidean Dist.",  f"{float(raw_eucl.mean()):.2f}" if raw_eucl is not None else "/")
            cols[3].metric("Max Euclidean",    f"{float(raw_eucl.max()):.2f}" if raw_eucl is not None else "/")

            # Download button
            pdf_bytes = _export_pdf(fig)
            st.download_button(
                "Download PDF", data=pdf_bytes,
                file_name=f"displacement_{model}_{dataset}_{tech}.pdf",
                mime="application/pdf",
            )
        except EmbeddingCacheNotFoundError as e:
            st.error(str(e))
        except Exception as exc:
            st.error(f"Error: {exc}")
            st.exception(exc)


def _render_collapse(dataset, model, techniques):
    st.subheader("Identity Collapse Detection")
    st.caption(
        "Measures if distinct identities merge toward similar embeddings after de-identification. "
        "Collapse ratio < 1 = identity structure weakened (good for privacy). "
        "Ratio > 1 = spread apart (bad: identity still distinguishable)."
    )

    if not techniques:
        st.warning("Select at least one technique in the sidebar.")
        return

    tech = st.selectbox("Technique", techniques, key="collapse_tech")

    if st.button("Analyze Collapse"):
        try:
            from deid.explore.embedding_analysis import (
                load_labels, prepare_collapse_data, compute_technique_summary,
                EmbeddingCacheNotFoundError,
            )
            from deid.explore.viz import plot_identity_dispersion

            labels_df = load_labels(get_loader().settings.root_dir, dataset)
            data = prepare_collapse_data(
                get_loader().settings.root_dir, model, dataset, tech, labels_df=labels_df,
            )

            if "error" in data:
                st.error(data["error"])
                return

            disp_df = data["dispersion_df"]
            if disp_df.empty:
                st.warning("No per-identity dispersion data (need 2+ images per identity).")
                return

            # Summary stats
            summary = data.get("summary", {})
            cols = st.columns(4)
            cols[0].metric("Images", summary.get("n_images", len(data["records"])))
            cols[1].metric("Mean Cosine Similarity", f"{summary.get('mean_cosine_similarity', 0):.3f}")
            cols[2].metric("Collapse Ratio (mean)", f"{disp_df['collapse_ratio'].mean():.3f}")
            collapsed_count = len(disp_df[disp_df["collapse_ratio"] < 1.0])
            total_identities = len(disp_df)
            cols[3].metric(f"Collapsed Identities", f"{collapsed_count}/{total_identities}")

            # Bar chart
            fig, sorted_df = plot_identity_dispersion(
                disp_df, dataset=dataset, technique=tech, top_n=20,
            )
            st.pyplot(fig)

            # Table
            st.subheader("Per-Identity Dispersion Metrics")
            st.dataframe(sorted_df.sort_values("collapse_ratio"), use_container_width=True)

            # Download
            pdf_bytes = _export_pdf(fig)
            st.download_button(
                "Download PDF", data=pdf_bytes,
                file_name=f"collapse_{model}_{dataset}_{tech}.pdf",
                mime="application/pdf",
            )

            csv_buf = io.StringIO()
            sorted_df.to_csv(csv_buf, index=False)
            st.download_button(
                "Download CSV", data=csv_buf.getvalue(),
                file_name=f"collapse_{model}_{dataset}_{tech}_metrics.csv",
                mime="text/csv",
            )

        except EmbeddingCacheNotFoundError as e:
            st.error(str(e))
        except Exception as exc:
            st.error(f"Error: {exc}")
            st.exception(exc)


def _render_comparison(dataset, model, proj_method, techniques):
    st.subheader("Multi-Technique Comparison")
    st.caption(
        "All techniques overlaid in a shared embedding projection. "
        "Compare displacement patterns: uniform scatter vs identity collapse vs rotation."
    )

    if len(techniques) < 2:
        st.warning("Select at least 2 techniques to compare.")
        return

    # Summary table first
    from deid.explore.embedding_analysis import (
        load_multi_technique_embeddings, compute_technique_summary,
        load_labels, EmbeddingCacheNotFoundError,
    )

    labels_df = load_labels(get_loader().settings.root_dir, dataset)
    try:
        all_recs = load_multi_technique_embeddings(
            get_loader().settings.root_dir, model, dataset, techniques, labels_df,
        )
    except EmbeddingCacheNotFoundError as e:
        st.error(str(e))
        return

    # Summary comparison table
    summary_rows = []
    for tech, recs in all_recs.items():
        s = compute_technique_summary(recs)
        if s:
            summary_rows.append({
                "Technique": tech,
                "Images": s.get("n_images", 0),
                "Mean Cosine Sim.": f"{s.get('mean_cosine_similarity', 0):.3f}",
                "Mean Displacement": f"{s.get('mean_euclidean_displacement', 0):.3f}",
            })

    if summary_rows:
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

    if st.button("Generate Comparison Plot"):
        try:
            import pandas as pd
            from deid.explore.embedding_analysis import prepare_comparison_data
            from deid.explore.viz import plot_technique_comparison

            cmp_data = prepare_comparison_data(
                get_loader().settings.root_dir, model, dataset, techniques,
                labels_df=labels_df, projection_method=proj_method,
            )

            if "error" in cmp_data:
                st.error(cmp_data["error"])
                return

            fig = plot_technique_comparison(
                orig_xy=cmp_data["orig_xy"],
                deid_xys={t: cmp_data["deid_xys"][f"deid_{t}"] for t in cmp_data["techniques"]},
                magnitudes=cmp_data["magnitudes_by_tech"],
                dataset=dataset,
            )

            st.pyplot(fig)

            pdf_bytes = _export_pdf(fig)
            st.download_button(
                "Download PDF", data=pdf_bytes,
                file_name=f"comparison_{model}_{dataset}_multi.pdf",
                mime="application/pdf",
            )

        except EmbeddingCacheNotFoundError as e:
            st.error(str(e))
        except Exception as exc:
            st.error(f"Error: {exc}")
            st.exception(exc)
