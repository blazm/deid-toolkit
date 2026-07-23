"""Summary tab — overview metrics across all techniques/datasets.

Displays score tables, ROC comparisons, distance histograms, and confusion matrices.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from deid.explore.data_loader import list_results, load_results_csv
from deid.explore.viz import (
    plot_score_summary,
    plot_roc_multi,
    plot_distance_histogram,
    plot_confusion_matrix,
)
from deid.explore.data_loader import get_loader
from deid.explore.radar_charts import (
    render_static_benchmark,
    render_loaded_charts,
)


def _export_fig(fig, base_name: str) -> None:
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format="pdf")
    buf.seek(0)
    st.download_button(
        label=f"Download {base_name} as PDF",
        data=buf,
        file_name=f"{base_name}.pdf",
        mime="application/pdf",
    )


def render() -> None:
    st.subheader("Summary")

    results = list_results()
    if not results:
        st.warning("No evaluation results found.")
        st.info(
            """
            **Get started:**

            ```
            deid select datasets arface ck+_fix
            deid select techniques deepprivacy2 blur
            deid select evaluation ssim lpips
            deid run all
            ```
            """
        )
        return

    # View selector
    view = st.radio(
        "View",
        ["Score Table", "ROC Comparison", "Distance Histograms", "Confusion Matrices", "Radar Charts"],
        horizontal=True,
    )

    # Dataset selector
    datasets = sorted(results.keys())
    selected_ds = st.selectbox("Dataset", datasets)

    # ---- Score Table ----
    if view == "Score Table":
        st.caption("Mean ± std score per technique per evaluation.")
        fig, stats_df = plot_score_summary(results)
        if not stats_df.empty:
            st.dataframe(stats_df, use_container_width=True)
        if not fig.empty:
            st.pyplot(fig)
            _export_fig(fig, "score_summary")

    # ---- ROC Comparison ----
    elif view == "ROC Comparison":
        st.caption("ROC curves for all techniques on the selected dataset.")
        ds_results = results.get(selected_ds, {})
        if not ds_results:
            st.warning("No results for this dataset.")
            return

        # Collect all eval names
        all_evals = set()
        for tech, tech_results in ds_results.items():
            all_evals.update(tech_results.keys())
        all_evals = sorted(all_evals)

        selected_eval = st.selectbox("Evaluation metric", all_evals)

        # Collect technique DataFrames
        tech_dfs = {}
        for tech_name, tech_results in sorted(ds_results.items()):
            if selected_eval in tech_results:
                csv_path = tech_results[selected_eval]
                df = load_results_csv(csv_path)
                score_col, label_col = None, None
                # Find score column
                for candidate in ["score", "cosine_similarity", "cossim", "similarity_score", "similarity", "cosine", "sim"]:
                    if candidate in df.columns:
                        score_col = candidate
                        break
                if not score_col:
                    score_candidates = [c for c in df.columns if "score" in c.lower() or "sim" in c.lower()]
                    score_col = score_candidates[0] if score_candidates else None
                # Find label column
                for candidate in ["ground_truth", "label", "gt"]:
                    if candidate in df.columns:
                        label_col = candidate
                        break

                if score_col and label_col:
                    tech_dfs[tech_name] = df

        if len(tech_dfs) < 2:
            st.warning(f"Need at least 2 techniques with valid score/label columns for ROC comparison. Found: {len(tech_dfs)}")
            return

        fig, info = plot_roc_multi(tech_dfs, selected_ds, selected_eval)
        if not fig.empty:
            st.pyplot(fig)
            _export_fig(fig, f"roc_{selected_eval}")

        # Show AUC/EER table
        if info:
            roc_df = []
            for item in info:
                roc_df.append({
                    "Technique": item["technique"],
                    "AUC": f"{item['auc']:.4f}",
                    "EER": f"{item['eer']:.4f}",
                    "Threshold": f"{item['threshold']:.3f}",
                })
            st.dataframe(roc_df, use_container_width=True)

    # ---- Distance Histograms ----
    elif view == "Distance Histograms":
        st.caption("Genuine vs impostor score distributions per technique.")
        ds_results = results.get(selected_ds, {})
        if not ds_results:
            st.warning("No results for this dataset.")
            return

        all_evals = set()
        for tech, tech_results in ds_results.items():
            all_evals.update(tech_results.keys())
        all_evals = sorted(all_evals)

        selected_eval = st.selectbox("Evaluation metric", all_evals)
        technique = st.selectbox("Technique", sorted(ds_results.keys()))

        if selected_eval in ds_results.get(technique, {}):
            csv_path = ds_results[technique][selected_eval]
            df = load_results_csv(csv_path)
            fig = plot_distance_histogram(df, technique, selected_ds, selected_eval)
            if not fig.empty:
                st.pyplot(fig)
                _export_fig(fig, f"dist_{selected_eval}")

    # ---- Confusion Matrices ----
    elif view == "Confusion Matrices":
        st.caption("Confusion matrices from emotion recognition evaluations.")
        ds_results = results.get(selected_ds, {})
        if not ds_results:
            st.warning("No results for this dataset.")
            return

        # Look for evaluations that have emotion predictions
        emotion_evals = []
        for tech_name, tech_results in ds_results.items():
            for ev_name, csv_path in tech_results.items():
                try:
                    df = load_results_csv(csv_path)
                    has_emotion = any("prediction" in c.lower() for c in df.columns)
                    if has_emotion:
                        emotion_evals.append((ev_name, csv_path, tech_name))
                except Exception:
                    pass

        if not emotion_evals:
            st.info("No emotion recognition results found. Confusion matrices will appear here once available.")
            return

        selected_ev_name, csv_path, selected_tech = st.selectbox(
            "Evaluation",
            [(n, str(p), t) for n, p, t in emotion_evals],
            format_func=lambda x: f"{x[0]} ({x[2]})",
        )

        df_results = load_results_csv(Path(csv_path))

        # Load labels
        loader = get_loader()
        labels_dir = Path(loader.settings.root_dir) / "datasets" / "labels"
        labels_df = None
        if labels_dir.exists():
            label_file = labels_dir / f"{selected_ds}_labels.csv"
            if label_file.exists():
                labels_df = load_results_csv(label_file)

        if labels_df is not None and not labels_df.empty:
            fig, stats_df = plot_confusion_matrix(df_results, labels_df, selected_ev_name, selected_tech, selected_ds)
            if not fig.empty:
                st.pyplot(fig)
                _export_fig(fig, f"confusion_{selected_ev_name}")
            if not stats_df.empty:
                st.dataframe(stats_df, use_container_width=True)
        else:
            st.warning("No labels CSV found for this dataset — confusion matrix requires dataset labels.")

    # ---- Radar Charts ----
    elif view == "Radar Charts":
        st.caption("Multi-dimensional comparison: each axis is an evaluation dimension.")

        radar_subview = st.radio(
            "Data source",
            ["Static Benchmark", "Loaded Evaluation Results"],
            horizontal=True,
            key="radar_source",
        )

        if radar_subview == "Static Benchmark":
            st.subheader("Static Benchmark Comparison")
            st.caption("Scores sourced from published papers (0.0 = worst, 1.0 = best).")
            fig_pdf, pdf_buf, svg_buf = render_static_benchmark()
            st.pyplot(fig_pdf)

            # SVG for interactive hover
            st.components.v1.html(svg_buf.read(), height=600, scrolling=False)

            st.download_button(
                label="Download Radar Chart as PDF",
                data=pdf_buf,
                file_name="radar_charts_benchmark.pdf",
                mime="application/pdf",
            )

        else:
            st.subheader("Loaded Evaluation Radar Chart")
            st.caption("Axes are aggregated means across datasets (DEID, DIV, 1-MSE, EX, GD).")

            datasets = sorted(results.keys())
            ds = st.selectbox("Dataset group", datasets)

            techs = sorted(results.get(ds, {}).keys())
            selected_techs = st.multiselect(
                "Techniques to compare",
                techs,
                default=techs[:min(3, len(techs))],
            )

            if not selected_techs:
                st.info("Select at least one technique to display.")
            else:
                fig, pdf_buf, svg_buf = render_loaded_charts(results, ds, selected_techs)
                st.pyplot(fig)
                st.components.v1.html(svg_buf.read(), height=700, scrolling=False)

                st.download_button(
                    label="Download Radar Chart as PDF",
                    data=pdf_buf,
                    file_name=f"radar_charts_{ds}.pdf",
                    mime="application/pdf",
                )

    # Bottom: per-technique detailed table
    st.subheader("All Results")
    table_rows = []
    for ds_name in sorted(results.keys()):
        for tech_name in sorted(results[ds_name].keys()):
            for ev_name, csv_path in sorted(results[ds_name][tech_name].items()):
                try:
                    df = load_results_csv(csv_path)
                    scores = df.iloc[:, 1] if len(df.columns) > 1 else df.iloc[:, 0]
                    table_rows.append({
                        "dataset": ds_name,
                        "technique": tech_name,
                        "metric": ev_name,
                        "mean": f"{scores.mean():.4f}",
                        "std": f"{scores.std():.4f}",
                        "count": len(scores),
                    })
                except Exception:
                    table_rows.append({
                        "dataset": ds_name,
                        "technique": tech_name,
                        "metric": ev_name,
                        "mean": "-",
                        "std": "-",
                        "count": 0,
                    })

    if table_rows:
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True)
