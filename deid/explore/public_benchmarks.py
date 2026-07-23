"""Public Benchmarks — read-only summary view without images.

Shows benchmark results, score tables, ROC/PR curves, and score distributions
so the public can compare techniques without accessing any sensitive biometric data.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from deid.explore.data_loader import get_loader, list_results, list_datasets, list_techniques, list_evaluations, load_results_csv
from deid.explore.viz import (
    plot_score_summary,
    plot_roc_multi,
    plot_distance_histogram,
    plot_score_distribution,
    plot_radar_chart,
)


def _export_pdf(fig: object, base_name: str) -> None:
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
    st.set_page_config(page_title="Public Benchmarks", layout="wide", page_icon="deid/assets/logo.svg")

    st.title("Public Benchmarks")
    st.caption("Summarized evaluation results. No individual images shown.")
    st.markdown("---")

    loader = get_loader()
    results = list_results()

    if not results:
        st.warning("No evaluation results available yet.")
        st.info(
            """
            **To populate benchmarks, run the pipeline:**

            ```
            deid select datasets arface ck+_fix
            deid select techniques deepprivacy2 blur
            deid select evaluation ssim lpips
            deid run all
            ```
            """
        )
        return

    # Overview stats
    n_techniques = len(set(t for techs in results.values() for t in techs.keys()))
    n_evaluations = len(set(e for techs in results.values() for es in techs.values() for e in es.keys()))
    n_datasets = len(results)
    col1, col2, col3 = st.columns(3)
    col1.metric("Datasets", str(n_datasets))
    col2.metric("Techniques", str(n_techniques))
    col3.metric("Metrics", str(n_evaluations))
    st.markdown("")

    # Filters
    filter_col1, filter_col2 = st.columns(2)
    with filter_col1:
        selected_ds = st.selectbox("Dataset", sorted(results.keys()))
    with filter_col2:
        techniques = sorted(results.get(selected_ds, {}).keys())
        selected_techniques = st.multiselect("Techniques", techniques, default=techniques[:min(3, len(techniques))])

    ds_results = results.get(selected_ds, {})
    if not ds_results:
        st.warning("No results for this dataset.")
        return

    # View selector
    view = st.radio(
        "View",
        ["Score Summary Table", "ROC Comparison", "Score Distributions", "Radar Comparison", "Detailed Results"],
        horizontal=True,
        key="public_view",
    )

    # ---- Score Summary Table ----
    if view == "Score Summary Table":
        st.subheader("Score Summary — " + selected_ds)
        fig, stats_df = plot_score_summary({t: {selected_ds: ds_results.get(t, {})} for t in selected_techniques if t in ds_results})
        if not stats_df.empty:
            st.dataframe(stats_df, use_container_width=True)
        if not fig.empty:
            st.pyplot(fig)
            _export_pdf(fig, f"summary_{selected_ds}")

    # ---- ROC Comparison ----
    elif view == "ROC Comparison":
        st.subheader("ROC Comparison — " + selected_ds)

        all_evals = set()
        for tech_name, tech_results in ds_results.items():
            all_evals.update(tech_results.keys())
        all_evals = sorted(all_evals)

        if not all_evals:
            st.info("No verification evaluations available for ROC curves.")
        else:
            selected_eval = st.selectbox("Evaluation metric", all_evals)

            tech_dfs = {}
            for tech_name in selected_techniques:
                if tech_name not in ds_results:
                    continue
                tech_results = ds_results[tech_name]
                if selected_eval not in tech_results:
                    continue
                csv_path = tech_results[selected_eval]
                df = load_results_csv(csv_path)

                score_col, label_col = None, None
                for candidate in ["score", "cosine_similarity", "cossim", "similarity_score", "similarity", "cosine", "sim"]:
                    if candidate in df.columns:
                        score_col = candidate
                        break
                if not score_col:
                    score_candidates = [c for c in df.columns if "score" in c.lower() or "sim" in c.lower()]
                    score_col = score_candidates[0] if score_candidates else None

                for candidate in ["ground_truth", "label", "gt"]:
                    if candidate in df.columns:
                        label_col = candidate
                        break

                if score_col and label_col:
                    tech_dfs[tech_name] = df

            if len(tech_dfs) < 2:
                st.warning(f"Need at least 2 techniques with verification scores. Found: {len(tech_dfs)}")
            else:
                fig, info = plot_roc_multi(tech_dfs, selected_ds, selected_eval)
                if not fig.empty:
                    st.pyplot(fig)
                    _export_pdf(fig, f"roc_public_{selected_eval}")

                if info:
                    roc_rows = [{
                        "Technique": item["technique"],
                        "AUC": f"{item['auc']:.4f}",
                        "EER": f"{item['eer']:.4f}",
                        "Threshold": f"{item['threshold']:.3f}",
                    } for item in info]
                    st.dataframe(roc_rows, use_container_width=True)

    # ---- Score Distributions ----
    elif view == "Score Distributions":
        st.subheader("Score Distributions — " + selected_ds)

        all_evals = set()
        for tech_results in ds_results.values():
            all_evals.update(tech_results.keys())
        all_evals = sorted(all_evals)

        selected_eval = st.selectbox("Evaluation metric", all_evals)
        technique = st.selectbox("Technique", sorted(ds_results.keys()))

        if selected_eval in ds_results.get(technique, {}):
            csv_path = ds_results[technique][selected_eval]
            df = load_results_csv(csv_path)

            score_col = None
            for candidate in ["score", "cosine_similarity", "cossim", "similarity_score", "similarity", "cosine", "sim"]:
                if candidate in df.columns:
                    score_col = candidate
                    break
            if not score_col:
                score_candidates = [c for c in df.columns if "score" in c.lower() or "sim" in c.lower() or "dist" in c.lower()]
                score_col = score_candidates[0] if score_candidates else None

            if score_col:
                fig = plot_score_distribution(df, score_col, technique, selected_ds, selected_eval)
                if not fig.empty:
                    st.pyplot(fig)
                    _export_pdf(fig, f"dist_public_{selected_eval}")
            else:
                st.info(f"No score column found for {selected_eval}. Available columns: {list(df.columns)}")

    # ---- Radar Comparison ----
    elif view == "Radar Comparison":
        st.subheader("Radar Comparison — " + selected_ds)

        all_evals = set()
        for tech_results in ds_results.values():
            all_evals.update(tech_results.keys())
        all_evals = sorted(all_evals)

        if not all_evals:
            st.info("No evaluation results available for radar chart.")
        else:
            fig = plot_radar_chart({t: {selected_ds: ds_results.get(t, {})} for t in selected_techniques if t in ds_results}, selected_ds)
            if fig is not None and not fig.empty:
                st.pyplot(fig)
                _export_pdf(fig, f"radar_{selected_ds}")
            else:
                st.info("Cannot compute radar chart — need at least 2 techniques with evaluated metrics.")

    # ---- Detailed Results ----
    elif view == "Detailed Results":
        st.subheader("Detailed Results — " + selected_ds)

        rows = []
        for tech_name in sorted(ds_results.keys()):
            for ev_name, csv_path in sorted(ds_results[tech_name].items()):
                try:
                    df = load_results_csv(csv_path)
                    scores = df.iloc[:, 1] if len(df.columns) > 1 else df.iloc[:, 0]
                    rows.append({
                        "Technique": tech_name,
                        "Metric": ev_name,
                        "Mean": f"{scores.mean():.4f}",
                        "Std": f"{scores.std():.4f}",
                        "Min": f"{scores.min():.4f}",
                        "Max": f"{scores.max():.4f}",
                        "Count": len(scores),
                        "Source": str(csv_path.name),
                    })
                except Exception:
                    rows.append({
                        "Technique": tech_name,
                        "Metric": ev_name,
                        "Mean": "-",
                        "Std": "-",
                        "Min": "-",
                        "Max": "-",
                        "Count": 0,
                        "Source": str(csv_path.name),
                    })

        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # Footer
    st.markdown("---")
    st.caption(
        "For full results including image galleries, "
        "[login to the Face De-Identification Toolkit](#) or run `deid explore` locally. "
        "This public section intentionally excludes all individual images to protect biometric privacy."
    )
