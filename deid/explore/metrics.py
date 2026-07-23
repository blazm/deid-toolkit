"""Metric scores viewer page."""
from __future__ import annotations

from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from deid.explore.data_loader import list_results, load_results_csv
from deid.explore.viz import (
    plot_score_distribution,
    plot_metric_table,
    plot_roc_curve,
    plot_pr_curve,
    plot_cmc_curve,
)


def _export_fig(fig, base_name: str) -> None:
    """Save a matplotlib figure as PDF via Streamlit download button."""
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
    st.subheader("Evaluation Metrics")

    results = list_results()
    if not results:
        st.warning("No evaluation results found. Run the pipeline first.")
        st.info(
            """
            **Get started:**

            ```
            deid select datasets arface
            deid select techniques blur
            deid select evaluation ssim
            deid run all
            ```
            """
        )
        return

    # Dataset selector
    datasets = sorted(results.keys())
    selected_ds = st.selectbox("Dataset", datasets)
    techniques = sorted(results[selected_ds].keys())
    selected_tech = st.selectbox("Technique", techniques)

    metrics = results[selected_ds][selected_tech]
    selected_metric_name = st.selectbox("Metric", list(metrics.keys()))
    csv_path = metrics[selected_metric_name]

    # Display
    st.write(f"Results from **{csv_path.name}**")
    try:
        df = load_results_csv(csv_path)
        st.dataframe(df, use_container_width=True)

        # CMC curve (identification results: probe + rank columns)
        has_rank = "rank" in df.columns and "true_identity" in df.columns
        if has_rank:
            cmc_fig = plot_cmc_curve(df, selected_tech, selected_ds, selected_metric_name)
            st.pyplot(cmc_fig)
            _export_fig(cmc_fig, f"cmc_{selected_metric_name}")

            # Summary stats
            rank1 = (df["rank"] == 1).sum() if "rank" in df.columns else 0
            rank5 = (df["rank"] <= 5).sum() if "rank" in df.columns else 0
            n = len(df)
            col1, col2 = st.columns(2)
            col1.metric("Rank@1", f"{rank1}/{n} ({100*rank1/n:.1f}%)")
            col2.metric("Rank@5", f"{rank5}/{n} ({100*rank5/n:.1f}%)")
        else:
            st.caption("CMC curve unavailable — requires rank + true_identity columns.")

        # ROC curve (verification-based evaluations: ground_truth + score)
        roc_fig, roc_auc = plot_roc_curve(df, selected_tech, selected_ds, selected_metric_name)
        if roc_auc is not None:
            st.metric("Area Under ROC Curve", f"{roc_auc:.3f}")
            st.pyplot(roc_fig)
            _export_fig(roc_fig, f"roc_{selected_metric_name}")
        else:
            st.caption("ROC curve unavailable — requires binary ground_truth + score columns.")

        # PR curve
        pr_fig, pr_ap = plot_pr_curve(df, selected_tech, selected_ds, selected_metric_name)
        if pr_ap is not None:
            st.metric("Average Precision", f"{pr_ap:.3f}")
            st.pyplot(pr_fig)
            _export_fig(pr_fig, f"pr_{selected_metric_name}")
        else:
            st.caption("PR curve unavailable — requires binary ground_truth + score columns.")

        # Generic score distribution
        score_cols = [c for c in df.columns if "score" in c.lower() or "sim" in c.lower() or "dist" in c.lower() or c == "score"]
        if score_cols:
            fig = plot_score_distribution(df, score_cols[0], selected_tech, selected_ds, selected_metric_name)
            st.pyplot(fig)
            _export_fig(fig, f"score_{selected_metric_name}")
    except Exception as exc:
        st.error(f"Could not load results: {exc}")
