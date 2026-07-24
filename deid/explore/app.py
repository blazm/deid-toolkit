"""Streamlit result browser for de-identification toolkit.

Unified tablist navigation: Home | Benchmarks | Survey | [Login] | [Docs/Results].
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from deid.explore.assets import logo_path

from deid.explore.data_loader import (
    get_loader,
    list_datasets,
    list_techniques,
    list_evaluations,
    list_results,
)


def require_login(default_redirect: str = "toolkit") -> bool:
    """Check if user is logged in. If not, redirect to login page immediately."""
    if st.session_state.get("logged_in", False):
        return True

    st.session_state.pending_page = default_redirect
    st.query_params["page"] = "login"
    st.rerun()
    return False  # never reached

# ── Initialize session state ─────────────────────────────────────
if "current_page" not in st.session_state:
    st.session_state.current_page = "home"
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = None
if "current_section" not in st.session_state:
    st.session_state.current_section = "private"  # "private" or "public"


# ── Page routing ─────────────────────────────────────────────────
def show_home():
    """Public research portfolio (new landing page) — uses same tabs as toolkit."""
    show_toolkit()


def show_login():
    """Render the full toolkit tabbed interface — tabs always visible on login page."""
    show_toolkit()


def show_public():
    """Render the public benchmarks page (no images, summarized results only)."""
    from deid.explore.public_benchmarks import render as render_public
    render_public()


def show_docs():
    """Toolkit documentation — datasets, techniques, CLI commands."""
    if require_login("docs"):
        from deid.explore.docs import render as render_docs
        render_docs()


def show_survey():
    """Human verification survey page."""
    from deid.explore.survey import render as render_survey
    render_survey()


def show_toolkit():
    """Render the DeID toolkit tabbed interface."""
    is_logged_in = st.session_state.get("logged_in", False)

    # ── Top bar: title (left) + logged-in status (right) ────
    if is_logged_in:
        top_col1, top_col2 = st.columns([6, 2])
        with top_col1:
            st.empty()
        with top_col2:
            st.markdown(
                f'<div style="text-align: right; font-size: 0.85rem; color: #666;">Logged in as **{st.session_state.username}**</div>',
                unsafe_allow_html=True,
            )
    else:
        st.empty()

    # ── Dynamic tab list — Docs/Results only appear when logged in ────
    # On st.rerun() the full DOM re-renders so the tab count can change.
    # This is only safe because we don't pass any state between renders
    # that depends on specific tab objects surviving a count change.
    tab_names = ["Home", "Benchmarks", "Survey", "Login"]
    if is_logged_in:
        tab_names.extend(["Docs", "Results", "Datasets"])

    tab_list = st.tabs(tab_names)

    with tab_list[0]:  # Home
        from deid.explore.landing import render as render_landing
        render_landing()

    with tab_list[1]:  # Benchmarks
        from deid.explore.public_benchmarks import render as render_public
        render_public()

    with tab_list[2]:  # Survey
        from deid.explore.survey import render as render_survey
        render_survey()

    with tab_list[3]:  # Login
        if is_logged_in:
            st.subheader("Account")
            if st.button("Logout", type="primary", use_container_width=True):
                st.session_state.logged_in = False
                st.session_state.username = None
                st.session_state.current_page = "home"
                st.rerun()
        else:
            from deid.explore.login_page import render as render_login
            render_login()

    # ── Protected tabs ────
    if is_logged_in:
        with tab_list[4]:
            from deid.explore.docs import render as render_docs
            render_docs()
        with tab_list[5]:
            _render_toolkit_results()
        with tab_list[6]:
            from deid.explore.datasets import render as render_datasets
            render_datasets()


def _render_toolkit_results():
    """Core toolkit results browser."""
    st.title("De-Identification Toolkit — Results Browser")
    st.markdown("---")

    # Manifest info
    manifest = get_loader().get_result_manifest()
    if manifest:
        ts = manifest.get("timestamp", "")
        with st.expander("Run Metadata"):
            st.write(f"**Run at:** {ts}")
            st.write(f"**Datasets:** {', '.join(manifest.get('datasets', []))}")
            st.write(f"**Techniques:** {', '.join(manifest.get('techniques', []))}")
            st.write(f"**Evaluations:** {', '.join(manifest.get('evaluation', []))}")

    # Filters
    st.sidebar.header("Filters")
    loader = get_loader()
    datasets = [(name, aligned) for name, aligned in list_datasets()]
    techniques = list_techniques()
    evaluations = list_evaluations()

    # Sidebar status
    if not datasets:
        st.sidebar.warning("No datasets found.")
        st.sidebar.caption("`deid list datasets`")
    if not techniques:
        st.sidebar.warning("No techniques found.")
        st.sidebar.caption("`deid run techniques`")
    if not evaluations:
        st.sidebar.warning("No evaluations found.")
        st.sidebar.caption("`deid run evaluation`")

    selected_dataset = st.sidebar.selectbox(
        "Dataset",
        [d[0] for d in datasets] if datasets else ["(none)"],
        index=0,
    )
    selected_technique = st.sidebar.selectbox(
        "Technique",
        techniques if techniques else ["(none)"],
        index=0,
    )
    selected_evaluation = st.sidebar.selectbox(
        "Metric",
        evaluations if evaluations else ["(none)"],
        index=0,
    )

    # Results sidebar
    results = list_results()
    st.sidebar.header("Results")
    if results:
        for ds, techs in sorted(results.items()):
            st.sidebar.write(f"**{ds}**")
            for tech, metrics in sorted(techs.items()):
                st.sidebar.write(f"  - {tech}: {', '.join(metrics.keys())}")
    else:
        st.sidebar.warning("No results found — run the pipeline first.")
        st.sidebar.caption("`deid run all`")

    # ── Tabs ─────────────────────────────────────────────
    (
        tab_compare, tab_summary, tab_unsupervised,
        tab_reid, tab_metrics, tab_clusters,
        tab_gallery, tab_techniques,
        tab_embedding_analysis, tab_interactive_viz,
    ) = st.tabs([
        "Compare", "Summary", "Embeddings", "Re-ID Risk",
        "Metrics", "Clusters", "Gallery", "Techniques",
        "Embedding Analysis", "Interactive Viewer",
    ])

    with tab_compare:
        from deid.explore.compare import render as render_compare
        render_compare(selected_dataset, selected_technique)

    with tab_summary:
        from deid.explore.summary import render as render_summary
        render_summary()

    with tab_metrics:
        from deid.explore.metrics import render as render_metrics
        render_metrics()

    with tab_clusters:
        from deid.explore.clusters import render as render_clusters
        render_clusters(selected_dataset, selected_technique)

    with tab_gallery:
        from deid.explore.gallery import render as render_gallery
        render_gallery(selected_dataset, selected_technique)

    with tab_techniques:
        from deid.explore.techniques_grid import render as render_techniques
        render_techniques()

    with tab_embedding_analysis:
        from deid.explore.embedding_analysis_tab import render as render_embedding_analysis
        render_embedding_analysis()

    with tab_interactive_viz:
        from deid.explore.interactive_embedding_tab import render as render_interactive_viz
        render_interactive_viz()

    with tab_unsupervised:
        st.subheader("Embedding Clustering — t-SNE/UMAP")
        st.caption("Project identity embeddings to 2D. Colored by available labels (gender, expression, ethnicity).")

        dataset_name = st.selectbox("Dataset", [d[0] for d in datasets] if datasets else ["(none)"], key="embed_dataset")
        technique_name = st.selectbox("Technique", techniques if techniques else ["(none)"], key="embed_technique")

        if dataset_name != "(none)" and technique_name != "(none)":
            labels_dir = Path(loader.settings.root_dir) / "datasets" / "labels"
            labels_df = None
            label_file = labels_dir / f"{dataset_name}_labels.csv"
            if label_file.exists():
                labels_df = pd.read_csv(label_file)
                if "Name" not in labels_df.columns:
                    name_col = next(
                        (c for c in labels_df.columns if c.lower() in {"name", "filename", "file_name"}),
                        None,
                    )
                    if name_col:
                        labels_df = labels_df.rename(columns={name_col: "Name"})

            st.info("Install dependencies: pip install scikit-learn umap-learn deepface")
            st.write("**Dataset labels found:**", not labels_df.empty)
            if labels_df is not None:
                st.write("Available metadata columns:", [c for c in labels_df.columns if c != "Name"])

    with tab_reid:
        st.subheader("Re-Identification Risk Assessment")
        st.caption("Measures how much identity information remains after de-identification.")

        dataset_name = st.selectbox("Dataset", [d[0] for d in datasets] if datasets else ["(none)"], key="reid_dataset")
        technique_name = st.selectbox("Technique", techniques if techniques else ["(none)"], key="reid_technique")

        if dataset_name != "(none)" and technique_name != "(none)":
            deid_path = loader.settings.deid_path / technique_name / dataset_name

            if deid_path.exists():
                st.info("Install dependencies: pip install scikit-learn deepface")
                st.write(
                    "**Images found:**",
                    len([f for f in deid_path.iterdir() if f.suffix.lower() in {".png", ".jpg", ".jpeg"}]),
                )
                st.write(
                    "To run re-ID risk assessment, use the CLI:\n"
                    "```\ndeid reid-risk --dataset %s --technique %s\n```",
                    dataset_name, technique_name,
                )
            else:
                st.warning("No de-identified images found for this technique/dataset.")


# ── Main dispatch ────────────────────────────────────────────────
page_map = {
    "home": show_home,
    "login": show_login,
    "public": show_public,
    "docs": show_docs,
    "survey": show_survey,
    "toolkit": show_toolkit,
}

# ── Query param routing (for sharable URLs) ───────────────────────
if "page" in st.query_params:
    qpage = st.query_params.get("page")
    if qpage and qpage in page_map:
        st.session_state.current_page = qpage

# ── Redirect to login for protected pages ────
protected_pages = {"docs", "toolkit"}
if st.session_state.current_page in protected_pages and not st.session_state.get("logged_in", False):
    st.session_state.pending_page = st.session_state.current_page
    st.query_params["page"] = "login"
    st.rerun()

page = st.session_state.current_page
if page not in page_map:
    page = "home"

page_map[page]()

# Navigation between public/private sections is now handled by the tablist above.
