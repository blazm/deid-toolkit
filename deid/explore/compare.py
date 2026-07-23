"""Side-by-side image comparison page."""
from __future__ import annotations

import streamlit as st
import streamlit.components.v1 as components

from deid.explore.data_loader import list_aligned_images, list_deid_images


def render(dataset: str, technique: str) -> None:
    st.subheader(f"Original vs De-identified — {dataset} / {technique}")

    aligned = list_aligned_images(dataset)
    deid = list_deid_images(dataset, technique)

    if not aligned or not deid:
        st.warning("No images available for comparison.")
        st.info(
            """
            **Get started:**

            ```
            deid select datasets arface
            deid select techniques blur
            deid run preprocess
            deid run techniques
            ```
            """
        )
        return

    total = min(len(aligned), len(deid))

    # View mode: grid or single
    view_mode = st.radio("View mode", ["Grid (all pairs)", "Single"], horizontal=True)

    if view_mode == "Grid (all pairs)":
        n_cols = st.slider("Columns", 2, 20, 8, key="compare_n_cols")
        n_rows = (total + n_cols - 1) // n_cols  # ceiling division

        for row in range(n_rows):
            cols = st.columns(n_cols)
            for col_idx in range(n_cols):
                img_idx = row * n_cols + col_idx
                if img_idx < total:
                    with cols[col_idx]:
                        name = aligned[img_idx].name
                        st.image(
                            str(aligned[img_idx]),
                            caption=f"Orig: {name}",
                            width="stretch",
                        )
                        name2 = deid[img_idx].name
                        st.image(
                            str(deid[img_idx]),
                            caption=f"Deid: {name2}",
                            width="stretch",
                        )
                else:
                    with cols[col_idx]:
                        st.empty()

        st.write(f"Total: {total} pairs · {n_cols} cols × {n_rows} rows")

    else:
        # Single pair view with slider
        max_idx = total - 1
        idx = st.slider("Image index", 0, max_idx, 0)

        col1, col2 = st.columns(2)
        with col1:
            name = aligned[idx].name
            st.image(str(aligned[idx]), caption=f"Original: {name}", width="stretch")
        with col2:
            name = deid[idx].name
            st.image(str(deid[idx]), caption=f"De-identified: {name}", width="stretch")

        st.write(f"Showing {idx + 1} of {total}")
