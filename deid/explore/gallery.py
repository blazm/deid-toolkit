"""Image gallery page with filters."""
from __future__ import annotations

import streamlit as st

from deid.explore.data_loader import list_aligned_images, list_deid_images


def render(dataset: str, technique: str) -> None:
    st.subheader(f"Image Gallery — {dataset}")

    show_deid = st.checkbox("Show de-identified", value=True)
    show_original = st.checkbox("Show original", value=True)

    cols = st.columns([1, 4])

    if show_original:
        images = list_aligned_images(dataset)
        if images:
            st.write(f"Original ({len(images)} images)")
            for img in images[:20]:  # Cap at 20 for performance
                with cols[1]:
                    st.image(str(img), use_container_width=True)

    if show_deid and technique:
        images = list_deid_images(dataset, technique)
        if images:
            st.write(f"De-identified with {technique} ({len(images)} images)")
            for img in images[:20]:
                with cols[1]:
                    st.image(str(img), use_container_width=True)
