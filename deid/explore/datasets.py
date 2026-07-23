"""Datasets tab — browse available datasets with metadata and sample images."""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from deid.explore.data_loader import (
    list_datasets,
    get_dataset_image_count,
    get_dataset_image_dimensions,
    get_dataset_attribute_columns,
    get_dataset_sample_images,
)


def render() -> None:
    st.subheader("Available Datasets")

    all_datasets = list_datasets()
    if not all_datasets:
        st.info("No datasets available. Run `deid run preprocess` to populate the dataset directories.")
        return

    # ── Dataset overview table ──
    rows = []
    for name, aligned in all_datasets:
        count = get_dataset_image_count(name)
        attrs = get_dataset_attribute_columns(name)
        dims = get_dataset_image_dimensions(name)
        if dims:
            w_vals = [d[0] for d in dims]
            h_vals = [d[1] for d in dims]
            unique_dims = {f"{w}x{h}" for w, h in zip(w_vals, h_vals)}
            dim_str = ", ".join(sorted(unique_dims))
        else:
            dim_str = "—"
        attr_str = ", ".join(attrs) if attrs else "(none)"
        rows.append({
            "Dataset": name,
            "Images": count,
            "Dimensions": dim_str,
            "Attributes": attr_str,
            "Aligned": "Yes" if aligned else "No",
        })

    df = __import__("pandas").DataFrame(rows)
    st.dataframe(df, use_container_width=True)

    st.markdown("---")

    # Detail per dataset
    selected = st.selectbox("Dataset", [r["Dataset"] for r in rows], key="datasets_select")

    cols = st.columns([1, 3])
    with cols[0]:
        sample = get_dataset_sample_images(selected)
        if sample:
            for img in sample:
                st.image(img, width=200)
        else:
            st.caption("No sample images available.")

    with cols[1]:
        count = get_dataset_image_count(selected)
        attrs = get_dataset_attribute_columns(selected)
        dims = get_dataset_image_dimensions(selected)
        st.write(f"**Images:** {count}  |  **Dimensions:** {rows[[r['Dataset'] for r in rows].index(selected)]['Dimensions']}")
        st.write(f"**Aligned:** {rows[[r['Dataset'] for r in rows].index(selected)]['Aligned']}  |  **Attributes:** {rows[[r['Dataset'] for r in rows].index(selected)]['Attributes']}")
        st.caption(f"Sample files: {' / '.join(img.name for img in sample[:3]) if sample else '—'}")
