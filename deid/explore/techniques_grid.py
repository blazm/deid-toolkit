"""Technique browser: shows all technique+dataset combos including completed ones."""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from deid.explore.data_loader import get_loader, list_aligned_images, list_deid_images


def render() -> None:
    st.subheader("Technique Gallery")
    st.caption("All technique+dataset combinations — including those with results but no script.")

    loader = get_loader()
    manifest = loader.get_result_manifest()
    results = loader.list_results()

    if not manifest and not results:
        st.info("No pipeline results found. Run the pipeline first to populate this view.")
        return

    # Collect all technique+dataset combos from both manifest and results
    all_techs: set[str] = set()
    all_datasets: set[str] = set()

    if manifest:
        for tech in manifest.get("techniques", []):
            all_techs.add(tech)
        for ds in manifest.get("datasets", []):
            all_datasets.add(ds)

    for ds_name in results:
        for tech_name in results[ds_name]:
            all_datasets.add(ds_name)
            all_techs.add(tech_name)

    if not all_techs or not all_datasets:
        st.info("No technique+dataset combinations found.")
        return

    # Build lookup: tech -> dataset -> has_results, has_script
    status_lookup: dict[str, dict[str, dict]] = {}
    for tech_name in all_techs:
        status_lookup[tech_name] = {}
        for ds_name in all_datasets:
            has_results = tech_name in results.get(ds_name, {}) and bool(results[ds_name][tech_name])
            tech_script = Path(loader.settings.techniques_path) / f"{tech_name}.py"
            has_script = tech_script.exists()
            # Also check built-in scripts
            try:
                from importlib import resources
                builtin = resources.files("deid.techniques") / f"{tech_name}.py"
                has_script = has_script or bool(builtin.exists())
            except Exception:
                pass
            status_lookup[tech_name][ds_name] = {
                "has_results": has_results,
                "has_script": has_script,
            }

    # Show grid
    sorted_techs = sorted(all_techs)
    sorted_ds = sorted(all_datasets)

    for tech in sorted_techs:
        with st.expander(tech, expanded=False):
            for ds in sorted_ds:
                status = status_lookup[tech][ds]
                if not status["has_results"] and not status["has_script"]:
                    continue  # skip nothing found

                icon = ""
                label = ""
                if status["has_results"] and status["has_script"]:
                    icon = "✅"
                    label = "Available"
                elif status["has_results"] and not status["has_script"]:
                    icon = "\U0001f4c1"
                    label = "Completed (script no longer available)"
                elif not status["has_results"] and status["has_script"]:
                    icon = "\U0001f4e6"
                    label = "Available (no results)"

                st.markdown(f"- {icon} **{ds}** — {label}")

                # Show aligned images preview if available
                aligned = list_aligned_images(ds)
                if aligned:
                    with st.expander("Preview aligned images", expanded=False):
                        cols = st.columns(min(4, len(aligned)))
                        for i, img in enumerate(aligned[:20]):  # max 20 previews
                            with cols[i % 4]:
                                st.image(str(img), use_container_width=True)
