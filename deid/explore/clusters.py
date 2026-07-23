"""Clustering visualization: before/after deidentification embedding comparison."""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from deid.explore.data_loader import (
    list_aligned_images,
    list_deid_images,
    get_loader,
)
from deid.explore.viz import plot_embedding_clustering


def render(dataset: str, technique: str) -> None:
    st.subheader(f"Embedding Cluster Comparison — {dataset}")

    if not technique or technique == "(none)":
        st.warning("Select a technique to view clustering.")
        return

    # Check that de-identified images exist
    deid_images = list_deid_images(dataset, technique)
    if not deid_images:
        st.warning("No de-identified images found for this technique/dataset.")
        return

    loader = get_loader()
    aligned_dir = loader.settings.aligned_path / dataset
    deid_dir = loader.settings.deid_path / technique / dataset

    method = st.selectbox(
        "Dimensionality reduction",
        ["umap", "tsne"],
        help="UMAP is faster; t-SNE gives finer local structure.",
    )
    n_neighbors = st.slider("Neighbors (UMAP)", 2, 50, 15, 1) if method == "umap" else 10

    st.info("Computing DeepFace VGG-Face embeddings and projecting to 2D — may take a moment.")

    if st.button("Compute Clusters"):
        try:
            from deepface import DeepFace
            DeepFace.settings.represent_mode = "flatten"

            import numpy as np
            import pandas as pd

            from umap import UMAP  # type: ignore
            from sklearn.manifold import TSNE  # type: ignore

            aligned_list = list_aligned_images(dataset)
            if not aligned_list:
                st.warning("No aligned images found for this dataset.")
                return

            images_to_process: list[tuple[str, Path, str]] = []
            for img in aligned_list:
                aligned_path = aligned_dir / img.name
                deid_path = deid_dir / img.name
                if aligned_path.exists() and deid_path.exists():
                    images_to_process.append((img.name, aligned_path, "original"))
                    images_to_process.append((img.name, deid_path, "deid"))

            n = len(images_to_process)
            progress_bar = st.progress(0, text=f"Computing {n} embeddings...")

            embeddings = []
            identities = []
            sources = []
            for i, (name, img_path, source) in enumerate(images_to_process):
                try:
                    rep = DeepFace.represent(
                        img_path=str(img_path),
                        model_name="VGG-Face",
                        enforce_detection=False,
                        silent=True,
                    )
                    if rep and rep[0].get("embedding"):
                        emb = np.array(rep[0]["embedding"])
                        embeddings.append(emb)
                        identities.append(name.split(".")[0])
                        sources.append(source)
                except Exception:
                    pass
                progress_bar.progress((i + 1) / n)

            if len(embeddings) < 10:
                st.error("Not enough embeddings extracted. Check that aligned and de-identified images exist.")
                return

            embeddings = np.array(embeddings)
            df = pd.DataFrame({
                "x": np.nan,
                "y": np.nan,
                "identity": identities,
                "source": sources,
            })

            st.info("Reducing dimensionality...")
            if method == "umap":
                try:
                    reducer = UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=0.1, random_state=42)
                    reduced = reducer.fit_transform(embeddings)
                except ImportError:
                    st.error("umap must be installed: pip install umap-learn")
                    return
            else:
                try:
                    # subsample for t-SNE if too many points
                    if len(embeddings) > 500:
                        idx = np.random.choice(len(embeddings), 500, replace=False)
                        sub = embeddings[idx]
                        sub_idx = idx
                    else:
                        sub = embeddings
                        sub_idx = np.arange(len(embeddings))
                    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(sub) - 1))
                    reduced_full = np.full((len(embeddings), 2), np.nan)
                    reduced_full[sub_idx] = tsne.fit_transform(sub)
                    reduced = reduced_full
                except ImportError:
                    st.error("sklearn must be installed: pip install scikit-learn")
                    return

            df["x"] = reduced[:, 0]
            df["y"] = reduced[:, 1]

            st.success("Clusters computed.")
            fig = plot_embedding_clustering(df, technique, dataset)
            st.pyplot(fig)

            # Per-cluster silhouette approximation
            n_ids = df["identity"].nunique()
            unique_embed = embeddings
            if n_ids > 1:
                from sklearn.metrics import silhouette_score  # type: ignore
                valid_sources = [s for s in df["source"].unique() if len(df[df["source"] == s]) > 1]
                for src in valid_sources:
                    mask = df["source"] == src
                    if mask.sum() > 1:
                        try:
                            sil = silhouette_score(
                                df.loc[mask, ["x", "y"]],
                                df.loc[mask, "identity"],
                            )
                            st.metric(f"Silhouette Score ({src.title()})", f"{sil:.3f}",
                                      help="Higher = tighter identity clusters (more identifiable).")
                        except (ValueError, ImportError):
                            pass
        except ImportError as e:
            st.error(f"Missing dependency: {e}")
            st.info("Install: pip install umap-learn scikit-learn deepface")
        except Exception as exc:
            st.error(f"Error: {exc}")
