"""Unsupervised visualization of identity embeddings before/after de-identification.

Uses DeepFace (or another embedding model) to extract embeddings from aligned
and de-identified images, then applies t-SNE/UMAP to project them into 2D
for visual inspection of cluster separation.

If label CSVs exist (gender, expression, ethnicity, etc.), they are used
to color-code the points so you can see whether identity clusters remain
compacted or spread out after de-identification.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def _load_labels(labels_dir: Path, dataset_name: str) -> Optional[pd.DataFrame]:
    """Load a per-dataset label CSV if it exists.

    Looks for ``{dataset_name}_labels.csv`` in ``labels_dir``.
    Returns a DataFrame with a ``Name`` column (image filename) and
    any number of metadata columns (Gender, Expression, Ethnicity, etc.).
    """
    label_file = labels_dir / f"{dataset_name}_labels.csv"
    if not label_file.exists():
        return None
    df = pd.read_csv(label_file)
    if "Name" not in df.columns:
        # Try common column names
        name_col = next((c for c in df.columns if c.lower() in {"name", "filename", "file_name", "image"}), None)
        if name_col:
            df = df.rename(columns={name_col: "Name"})
    return df


def _extract_embeddings(
    image_dir: Path,
    labels_df: Optional[pd.DataFrame],
    n_images: int = 500,
    seed: int = 42,
) -> tuple[np.ndarray, list[str], Optional[pd.DataFrame]]:
    """Extract DeepFace embeddings for images in ``image_dir``.

    Returns:
        embeddings: (n_images, 512) array of DeepFace embeddings
        names: list of image filenames
        subset_labels: the rows from labels_df matching these images (same order)
    """
    import cv2
    from deepface import DeepFace

    image_files = sorted(
        [f for f in image_dir.iterdir() if f.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    )

    if len(image_files) > n_images:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(image_files), n_images, replace=False)
        image_files = [image_files[i] for i in idx]

    names = [f.name for f in image_files]
    embeddings = []

    for img_path in image_files:
        try:
            emb = DeepFace.represent(
                str(img_path),
                model_name="VGG-Face",
                detector_backend="mtcnn",
                enforce_detection=False,
            )
            # DeepFace returns a list of embeddings (one per detected face)
            if emb:
                embeddings.append(emb[0]["embedding"])
        except Exception:
            pass  # skip images without faces

    embeddings = np.array(embeddings, dtype=np.float32)
    # Filter out images that failed embedding extraction
    valid = np.arange(len(names))[: len(embeddings)]
    names = [names[i] for i in valid]

    subset_labels = None
    if labels_df is not None:
        # Match by name
        subset_labels = labels_df[labels_df["Name"].isin(names)].copy()
        # Reorder to match embeddings order
        subset_labels = subset_labels.set_index("Name").loc[names].reset_index()

    return embeddings, names, subset_labels


def project_embeddings(
    embeddings_original: np.ndarray,
    embeddings_deid: np.ndarray,
    labels: Optional[pd.DataFrame],
    method: str = "tsne",
    n_components: int = 2,
    perplexity: float = 30.0,
    n_iter: int = 1000,
) -> dict[str, np.ndarray]:
    """Project both original and de-identified embeddings into 2D.

    Returns a dict with keys ``"original"`` and ``"deid"`` containing
    (n, 2) projection arrays. If labels exist, also returns ``"color"``
    with the column name used for coloring.
    """
    try:
        from sklearn.manifold import TSNE
        projector = TSNE(n_components=n_components, perplexity=perplexity, n_iter=n_iter, random_state=42)
    except ImportError:
        TSNE = None
        try:
            from umap import UMAP
            projector = UMAP(n_components=n_components, random_state=42, n_neighbors=15)
        except ImportError:
            raise ImportError("Install scikit-learn or umap-learn: pip install scikit-learn umap-learn")

    # Concatenate for joint projection
    combined = np.vstack([embeddings_original, embeddings_deid])
    projection = projector.fit_transform(combined)
    n = len(embeddings_original)

    result = {"original": projection[:n], "deid": projection[n:]}
    return result


def compute_reid_risk(
    embeddings_original: np.ndarray,
    embeddings_deid: np.ndarray,
    threshold: float = 0.4,
) -> dict:
    """Compute re-identification risk metrics.

    For each de-identified image, find the closest original image by
    cosine similarity. If the closest match has similarity above
    the threshold, the de-identified image is considered
    re-identifiable.

    Returns:
        A dict with keys:
            - reid_rate: fraction of de-identified images that are re-identifiable
            - mean_similarity: mean similarity to closest original
            - max_similarity: max similarity to closest original
            - histogram: (n_bins,) array of distances
    """
    from sklearn.metrics.pairwise import cosine_similarity

    sim_matrix = cosine_similarity(embeddings_deid, embeddings_original)  # (n_deid, n_orig)
    max_sim = sim_matrix.max(axis=1)
    mean_sim = sim_matrix.mean(axis=1)

    reid_rate = float((max_sim >= threshold).mean())
    max_sim_val = float(max_sim.max())
    mean_sim_val = float(max_sim.mean())

    # Histogram of max similarities
    hist, bin_edges = np.histogram(max_sim, bins=20, range=(0, 1))

    return {
        "reid_rate": reid_rate,
        "mean_similarity": mean_sim_val,
        "max_similarity": max_sim_val,
        "histogram": hist.tolist(),
        "bin_edges": bin_edges.tolist(),
    }


def plot_before_after(
    projection: dict[str, np.ndarray],
    labels: Optional[pd.DataFrame],
    color_column: Optional[str] = None,
    output_dir: Optional[Path] = None,
) -> None:
    """Plot 2D embedding projections side-by-side for original vs de-identified.

    If labels are provided, points are colored by the specified metadata column.
    """
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Determine color mapping
    color_col = color_column
    if color_col is None and labels is not None:
        # Auto-select first useful column
        for col in ["Gender", "Expression", "Ethnicity", "Race", "Age", "Sex"]:
            if col in labels.columns and labels[col].notna().any():
                color_col = col
                break

    cmap = "viridis"
    if color_col:
        unique_vals = labels[color_col].dropna().unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_vals)))
        color_map = dict(zip(unique_vals, colors))
    else:
        colors = None

    # Plot original
    scatter1 = ax1.scatter(
        projection["original"][:, 0],
        projection["original"][:, 1],
        c=[color_map.get(labels.iloc[i][color_col], None) for i in range(len(projection["original"]))]
        if color_col
        else "C0",
        alpha=0.6,
        s=20,
    )
    ax1.set_title("Original — 2D Embeddings")
    ax1.set_xlabel("t-SNE/UMAP dim 1")
    ax1.set_ylabel("t-SNE/UMAP dim 2")

    # Plot de-identified
    scatter2 = ax2.scatter(
        projection["deid"][:, 0],
        projection["deid"][:, 1],
        c=[color_map.get(labels.iloc[i][color_col], None) for i in range(len(projection["deid"]))]
        if color_col
        else "C1",
        alpha=0.6,
        s=20,
    )
    ax2.set_title("De-identified — 2D Embeddings")
    ax2.set_xlabel("t-SNE/UMAP dim 1")
    ax2.set_ylabel("t-SNE/UMAP dim 2")

    if color_col:
        legend_elements = [
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color_map[v], markersize=6)
            for v in labels[color_col].dropna().unique()
        ]
        fig.legend(legend_elements, [str(v) for v in labels[color_col].dropna().unique()], loc="lower center", ncol=len(legend_elements))

    plt.tight_layout()

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / "embedding_projection_before_after.pdf", dpi=150, format="pdf")
        plt.savefig(output_dir / "embedding_projection_before_after.png", dpi=150)
    plt.show()


def plot_reid_risk(risk: dict, output_dir: Optional[Path] = None) -> None:
    """Plot re-identification risk histogram."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        (np.array(risk["bin_edges"])[:-1] + np.array(risk["bin_edges"])[1:]) / 2,
        risk["histogram"],
        width=np.diff(risk["bin_edges"]),
        alpha=0.7,
        edgecolor="black",
    )
    ax.axvline(0.4, color="red", linestyle="--", label="Threshold (0.4)")
    ax.set_title(f"Re-identification Risk — Distribution of Max Similarity\n"
                 f"Re-ID Rate: {risk['reid_rate']:.1%}")
    ax.set_xlabel("Max cosine similarity to closest original")
    ax.set_ylabel("Count")
    ax.legend()
    plt.tight_layout()

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / "reid_risk_histogram.pdf", dpi=150, format="pdf")
        plt.savefig(output_dir / "reid_risk_histogram.png", dpi=150)
    plt.show()
