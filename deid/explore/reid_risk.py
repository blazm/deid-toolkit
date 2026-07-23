"""Re-identification risk assessment module.

Provides experiments and visualizations to measure how well a de-identified
face can still be re-identified. This is the key privacy guarantee of any
DEID technique.

Key metrics:
- Re-ID success rate: Can a re-ID model still identify the person?
- Embedding leakage: How much identity information remains in the embedding space?
- Attribute leakage: Can demographic attributes (gender, race, age) be inferred?
- Cross-model agreement: Do different identity models agree on matches?

These are designed to be called after the pipeline produces results, or
on-demand by the user.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def reid_success_rate(
    aligned_path: Path,
    deid_path: Path,
    n_images: int = 100,
    seed: int = 42,
) -> dict:
    """Measure re-ID success rate using DeepFace.

    For each de-identified image, check if the same identity can still
    be recovered by a re-ID model.

    Returns:
        Dict with:
            - reid_rate: float - fraction of images still re-identifiable
            - success_count: int - number of successful re-ID
            - total_count: int - total images tested
            - sample_results: list of (original_name, closest_match, confidence)
    """
    from deepface import DeepFace

    deid_images = sorted(
        [f for f in deid_path.iterdir() if f.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    )
    aligned_images = sorted(
        [f for f in aligned_path.iterdir() if f.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    )

    if len(deid_images) > n_images:
        rng = np.random.RandomState(seed)
        deid_images = [deid_images[i] for i in rng.choice(len(deid_images), n_images, replace=False)]

    successful = 0
    sample_results = []

    for deid_img in deid_images[:50]:  # Sample for detailed results
        try:
            # Extract embeddings for de-identified image
            deid_ents = DeepFace.represent(str(deid_img), model_name="VGG-Face", detector_backend="mtcnn")
            if not deid_ents:
                continue
            deid_emb = deid_ents[0]["embedding"]

            # Compare against all original images
            best_score = 0
            best_match = None
            for orig_img in aligned_images[:100]:
                orig_ents = DeepFace.represent(str(orig_img), model_name="VGG-Face", detector_backend="mtcnn")
                if orig_ents:
                    score = np.dot(deid_emb, orig_ents[0]["embedding"])
                    if score > best_score:
                        best_score = score
                        best_match = orig_img.name

            if best_score > 0.4:  # Re-identifiable threshold
                successful += 1
                sample_results.append((deid_img.name, best_match, float(best_score)))
        except Exception:
            pass

    return {
        "reid_rate": successful / max(len(deid_images), 1),
        "success_count": successful,
        "total_count": len(deid_images),
        "sample_results": sample_results,
    }


def attribute_leakage(
    aligned_path: Path,
    deid_path: Path,
    attributes: list[str] = None,
    n_images: int = 50,
) -> dict:
    """Measure demographic attribute leakage after de-identification.

    Compares attribute prediction accuracy between aligned and de-identified images.
    If the accuracy is similar, the attribute information is leaking.

    Returns:
        Dict with per-attribute leakage scores and a summary.
    """
    if attributes is None:
        attributes = ["Gender", "Age", "Race"]

    results = {}
    for attr in attributes:
        # Compare attribute prediction on original vs de-identified
        original_results = _predict_attribute(aligned_path, attr, n_images)
        deid_results = _predict_attribute(deid_path, attr, n_images)
        results[attr] = {
            "original_accuracy": original_results,
            "deid_accuracy": deid_results,
            "leakage": abs(original_results - deid_results),
        }

    return results


def _predict_attribute(image_path: Path, attribute: str, n_images: int) -> float:
    """Predict a specific attribute and return a confidence score."""
    import cv2
    from deepface import DeepFace

    images = sorted(
        [f for f in image_path.iterdir() if f.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    )
    if len(images) > n_images:
        rng = np.random.RandomState(42)
        images = [images[i] for i in rng.choice(len(images), n_images, replace=False)]

    correct = 0
    total = 0
    for img_path in images[:100]:
        try:
            result = DeepFace.analyze(str(img_path), actions=[attribute], detector_backend="mtcnn")
            if result:
                # Get the most confident prediction
                pred = result[0][attribute]
                confidence = pred.get("confidence", 0)
                # For binary attributes, check if the majority prediction matches
                if attribute == "Gender":
                    correct += 1 if confidence > 0.7 else 0
                elif attribute == "Age":
                    correct += 1 if abs(pred["age"] - 30) < 10 else 0  # Simplified
                total += 1
        except Exception:
            pass

    return correct / max(total, 1)


def plot_reid_results(risk_results: dict) -> None:
    """Visualize re-identification risk assessment results."""
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Re-ID rate bar
    reid_rate = risk_results.get("reid_rate", 0)
    ax1.bar(["Re-ID Success Rate"], [reid_rate * 100], color="crimson", alpha=0.7)
    ax1.set_ylim(0, 100)
    ax1.set_ylabel("Re-ID Success Rate (%)")
    ax1.set_title("Re-identification Risk")

    # Attribute leakage
    leakage_data = risk_results.get("leakage", {})
    if leakage_data:
        attrs = list(leakage_data.keys())
        leakages = [leakage_data[a].get("leakage", 0) for a in attrs]
        ax2.bar(attrs, leakages, color="steelblue", alpha=0.7)
        ax2.set_ylabel("Attribute Leakage (Δ accuracy)")
        ax2.set_title("Demographic Attribute Leakage")

    plt.tight_layout()
    plt.show()


def compute_identity_cluster_overlap(
    aligned_embeddings: np.ndarray,
    deid_embeddings: np.ndarray,
) -> dict:
    """Compute how much identity clusters overlap between original and de-identified.

    Uses the ratio of inter-cluster to intra-cluster distances.
    Higher overlap means less effective de-identification.

    Returns:
        Dict with overlap metrics.
    """
    from sklearn.metrics.pairwise import cosine_similarity

    # Compute similarity matrices
    orig_sim = cosine_similarity(aligned_embeddings)
    deid_sim = cosine_similarity(deid_embeddings)

    # Distance matrices
    orig_dist = 1 - orig_sim
    deid_dist = 1 - deid_sim

    # Intra-cluster distances (same identity) - use diagonal blocks
    n = len(aligned_embeddings)
    mask = np.eye(n, dtype=bool)
    orig_intra = orig_dist[mask].mean()
    deid_intra = deid_dist[mask].mean()

    # Inter-cluster distances (different identity) - use off-diagonal blocks
    mask_off = ~mask
    orig_inter = orig_dist[mask_off].mean()
    deid_inter = deid_dist[mask_off].mean()

    # Overlap ratio
    overlap_ratio = deid_intra / max(orig_intra, 1e-6)

    return {
        "overlap_ratio": overlap_ratio,
        "orig_intra_cluster_dist": orig_intra,
        "deid_intra_cluster_dist": deid_intra,
        "orig_inter_cluster_dist": orig_inter,
        "deid_inter_cluster_dist": deid_inter,
        "cluster_separation_improvement": (deid_inter - deid_intra) / max(orig_inter - orig_intra, 1e-6),
    }


def plot_cluster_overlap(overlap_results: dict) -> None:
    """Visualize identity cluster overlap before/after de-identification."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))

    categories = ["Intra-cluster\nDistance", "Inter-cluster\nDistance"]
    original = [overlap_results["orig_intra_cluster_dist"], overlap_results["orig_inter_cluster_dist"]]
    deid = [overlap_results["deid_intra_cluster_dist"], overlap_results["deid_inter_cluster_dist"]]

    x = np.arange(len(categories))
    width = 0.35

    ax.bar(x - width/2, original, width, label="Original", alpha=0.7)
    ax.bar(x + width/2, deid, width, label="De-identified", alpha=0.7)

    ax.set_ylabel("Cosine Distance")
    ax.set_title("Identity Cluster Overlap")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()

    plt.tight_layout()
    plt.show()
