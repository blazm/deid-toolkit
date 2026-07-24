"""Embedding space analysis for face de-identification.

Loads cached embeddings from evaluation scripts, computes joint 2D projections,
and provides displacement/collapse/comparison metrics and visualizations.

Three core analyses:
1. **Displacement** – per-image arrows in embedding space (original → deid)
2. **Identity Collapse** – do distinct identities merge after de-identification?
3. **Technique Comparison** – overlay multiple techniques in shared projection
"""
from __future__ import annotations

import pickle
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class EmbeddingRecord:
    """One paired embedding record for a single image."""
    image_name: str          # e.g. "KlemenGrm.jpg"
    identity: str | None     # derived from filename stem or label CSV
    original: np.ndarray     # (dim,) numpy float32
    deid: np.ndarray         # (dim,) numpy float32


class EmbeddingCacheNotFoundError(Exception):
    """Raised when no cached embeddings exist for the requested model/dataset/technique."""
    pass


# ---------------------------------------------------------------------------
# Cache discovery and loading
# ---------------------------------------------------------------------------

# Model name → cache subdirectory pattern under root_dir/preprocess/temp/
# All models: {model}/{dataset}/original/ (shared) + {dataset}/deid/{technique}/ (per technique)

_MODEL_CACHE_CONFIG = {
    "adaface": {
        "original_key": "{ds}/original",      # Shared originals (same as swinface)
        "deid_key": "{ds}/deid/{tech}",       # Per technique: fri/deid/blur/
        "format": "torch_tensor",             # torch.Tensor saved via .cpu(), pickle
    },
    "swinface": {
        "original_key": "{ds}/original",      # Shared originals
        "deid_key": "{ds}/deid/{tech}",       # Per technique: celeba-test_aligned/deid/blur/
        "format": "torch_dict",               # dict[str, torch.Tensor], key="Recognition"
    },
    "deepface_vggface": {
        "original_key": "{ds}/original",      # Shared originals
        "deid_key": "{ds}/deid/{tech}",       # Per technique
        "format": "numpy",                    # raw numpy.ndarray
    },
}


def discover_embedding_models(root_dir: str) -> list[str]:
    """Return list of embedding models with cached data under root_dir."""
    temp_dir = Path(root_dir) / "preprocess" / "temp"
    if not temp_dir.is_dir():
        return []

    found = []
    for model_name in _MODEL_CACHE_CONFIG:
        model_dir = temp_dir / model_name
        if model_dir.is_dir() and any(model_dir.rglob("*.pkl")):
            found.append(model_name)
    return sorted(found)


def _normalize_embedding(raw_object: Any, fmt: str) -> np.ndarray:
    """Convert a cached pickle object to (dim,) numpy float32."""
    if fmt == "numpy":
        # Already numpy or list
        arr = np.asarray(raw_object, dtype=np.float32)
        return arr.flatten()

    if fmt in ("torch_tensor", "torch_dict"):
        import torch

        if fmt == "torch_dict" and isinstance(raw_object, dict):
            # SWINFace: {"Recognition": tensor, ...}
            raw_object = raw_object.get("Recognition", raw_object)

        if isinstance(raw_object, torch.Tensor):
            tensor = raw_object.detach().cpu()
            return tensor.numpy().astype(np.float32).flatten()

        # Fallback: try numpy conversion
        return np.asarray(raw_object, dtype=np.float32).flatten()

    # Last resort
    return np.asarray(raw_object, dtype=np.float32).flatten()


def _load_pkl_embeddings(cache_dir: Path, fmt: str) -> dict[str, np.ndarray]:
    """Load all .pkl files in a directory. Returns {filename_stem: embedding}."""
    if not cache_dir.is_dir():
        return {}

    embeddings = {}
    for pkl_file in sorted(cache_dir.glob("*.pkl")):
        fname = pkl_file.name  # e.g. "001_ang_take000_img_0034.jpg.pkl"
        img_name = fname.rsplit(".pkl", 1)[0]  # strip .pkl suffix

        try:
            with open(pkl_file, "rb") as f:
                raw = pickle.load(f)
            emb = _normalize_embedding(raw, fmt)
            embeddings[img_name] = emb
        except Exception as exc:
            logger.warning("Failed to load %s: %s", pkl_file.name, exc)

    return embeddings


def find_cache_dirs(
    root_dir: str,
    model_name: str,
    dataset: str,
    technique: str,
) -> tuple[Path, Path]:
    """Return (original_cache_dir, deid_cache_dir) for a model/dataset/technique.

    Raises EmbeddingCacheNotFoundError if no .pkl files found in either dir.
    """
    config = _MODEL_CACHE_CONFIG[model_name]
    temp_root = Path(root_dir) / "preprocess" / "temp" / model_name

    ds = dataset
    tech = technique
    orig_key = config["original_key"].format(ds=ds, tech=tech)
    deid_key = config["deid_key"].format(ds=ds, tech=tech)

    # All models share the same cache layout: {ds}/original/ + {ds}/deid/{tech}/
    orig_dir = temp_root / orig_key   # already "{ds}/original"
    deid_dir = temp_root / deid_key   # already "{ds}/deid/{tech}"

    has_orig = (orig_dir.is_dir() and any(orig_dir.glob("*.pkl")))
    has_deid = (deid_dir.is_dir() and any(deid_dir.glob("*.pkl")))

    if not has_orig and not has_deid:
        raise EmbeddingCacheNotFoundError(
            f"No cached embeddings for {model_name}/{dataset}/{technique}.\n"
            f"  Searched: {orig_dir}, {deid_dir}\n"
            f"  Run the evaluation pipeline first (e.g. 'deid run evaluation')."
        )

    return orig_dir, deid_dir


def load_paired_embeddings(
    root_dir: str,
    model_name: str,
    dataset: str,
    technique: str,
    labels_df: Optional[pd.DataFrame] = None,
) -> list[EmbeddingRecord]:
    """Load paired (original, deid) embeddings from cache for one technique.

    Returns list of EmbeddingRecord sorted by image_name.
    Only includes images present in BOTH original and deid caches.
    """
    config = _MODEL_CACHE_CONFIG[model_name]
    orig_dir, deid_dir = find_cache_dirs(root_dir, model_name, dataset, technique)

    orig_embs = _load_pkl_embeddings(orig_dir, config["format"])
    deid_embs = _load_pkl_embeddings(deid_dir, config["format"])

    # Match by image name (present in both caches)
    common_names = sorted(set(orig_embs.keys()) & set(deid_embs.keys()))

    if not common_names:
        logger.warning(
            "No paired embeddings found for %s/%s/%s (orig=%d, deid=%d files)",
            model_name, dataset, technique, len(orig_embs), len(deid_embs),
        )
        return []

    # Load identity labels if available
    label_map: dict[str, dict[str, Any]] = {}
    if labels_df is not None and "Name" in labels_df.columns:
        for _, row in labels_df.iterrows():
            label_map[row["Name"]] = row.to_dict()

    records = []
    for name in common_names:
        identity = name.rsplit(".", 1)[0]  # stem as default identity
        extra = {}

        if name in label_map:
            lr = label_map[name]
            # Try common identity column names
            for col in ("Identity", "identity", "Person", "person_id"):
                if col in lr and pd.notna(lr[col]):
                    identity = str(lr[col])
                    break
            extra = {k: v for k, v in lr.items() if k not in ("Name", "Identity", "identity")}

        records.append(EmbeddingRecord(
            image_name=name,
            identity=identity,
            original=orig_embs[name],
            deid=deid_embs[name],
        ))

    return records


def load_multi_technique_embeddings(
    root_dir: str,
    model_name: str,
    dataset: str,
    techniques: list[str],
    labels_df: Optional[pd.DataFrame] = None,
) -> dict[str, list[EmbeddingRecord]]:
    """Load embeddings for multiple techniques.

    Returns {technique: [EmbeddingRecord]} dict.
    """
    result = {}
    for tech in techniques:
        try:
            recs = load_paired_embeddings(root_dir, model_name, dataset, tech, labels_df)
            if recs:
                result[tech] = recs
            else:
                logger.warning("No paired embeddings for technique %s", tech)
        except EmbeddingCacheNotFoundError as e:
            logger.warning(str(e))

    return result


# ---------------------------------------------------------------------------
# Label loading
# ---------------------------------------------------------------------------

def load_labels(root_dir: str, dataset: str) -> Optional[pd.DataFrame]:
    """Load label CSV for a dataset if it exists."""
    labels_dir = Path(root_dir) / "datasets" / "labels"
    label_file = labels_dir / f"{dataset}_labels.csv"

    if not label_file.exists():
        return None

    try:
        df = pd.read_csv(label_file)
        # Normalize column name for image filename
        if "Name" not in df.columns:
            for col in ("filename", "file_name", "image"):
                if col in df.columns:
                    df = df.rename(columns={col: "Name"})
                    break
        return df
    except Exception as exc:
        logger.warning("Failed to load labels for %s: %s", dataset, exc)
        return None


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------

def project_embeddings_joint(
    embedding_sets: dict[str, np.ndarray],
    method: str = "umap",
    n_components: int = 2,
    perplexity: float = 30.0,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Project multiple embedding sets to shared 2D space via joint UMAP/t-SNE/PCA.

    Args:
        embedding_sets: {label: (N_i, dim)} dict of embedding arrays
        method: "umap", "tsne", or "pca"
        perplexity: t-SNE perplexity
        n_neighbors: UMAP neighbor count
        min_dist: UMAP minimum distance
        seed: random state seed

    Returns:
        {label: (N_i, 2)} dict of 2D projected arrays
    """
    # Deduplicate: only project unique embeddings once.
    # This ensures identical vectors get identical 2D positions across techniques.
    labels = sorted(embedding_sets.keys())
    counts = [len(embedding_sets[l]) for l in labels]
    combined = np.vstack([embedding_sets[l] for l in labels]).astype(np.float32)

    total = len(combined)

    # Find unique rows + inverse mapping so duplicates share the same projection
    _, unique_idx, inv_idx = np.unique(combined, axis=0, return_index=True, return_inverse=True)
    unique_combined = combined[unique_idx]
    n_unique = len(unique_combined)

    if method == "pca":
        from sklearn.decomposition import PCA

        projections_unique = PCA(
            n_components=n_components,
            random_state=seed,
        ).fit_transform(unique_combined)

    elif method == "tsne":
        from sklearn.manifold import TSNE

        # t-SNE is O(n^2); subsample if needed
        max_tsne = 3000
        if n_unique > max_tsne:
            rng = np.random.RandomState(seed)
            sub_idx = rng.choice(n_unique, max_tsne, replace=False)
            sub_idx.sort()
            proj_full = np.zeros((n_unique, n_components), dtype=np.float32)
            proj_result = TSNE(
                n_components=n_components,
                perplexity=min(perplexity, max_tsne // 2 - 1),
                random_state=seed,
                n_iter=1000,
            ).fit_transform(unique_combined[sub_idx])
            # Map back — non-subsampled rows get nearest-neighbor projection
            proj_full[sub_idx] = proj_result
            remaining = np.ones(n_unique, dtype=bool)
            remaining[sub_idx] = False
            if remaining.any():
                from sklearn.neighbors import NearestNeighbors
                nn = NearestNeighbors(n_neighbors=1)
                nn.fit(unique_combined[sub_idx])
                _, rem_map = nn.kneighbors(unique_combined[remaining])
                proj_full[remaining] = proj_result[rem_map.ravel()]
            projections_unique = proj_full
        else:
            projections_unique = TSNE(
                n_components=n_components,
                perplexity=min(perplexity, n_unique // 2 - 1),
                random_state=seed,
                n_iter=1000,
            ).fit_transform(unique_combined)

    else:
        # UMAP — handles large N efficiently
        try:
            from umap import UMAP
        except ImportError:
            raise ImportError(
                "UMAP required for projection. Install: pip install umap-learn"
            )

        projections_unique = UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=seed,
        ).fit_transform(unique_combined)

    # Expand back to full size using inverse mapping (duplicates share projection)
    projections_full = projections_unique[inv_idx]

    # Split back by original counts
    result = {}
    start = 0
    for label, count in zip(labels, counts):
        end = start + count
        result[label] = projections_full[start:end]
        start = end

    return result


# ---------------------------------------------------------------------------
# Displacement analysis
# ---------------------------------------------------------------------------

def compute_displacements(
    orig_xy: np.ndarray,  # (N, 2)
    deid_xy: np.ndarray,  # (N, 2)
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-image displacement vectors and magnitudes.

    Returns (vectors[N,2], magnitudes[N]).
    """
    vectors = deid_xy - orig_xy
    magnitudes = np.linalg.norm(vectors, axis=1)
    return vectors, magnitudes


# ---------------------------------------------------------------------------
# Identity collapse analysis
# ---------------------------------------------------------------------------

def compute_identity_dispersion(
    original_embs: dict[str, np.ndarray],  # {image_name: embedding}
    deid_embs: dict[str, np.ndarray],      # {image_name: embedding}
    identity_map: dict[str, str],          # {image_name: identity_label}
) -> pd.DataFrame:
    """Compute per-identity dispersion metrics before and after de-identification.

    Dispersion = mean pairwise cosine distance within an identity group.
    collapse_ratio = deid_dispersion / orig_dispersion.

    Returns DataFrame with columns:
        identity, n_images, dispersion_before, dispersion_after,
        collapse_ratio, mean_displacement_norm
    """
    identities = sorted(set(identity_map.values()))
    rows = []

    for ident in identities:
        # Get images belonging to this identity
        img_names = [n for n, i in identity_map.items() if i == ident]
        img_names = [n for n in img_names if n in original_embs and n in deid_embs]

        if len(img_names) < 2:
            continue

        orig_group = np.array([original_embs[n] for n in img_names])
        deid_group = np.array([deid_embs[n] for n in img_names])

        # Mean pairwise cosine distance (1 - cosine similarity)
        def mean_pairwise_cosine_dist(embs):
            norms = np.linalg.norm(embs, axis=1, keepdims=True)
            norms = np.where(norms < 1e-10, 1e-10, norms)
            normalized = embs / norms
            sim_matrix = normalized @ normalized.T
            # Exclude diagonal
            n = len(embs)
            mask = ~np.eye(n, dtype=bool)
            return float(1 - sim_matrix[mask].mean())

        orig_disp = mean_pairwise_cosine_dist(orig_group)
        deid_disp = mean_pairwise_cosine_dist(deid_group)
        collapse_ratio = deid_disp / max(orig_disp, 1e-6)

        # Mean displacement (in original embedding space)
        displacements = []
        for n in img_names:
            diff = deid_embs[n] - original_embs[n]
            norm_o = np.linalg.norm(original_embs[n])
            norm_d = np.linalg.norm(deid_embs[n])
            if norm_o > 1e-10 and norm_d > 1e-10:
                displacements.append(np.linalg.norm(diff))
        mean_disp = float(np.mean(displacements)) if displacements else 0.0

        rows.append({
            "identity": ident,
            "n_images": len(img_names),
            "dispersion_before": round(orig_disp, 4),
            "dispersion_after": round(deid_disp, 4),
            "collapse_ratio": round(collapse_ratio, 4),
            "mean_displacement_norm": round(mean_disp, 4),
        })

    if not rows:
        return pd.DataFrame(columns=[
            "identity", "n_images", "dispersion_before", "dispersion_after",
            "collapse_ratio", "mean_displacement_norm"
        ])

    return pd.DataFrame(rows).sort_values("collapse_ratio").reset_index(drop=True)


def compute_technique_summary(
    records: list[EmbeddingRecord],
) -> dict[str, float]:
    """Compute summary statistics for a technique's embeddings.

    Returns dict with mean/cosine_similarity_drop, avg_displacement, etc.
    """
    if not records:
        return {}

    orig = np.array([r.original for r in records])
    deid = np.array([r.deid for r in records])

    # Mean cosine similarity between original and deid (same image)
    norms_o = np.linalg.norm(orig, axis=1, keepdims=True)
    norms_d = np.linalg.norm(deid, axis=1, keepdims=True)
    norms_o = np.where(norms_o < 1e-10, 1e-10, norms_o)
    norms_d = np.where(norms_d < 1e-10, 1e-10, norms_d)

    cos_sim = (orig * deid).sum(axis=1) / (norms_o.squeeze() * norms_d.squeeze())

    # Mean Euclidean displacement
    euclidean = np.linalg.norm(orig - deid, axis=1)

    return {
        "n_images": len(records),
        "mean_cosine_similarity": float(cos_sim.mean()),
        "std_cosine_similarity": float(cos_sim.std()),
        "min_cosine_similarity": float(cos_sim.min()),
        "max_cosine_similarity": float(cos_sim.max()),
        "mean_euclidean_displacement": float(euclidean.mean()),
        "std_euclidean_displacement": float(euclidean.std()),
    }


# ---------------------------------------------------------------------------
# Build data for all 3 visualizations
# ---------------------------------------------------------------------------

def prepare_displacement_data(
    root_dir: str,
    model_name: str,
    dataset: str,
    technique: str,
    labels_df: Optional[pd.DataFrame] = None,
    projection_method: str = "umap",
) -> dict:
    """Prepare all data needed for the displacement plot.

    Returns dict with keys:
        records, orig_xy, deid_xy, vectors, magnitudes,
        image_names, identities, labels_df
    """
    if labels_df is None:
        labels_df = load_labels(root_dir, dataset)
    records = load_paired_embeddings(root_dir, model_name, dataset, technique, labels_df)

    if not records:
        return {"error": "No paired embeddings found."}

    # Build embedding sets for joint projection
    orig_embs = np.array([r.original for r in records])
    deid_embs = np.array([r.deid for r in records])

    em_set = {"original": orig_embs, "deid": deid_embs}
    projections = project_embeddings_joint(em_set, method=projection_method)

    orig_xy = projections["original"]
    deid_xy = projections["deid"]

    # 2D displacement vectors from UMAP projection (for direction visualization)
    vectors_2d, magnitudes_2d = compute_displacements(orig_xy, deid_xy)

    # Raw embedding-space displacement magnitudes (the metric that matters)
    raw_euclidean = np.linalg.norm(orig_embs - deid_embs, axis=1)
    norms_o = np.linalg.norm(orig_embs, axis=1, keepdims=True)
    norms_d = np.linalg.norm(deid_embs, axis=1, keepdims=True)
    norms_o = np.where(norms_o < 1e-10, 1e-10, norms_o)
    norms_d = np.where(norms_d < 1e-10, 1e-10, norms_d)
    cos_sim = (orig_embs * deid_embs).sum(axis=1) / (norms_o.squeeze() * norms_d.squeeze())

    return {
        "records": records,
        "orig_xy": orig_xy,
        "deid_xy": deid_xy,
        "vectors": vectors_2d,
        "magnitudes": magnitudes_2d,           # 2D UMAP displacement (for arrow scaling)
        "raw_euclidean_displacement": raw_euclidean,  # True embedding-space Euclidean distance
        "cosine_similarity": cos_sim,            # Cosine similarity between orig/deid pairs
        "image_names": [r.image_name for r in records],
        "identities": [r.identity for r in records],
        "labels_df": labels_df,
    }


def prepare_collapse_data(
    root_dir: str,
    model_name: str,
    dataset: str,
    technique: str,
    labels_df: Optional[pd.DataFrame] = None,
) -> dict:
    """Prepare all data needed for the identity collapse analysis.

    Returns dict with keys:
        dispersion_df, orig_embs, deid_embs, identity_map, records, summary
    """
    if labels_df is None:
        labels_df = load_labels(root_dir, dataset)
    records = load_paired_embeddings(root_dir, model_name, dataset, technique, labels_df)

    if not records:
        return {"error": "No paired embeddings found."}

    orig_embs = {r.image_name: r.original for r in records}
    deid_embs = {r.image_name: r.deid for r in records}
    identity_map = {r.image_name: r.identity for r in records}

    dispersion_df = compute_identity_dispersion(orig_embs, deid_embs, identity_map)
    summary = compute_technique_summary(records)

    return {
        "dispersion_df": dispersion_df,
        "orig_embs": orig_embs,
        "deid_embs": deid_embs,
        "identity_map": identity_map,
        "records": records,
        "summary": summary,
    }


def prepare_comparison_data(
    root_dir: str,
    model_name: str,
    dataset: str,
    techniques: list[str],
    labels_df: Optional[pd.DataFrame] = None,
    projection_method: str = "umap",
) -> dict:
    """Prepare all data needed for multi-technique comparison.

    Returns dict with keys:
        orig_xy, deid_xys (dict[tech: xy]), magnitudes (dict[tech: mag]),
        image_names, technique_summaries
    """
    if labels_df is None:
        labels_df = load_labels(root_dir, dataset)
    all_records = load_multi_technique_embeddings(
        root_dir, model_name, dataset, techniques, labels_df
    )

    if not all_records:
        return {"error": "No embeddings found for any technique."}

    # Find common image names across all techniques + originals
    name_sets = [set(r.image_name for r in recs) for recs in all_records.values()]
    common_names = sorted(set.intersection(*name_sets)) if name_sets else []

    if len(common_names) < 3:
        return {"error": f"Too few common images across techniques ({len(common_names)})."}

    # Build aligned arrays (same order for all techniques)
    def _build_arrays(recs_list):
        """From records, extract embs in common_names order."""
        name_to_emb_orig = {r.image_name: r.original for r in recs_list}
        name_to_emb_deid = {r.image_name: r.deid for r in recs_list}
        orig = np.array([name_to_emb_orig[n] for n in common_names])
        deid = np.array([name_to_emb_deid[n] for n in common_names])
        return orig, deid

    # Use first technique's originals (all should have same original cache)
    first_tech = list(all_records.keys())[0]
    first_orig, _ = _build_arrays(all_records[first_tech])
    shared_orig = first_orig  # originals are identical across techniques

    em_set = {"original": shared_orig}
    deid_by_tech = {}

    for tech, recs in all_records.items():
        _, deid_arr = _build_arrays(recs)
        em_set[f"deid_{tech}"] = deid_arr
        deid_by_tech[tech] = deid_arr

    projections = project_embeddings_joint(em_set, method=projection_method)
    orig_xy = projections["original"]

    # Compute displacements per technique
    vectors_by_tech = {}
    magnitudes_by_tech = {}
    for tech in all_records:
        v, m = compute_displacements(orig_xy, projections[f"deid_{tech}"])
        vectors_by_tech[tech] = v
        magnitudes_by_tech[tech] = m

    # Summary per technique
    summaries = {}
    for tech, recs in all_records.items():
        summaries[tech] = compute_technique_summary(recs)

    return {
        "orig_xy": orig_xy,
        "deid_xys": projections,  # includes all deid_* keys
        "vectors_by_tech": vectors_by_tech,
        "magnitudes_by_tech": magnitudes_by_tech,
        "image_names": common_names,
        "techniques": list(all_records.keys()),
        "technique_summaries": summaries,
    }
