"""VGG-Face identity verification via DeepFace (bypasses broken .t7 loader).

Uses DeepFace's built-in VGG-Face model to extract 4096-d embeddings and compute
cosine similarity for each image pair, matching the standard eval script interface.

Embeddings are cached as .pkl files under root_dir/preprocess/temp/deepface_vggface/
so that:
  - Re-runs skip TF inference entirely (minutes instead of hours).
  - Embeddings are available for downstream visualization (t-SNE, UMAP, clusters).
"""
from __future__ import annotations

import os
import pickle
import numpy as np
from deepface import DeepFace
import utils as util
from tqdm import tqdm

EVALUATION_NAME = "deepface_vggface"


def _get_embedding(image_path: str, cache_dir: str, log_dir: str) -> "np.ndarray | None":
    """Return a VGG-Face embedding, using a pickle cache if available."""
    image_name = os.path.basename(image_path)
    cache_file = os.path.join(cache_dir, f"{image_name}.pkl")

    if os.path.exists(cache_file):
        try:
            with open(cache_file, "rb") as f:
                emb = pickle.load(f)
            return np.array(emb)
        except Exception:
            pass  # stale/corrupt cache — recompute below

    emb = DeepFace.represent(
        img_path=image_path, model_name="VGG-Face", detector_backend="skip"
    )
    feat = emb[0]["embedding"]

    try:
        with open(cache_file, "wb") as f:
            pickle.dump(feat, f)
    except Exception:
        util.log(os.path.join(log_dir, "deepface_vggface.txt"),
                 f"Failed to cache embedding for {image_name}")

    return np.array(feat)


def main():
    args = util.read_args()
    aligned_path = args.aligned_path
    deid_path = args.deidentified_path
    path_to_save = args.save_path
    path_to_log = args.dir_to_log

    ds = util.get_dataset_name_from_path(aligned_path)
    tech = util.get_technique_name_from_path(deid_path)

    # Shared embedding cache: {dataset}/original/ (shared) + {dataset}/deid/{technique}/ (per technique)
    temp_dir = util.get_temp_dir(args.root_dir, EVALUATION_NAME)
    cache_aligned = os.path.join(temp_dir, ds, "original")
    cache_deid   = os.path.join(temp_dir, ds, "deid", tech)
    os.makedirs(cache_aligned, exist_ok=True)
    os.makedirs(cache_deid,   exist_ok=True)

    genu_names_a, genu_ids_a, genu_names_b, genu_ids_b = util.read_pairs_file(args.genuine_pairs_filepath)
    impo_names_a, impo_ids_a, impo_names_b, impo_ids_b = util.read_pairs_file(args.impostor_pairs_filepath)

    names_a = genu_names_a + impo_names_a
    names_b = genu_names_b + impo_names_b
    ids_a = genu_ids_a + impo_ids_a
    ids_b = genu_ids_b + impo_ids_b
    ground_truth = np.array([int(a == b) for a, b in zip(ids_a, ids_b)])

    metrics_df = util.Metrics(name_score="cossim")
    scores = []

    for i, (name_a, name_b, gt) in enumerate(
        tqdm(zip(names_a, names_b, ground_truth), total=len(names_a),
             desc=f"deepface_vggface | {ds}-{tech}")
    ):
        if util._TEST_SINGLE and i > 0:
            break

        path_a = os.path.join(aligned_path, name_a)
        path_b = os.path.join(deid_path, name_b)

        if not os.path.exists(path_a):
            util.log(os.path.join(path_to_log, "deepface_vggface.txt"), f"({ds}) Missing: {path_a}")
            continue
        if not os.path.exists(path_b):
            util.log(os.path.join(path_to_log, "deepface_vggface.txt"), f"({tech}) Missing: {path_b}")
            continue

        try:
            feat_a = _get_embedding(path_a, cache_aligned, path_to_log)
            feat_b = _get_embedding(path_b, cache_deid,   path_to_log)
            if feat_a is None or feat_b is None:
                continue

            norm_a = np.linalg.norm(feat_a)
            norm_b = np.linalg.norm(feat_b)
            if norm_a < 1e-10 or norm_b < 1e-10:
                cos_sim = 0.0
            else:
                cos_sim = np.dot(feat_a, feat_b) / (norm_a * norm_b)

            scores.append(cos_sim)
            metrics_df.add_score(name_a, cos_sim)
            metrics_df.add_column_value("ground_truth", gt)

        except Exception as exc:
            util.log(os.path.join(path_to_log, "deepface_vggface.txt"), f"Error on {name_a}/{name_b}: {exc}")
            continue

    if scores:
        pass  # MIN/MAX debug prints commented out
    else:
        print("No scores computed.")

    metrics_df.save_to_csv(path_to_save)
    print(f"deepface_vggface saved into {path_to_save}")
    print(f"Embedding cache: aligned={cache_aligned}, deid={cache_deid}")


if __name__ == "__main__":
    main()
