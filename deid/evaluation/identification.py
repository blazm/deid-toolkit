"""Identification evaluation: gallery search producing CMC curve data.

Loads cached .pkl embeddings (produced by verification scripts), splits
images into gallery/probe per identity using label CSVs, ranks gallery
identities by cosine similarity, and outputs per-probe match data for CMC.

Supports all 4 verification models: arcface, adaface_optimized, swinface, deepface_vggface.
Embeddings are read from root_dir/preprocess/temp/{model}/{dataset}/original/.

CSV output columns (per probe):
    probe_image, true_identity, rank, rank1_correct, rank2_correct, ..., rank20_correct
"""
from __future__ import annotations

import argparse
import csv
import os
import pickle
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import utils as util


def load_cached_embeddings(cache_dir: str) -> dict[str, np.ndarray]:
    """Load all .pkl embeddings from a directory.

    Returns {image_name_without_ext: numpy_embedding} dict.
    """
    embs = {}
    if not os.path.isdir(cache_dir):
        return embs
    for f in os.listdir(cache_dir):
        if f.endswith(".pkl"):
            try:
                with open(os.path.join(cache_dir, f), "rb") as fh:
                    data = pickle.load(fh)
                name = f[:-4]  # strip .pkl
                if isinstance(data, dict):
                    # SWINFace format: {"Recognition": tensor}
                    for _, v in data.items():
                        embs[name] = np.array(v.cpu() if hasattr(v, "cpu") else v)
                elif hasattr(data, "cpu"):
                    embs[name] = data.cpu().numpy()
                else:
                    embs[name] = np.array(data)
            except Exception as e:
                print(f"  [WARN] Failed to load {f}: {e}")
    return embs


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def main() -> None:
    parser = argparse.ArgumentParser(description="Identification evaluation (gallery search)")
    parser.add_argument("aligned_path", type=str, help="Path to aligned (original) images")
    parser.add_argument("deidentified_path", type=str, help="Path to de-identified images")
    parser.add_argument("--dataset_name", type=str, default="")
    parser.add_argument("--technique_name", type=str, default="")
    parser.add_argument("--impostor_pairs_filepath", type=str, default="")
    parser.add_argument("--genuine_pairs_filepath", type=str, default="")
    parser.add_argument("--save_path", type=str, help="Output CSV path")
    parser.add_argument("--dir_to_log", type=str, default=".")
    parser.add_argument("--root_dir", type=str, default=".", help="Root directory (for cache lookup)")
    parser.add_argument("--model", type=str, default="swinface",
                        choices=["arcface", "adaface_optimized", "swinface", "deepface_vggface"],
                        help="Embedding model to use (reads cached .pkl files)")
    parser.add_argument("--gallery_ratio", type=float, default=0.5,
                        help="Fraction of each identity's images to use as gallery (0-1)")
    parser.add_argument("--labels_path", type=str, default="",
                        help="Path to labels CSV (Name,Identity columns). Auto-detected if empty.")
    args = parser.parse_args()

    dataset_name = util.get_dataset_name_from_path(args.aligned_path)
    technique_name = util.get_technique_name_from_path(args.deidentified_path)

    # --- Load identity labels ---
    labels_path = args.labels_path
    if not labels_path:
        labels_dir = Path(args.aligned_path).parent.parent / "labels"
        candidate = labels_dir / f"{dataset_name}_labels.csv"
        if candidate.exists():
            labels_path = str(candidate)
        else:
            print(f"No labels file found for {dataset_name} — skipping identification.")
            return

    identity_images: dict[str, list[str]] = defaultdict(list)
    with open(labels_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("Name", row.get("name", ""))
            identity = row.get("Identity", row.get("identity", row.get("ID", "")))
            if name and identity and identity.strip():
                identity_images[identity].append(name)

    if not identity_images:
        print(f"No identity labels found in {labels_path} — skipping.")
        return

    # --- Split into gallery/probe per identity ---
    gallery_names_by_id: dict[str, list[str]] = {}  # identity -> [image_names]
    probe_list: list[tuple[str, str]] = []  # [(image_name, true_identity)]

    for identity, images in sorted(identity_images.items()):
        n = len(images)
        n_gallery = max(1, int(round(n * args.gallery_ratio)))
        gallery_imgs = sorted(images[:n_gallery])
        probe_imgs = sorted(images[n_gallery:])
        if not probe_imgs:
            continue
        gallery_names_by_id[identity] = gallery_imgs
        for img_name in probe_imgs:
            probe_list.append((img_name, identity))

    if not probe_list:
        print("No probe images after split — skipping.")
        return

    # --- Load cached embeddings ---
    # Original (gallery) embeddings: shared across techniques
    temp_dir = util.get_temp_dir(args.root_dir, args.model)
    cache_original  = os.path.join(temp_dir, dataset_name, "original")
    # De-identified (probe) embeddings: per technique
    cache_deid      = os.path.join(temp_dir, dataset_name, "deid", technique_name)

    print(f"Loading cached embeddings ({args.model}) ...")
    print(f"  Original cache: {cache_original}")
    print(f"  De-identified cache: {cache_deid}")

    orig_embs = load_cached_embeddings(cache_original)
    deid_embs = load_cached_embeddings(cache_deid)

    if not orig_embs:
        print(f"No cached original embeddings found in {cache_original}")
        print(f"  Run verification (e.g. `{args.model}`) first to generate cache.")
        return

    # --- Build gallery embedding map: identity -> matrix of embeddings ---
    gallery_by_id: dict[str, np.ndarray] = {}
    for identity, img_names in gallery_names_by_id.items():
        embs_list = []
        for name in img_names:
            if name in orig_embs:
                embs_list.append(orig_embs[name])
        if embs_list:
            gallery_by_id[identity] = np.array(embs_list)

    if not gallery_by_id:
        print("No gallery embeddings matched — skipping.")
        return

    total_gallery = sum(len(v) for v in gallery_by_id.values())
    print(f"Gallery: {len(gallery_by_id)} identities, {total_gallery} images")
    print(f"Probes:  {len(probe_list)} images")

    # --- Gallery search ---
    probe_results: list[dict] = []
    max_probes_to_process = min(len(probe_list), 20)  # we only need rank-1..20 for CMC

    for i, (img_name, true_identity) in enumerate(
        tqdm(probe_list, desc=f"identification({args.model}) | {dataset_name}-{technique_name}")
    ):
        if util._TEST_SINGLE and i > 0:
            break

        # Try de-identified cache first, fall back to original (validation baseline)
        probe_emb = None
        if img_name in deid_embs:
            probe_emb = deid_embs[img_name]
        elif img_name in orig_embs:
            probe_emb = orig_embs[img_name]  # validation: aligned vs aligned
        else:
            continue

        # Rank all identities by best cosine similarity to any gallery image
        ranked: list[tuple[float, str]] = []
        for identity, gallery_embs in gallery_by_id.items():
            norms = np.linalg.norm(gallery_embs, axis=1)
            if norms.min() < 1e-10:
                continue
            sims = np.dot(gallery_embs, probe_emb) / (norms * np.linalg.norm(probe_emb))
            ranked.append((float(sims.max()), identity))

        if not ranked:
            continue
        ranked.sort(key=lambda x: x[0], reverse=True)

        rank = next((k + 1 for k, (_, ident) in enumerate(ranked) if ident == true_identity), len(ranked) + 1)
        probe_results.append({
            "probe_image": img_name,
            "true_identity": true_identity,
            "rank": min(rank, max_probes_to_process + 1),  # cap for CMC purposes
        })

    if not probe_results:
        print("No probe results — skipping.")
        return

    # --- Save CSV with per-rank correctness columns ---
    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    top_k = min(max_probes_to_process, 20)
    fieldnames = ["probe_image", "true_identity", "rank"] + [f"rank{k}_correct" for k in range(1, top_k + 1)]

    for r in probe_results:
        for k in range(1, top_k + 1):
            r[f"rank{k}_correct"] = int(r["rank"] <= k)

    with open(args.save_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(probe_results)

    # Summary stats
    total = len(probe_results)
    print(f"\nIdentification ({args.model}) saved to {args.save_path}")
    print(f"  Total probes: {total}")
    for k in [1, 5, 10]:
        if k <= top_k:
            hits = sum(r[f"rank{k}_correct"] for r in probe_results)
            print(f"  Rank@{k:>2}: {hits}/{total} ({100*hits/total:.1f}%)")


if __name__ == "__main__":
    main()
