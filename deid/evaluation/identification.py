"""Identification evaluation: gallery search producing CMC curve data.

Loads identity labels, splits each identity's images into gallery and probe,
computes DeepFace embeddings for probe and gallery images, ranks gallery
identities by cosine similarity, and outputs per-probe match data for CMC.

CSV output columns:
    probe_image, true_identity, rank, matched_identity, rank1_score,
    rank5_score, rank1_correct
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
from tqdm import tqdm

# Ensure evaluation dir is on path for utils
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import utils as util


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def main() -> None:
    parser = argparse.ArgumentParser(description="Identification evaluation")
    parser.add_argument("aligned_path", type=str, help="Path to aligned (original) images")
    parser.add_argument("deidentified_path", type=str, help="Path to de-identified images")
    parser.add_argument("--dataset_name", type=str, default="")
    parser.add_argument("--technique_name", type=str, default="")
    parser.add_argument("--impostor_pairs_filepath", type=str, default="")
    parser.add_argument("--genuine_pairs_filepath", type=str, default="")
    parser.add_argument("--save_path", type=str, help="Output CSV path")
    parser.add_argument("--dir_to_log", type=str, default=".")
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
        # Auto-detect from standard labels directory
        labels_dir = Path(args.aligned_path).parent.parent / "labels"
        candidate = labels_dir / f"{dataset_name}_labels.csv"
        if candidate.exists():
            labels_path = str(candidate)
        else:
            print(f"No labels file found for {dataset_name} — skipping identification evaluation.")
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
    gallery_dir = Path(args.deidentified_path).parent / "gallery"
    gallery_dir.mkdir(exist_ok=True)
    probe_dir = Path(args.deidentified_path).parent / "probe"
    probe_dir.mkdir(exist_ok=True)

    aligned_path = Path(args.aligned_path)
    deid_path = Path(args.deidentified_path)
    gallery_images: dict[str, list[str]] = defaultdict(list)  # identity -> [probe_image_name]
    probe_images: list[str] = []  # (probe_name, true_identity)

    for identity, images in sorted(identity_images.items()):
        n = len(images)
        n_gallery = max(1, int(round(n * args.gallery_ratio)))
        gallery_imgs = sorted(images[:n_gallery])
        probe_imgs = sorted(images[n_gallery:])
        if not probe_imgs:
            continue  # need at least one probe per identity

        gallery_images[identity] = gallery_imgs
        for img_name in probe_imgs:
            probe_images.append(img_name)

    if not probe_images:
        print("No probe images after split — skipping.")
        return

    # --- Extract embeddings using DeepFace ---
    from deepface import DeepFace
    DeepFace.settings.represent_mode = "flatten"

    print(f"Extracting gallery embeddings ({sum(len(v) for v in gallery_images.values()} images)...")
    gallery_embeddings: dict[str, np.ndarray] = {}  # identity -> [embedding]
    gallery_image_names: dict[str, list[str]] = {}  # identity -> [img_name]

    for identity, imgs in sorted(gallery_images.items()):
        gallery_image_names[identity] = []
        gallery_embeddings[identity] = []
        for img_name in imgs:
            aligned_path_img = aligned_path / img_name
            if not aligned_path_img.exists():
                continue
            try:
                rep = DeepFace.represent(
                    img_path=str(aligned_path_img),
                    model_name="VGG-Face",
                    enforce_detection=False,
                    silent=True,
                )
                if rep:
                    emb = np.array(rep[0]["embedding"])
                    gallery_embeddings[identity].append(emb)
                    gallery_image_names[identity].append(img_name)
            except Exception:
                continue

    # Filter identities with no embeddings
    gallery_images = {k: v for k, v in gallery_images.items() if v}
    gallery_embeddings = {k: np.array(v) for k, v in gallery_embeddings.items() if v}
    gallery_image_names = {k: v for k, v in gallery_image_names.items() if v}

    if not gallery_images:
        print("No gallery embeddings extracted — skipping.")
        return

    print(f"Extracting probe embeddings ({len(probe_images)} images)...")
    probe_results: list[dict] = []

    for i, img_name in enumerate(tqdm(probe_images, desc=f"identification | {dataset_name}-{technique_name}")):
        if util._TEST_SINGLE and i > 0:
            break
        probe_path_img = deid_path / img_name
        if not probe_path_img.exists():
            continue

        # Find true identity for this probe image
        true_identity = None
        for identity, images in identity_images.items():
            if img_name in images:
                true_identity = identity
                break
        if not true_identity:
            continue

        try:
            probe_rep = DeepFace.represent(
                img_path=str(probe_path_img),
                model_name="VGG-Face",
                enforce_detection=False,
                silent=True,
            )
            if not probe_rep or not probe_rep[0]["embedding"]:
                continue
            probe_emb = np.array(probe_rep[0]["embedding"])
        except Exception:
            continue

        # Compute similarity to all gallery images across all identities
        ranked: list[tuple[float, str]] = []  # (sim, identity)
        for identity, gallery_embs in gallery_embeddings.items():
            sims = np.dot(gallery_embs, probe_emb) / (
                np.linalg.norm(gallery_embs, axis=1) * np.linalg.norm(probe_emb)
            )
            best_sim = float(sims.max())
            ranked.append((best_sim, identity))

        ranked.sort(key=lambda x: x[0], reverse=True)

        rank1_match = ranked[0][1] == true_identity
        rank5_matches = [r[1] for r in ranked[:5]]
        rank5_match = true_identity in rank5_matches
        rank = next((i + 1 for i, (_, ident) in enumerate(ranked) if ident == true_identity), len(ranked) + 1)

        probe_results.append({
            "probe_image": img_name,
            "true_identity": true_identity,
            "rank": rank,
            "rank1_score": ranked[0][0] if ranked else 0.0,
            "rank5_score": ranked[4][0] if len(ranked) > 4 else 0.0,
            "rank1_correct": int(rank1_match),
            "rank5_correct": int(rank5_match),
        })

    if not probe_results:
        print("No probe results — skipping.")
        return

    # --- Save CSV ---
    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    fieldnames = ["probe_image", "true_identity", "rank", "rank1_score", "rank5_score",
                   "rank1_correct", "rank5_correct"]
    with open(args.save_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(probe_results)

    # Summary stats
    rank1_hits = sum(r["rank1_correct"] for r in probe_results)
    rank5_hits = sum(r["rank5_correct"] for r in probe_results)
    total = len(probe_results)
    print(f"Identification saved to {args.save_path}")
    print(f"  Rank@1: {rank1_hits}/{total} ({100*rank1_hits/total:.1f}%)")
    print(f"  Rank@5: {rank5_hits}/{total} ({100*rank5_hits/total:.1f}%)")


if __name__ == "__main__":
    main()
