"""Survey API — generates pairs, handles submissions, manages quotas."""
from __future__ import annotations

import os
import random
import json
import time
from pathlib import Path
from typing import Optional

import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx

# ── Configuration ───────────────────────────────
SURVEY_DIR = Path(__file__).parent.parent.parent / "results" / "human_survey"
MAX_SESSIONS_PER_IP = 10  # max 10 sessions per IP
MAX_PAIRS_PER_SESSION = 10  # 5 same + 5 different


def reset_survey_results() -> int:
    """Delete all survey responses and reset rate limits. Returns number of files removed."""
    count = 0
    if SURVEY_DIR.exists():
        for f in SURVEY_DIR.glob("response_*.json"):
            f.unlink()
            count += 1
        for f in SURVEY_DIR.glob("*_session_count.json"):
            f.unlink()
            count += 1
    return count

# Quota tracking — tracks how many times each technique has been used
_technique_quota: dict[str, int] = {}
_technique_target = 0  # will be set dynamically

# ── Dataset label detection ─────────────────────
def get_dataset_labels(dataset_name: str, base_dir: Path = Path("datasets")) -> list[str]:
    """Detect available labels for a dataset.

    Looks for label files: {dataset_name}_labels.csv
    Returns list of available label types (gender, expression, age, ethnicity).
    """
    label_file = base_dir / f"{dataset_name}_labels.csv"
    if not label_file.exists():
        return []

    available = []
    with open(label_file, "r") as f:
        header = f.readline().strip().lower()
        columns = header.split(",")
        if "gender" in columns:
            available.append("gender")
        if "expression" in columns:
            available.append("expression")
        if "age" in columns:
            available.append("age")
        if "ethnicity" in columns:
            available.append("ethnicity")
    return available


# ── Pair generation ─────────────────────────────
def _get_techniques(datasets_dir: Path) -> list[str]:
    """Get list of available de-identification techniques."""
    techniques = []
    if datasets_dir.exists():
        for item in datasets_dir.iterdir():
            if item.is_dir() and item.name != "labels":
                techniques.append(item.name)
    return sorted(techniques)


def _update_technique_quota(techniques: list[str]) -> None:
    """Update technique quota to ensure equal representation."""
    global _technique_quota, _technique_target
    if not techniques:
        return
    # Calculate target quota per technique
    total_quota = MAX_PAIRS_PER_SESSION * 100  # target 100 pairs per technique
    _technique_target = total_quota // len(techniques)

    # Initialize any new techniques
    for tech in techniques:
        if tech not in _technique_quota:
            _technique_quota[tech] = 0


def generate_pairs(dataset_name: str, num_pairs: int = 10) -> list[dict]:
    """Generate a batch of pairs for the survey.

    Pairs are shuffled together (validation + de-identification mixed) so the user
    cannot distinguish which type they are viewing.

    Args:
        dataset_name: Name of the dataset to use (must match an actual dataset folder)
        num_pairs: Number of de-identification pairs to generate (default 10)

    Returns:
        List of pair dictionaries with image paths and metadata
    """
    from deid.explore.data_loader import get_loader

    loader = get_loader()
    aligned_dir = Path(loader.settings.aligned_path) / dataset_name
    deid_path = Path(loader.settings.deid_path)
    labels_dir = Path(loader.settings.root_dir) / "datasets" / "labels"

    if not aligned_dir.exists():
        raise ValueError(f"Dataset not found: {dataset_name}")

    label_types = get_dataset_labels(dataset_name, labels_dir)
    image_files = [f for f in aligned_dir.iterdir() if f.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    if not image_files:
        raise ValueError(f"No images found in {aligned_dir}")

    # ── Validation pairs (aligned-to-aligned) ──
    pairs = []
    val_per_type = max(5, num_pairs // 4)

    for _ in range(val_per_type):
        img_file = random.choice(image_files)
        deid_img = deid_path / "blur" / dataset_name / img_file.name
        if deid_img.exists():
            pairs.append({
                "pair_type": "validation_same",
                "display": "aligned",
                "image1_path": str(img_file),
                "image2_path": str(deid_img),
                "dataset": dataset_name,
                "available_labels": label_types,
                "ground_truth": "same",
            })

    for _ in range(val_per_type):
        if len(image_files) < 2:
            break
        img1, img2 = random.sample(image_files, 2)
        pairs.append({
            "pair_type": "validation_different",
            "display": "aligned",
            "image1_path": str(img1),
            "image2_path": str(img2),
            "dataset": dataset_name,
            "available_labels": label_types,
            "ground_truth": "different",
        })

    # ── De-identification pairs ──
    techniques = [
        t for t in _get_techniques(deid_path)
        if (deid_path / t / dataset_name).is_dir()
    ]

    if not techniques:
        random.shuffle(pairs)
        return pairs

    _update_technique_quota(techniques)

    for _ in range(num_pairs // 2):
        img_file = random.choice(image_files)
        if _technique_target > 0:
            weights = []
            for tech in techniques:
                quota_diff = _technique_target - _technique_quota.get(tech, 0)
                weights.append(max(1, quota_diff))
            selected_tech = random.choices(techniques, weights=weights, k=1)[0]
        else:
            selected_tech = random.choice(techniques)
        deid_img = deid_path / selected_tech / dataset_name / img_file.name
        if not deid_img.exists():
            continue
        pairs.append({
            "pair_type": "deid_same",
            "display": "deid",
            "original_path": str(img_file),
            "deid_path": str(deid_img),
            "technique": selected_tech,
            "dataset": dataset_name,
            "available_labels": label_types,
            "ground_truth": "same",
        })
        _technique_quota[selected_tech] = _technique_quota.get(selected_tech, 0) + 1

    for _ in range(num_pairs - num_pairs // 2):
        if len(image_files) < 2:
            break
        img1, img2 = random.sample(image_files, 2)
        selected_tech = random.choice(techniques)
        deid_img1 = deid_path / selected_tech / dataset_name / img1.name
        deid_img2 = deid_path / selected_tech / dataset_name / img2.name
        if not deid_img1.exists() or not deid_img2.exists():
            continue
        pairs.append({
            "pair_type": "deid_different",
            "display": "deid",
            "original1_path": str(img1),
            "original2_path": str(img2),
            "deid1_path": str(deid_img1),
            "deid2_path": str(deid_img2),
            "technique": selected_tech,
            "dataset": dataset_name,
            "available_labels": label_types,
            "ground_truth": "different",
        })

    # ── Shuffle all pairs together ──
    random.shuffle(pairs)
    return pairs


# ── Submission handling ─────────────────────────
def get_ip_address() -> str:
    """Get the client's IP address."""
    ctx = get_script_run_ctx()
    if ctx:
        client = getattr(ctx, "client", None)
        if client:
            ip = getattr(client, "ip", None)
            if ip:
                return ip
    return "unknown"


def check_rate_limit() -> bool:
    """Check if the current IP has exceeded the session limit."""
    ip = get_ip_address()
    if ip == "unknown":
        return True  # Allow if we can't determine IP

    # Load session count
    session_count_file = SURVEY_DIR / f"{ip}_session_count.json"
    if not session_count_file.exists():
        return True

    with open(session_count_file, "r") as f:
        data = json.load(f)

    current_count = data.get("count", 0)
    return current_count >= MAX_SESSIONS_PER_IP


def submit_responses(responses: list[dict], session_id: str) -> str:
    """Submit survey responses.

    Args:
        responses: List of responses
        session_id: Unique session ID

    Returns:
        Status message
    """
    # Check rate limit
    if not check_rate_limit():
        return "error", "Rate limit exceeded. Please wait before trying again."

    # Create response directory
    SURVEY_DIR.mkdir(parents=True, exist_ok=True)

    # Load or create session count
    ip = get_ip_address()
    session_count_file = SURVEY_DIR / f"{ip}_session_count.json"
    if session_count_file.exists():
        with open(session_count_file, "r") as f:
            data = json.load(f)
    else:
        data = {"count": 0}

    data["count"] += 1
    with open(session_count_file, "w") as f:
        json.dump(data, f)

    # Save response
    response_file = SURVEY_DIR / f"response_{session_id}.json"
    response_data = {
        "session_id": session_id,
        "submitted_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "ip_address": ip,
        "responses": responses,
    }

    with open(response_file, "w") as f:
        json.dump(response_data, f, indent=2)

    return "success", "Responses submitted successfully!"


# ── Results loading ─────────────────────────────
def get_survey_results() -> list[dict]:
    """Load all survey responses."""
    if not SURVEY_DIR.exists():
        return []

    responses = []
    for file in SURVEY_DIR.glob("response_*.json"):
        with open(file, "r") as f:
            responses.append(json.load(f))
    return responses
