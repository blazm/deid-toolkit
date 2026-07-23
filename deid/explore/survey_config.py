"""Survey configuration — datasets, attributes, and settings for human verification."""
from __future__ import annotations

# List of datasets available for human verification survey
AVAILABLE_DATASETS = [
    "arface",
    "arc2face",
    "celeba",
    "colorferet",
    "fri",
    "lfw",
    "muct",
    "raf_db",
]

# Available labels for each dataset (for attribute evaluation)
DATASETS_WITH_LABELS = {
    "arface": ["gender"],
    "arc2face": [],
    "celeba": ["gender", "expression"],
    "colorferet": ["gender"],
    "fri": ["gender", "expression"],
    "lfw": ["gender", "expression"],
    "muct": ["gender"],
    "raf_db": ["expression"],
}

# Default attribute to evaluate in the survey
# Options: "id", "gender", "expression", "age"
DEFAULT_EVALUATION_ATTRIBUTE = "id"

# Attributes available for evaluation
EVALUATION_ATTRIBUTES = ["id", "gender", "expression", "age"]

# Max pairs per survey session
MAX_PAIRS_PER_SESSION = 10

# Max sessions per IP address
MAX_SESSIONS_PER_IP = 10
