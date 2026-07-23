"""Built-in evaluation metrics.

Evaluations are organized into categories:
- identity_verification/: AdaFace, insightface, swinface, vgg-face
- data_utility/: DAN, ResNet18 (requires external model downloads)

Evaluations are executed as subprocesses in each evaluation's conda environment.
To add a custom evaluation, create a Python/Shell script in your workspace's
evaluation/ directory. The script must accept:
    python script.py aligned_path deidentified_path \
        --dataset_name NAME --technique_name TECH \
        --impostor_pairs_filepath PATH --genuine_pairs_filepath PATH \
        --save_path PATH
"""
from __future__ import annotations

from importlib import resources


def list_builtin_evaluations() -> list[str]:
    """Return sorted names of all bundled evaluation scripts."""
    pkg = resources.files("deid.evaluation")
    return sorted(
        p.stem for p in pkg.iterdir()
        if p.suffix == ".py" and p.name not in {"__init__.py", "utils.py"}
    )


def list_builtin_evaluation_dirs() -> list[str]:
    """List subdirectories (identity_verification, data_utility)."""
    pkg = resources.files("deid.evaluation")
    return sorted(
        p.name for p in pkg.iterdir()
        if p.is_dir() and p.name not in {"utils", "__pycache__"}
    )
