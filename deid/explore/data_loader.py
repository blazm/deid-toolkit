"""Data loading helpers for the Streamlit explore app."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from deid.config.loader import ConfigLoader


def get_loader(workspace: str = None) -> ConfigLoader:
    """Get or create a ConfigLoader singleton.

    Args:
        workspace: Optional workspace directory to use. Defaults to the
                   logged-in user's workspace or the default root_dir.
    """
    if not hasattr(get_loader, "_instance"):
        get_loader._instance = None
    if get_loader._instance is None:
        if workspace:
            loader = ConfigLoader(config_yaml_path=Path(workspace) / "deid-config.yaml")
        else:
            loader = ConfigLoader()
        get_loader._instance = loader
    return get_loader._instance  # type: ignore[attr-defined]


def reset_loader() -> None:
    """Reset the singleton (used by logout, workspace switch)."""
    get_loader._instance = None


def list_datasets() -> list[tuple[str, bool]]:
    """Return (name, is_aligned) for all available datasets."""
    loader = get_loader()
    original = set(loader.load_datasets())
    aligned = set(loader.load_aligned_datasets())
    all_names = sorted(original | aligned)
    return [(n, n in aligned) for n in all_names]


def list_techniques() -> list[str]:
    loader = get_loader()
    return loader.load_techniques()


def list_evaluations() -> list[str]:
    loader = get_loader()
    return loader.load_evaluations()


def list_results() -> dict[str, dict[str, dict[str, Path]]]:
    """Return {dataset: {technique: {metric: csv_path}}}."""
    loader = get_loader()
    return loader.list_results()


def load_results_csv(csv_path: Path) -> pd.DataFrame:
    """Load a results CSV into a DataFrame."""
    return pd.read_csv(csv_path)


def list_aligned_images(dataset: str) -> list[Path]:
    """List aligned images for a dataset."""
    loader = get_loader()
    aligned_dir = loader.settings.aligned_path / dataset
    if not aligned_dir.is_dir():
        return []
    return sorted(
        p for p in aligned_dir.glob("*")
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    )


def list_deid_images(dataset: str, technique: str) -> list[Path]:
    """List de-identified images for a dataset+technique."""
    loader = get_loader()
    deid_dir = loader.settings.deid_path / technique / dataset
    if not deid_dir.is_dir():
        return []
    return sorted(
        p for p in deid_dir.glob("*")
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    )


def get_dataset_image_count(dataset: str) -> int:
    """Return the number of aligned images for a dataset."""
    return len(list_aligned_images(dataset))


def get_dataset_image_dimensions(dataset: str, sample_size: int = 3) -> list[tuple[int, int, int]]:
    """Return [(width, height, channels), ...] from a small random sample of images."""
    try:
        from PIL import Image
    except ImportError:
        return []

    images = list_aligned_images(dataset)
    if not images:
        original_dir = get_loader().settings.original_path / dataset
        if original_dir.is_dir():
            images = sorted(
                p for p in original_dir.glob("*")
                if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
            )
    if not images:
        return []

    # Sample up to sample_size images
    step = max(1, len(images) // sample_size)
    dims = []
    for p in images[:sample_size]:
        try:
            with Image.open(p) as img:
                w, h = img.size
                dims.append((w, h, img.mode))
        except Exception:
            pass
    return dims


def get_dataset_attribute_columns(dataset: str) -> list[str]:
    """Return available attribute columns from the labels CSV for a dataset."""
    loader = get_loader()
    root = Path(loader.settings.root_dir)
    labels_dir = root / "datasets" / "labels"
    label_file = labels_dir / f"{dataset}_labels.csv"
    if not label_file.is_file():
        return []
    try:
        df = pd.read_csv(label_file)
        exclude = {"Name", "name", "filename", "file_name"}
        return [c for c in df.columns if c not in exclude]
    except Exception:
        return []


def get_dataset_sample_images(dataset: str, n: int = 5) -> list[Path]:
    """Return up to n aligned image paths for preview (skip first 10 to avoid top-of-file bias)."""
    images = list_aligned_images(dataset)
    if not images:
        original_dir = get_loader().settings.original_path / dataset
        if original_dir.is_dir():
            images = sorted(
                p for p in original_dir.glob("*")
                if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
            )
    if not images:
        return []
    start = min(10, len(images))
    return images[start:start + n]
