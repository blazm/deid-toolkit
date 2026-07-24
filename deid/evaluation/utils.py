"""Shared utilities for evaluation scripts."""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

from PIL import Image

_TEST_SINGLE = int(os.environ.get("DEID_TEST_SINGLE", "0"))


def read_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("aligned_path", type=str)
    parser.add_argument("deidentified_path", type=str)
    parser.add_argument("--dataset_name", type=str, default="")
    parser.add_argument("--technique_name", type=str, default="")
    parser.add_argument("--impostor_pairs_filepath", type=str, default="")
    parser.add_argument("--genuine_pairs_filepath", type=str, default="")
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--dir_to_log", type=str, default=".")
    parser.add_argument("--root_dir", type=str, default=".")
    parser.add_argument("--eval_package_dir", type=str, default=".")
    return parser.parse_args()


def resize_if_different(img0: Image, img1: Image) -> Image:
    """Resize img0 to match img1's size if shapes differ."""
    if img0.size != img1.size:
        return img0.resize(img1.size)
    return img0


class Metrics:
    """Accumulates per-row metrics and writes a CSV."""

    def __init__(self, name_score: str = "score") -> None:
        self._scores: dict[str, float] = {}
        self._columns: dict[str, list] = {}
        self._name_score = name_score

    def add_score(self, img: str, metric_result: float) -> None:
        self._scores[img] = float(metric_result)

    def add_column_value(self, col: str, value: object) -> None:
        if col not in self._columns:
            self._columns[col] = []
        self._columns[col].append(value)

    def save_to_csv(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        img_names = sorted(self._scores.keys())
        col_names = sorted(self._columns.keys())
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["image", self._name_score] + col_names
            writer.writerow(header)
            for img in img_names:
                row = [img, self._scores[img]]
                for cn in col_names:
                    idx = img_names.index(img)
                    row.append(self._columns[cn][idx] if idx < len(self._columns[cn]) else "")
                writer.writerow(row)


def log(path: str, msg: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as f:
        f.write(msg + "\n")


def get_dataset_name_from_path(path: str) -> str:
    return Path(path).name


def get_technique_name_from_path(path: str) -> str:
    """Extract technique name from de-identified image path.

    Standard layout: {root}/datasets/deidentified/{technique}/{dataset}/
    Returns parent directory name (the technique folder).
    """
    return Path(path).parent.name


def read_pairs_file(filepath: str) -> tuple[list[str], list[str], list[str], list[str]]:
    """Read a pair file of format: id_a name_a id_b name_b.

    Returns (names_a, ids_a, names_b, ids_b).
    """
    names_a, ids_a, names_b, ids_b = [], [], [], []
    with open(filepath) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 4:
                continue
            ids_a.append(parts[0])
            names_a.append(parts[1])
            ids_b.append(parts[2])
            names_b.append(parts[3])
    return names_a, ids_a, names_b, ids_b


def get_temp_dir(root_dir: str, eval_name: str) -> str:
    """Return a temporary directory under root_dir for cached features of an evaluation."""
    temp_path = os.path.join(root_dir, "preprocess", "temp", eval_name)
    os.makedirs(temp_path, exist_ok=True)
    return temp_path
