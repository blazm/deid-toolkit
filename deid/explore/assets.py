"""Shared assets for the explore app."""
from __future__ import annotations

from pathlib import Path

# Root of the deid package
_PACKAGE_DIR = Path(__file__).resolve().parent

def logo_path() -> str:
    """Return the absolute path to the logo SVG."""
    return str(_PACKAGE_DIR.parent.parent / "assets" / "deid-toolkit-website-logo.svg")
