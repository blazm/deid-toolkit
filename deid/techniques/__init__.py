"""Built-in deidentification techniques.

Techniques are executed as subprocesses in each technique's conda environment.
To add a custom technique, create a Python script in your workspace's
techniques/ directory. The script must accept:
    python script.py aligned_path save_path [extra_args]
"""
from __future__ import annotations

from importlib import resources


def list_builtin_techniques() -> list[str]:
    """Return sorted names of all bundled technique scripts (without .py extension)."""
    pkg = resources.files("deid.techniques")
    return sorted(
        p.stem for p in pkg.iterdir()
        if p.suffix == ".py" and p.name != "__init__.py"
    )
