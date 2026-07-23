"""Backwards-compatible import shim.

Moved to ``deid.reports.pdf_export`` — import from there instead.
This file remains for existing code that still imports from explore/.
"""
from __future__ import annotations

try:
    from deid.reports.pdf_export import (
        export_results_to_pdf,
        generate_summary_report,
    )
except ImportError:
    # If reports package is not installed, provide no-op fallback
    def export_results_to_pdf(*args, **kwargs):  # type: ignore
        return []

    def generate_summary_report(*args, **kwargs):  # type: ignore
        from pathlib import Path
        return Path(".")

__all__ = ["export_results_to_pdf", "generate_summary_report"]
