"""PDF report generation for pipeline results.

Imported by ``deid/pipeline.py`` (compute side) — moved out of the
explore module so that PDF reports can be generated without Streamlit.
"""

from deid.reports.pdf_export import export_results_to_pdf, generate_summary_report

__all__ = ["export_results_to_pdf", "generate_summary_report"]
