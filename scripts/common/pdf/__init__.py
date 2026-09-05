"""
scripts/common/pdf/ — country-agnostic filing-PDF handling.

    pipeline.extract_paragraphs(parts, backend=..., country_code=...)

Everything about parsing a filing PDF lives here except the one thing that
genuinely differs by regulator — which strings that regulator's template
repeats as page furniture — which lives in profiles.py. See
docs/analytics/pdf-backend-poc.md for the measured comparison behind the
default backend choice.
"""

from scripts.common.pdf import blocks, pipeline, profiles, render  # noqa: F401
from scripts.common.pdf.pipeline import (  # noqa: F401
    PageTriage, PdfExtractor, extract_paragraphs, triage_from_config,
)
