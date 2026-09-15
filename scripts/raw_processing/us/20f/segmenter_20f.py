"""
scripts/us/20f/20f_segmenter.py — the whole 20-F annual report as ONE
section, not a real per-heading segmenter.

Same rationale as scripts/us/proxy/proxy_segmenter.py: a 20-F's item
structure (Item 4 "Information on the Company", Item 5 "Operating and
Financial Review and Prospects", Item 3 "Key Information" incl. risk
factors, ...) doesn't match the 10-K's Item 1/1A/7 numbering that
section_segmenter.py's heading regex targets, and only 5 firms in this
universe file as 20-F (ASML, HMC, TM, TSM, UL) — not enough volume to
justify writing and validating a dedicated heading parser. Whole-document
extraction lets the same prefilter (lexical + semantic scoring) find the
AI-relevant paragraphs regardless of which named section they sit in.

Reuses section_segmenter.clean_html_to_lines() (already form-agnostic
Markdown conversion) unchanged.
"""

from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/us/
from section_segmenter import clean_html_to_lines  # noqa: F401 (re-exported for 02_extract_sections.py)

TARGET_ITEMS = {"0": "Full Document"}


def general_segment(lines: list[str], form: str = "20-F", ticker: str | None = None) -> dict[str, str]:
    return {"0": "\n".join(lines)}


def known_issue(ticker: str, form: str, item_key: str | None = None) -> dict | None:
    """No known-issue tracking for 20-F yet -- always None, matching
    known_segmenter_issues.yaml's contract without a populated file."""
    return None
