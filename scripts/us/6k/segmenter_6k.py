"""
scripts/us/6k/segmenter_6k.py — the whole 6-K document as ONE section, not a
real per-Item segmenter.

6-K has no numbered-Item structure at all (unlike 8-K's event-type codes) —
it's a foreign private issuer's catch-all "current report," anything the
issuer would have had to make public at home or on its own exchange: an
interim/quarterly result, a press release, a shareholder circular, a
dividend notice. Same pragmatic whole-document choice already made for
8-K/proxy/Chile's Memoria Anual: item_key="0", let the existing prefilter
(lexical + semantic) find the AI-relevant paragraphs wherever they land.

Reuses section_segmenter.clean_html_to_lines() (form-agnostic Markdown
conversion) unchanged.
"""

from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/us/
from section_segmenter import clean_html_to_lines  # noqa: F401 (re-exported for 02_extract_sections.py)

TARGET_ITEMS = {"0": "Full Document"}


def general_segment(lines: list[str], form: str = "6-K", ticker: str | None = None) -> dict[str, str]:
    return {"0": "\n".join(lines)}


def known_issue(ticker: str, form: str, item_key: str | None = None) -> dict | None:
    """No known-issue tracking for 6-K yet -- always None, matching
    known_segmenter_issues.yaml's contract without a populated file."""
    return None
