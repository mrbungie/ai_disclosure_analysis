"""
scripts/us/8k/8k_segmenter.py — the whole 8-K document as ONE section, not a
real per-Item segmenter.

An 8-K's "Item" numbers (1.01, 2.02, 5.02, 7.01, 8.01, ...) are event-type
codes, not the stable narrative-section numbering that section_segmenter.py
was built for (10-K Item 1/1A/7 always mean the same thing; 8-K Item 7.01
"Regulation FD Disclosure" can carry anything from an AI product launch
press release to an unrelated investor-day slide deck). Most of an 8-K's
substance also lives in an attached EX-99.1 press-release exhibit, not the
cover-page body — a real segmenter would need to decide how to handle
exhibits, which is real work not justified to build blind. Same pragmatic
choice already made for Chile's Memoria Anual and US proxy (DEF 14A): whole
document as item_key="0", let the existing prefilter (lexical + semantic)
find the AI-relevant paragraphs regardless of which numbered Item they came
from.

Reuses section_segmenter.clean_html_to_lines() (form-agnostic Markdown
conversion) unchanged.
"""

from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/us/
from section_segmenter import clean_html_to_lines  # noqa: F401 (re-exported for 02_extract_sections.py)

TARGET_ITEMS = {"0": "Full Document"}


def general_segment(lines: list[str], form: str = "8-K", ticker: str | None = None,
                    html_path: Path | None = None) -> dict[str, str]:
    return {"0": "\n".join(lines)}


def known_issue(ticker: str, form: str, item_key: str | None = None) -> dict | None:
    """No known-issue tracking for 8-K yet -- always None, matching
    known_segmenter_issues.yaml's contract without a populated file."""
    return None
