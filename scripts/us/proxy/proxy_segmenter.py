"""
scripts/us/proxy/proxy_segmenter.py — the whole DEF 14A document as ONE
section, not a real per-heading segmenter.

A proxy statement doesn't have the 10-K's stable Item 1/1A/7 numbering
(section_segmenter.py's heading regex is 10-K/10-Q specific) — its
AI-governance-relevant content lives under headings like "Board
Committees", "Risk Oversight", "Compensation Discussion and Analysis"
that vary in wording and placement across companies. Writing and
validating a real heading-based proxy segmenter is real work
(docs/document_expansion_plan.md Fase 3 flags this explicitly) that
wasn't justified to build blind, so this ships the honest, simpler
alternative instead: treat the full document as `item_key="0"` and let
the SAME prefilter (lexical + semantic scoring, already proven on 10-K)
find the AI/governance-relevant paragraphs inside it -- the prefilter
doesn't care which named section a paragraph came from, only whether the
paragraph itself looks AI-relevant. This is the same "whole document,
no Item structure" choice already made for Chile's Memoria Anual
(scripts/cl/cmf_pdf_paragraphs.py; item_key='0' there too).

Reuses section_segmenter.clean_html_to_lines() (already form-agnostic
Markdown conversion) unchanged.
"""

from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/us/
from section_segmenter import clean_html_to_lines  # noqa: F401 (re-exported for 02_extract_sections.py)

TARGET_ITEMS = {"0": "Full Document"}


def general_segment(lines: list[str], form: str = "DEF 14A") -> dict[str, str]:
    return {"0": "\n".join(lines)}


def known_issue(ticker: str, form: str, item_key: str | None = None) -> dict | None:
    """No known-issue tracking for proxy yet (docs/document_expansion_plan.md
    Fase 3 is new) -- always None, matching known_segmenter_issues.yaml's
    contract without a populated file."""
    return None
