"""
section_segmenter.py — shared HTML->Markdown->Item-segments logic, used by
scripts/04_extract_sections.py (extracts Item 1/1A/7 for the pipeline) and
verif_scripts/section_audit/section_audit.py (audits ALL items). One
source of truth so the two never drift apart.

Every "Item N" heading appears (at least) twice in a 10-K: once as a
table-of-contents ROW (a markdown hyperlink to the real section:
"[Item 1.](#anchor) [Business](#anchor) [5](#anchor)") and once as the
real heading itself (the anchor TARGET — plain text, no link, because it
doesn't link to itself: "ITEM 1. BUSINESS"). Telling these apart used to
be done by a length threshold on the extracted text (TOC rows happen to
be short, real sections happen to be long) — an arbitrary number with no
structural justification and no comment explaining it (checked: never
had one, see git history).

Two structural signals replace it:
1. `is_link_row` — a TOC row IS a stack of markdown links; most real
   heading lines aren't wrapped in link syntax. Cheap first pass, but
   under-catches: an inapplicable item ("Item 1B. Unresolved Staff
   Comments ... N/A") often has no page link in the TOC either, since
   there's nothing to jump to — link presence alone would treat it as a
   real heading.
2. TOC-PREFIX detection — the actual structural fact that survives every
   filer's formatting quirk: a table of contents lists each item ONCE, in
   increasing order (Item 1, 1A, 1B, ... 16), and it's always the FIRST
   thing in the document to do so. So the maximal prefix of candidates,
   starting from the very first one found, whose item numbers strictly
   increase with no repeat, IS the TOC — whatever it looks like (link or
   not). The moment a candidate repeats an item number already seen (the
   real "Item 1" heading, appearing again after the TOC listed it once),
   the prefix ends; everything from there on is body content. A window/
   density heuristic ("candidates packed within N lines") was tried first
   and rejected: it also swept up real headings that happen to immediately
   follow the TOC (the common case), which is exactly the text this whole
   segmenter is supposed to keep.
"""

import re
from pathlib import Path

import pandas as pd
from bs4 import XMLParsedAsHTMLWarning
from markdownify import markdownify as md
import warnings

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# Space-tolerant and Markdown-tolerant separator: the punctuation SEC filers
# use between "Item N" and the section title (en/em dashes, colons,
# markdown link brackets, table pipes, NBSPs).
SEP = r"[|#*_\s\-–—:\[\]]*"

# Digits and the A/B/C sub-item letter are captured separately, in one of
# 3 formats filers actually use — matching only "\d[A-C]?" silently drops
# the letter for the other two, colliding e.g. "Item 1(B)" with "Item 1"
# itself:
#   1A.   letter directly suffixed, no punctuation      ("Item 1A.")
#   1.C.  letter after a period, no space                ("Item 1.C.")
#   1 (B) letter parenthesized, optionally after a space  ("Item 1 (B)")
# A bare letter is only accepted right after the digit (optionally through
# a single period) or inside parens — never after a plain space, which
# would false-match a following word's first letter (e.g. "Item 14.
# Certain Relationships..." must NOT read as "Item 14C").
ITEM_RE = re.compile(
    rf"^\s*(?:{SEP})\s*Item{SEP}(\d{{1,2}})(?:\.?([A-C])|\s*\(([A-C])\))?\.?{SEP}",
    re.IGNORECASE,
)


def _item_key(match: re.Match) -> str:
    return match.group(1) + (match.group(2) or match.group(3) or "").upper()

# A small number of filers use a non-standard Item 1 heading instead of
# "Item 1. Business" (e.g. Honeywell's "ABOUT HONEYWELL", a lone plain-text
# heading line — still structurally a real heading, not a TOC row, just not
# spelled "Item 1"). Documented per-company exceptions, not a broad regex
# relaxation that would risk false positives elsewhere.
ITEM_ALIASES: dict[str, re.Pattern] = {
    "1": re.compile(r"^\s*ABOUT\s+HONEYWELL\s*$", re.IGNORECASE),
}

# A TOC row is literally a markdown hyperlink: "[text](#anchor)". A real
# heading is the anchor's target, plain text, never wrapped in link syntax.
LINK_RE = re.compile(r"\[[^\]]+\]\([^)]+\)")


def is_link_row(line: str) -> bool:
    return bool(LINK_RE.search(line))


def clean_html_to_lines(html_path: Path) -> list[str]:
    with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
        html_content = f.read()
    markdown_text = md(html_content, heading_style="ATX", strip=["script", "style"])
    lines = [line.strip() for line in markdown_text.split("\n")]
    return [line for line in lines if line]


def item_sort_key(item_key: str) -> tuple[int, int]:
    """'1' -> (1, 0), '1A' -> (1, 1), '7A' -> (7, 1), '16' -> (16, 0)."""
    m = re.match(r"(\d+)([A-C]?)", item_key.upper())
    assert m is not None
    num = int(m.group(1))
    letter_rank = {"": 0, "A": 1, "B": 2, "C": 3}[m.group(2)]
    return (num, letter_rank)


def _raw_candidates(lines: list[str]) -> list[tuple[str, int]]:
    """Every line that looks like an Item heading, TOC or real, in document
    order — no filtering yet."""
    found = []
    for i, line in enumerate(lines):
        m = ITEM_RE.search(line)
        if m:
            found.append((_item_key(m), i))
            continue
        for item_key, alias_re in ITEM_ALIASES.items():
            if alias_re.search(line):
                found.append((item_key, i))
    return found


# A real TOC lists most/all of a 10-K's ~15 items, packed within a couple
# hundred lines. A short or sparse strictly-increasing run isn't a TOC —
# e.g. a filer whose TOC doesn't spell "Item N" for most rows (only its
# real body headings match at all): treating that as "the TOC" would wrongly
# discard the one real heading actually found. Both guards matter: without
# MIN_TOC_ITEMS, 1-2 genuinely early body headings get misread as a TOC;
# without MAX_TOC_GAP, an unrelated later match could stay "monotonic" and
# get swept into a TOC prefix that isn't really one contiguous block.
MIN_TOC_ITEMS = 5
MAX_TOC_GAP = 150


def _toc_prefix_end(raw: list[tuple[str, int]]) -> int:
    """Index into `raw` (not a line number) of the first candidate that is
    NOT part of the initial strictly-increasing, no-repeat, tightly-packed
    run starting at raw[0] — i.e. where the TOC ends and body content
    begins. Returns 0 (no TOC prefix at all) if that initial run is too
    short/sparse to plausibly be a real table of contents."""
    seen: set[str] = set()
    last_rank = (-1, -1)
    last_line_idx = None
    prefix_len = 0
    for item_key, line_idx in raw:
        rank = item_sort_key(item_key)
        if item_key in seen or rank <= last_rank:
            break
        if last_line_idx is not None and line_idx - last_line_idx > MAX_TOC_GAP:
            break
        seen.add(item_key)
        last_rank = rank
        last_line_idx = line_idx
        prefix_len += 1
    return prefix_len if prefix_len >= MIN_TOC_ITEMS else 0


def general_segment(lines: list[str]) -> dict[str, str]:
    """One segment per recognizable "Item N[A-C]" heading. The TOC prefix
    (see _toc_prefix_end) is dropped outright; among the remaining
    candidates, link rows (is_link_row — an occasional in-body cross-
    reference styled as a link) are also dropped, the LAST occurrence per
    item is kept, then candidates are filtered to a monotonically
    non-decreasing item-number sequence in document order, matching how
    10-Ks are actually laid out. This is a heuristic, not a guarantee, but
    it no longer depends on any arbitrary length number."""
    raw = _raw_candidates(lines)
    body = raw[_toc_prefix_end(raw):]

    candidates_by_item: dict[str, int] = {}
    for item_key, line_idx in body:
        if is_link_row(lines[line_idx]):
            continue
        candidates_by_item[item_key] = line_idx  # overwrite -> last real occurrence wins

    ordered = sorted(candidates_by_item.items(), key=lambda kv: kv[1])

    kept = []
    last_rank = (-1, -1)
    for item_key, line_idx in ordered:
        rank = item_sort_key(item_key)
        if rank >= last_rank:
            kept.append((item_key, line_idx))
            last_rank = rank

    segments = {}
    for i, (item_key, start_idx) in enumerate(kept):
        end_idx = kept[i + 1][1] if i + 1 < len(kept) else len(lines)
        segments[item_key] = "\n".join(lines[start_idx:end_idx])
    return segments


def load_filing_sections(config: dict) -> pd.DataFrame:
    """Reads every extraction run's part-files (04_extract_sections.py
    writes filing_sections__run=<run_id>__part=<NNNN>__<comment>.parquet, never
    a single overwritten file) and collapses to one row per
    (accession_number, section_name), keeping the most recent run_date. Glob,
    don't hardcode a single filename — new runs just add more part files."""
    sections_dir = Path(config["paths"]["interim_sections"])
    part_paths = sorted(sections_dir.glob("filing_sections__run=*__part=*.parquet"))
    if not part_paths:
        raise FileNotFoundError(
            f"No filing_sections__run=*__part=*.parquet files under {sections_dir}. "
            "Run 04_extract_sections.py first."
        )
    sections = pd.concat((pd.read_parquet(p) for p in part_paths), ignore_index=True)
    sections = sections.sort_values("run_date").drop_duplicates(
        subset=["accession_number", "section_name"], keep="last"
    )
    return sections
