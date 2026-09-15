"""
scripts/us/section_segmenter.py — shared HTML->Markdown->Item-segments
logic for SEC EDGAR filings (US-specific: SEC's "Item N" heading
convention doesn't generalize to other countries' filing formats — see
scripts/common/section_extraction.py's docstring for how a different
country would plug in its own segmenter instead). Used by
scripts/us/10k/02_extract_sections.py (extracts Item 1/1A/7),
scripts/us/10q/02_extract_sections.py (extracts Item 2/1A), and
scripts/verif/section_audit/section_audit.py (audits ALL items). One
source of truth so those never drift apart.

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
1. `_is_toc_formatted_row` — a TOC/index row states an item's NUMBER and
   its TITLE as two separate pieces of content (two markdown links, or
   two non-empty cells of a "|"-delimited layout table, or both); a real
   heading is always one piece of content, however oddly a given filer's
   HTML->markdown conversion styles it (plain text, a single "|"-padded
   layout row, or a link wrapped around just the title). Counting
   distinct non-empty segments tells these apart where cruder checks
   (any link on the line? any "|" on the line?) don't.
2. TOC-PREFIX detection (`_toc_prefix_end`) — the structural fact that
   survives every filer's formatting quirk: a table of contents lists
   every item ONCE, and it's always the FIRST thing in the document to do
   so. So the maximal prefix of candidates, starting from the very first
   one found, that stays TOC-row-formatted (signal 1) and tightly packed
   (MAX_TOC_GAP) IS the TOC; the first candidate that ISN'T TOC-formatted
   is unambiguously the first real heading — regardless of what item
   number it carries. This replaced an earlier version keyed on item
   NUMBERS (strictly-increasing, no-repeat): it broke on a 10-Q's single
   combined TOC, which lists Part I's items then RESTARTS at Part II's
   item 1 — a rank decrease inside the TOC itself, indistinguishable by
   rank alone from real body content starting. A density heuristic
   ("candidates packed within N lines") was tried even earlier and
   rejected too: it swept up real headings that happen to immediately
   follow the TOC (the common case), which is exactly the text this whole
   segmenter is supposed to keep.

A 10-Q additionally numbers Part I (Items 1-4) and Part II (Items 1-6)
INDEPENDENTLY, restarting at 1 — `_part_windows` splits the document at
explicit "PART I"/"PART II" markers so each Part's items segment
independently (see its docstring and `general_segment`'s for the details
and the false starts that led here).
"""

import gzip
import re
from pathlib import Path

import pandas as pd
import yaml
from bs4 import XMLParsedAsHTMLWarning
from markdownify import markdownify as md
import warnings

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# scripts/raw_processing/us/known_segmenter_issues.yaml — documented, investigated extraction
# limitations (tickers whose real content the segmenter structurally can't
# find, or can't find yet) so a benchmark/extraction failure on one of
# these doesn't get re-diagnosed from scratch every session. See
# known_issue() below and that file's own header for the schema.
_KNOWN_ISSUES_PATH = Path(__file__).resolve().parent / "known_segmenter_issues.yaml"
_known_issues_cache: list[dict] | None = None


def _load_known_issues() -> list[dict]:
    global _known_issues_cache
    if _known_issues_cache is None:
        if _KNOWN_ISSUES_PATH.exists():
            with open(_KNOWN_ISSUES_PATH) as f:
                _known_issues_cache = (yaml.safe_load(f) or {}).get("issues", [])
        else:
            _known_issues_cache = []
    return _known_issues_cache


def known_issue(ticker: str, form: str, item_key: str | None = None) -> dict | None:
    """Look up scripts/raw_processing/us/known_segmenter_issues.yaml for a documented,
    investigated extraction limitation matching this ticker/form(/item).
    Returns the matching issue dict (category, confirmed, note, ...) or
    None. Consult this before re-diagnosing a benchmark or extraction
    failure from scratch — it might already be understood."""
    for issue in _load_known_issues():
        if ticker not in issue.get("tickers", []):
            continue
        if form not in issue.get("forms", []):
            continue
        if item_key is not None and issue.get("items") and item_key not in issue["items"]:
            continue
        return issue
    return None

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
# The trailing `(?!\d)` matters: no real 10-K/10-Q Item exceeds 16, but
# filing narrative frequently cites a DIFFERENT numbering system by the
# same phrase — "Item 103 of SEC Regulation S-K" (GPC's 10-Q) — where
# `\d{1,2}` would otherwise silently truncate "103" to "10" and match a
# nonexistent "Item 10" heading. Requiring the digit run NOT be followed
# by another digit rejects any 3+-digit citation outright.
# The `(?![a-z])` after the bare-letter branch matters too: some filers
# (URI, FFIV) render the heading with NO space between the period and the
# title — "Item 4.Controls and Procedures" — where the title's own leading
# capital "C" would otherwise get read as a dotted letter-suffix, matching
# a nonexistent "Item 4C" instead of "Item 4". A real letter suffix is
# always followed by a period, space, dash, or the string's end — never
# directly by more lowercase letters spelling out a word.
# "Items?" (plural, optional "s") matters too: some 10-Q filers (NTRS)
# combine two items whose content overlaps into ONE heading — "Items 2.
# and 3. Management's Discussion and Analysis... and Quantitative and
# Qualitative Disclosures..." — plain "Item" never matches "Items", so
# this heading was invisible to the segmenter entirely. Matching the
# FIRST digit after "Items" still correctly captures the primary item
# (2, our actual 10-Q extraction target) even though the second item
# folded into the same heading can't get its own separate boundary.
# Some filers (EXPE confirmed) prefix the real body heading with the Part
# label itself — "Part I. Item\xa02. Management's Discussion..." — instead
# of restating "Item 2" alone. Without this, ITEM_RE's `^\s*(?:SEP)*`
# can't skip "Part I." (SEP is punctuation-only, no letters), so the whole
# line silently fails to match and the heading is invisible. The optional
# leading group only accepts "Part" + a roman numeral + SEP punctuation
# directly before "Item" — never arbitrary words — so it can't swallow an
# unrelated sentence that merely mentions "Part" earlier in the line.
ITEM_RE = re.compile(
    rf"^\s*(?:{SEP})\s*(?:Part{SEP}[IVX]{{1,3}}\.?{SEP})?Items?{SEP}(\d{{1,2}})(?!\d)(?:\.?([A-C])(?![a-z])|\s*\(([A-C])\))?\.?{SEP}",
    re.IGNORECASE,
)


def _item_key(match: re.Match) -> str:
    return match.group(1) + (match.group(2) or match.group(3) or "").upper()

# A small number of filers use a non-standard heading instead of the
# expected "Item N. Title" text (e.g. Honeywell's 10-K Item 1 heading is
# just "ABOUT HONEYWELL", a lone plain-text line — still structurally a
# real heading, not a TOC row, just not spelled "Item 1"; Illumina's
# 10-Q Item 2 heading is "MANAGEMENT'S DISCUSSION & ANALYSIS" — the TOC
# entry says "Item 2" as usual, but the real body heading never restates
# it). Documented per-company/per-form exceptions, not a broad regex
# relaxation that would risk false positives elsewhere. Keyed by FORM
# because the same alias text can mean a different item on each form
# (a bare "MANAGEMENT'S DISCUSSION..." heading is Item 7 on a 10-K, Item
# 2 on a 10-Q) — general_segment's `form` parameter selects which set
# applies; unrecognized/omitted forms get no aliases at all (regex-only).
# A bare "MANAGEMENT'S DISCUSSION AND/& ANALYSIS..." heading (no "Item N"
# restated anywhere near it) turned out NOT to be an Illumina-only quirk —
# GE's 10-K uses the exact same bare heading for Item 7 ("MANAGEMENT'S
# DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION AND RESULTS OF OPERATIONS
# (MD&A)."), confirmed by direct inspection. One shared pattern, applied to
# whichever item number it means on each form (Item 7 on a 10-K, Item 2 on
# a 10-Q — see ITEM_ALIASES' docstring above).
# Two more confirmed variants of the same bare heading, folded into one
# alias regex rather than a new dict entry (still keyed to the same fixed
# item number, form-by-form):
#   - ETR: "Management's FINANCIAL Discussion and Analysis" — one extra
#     word inserted between "Management's" and "Discussion" (combined-
#     utility-registrant filing convention).
#   - MRNA: "2. MANAGEMENT'S DISCUSSION AND ANALYSIS..." — a bare item
#     number (no "Item" word) directly prefixing the same phrase. The
#     digit prefix is only accepted glued to this exact phrase, so it
#     can't false-match an unrelated numbered list line elsewhere.
_MDA_ALIAS_RE = re.compile(
    r"^\s*\|?\s*(?:\d{1,2}\.\s*)?MANAGEMENT.S\s+(?:FINANCIAL\s+)?DISCUSSION\s*(?:&|AND)\s*ANALYSIS\b",
    re.IGNORECASE)

ITEM_ALIASES: dict[str, dict[str, re.Pattern]] = {
    "10-K": {
        "1": re.compile(r"^\s*ABOUT\s+HONEYWELL\s*$", re.IGNORECASE),
        "7": _MDA_ALIAS_RE,
    },
    "10-Q": {
        "2": _MDA_ALIAS_RE,
    },
}

# A TOC row is literally a markdown hyperlink: "[text](#anchor)". A real
# heading is the anchor's target, plain text, never wrapped in link syntax.
LINK_RE = re.compile(r"\[[^\]]+\]\([^)]+\)")


def is_link_row(line: str) -> bool:
    return bool(LINK_RE.search(line))


def _item_heading_is_link_wrapped(line: str) -> bool:
    """True if the "Item N" TEXT ITSELF sits inside a markdown link's
    bracket span — the actual TOC-row signature — as opposed to merely
    having a link SOMEWHERE on the line. On some iXBRL/Workiva-style
    filers the real heading also contains a link, but only around the
    section TITLE, with "Item N" itself plain text outside it:
        TOC row (IP):     "| [Item 1.](#a) | ... [Financial Statements](#a)"
        TOC row (CPRT):   "| [Item 1 - Financial Statements](#a) | ..."
        real heading (IP): "ITEM 1.[Financial Statements](#a)"
    In every TOC case the "[" opens BEFORE the word "Item"; in the real-
    heading case it only opens AFTER "Item N", around the title. (Simply
    checking for a *complete* "[...]" pair within the match fails on
    CPRT's TOC row: SEP's "-" swallows past the title's opening "( " and
    ITEM_RE stops before the closing "]", so the match itself never
    contains a "]" at all — yet it's still unambiguously a TOC row.)"""
    m = ITEM_RE.search(line)
    if not m:
        return False
    matched = m.group(0)
    open_pos = matched.find("[")
    if open_pos == -1:
        return False
    item_pos = matched.upper().find("ITEM")
    return item_pos == -1 or open_pos < item_pos


def _part_line_is_bare_label(line: str) -> bool:
    """True if the line is JUST "PART I" / "PART II" (or that plus pure
    punctuation), nothing else — the signature of a lone TOC section
    divider that introduces its own table right after (BLK, CHD, EG
    confirmed: "PART I" / "PART II" as standalone lines directly above
    that Part's own pipe-table TOC), as opposed to a real heading, which
    always carries a title on the same line ("PART I – FINANCIAL
    INFORMATION", "Part I - Item 1. Business", "PART I. FINANCIAL
    INFORMATION" — confirmed on HIG, BLK's/TSN's real headings). Checked
    by what's left on the line after the "PART N" match itself: a real
    heading's remainder is a real word or two; a bare label's remainder
    is empty or pure punctuation/whitespace."""
    m = _PART_RE.search(line)
    if not m:
        return False
    rest = re.sub(r"[\s\-–—:.,]+", "", line[m.end():])
    return not rest


def _part_heading_is_narrative_mention(line: str) -> bool:
    """True when "Part N" is embedded in an ordinary sentence ABOUT that
    Part, not stating it (ALL's 10-K confirmed: "Part\xa0III of this Form
    10-K incorporates by reference certain information from the
    registrant's definitive proxy statement..." — a cross-reference
    sentence, not Part III's own opening heading). A real heading's title
    always starts with a capital letter directly after the "Part N" match
    (however punctuated: "PART I – FINANCIAL INFORMATION", "Part I, Item
    3. Quantitative..."); a narrative mention continues in lowercase, as
    the next word of the sentence ("of this Form...")."""
    m = _PART_RE.search(line)
    if not m:
        return False
    rest = line[m.end():].lstrip(" \t\xa0-–—:.,|")
    return bool(rest) and rest[0].isalpha() and rest[0].islower()


def _part_heading_is_link_wrapped(line: str) -> bool:
    """Same signature as `_item_heading_is_link_wrapped`, for `_PART_RE`
    matches. TSN's 10-Q TOC lists "PART I"/"PART II" as bare markdown links
    with no surrounding "|" table ("[PART I. FINANCIAL INFORMATION](#a)"),
    so `_part_windows`' old "no pipe = real heading" rule picked the TOC
    row as the window boundary, dragging the whole rest of the document
    (including the real Part I content) into a mislabeled "Part II"
    window. A real "PART I" heading is bare text, never link-wrapped."""
    m = _PART_RE.search(line)
    if not m:
        return False
    matched = m.group(0)
    open_pos = matched.find("[")
    if open_pos == -1:
        return False
    part_pos = matched.upper().find("PART")
    return part_pos == -1 or open_pos < part_pos


def clean_html_to_lines(html_path: Path) -> list[str]:
    """Filing HTML is stored gzip-compressed (.html.gz — iXBRL-era 10-Ks
    compress ~90%+, see scripts/us/10k/01_fetch_filings.py). Transparently
    reads plain .html too, for any file saved before that switch."""
    html_path = Path(html_path)
    opener = gzip.open if html_path.suffix == ".gz" else open
    with opener(html_path, "rt", encoding="utf-8", errors="ignore") as f:
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


# Same signal as `_part_heading_is_narrative_mention`, generalized to any
# regex match: a real heading's title starts with a capital letter right
# after the match; a narrative CROSS-REFERENCE to that item number
# continues the sentence in lowercase (INCY confirmed: "Items\xa010 (as to
# directors and Delinquent Section\xa016(a) Reports), 11, 12, 13 and 14 is
# incorporated by reference..." — a front-matter incorporation-by-
# reference notice, not Item 10's own heading; ITEM_RE still matches
# "Items 10" since "Items" with a trailing "s" is deliberately accepted —
# see ITEM_RE's docstring on NTRS). Left unfiltered by this check
# previously, such a mention can rank ahead of every real, correctly-
# ordered Item heading that follows it, and the strictly-increasing `kept`
# filter in `_segment_window` then rejects ALL of them as "out of order" —
# confirmed on INCY: one such mention at line 57 (rank 10) silently wiped
# out real Items 1 through 7A found hundreds of lines later.
_LEADING_AND_ITEM_RE = re.compile(r"^and\s+\d{1,2}[A-C]?\.?\s*", re.IGNORECASE)


def _match_is_narrative_mention(line: str, match_end: int) -> bool:
    rest = line[match_end:].lstrip(" \t\xa0-–—:.,([|")
    # A combined heading (FCX: "Items 7. and 7A. Management's Discussion
    # and Analysis...", NTRS's analogous "Items 2. and 3." — see ITEM_RE's
    # docstring) continues in lowercase too ("and 7A. Management's...").
    # Stripping that specific "and N[A-C]?." continuation before the
    # casing check tells it apart from an actual narrative sentence,
    # which never has this shape right after the item number.
    rest = _LEADING_AND_ITEM_RE.sub("", rest, count=1)
    return bool(rest) and rest[0].isalpha() and rest[0].islower()


#: A 10-Q's own items never exceed 6 (Part I: 1-4; Part II: 1,1A,2-6) — a
#: matched number past that is with near-certainty a cross-reference to
#: the ANNUAL 10-K's numbering (which does go past 6), not a real 10-Q
#: heading (TAP confirmed: "—Item\xa08 Financial Statements, Note\xa018,
#: \"Commitments and Contingencies\" in our Annual Report..." — a citation,
#: not Item 8 of THIS filing, since no such item exists on a 10-Q at
#: all). Left unfiltered, such a mention ranks ahead of the real, later
#: Item 2/3/4 headings and the strictly-increasing `kept` filter in
#: `_segment_window` rejects them all as "out of order" — the same
#: failure shape `_match_is_narrative_mention` targets, but this one's
#: title text ("Financial Statements") happens to still be capitalized,
#: so the casing signal alone doesn't catch it.
_MAX_10Q_ITEM = 6

#: O (Realty Income) restates its 10-Q's MD&A section as "Item\xa07:" —
#: a filer-specific numbering quirk (confirmed by direct inspection: the
#: real heading text is unambiguously the MD&A section, just mislabeled
#: with the 10-K's item number instead of the 10-Q's own "Item 2").
#: Since ITEM_RE already matches it (as item "7", which the guard above
#: would otherwise reject on a 10-Q), it's relabeled to "2" instead of
#: dropped — recovering real content a plain reject would just discard.
_O_MDA_MISLABEL_RE = re.compile(r"^Management.S\s+Discussion", re.IGNORECASE)


def _raw_candidates(lines: list[str], form: str = "10-K") -> list[tuple[str, int]]:
    """Every line that looks like an Item heading, TOC or real, in document
    order — no filtering yet. `form` selects which ITEM_ALIASES apply."""
    found = []
    aliases = ITEM_ALIASES.get(form, {})
    for i, line in enumerate(lines):
        m = ITEM_RE.search(line)
        if m:
            key = _item_key(m)
            base_num = int(m.group(1))
            if form == "10-Q" and base_num > _MAX_10Q_ITEM:
                if _O_MDA_MISLABEL_RE.match(line[m.end():].lstrip(" \t\xa0-–—:.,")):
                    found.append(("2", i))
                continue
            if not _match_is_narrative_mention(line, m.end()):
                found.append((key, i))
            continue
        for item_key, alias_re in aliases.items():
            if alias_re.search(line):
                found.append((item_key, i))
    return found


# A real TOC lists most/all of a 10-K's ~15 items, packed within a couple
# hundred lines. A short or sparse run isn't a TOC — e.g. a filer whose TOC
# doesn't spell "Item N" for most rows (only its real body headings match
# at all): treating that as "the TOC" would wrongly discard the one real
# heading actually found. Both guards matter: without MIN_TOC_ITEMS, 1-2
# genuinely early body headings get misread as a TOC; without MAX_TOC_GAP,
# an unrelated later match could stay in-format and get swept into a TOC
# prefix that isn't really one contiguous block.
MIN_TOC_ITEMS = 5
MAX_TOC_GAP = 150


def _is_toc_formatted_row(line: str) -> bool:
    """True for a TOC/index row, false for a real heading — however either
    is styled by a given filer's HTML->markdown conversion. Two
    independent signals, either one sufficient:
    1. `_item_heading_is_link_wrapped` — the "Item N" text itself opens
       inside a markdown link (whether the link covers just "Item N", or
       "Item N - Title" together, or the whole row) — the TOC-row
       signature ITEM_RE's own match can't always see structurally (see
       that function's docstring).
    2. Two or more non-empty cells of a "|"-delimited layout table
       ("| Item 1. | | | Consolidated Financial Statements | | |" — PSA)
       — a plain-text TOC row with no links at all, but item number and
       title still stated as two separate pieces of content. A real
       heading, even when a filer also wraps it in a "|"-padded layout
       row ("| ITEM 1. BUSINESS | | |" — ALK), is always ONE piece of
       content — exactly one non-empty cell, not two."""
    if _item_heading_is_link_wrapped(line):
        return True
    non_empty_cells = [c for c in (cell.strip() for cell in line.split("|")) if c]
    return len(non_empty_cells) >= 2


PROSE_MIN_LEN = 120

# Filers confirmed, by direct inspection, to table-format EVERY real body
# heading exactly like a TOC row ("| ITEM 1. | | | BUSINESS | | |"), not
# just the TOC itself — all Workiva-generated iXBRL 10-Ks. For these
# specific filers only, `_toc_prefix_end` additionally stops the TOC
# prefix at the first table-formatted candidate immediately followed by
# real prose (`_is_followed_by_prose`), since shape alone can't tell their
# real headings from TOC rows. Tried this as a general rule first — it
# regressed roughly a third of a 80-filing sample (WDC lost all 3 target
# items outright; MA, INCY, PXD, TDY, CTAS, IR, BXP, JNJ lost most of
# their Item 1/1A/7 content) — false-positive "prose" lines (a short
# sub-heading, a page-break artifact) exist elsewhere in this corpus that
# happen to clear `PROSE_MIN_LEN` with no "|", so the extra check must be
# scoped to filers that actually need it, not applied corpus-wide.
WORKIVA_TABLE_HEADING_TICKERS = {"WFC", "SYK", "ADM", "KR", "AIG", "PCAR", "WMT", "UAA", "LHX"}


_JUNK_LINE_MAX_LEN = 3  # a lone zero-width space, bullet, or similar filler
_JUNK_LOOKAHEAD = 5     # how many such junk lines to skip before giving up


def _is_followed_by_prose(lines: list[str], line_idx: int, min_len: int = PROSE_MIN_LEN) -> bool:
    """True if a real paragraph follows `line_idx` within a few lines —
    long, no "|" table formatting — rather than another TOC row. Some
    filers (KR confirmed) insert a near-invisible filler line (a lone
    zero-width space, `\\u200b`) between a real table-formatted heading and
    its body paragraph; `clean_html_to_lines` keeps it (Python's `.strip()`
    doesn't treat `\\u200b` as whitespace), so checking only the immediate
    next line missed the real prose sitting one line further. Skipping a
    short run of such near-empty junk lines before giving up on "no prose
    follows" fixes that without weakening the TOC-row rejection: a genuine
    TOC entry is followed by another TOC entry (another "|" row) however
    many junk lines separate them, never by a real paragraph.
    See `WORKIVA_TABLE_HEADING_TICKERS` for why this check exists and why
    it is NOT applied to every filer."""
    idx = line_idx + 1
    skipped = 0
    while idx < len(lines) and skipped <= _JUNK_LOOKAHEAD:
        candidate = lines[idx]
        if len(candidate) <= _JUNK_LINE_MAX_LEN:
            idx += 1
            skipped += 1
            continue
        return len(candidate) >= min_len and "|" not in candidate
    return False


def _toc_prefix_end(raw: list[tuple[str, int]], lines: list[str], ticker: str | None = None) -> int:
    """Index into `raw` (not a line number) of the first candidate that is
    NOT part of the initial run of TABLE-ROW-FORMATTED candidates starting
    at raw[0] — i.e. where the TOC ends and body content begins. Returns 0
    (no TOC prefix at all) if that initial run is too short/sparse to
    plausibly be a real table of contents.

    Previously this walked the candidates' ITEM NUMBERS, requiring a
    strictly-increasing, no-repeat run — but a 10-Q's single combined TOC
    lists Part I's items (1-4) then RESTARTS at Part II's item 1, so the
    TOC itself contains a rank decrease after only 4 items (below
    MIN_TOC_ITEMS), indistinguishable by rank alone from real body content
    starting (the ambiguity _part_windows' docstring describes). Table-row
    formatting sidesteps it entirely: both Part I's and Part II's listings
    in the combined TOC are table rows, so the prefix keeps extending
    across the restart; the first candidate that ISN'T a table row is
    unambiguously the first real heading, regardless of what number it
    is.

    For `ticker in WORKIVA_TABLE_HEADING_TICKERS` only, one more stop
    condition applies: those filers table-format every real body heading
    too, so a table-formatted candidate immediately followed by an actual
    paragraph (`_is_followed_by_prose`) ends the prefix right there."""
    check_prose = ticker in WORKIVA_TABLE_HEADING_TICKERS
    last_line_idx = None
    prefix_len = 0
    for _item_key, line_idx in raw:
        if not _is_toc_formatted_row(lines[line_idx]):
            break
        if check_prose and _is_followed_by_prose(lines, line_idx):
            break
        if last_line_idx is not None and line_idx - last_line_idx > MAX_TOC_GAP:
            break
        last_line_idx = line_idx
        prefix_len += 1
    return prefix_len if prefix_len >= MIN_TOC_ITEMS else 0


# A 10-K numbers items 1 through 16 once, globally. A 10-Q numbers Part I
# (Items 1-4) and Part II (Items 1-6) INDEPENDENTLY — Part II's "Item 1" is
# Legal Proceedings, unrelated to Part I's "Item 1" Financial Statements,
# and it comes right after Part I's own Item 4. Tried inferring the
# restart purely from item-rank decreasing; broke on real 10-Qs whose
# single combined TOC lists BOTH parts' items back to back (so the TOC
# ITSELF contains a rank decrease — indistinguishable from a real Part II
# starting, by rank alone). Every 10-K/10-Q instead labels this
# explicitly, in the text: a literal "PART I" / "PART II" heading. Using
# that as the authoritative split point sidesteps the ambiguity entirely
# — each Part's own window of lines gets the ORIGINAL single-sequence
# logic (TOC-prefix + monotonic ordering) run independently, and since the
# combined TOC sits before Part I's real heading, it's excluded from both
# windows automatically, not something either window needs to strip again.
_PART_RE = re.compile(rf"^\s*(?:{SEP})\s*PART{SEP}(I{{1,3}}|IV)\b", re.IGNORECASE)
_ROMAN_RANK = {"I": 1, "II": 2, "III": 3, "IV": 4}

# Many large filers (GE, INTC, WFC, CAH, MCD, ILMN's 10-Q, ...) include a
# SECOND, formal index near the END of the document — a real, named SEC-
# filing convention ("FORM 10-K CROSS REFERENCE INDEX" / "FORM 10-Q
# CROSS-REFERENCE INDEX") mapping each Item number to a page number, distinct
# from the front-matter TOC. For these filers the front TOC and body headings
# often DON'T restate "Item N"/"Part N" in a form our regex matches (company-
# specific bare headings instead — "ABOUT GE AEROSPACE" for Item 1,
# "MANAGEMENT'S DISCUSSION AND ANALYSIS..." for Item 7, both without "Item N"
# anywhere nearby), leaving this trailing index as the ONLY textually-
# matchable "Part I"/"Item N" occurrences in the whole document.
# _part_windows' fallback ("last occurrence of any form") then anchors
# windows at ~95%+ through the document, excluding virtually all real
# content (confirmed on GE: window "I" started at line 2448 of 2543).
_CROSS_REF_INDEX_RE = re.compile(r"CROSS[\s-]*REFERENCE\s+INDEX", re.IGNORECASE)


def _trailing_index_start(lines: list[str]) -> int:
    """Absolute line index where a TRAILING cross-reference index begins
    (its own header line, e.g. "FORM 10-K CROSS REFERENCE INDEX"), or
    len(lines) if none found — i.e. no candidate should be excluded.
    Some filers (PNC) instead put a MULTI-PAGE cross-reference index near
    the FRONT of the document, repeating "Cross-Reference Index...
    (continued)" on every one of its own pages — real Item content then
    follows it. Treating that last "(continued)" repeat as a trailing-
    index start would exclude the entire rest of the document, including
    the real content (confirmed regression: PNC's Item 1/1A/7 all
    vanished this way). The last occurrence is only trustworthy as a
    genuine trailing index when it falls in roughly the back HALF of the
    document — GE's/ILMN's real trailing indexes sit past 95% through;
    PNC's front-loaded one only reaches ~4% through even at its last
    repeat."""
    last = None
    for i, line in enumerate(lines):
        if _CROSS_REF_INDEX_RE.search(line):
            last = i
    if last is None or last < len(lines) / 2:
        return len(lines)
    return last


def _part_windows(lines: list[str]) -> list[tuple[str, int, int]]:
    """[(part_label, window_start, window_end), ...] in document order.
    Falls back to a single ("I", 0, len(lines)) window when no explicit
    PART marker is found (or only one) — covers filings that don't use
    "PART" labeling at all, same as before this function existed.

    Unlike Item headings, a PART marker's real heading is NOT reliably
    distinguishable from its TOC row by link-syntax alone: some filers
    (iXBRL/Workiva-style, e.g. IP's 10-Qs) render even the real "PART I"/
    "PART II" heading as a self-referencing anchor link, identical in
    form to the TOC row. What IS reliable: an index/TOC's PART row is
    always one cell of a markdown table (the whole line is a "|"-
    delimited table row, however it's linked inside); a real PART heading
    is always its own standalone line, plain text or a lone link, never a
    table row. So prefer a non-table-row occurrence (the real heading,
    whether linked or not); some filers (e.g. PSA's 10-Qs) never repeat
    "PART I" as a real heading at all (they go from the TOC row straight
    into "Item 1." with no restated Part heading) — for those, fall back
    to the last occurrence of any form rather than finding nothing, since
    a too-early window start is still better than silently dropping the
    Part split.

    FIRST non-table occurrence, not last: some filers (MSFT) repeat "PART
    I"/"PART II" as a running PAGE HEADER on every page of that Part's
    content — dozens of non-table occurrences, not one. Taking the LAST
    of those (the original design, sized for the common case of exactly
    one restated heading) picks the Part's very LAST page as the window
    start, excluding essentially all of that Part's real content from its
    own window (confirmed: MSFT's Item 1/1A vanished entirely this way —
    the window started 34 "PART I" page-headers too late)."""
    trailing_index_start = _trailing_index_start(lines)
    # Every occurrence of each label is kept (not just the first), grouped
    # into two tiers — bare-text-or-link (never a TOC row) and plain
    # non-table (could still be a TOC row on filers with no "|" in their
    # TOC at all) — falling back to table-row occurrences only when a
    # label has neither. A single "first occurrence wins" pick (the
    # original design) turned out unsafe in BOTH directions: some filers
    # repeat a bare "PART I" / "PART II" pair as a running header sitting
    # right at/before the TOC — TWICE, close together — before the real,
    # correctly-spaced heading later on (PVH, WBA confirmed: first pair
    # only ~15-25 lines apart, real pair 1000+ lines apart); a fixed
    # "first" pick locks onto the bogus early pair. Keeping every
    # occurrence and validating candidate PAIRS by span (below) recovers
    # the real heading without reintroducing the MSFT running-page-header
    # regression the original "first, not last" rule fixed (MSFT's real
    # heading IS its first occurrence, so the earliest-valid-candidate
    # search still lands on it).
    # Link-wrapped non-table lines are excluded outright — a real heading
    # never links to itself (see `_part_heading_is_link_wrapped`), so
    # these are TOC rows regardless of position (TSN confirmed). A bare
    # "PART I" divider line with no title (BLK/CHD/EG's lone-line-above-
    # its-own-TOC-table pattern) is NOT excluded here, unlike the
    # original single-candidate design: it turned out to ALSO be the
    # genuine section-opening heading on other filers (ALL confirmed:
    # "Part\xa0I" bare, directly above the real "Item\xa01.\xa0Business").
    # The span validation below (not a pre-filter on this one shape)
    # is what tells the two apart — BLK/CHD/EG's bogus pair sits right
    # next to its own TOC table (small span, rejected), ALL's real pair
    # spans the whole Part I content (large span, accepted).
    non_table_occ: dict[str, list[int]] = {}
    table_occ: dict[str, list[int]] = {}
    for i, line in enumerate(lines):
        if i >= trailing_index_start:
            break  # everything from here on is the trailing cross-reference index
        m = _PART_RE.search(line)
        if not m or _part_heading_is_narrative_mention(line):
            continue
        label = m.group(1).upper()
        if "|" in line:
            table_occ.setdefault(label, []).append(i)
            continue
        if not _part_heading_is_link_wrapped(line):
            non_table_occ.setdefault(label, []).append(i)

    def pool(label: str) -> list[int]:
        return non_table_occ.get(label) or table_occ.get(label) or []

    ranks_present = sorted(
        {l for l in set(non_table_occ) | set(table_occ) if l in _ROMAN_RANK},
        key=lambda l: _ROMAN_RANK[l])
    if len(ranks_present) < 2:
        return [("I", 0, len(lines))]

    # A real Part I always contains substantial content (at minimum Item
    # 1's financial statements) — a candidate pair narrower than this is
    # near-certainly TOC/page-header noise, not a genuine section
    # boundary (confirmed: ABT's only bare pair sits 4 lines apart, a
    # trailing running-header artifact, not real headings).
    MIN_PART_SPAN = 100

    chosen: list[tuple[str, int]] = []
    prev_idx = -1
    for rank_i, label in enumerate(ranks_present):
        candidates = [i for i in pool(label) if i > prev_idx]
        if not candidates:
            break
        if rank_i == len(ranks_present) - 1:
            chosen_idx = candidates[0]
        else:
            next_pool = pool(ranks_present[rank_i + 1])
            chosen_idx = next(
                (c for c in candidates if any(n - c >= MIN_PART_SPAN for n in next_pool)),
                candidates[0])  # no validated pair found; best effort, unchanged from before
        chosen.append((label, chosen_idx))
        prev_idx = chosen_idx

    if len(chosen) < 2 or chosen[1][1] - chosen[0][1] < MIN_PART_SPAN:
        # Even the best candidate pair found is implausibly narrow (no
        # real alternative existed — confirmed on DUK, whose only "PART
        # I"/"PART II" occurrences are a single adjacent table-row pair
        # with no real body heading restated at all). Trust the Item-
        # level segmentation over a fabricated Part split.
        return [("I", 0, len(lines))]

    windows = []
    for i, (label, start_idx) in enumerate(chosen):
        end_idx = chosen[i + 1][1] if i + 1 < len(chosen) else len(lines)
        windows.append((label, start_idx, end_idx))
    return windows


def _segment_window(
    lines: list[str],
    window_start: int,
    window_end: int,
    suffix: str,
    toc_end_line: int,
    assume_starts_at_item1: bool = False,
    form: str = "10-K",
) -> dict[str, str]:
    """The original single-sequence segmentation logic, scoped to one
    [window_start, window_end) slice of lines. `toc_end_line` is the
    absolute line index of the last row in the document's ONE, GLOBAL TOC
    prefix (see general_segment) — candidates at or before it are TOC
    noise and dropped; everything after is body, regardless of which
    window it falls in. Re-running _toc_prefix_end per window (the first
    Part-windowing attempt) was wrong: once windowing already excludes the
    real TOC block (which always precedes Part I's own start), a window's
    OWN real headings are themselves a strictly-increasing, tightly-packed
    run that _toc_prefix_end can't distinguish from an actual TOC — it
    stripped Part I's real Item 1/1A/1B/1C/2/3/4 headings outright on
    filings like APD, where the TOC sits at absolute lines 94-124 but the
    "PART I" marker (and window) only starts at line 187."""
    window_lines = lines[window_start:window_end]
    raw_local = _raw_candidates(window_lines, form=form)
    body = [(key, window_start + i) for key, i in raw_local if window_start + i > toc_end_line]

    candidates_by_item: dict[str, int] = {}
    for item_key, line_idx in body:
        # _item_heading_is_link_wrapped, not is_link_row: a real heading on
        # some filers (e.g. IP's 10-Qs) also carries a link around its
        # title, which is_link_row's "any link on the line" test can't
        # tell apart from an actual TOC row.
        if _item_heading_is_link_wrapped(lines[line_idx]):
            continue
        # FIRST real occurrence wins, not last: some filers (MSFT) repeat
        # a running-page-header form of the item heading ("Item 1") many
        # times throughout that section's own pages, in ADDITION to the
        # one real title line ("ITEM 1. BUSINESS"). Overwriting on every
        # occurrence used to pick the section's very LAST page as its
        # start, losing virtually all of it (confirmed: MSFT's Item 1A
        # collapsed from its real ~74K chars down to a ~2K tail this way).
        # The first non-link, non-toc-row occurrence of a key is always
        # at or before that section's real content begins.
        if item_key not in candidates_by_item:
            candidates_by_item[item_key] = line_idx

    ordered = sorted(candidates_by_item.items(), key=lambda kv: kv[1])
    kept = []
    last_rank = (-1, -1)
    for item_key, line_idx in ordered:
        rank = item_sort_key(item_key)
        if rank >= last_rank:
            kept.append((item_key, line_idx))
            last_rank = rank

    # Some filers (CPRT, GPC, MLM) never restate a real "Item 1" heading in
    # text at all — the financial statements just start under their own
    # natural headers ("Copart, Inc. / Consolidated Balance Sheets") right
    # after the TOC, with no "Item 1" marker anywhere to find. No text
    # search can recover a heading that was never written. But Item 1 is
    # ALWAYS Part I's first item on both forms (a fixed, known fact, same
    # as _PART_I_ITEM_KEYS) — so when the caller has verified this window
    # really is Part I and its real "1" heading is missing, the body's
    # leading span (up to whatever real heading WAS found first) can only
    # be Item 1's content by elimination; label it as such instead of
    # leaving it a silent gap.
    #
    # Checking "is '1' ANYWHERE in kept", not just "is it kept[0]": some
    # filers (NTRS) order their body content out of Item-number sequence —
    # a combined "Items 2. and 3." MD&A heading comes BEFORE the real
    # "Item 1" heading further down. kept[0] != "1" there doesn't mean
    # Item 1 is missing, just that it isn't first; inserting a synthetic
    # "1" anyway would override real content with a 1-line stub (confirmed
    # regression: NTRS's real Item 1 collapsed from ~30K chars to "PART I –
    # FINANCIAL INFORMATION", 30 chars).
    body_start = max(window_start, toc_end_line + 1)
    if assume_starts_at_item1 and "1" not in dict(kept) and body_start < window_end:
        kept.insert(0, ("1", body_start))

    segments = {}
    for i, (item_key, start_idx) in enumerate(kept):
        end_idx = kept[i + 1][1] if i + 1 < len(kept) else window_end
        segments[item_key + suffix] = "\n".join(lines[start_idx:end_idx])
    return segments


def general_segment(lines: list[str], form: str = "10-K", ticker: str | None = None) -> dict[str, str]:
    """One segment per recognizable "Item N[A-C]" heading. `form` ("10-K"
    or "10-Q") selects which ITEM_ALIASES apply — see that dict's
    docstring for why the same alias text can mean a different item
    depending on the form. The TOC prefix
    (see _toc_prefix_end) is computed ONCE, globally, over the whole
    document — a table of contents only ever appears once, before
    anything else, regardless of how many Part windows follow it. That
    global boundary is then applied when segmenting every window (see
    _segment_window) instead of re-deriving a per-window TOC, which broke
    on windows whose own real headings looked like a plausible TOC once
    the actual TOC (sitting before the window even starts) was excluded.

    A 10-K's items never repeat across its 4 Part windows (Part I ends at
    Item 4, Part II picks up at Item 5, ...), so every key there stays
    bare ("1", "1A", "7", "8", ...) exactly as before Part-windowing
    existed — 02_extract_sections.py's TARGET_ITEMS lookups are
    untouched. A 10-Q's Part II genuinely REUSES Part I's item numbers
    (Item 1 = Legal Proceedings, not Financial Statements) — only a real
    key COLLISION like that gets suffixed "__partII" (checked per-key as
    windows are merged, not by window index — an index-based rule would
    incorrectly suffix a 10-K's later, non-colliding Part II/III/IV items
    too). This is a heuristic, not a guarantee, but it no longer depends
    on any arbitrary length number.

    _PART_I_ITEM_KEYS below covers the two cases actual per-key collision
    detection alone MISSES: a filer (e.g. CPRT) whose Part I never
    restates its real heading in text at all, so window I's own segments
    dict never contains "1" — nothing to collide against, so Part II's
    real "1" (Legal Proceedings) would otherwise fall through and silently
    become the bare "1" key, masquerading as Part I's Financial
    Statements. Every key SEC Form 10-K/10-Q's Part I can ever contain is
    fixed and known ahead of time (Item 1/1A/1B/1C/2/3/4) — reserving it
    for the FIRST window regardless of what that window actually found
    fixes this without reverting to a pure index-based rule (which would
    wrongly suffix a 10-K's legitimate, non-colliding Part II+ items)."""
    all_raw = _raw_candidates(lines, form=form)
    toc_prefix_len = _toc_prefix_end(all_raw, lines, ticker=ticker)
    toc_end_line = all_raw[toc_prefix_len - 1][1] if toc_prefix_len > 0 else -1

    windows = _part_windows(lines)
    # Only trust "the first window IS Part I" when _part_windows actually
    # found a real "I" label to start from. A handful of filers (DUK, AEP)
    # are COMBINED filings for multiple registrants, each repeating its own
    # PART I-IV structure — _part_windows' "last occurrence wins" picks up
    # a later registrant's marker and produces a first window labeled
    # "II"/"III"/etc. Reserving Part I's keys for a window that isn't
    # actually Part I would wrongly suffix real content (confirmed: DUK's
    # Item 1/1A/1B/1C/2/3/4 all ended up misdetected under a catch-all
    # "IV" window and got wrongly suffixed "__partIV"). Safer to skip the
    # reservation entirely than trust window semantics we can't verify.
    reservation_active = windows[0][0] == "I"

    segments: dict[str, str] = {}
    for label, start_idx, end_idx in windows:
        window_segments = _segment_window(
            lines, start_idx, end_idx, suffix="", toc_end_line=toc_end_line,
            assume_starts_at_item1=reservation_active and label == "I",
            form=form,
        )
        for key, text in window_segments.items():
            reserved_for_first_window = (
                reservation_active and label != "I" and key in _PART_I_ITEM_KEYS
            )
            if key in segments or reserved_for_first_window:
                segments[f"{key}__part{label}"] = text
            else:
                segments[key] = text

    _recover_item_8(lines, segments)
    return segments


# The complete, fixed set of Item keys Part I can ever contain in either
# form (10-K: 1, 1A, 1B, 1C, 2, 3, 4; 10-Q: 1, 2, 3, 4 — a strict subset).
# Not a tuning knob: this is SEC Form 10-K/10-Q's actual, legally fixed
# structure, the same kind of fact TARGET_ITEMS already leans on elsewhere
# (e.g. "7" is always MD&A on a 10-K).
_PART_I_ITEM_KEYS = {"1", "1A", "1B", "1C", "2", "3", "4"}


# Item 8 (Financial Statements) frequently ISN'T bounded correctly by the
# generic "next Item heading" logic that works for every other item, for
# two distinct reasons found by direct inspection:
#   1. A one-line STUB — "The information required by this Item is set
#      forth in our Consolidated Financial Statements... included in this
#      Annual Report" (or "submitted as a separate section... See Part IV,
#      Item 15") — with the real statements starting elsewhere, under
#      their own natural headers, never repeating "Item 8" nearby at all.
#   2. "Item 8" reused as a running PAGE-HEADER stamp throughout the
#      financial statements/audit report (seen verbatim in MSFT's filing:
#      "PART II / Item 8 / Income Taxes – Uncertain Tax Positions..." —
#      that's a page footer, not a heading). "Last occurrence wins" then
#      picks whichever page happened to repeat the header last, often near
#      the END of the real content, producing a tiny leftover segment.
# Neither case leaves a real Item-8-labeled marker at the actual start of
# content, so no boundary-heading logic can find it. What both cases share:
# a real Item 8 (any real company with revenue) is never a few thousand
# characters — the recovery just fires whenever the segment is
# implausibly short, and looks for a near-universal alternate anchor every
# audited 10-K's financial statements begin with: the opening line of the
# audit report. This does NOT extend to other items — it's specific to
# Item 8 because that phrase is uniquely load-bearing for it.
_AUDIT_REPORT_ANCHOR_RE = re.compile(
    r"Report of Independent Registered Public Accounting Firm", re.IGNORECASE
)
# No real company's audited financial statements + notes fit under this
# many characters — anything shorter is a stub, a page-header artifact, or
# a truncated segment, not real Item 8 content.
_ITEM_8_MIN_PLAUSIBLE_CHARS = 3000


def _recover_item_8(lines: list[str], segments: dict[str, str]) -> None:
    current = segments.get("8", "")
    if len(current) >= _ITEM_8_MIN_PLAUSIBLE_CHARS:
        return

    # Some filers issue TWO audit opinions back to back — one on internal
    # control (ICFR), one on the financial statements themselves — with
    # Part II/III administrative stubs (Items 9B-15, "None"/incorporated-
    # by-reference one-liners) packed in BETWEEN them, not after both.
    # Tried "next kept item" and then "next item >= 10" as the end
    # boundary; both failed on real filings (MNST, CZR) because those
    # stub items sit before the real numbers, not after — there is no
    # single Item-number boundary that reliably marks the end across
    # filers. The LAST occurrence of the anchor phrase is the one that
    # actually precedes the real financial statements (confirmed: CZR's
    # first occurrence at line 1226 was the ICFR opinion; a second real
    # occurrence followed at 1591, right before the actual numbers).
    anchor_candidates = [
        i for i, line in enumerate(lines) if _AUDIT_REPORT_ANCHOR_RE.search(line) and not is_link_row(line)
    ]
    if not anchor_candidates:
        return  # no recovery anchor found; leave the stub (or empty) as-is
    anchor_idx = anchor_candidates[-1]

    # End of document, not the next Item boundary — Item 8's real content
    # (the full financial statements + notes) runs to (near) the end of
    # the filing's substantive text in every case checked; a stray later
    # Item match is exhibit-index/signature-block noise, safer to include
    # than risk cutting off real content again.
    recovered = "\n".join(lines[anchor_idx:])
    if len(recovered) > len(current):
        segments["8"] = recovered


def load_extraction_trace(config: dict, output_prefix: str = "filing_sections") -> pd.DataFrame:
    """The FULL trace: one row per (filing, target item) — found or not —
    from every extraction run's part-files (glob'd, never a single
    hardcoded filename; new runs just add more parts). `output_prefix`
    selects which instrument: "filing_sections" (10-K, the default — kept
    for backward compatibility with every existing caller) or
    "filing_sections_10q" (10-Q — never pooled with 10-K, see the
    project's data-scope decision). Columns: accession_number, ticker,
    filing_date, item_key, section_name, found, section_text, char_len,
    word_count, note, run_id, run_date, run_comment, part_num, part_file.
    Collapsed to the most recent run_date per (accession_number, item_key)
    so a rerun's rows supersede the old ones."""
    sections_dir = Path(config["storage"]["interim_sections"])
    part_paths = sorted(sections_dir.glob(f"{output_prefix}__run=*__part=*.parquet"))
    if not part_paths:
        raise FileNotFoundError(
            f"No {output_prefix}__run=*__part=*.parquet files under {sections_dir}. "
            "Run 02_extract_sections.py first."
        )
    trace = pd.concat((pd.read_parquet(p) for p in part_paths), ignore_index=True)
    trace = trace.sort_values("run_date").drop_duplicates(
        subset=["accession_number", "item_key"], keep="last"
    )
    return trace


def load_filing_sections(config: dict, output_prefix: str = "filing_sections") -> pd.DataFrame:
    """Just the FOUND sections (found == True) from load_extraction_trace —
    what a consumer that only cares about actual extracted text wants.
    For the full found/not-found picture, use load_extraction_trace."""
    trace = load_extraction_trace(config, output_prefix=output_prefix)
    return trace[trace["found"]].reset_index(drop=True)
