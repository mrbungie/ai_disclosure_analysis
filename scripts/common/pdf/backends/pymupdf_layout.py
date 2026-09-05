"""
scripts/common/pdf/backends/pymupdf_layout.py — the CPU backend: PyMuPDF's
own layout analysis plus the content heuristics this project accumulated
while auditing real CMF filings. This is scripts/cl/cmf_pdf_paragraphs.py,
moved here and made country-agnostic (its CMF-specific regexes now live in
scripts/common/pdf/profiles.py) — the heuristics themselves were never
Chilean, they were about what PDF text extraction does to any document.

Its known limits, measured against ground truth on Banco de Chile's
Memoria Anual 2024 (see docs/analytics/pdf-backend-poc.md) and NOT fixable
with more regexes, are why the VLM backends exist:
  - it recovered 42 of 52 numeric cells on a 4-table liquidity page,
    silently dropping a whole column,
  - it has no notion of page furniture beyond these heuristics, so a
    margin marker at the same height as a paragraph gets spliced INTO
    that paragraph's text,
  - it cannot see a page that has no text layer at all.
It is kept, not replaced, because it is ~150x faster per page and is the
right tool for the pages that don't need more (see pipeline.py's triage).
"""

import bisect
import os
import re
from collections import Counter

import pymupdf

from scripts.common.pdf import blocks as B
from scripts.common.pdf.profiles import PdfProfile, get_profile

_DATE_TOKEN_RE = re.compile(r"^\d{1,2}[/-][A-Za-zÀ-ÿ]{0,4}[/-]?\d{2,4}$")
_NUMERIC_TOKEN_RE = re.compile(r"^\(?-?[\d.,]+%?\)?$")
_ALL_CAPS_HEADING_RE = re.compile(r"^[^a-zà-ÿ]+$")
_NUMBERED_TOC_ENTRY_RE = re.compile(r"^\d{1,3}\.?\s+\S")
_HEAVY_WEIGHT_FONT_RE = re.compile(r"(bold|black|heavy|semibold|extrabold|medium)", re.IGNORECASE)
_LIGHT_WEIGHT_FONT_RE = re.compile(r"(light|regular|book|thin|roman)", re.IGNORECASE)
# A prose block that doesn't end in one of these is, empirically, not a
# finished paragraph — verified on a real filing where PyMuPDF split one
# continuous sentence into 3 same-page blocks with nothing between them.
_SENTENCE_END_RE = re.compile(r"[.!?:]['\"”’)]*\s*$")


def _normalize_for_repeat_check(text: str) -> str:
    """Collapses digit runs (page numbers, dates) so the same header
    template with a different page number counts as the same text."""
    return re.sub(r"\d+", "#", text.strip())


def _extract_text_blocks(page) -> list[tuple]:
    """[(bbox, raw_text, font_names), ...] for one page's TEXT blocks.
    Span font NAMES are kept alongside the text because
    `_is_uniform_heavy_font_label` needs them and PyMuPDF's span `flags`
    bold bit can't substitute (verified: flags was constant across both a
    bold sub-heading and a real sentence that merely started with a bold
    marker — that PDF encodes weight only in the font's PostScript name)."""
    out = []
    for block in page.get_text("dict")["blocks"]:
        if block.get("type") != 0:
            continue
        lines_text, font_names = [], set()
        for line in block["lines"]:
            lines_text.append("".join(span["text"] for span in line["spans"]))
            font_names.update(span["font"] for span in line["spans"] if span["text"].strip())
        out.append((tuple(block["bbox"]), "\n".join(lines_text), frozenset(font_names)))
    return out


def _find_repeated_furniture(pages_blocks: list[list[tuple]], min_page_fraction: float) -> set[str]:
    """A block whose normalized text appears on at least `min_page_fraction`
    of a document's pages is template furniture, not content — verified
    needed on AFP Capital's Memoria Integrada, whose header/footer template
    is completely different from CMF's own. Every filer's PDF template is
    its own, so a hardcoded regex per template doesn't scale, but "this
    exact text reappears on a third of the pages" holds regardless."""
    counts = Counter()
    for page_blocks in pages_blocks:
        seen = set()
        for _bbox, raw_text, _fonts in page_blocks:
            norm = _normalize_for_repeat_check(raw_text)
            if norm and norm not in seen:
                counts[norm] += 1
                seen.add(norm)
    threshold = max(3, len(pages_blocks) * min_page_fraction)
    return {norm for norm, count in counts.items() if count >= threshold}


def _bbox_overlaps(bbox, table_bboxes, threshold: float = 0.4) -> bool:
    """True if bbox is mostly covered by a table's bbox — a block inside a
    detected table region is that table's content, not prose, even though
    PyMuPDF's block and table segmentation run independently. 0.4 rather
    than something near 1.0: verified on a real block ("PATRIMONIO TOTAL",
    a table's own last row) whose bbox included a trailing blank line
    stretching past the table's bottom edge, measuring only 48% overlap
    despite being entirely the table's content."""
    x0, y0, x1, y1 = bbox
    area = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    if area == 0:
        return False
    for tx0, ty0, tx1, ty1 in table_bboxes:
        inter = (max(0.0, min(x1, tx1) - max(x0, tx0))
                 * max(0.0, min(y1, ty1) - max(y0, ty0)))
        if inter / area > threshold:
            return True
    return False


def _is_numeric_heavy(text: str, threshold: float = 0.4) -> bool:
    """Fallback for tables page.find_tables() misses ENTIRELY — verified on
    a real filing whose segment-revenue table had no ruling lines to grid
    on, so it never appeared in tables.tables at all. threshold=0.4, not
    >0.5: a label-plus-values table row is close to a 50/50 split almost by
    definition (verified on "Otras ganancias (pérdidas) 133.311 133.311
    133.311", exactly 3 numeric of 6 tokens)."""
    tokens = text.split()
    if not tokens:
        return False
    numeric = sum(1 for t in tokens if _NUMERIC_TOKEN_RE.match(t) or _DATE_TOKEN_RE.match(t))
    return numeric / len(tokens) >= threshold


def _is_heading_like(text: str) -> bool:
    """Short ALL-CAPS / numbered blocks: section headings, table-of-contents
    entries, and the section-navigation strip printed across the top of
    every glossy Memoria page — e.g. "PRINCIPALES INDICADORES", "9.
    Elección de nuevo Directorio.", "GOBIERNO CORPORATIVO".

    This heuristic CANNOT tell a real in-document heading from navigation
    chrome: both are short, capitalized, verbless blocks. They are all
    treated as furniture, which is what this backend has always done and
    what keeps the nav strip out of the corpus — at the cost of also
    losing real headings. Recovering the distinction needs an actual layout
    model, and is one of the concrete things the VLM backends buy (they
    return `header` for the nav strip and `title` for the heading).

    Deliberately NARROW (<=10 words): a false positive here silently drops
    real content, which is worse for a disclosure-detection corpus than a
    false negative leaving one heading unfiltered."""
    words = text.split()
    if not words or len(words) > 10 or not any(c.isalpha() for c in text):
        return False
    return bool(_ALL_CAPS_HEADING_RE.match(text) or _NUMBERED_TOC_ENTRY_RE.match(text))


def _is_uniform_heavy_font_label(font_names: frozenset, word_count: int) -> bool:
    """Font-weight signal for sub-headings `_is_heading_like` can't catch by
    content — verified on Banco de Chile 2024 notes: "(b) Instrumentos
    financieros de deuda:" renders entirely in 'BCH_0515-Medium', while a
    real sentence starting with the same "(b)" marker MIXES that font (the
    marker) with 'BCH_0515-Light' (the sentence). So the signal is "every
    span is one font AND that font reads as a heavier weight", not "a bold
    span is present" — which the real-sentence case would also trigger."""
    if word_count > 20 or len(font_names) != 1:
        return False
    (font,) = font_names
    return bool(_HEAVY_WEIGHT_FONT_RE.search(font)) and not _LIGHT_WEIGHT_FONT_RE.search(font)


def _clean_prose_text(block_text: str) -> str:
    """Rejoins a block's wrapped lines into one paragraph — the whole reason
    to work at block rather than line granularity. A hyphen at a line break
    (word split across the wrap) rejoins without a space."""
    text = re.sub(r"-\n", "", block_text)
    return re.sub(r"\s+", " ", re.sub(r"\s*\n\s*", " ", text)).strip()


def _table_to_text(rows) -> str:
    """One table -> one block of pipe-delimited rows, the same convention
    scripts/common/build_duckdb.py already uses for a US table paragraph,
    so a downstream reader treats both sources' tables identically."""
    lines = []
    for row in rows:
        cells = [(c or "").strip() for c in row]
        if any(cells):
            lines.append(" | ".join(cells))
    return "\n".join(lines)


def _column_bounds(x0_values: list[float], min_gap: float = 100.0) -> list[float]:
    """Clusters block left edges into columns, returning the split points.
    General to any column COUNT — verified needed on a real 3-column
    Memoria page, where a fixed left/right midpoint split put columns 1 and
    2 in the same bucket and still interleaved their sentences. min_gap
    100pt is large enough not to read a bullet's ~20pt hanging indent as a
    new column, small enough to separate real columns (~480-570pt apart on
    that page)."""
    if not x0_values:
        return []
    xs = sorted(set(x0_values))
    clusters = [[xs[0]]]
    for x in xs[1:]:
        (clusters.append([x]) if x - clusters[-1][-1] > min_gap else clusters[-1].append(x))
    return [(clusters[i][-1] + clusters[i + 1][0]) / 2 for i in range(len(clusters) - 1)]


def _reading_order_key(bbox, bounds: list[float]) -> tuple:
    """(column, y0) — verified needed on a real Memoria page where a naive
    top-to-bottom sort stitched two different columns' sentences together
    mid-sentence. Column comes from the block's LEFT edge, with no
    "full-width" special case: an earlier version keyed that off block
    WIDTH and broke a single-column document, sorting a paragraph's short
    last line into a different column than the rest of its own paragraph."""
    x0, y0, _x1, _y1 = bbox
    return (bisect.bisect_right(bounds, x0), y0)


def _merge_prose_fragments(items: list[B.Block]) -> list[B.Block]:
    """Joins a run of body blocks into one paragraph whenever a block
    doesn't end in terminal punctuation, stopping at the first one that
    does (or at a table / end of page, which always ends a run).

    Needed because PyMuPDF splits one continuous sentence into several
    same-page blocks with nothing between them — verified on a real filing
    where "En otras ganancias (pérdidas), se obtuvo... / a reversa de
    provisión... / ...revalorización realizada a los terrenos de la
    Sociedad." came back as 3 blocks, one sentence, one page.

    Furniture does NOT break a run. A repeated page header sitting between
    two halves of one sentence is invisible in the printed page's reading
    flow, and letting it end the run would leave the sentence split — this
    backend used to DELETE furniture outright, so it merged across it for
    free; now that furniture is classified and carried (so a caller can
    audit it), the merge has to skip it explicitly to keep that behaviour.
    """
    merged: list[B.Block] = []
    last_body: B.Block | None = None
    for item in items:
        if item.type in B.FURNITURE:
            merged.append(item)
            continue
        if (item.type == B.BODY and last_body is not None
                and not _SENTENCE_END_RE.search(last_body.text)):
            last_body.text += " " + item.text
            continue
        merged.append(item)
        last_body = item if item.type == B.BODY else None
    return merged


class PyMuPdfBackend:
    """Text-layer backend. Pure CPU, so it parallelizes across processes."""

    name = "pymupdf"
    reads_pixels = False

    def __init__(self, country_code: str = "", profile: PdfProfile | None = None, **_):
        # **_ swallows the VLM-only settings (dpi, runtime, server_url) that
        # config passes to whichever backend is selected — this one reads the
        # PDF's own text layer, so none of them mean anything here.
        self.profile = profile or get_profile(country_code)
        self.max_workers = os.cpu_count() or 1
        self._repeated: set[str] = set()

    def start(self) -> None:
        pass

    def close(self) -> None:
        pass

    def prepare_document(self, pdf_parts: list[bytes]) -> None:
        """Frequency-based furniture detection needs to see the WHOLE
        document before any page can be classified, so it can't live in
        parse_page. Called once per document by the runner; a backend that
        doesn't need it (every VLM backend) inherits the no-op in
        pipeline.py."""
        pages_blocks = []
        for pdf_bytes in pdf_parts:
            with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
                pages_blocks.extend(_extract_text_blocks(page) for page in doc)
        self._repeated = _find_repeated_furniture(pages_blocks, self.profile.repeat_page_fraction)

    def layout_signals(self, page) -> dict:
        """The cheap, structural read on a page: how much text it has, how
        many tables, how many text columns.

        Lives on the backend rather than in pipeline.py's triage because it
        has to apply the SAME furniture filter parse_page does. Computing
        column count over every block instead — which an earlier version of
        the triage did — counts the section-navigation strip across the top
        of a Memoria page as text, and its evenly-spread left edges bridge
        the gutter between the real columns: the two-column page carrying
        this project's Chilean AI-disclosure narrative (Banco de Chile 2024
        p.36) came back as "1 column" and was routed to this backend, which
        is exactly the page it handles worst.
        """
        raw_blocks = _extract_text_blocks(page)
        content_x0 = [
            bbox[0] for bbox, raw_text, font_names in raw_blocks
            if not self.profile.is_furniture(raw_text)
            and _normalize_for_repeat_check(raw_text) not in self._repeated
            and (text := _clean_prose_text(raw_text))
            and not _is_heading_like(text)
            and not _is_uniform_heavy_font_label(font_names, len(text.split()))
        ]
        return {
            "text_chars": len(page.get_text().strip()),
            "n_tables": len(page.find_tables().tables),
            "n_columns": len(_column_bounds(content_x0)) + 1,
            "n_images": len(page.get_images()),
        }

    def parse_page(self, page, page_no: int) -> list[B.Block]:
        width, height = page.rect.width or 1.0, page.rect.height or 1.0

        def norm_bbox(bb):
            return (bb[0] / width, bb[1] / height, bb[2] / width, bb[3] / height)

        tables = page.find_tables()
        table_bboxes = [t.bbox for t in tables.tables]
        placed: list[tuple[tuple, B.Block]] = []

        for table in tables.tables:
            text = _table_to_text(table.extract())
            if text.strip():
                placed.append((table.bbox, B.Block(type=B.TABLE, text=text, page=page_no,
                                                   bbox=norm_bbox(table.bbox), raw_type="find_tables")))

        for bbox, raw_text, font_names in _extract_text_blocks(page):
            if self.profile.is_furniture(raw_text):
                placed.append((bbox, B.Block(type=B.PAGE_HEADER, text=raw_text.strip(), page=page_no,
                                             bbox=norm_bbox(bbox), raw_type="profile_furniture")))
                continue
            if _normalize_for_repeat_check(raw_text) in self._repeated:
                placed.append((bbox, B.Block(type=B.PAGE_HEADER, text=raw_text.strip(), page=page_no,
                                             bbox=norm_bbox(bbox), raw_type="repeated_furniture")))
                continue
            if _bbox_overlaps(bbox, table_bboxes):
                continue  # already emitted as part of its table
            text = _clean_prose_text(raw_text)
            if not text:
                continue
            if _is_numeric_heavy(text):
                # find_tables() missed this one entirely — keep the ORIGINAL
                # per-line layout, since space-joining would smash separate
                # row/column values together with no delimiter at all.
                block_type, block_text, raw = B.TABLE, raw_text.strip(), "numeric_heavy"
            elif _is_heading_like(text) or _is_uniform_heavy_font_label(font_names, len(text.split())):
                # Furniture, not B.HEADING — see _is_heading_like on why
                # this backend can't separate the two.
                block_type, block_text, raw = B.PAGE_HEADER, text, "heading_or_toc_noise"
            else:
                block_type, block_text, raw = B.BODY, text, "text_block"
            placed.append((bbox, B.Block(type=block_type, text=block_text, page=page_no,
                                         bbox=norm_bbox(bbox), raw_type=raw)))

        # Column bounds come from CONTENT blocks only. The nav strip across
        # the top of a Memoria page has left edges spread evenly across the
        # full page width, which bridges the gutter between the real text
        # columns and collapses them into one cluster — verified on Banco de
        # Chile 2024 p.36, where including furniture made a two-column page
        # look single-column and the top-to-bottom sort then stitched the two
        # columns' sentences together mid-paragraph.
        bounds = _column_bounds([bbox[0] for bbox, block in placed
                                 if block.type not in B.FURNITURE])
        placed.sort(key=lambda pair: _reading_order_key(pair[0], bounds))
        return _merge_prose_fragments([block for _bbox, block in placed])
