"""
scripts/cl/cmf_pdf_paragraphs.py — extracts paragraphs (prose + table) from
a CMF PDF (Memoria Anual / Análisis Razonado), one document at a time.

WHY this needs DIFFERENT rules than scripts/common/build_duckdb.py's
paragraph SQL (which is tuned for scripts/us's HTML-derived, one-line-
per-paragraph markdown text): verified directly against real Empresas
Copec filings (2026-09-02) that plain PDF text extraction has NONE of
that structure —
  1. A markdown line already IS a full paragraph/heading; a PDF text LINE
     is just wherever the page's column width forced a line break, so a
     single real paragraph/sentence is split across many short lines —
     "one prose line = one paragraph" (build_duckdb.py's rule) would
     shred every real paragraph into fragments.
  2. Markdown tables carry "|" delimiters; a PDF has none — every table
     cell becomes its own bare line ("Activos corrientes" / "6.723" /
     "6.796" / "(74)" / "(1,1%)", each on its own line), indistinguishable
     from prose by any line-content rule build_duckdb.py uses.
  3. CMF's own PDF template repeats a page header (company/RUT/period)
     and footer (page number, print date) on EVERY page — pure noise, no
     equivalent in the US pipeline (there, that's HTTP/HTML boilerplate
     already stripped upstream by clean_html_to_lines).

REUSED, not reinvented: sentence-boundary splitting within a prose
paragraph (abbreviation masking + capital-letter-gated split) is the same
problem in either source, so once this module has clean PARAGRAPH-level
text, scripts/common/build_duckdb.py's existing sentence logic is the
right tool for the next level down — this module stops at paragraphs,
sentence splitting is deliberately not duplicated here.

Unit of extraction is PyMuPDF's own text BLOCK (page.get_text("blocks")),
not a raw line — a block already groups PyMuPDF's own layout analysis of
what belongs together, which turned out to line up with real paragraphs
far better than any line-based heuristic (verified: a narrative block on
a real Memoria page came back as one coherent multi-sentence paragraph,
not fragments). Tables are extracted separately via PyMuPDF's own
page.find_tables() (real structural table detection, not a text-content
guess) and excluded from the prose stream by bounding-box overlap so a
table's numbers never leak into "prose" paragraphs.
"""

import re

import fitz  # pymupdf

# The exact two-block header CMF's legacy PDF template repeats on every
# page of an Análisis Razonado — verified across 3 consecutive pages of a
# real filing, identical text each time bar the page number. Matched on
# CONTENT, not position, so a template variant that reorders these fields
# still gets caught. Kept as a fast, doesn't-need-the-whole-document
# check ALONGSIDE _find_repeated_boilerplate below (not replaced by it):
# it also catches this header on documents short enough, or with few
# enough distinct pages, that the repetition-frequency check wouldn't
# fire confidently.
_PAGE_HEADER_RE = re.compile(r"^Sociedad\s.*Rut\s.*Periodo\s.*Tipo de Balance", re.DOTALL)
_PAGE_FOOTER_RE = re.compile(r"^Página\s+\d+\s+de\s+\d+")
# A running header on glossy "Memoria Integrada" reports (e.g. Banco de
# Chile 2024): "Memoria Anual 2024 • Estados Financieros Consolidados" on
# one page, "Memoria Anual 2024 • Personas" on another — same running
# header, but with the CURRENT SECTION NAME as a varying suffix, so it
# never repeats often enough (same exact string) to trip
# _find_repeated_boilerplate's frequency check below. Matched on the
# fixed "Memoria Anual|Integrada <year> •" prefix alone, since that part
# is genuinely constant regardless of which section title follows it.
_RUNNING_HEADER_RE = re.compile(r"^Memoria (Anual|Integrada) \d{4}\s*[•·]")


def _is_boilerplate(text: str) -> bool:
    return bool(_PAGE_HEADER_RE.match(text) or _PAGE_FOOTER_RE.match(text) or _RUNNING_HEADER_RE.match(text))


def _normalize_for_repeat_check(text: str) -> str:
    """Collapses digit runs (page numbers, dates) so the SAME header
    template with a different page number still counts as the same
    repeated text."""
    return re.sub(r"\d+", "#", text.strip())


def _extract_text_blocks(page) -> list[tuple]:
    """[(bbox, raw_text, font_names), ...] for one page's TEXT blocks
    (type==0; image blocks are skipped, same content page.get_text("blocks")
    would return but with each block's distinct span font NAMES kept
    alongside — needed by _is_uniform_heavy_font_label. One get_text("dict")
    call replaces what used to be a get_text("blocks") call; same
    underlying PyMuPDF layout segmentation, just a richer return shape."""
    out = []
    for block in page.get_text("dict")["blocks"]:
        if block.get("type") != 0:
            continue
        lines_text = []
        font_names = set()
        for line in block["lines"]:
            line_text = "".join(span["text"] for span in line["spans"])
            lines_text.append(line_text)
            font_names.update(span["font"] for span in line["spans"] if span["text"].strip())
        raw_text = "\n".join(lines_text)
        out.append((tuple(block["bbox"]), raw_text, frozenset(font_names)))
    return out


def _find_repeated_boilerplate(pages_blocks: list[list[tuple]], min_page_fraction: float = 0.3) -> set[str]:
    """General page-header/footer detector: a block whose normalized text
    appears on at least `min_page_fraction` of a document's pages is
    template boilerplate, not content — verified needed on a real filing
    (AFP Capital's Memoria Integrada) whose header/footer ("Memoria
    Integrada 2024 / AFP Capital PÁGINA 107 ... SOBRE NOSOTROS SISTEMA DE
    PENSIONES GOBIERNO CORPORATIVO Y MARCO DE ACTUACIÓN") is a completely
    different template from the "Sociedad/Rut/Periodo/Tipo de Balance"
    one _PAGE_HEADER_RE catches — every filer's PDF template is its own,
    so a hardcoded regex per template doesn't scale, but "this exact text
    reappears on a third of the document's pages" holds regardless of
    which filer or which template produced it."""
    from collections import Counter

    counts = Counter()
    n_pages = len(pages_blocks)
    for blocks in pages_blocks:
        seen_this_page = set()
        for _bbox, raw_text, _fonts in blocks:
            norm = _normalize_for_repeat_check(raw_text)
            if norm and norm not in seen_this_page:
                counts[norm] += 1
                seen_this_page.add(norm)
    threshold = max(3, n_pages * min_page_fraction)
    return {norm for norm, count in counts.items() if count >= threshold}


def _bbox_overlaps(bbox, table_bboxes, threshold: float = 0.4) -> bool:
    """True if bbox's area is mostly (>threshold) covered by any one
    table bbox — a block entirely inside a detected table region is that
    table's content, not prose, even though PyMuPDF's block segmentation
    and its table segmentation run independently and don't share
    boundaries exactly. 0.4, not something closer to 1.0: verified on a
    real block ("PATRIMONIO TOTAL" — a table's own last row) whose bbox
    included a trailing blank line stretching past the table's bottom
    edge, measuring only 48% overlap despite being entirely the table's
    own content — a stricter threshold silently leaked real table rows
    into the prose stream."""
    x0, y0, x1, y1 = bbox
    area = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    if area == 0:
        return False
    for tx0, ty0, tx1, ty1 in table_bboxes:
        ix0, iy0 = max(x0, tx0), max(y0, ty0)
        ix1, iy1 = min(x1, tx1), min(y1, ty1)
        inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
        if inter / area > threshold:
            return True
    return False


# A fallback for tables page.find_tables() misses ENTIRELY (verified on
# a real filing: a segment-revenue breakdown table came back with zero
# ruling lines PyMuPDF's table detector could grid on, so it never
# appeared in tables.tables at all — not a partial-overlap miss like
# _bbox_overlaps handles, a block find_tables() never saw in the first
# place). Same principle as build_duckdb.py's isolated-dash fallback for
# US: a structural signal first (find_tables), a content-shape signal to
# catch what the structural one misses.
#
# Also matches bare dates ("01/01/2022", "29-mar-22") — verified needed
# on a real Análisis Razonado where a table's date-header ROW leaked into
# prose as "01/01/2022 01/01/2022 01/01/2022 01/01/2022", every token a
# date so none matched the plain-numeric pattern below.
_DATE_TOKEN_RE = re.compile(r"^\d{1,2}[/-][A-Za-zÀ-ÿ]{0,4}[/-]?\d{2,4}$")
_NUMERIC_TOKEN_RE = re.compile(r"^\(?-?[\d.,]+%?\)?$")


def _is_numeric_heavy(text: str, threshold: float = 0.4) -> bool:
    """threshold=0.4, not >0.5: verified on a real leaked table row
    ("Otras ganancias (pérdidas) 133.311 133.311 133.311") where exactly
    half the tokens (3/6: the figures) are numeric and the other half
    (2 words + one parenthesized word) are the row's own label — a
    >0.5 threshold missed this by construction, since a label-plus-values
    table row is close to a 50/50 split almost by definition."""
    tokens = text.split()
    if not tokens:
        return False
    numeric = sum(1 for t in tokens if _NUMERIC_TOKEN_RE.match(t) or _DATE_TOKEN_RE.match(t))
    return numeric / len(tokens) >= threshold


# Section headings / table-of-contents / agenda entries that PyMuPDF's
# block segmentation returns as their own "prose" block, verified on real
# Memoria/Análisis Razonado pages sampled across ~10 filers while auditing
# extraction quality (2026-09-03) — e.g. "PRINCIPALES INDICADORES",
# "9. Elección de nuevo Directorio.", "23 Gobierno Corporativo". These
# carry no disclosure narrative (no verb, no claim), just page furniture,
# so they're dropped outright rather than kept or reclassified as
# 'table' — reclassifying wouldn't help, since scripts/common/ai_embed.py
# embeds every paragraph_index regardless of content_type.
#
# Deliberately NARROW (word count <=10) — a false positive here silently
# drops real content, which is worse for a disclosure-detection corpus
# than a false negative that leaves one heading unfiltered. Known
# remaining gap, accepted rather than chased with more regexes: a
# short Title-Case caption with no leading/trailing number and no
# ALL-CAPS ("María Cecilia Facetti Presidente de Grupo Cintac") isn't
# caught by either rule below.
_ALL_CAPS_HEADING_RE = re.compile(r"^[^a-zà-ÿ]+$")  # no lowercase letters anywhere
_NUMBERED_TOC_ENTRY_RE = re.compile(r"^\d{1,3}\.?\s+\S")  # "9. Elección..." / "23 Gobierno..."


def _is_heading_or_toc_noise(text: str) -> bool:
    words = text.split()
    if not words or len(words) > 10:
        return False
    if not any(c.isalpha() for c in text):
        return False  # pure numbers/dates are _is_numeric_heavy's job, not this one
    return bool(_ALL_CAPS_HEADING_RE.match(text) or _NUMBERED_TOC_ENTRY_RE.match(text))


# Font-weight signal for note/section sub-headings that _is_heading_or_toc_noise
# can't catch by content alone — verified on real Banco de Chile 2024 EEFF
# notes (2026-09-03) via page.get_text("dict") span data: "(b) Instrumentos
# financieros de deuda:" renders ENTIRELY in one font, 'BCH_0515-Medium'
# (a heavier weight than the body text's 'BCH_0515-Light') — while a REAL
# sentence that merely starts with the same "(b)" marker, e.g. "(b) Con
# fecha 25 de enero de 2024, el Directorio del Banco...", mixes
# 'BCH_0515-Medium' (just the marker) with 'BCH_0515-Light' (the actual
# sentence) in the SAME block. So "every span uses one single font, and
# that font's name reads as a heavier weight" is the signal — not "any
# bold span present", which the real-sentence case would also trigger.
#
# Font-NAME heuristic, not PyMuPDF's span `flags` bold bit: verified flags
# was constant (4, the serif bit only) across BOTH the label and the real
# sentence above — this PDF encodes weight purely in the font's PostScript
# name, not the flags bitfield, so flags can't distinguish them here.
#
# Known gap, accepted: a template whose headings and body share ONE font
# name (weight conveyed only by size) won't be caught by this — same
# "narrow, evidence-backed rule over a broader risky one" tradeoff as
# _is_heading_or_toc_noise's word-count cap.
_HEAVY_WEIGHT_FONT_RE = re.compile(r"(bold|black|heavy|semibold|extrabold|medium)", re.IGNORECASE)
_LIGHT_WEIGHT_FONT_RE = re.compile(r"(light|regular|book|thin|roman)", re.IGNORECASE)


def _is_uniform_heavy_font_label(font_names: frozenset, word_count: int) -> bool:
    if word_count > 20 or len(font_names) != 1:
        return False
    (font,) = font_names
    return bool(_HEAVY_WEIGHT_FONT_RE.search(font)) and not _LIGHT_WEIGHT_FONT_RE.search(font)


def _clean_prose_text(block_text: str) -> str:
    """Rejoins a block's wrapped lines into one continuous paragraph —
    the whole reason to work at block granularity instead of line
    granularity. A hyphen right at a line break (word split across the
    wrap) is rejoined without a space; every other line break becomes a
    single space."""
    text = re.sub(r"-\n", "", block_text)
    text = re.sub(r"\s*\n\s*", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _table_to_text(rows: list[list[str | None]]) -> str:
    """One table -> one paragraph_text, pipe-delimited rows — same
    convention scripts/common/build_duckdb.py uses for a US table
    paragraph (content_type='table', whole table as one row), so a
    downstream reader treats both sources' tables identically."""
    lines = []
    for row in rows:
        cells = [(c or "").strip() for c in row]
        if any(cells):
            lines.append(" | ".join(cells))
    return "\n".join(lines)


# A prose block that DOESN'T end in one of these is, empirically, not a
# finished paragraph — verified on a real filing where PyMuPDF split one
# continuous sentence into 3 separate SAME-PAGE blocks with no page break
# or table between them at all ("En otras ganancias (pérdidas), se
# obtuvo..." / "a reversa de provisión..." / "...efecto en revalorización
# realizada a los terrenos de la Sociedad." — one sentence, one page,
# 3 blocks). A heading (e.g. "ANTECEDENTES FINANCIEROS") has no trailing
# punctuation either and WOULD wrongly merge into whatever follows it —
# accepted false-negative for now (an unmerged heading is at worst its
# own tiny paragraph, already true of the unmerged rule; a wrongly-merged
# heading loses nothing that a downstream reader can't still see, since
# the heading's ALL-CAPS text stays right there at the merged text's
# start) — no case of an actual bad merge found sampling real output
# across 4 companies/9 documents while building this.
_SENTENCE_END_RE = re.compile(r"[.!?:]['\"”’)]*\s*$")


def _merge_prose_fragments(items: list[dict]) -> list[dict]:
    """Second pass over one page's items, already in reading order: joins
    a run of consecutive 'prose' entries into one paragraph whenever an
    entry doesn't end in terminal punctuation, stopping at the first one
    that does (or at a table/end-of-page, which always ends a run)."""
    merged: list[dict] = []
    for item in items:
        if (item["content_type"] == "prose" and merged
                and merged[-1]["content_type"] == "prose"
                and not _SENTENCE_END_RE.search(merged[-1]["paragraph_text"])):
            merged[-1]["paragraph_text"] = merged[-1]["paragraph_text"] + " " + item["paragraph_text"]
        else:
            merged.append(dict(item))
    return merged


def _column_bounds(x0_values: list[float], min_gap: float = 100.0) -> list[float]:
    """Clusters a page's block x0 positions into columns and returns the
    split points between them. General to any column COUNT (verified
    needed on a real 3-column Memoria page — a fixed left/right midpoint
    split, the first version of this function, put columns 1 and 2 of 3
    in the same bucket and still interleaved their sentences).
    `min_gap`=100pt: large enough to not treat a bullet/hanging-indent's
    ~20pt offset as a new column (verified against a real bulleted block
    whose wrapped line was indented 21pt from its own first line), small
    enough to still separate real columns (verified ~480-570pt apart on
    the same page)."""
    if not x0_values:
        return []
    xs = sorted(set(x0_values))
    clusters = [[xs[0]]]
    for x in xs[1:]:
        if x - clusters[-1][-1] > min_gap:
            clusters.append([x])
        else:
            clusters[-1].append(x)
    return [(clusters[i][-1] + clusters[i + 1][0]) / 2 for i in range(len(clusters) - 1)]


def _reading_order_key(item_bbox: tuple, bounds: list[float]) -> tuple:
    """Sort key approximating reading order on a possibly-multi-column
    page: (column, y0) — verified needed on a real Memoria page where a
    naive top-to-bottom-only sort interleaved sentences from different
    columns into nonsense ("Los contratos que AAISA mantiene con sus
    clientes Al momento de ingresar a AAISA, los trabajadores
    arrendatarios cuentan con cláusulas que además de reciben
    información..." — two different sentences from two different
    columns, stitched mid-sentence).

    Column = which x0-cluster this block's LEFT edge falls into — no
    separate "full-width" special case (an earlier version had one, keyed
    off block WIDTH; removed after it broke a real single-column
    document: a paragraph's short last line — narrower than the 60%-of-
    page-width cutoff purely because text wrapping ended early, NOT
    because it was in a different column — got treated as "not full
    width" and sorted into a different column bucket than the rest of
    its OWN paragraph, even though its x0 was identical to every other
    line's). A block's LEFT edge is what actually indicates its column;
    width doesn't, and a heading/table spanning multiple columns still
    starts at the leftmost column's x0, so it naturally sorts into
    column 0 without needing a special case."""
    import bisect
    x0, y0, _x1, _y1 = item_bbox
    return (bisect.bisect_right(bounds, x0), y0)


def _sanitize_text(text: str) -> str:
    """Some CMF PDFs decode through PyMuPDF with lone UTF-16 surrogate
    codepoints (a broken font/cmap in the source PDF, not a bug in our
    extraction) — verified on a real Análisis Razonado where this crashed
    pandas/pyarrow's parquet write with UnicodeEncodeError ("surrogates
    not allowed") only once ~1300 documents in, i.e. rare but real.
    Round-tripping through utf-8 with errors='replace' swaps any such
    codepoint for U+FFFD instead of failing the whole run over one
    unrepresentable character in one paragraph."""
    return text.encode("utf-8", errors="replace").decode("utf-8")


def extract_paragraphs(pdf_bytes: bytes) -> list[dict]:
    """Returns [{content_type: 'prose'|'table', paragraph_index, page,
    paragraph_text}, ...] for one PDF. `paragraph_index` is a stable
    0-based running counter across the whole document, in true reading
    order (top-to-bottom per page, tables and prose blocks interleaved by
    position — NOT tables-then-prose, which would silently misorder a
    paragraph that continues after a table sitting between it and its
    continuation) — same role as scripts/us's paragraph_index (a pointer,
    not a claim about true document order down to the sentence, though
    here it happens to track it more closely)."""
    paragraphs = []
    idx = 0
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        # Cache every page's raw blocks once — needed twice (the repeated-
        # boilerplate frequency count below, then the real extraction
        # pass) and get_text("dict") isn't free to call twice per page.
        pages_blocks = [_extract_text_blocks(page) for page in doc]
        repeated_boilerplate = _find_repeated_boilerplate(pages_blocks)

        for page_no, page in enumerate(doc):
            tables = page.find_tables()
            table_bboxes = [t.bbox for t in tables.tables]

            page_items = []  # [(bbox, item_dict), ...] — sorted into reading order below

            for table in tables.tables:
                table_text = _table_to_text(table.extract())
                if table_text.strip():
                    page_items.append((table.bbox, {
                        "content_type": "table", "page": page_no, "paragraph_text": table_text,
                    }))

            for bbox, raw_text, font_names in pages_blocks[page_no]:
                if _is_boilerplate(raw_text):
                    continue
                if _normalize_for_repeat_check(raw_text) in repeated_boilerplate:
                    continue
                if _bbox_overlaps(bbox, table_bboxes):
                    continue
                text = _clean_prose_text(raw_text)
                if not text:
                    continue
                word_count = len(text.split())
                if _is_numeric_heavy(text):
                    # find_tables() missed this one entirely (see
                    # _is_numeric_heavy's comment) — keep the ORIGINAL
                    # per-line layout, not the space-joined prose text,
                    # since joining would smash separate row/column
                    # values together with no delimiter at all.
                    page_items.append((bbox, {
                        "content_type": "table", "page": page_no, "paragraph_text": raw_text.strip(),
                    }))
                elif _is_heading_or_toc_noise(text) or _is_uniform_heavy_font_label(font_names, word_count):
                    continue
                else:
                    page_items.append((bbox, {
                        "content_type": "prose", "page": page_no, "paragraph_text": text,
                    }))

            column_bounds = _column_bounds([bbox[0] for bbox, _ in page_items])
            page_items.sort(key=lambda pair: _reading_order_key(pair[0], column_bounds))
            for item in _merge_prose_fragments([item for _, item in page_items]):
                item["paragraph_index"] = idx
                item["paragraph_text"] = _sanitize_text(item["paragraph_text"])
                paragraphs.append(item)
                idx += 1
    return paragraphs
