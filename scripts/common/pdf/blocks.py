"""
scripts/common/pdf/blocks.py — the country-agnostic vocabulary every PDF
backend maps into, and the collapse from that vocabulary down to the
`paragraphs` contract the rest of the pipeline already speaks
(content_type / paragraph_index / page / paragraph_text — see
scripts/common/build_duckdb.py's _cl_paragraph_select_sql).

WHY a taxonomy at all, when the downstream contract only has two content
types: because "is this page furniture or is it disclosure narrative?" is
the question every backend answers, and each answers it in its OWN
vocabulary — MinerU says `header`/`footer`/`page_number`/`aside_text`,
dots.mocr says `Page-header`/`Page-footer`, PaddleOCR-VL says something
else again, and scripts/cl/cmf_pdf_paragraphs.py's PyMuPDF heuristics
answered it with a pile of hand-written regexes. Collapsing straight to
'prose'/'table' at the backend boundary would throw that answer away and
force each backend to re-implement the drop policy. So backends classify
into THIS enum, and the drop policy lives here, once.

The taxonomy is deliberately the INTERSECTION of what the backends
actually emit, not a union of every label any of them has — a label no
backend can produce reliably is a label the pipeline can't depend on.
"""

from dataclasses import dataclass, field

# Narrative: this is the disclosure text the thesis is actually about.
BODY = "body"            # a running prose paragraph
HEADING = "heading"      # section/sub-section title
LIST_ITEM = "list_item"  # bulleted/numbered item
CAPTION = "caption"      # figure/table caption
FOOTNOTE = "footnote"    # footnote or table footnote
TABLE = "table"          # a whole table, serialized (HTML if the backend gives it)
FORMULA = "formula"      # LaTeX
FIGURE = "figure"        # image region; text-free by definition

# Page furniture: never disclosure narrative, always dropped.
PAGE_HEADER = "page_header"
PAGE_FOOTER = "page_footer"
PAGE_NUMBER = "page_number"
MARGIN_NOTE = "margin_note"  # CMF/GRI reference markers printed in the gutter

#: Types dropped before anything reaches the corpus. `margin_note` is in
#: here on evidence, not taste: on Banco de Chile's Memoria 2024 p.36 the
#: CMF cross-reference markers ("3.1.v", "3.6.vii", "3.6.ix") sit in the
#: LEFT GUTTER at the same vertical position as a body paragraph, and the
#: old PyMuPDF reading-order sort spliced them INTO the middle of a real
#: sentence — "...altos niveles de participación en los 3.6.ix procesos de
#: adaptación competitiva..." — inside the very page that carries this
#: project's Chilean AI-disclosure narrative.
FURNITURE = frozenset({PAGE_HEADER, PAGE_FOOTER, PAGE_NUMBER, MARGIN_NOTE})

#: Types that carry no text worth storing even when kept.
TEXTLESS = frozenset({FIGURE})

#: How each type maps onto the two `content_type` values the downstream
#: parquet/DuckDB contract already has. A table stays a table (build_duckdb
#: routes content_type='table' away from sentence splitting); everything
#: else that survives is prose, because that's what the sentence splitter
#: and the embedder expect. Headings/captions/footnotes ARE kept: a heading
#: like "Innovación y digitalización" is a real signal for AI-disclosure
#: detection, and dropping it (as the old heuristic did) removed context
#: the classifier could use.
CONTENT_TYPE = {
    BODY: "prose", HEADING: "prose", LIST_ITEM: "prose", CAPTION: "prose",
    FOOTNOTE: "prose", FORMULA: "prose", TABLE: "table",
}


@dataclass
class Block:
    """One layout element on one page, in reading order.

    `bbox` is normalized to 0-1 of page width/height so it means the same
    thing whether the backend saw a 200-dpi raster (VLM backends) or the
    PDF's own point-space coordinates (the PyMuPDF backend) — the old code
    mixed the two and could only ever compare boxes within one backend.
    """
    type: str
    text: str
    page: int
    bbox: tuple[float, float, float, float] | None = None
    #: The backend's own label, kept verbatim for auditing a mis-mapping
    #: without having to re-run the whole extraction.
    raw_type: str = ""
    meta: dict = field(default_factory=dict)


def _sanitize(text: str) -> str:
    """Some CMF PDFs decode with lone UTF-16 surrogate codepoints (a broken
    font/cmap in the source PDF, not a bug here) — verified on a real
    Análisis Razonado where this crashed pyarrow's parquet write with
    "surrogates not allowed", once in ~1300 documents. Round-tripping
    through utf-8 with errors='replace' costs one U+FFFD instead of the
    whole run."""
    return text.encode("utf-8", errors="replace").decode("utf-8")


def blocks_to_paragraphs(blocks: list[Block], *, keep_furniture: bool = False) -> list[dict]:
    """Collapses a document's blocks into the paragraph rows
    scripts/cl/02_extract_text.py writes to parquet.

    `paragraph_index` is a dense 0-based counter over the KEPT blocks in
    reading order — same role as scripts/us's paragraph_index (a stable
    pointer, not a claim about true document order down to the sentence).

    `keep_furniture=True` keeps the dropped types with their real
    block_type, for auditing what a backend classified as furniture without
    re-running it; it is never used by the production path.
    """
    rows = []
    for block in blocks:
        if block.type in TEXTLESS:
            continue
        if block.type in FURNITURE and not keep_furniture:
            continue
        text = _sanitize(block.text or "").strip()
        if not text:
            continue
        rows.append({
            "content_type": CONTENT_TYPE.get(block.type, "prose"),
            "block_type": block.type,
            "page": block.page,
            "paragraph_text": text,
        })
    for index, row in enumerate(rows):
        row["paragraph_index"] = index
    return rows
