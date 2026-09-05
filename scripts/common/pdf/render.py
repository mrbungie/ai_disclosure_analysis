"""
scripts/common/pdf/render.py — turning filing PDFs into the two things a
backend can consume: page rasters (for the VLM backends, which read pixels)
and a text-layer probe (for deciding whether a page even HAS extractable
text). Country-agnostic on purpose: a page raster is a page raster whether
the filer is Chilean or not.

MULTI-PART DOCUMENTS. A large Memoria arrives as N separate `.pdf.gz`
parts (see scripts/cl/01_fetch_filings.py's _save_pdf_parts). Each part is
its own standalone PDF with its own page numbering from 0, so page numbers
are only meaningful alongside the part they came from — `PageRef` carries
both, and `iter_pages` renumbers `page` densely across the whole document
while keeping `source_part` for lineage. This was open-coded inside
scripts/cl/02_extract_text.py; it isn't Chile-specific in any way, so it
lives here now.
"""

import gzip
from dataclasses import dataclass
from pathlib import Path

import pymupdf

#: 200 dpi. Below ~150 the VLM backends start dropping digits out of dense
#: financial tables; above ~250 the vision encoder's token count (and so
#: latency) climbs with no accuracy gain measured on the Banco de Chile
#: Memoria 2024 test pages.
DEFAULT_DPI = 200


@dataclass
class PageRef:
    """One page of one document, wherever it physically came from."""
    page: int          # dense index across the whole (possibly multi-part) document
    source_part: int   # which .pdf.gz part produced it
    part_page: int     # page index WITHIN that part
    page_obj: object   # pymupdf.Page — only valid while its document is open


def read_pdf_parts(local_path: str) -> list[bytes]:
    """`local_path` is the manifest's ';'-joined list of .pdf.gz part paths
    (one entry for a single-part document). Returns decompressed PDF bytes
    in part order."""
    parts = [p for p in local_path.split(";") if p]
    missing = [p for p in parts if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"missing PDF part(s): {missing}")
    out = []
    for path in parts:
        with gzip.open(path, "rb") as fh:
            out.append(fh.read())
    return out


def iter_pages(pdf_parts: list[bytes]):
    """Yields (PageRef, document) for every page of every part, in order.

    Yielded as a generator over OPEN documents rather than a materialized
    list: a 466-page Memoria's pages are consumed one at a time by every
    backend, and holding all of them (or all their rasters) at once is the
    difference between 200 MB and several GB of RSS on the big filers.
    """
    page_no = 0
    for part_num, pdf_bytes in enumerate(pdf_parts):
        with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
            for part_page, page in enumerate(doc):
                yield PageRef(page=page_no, source_part=part_num,
                              part_page=part_page, page_obj=page), doc
                page_no += 1


def render_page(page, dpi: int = DEFAULT_DPI):
    """One page -> a PIL RGB image. PIL, not raw bytes: every VLM backend's
    processor takes PIL images, and going through a PNG encode/decode round
    trip just to hand them one is pure waste."""
    from PIL import Image

    pix = page.get_pixmap(dpi=dpi)
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def text_layer_chars(page) -> int:
    """How many characters this page's embedded text layer holds.

    Zero means the page is a raster with no text layer — a scan, or a
    design-tool export that outlined its type. Those pages are INVISIBLE to
    any text-layer extractor (the old PyMuPDF path returned nothing at all
    for them and the document silently lost those pages), and they are
    exactly the pages a VLM backend exists to recover. Used by
    `scripts/common/pdf/pipeline.py`'s triage to decide which pages are
    worth a VLM call.
    """
    return len(page.get_text().strip())
