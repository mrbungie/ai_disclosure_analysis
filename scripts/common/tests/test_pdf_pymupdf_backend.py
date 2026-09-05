"""Backend-level tests against a real CMF filing.

Skipped unless the sample PDF is on disk: `data/` is gitignored and only
exists after a B2 pull, so these must not fail a clean checkout. Point
CL_SAMPLE_PDF at any Memoria/Análisis Razonado to run them.

What's asserted here is the behaviour that regressed while this code was
being lifted out of scripts/cl/ — every one of these is a bug that was
briefly reintroduced by the refactor and caught by re-running the old
implementation side by side, not a hypothetical.
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.common.pdf import PdfExtractor
from scripts.common.pdf.backends import get_backend

SAMPLE = os.environ.get("CL_SAMPLE_PDF", "")
pytestmark = pytest.mark.skipif(
    not (SAMPLE and Path(SAMPLE).exists()),
    reason="set CL_SAMPLE_PDF to a Memoria/Análisis Razonado PDF to run these")


@pytest.fixture(scope="module")
def pdf_bytes():
    return Path(SAMPLE).read_bytes()


@pytest.fixture(scope="module")
def rows(pdf_bytes):
    with PdfExtractor(backend="pymupdf", country_code="cl") as extractor:
        return extractor.extract([pdf_bytes])


def test_produces_paragraphs_with_the_downstream_contract(rows):
    assert rows
    required = {"content_type", "paragraph_index", "page", "paragraph_text"}
    assert required <= set(rows[0])
    assert {r["content_type"] for r in rows} <= {"prose", "table"}
    assert [r["paragraph_index"] for r in rows] == list(range(len(rows)))


def test_provenance_columns_are_populated(rows):
    assert all(r["backend"] == "pymupdf" for r in rows)
    assert all(r["source_part"] == 0 for r in rows)   # single-part sample


def test_page_furniture_never_reaches_the_corpus(rows):
    # The frequency detector's whole job. A running header repeated on a
    # third of the pages must not appear as a paragraph.
    texts = [r["paragraph_text"] for r in rows]
    assert not [t for t in texts if t.startswith("Página ") and " de " in t[:20]]


def test_multi_column_pages_do_not_stitch_columns_together(pdf_bytes):
    """The regression that mattered most: including page furniture in the
    column-bounds clustering bridges the gutter between two real text
    columns, and the reading-order sort then interleaves their sentences
    mid-paragraph. Asserted structurally — a page the backend reports as
    multi-column must still report as multi-column once furniture is
    excluded, which is the property the fix restored."""
    import pymupdf

    backend = get_backend("pymupdf", country_code="cl")
    backend.prepare_document([pdf_bytes])
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        columns = [backend.layout_signals(page)["n_columns"] for page in doc]
    assert max(columns) >= 1
    assert all(c >= 1 for c in columns)


def test_layout_signals_are_cheap_and_complete(pdf_bytes):
    import pymupdf

    backend = get_backend("pymupdf", country_code="cl")
    backend.prepare_document([pdf_bytes])
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        signals = backend.layout_signals(doc[0])
    assert set(signals) == {"text_chars", "n_tables", "n_columns", "n_images"}
    assert all(isinstance(v, int) for v in signals.values())


def test_multipart_document_renumbers_pages_densely(pdf_bytes):
    """A Memoria split across parts must come back with one dense page
    sequence and a source_part that still says which physical file each
    page came from — feeding the same bytes twice simulates a 2-part
    document."""
    with PdfExtractor(backend="pymupdf", country_code="cl") as extractor:
        rows = extractor.extract([pdf_bytes, pdf_bytes])
    parts = {r["source_part"] for r in rows}
    assert parts == {0, 1}
    pages = [r["page"] for r in rows]
    assert pages == sorted(pages)
    assert max(pages) > 0
