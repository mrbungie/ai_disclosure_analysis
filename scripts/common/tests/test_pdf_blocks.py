"""Unit tests for the country-agnostic PDF layer's two pure pieces: the
block taxonomy's drop/collapse policy, and the CMF furniture profile.

Nothing here needs a PDF or a GPU — the backends are tested against real
filings in scripts/common/tests/test_pdf_pymupdf_backend.py, which skips
when the sample isn't present.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.common.pdf import blocks as B
from scripts.common.pdf.profiles import get_profile


def test_furniture_is_dropped_and_indices_stay_dense():
    rows = B.blocks_to_paragraphs([
        B.Block(type=B.BODY, text="Primer párrafo.", page=0),
        B.Block(type=B.PAGE_HEADER, text="Memoria Anual 2024 • Personas", page=0),
        B.Block(type=B.MARGIN_NOTE, text="CMF 3.1.v", page=0),
        B.Block(type=B.BODY, text="Segundo párrafo.", page=0),
    ])
    assert [r["paragraph_text"] for r in rows] == ["Primer párrafo.", "Segundo párrafo."]
    # Dense, not "the index this block had before its neighbours were
    # dropped" — downstream keys on (document_id, paragraph_index).
    assert [r["paragraph_index"] for r in rows] == [0, 1]


def test_keep_furniture_is_opt_in_and_audit_only():
    blocks = [B.Block(type=B.PAGE_HEADER, text="Página 3 de 40", page=0)]
    assert B.blocks_to_paragraphs(blocks) == []
    kept = B.blocks_to_paragraphs(blocks, keep_furniture=True)
    assert kept[0]["block_type"] == B.PAGE_HEADER


def test_figures_are_dropped_even_when_furniture_is_kept():
    # A figure carries no text by definition; keeping it would put an empty
    # paragraph in the corpus and shift every following paragraph_index.
    blocks = [B.Block(type=B.FIGURE, text="", page=0)]
    assert B.blocks_to_paragraphs(blocks, keep_furniture=True) == []


def test_only_tables_map_to_content_type_table():
    types = [B.BODY, B.HEADING, B.LIST_ITEM, B.CAPTION, B.FOOTNOTE, B.FORMULA, B.TABLE]
    rows = B.blocks_to_paragraphs([B.Block(type=t, text="x", page=0) for t in types])
    assert [r["content_type"] for r in rows] == ["prose"] * 6 + ["table"]


def test_surrogates_survive_a_parquet_safe_round_trip():
    # A real Análisis Razonado decoded through PyMuPDF with a lone UTF-16
    # surrogate and crashed pyarrow's parquet write. The paragraph must
    # survive, with the bad codepoint replaced.
    row = B.blocks_to_paragraphs([B.Block(type=B.BODY, text="ingresos \ud800 totales", page=0)])[0]
    assert "ingresos" in row["paragraph_text"] and "totales" in row["paragraph_text"]
    row["paragraph_text"].encode("utf-8")


def test_cl_profile_matches_both_cmf_templates_and_not_prose():
    profile = get_profile("cl")
    assert profile.is_furniture("Sociedad X\nRut 99.999.999-9\nPeriodo 2024\nTipo de Balance C")
    assert profile.is_furniture("Página 12 de 40")
    assert profile.is_furniture("Memoria Anual 2024 • Estados Financieros Consolidados")
    assert profile.is_furniture("Memoria Integrada 2023 · Personas")
    assert not profile.is_furniture(
        "Otro avance significativo ha sido el programa de exploración del uso "
        "de herramientas de inteligencia artificial.")


def test_unknown_country_still_gets_the_general_detector():
    profile = get_profile("br")
    assert profile.is_furniture("Página 12 de 40") is False   # no CMF patterns
    assert profile.repeat_page_fraction == 0.3                # frequency rule still applies


def test_page_deadline_fires_and_restores_the_previous_handler():
    """The watchdog exists because a VLM page call hung indefinitely while
    the backends were being benchmarked (see scripts/common/pdf/watchdog.py).
    Both halves matter: it must fire, and it must not leave a SIGALRM
    handler installed that some later part of the run trips over."""
    import signal
    import time

    import pytest

    from scripts.common.pdf.watchdog import PageTimeout, page_deadline

    sentinel = signal.getsignal(signal.SIGALRM)
    with pytest.raises(PageTimeout):
        with page_deadline(1):
            time.sleep(3)
    assert signal.getsignal(signal.SIGALRM) is sentinel

    # A call that finishes in time must cancel the timer, not leave it armed.
    with page_deadline(5):
        pass
    time.sleep(0.2)
    assert signal.getsignal(signal.SIGALRM) is sentinel


def test_page_deadline_is_a_noop_when_disabled():
    import time

    from scripts.common.pdf.watchdog import page_deadline

    with page_deadline(0):
        time.sleep(0.05)   # no timer armed, no exception
