"""
scripts/common/pdf/pipeline.py — the country-agnostic entry point:
PDF bytes in, paragraph rows out, whichever backend you name.

    from scripts.common.pdf import pipeline
    rows = pipeline.extract_paragraphs(pdf_parts, backend="mineru", country_code="cl")

WHY THERE IS A TRIAGE STEP. The Chilean corpus is ~2,100 PDFs and roughly
300,000 pages. At the ~11 s/page a VLM costs through plain `transformers`
that is several hundred GPU-hours; even served through vLLM it's a
multi-hour run for the whole corpus. But most pages don't need a VLM: a
single-column page with a clean text layer and no tables is a page the
CPU backend already handles correctly in ~2 ms.

So `PageTriage` decides, per page, which backend earns the call. The rule
is deliberately CONSERVATIVE — it sends a page to the VLM whenever there's
any reason to think the cheap path would degrade it, because a wrong
"cheap" decision silently corrupts the corpus while a wrong "expensive"
decision only costs seconds. Set `triage=None` to run one backend over
every page (what you want for a small run, or to reproduce the POC).
"""

from dataclasses import dataclass

from scripts.common.pdf import blocks as B
from scripts.common.pdf.backends import get_backend
from scripts.common.pdf.render import iter_pages, read_pdf_parts, text_layer_chars


@dataclass(frozen=True)
class PageTriage:
    """Per-page routing between a cheap and an expensive backend."""

    #: A page with fewer than this many extractable characters is either
    #: blank or a raster. Rasters are invisible to the text-layer backend
    #: (it returns nothing and the page is silently lost from the
    #: document), so they always go to the VLM.
    min_text_chars: int = 40
    #: Any detected table sends the page to the VLM. Measured, not assumed:
    #: on a real 4-table liquidity page the text-layer backend recovered 42
    #: of 52 numeric cells, dropping an entire column with no error.
    vlm_if_tables: bool = True
    #: More than one text column sends the page to the VLM — multi-column
    #: reading order is where the text-layer backend spliced a margin
    #: marker into the middle of a real sentence.
    vlm_if_multi_column: bool = True

    def needs_vlm(self, page, signals: dict) -> bool:
        """`signals` comes from the cheap backend's own layout_signals() —
        the point being that the triage asks the CHEAP backend what it sees,
        and the answer to "should something better look at this page" is
        derived from the same view that backend would parse from."""
        if signals["text_chars"] < self.min_text_chars:
            # No usable text layer. Either a raster page (which the cheap
            # backend cannot see at all — those must go to the VLM) or a
            # genuinely blank section divider, which is worth nothing to
            # either backend, so don't spend a VLM call on it.
            return _page_has_ink(page)
        if self.vlm_if_tables and signals["n_tables"] > 0:
            return True
        if self.vlm_if_multi_column and signals["n_columns"] > 1:
            return True
        return False


def _page_has_ink(page) -> bool:
    return bool(page.get_images()) or bool(page.get_drawings())


class PdfExtractor:
    """Holds the backend(s) across many documents.

    This is a class and not just a function because of what a VLM backend
    costs to construct: `MinerU2.5-Pro` is 2.2 GB of weights, and building
    a fresh backend per document — which a plain function taking
    `backend="mineru"` would do — reloads them for every filing in a
    2,100-document corpus. One extractor, reused, loads them once.

    Not thread-safe, and deliberately so: one extractor owns one GPU
    context. Run several only if you have several GPUs.
    """

    def __init__(self, *, backend: str = "pymupdf", country_code: str = "",
                 triage: PageTriage | None = None, fallback_backend: str = "pymupdf",
                 backend_kwargs: dict | None = None, keep_furniture: bool = False):
        self.keep_furniture = keep_furniture
        self.triage = triage
        self.primary = get_backend(backend, country_code=country_code, **(backend_kwargs or {}))
        # The cheap backend is only built when triage can actually route to
        # it — asking for triage with backend == fallback_backend is a no-op
        # configuration, not an error.
        self.cheap = (get_backend(fallback_backend, country_code=country_code)
                      if triage is not None and backend != fallback_backend else None)
        #: Pages routed to each backend, cumulative across documents. Worth
        #: logging at the end of a run: it's the only direct read on whether
        #: triage is paying for itself.
        self.pages_by_backend: dict[str, int] = {}

    @property
    def max_workers(self) -> int:
        """The most restrictive of the backends in play — a pipeline that
        can route to a GPU backend is GPU-bound even if most pages don't."""
        engines = [e for e in (self.primary, self.cheap) if e is not None]
        return min(getattr(e, "max_workers", 1) for e in engines)

    def extract(self, pdf_parts: list[bytes]) -> list[dict]:
        """PDF part bytes -> paragraph rows.

        Each row carries `content_type` / `paragraph_index` / `page` /
        `paragraph_text` (the existing downstream contract — see
        build_duckdb.py's _cl_paragraph_select_sql) plus `block_type`,
        `source_part` and `backend`, which are new and additive: nothing
        downstream selects them yet, and they're what makes it possible to
        ask "which backend produced this paragraph, and what did it think
        the paragraph WAS" without re-running extraction.
        """
        for engine in (self.primary, self.cheap):
            if engine is not None and hasattr(engine, "prepare_document"):
                engine.prepare_document(pdf_parts)

        all_blocks: list[B.Block] = []
        part_of_page: dict[int, int] = {}
        for ref, _doc in iter_pages(pdf_parts):
            engine = self.primary
            if self.cheap is not None:
                signals = self.cheap.layout_signals(ref.page_obj)
                if not self.triage.needs_vlm(ref.page_obj, signals):
                    engine = self.cheap
            self.pages_by_backend[engine.name] = self.pages_by_backend.get(engine.name, 0) + 1
            part_of_page[ref.page] = ref.source_part
            for block in engine.parse_page(ref.page_obj, ref.page):
                block.meta["backend"] = engine.name
                all_blocks.append(block)

        rows = B.blocks_to_paragraphs(all_blocks, keep_furniture=self.keep_furniture)
        # blocks_to_paragraphs drops blocks; re-deriving the same filter here
        # is what lets each surviving ROW be zipped back to the BLOCK it came
        # from, so per-row provenance (which part, which backend) doesn't
        # have to be threaded through the taxonomy layer.
        kept = [b for b in all_blocks
                if b.type not in B.TEXTLESS
                and (self.keep_furniture or b.type not in B.FURNITURE)
                and (b.text or "").strip()]
        for row, block in zip(rows, kept):
            row["source_part"] = part_of_page.get(block.page, 0)
            row["backend"] = block.meta.get("backend", "")
        return rows

    def close(self) -> None:
        for engine in (self.primary, self.cheap):
            if engine is not None:
                engine.close()

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()


def extract_paragraphs(pdf_parts: list[bytes], **kwargs) -> list[dict]:
    """One-shot convenience wrapper. Fine for a handful of documents or a
    CPU backend; for a corpus run on a VLM backend, build one
    `PdfExtractor` and reuse it — see the class docstring."""
    with PdfExtractor(**kwargs) as extractor:
        return extractor.extract(pdf_parts)


def extract_paragraphs_from_manifest_path(local_path: str, **kwargs) -> list[dict]:
    """Convenience wrapper over the manifest's ';'-joined part path list —
    the shape scripts/cl/02_extract_text.py actually holds."""
    return extract_paragraphs(read_pdf_parts(local_path), **kwargs)


def triage_from_config(pdf_config: dict) -> PageTriage | None:
    """Builds a PageTriage from configs/<country>/config.yaml's `pdf.triage`
    block, or None when it's disabled (run one backend over every page)."""
    triage = (pdf_config or {}).get("triage") or {}
    if not triage.get("enabled"):
        return None
    return PageTriage(
        min_text_chars=triage.get("min_text_chars", 40),
        vlm_if_tables=triage.get("vlm_if_tables", True),
        vlm_if_multi_column=triage.get("vlm_if_multi_column", True),
    )
