"""
scripts/common/pdf/backends/base.py — the one interface every PDF backend
implements, and the two facts about a backend the RUNNER needs to know
before it can schedule work.

`parse_page` rather than `parse_document`: every backend this project has
evaluated (PyMuPDF layout analysis, MinerU2.5, dots.mocr, PaddleOCR-VL)
works one page at a time, and a per-page interface is what lets the
pipeline mix them — cheap backend for the easy pages, VLM for the hard
ones — without either backend knowing the other exists.
"""

from typing import Protocol

from scripts.common.pdf.blocks import Block


class PdfBackend(Protocol):
    name: str

    #: How many documents may be parsed CONCURRENTLY with this backend.
    #: This is not a tuning knob, it's a hardware fact: the PyMuPDF backend
    #: is pure CPU and scales across os.cpu_count() worker processes (which
    #: is what scripts/cl/02_extract_text.py has always done), while a VLM
    #: backend holds model weights on ONE GPU — forking it 32 ways doesn't
    #: make it faster, it makes it an out-of-memory crash. A backend that
    #: reports 1 must be run in-process, sequentially.
    max_workers: int

    #: True if the backend reads PIXELS (so it can recover a scanned page
    #: with no text layer at all), False if it reads the PDF's embedded
    #: text layer (so a scanned page is invisible to it).
    reads_pixels: bool

    def parse_page(self, page, page_no: int) -> list[Block]:
        """One pymupdf.Page -> its blocks, in reading order, classified
        into scripts/common/pdf/blocks.py's taxonomy."""
        ...

    def start(self) -> None:
        """Load models / warm up. Separate from __init__ so constructing a
        backend to ask its `max_workers` costs nothing — the runner needs
        that answer BEFORE it decides whether to spawn a process pool."""

    def close(self) -> None:
        """Release GPU memory."""
