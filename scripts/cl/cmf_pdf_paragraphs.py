"""
scripts/cl/cmf_pdf_paragraphs.py — back-compat shim.

Everything that used to live here now lives in `scripts/common/pdf/`,
because none of it was Chile-specific: block segmentation, reading order,
column detection, table extraction and the fragment-merge rules are facts
about PDFs, not about the CMF. The one genuinely Chilean part — which
strings CMF's own templates repeat as page furniture — is
`scripts/common/pdf/profiles.CL`.

Kept as a shim rather than deleted: it is imported by
scripts/cl/02_extract_text.py's older revisions and by ad-hoc analysis
notebooks, and the function contract is unchanged (byte-identical output
verified against the pre-refactor implementation on Banco de Chile's
Memoria Anual 2024, 4,062 paragraphs, and Salmones Camanchaca's Análisis
Razonado 2024-Q4, 131 paragraphs).

New code should call `scripts.common.pdf.pipeline.extract_paragraphs`
directly — it takes a `backend=` argument, which is the whole point of the
move (see docs/analytics/pdf-backend-poc.md).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.common.pdf import pipeline


def extract_paragraphs(pdf_bytes: bytes) -> list[dict]:
    """One PDF's paragraphs via the CPU (PyMuPDF) backend, with Chile's
    furniture profile. Returns [{content_type, paragraph_index, page,
    paragraph_text}, ...] — plus `block_type`/`source_part`/`backend`,
    which are additive and ignored by existing callers."""
    return pipeline.extract_paragraphs([pdf_bytes], backend="pymupdf", country_code="cl")
