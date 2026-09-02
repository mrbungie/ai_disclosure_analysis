"""
scripts/cl/cmf_pdf_text.py — local PDF text extraction via PyMuPDF, used in
place of the hosted mcp-cmf-chile server's WASM PDF→Markdown conversion
(see cmf_direct_client.py's docstring for why: no 4MB cap, no multi-minute
round trip). Plain text, not Markdown — good enough for this project's
purpose (finding AI-disclosure narrative sentences), same scoping
principle as scripts/us/ not needing exact financial-table structure
either. Verified directly against a real filing: PyMuPDF's flattened
table text reproduced the exact same figures as the hosted server's
Markdown-table conversion for the same document (Empresas Copec Análisis
Razonado 2021 Q1).
"""

import fitz  # pymupdf


def pdf_bytes_to_text(pdf_bytes: bytes) -> str:
    """Concatenates every page's text, double-newline-separated."""
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        return "\n\n".join(page.get_text() for page in doc)


def pdf_parts_to_text(parts: list[bytes]) -> str:
    """Multi-part documents (see cmf_direct_client.fetch_pdf_parts) — each
    part's text, in order, separated by a page-break-style marker so a
    downstream reader can still tell where one part ended and the next
    began (useful lineage, costs nothing)."""
    return "\n\n----- [siguiente parte del documento] -----\n\n".join(
        pdf_bytes_to_text(part) for part in parts
    )
