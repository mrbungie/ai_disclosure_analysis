"""
scripts/common/pdf/profiles.py — the ONLY country-specific part of PDF
handling: which strings a given regulator's PDF template repeats as page
furniture.

Everything else about parsing a filing PDF (reading order, column
detection, table extraction, block classification, multi-part page
numbering) turned out to be country-agnostic once it was separated from
these regexes — which is why this file is small and the rest of
scripts/common/pdf/ has no mention of Chile in it.

A new country adds a Profile here and nothing else. A country whose
filings are HTML doesn't need one at all.
"""

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class PdfProfile:
    country_code: str
    #: Patterns matched against a block's RAW text (anchored at the start).
    #: A hit means page furniture, dropped before the corpus.
    furniture_patterns: tuple = ()
    #: A block whose text repeats on at least this fraction of a document's
    #: pages is furniture regardless of what it says. See
    #: `_find_repeated_furniture` in the PyMuPDF backend.
    repeat_page_fraction: float = 0.3

    def is_furniture(self, text: str) -> bool:
        return any(p.match(text) for p in self.furniture_patterns)


# CMF (Chile). Each pattern below was verified against a real filing while
# building scripts/cl/cmf_pdf_paragraphs.py; they are kept because a
# frequency-based detector alone can't catch furniture on a document short
# enough that nothing repeats often enough to clear the threshold.
CL = PdfProfile(
    country_code="cl",
    furniture_patterns=(
        # CMF's legacy Análisis Razonado template repeats this exact
        # two-block header on every page — verified identical across 3
        # consecutive pages of a real filing bar the page number. Matched
        # on CONTENT, not position, so a template variant that reorders
        # these fields still gets caught.
        re.compile(r"^Sociedad\s.*Rut\s.*Periodo\s.*Tipo de Balance", re.DOTALL),
        re.compile(r"^Página\s+\d+\s+de\s+\d+"),
        # Glossy "Memoria Integrada" running header, e.g. Banco de Chile
        # 2024: "Memoria Anual 2024 • Estados Financieros Consolidados" on
        # one page, "Memoria Anual 2024 • Personas" on another. The SECTION
        # NAME varies, so the exact string never repeats often enough to
        # trip the frequency check — only the "Memoria Anual|Integrada
        # <year> •" prefix is genuinely constant, so that's what's matched.
        re.compile(r"^Memoria (Anual|Integrada) \d{4}\s*[•·]"),
    ),
)

PROFILES = {"cl": CL}

#: A country with no profile yet still gets the frequency-based furniture
#: detector, which is the general half of the mechanism and needs no
#: per-regulator knowledge at all.
DEFAULT = PdfProfile(country_code="")


def get_profile(country_code: str) -> PdfProfile:
    return PROFILES.get((country_code or "").lower(), DEFAULT)
