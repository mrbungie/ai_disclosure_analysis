"""Parse numerical facts from one cached inline-XBRL filing.

Unlike SEC Company Facts, every output row retains the filing accession and
filing date that made the value public.  Downstream as-of joins can therefore
never select a value disclosed after their anchor date.
"""

from __future__ import annotations

import re
import warnings
from typing import Any

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning


def _tag_name(tag: Any) -> str:
    return str(getattr(tag, "name", "")).lower()


def _context_dates(soup: BeautifulSoup) -> dict[str, tuple[str | None, str, bool]]:
    contexts: dict[str, tuple[str | None, str, bool]] = {}
    for context in soup.find_all(lambda tag: _tag_name(tag).endswith(":context") or _tag_name(tag) == "context"):
        context_id = context.get("id")
        period = context.find(lambda tag: _tag_name(tag).endswith(":period") or _tag_name(tag) == "period")
        if not context_id or period is None:
            continue
        has_dimensions = context.find(
            lambda tag: _tag_name(tag).endswith(":explicitmember")
            or _tag_name(tag).endswith(":typedmember")
        ) is not None
        instant = period.find(lambda tag: _tag_name(tag).endswith(":instant") or _tag_name(tag) == "instant")
        if instant is not None:
            contexts[str(context_id)] = (None, instant.get_text(strip=True), has_dimensions)
            continue
        start = period.find(lambda tag: _tag_name(tag).endswith(":startdate") or _tag_name(tag) == "startdate")
        end = period.find(lambda tag: _tag_name(tag).endswith(":enddate") or _tag_name(tag) == "enddate")
        if start is not None and end is not None:
            contexts[str(context_id)] = (start.get_text(strip=True), end.get_text(strip=True), has_dimensions)
    return contexts


def _number(text: str, scale: str | None, sign: str | None, format_: str | None) -> float | None:
    value = text.strip().replace("\u00a0", "").replace("−", "-")
    if not value or value in {"—", "-", "–"}:
        return None
    negative = value.startswith("(") and value.endswith(")")
    value = value.strip("()").replace("$", "").replace(" ", "")
    if format_ and "numcommadot" in format_.lower():
        value = value.replace(".", "").replace(",", ".")
    else:
        value = value.replace(",", "")
    value = re.sub(r"[^0-9.+-]", "", value)
    try:
        parsed = float(value)
    except ValueError:
        return None
    if negative or sign == "-":
        parsed = -abs(parsed)
    return parsed * (10 ** int(scale or "0"))


def parse_inline_xbrl(
    html: str,
    *,
    ticker: str,
    accession_number: str,
    filing_date: str,
    concepts: set[str] | None = None,
) -> list[dict[str, object]]:
    """Return numeric duration/instant facts, tied to this filing's metadata."""
    warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
    soup = BeautifulSoup(html, "lxml")
    contexts = _context_dates(soup)
    facts: list[dict[str, object]] = []
    for tag in soup.find_all(lambda node: _tag_name(node).endswith(":nonfraction")):
        concept = tag.get("name")
        context_ref = tag.get("contextref") or tag.get("contextRef")
        if not concept or not context_ref or context_ref not in contexts:
            continue
        if concepts is not None and concept not in concepts:
            continue
        value = _number(tag.get_text("", strip=True), tag.get("scale"), tag.get("sign"), tag.get("format"))
        if value is None:
            continue
        period_start, period_end, has_dimensions = contexts[context_ref]
        facts.append({
            "ticker": ticker,
            "accession_number": accession_number,
            "filing_date": filing_date,
            "context_id": str(context_ref),
            "has_dimensions": has_dimensions,
            "concept": str(concept),
            "numeric_value": value,
            "period_type": "instant" if period_start is None else "duration",
            "period_start": period_start,
            "period_end": period_end,
        })
    return facts


#: Elements that carry a `contextRef` but are never themselves a reportable
#: fact — footnote links and the two "these numbers don't need a fact tag"
#: XBRL housekeeping elements. A handful of filers (small/newly-public ones
#: still under transition relief; verified: DDOG's and PLTR's first
#: post-IPO 10-Ks) file traditional (non-inline) XBRL — the numeric facts
#: are tagged directly, `<us-gaap:Revenues contextRef="c1" ...>123</...>`,
#: in a standalone `*_htm.xml`-adjacent XBRL INSTANCE DOCUMENT, not as
#: `<ix:nonFraction name="...">` wrapped around numbers inside the primary
#: HTML. `parse_inline_xbrl` cannot see these at all (no `ix:` tags exist in
#: this dialect); `parse_xbrl_instance` handles that dialect instead.
_NON_FACT_TAGS = {"contextref", "unitref", "footnote", "link:footnote", "footnotelink", "link:footnotelink"}


def parse_xbrl_instance(
    xml: str,
    *,
    ticker: str,
    accession_number: str,
    filing_date: str,
    concepts: set[str] | None = None,
) -> list[dict[str, object]]:
    """Return numeric duration/instant facts from a traditional (non-inline)
    XBRL instance document — same output schema as `parse_inline_xbrl` so
    callers can union the two without caring which dialect a filing used.

    Parsed with `features="xml"` (case-preserving), unlike
    `parse_inline_xbrl`'s HTML-mode `"lxml"`: HTML-mode parsing lowercases
    tag names, which is harmless there because the concept lives in the
    `name` ATTRIBUTE (never lowercased), but would be fatal here — the
    concept IS the tag name (`<us-gaap:Revenues>`) and the downstream
    concept-priority mapping is case-sensitive."""
    soup = BeautifulSoup(xml, "xml")
    contexts = _context_dates(soup)
    facts: list[dict[str, object]] = []
    for tag in soup.find_all(lambda node: node.get("contextRef") is not None
                             and _tag_name(node) not in _NON_FACT_TAGS):
        # `features="xml"` namespace-resolves tags: `.name` is the bare
        # local name ("Revenues"), the "us-gaap:" prefix that downstream
        # concept matching needs lives separately in `.prefix` — verified
        # against DDOG's raw XML, which does declare and use the prefix
        # (`<us-gaap:Revenues ...>`); BS4 just doesn't fold it back into
        # `.name` the way HTML-mode parsing does for `ix:nonFraction`.
        concept = f"{tag.prefix}:{tag.name}" if tag.prefix else tag.name
        context_ref = tag.get("contextRef")
        if not concept or not context_ref or context_ref not in contexts:
            continue
        if concepts is not None and concept not in concepts:
            continue
        if tag.find(True) is not None:  # has child elements -> not a leaf numeric fact
            continue
        value = _number(tag.get_text("", strip=True), None, tag.get("sign"), None)
        if value is None:
            continue
        period_start, period_end, has_dimensions = contexts[context_ref]
        facts.append({
            "ticker": ticker,
            "accession_number": accession_number,
            "filing_date": filing_date,
            "context_id": str(context_ref),
            "has_dimensions": has_dimensions,
            "concept": str(concept),
            "numeric_value": value,
            "period_type": "instant" if period_start is None else "duration",
            "period_start": period_start,
            "period_end": period_end,
        })
    return facts
