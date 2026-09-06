"""
scripts/it/xbrl_filings_client.py — client for filings.xbrl.org, the public
ESEF aggregator this project uses instead of crawling Italy's national
storage mechanism (see docs/international_expansion_plan.md).

WHY THIS SOURCE AND NOT CONSOB/eMarket Storage. Italian regulated filings
are published through SDIR/storage operators authorised by CONSOB —
eMarket Storage (Teleborsa) is the main one. It has more than this
aggregator does (half-year reports, price-sensitive releases) but exposes
no API, and its documents are PDF, which would put Italy back on the
scanned-PDF path measured in docs/analytics/pdf-backend-poc.md. ESEF
filings here are XHTML, which is the shape scripts/us/ already parses.

WHAT ESEF GIVES AND WHAT IT DOESN'T. Since FY2021 every EU-listed issuer
must publish its ANNUAL financial report as XHTML, and the mandate covers
the whole report — the management report ("relazione sulla gestione")
included, not just the tagged financial statements. Measured on 20 random
Italian issuers, 18 do carry the management report; 2 filed the financial
statements alone (Enel, checked separately, is a third). That ~10% is why
`probe_report` exists: a filing with no narrative is a fact to record at
fetch time, not a surprise for the extractor.

There is no half-year and nothing before FY2021 here, by construction.

POLITENESS. This is a free service run for the XBRL community, not a
commercial API. One request at a time, a delay between them, and a
User-Agent that says who is calling.
"""

from __future__ import annotations

import http.client
import io
import json
import re
import time
import zipfile
import urllib.error
import urllib.parse
import urllib.request

BASE = "https://filings.xbrl.org"
USER_AGENT = ("mib-tesis-ai-disclosure/1.0 (academic research; "
              "contact via github.com/mrbungie)")
#: Seconds between requests. Downloads are 10-140 MB each, so the transfer
#: itself already paces the crawl; this only matters for the index pages.
REQUEST_DELAY = 0.5

#: Markers that a report contains the management/governance narrative
#: rather than only the tagged financial statements. Matched against the
#: de-tagged text. Italian first, English second: some issuers file the
#: English version of the annual report as their ESEF document.
_NARRATIVE_MARKERS = re.compile(
    r"relazione sulla gestione|relazione degli amministratori|"
    r"relazione sul governo societario|management report|"
    r"report of the board of directors|corporate governance report",
    re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def _encode(url: str) -> str:
    """Percent-encodes the PATH only, leaving the scheme/host alone.

    One real filing's report path contains spaces — ".../8156...-it GHC
    RFA2023 v200324/reports/..." — which urllib rejects outright as
    "control characters". The aggregator publishes the path unencoded, so
    encoding it here is what makes the URL usable at all."""
    parts = urllib.parse.urlsplit(url)
    return urllib.parse.urlunsplit(parts._replace(
        path=urllib.parse.quote(parts.path, safe="/%")))


def _request(url: str, accept: str, attempts: int = 3) -> bytes:
    """Retries on transient transport failures. These reports run to 140 MB
    and one of 870 came back as an IncompleteRead after 16 MB — over a
    corpus that is a guaranteed occasional loss, and a retry costs nothing
    when nothing is wrong."""
    request = urllib.request.Request(_encode(url),
                                     headers={"User-Agent": USER_AGENT, "Accept": accept})
    last = None
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=900) as response:
                return response.read()
        except (urllib.error.URLError, http.client.IncompleteRead, OSError) as error:
            last = error
            time.sleep(2 ** attempt)
    raise last


def get_json(path_or_url: str) -> dict:
    url = path_or_url if path_or_url.startswith("http") else BASE + path_or_url
    time.sleep(REQUEST_DELAY)
    return json.loads(_request(url, "application/json"))


def iter_filings(country: str, page_size: int = 200):
    """Yields (filing_attributes, entity_name, entity_identifier) for every
    ESEF filing from `country`.

    `include=entity` is what makes this one pass instead of two: the entity
    name and LEI arrive alongside the filing rather than needing a lookup
    per issuer. Filings whose entity relationship is missing are still
    yielded, with empty issuer fields — dropping them would silently
    shrink the corpus over a metadata gap in the aggregator.
    """
    url = (f"{BASE}/api/filings?filter[country]={urllib.parse.quote(country)}"
           f"&page[size]={page_size}&include=entity")
    while url:
        payload = get_json(url)
        names = {e["id"]: e["attributes"] for e in payload.get("included", [])
                 if e.get("type") == "entity"}
        for row in payload["data"]:
            entity = (row.get("relationships", {}).get("entity", {}) or {}).get("data")
            attributes = names.get(entity["id"], {}) if entity else {}
            yield row["attributes"], attributes.get("name", ""), attributes.get("identifier", "")
        nxt = (payload.get("links") or {}).get("next")
        url = (BASE + nxt) if nxt and str(nxt).startswith("/") else nxt


def download_report(report_url: str) -> bytes:
    """The report XHTML itself. Not the .zip package: the package adds the
    taxonomy files, which this project never reads — the narrative it does
    read is in this one file."""
    time.sleep(REQUEST_DELAY)
    return _request(BASE + report_url if report_url.startswith("/") else report_url,
                    "text/html,application/xhtml+xml")


def download_report_from_package(package_url: str) -> bytes:
    """The report XHTML pulled out of the ESEF ZIP package.

    Needed because 8 of 870 Italian filings publish a package but no
    standalone `report_url` — the aggregator simply has no direct link for
    them. Skipping those would drop real annual reports for a reason that
    is an artefact of the index, not of the filing.

    Picks the largest .xhtml/.html under a `reports/` directory: an ESEF
    package also carries the taxonomy and, sometimes, small auxiliary HTML,
    and the annual report is by a wide margin the biggest of them."""
    time.sleep(REQUEST_DELAY)
    payload = _request(
        BASE + package_url if package_url.startswith("/") else package_url, "application/zip")
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        candidates = [i for i in archive.infolist()
                      if i.filename.lower().endswith((".xhtml", ".html"))
                      and "/reports/" in i.filename.lower()]
        if not candidates:
            candidates = [i for i in archive.infolist()
                          if i.filename.lower().endswith((".xhtml", ".html"))]
        if not candidates:
            raise ValueError("no XHTML report inside the ESEF package")
        return archive.read(max(candidates, key=lambda i: i.file_size))


def probe_report(xhtml: bytes) -> dict:
    """Cheap facts about a report, computed from bytes already in memory.

    This is NOT extraction — no paragraphs, no sections, nothing stored but
    three numbers and a boolean. It exists because "does this filing
    contain narrative at all" decides whether the document belongs in an
    AI-disclosure corpus, and answering it later means re-reading tens of
    GB. `has_narrative` is a marker match, so it can be wrong in both
    directions; it is recorded as a signal to filter and audit on, never as
    a parse result.
    """
    text = _WS_RE.sub(" ", _TAG_RE.sub(" ", xhtml.decode("utf-8", errors="replace")))
    return {
        "n_bytes": len(xhtml),
        "n_words": len(text.split()),
        "has_narrative": bool(_NARRATIVE_MARKERS.search(text)),
    }
