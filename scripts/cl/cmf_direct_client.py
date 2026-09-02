"""
scripts/cl/cmf_direct_client.py — direct HTTP client for www.cmfchile.cl's
legacy PHP endpoints, ported from vendor/mcp-cmf-chile (TypeScript, MIT) —
specifically src/client/anti-bot.ts and the relevant parts of
src/client/cmf-client.ts (getLegacy/postLegacy/fetchCmfBinario).

WHY a direct port instead of the hosted MCP server (scripts/cl/cmf_mcp_client.py,
kept for reference/fallback): the hosted server converts every PDF to
Markdown server-side via a WASM engine with a hard 4MB input cap
(vendor/mcp-cmf-chile/src/pdf.ts:LIMITE_PDF_BYTES) and took ~2.5 minutes per
large Memoria Anual in practice — most Memoria PDFs (glossy annual reports,
often 5-20MB) simply cannot be converted there. Downloading the PDF
ourselves and extracting text locally (see cmf_pdf_text.py) removes both
the size cap and the multi-minute round trip.

No API key or env var needed for anything used here: CMF_API_KEY only
gates the separate api.sbif.cl macro-indicator tools (see vendor/mcp-cmf-
chile/CLAUDE.md's Gotchas — "Las tools cmf_api_* necesitan CMF_API_KEY...
en local no está, y responden un error que lo dice"), which this project
never calls. The WAF challenge below is pure HTTP, no JS execution, no
captcha, no key.

Ported, not translated line-by-line: `requests.Session` already IS a
cookie jar with automatic Set-Cookie handling, so the TypeScript CookieJar
class collapses to "just use a Session"; the manual per-host RateLimiter
becomes a single module-level throttle (this project talks to exactly one
CMF host, never several concurrently).
"""

import re
import time
from urllib.parse import urlencode, urljoin

import requests

BASE = "https://www.cmfchile.cl"
UA_DEFAULT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"

_MIN_INTERVAL_S = 1.1  # matches mcp-cmf-chile's own cmf-client.ts rateLimitMs — be an equally polite citizen of CMF's site
_last_call_ts = 0.0


class CmfSiteError(Exception):
    """CMF's legacy site returned something we couldn't work with (a
    challenge we couldn't resolve, a non-2xx after retries, etc.)."""


def _throttle() -> None:
    global _last_call_ts
    elapsed = time.monotonic() - _last_call_ts
    if elapsed < _MIN_INTERVAL_S:
        time.sleep(_MIN_INTERVAL_S - elapsed)
    _last_call_ts = time.monotonic()


def _es_challenge(body: str) -> bool:
    """Port of anti-bot.ts:esChallenge — the F5 ASM anti-bot challenge
    page is always short and carries one of these markers."""
    return len(body) < 4000 and (
        "cookiesession8341" in body or "fwb_dat" in body or "eval(function" in body
    )


def _extraer_challenge(body: str) -> tuple[str, str] | None:
    """Port of anti-bot.ts:extraerChallenge."""
    fwb = re.search(r"""fwb_dat["']?\s*[:=]\s*["']([A-Za-z0-9+/=]+)["']""", body)
    if not fwb:
        return None
    md5 = re.search(r"cookiesession8341\s*=\s*([a-f0-9]{32})", body)
    return fwb.group(1), (md5.group(1) if md5 else "0" * 32)


def _decode_body(content: bytes) -> str:
    """Port of cmf-client.ts:decodificarBody — UTF-8 if valid, else
    windows-1252 (CMF's legacy PHP pages are not consistently UTF-8)."""
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError:
        return content.decode("windows-1252")


def _request(session: requests.Session, method: str, url: str, **kwargs) -> requests.Response:
    """Core request with F5 ASM anti-bot challenge resolution — port of
    anti-bot.ts:resolverChallenge. `requests.Session` already persists
    Set-Cookie across calls, so no manual cookie jar is needed here."""
    _throttle()
    kwargs.setdefault("headers", {})
    kwargs["headers"].setdefault("User-Agent", UA_DEFAULT)
    resp = session.request(method, url, timeout=30, **kwargs)
    body_preview = resp.text[:4000] if resp.content else ""
    if not _es_challenge(body_preview):
        return resp

    challenge = _extraer_challenge(resp.text)
    if challenge is None:
        return resp  # unrecognized challenge shape: return as-is, let the caller see the raw (broken) response

    fwb_dat, md5 = challenge
    sep = "&" if "?" in url else "?"
    challenge_url = f"{url}{sep}cookiesession8341={md5}"
    _throttle()
    session.post(
        challenge_url,
        data=f"fwb_dat={fwb_dat}",
        headers={"Content-Type": "text/html", "User-Agent": UA_DEFAULT},
        timeout=30,
    )
    # Retry the original request — the session now carries the resolved cookiesession1 cookie.
    _throttle()
    return session.request(method, url, timeout=30, **kwargs)


def get_legacy(session: requests.Session, path: str, params: dict | None = None) -> str:
    """Port of cmf-client.ts:getLegacy (minus its optional KV cache — this
    project's own manifest/local-file idempotency plays that role)."""
    params = {k: v for k, v in (params or {}).items() if v not in (None, "")}
    url = f"{BASE}{path}"
    if params:
        url += f"?{urlencode(params)}"
    resp = _request(session, "GET", url)
    return _decode_body(resp.content)


def post_legacy(session: requests.Session, path: str, body: dict) -> str:
    """Port of cmf-client.ts:postLegacy."""
    url = f"{BASE}{path}"
    form = {k: v for k, v in body.items() if v is not None}
    resp = _request(
        session, "POST", url,
        data=form,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    return _decode_body(resp.content)


def fetch_binary(session: requests.Session, url: str) -> bytes:
    """Port of cmf-client.ts:fetchCmfBinario — raw document bytes (PDF),
    no text decoding."""
    resp = _request(session, "GET", url)
    return resp.content


def resolve_doc_url(href: str) -> str:
    """Port of util/nombres.ts:urlDocumentoCmf — normalizes a relative
    href from a CMF listing page into an absolute document URL, refusing
    one that still contains a directory traversal after normalization
    (see that function's own comment for the CodeQL finding this guards
    against)."""
    ruta = re.sub(r"^(?:\.\./)+", "/institucional/", href)
    if "../" in ruta:
        raise CmfSiteError(f"Document path still has a directory traversal after normalizing: {href!r}")
    return urljoin(BASE, ruta)


_VERDOCTO_RE = re.compile(r"verDocto\('([^']+)'\)")


def fetch_pdf_parts(session: requests.Session, doc_url: str) -> list[bytes]:
    """Resolves a CMF document link to one or more raw PDF byte-strings.

    Two shapes observed on the real site: an EEFF-tab document
    (safec_ifrs_verarchivo.php) links straight to a PDF. A Memoria Anual
    document (ver_sgd.php) instead returns a small HTML wrapper page
    listing one `verDocto('...&secuencia=N')` link per PART of the
    memoria (a real Empresas Copec 2021 memoria was 2 parts, 5.5MB +
    4.0MB — well past the hosted MCP server's 4MB single-file conversion
    cap, which is exactly why this project downloads and extracts PDFs
    itself rather than using that server's conversion for Memoria).
    Detected by content, not by URL pattern (%PDF-1.x magic bytes vs.
    not), since either shape could in principle appear behind either
    kind of link."""
    content = fetch_binary(session, doc_url)
    if content.startswith(b"%PDF-"):
        return [content]
    body = _decode_body(content)
    part_urls = _VERDOCTO_RE.findall(body)
    if not part_urls:
        raise CmfSiteError(f"Document at {doc_url} is neither a PDF nor a recognized ver_sgd wrapper page")
    return [fetch_binary(session, part_url) for part_url in part_urls]


def new_session() -> requests.Session:
    return requests.Session()
