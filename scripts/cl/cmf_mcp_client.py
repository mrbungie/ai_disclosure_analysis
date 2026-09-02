"""
scripts/cl/cmf_mcp_client.py — thin JSON-RPC client for the hosted
mcp-cmf-chile server (github.com/JoaquinMulet/mcp-cmf-chile), the data
source for scripts/cl/ the same way edgartools is for scripts/us/.

WHY a hosted MCP server instead of hand-rolling requests against
www.cmfchile.cl directly (the way scripts/us/edgar_fetch.py talks to SEC
EDGAR): CMF's legacy site sits behind an F5 ASM anti-bot challenge
(cookiesession) that mcp-cmf-chile already solves, rate-limits itself
against CMF, and converts the audited PDFs (EEFF, Análisis Razonado,
Memoria) to Markdown server-side via pdf-inspector — replicating any of
that here would be undocumented, unmaintained, and break silently the
day CMF's WAF changes. Free, open-source (MIT), no API key. Same
reasoning as edgartools: use the established wrapper, not raw requests.

Verified directly against the real server (2026-09-02): stateless HTTP —
no MCP `initialize` handshake or session id is required before
`tools/call`, confirmed by calling tools/call cold and getting a normal
result.

Usage:
    from cmf_mcp_client import call_tool
    result = call_tool("cmf_empresa_memoria_anual", {"rut": "90690000", "anio": "2021"})
"""

import json
import time

import requests

MCP_ENDPOINT = "https://cmf-mcp.kumocloud.cl/mcp"

# Self-throttle: this is a free, single-instance community server, not an
# API with its own rate-limit contract with us. The server ALREADY
# rate-limits its own outbound calls to CMF (1100ms/host, see its
# cmf-client.ts) — this throttle is about not hammering the shared
# front door with a batch job, independent of that.
_MIN_INTERVAL_S = 0.6
_last_call_ts = 0.0


class CmfMcpError(Exception):
    """A JSON-RPC error, a tool-level error (isError:true), or an
    unparseable response from the mcp-cmf-chile server."""


def _throttle() -> None:
    global _last_call_ts
    elapsed = time.monotonic() - _last_call_ts
    if elapsed < _MIN_INTERVAL_S:
        time.sleep(_MIN_INTERVAL_S - elapsed)
    _last_call_ts = time.monotonic()


def _parse_sse(text: str) -> dict:
    """The server always responds Content-Type: text/event-stream, one
    `data: <json>` line per call (no real multi-event streaming observed
    for these tools) — take the LAST data: line in case that ever
    changes, rather than assuming exactly one."""
    lines = [line[len("data: "):] for line in text.splitlines() if line.startswith("data: ")]
    if not lines:
        raise CmfMcpError(f"No SSE 'data:' line in response: {text[:300]!r}")
    return json.loads(lines[-1])


def call_tool(name: str, arguments: dict, request_id: int = 1, retries: int = 3, timeout: int = 60) -> dict:
    """Calls one MCP tool. Returns the tool's `structuredContent` (falls
    back to the raw `result` dict if a tool has none). Raises
    CmfMcpError on a JSON-RPC error, a tool-level error (isError:true —
    e.g. CMF's own site not responding for that request), or an
    unparseable response — NOT on legitimate "no data for this
    period/company" results, which the tools here return as ok with an
    empty documents list (see cmf_empresa_eeff / cmf_empresa_memoria_anual
    in the upstream repo)."""
    payload = {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": "tools/call",
        "params": {"name": name, "arguments": arguments},
    }
    for attempt in range(retries):
        _throttle()
        try:
            resp = requests.post(
                MCP_ENDPOINT,
                json=payload,
                headers={"Accept": "application/json, text/event-stream", "Content-Type": "application/json"},
                timeout=timeout,
            )
            resp.raise_for_status()
            data = _parse_sse(resp.text)
            if "error" in data:
                raise CmfMcpError(f"{name}({arguments}): JSON-RPC error: {data['error']}")
            result = data["result"]
            if result.get("isError"):
                text = "; ".join(c.get("text", "") for c in result.get("content", []))
                raise CmfMcpError(f"{name}({arguments}): tool error: {text}")
            return result.get("structuredContent") or result
        except (requests.RequestException, CmfMcpError, json.JSONDecodeError, KeyError):
            if attempt < retries - 1:
                time.sleep(1.5 * (attempt + 1))
                continue
            raise
    raise CmfMcpError(f"{name}({arguments}): exhausted retries")  # unreachable, satisfies type checkers


def documento_markdown_full(url: str, max_chars: int = 100_000) -> str:
    """Fetches a CMF document's FULL Markdown text via cmf_documento_markdown,
    looping over its pagination (offset_chars/max_chars, see the tool's own
    description) until the server stops reporting a truncated tail."""
    chunks = []
    offset = 0
    while True:
        result = call_tool(
            "cmf_documento_markdown",
            {"url": url, "max_chars": max_chars, "offset_chars": offset},
        )
        markdown = result.get("markdown", "")
        chunks.append(markdown)
        if not result.get("markdown_truncado"):
            break
        offset += len(markdown)
    return "".join(chunks)
