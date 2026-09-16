"""
scripts/us/earnings_calls/03_fill_gaps_equibles.py — fills the Hugging Face
earnings-call dataset's coverage gaps (docs/problemas_academicos.md #5;
see 01_fetch_transcripts.py's docstring for why this instrument matters)
using the Equibles MCP server (https://mcp.equibles.com/mcp).

WHY A SEPARATE MANIFEST FILE, NOT THE SAME ONE 01_fetch_transcripts.py
WRITES. That script REBUILDS `filing_manifest_earnings_calls.parquet` from
scratch on every run (`manifest = pd.DataFrame(rows); manifest.to_parquet(...)`),
scoped only to what it finds in the HF dataset — a second source's rows
appended there today would be silently wiped the next time someone re-runs
01_fetch_transcripts.py. This script instead owns its own manifest,
`filing_manifest_earnings_calls_equibles.parquet`, and its own storage tree,
`data/raw/earnings_calls_equibles/`. The two sources are unioned back
together at read time in two places — scripts/us/earnings_calls/
02_extract_sections.py (paragraph extraction) and scripts/bronze/
manifests.py (bronze.filing_manifest) — neither of which can
overwrite the other's file. Idempotent within itself too: re-running only
fetches quarters not already in ITS OWN manifest.

WHY MCP OVER HTTP, NOT AN SDK. Equibles exposes a remote MCP server
(streamable-HTTP transport, JSON-RPC 2.0, `POST /mcp`), authenticated by an
API key (`EQUIBLES_API_KEY` env var — free key at
https://equibles.com/dashboard/apikeys). No `mcp` Python package is
required: the transport is a handful of JSON-RPC calls, each answered as a
single SSE `data:` frame (confirmed: no session negotiation, no streaming
needed for this use case) — implemented here with nothing beyond the
standard library's `urllib`.

WHAT "GAP" MEANS HERE. A ticker/fiscal-quarter pair in [2021, 2025] that
appears in NEITHER the HF manifest NOR this script's own manifest. Firms
skip quarters for real reasons (no call held, fiscal-year transition); this
only fills quarters Equibles confirms have a transcript on file
(`ListInvestorEvents`' Transcript column == "available"), never invents one.

RESPONSE FORMAT. Both `ListInvestorEvents` and `GetEarningsCallTranscript`
return Markdown text (a table, and a "**Speaker (Role)**\\n\\ntext" turn
list respectively) inside the MCP tool result's `content[0].text` — not
structured JSON. Parsed here with regex, not a JSON schema.

Usage:
    export EQUIBLES_API_KEY=eq_...
    uv run python scripts/us/earnings_calls/03_fill_gaps_equibles.py
    uv run python scripts/us/earnings_calls/03_fill_gaps_equibles.py --tickers DDOG,MRVL --dry-run
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "common"))
import pipeline_logger  # noqa: E402

MCP_URL = "https://mcp.equibles.com/mcp"
FROM_YEAR, TO_YEAR = 2021, 2026
TURN_LIMIT = 200  # server max per GetEarningsCallTranscript call
RATE_LIMIT_SECONDS = 0.5
MAX_RETRIES = 4

FORM_TYPE = "Earnings call transcript"
SOURCE = "equibles:GetEarningsCallTranscript"

_SPEAKER_LINE = re.compile(r"^\*\*(.+?)\*\*\s*$")
_TURNS_HEADER = re.compile(r"Turns:\s*\d+-\d+\s*of\s*(\d+)")
_EVENTS_ROW = re.compile(
    r"^\|\s*([0-9a-fA-F-]{36})\s*\|\s*(Earnings Call)\s*\|\s*([0-9-]+)\s*\|"
    r"\s*(.*?)\s*\|\s*FY(\d{4})\s*Q(\d)\s*\|\s*(\S+)\s*\|\s*(available|not available)\s*\|",
    re.MULTILINE,
)


class EquiblesError(RuntimeError):
    pass


def _mcp_call(api_key: str, tool: str, arguments: dict, id_: int = 1) -> dict:
    """One JSON-RPC `tools/call` round trip. Retries transient failures
    (429/5xx/network) with exponential backoff; anything else (401 bad key,
    a tool-level error) raises immediately — retrying a bad key just burns
    time until the user fixes it."""
    body = {"jsonrpc": "2.0", "id": id_, "method": "tools/call",
            "params": {"name": tool, "arguments": arguments}}
    data = json.dumps(body).encode()
    req = urllib.request.Request(MCP_URL, data=data, headers={
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        "Authorization": f"Bearer {api_key}",
    })
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = resp.read().decode(errors="replace")
                break
        except urllib.error.HTTPError as e:
            if e.code == 401:
                raise EquiblesError(
                    f"401 from Equibles — bad/missing EQUIBLES_API_KEY: {e.read().decode(errors='replace')}"
                ) from e
            if e.code in (429, 500, 502, 503, 504) and attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
                last_err = e
                continue
            raise EquiblesError(f"HTTP {e.code}: {e.read().decode(errors='replace')}") from e
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
                last_err = e
                continue
            raise EquiblesError(f"network error after {MAX_RETRIES} attempts: {e}") from e
    else:
        raise EquiblesError(f"exhausted retries: {last_err}")

    # the server answers either one SSE `data:` frame or a plain JSON body
    line = next((l for l in raw.split("\n") if l.startswith("data:")), None)
    try:
        payload = json.loads(line[len("data:"):].strip() if line is not None else raw)
    except json.JSONDecodeError as e:
        raise EquiblesError(f"unparseable response: {raw[:500]}") from e
    if "error" in payload:
        raise EquiblesError(f"{tool} error: {payload['error']}")
    result = payload["result"]
    if result.get("isError"):
        raise EquiblesError(f"{tool} tool error: {result}")
    texts = [c["text"] for c in result.get("content", []) if c.get("type") == "text"]
    return {"text": "\n".join(texts)}


def list_earnings_events(api_key: str, ticker: str) -> list[dict]:
    """[{fiscal_year, fiscal_quarter, transcript_available}, ...] for one
    ticker, from ListInvestorEvents' Markdown table. `limit=100` covers
    every quarter back to well before 2021 for a quarterly reporter."""
    out = _mcp_call(api_key, "ListInvestorEvents",
                    {"ticker": ticker, "eventType": "EarningsCall", "limit": 100})
    events = []
    for m in _EVENTS_ROW.finditer(out["text"]):
        _event_id, _type, _date, _title, fy, fq, _status, transcript = m.groups()
        events.append({"fiscal_year": int(fy), "fiscal_quarter": int(fq),
                       "transcript_available": transcript == "available"})
    return events


def _parse_transcript_page(text: str) -> tuple[list[dict], int, str | None]:
    """(turns_on_this_page, total_turns, call_date) from one
    GetEarningsCallTranscript response's Markdown text."""
    total_m = _TURNS_HEADER.search(text)
    total = int(total_m.group(1)) if total_m else 0
    date_m = re.search(r"Call date \(UTC\):\s*([0-9-]+)", text)
    call_date = date_m.group(1) if date_m else None

    lines = text.split("\n")
    turns: list[dict] = []
    current_speaker = None
    current_lines: list[str] = []

    def flush():
        if current_speaker is not None:
            body = " ".join(l.strip() for l in current_lines if l.strip())
            if body:
                turns.append({"speaker": current_speaker, "text": body})

    for line in lines:
        m = _SPEAKER_LINE.match(line)
        if m:
            flush()
            current_speaker = m.group(1).strip()
            current_lines = []
        elif current_speaker is not None:
            current_lines.append(line)
    flush()
    return turns, total, call_date


def fetch_full_transcript(api_key: str, ticker: str, fiscal_year: int, fiscal_quarter: int) -> dict:
    """Pages through GetEarningsCallTranscript (server max 200 turns/call)
    until every turn is collected. Returns a payload shaped like
    01_fetch_transcripts.py's HF-sourced one (`structured_content`,
    `content`, ...) so 02_extract_sections.py needs no source-specific
    branch."""
    offset = 0
    all_turns: list[dict] = []
    call_date = None
    total = None
    while total is None or offset < total:
        out = _mcp_call(api_key, "GetEarningsCallTranscript", {
            "ticker": ticker, "fiscalYear": fiscal_year, "fiscalQuarter": fiscal_quarter,
            "limit": TURN_LIMIT, "offset": offset,
        })
        turns, page_total, page_date = _parse_transcript_page(out["text"])
        if not turns:
            break
        all_turns.extend(turns)
        call_date = call_date or page_date
        total = page_total
        offset += len(turns)
        time.sleep(RATE_LIMIT_SECONDS)
    content = "\n\n".join(f"{t['speaker']}: {t['text']}" for t in all_turns)
    return {"symbol": ticker, "quarter": fiscal_quarter, "year": fiscal_year,
            "date": call_date or "", "content": content, "structured_content": all_turns,
            "company_name": "", "company_id": ""}


def existing_periods(manifest_paths: list[Path]) -> set[tuple[str, int, int]]:
    """{(ticker, fiscal_year, fiscal_quarter), ...} already covered by ANY
    manifest — reading, never writing, the other source's file."""
    covered = set()
    for path in manifest_paths:
        if not path.exists():
            continue
        df = pd.read_parquet(path, columns=["ticker", "period_end_date"])
        for ticker, period in zip(df["ticker"], df["period_end_date"]):
            m = re.match(r"(\d{4})Q(\d)", str(period))
            if m:
                covered.add((str(ticker).upper().strip().replace(".", "-"), int(m.group(1)), int(m.group(2))))
    return covered


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tickers", default=None,
                        help="Comma-separated subset to process (default: every gap ticker in the universe).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report gaps found per ticker; fetch nothing.")
    parser.add_argument("--limit-tickers", type=int, default=None,
                        help="Cap how many tickers to process this run (politeness / testing).")
    args = parser.parse_args()

    api_key = os.environ.get("EQUIBLES_API_KEY")
    if not api_key:
        print("EQUIBLES_API_KEY not set. Get a free key at https://equibles.com/dashboard/apikeys "
              "and `export EQUIBLES_API_KEY=eq_...` before running this script.")
        return

    config = yaml.safe_load((REPO_ROOT / "configs" / "us" / "config.yaml").read_text())
    manifest_dir = REPO_ROOT / config["storage"]["interim_manifests"]
    raw_dir = REPO_ROOT / "data" / "raw" / "earnings_calls_equibles"
    hf_manifest_path = manifest_dir / "filing_manifest_earnings_calls.parquet"
    own_manifest_path = manifest_dir / "filing_manifest_earnings_calls_equibles.parquet"

    universe = pd.read_csv(REPO_ROOT / "configs" / "us" / "universe.csv")
    if args.tickers:
        wanted = [t.strip().upper() for t in args.tickers.split(",")]
        universe = universe[universe["ticker"].str.upper().isin(wanted)]
    if args.limit_tickers:
        universe = universe.head(args.limit_tickers)
    cik_by_ticker = dict(zip(universe["ticker"], universe["cik"]))

    covered = existing_periods([hf_manifest_path, own_manifest_path])
    pipeline_logger.log_event(pipeline_step="us_fill_gaps_equibles", level="INFO",
                              message=f"{len(covered):,} ticker-quarters already covered "
                                      f"(HF + prior Equibles runs) across {FROM_YEAR}-{TO_YEAR}",
                              log_dir=manifest_dir)

    existing_own = pd.read_parquet(own_manifest_path) if own_manifest_path.exists() else pd.DataFrame()
    existing_ids = set(existing_own["document_id"]) if len(existing_own) else set()

    new_rows = []
    n_tickers_with_gaps = 0
    n_total_tickers = len(universe)
    for i, ticker in enumerate(universe["ticker"], start=1):
        print(f"[{i}/{n_total_tickers}] {ticker}...", flush=True)
        norm = str(ticker).upper().strip().replace(".", "-")
        try:
            events = list_earnings_events(api_key, ticker)
        except EquiblesError as e:
            pipeline_logger.log_event(pipeline_step="us_fill_gaps_equibles", level="ERROR",
                                      message=f"{ticker}: {e}", log_dir=manifest_dir)
            continue
        time.sleep(RATE_LIMIT_SECONDS)

        gaps = [e for e in events
               if FROM_YEAR <= e["fiscal_year"] <= TO_YEAR and e["transcript_available"]
               and (norm, e["fiscal_year"], e["fiscal_quarter"]) not in covered]
        if not gaps:
            continue
        n_tickers_with_gaps += 1
        print(f"{ticker}: {len(gaps)} gap quarter(s) with a transcript on file "
              f"{[(g['fiscal_year'], g['fiscal_quarter']) for g in gaps]}")
        if args.dry_run:
            continue

        for gap in gaps:
            fy, fq = gap["fiscal_year"], gap["fiscal_quarter"]
            document_id = f"{norm}_{fy}Q{fq}"
            if document_id in existing_ids:
                continue
            try:
                payload = fetch_full_transcript(api_key, ticker, fy, fq)
            except EquiblesError as e:
                pipeline_logger.log_event(pipeline_step="us_fill_gaps_equibles", level="ERROR",
                                          message=f"{ticker} {fy}Q{fq}: {e}", log_dir=manifest_dir)
                continue
            if not payload["structured_content"]:
                pipeline_logger.log_event(pipeline_step="us_fill_gaps_equibles", level="WARNING",
                                          message=f"{ticker} {fy}Q{fq}: event listed as available "
                                                  f"but 0 turns returned; skipped", log_dir=manifest_dir)
                continue

            path = raw_dir / norm / f"{document_id}.json.gz"
            path.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(path, "wt", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False)

            new_rows.append({
                "document_id": document_id, "ticker": ticker, "cik": cik_by_ticker.get(ticker, ""),
                "source": SOURCE, "form_type": FORM_TYPE, "filing_type": "earnings_call",
                "filing_date": (payload["date"] or "")[:10], "period_end_date": f"{fy}Q{fq}",
                "local_path": str(path), "format": "json", "download_status": "completed",
                "n_bytes": path.stat().st_size, "n_turns": len(payload["structured_content"]),
                "n_chars": len(payload["content"]), "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc),
            })
            pipeline_logger.log_event(pipeline_step="us_fill_gaps_equibles", level="SUCCESS",
                                      message=f"{ticker} FY{fy}Q{fq}: {len(payload['structured_content'])} turns",
                                      log_dir=manifest_dir)

    pipeline_logger.log_event(
        pipeline_step="us_fill_gaps_equibles", level="INFO",
        message=f"{n_tickers_with_gaps} tickers had a fillable gap"
                + (" (dry run, nothing fetched)" if args.dry_run else f"; {len(new_rows)} new transcripts fetched"),
        log_dir=manifest_dir)

    if args.dry_run or not new_rows:
        return

    combined = pd.concat([existing_own, pd.DataFrame(new_rows)], ignore_index=True) if len(existing_own) else pd.DataFrame(new_rows)
    combined = combined.drop_duplicates("document_id", keep="last")
    combined.to_parquet(own_manifest_path, index=False)
    pipeline_logger.log_event(
        pipeline_step="us_fill_gaps_equibles", level="SUCCESS",
        message=f"{len(combined):,} total Equibles transcripts, {combined.ticker.nunique()} tickers -> {own_manifest_path}",
        log_dir=manifest_dir)


if __name__ == "__main__":
    main()
