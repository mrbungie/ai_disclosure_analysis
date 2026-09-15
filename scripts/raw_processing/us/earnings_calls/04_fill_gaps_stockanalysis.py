"""
scripts/us/earnings_calls/04_fill_gaps_stockanalysis.py — third, independent
earnings-call source: stockanalysis.com, scraped via local headless Chrome.

WHY A THIRD SOURCE. 03_fill_gaps_equibles.py covers most of the gaps the
Hugging Face bulk dataset (01_fetch_transcripts.py) leaves behind, but
Equibles' free/shared MCP plan caps out at 100 requests/day — a run over
the full ~517-ticker universe hits that ceiling with tickers still
unprocessed. Those tickers are NOT missing data on Equibles' side (every
one of them failed with "Daily call limit exceeded", not "not found");
waiting a day and rerunning 03 would eventually clear them. This script
exists for the tickers a same-day pass still needs: it hits a different,
non-rate-limited source instead of waiting on Equibles' clock.

WHY LOCAL HEADLESS CHROME, NOT `requests`/`urllib`. stockanalysis.com has
no public API and returns the transcript inline in server-rendered HTML —
but its anti-bot layer 400s a bare urllib/requests client after 2-3
sequential fetches even with a browser User-Agent and multi-second
delays (confirmed empirically). A real Chrome instance (`--headless=new
--dump-dom`) presents a genuine browser fingerprint and was not blocked
in the same testing. This is heavier per page (~5-10s, a subprocess
spawn) than an API call, which is exactly why this script is the
fallback, not the primary source — 01 (HF) and 03 (Equibles) are always
tried first.

WHY A THIRD MANIFEST FILE, NOT A MERGE. Same rule as 03: this never reads
FROM or writes TO filing_manifest_earnings_calls.parquet (HF, rewritten
wholesale by 01 on every run) or filing_manifest_earnings_calls_equibles
.parquet (03's own file). 02_extract_sections.py concatenates all three
at read time. A ticker's transcript quarter is fetched from whichever
source had it available FIRST — once any manifest lists a (ticker, year,
quarter), the other two scripts treat it as covered and skip it.

SLUG RESOLUTION. stockanalysis.com URLs use a lowercased ticker slug that
usually — but not always — matches the SEC ticker. `.` in a class-share
ticker (BRK.B) does not map to any slug stockanalysis.com recognizes
under any of the variants tried (also true for Berkshire specifically
because it doesn't hold conference calls at all); such tickers 404 on
every variant and are logged and skipped, not retried.

FISCAL, NOT CALENDAR, YEAR/QUARTER. stockanalysis.com labels each
transcript with the company's own fiscal year/quarter (e.g. MRVL's "Q1
2027" call happened in calendar May 2026) — the same convention Equibles'
ListInvestorEvents already uses (03's FY{n} Qn parsing) and what the
existing_periods() gap-detection compares against. Using the company's
own label, not a recomputed calendar quarter, is what keeps this
consistent with the other two sources.

Usage:
    uv run python scripts/us/earnings_calls/04_fill_gaps_stockanalysis.py [--tickers AAPL,MSFT] [--dry-run] [--limit-tickers N]
"""

import argparse
import gzip
import html as ihtml
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "common"))

import pipeline_logger
from importlib import import_module
_equibles = import_module("03_fill_gaps_equibles")
existing_periods = _equibles.existing_periods

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
BASE_URL = "https://stockanalysis.com"
FROM_YEAR, TO_YEAR = 2021, 2026
FORM_TYPE = "Earnings call transcript"
SOURCE = "stockanalysis.com:headless-chrome"
#: seconds between page fetches — same host, so this is politeness, not a
#: documented rate limit (none is published).
RATE_LIMIT_SECONDS = 2.0
DOM_TIMEOUT_MS = 8000
MAX_RETRIES = 3

_EVENT_RE = re.compile(
    r'fiscalYear:(\d+),quarterLabel:"([^"]+)",detailSlug:"([^"]+)",eventDate:"([^"]+)"')
_QUARTER_LABEL_RE = re.compile(r"^Q([1-4])\s+(\d{4})$")
_SPEAKER_RE = re.compile(r'<div class="text-lg font-bold text-default[^"]*">([^<]+)</div>')
_SENTENCE_RE = re.compile(r'<span class="transcript-sentence[^"]*"[^>]*>([^<]*)</span>')
_TITLE_RE = re.compile(r"<title>([^<]*)</title>")
_COMPANY_NAME_RE = re.compile(r'og:title" content="([^(]+)\(')
#: A renamed/rebranded ticker's OLD slug listing page (`/stocks/<old>/
#: transcripts/`) redirects server-side to the CURRENT company's content,
#: but that redirect only applies to the listing — a SPECIFIC quarter's
#: detail URL built from the old slug (`/stocks/<old>/transcripts/<id>/`)
#: does not resolve the same way and silently yields a transcript page
#: with zero speaker turns. Verified against SQ: Block's ticker changed
#: from SQ to XYZ; `/stocks/sq/transcripts/` renders with
#: `<title>Block (XYZ) Earnings Call Transcripts</title>` — the CURRENT
#: ticker is right there in the title even though the URL still says
#: "sq" — but every `/stocks/sq/transcripts/<old-detail-id>/` fetch
#: returns 0 turns. Using the canonical ticker from the title for all
#: subsequent list/fetch calls (not the originally-probed slug) fixes it;
#: confirmed XYZ's listing carries SQ's full history (Q1/Q2 2025 present).
_CANONICAL_TICKER_RE = re.compile(r"\(([A-Z]{1,6}(?:\.[A-Z])?)\)")


class FetchError(RuntimeError):
    pass


def _dump_dom(url: str) -> str:
    """One headless Chrome invocation, rendered DOM as a string. Retries
    transient subprocess/network failures; a 404 page is returned as-is
    (its title reads "404 - Page not found") for the caller to detect."""
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            result = subprocess.run(
                [CHROME, "--headless=new", "--disable-gpu", "--no-sandbox",
                 f"--virtual-time-budget={DOM_TIMEOUT_MS}", "--dump-dom", url],
                capture_output=True, text=True, timeout=30)
            if result.returncode == 0 and result.stdout:
                return result.stdout
            last_err = result.stderr[-500:] if result.stderr else "empty output"
        except subprocess.TimeoutExpired as e:
            last_err = str(e)
        time.sleep(1.5)
    raise FetchError(f"headless chrome failed after {MAX_RETRIES} attempts: {last_err}")


def _slug_candidates(ticker: str) -> list[str]:
    t = ticker.strip()
    candidates = [t.lower(), t.lower().replace(".", "-")]
    if "." in t:
        candidates.append(t.split(".")[0].lower())
    seen, out = set(), []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def resolve_slug(ticker: str) -> str | None:
    for slug in _slug_candidates(ticker):
        dom = _dump_dom(f"{BASE_URL}/stocks/{slug}/transcripts/")
        title_m = _TITLE_RE.search(dom)
        if title_m and "404" not in title_m.group(1):
            canonical_m = _CANONICAL_TICKER_RE.search(title_m.group(1))
            if canonical_m and canonical_m.group(1).lower() != slug:
                return canonical_m.group(1).lower()
            return slug
        time.sleep(RATE_LIMIT_SECONDS)
    return None


def list_events(slug: str) -> list[dict]:
    """Every Q1-Q4 earnings-call event on the ticker's transcript index
    page (conference talks / AGMs / investor days carry non-"Qn YYYY"
    quarterLabels and are excluded — they are not earnings calls)."""
    dom = _dump_dom(f"{BASE_URL}/stocks/{slug}/transcripts/")
    events = []
    for fiscal_year, quarter_label, slug_id, event_date in _EVENT_RE.findall(dom):
        m = _QUARTER_LABEL_RE.match(quarter_label)
        if not m:
            continue
        events.append({
            "fiscal_quarter": int(m.group(1)), "fiscal_year": int(fiscal_year),
            "detail_slug": slug_id, "event_date": event_date,
        })
    return events


def fetch_transcript(slug: str, detail_slug: str) -> dict:
    dom = _dump_dom(f"{BASE_URL}/stocks/{slug}/transcripts/{detail_slug}/")
    start = dom.find('aria-label="Full transcript"')
    if start < 0:
        return {"structured_content": [], "content": "", "company_name": None}
    region = dom[start:]
    speaker_matches = list(_SPEAKER_RE.finditer(region))
    turns = []
    for i, m in enumerate(speaker_matches):
        speaker = ihtml.unescape(m.group(1)).strip()
        seg_end = speaker_matches[i + 1].start() if i + 1 < len(speaker_matches) else len(region)
        segment = region[m.end():seg_end]
        sentences = _SENTENCE_RE.findall(segment)
        text = ihtml.unescape(" ".join(s.strip() for s in sentences if s.strip()))
        if text:
            turns.append({"speaker": speaker, "text": text})
    company_m = _COMPANY_NAME_RE.search(dom)
    company_name = company_m.group(1).strip() if company_m else None
    content = "\n".join(f"{t['speaker']}: {t['text']}" for t in turns)
    return {"structured_content": turns, "content": content, "company_name": company_name}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tickers", default=None,
                        help="Comma-separated subset to process (default: every gap ticker in the universe).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report gaps found per ticker; fetch nothing.")
    parser.add_argument("--limit-tickers", type=int, default=None,
                        help="Cap how many tickers to process this run.")
    args = parser.parse_args()

    if not Path(CHROME).exists():
        print(f"Chrome not found at {CHROME}; edit the CHROME constant.")
        return

    config = yaml.safe_load((REPO_ROOT / "configs" / "us" / "config.yaml").read_text())
    manifest_dir = REPO_ROOT / config["storage"]["interim_manifests"]
    raw_dir = REPO_ROOT / "data" / "raw" / "earnings_calls_stockanalysis"
    hf_manifest_path = manifest_dir / "filing_manifest_earnings_calls.parquet"
    equibles_manifest_path = manifest_dir / "filing_manifest_earnings_calls_equibles.parquet"
    own_manifest_path = manifest_dir / "filing_manifest_earnings_calls_stockanalysis.parquet"

    universe = pd.read_csv(REPO_ROOT / "configs" / "us" / "universe.csv")
    if args.tickers:
        wanted = [t.strip().upper() for t in args.tickers.split(",")]
        universe = universe[universe["ticker"].str.upper().isin(wanted)]
    if args.limit_tickers:
        universe = universe.head(args.limit_tickers)
    cik_by_ticker = dict(zip(universe["ticker"], universe["cik"]))

    covered = existing_periods([hf_manifest_path, equibles_manifest_path, own_manifest_path])
    pipeline_logger.log_event(pipeline_step="us_fill_gaps_stockanalysis", level="INFO",
                              message=f"{len(covered):,} ticker-quarters already covered "
                                      f"(HF + Equibles + prior stockanalysis runs) across {FROM_YEAR}-{TO_YEAR}",
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
            slug = resolve_slug(ticker)
        except FetchError as e:
            pipeline_logger.log_event(pipeline_step="us_fill_gaps_stockanalysis", level="ERROR",
                                      message=f"{ticker}: slug resolution failed: {e}", log_dir=manifest_dir)
            continue
        if slug is None:
            pipeline_logger.log_event(pipeline_step="us_fill_gaps_stockanalysis", level="WARNING",
                                      message=f"{ticker}: no stockanalysis.com page under any slug variant "
                                              f"tried ({_slug_candidates(ticker)}); skipped", log_dir=manifest_dir)
            continue
        time.sleep(RATE_LIMIT_SECONDS)

        try:
            events = list_events(slug)
        except FetchError as e:
            pipeline_logger.log_event(pipeline_step="us_fill_gaps_stockanalysis", level="ERROR",
                                      message=f"{ticker}: {e}", log_dir=manifest_dir)
            continue
        time.sleep(RATE_LIMIT_SECONDS)

        gaps = [e for e in events
               if FROM_YEAR <= e["fiscal_year"] <= TO_YEAR
               and (norm, e["fiscal_year"], e["fiscal_quarter"]) not in covered]
        if not gaps:
            continue
        n_tickers_with_gaps += 1
        print(f"{ticker} ({slug}): {len(gaps)} gap quarter(s) "
              f"{[(g['fiscal_year'], g['fiscal_quarter']) for g in gaps]}")
        if args.dry_run:
            continue

        for gap in gaps:
            fy, fq = gap["fiscal_year"], gap["fiscal_quarter"]
            document_id = f"{norm}_{fy}Q{fq}"
            if document_id in existing_ids:
                continue
            try:
                payload_extra = fetch_transcript(slug, gap["detail_slug"])
            except FetchError as e:
                pipeline_logger.log_event(pipeline_step="us_fill_gaps_stockanalysis", level="ERROR",
                                          message=f"{ticker} {fy}Q{fq}: {e}", log_dir=manifest_dir)
                continue
            time.sleep(RATE_LIMIT_SECONDS)
            if not payload_extra["structured_content"]:
                pipeline_logger.log_event(pipeline_step="us_fill_gaps_stockanalysis", level="WARNING",
                                          message=f"{ticker} {fy}Q{fq}: event listed but 0 turns parsed; skipped",
                                          log_dir=manifest_dir)
                continue

            payload = {
                "symbol": ticker, "quarter": fq, "year": fy, "date": gap["event_date"],
                "content": payload_extra["content"],
                "structured_content": payload_extra["structured_content"],
                "company_name": payload_extra["company_name"] or ticker,
                "company_id": ticker,
            }
            path = raw_dir / norm / f"{document_id}.json.gz"
            path.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(path, "wt", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False)

            new_rows.append({
                "document_id": document_id, "ticker": ticker, "cik": cik_by_ticker.get(ticker, ""),
                "source": SOURCE, "form_type": FORM_TYPE, "filing_type": "earnings_call",
                "filing_date": gap["event_date"], "period_end_date": f"{fy}Q{fq}",
                "local_path": str(path), "format": "json", "download_status": "completed",
                "n_bytes": path.stat().st_size, "n_turns": len(payload["structured_content"]),
                "n_chars": len(payload["content"]), "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc),
            })
            pipeline_logger.log_event(pipeline_step="us_fill_gaps_stockanalysis", level="SUCCESS",
                                      message=f"{ticker} FY{fy}Q{fq}: {len(payload['structured_content'])} turns",
                                      log_dir=manifest_dir)

    pipeline_logger.log_event(
        pipeline_step="us_fill_gaps_stockanalysis", level="INFO",
        message=f"{n_tickers_with_gaps} tickers had a fillable gap"
                + (" (dry run, nothing fetched)" if args.dry_run else f"; {len(new_rows)} new transcripts fetched"),
        log_dir=manifest_dir)

    if args.dry_run or not new_rows:
        return

    combined = pd.concat([existing_own, pd.DataFrame(new_rows)], ignore_index=True) if len(existing_own) else pd.DataFrame(new_rows)
    combined = combined.drop_duplicates("document_id", keep="last")
    combined.to_parquet(own_manifest_path, index=False)
    pipeline_logger.log_event(
        pipeline_step="us_fill_gaps_stockanalysis", level="SUCCESS",
        message=f"{len(combined):,} total stockanalysis.com transcripts, {combined.ticker.nunique()} tickers -> {own_manifest_path}",
        log_dir=manifest_dir)


if __name__ == "__main__":
    main()
