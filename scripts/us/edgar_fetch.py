"""
scripts/us/edgar_fetch.py — SEC EDGAR fetch logic, specific to this
country/source. Shared by scripts/us/10k/01_fetch_filings.py and
scripts/us/10q/01_fetch_filings.py. Uses edgartools (Company.get_filings +
Filing.html()) instead of hand-rolled requests calls:

- edgartools' HTTP client (httpx-based, see edgar.httpclient) reuses
  connections across requests — plain module-level `requests.get()` calls
  (the old approach) open a fresh connection every time, paying a TCP+TLS
  handshake per request instead of keep-alive.
- Its /Archives/edgar/data responses are cached FOREVER on disk (see
  edgar.httpclient.clear_empty_cached_responses' docstring) — re-running
  this script never re-hits the network for a filing it already has,
  without us hand-rolling that check ourselves.
- Filing.html() fetches exactly one filing's primary document — NOT
  Filings.download(), which downloads whole daily/quarterly bulk index
  archives (every filer, not just ours) and filters locally; wrong tool
  for a targeted subset of ~500 companies.

10-K and 10-Q are fetched via the SAME function (parameterized by form)
but into SEPARATE output paths — no pooling, per the project's data-scope
decision (10-K panel core + 10-Q shock series, never merged).

Storage: primary documents are gzip-compressed on save (.html.gz) — this
is OUR OWN mirror (what section_segmenter.py reads), on top of edgar's own
cache. iXBRL-era 10-Ks/10-Qs are ~90%+ repeated markup/XBRL-context
boilerplate that compresses hard (a real AAPL 10-K: 1.5MB -> 111KB).
"""

import gzip
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import edgar
import pandas as pd
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))  # scripts/common/
import pipeline_logger

MAX_WORKERS = 8
_configured = False


def configure(user_agent: str, local_storage_dir: Path):
    """Point edgartools at OUR repo-internal, gitignored cache dir instead
    of its default ~/.edgar_cache — keeps everything the pipeline touches
    inside the repo tree (still gitignored, never committed)."""
    global _configured
    if _configured:
        return
    local_storage_dir.mkdir(parents=True, exist_ok=True)
    edgar.set_identity(user_agent)
    edgar.set_local_storage_path(str(local_storage_dir))
    edgar.use_local_storage(True)
    _configured = True


def _fetch_one_company(row, form, start_date, end_date, allow_amendments, html_dir, existing_manifest_df):
    """Runs in a worker thread. Returns (firm_info, manifest_rows)."""
    ticker = row["ticker"]
    cik = row["cik"]

    manifest_rows = []
    firm_info = row.to_dict()

    try:
        company = edgar.Company(int(cik))
        filings = company.get_filings(
            form=form, filing_date=f"{start_date}:{end_date}", amendments=allow_amendments,
        )
        firm_info["sic"] = str(getattr(company, "sic", "") or "")
        # `industry` since edgartools 5.55 (was `sic_description`). Read both, and
        # do NOT swallow a miss: a silent "" here is what left industry_group empty
        # for all 517 firms and stayed empty, because script 00 only carries the
        # previous value forward.
        industry = getattr(company, "industry", None) or getattr(company, "sic_description", None)
        if not industry and not firm_info.get("industry_group"):
            pipeline_logger.log_event(
                pipeline_step="edgar_fetch", level="WARNING",
                message=f"No industry/sic_description for {ticker} (CIK {cik}); "
                        f"industry_group left empty",
                ticker=ticker, cik=cik)
        firm_info["industry_group"] = industry or firm_info.get("industry_group", "")
    except Exception as e:
        pipeline_logger.log_event(
            pipeline_step="edgar_fetch", level="ERROR",
            message=f"Error looking up {ticker} (CIK {cik}): {e}",
            ticker=ticker, cik=cik, details={"error": str(e)},
        )
        return firm_info, manifest_rows

    # Iterate the Filing objects directly (EntityFilings is iterable) rather
    # than round-tripping through to_pandas() and re-querying by accession
    # number to fetch content — that re-query (company.get_filings(
    # accession_number=...).latest()) intermittently returned None for a
    # small fraction of filings ("'NoneType' object has no attribute
    # 'download'"), for reasons not worth chasing when the fix is simply not
    # doing a redundant second lookup in the first place.
    for f in filings:
        acc_num = f.accession_number
        filing_date = pd.to_datetime(f.filing_date).date()
        year = filing_date.year
        report_date_raw = getattr(f, "period_of_report", None)
        report_date = pd.to_datetime(report_date_raw).date() if report_date_raw else None

        # form.replace(" ", "") -- "DEF 14A" has a space, which is otherwise
        # a valid (if annoying) filename character; sanitized so every form
        # gets a clean single-token filename, not just the hyphenated ones
        # (10-K/10-Q) this originally shipped with.
        filename = f"{ticker}_{year}_{form.replace(' ', '')}_{acc_num}.html.gz"
        local_path = html_dir / filename

        existing_row = None
        if existing_manifest_df is not None:
            match = existing_manifest_df[existing_manifest_df["accession_number"] == acc_num]
            if not match.empty:
                existing_row = match.iloc[0]

        download_status = existing_row["download_status"] if existing_row is not None else "pending"
        parse_status = existing_row["parse_status"] if existing_row is not None else "pending"
        created_at = existing_row.get("created_at", datetime.now()) if existing_row is not None else datetime.now()

        # OUR OWN idempotency check, on top of edgartools' own cache: if we
        # already wrote this filing's gzip mirror, don't even ask edgartools
        # for it again.
        if local_path.exists() and local_path.stat().st_size > 0:
            download_status = "completed"
        else:
            try:
                html = f.html()
                with gzip.open(local_path, "wt", encoding="utf-8") as out:
                    out.write(html)
                download_status = "completed"
                pipeline_logger.log_event(
                    pipeline_step="edgar_fetch", level="SUCCESS",
                    message="Filing fetched and cached",
                    ticker=ticker, cik=cik, accession_number=acc_num,
                    details={"stored_bytes": local_path.stat().st_size},
                )
            except Exception as e:
                download_status = f"failed: {e}"
                pipeline_logger.log_event(
                    pipeline_step="edgar_fetch", level="ERROR",
                    message=f"Error fetching {ticker} {acc_num}: {e}",
                    ticker=ticker, cik=cik, accession_number=acc_num, details={"error": str(e)},
                )

        manifest_rows.append({
            "document_id": acc_num,
            "cik": cik,
            "ticker": ticker,
            "country": row.get("country", "US"),
            "source": row.get("source", "SEC_EDGAR"),
            "form_type": form,
            "filing_date": filing_date,
            "period_end_date": report_date,
            "accession_number": acc_num,
            "sec_url": getattr(f, "url", ""),
            "local_path": str(local_path),
            "download_status": download_status,
            "parse_status": parse_status,
            "prefilter_status": existing_row.get("prefilter_status", "pending") if existing_row is not None else "pending",
            "llm_status": existing_row.get("llm_status", "pending") if existing_row is not None else "pending",
            "priority_score": existing_row.get("priority_score", 1.0) if existing_row is not None else 1.0,
            "batch_id": existing_row.get("batch_id", "") if existing_row is not None else "",
            "created_at": created_at,
            "updated_at": datetime.now(),
        })

    return firm_info, manifest_rows


def _safe_write_universe(universe_path: Path, full_universe_df: pd.DataFrame, updated_rows: list[dict]) -> None:
    """Writes `universe_path` by MERGING `updated_rows` into the FULL
    original universe (by ticker), never by replacing the file with just
    `updated_rows` outright.

    Real incident this fixes (2026-09-04): `universe_path` (firm_universe.
    parquet) is a single SHARED file every fetch script (10-K/10-Q/proxy/
    8-K) reads AND writes -- unlike every other output in this pipeline
    (prefilter_scores, ai_frames, golden_set labels, ...), which is
    append-only `__run=...__part=...parquet`. The periodic checkpoint used
    to `pd.DataFrame(updated_universe).to_parquet(universe_path)` with
    `updated_universe` containing only the tickers processed SO FAR in
    THIS run -- a killed/interrupted run (Ctrl-C, an exception escaping
    the checkpoint window) truncated the shared 517-firm universe down to
    however many tickers had been processed at that point (confirmed:
    517 -> 150 from one interrupted run). Recovered from the last B2 push,
    but the file being a single mutable target made that recovery
    necessary at all. Merging against the full baseline means an
    interrupted run can only ever update a subset of rows, never drop the
    rest."""
    updated_by_ticker = {row["ticker"]: row for row in updated_rows}
    merged = [updated_by_ticker.get(row["ticker"], row) for row in full_universe_df.to_dict("records")]
    # A ticker in updated_rows but NOT in the original baseline (shouldn't
    # happen -- fetch_filings only ever iterates full_universe_df -- but
    # checked rather than silently dropped if it ever does).
    known_tickers = {row["ticker"] for row in full_universe_df.to_dict("records")}
    merged.extend(row for row in updated_rows if row["ticker"] not in known_tickers)
    pd.DataFrame(merged).to_parquet(universe_path, index=False)


def fetch_filings(universe_df, form, start_date, end_date, allow_amendments, html_dir, manifest_path, universe_path):
    """Fetches `form` filings for every (ticker, cik) in universe_df between
    start_date/end_date, writes gzip'd HTML to html_dir, and builds/updates
    manifest_path. Idempotent: rerun skips any filing whose gzip file
    already exists — no re-fetch, no re-write."""
    html_dir.mkdir(parents=True, exist_ok=True)
    full_universe_df = universe_df.copy()  # see _safe_write_universe

    # _fully_done(ticker) below matches manifest rows by TICKER alone (not
    # CIK) — if universe_df ever contains the same ticker twice (found once,
    # for real: XOM listed under two different CIKs in configs/us/universe.csv
    # after an untracked corporate-CIK-reassignment addition), each
    # duplicate row independently re-triggers the "already done, carry
    # forward existing rows" branch, appending that ticker's full existing
    # row set again — doubling it every single rerun (2 dupes/run compounds
    # to 4x, 16x... across repeated fetch invocations; confirmed exactly
    # this shape in filing_manifest_10q.parquet). Deduping here is a
    # backstop against that recurring, independent of also fixing the
    # source data in universe.csv — first row wins.
    n_before = len(universe_df)
    universe_df = universe_df.drop_duplicates(subset=["ticker"], keep="first")
    if len(universe_df) < n_before:
        pipeline_logger.log_event(
            pipeline_step="edgar_fetch", level="WARNING",
            message=(
                f"universe_df had {n_before - len(universe_df)} duplicate ticker row(s) "
                f"(same ticker, different CIK) — kept the first, dropped the rest."
            ),
        )

    existing_manifest_df = None
    if manifest_path.exists():
        try:
            existing_manifest_df = pd.read_parquet(manifest_path)
        except Exception:
            pass

    updated_universe = []
    manifest_data = []

    def _fully_done(ticker):
        """True if every known-existing manifest row for this ticker is
        already completed AND its gzip file is actually on disk — lets a
        retry pass skip re-querying edgartools for the whole company's
        filing list just to retry a handful of other companies' failures.
        Without this, a rerun that only needs to retry 14/517 companies
        still re-discovers all 517 from scratch every time."""
        if existing_manifest_df is None:
            return False
        rows = existing_manifest_df[
            (existing_manifest_df["ticker"] == ticker) & (existing_manifest_df["form_type"] == form)
        ]
        if rows.empty:
            return False
        return bool((rows["download_status"] == "completed").all()) and all(
            Path(p).exists() and Path(p).stat().st_size > 0 for p in rows["local_path"]
        )

    skipped_tickers = set()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {}
        for _, row in universe_df.iterrows():
            ticker = row["ticker"]
            if _fully_done(ticker):
                skipped_tickers.add(ticker)
                updated_universe.append(row.to_dict())
                manifest_data.extend(
                    existing_manifest_df[
                        (existing_manifest_df["ticker"] == ticker) & (existing_manifest_df["form_type"] == form)
                    ].to_dict("records")
                )
                continue
            futures[executor.submit(
                _fetch_one_company, row, form, start_date, end_date, allow_amendments, html_dir, existing_manifest_df,
            )] = ticker

        if skipped_tickers:
            print(f"Skipping {len(skipped_tickers)} companies already fully fetched (no network call).")

        completed_since_checkpoint = 0
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Fetching {form}"):
            ticker = futures[future]
            try:
                firm_info, manifest_rows = future.result()
            except Exception as e:
                # A worker thread's uncaught exception (e.g. a raw httpx
                # timeout escaping _fetch_one_company's own try/except)
                # used to propagate out of future.result() and crash the
                # WHOLE run here — losing every other company's already-
                # fetched manifest rows (the gzip files stayed on disk, but
                # the run never got to write filing_manifest*.parquet at
                # all). One company's failure must never cost everyone
                # else's progress.
                pipeline_logger.log_event(
                    pipeline_step="edgar_fetch", level="ERROR",
                    message=f"Unhandled error processing {ticker}, skipping company: {e}",
                    ticker=ticker, details={"error": str(e)},
                )
                continue
            updated_universe.append(firm_info)
            manifest_data.extend(manifest_rows)

            # Checkpoint periodically so a later crash (or Ctrl-C) doesn't
            # lose everything fetched so far in THIS run either.
            completed_since_checkpoint += 1
            if completed_since_checkpoint >= 25:
                _safe_write_universe(universe_path, full_universe_df, updated_universe)
                pd.DataFrame(manifest_data).to_parquet(manifest_path, index=False)
                completed_since_checkpoint = 0

    _safe_write_universe(universe_path, full_universe_df, updated_universe)

    if manifest_data:
        manifest_df = pd.DataFrame(manifest_data)
        manifest_df.to_parquet(manifest_path, index=False)
        pipeline_logger.log_event(
            pipeline_step="edgar_fetch", level="SUCCESS",
            message=f"Built {form} manifest with {len(manifest_df)} filings at {manifest_path}",
            details={"filing_count": len(manifest_df), "form": form},
        )
    else:
        pipeline_logger.log_event(
            pipeline_step="edgar_fetch", level="WARNING",
            message=f"No {form} filings found for the configured universe/date range.",
        )
