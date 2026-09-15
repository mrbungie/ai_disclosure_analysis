"""
scripts/us/sec_letters/01_fetch_filings.py — fetches SEC comment-letter
threads (SEC's own letter = form UPLOAD, the company's reply = form
CORRESP) for the firm universe. See docs/document_expansion_plan.md
Fase 2.

NOT built on edgar_fetch.py's generic HTML fetcher: `Filing.html()`
returns None for UPLOAD/CORRESP (these aren't HTML documents in
edgartools' sense) -- verified directly. edgartools instead exposes a
purpose-built `Filing.correspondence()` that returns the WHOLE THREAD
(every UPLOAD/CORRESP entry belonging to the same SEC review episode,
correctly typed as SEC_COMMENT / COMPANY_RESPONSE / REVIEW_COMPLETE)
with real text in `.body` -- richer and structurally cleaner than
reconstructing a thread from separately-fetched raw documents ourselves.

Only iterates each company's UPLOAD filings (the SEC's own letters) --
each one's `.correspondence()` call already returns the CORRESP replies
in the same thread, so iterating CORRESP filings too would just refetch
the same threads a second time. Deduped by each entry's own
`accession_no` (one thread can span several UPLOAD filings if the SEC
sent more than one letter in the same review).

Output: one gzip'd .txt per correspondence entry (data/raw/sec_letters/)
+ a manifest parquet (data/interim/manifests/filing_manifest_sec_letters.parquet)
with one row per entry -- ticker, cik, accession_no, correspondence_type,
dates, referenced_form, local_path.

Usage:
    uv run python scripts/us/sec_letters/01_fetch_filings.py
"""

import gzip
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import edgar
import pandas as pd
import yaml
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common"))
import pipeline_logger

MAX_WORKERS = 8
_seen_lock = threading.Lock()


def _fetch_company_threads(cik: int, ticker: str, start_date: str, end_date: str,
                            text_dir: Path, seen_accessions: set) -> list[dict]:
    rows = []
    try:
        company = edgar.Company(cik)
        upload_filings = company.get_filings(form="UPLOAD", filing_date=f"{start_date}:{end_date}")
    except Exception as e:
        pipeline_logger.log_event(
            pipeline_step="sec_letters_fetch", level="ERROR",
            message=f"Error looking up {ticker} (CIK {cik}): {e}", ticker=ticker, cik=cik)
        return rows

    for f in upload_filings:
        try:
            thread = f.correspondence()
        except Exception as e:
            pipeline_logger.log_event(
                pipeline_step="sec_letters_fetch", level="ERROR",
                message=f"Error fetching correspondence thread for {ticker} {f.accession_number}: {e}",
                ticker=ticker, cik=cik, accession_number=f.accession_number)
            continue

        # `.correspondence()` returns None (not an exception) for some
        # filings -- verified 82/517 companies hit this and, before this
        # check, crashed the WHOLE company's processing (the AttributeError
        # on the next line propagated past this function's try/except,
        # caught only by main()'s outer "Unhandled error, skip company"
        # handler, discarding any OTHER good UPLOAD filings that same
        # company had already found). Skipping just this one filing, not
        # the company, is the actual fix.
        if thread is None:
            pipeline_logger.log_event(
                pipeline_step="sec_letters_fetch", level="WARNING",
                message=f"correspondence() returned None for {ticker} {f.accession_number}, skipping this filing",
                ticker=ticker, cik=cik, accession_number=f.accession_number)
            continue

        for entry in thread.entries:
            with _seen_lock:
                if entry.accession_no in seen_accessions:
                    continue
                seen_accessions.add(entry.accession_no)

            body = entry.body or ""
            filename = f"{ticker}_{entry.accession_no}.txt.gz"
            local_path = text_dir / filename
            if not (local_path.exists() and local_path.stat().st_size > 0):
                with gzip.open(local_path, "wt", encoding="utf-8") as out:
                    out.write(body)

            # `errors="coerce"` -- verified real, not hypothetical: edgartools
            # sometimes parses `response_date` out of the letter's own body
            # text, and gets a whole sentence instead of a date (e.g. "May 11,
            # 2022, to the Staff's comment letter dated April 27, 2022. To
            # assist your review, we have included..."). Without coercion,
            # pd.to_datetime raises and crashes the WHOLE company (6/517 hit
            # this before the fix, same "one bad entry costs everyone else's
            # progress" failure mode as the None-thread bug above).
            filing_date_parsed = pd.to_datetime(entry.filing_date, errors="coerce") if entry.filing_date else None
            response_date_parsed = pd.to_datetime(
                getattr(entry, "response_date", None), errors="coerce") if getattr(entry, "response_date", None) else None
            rows.append({
                "document_id": entry.accession_no,
                "cik": cik,
                "ticker": ticker,
                "country": "US",
                "source": "SEC_EDGAR",
                "form_type": entry.form,
                "correspondence_type": str(entry.correspondence_type),
                "filing_date": filing_date_parsed.date() if pd.notna(filing_date_parsed) else None,
                "response_date": response_date_parsed.date() if pd.notna(response_date_parsed) else None,
                "referenced_form": getattr(entry, "referenced_form", None),
                "accession_number": entry.accession_no,
                "local_path": str(local_path),
                "n_chars": len(body),
                "download_status": "completed",
                "parse_status": "pending",
                "created_at": datetime.now(),
                "updated_at": datetime.now(),
            })
    return rows


def main():
    with open("configs/us/config.yaml") as f:
        config = yaml.safe_load(f)

    universe_path = Path(config["storage"]["interim_manifests"]) / "firm_universe.parquet"
    if not universe_path.exists():
        print(f"Universe file not found at {universe_path}. Run scripts/us/00_build_firm_universe.py first.")
        return
    universe_df = pd.read_parquet(universe_path).drop_duplicates(subset=["ticker"], keep="first")

    edgar.set_identity(config["sec"]["user_agent"])
    edgar.set_local_storage_path(str(Path(".edgartools_data") / "sec_letters"))
    edgar.use_local_storage(True)

    text_dir = Path(config["storage"]["raw_sec_letters"])
    text_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(config["storage"]["interim_manifests"]) / "filing_manifest_sec_letters.parquet"

    existing = pd.read_parquet(manifest_path) if manifest_path.exists() else None
    seen_accessions = set(existing["accession_number"]) if existing is not None else set()
    fully_done_tickers = set()
    if existing is not None:
        # A ticker with any manifest rows already has its UPLOAD filings'
        # threads fully expanded (no partial state per-thread) -- skip its
        # network calls entirely on a rerun.
        fully_done_tickers = set(existing["ticker"].unique())

    corr_cfg = config["corpus"]["filings_correspondence"]
    filing_date = corr_cfg["filing_date"]

    all_rows = list(existing.to_dict("records")) if existing is not None else []
    pending = [row for _, row in universe_df.iterrows() if row["ticker"] not in fully_done_tickers]
    if len(pending) < len(universe_df):
        print(f"Skipping {len(universe_df) - len(pending)} companies already fetched (no network call).")

    completed_since_checkpoint = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(_fetch_company_threads, int(row["cik"]), row["ticker"],
                             filing_date["from"], filing_date["to"], text_dir, seen_accessions): row["ticker"]
            for row in pending
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="SEC comment-letter threads"):
            ticker = futures[future]
            try:
                new_rows = future.result()
            except Exception as e:
                pipeline_logger.log_event(
                    pipeline_step="sec_letters_fetch", level="ERROR",
                    message=f"Unhandled error processing {ticker}, skipping company: {e}",
                    ticker=ticker, details={"error": str(e)})
                continue
            all_rows.extend(new_rows)
            completed_since_checkpoint += 1
            if completed_since_checkpoint >= 25:
                pd.DataFrame(all_rows).to_parquet(manifest_path, index=False)
                completed_since_checkpoint = 0

    manifest_df = pd.DataFrame(all_rows)
    manifest_df.to_parquet(manifest_path, index=False)
    print(f"\n{len(manifest_df):,} correspondence entries -> {manifest_path}")
    if len(manifest_df):
        print(manifest_df["correspondence_type"].value_counts().to_string())
    pipeline_logger.log_event(
        pipeline_step="sec_letters_fetch", level="SUCCESS",
        message=f"Built SEC comment-letter manifest with {len(manifest_df)} entries",
        details={"entry_count": len(manifest_df)})


if __name__ == "__main__":
    main()
