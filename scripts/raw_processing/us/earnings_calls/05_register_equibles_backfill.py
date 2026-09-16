"""
scripts/raw_processing/us/earnings_calls/05_register_equibles_backfill.py —
manifest of the Equibles backfill transcripts, the fourth earnings-call source.

The backfill files (data/raw/earnings_calls_equibles_backfill/<TICKER>/
<TICKER>_<eventdate>_FY<fy>Q<q>.json.gz, same payload as the Equibles raw
files plus `equibles_event_date`) were fetched for quarters the other three
sources miss. Their fiscal labels carry the same fiscal-year conventions as
Equibles, so the document id is keyed by the event date instead of the label:
`{TICKER}_EQB_{YYYY-MM-DD}` never collides with a `{TICKER}_{YYYY}Q{N}` id of
another source. Metadata date = the event date, fiscal period = the payload's
year and quarter; scripts/bronze/call_transcripts.py checks both against the
text like every other source (lowest source priority).

Writes data/interim/manifests/filing_manifest_earnings_calls_equibles_backfill.parquet
(rebuilt from the files on disk; no network).

Usage:
    .venv/bin/python scripts/raw_processing/us/earnings_calls/05_register_equibles_backfill.py
"""

from __future__ import annotations

import gzip
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[4]
SOURCE = "equibles_backfill:GetEarningsCallTranscript"
FORM_TYPE = "Earnings call transcript"
FILE_NAME = re.compile(r"^(?P<ticker>.+)_(?P<date>\d{4}-\d{2}-\d{2})_FY(?P<fy>\d{4})Q(?P<q>[1-4])\.json\.gz$")


def main() -> None:
    config = yaml.safe_load((REPO_ROOT / "configs" / "us" / "config.yaml").read_text())
    raw_dir = REPO_ROOT / "data" / "raw" / "earnings_calls_equibles_backfill"
    universe = pd.read_csv(REPO_ROOT / "configs" / "us" / "universe.csv", dtype={"cik": str})
    cik_by_ticker = dict(zip(universe["ticker"].str.replace(".", "-", regex=False), universe["cik"]))
    now = datetime.now(timezone.utc)
    rows = []
    for path in sorted(raw_dir.glob("*/*.json.gz")):
        m = FILE_NAME.match(path.name)
        if not m:
            raise ValueError(f"unexpected backfill file name: {path}")
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            payload = json.load(fh)
        rows.append({
            "document_id": f"{m['ticker']}_EQB_{m['date']}", "ticker": m["ticker"].replace("-", "."),
            "cik": int(cik_by_ticker[m["ticker"]]) if m["ticker"] in cik_by_ticker else 0,
            "source": SOURCE, "form_type": FORM_TYPE, "filing_type": "earnings_call",
            "filing_date": payload.get("equibles_event_date") or m["date"], "period_end_date": f"{m['fy']}Q{m['q']}",
            "local_path": str(path), "format": "json", "download_status": "completed",
            "n_bytes": path.stat().st_size, "n_turns": len(payload.get("structured_content") or []),
            "n_chars": len(payload.get("content") or ""), "created_at": now, "updated_at": now,
        })
    manifest = pd.DataFrame(rows)
    out = REPO_ROOT / config["storage"]["interim_manifests"] / "filing_manifest_earnings_calls_equibles_backfill.parquet"
    manifest.to_parquet(out, index=False)
    print(f"{len(manifest)} backfill transcripts, {manifest['ticker'].nunique()} tickers -> {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
