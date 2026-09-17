#!/usr/bin/env python
"""
Build a clean parquet of fs (FactSet) earnings-release dates from the raw
per-batch JSON pull (`data/raw/fs/report_dates/fs_report_dates_*.json`, one
dict per file mapping factset_id -> {fiscal_date_ann, eps_rpt_date_ann,
source_doc_ann, fiscal_date_qtr, eps_rpt_date_qtr, source_doc_qtr}, each a
parallel list over periods).

This is the point-in-time FALLBACK date source (docs/plans/fs_gold_replacement.md
S2): fs has no filing/report date on its STND fundamentals tables, only
FF_EPS_RPT_DATE (earnings release date, 0-21 days before the SEC filing) and
FF_SOURCE_DOC (the form the fiscal period was disclosed in, e.g. "10-K").
The EDGAR-matched filing_date is still the PRIMARY source; this table is only
consulted when no EDGAR filing matches a period within tolerance.

Usage: uv run python scripts/sources/fs/parse_report_dates.py
Rerunnable: re-reads every fs_report_dates_*.json each time.

Output: data/raw/fs/report_dates/fs_report_dates_parsed.parquet
  ticker, factset_id, period ("annual"|"quarterly"), period_end (date),
  eps_rpt_date (date), source_doc (str, e.g. "10-K"/"10-Q")
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).parent.parent.parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw" / "fs" / "report_dates"
OUT_PATH = RAW_DIR / "fs_report_dates_parsed.parquet"

# id_map.parquet gives factset_id -> ticker (report_dates JSON is keyed by factset_id)
ID_MAP = REPO_ROOT / "data" / "raw" / "fs" / "reference" / "id_map.parquet"


def parse_date(s: str | None, fmt: str) -> str | None:
    if not s:
        return None
    return s


def main() -> None:
    files = sorted(glob.glob(str(RAW_DIR / "fs_report_dates_*.json")))
    print(f"Found {len(files)} batch files")
    rows = []
    for f in files:
        batch = json.loads(Path(f).read_text())
        for factset_id, entry in batch.items():
            for period, fkey, rkey, skey in (
                ("annual", "fiscal_date_ann", "eps_rpt_date_ann", "source_doc_ann"),
                ("quarterly", "fiscal_date_qtr", "eps_rpt_date_qtr", "source_doc_qtr"),
            ):
                fiscal_dates = entry.get(fkey) or []
                rpt_dates = entry.get(rkey) or []
                source_docs = entry.get(skey) or []
                for i, fd in enumerate(fiscal_dates):
                    if not fd:
                        continue
                    rd = rpt_dates[i] if i < len(rpt_dates) else None
                    sd = source_docs[i] if i < len(source_docs) else None
                    rows.append((factset_id, period, fd, rd, sd))

    long = pl.DataFrame(rows, schema=["factset_id", "period", "fiscal_date_raw", "eps_rpt_date_raw", "source_doc"],
                        orient="row")
    print(f"Parsed {long.height} (factset_id, period, period_end) rows")

    # fiscal_date_raw is "MM/DD/YYYY", eps_rpt_date_raw is "YYYYMMDD" or null
    long = long.with_columns([
        pl.col("fiscal_date_raw").str.strptime(pl.Date, "%m/%d/%Y", strict=False).alias("period_end"),
        pl.col("eps_rpt_date_raw").str.strptime(pl.Date, "%Y%m%d", strict=False).alias("eps_rpt_date"),
    ]).drop(["fiscal_date_raw", "eps_rpt_date_raw"])

    # de-duplicate: same (factset_id, period, period_end) can appear in more than one
    # batch overlap window; keep the row with a non-null eps_rpt_date if any do
    long = (long.sort(["factset_id", "period", "period_end", "eps_rpt_date"], nulls_last=True)
            .unique(subset=["factset_id", "period", "period_end"], keep="last"))

    id_map = pl.read_parquet(ID_MAP).select("factset_id", "ticker")
    long = long.join(id_map, on="factset_id", how="left")
    missing_ticker = long.filter(pl.col("ticker").is_null())
    if missing_ticker.height:
        print(f"WARNING: {missing_ticker.height} rows have no ticker match in id_map "
              f"({missing_ticker['factset_id'].n_unique()} distinct factset_ids)")

    long = long.select(["ticker", "factset_id", "period", "period_end", "eps_rpt_date", "source_doc"]).sort(
        ["ticker", "period", "period_end"])
    long.write_parquet(OUT_PATH)
    print(f"Wrote {OUT_PATH}: {long.height} rows, {long['factset_id'].n_unique()} factset_ids, "
          f"{long['ticker'].n_unique()} tickers")
    for period in ("annual", "quarterly"):
        sub = long.filter(pl.col("period") == period)
        print(f"  {period}: {sub.height} rows, eps_rpt_date non-null {sub['eps_rpt_date'].is_not_null().mean():.1%}, "
              f"period_end range {sub['period_end'].min()} -> {sub['period_end'].max()}")


if __name__ == "__main__":
    main()
