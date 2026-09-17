#!/usr/bin/env python
"""
FactSet market cap parser: read raw JSON batches, output long-format parquet.
Usage: uv run python scripts/sources/fs/parse_market_cap.py
"""
import json
import glob
from pathlib import Path
from datetime import datetime
import polars as pl

def parse_market_cap_batches():
    """Parse all FactSet market cap batch JSONs into a long-format parquet."""
    repo_root = Path(__file__).parent.parent.parent.parent
    batch_dir = repo_root / "data/raw/fs/market_cap"
    output_file = batch_dir / "market_cap_daily.parquet"

    # Firm universe (silver, sp500_2021_start_panel — 499 tickers, includes FRC which
    # the gold spine is missing) for delisting-date truncation.
    firm_df = pl.read_parquet(repo_root / "data/silver/firm_universe.parquet")
    firm_df = firm_df.filter(
        pl.col("membership_groups").list.contains("sp500_2021_start_panel")
    )

    # Load ID mapping for ticker to factset_id mapping. Only (ticker, factset_id)
    # pairs present here are kept — anything else in the raw batches is junk.
    id_map_file = repo_root / "data/raw/fs/reference/id_map.parquet"
    id_map_dict = {}  # {factset_id: ticker}
    if id_map_file.exists():
        id_map = pl.read_parquet(id_map_file)
        for row in id_map.to_dicts():
            if row["factset_id"] is not None:
                id_map_dict[row["factset_id"]] = row["ticker"]
        print(f"Loaded ID mapping with {len(id_map_dict)} entries")

    # Collect all rows
    rows = []
    factset_ids_seen = set()

    # Read all batch files
    batch_files = sorted(glob.glob(str(batch_dir / "fs_marketcap_batch_*.json")))
    print(f"Processing {len(batch_files)} batch files...")

    for batch_file in batch_files:
        with open(batch_file) as f:
            batch_data = json.load(f)

        for factset_id, metrics in batch_data.items():
            factset_ids_seen.add(factset_id)

            # Only keep (ticker, factset_id) pairs present in id_map — drop anything
            # else outright rather than guessing a ticker from the id.
            if factset_id not in id_map_dict:
                continue
            ticker = id_map_dict[factset_id]

            dates = metrics.get("date", [])
            market_caps = metrics.get("market_cap", [])

            # Skip if no data
            if not dates:
                continue

            # Helper to safely extract value
            def safe_get(arr, i):
                if arr is None or i >= len(arr):
                    return None
                val = arr[i]
                if val is None or val == "@NA" or val == "":
                    return None
                return val

            # Build rows for this ticker
            n_obs = len(dates)
            for i in range(n_obs):
                rows.append({
                    "ticker": ticker,
                    "factset_id": factset_id,
                    "date": dates[i],
                    "market_cap": safe_get(market_caps, i),
                })

    # Create DataFrame
    df = pl.DataFrame(rows, schema={
        "ticker": pl.Utf8,
        "factset_id": pl.Utf8,
        "date": pl.Int32,
        "market_cap": pl.Float64,
    })

    # Convert date from YYYYMMDD int to Date
    df = df.with_columns(
        pl.col("date").cast(pl.Utf8).str.slice(0, 4).cast(pl.Int32).alias("year"),
        pl.col("date").cast(pl.Utf8).str.slice(4, 2).cast(pl.Int32).alias("month"),
        pl.col("date").cast(pl.Utf8).str.slice(6, 2).cast(pl.Int32).alias("day"),
    )
    df = df.with_columns(
        pl.datetime(pl.col("year"), pl.col("month"), pl.col("day")).cast(pl.Date).alias("date_parsed")
    ).select(
        pl.col("ticker"),
        pl.col("factset_id"),
        pl.col("date_parsed").alias("date"),
        pl.col("market_cap"),
    )

    # Deduplicate on (ticker, date), preferring non-null market_cap
    df_sorted = df.with_columns(
        pl.col("market_cap").is_null().cast(pl.Int32).alias("_is_null")
    ).sort(["ticker", "date", "_is_null"]).drop("_is_null")

    df_dedup = df_sorted.unique(subset=["ticker", "date"], keep="first")

    # Truncate market_cap after delisting_date (same rationale as prices: FactSet
    # holds some delisted securities' last value frozen well past the actual
    # delisting date).
    delisting_dates = {}
    for row in firm_df.to_dicts():
        if row["delisting_date"] is not None:
            delisting_dates[row["ticker"]] = row["delisting_date"]

    if delisting_dates:
        print(f"Truncating {len(delisting_dates)} tickers at delisting date...")
        rows_final = []
        for row in df_dedup.to_dicts():
            ticker = row["ticker"]
            date_val = row["date"]
            if ticker in delisting_dates:
                delisting_date = delisting_dates[ticker]
                if hasattr(delisting_date, "date"):
                    delisting_date = delisting_date.date()
                if date_val > delisting_date:
                    row["market_cap"] = None
            rows_final.append(row)
        df_dedup = pl.DataFrame(rows_final, schema=df_dedup.schema)

    # Write parquet
    df_dedup.write_parquet(output_file)

    # Coverage stats
    unique_tickers = df_dedup["ticker"].n_unique()
    total_rows = len(df_dedup)
    date_min = df_dedup["date"].min()
    date_max = df_dedup["date"].max()

    print(f"\n=== Market Cap Coverage ===")
    print(f"Unique tickers in data: {unique_tickers}")
    print(f"Total rows: {total_rows}")
    print(f"Date range: {date_min} to {date_max}")
    print(f"Unique factset_ids: {len(factset_ids_seen)}")

    universe_tickers = set(firm_df["ticker"].drop_nulls().to_list())
    tickers_in_data = set(df_dedup["ticker"].unique().to_list())
    missing_tickers = sorted(universe_tickers - tickers_in_data)
    print(f"\nUniverse tickers missing from data ({len(missing_tickers)}):")
    for t in missing_tickers:
        print(f"  - {t}")

    print(f"\nUnits: market_cap is in MILLIONS of USD (curn='LOCAL', all US listings "
          f"trade in USD; e.g. AAPL 2026-09-17 = {df_dedup.filter((pl.col('ticker')=='AAPL') & (pl.col('date')==date_max))['market_cap'].item() if df_dedup.filter((pl.col('ticker')=='AAPL') & (pl.col('date')==date_max)).height else 'n/a'} => ~$4.85T market cap).")
    print(f"Output: {output_file}")
    return df_dedup, missing_tickers

if __name__ == "__main__":
    parse_market_cap_batches()
