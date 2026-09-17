#!/usr/bin/env python
"""
FactSet daily prices parser: read raw JSON batches, output long-format parquet.
Usage: uv run python scripts/sources/fs/parse_prices.py
"""
import json
import glob
from pathlib import Path
from datetime import datetime
import polars as pl

def parse_prices_batches():
    """Parse all FactSet price batch JSONs into a long-format parquet."""
    repo_root = Path(__file__).parent.parent.parent.parent
    batch_dir = repo_root / "data/raw/fs/prices/daily"
    output_file = batch_dir / "prices_daily.parquet"

    # Load firm universe (silver, sp500_2021_start_panel membership — 499 tickers,
    # includes FRC which the gold spine is missing) for universe membership and
    # delisting dates.
    firm_df = pl.read_parquet(repo_root / "data/silver/firm_universe.parquet")
    firm_df = firm_df.filter(
        pl.col("membership_groups").list.contains("sp500_2021_start_panel")
    )
    universe_tickers = set(firm_df["ticker"].drop_nulls().to_list())
    print(f"Universe: {len(universe_tickers)} tickers")

    # Load ID mapping table for corrections
    id_map_file = repo_root / "data/raw/fs/reference/id_map.parquet"
    id_map_dict = {}  # {factset_id: universe_ticker}
    if id_map_file.exists():
        id_map = pl.read_parquet(id_map_file)
        # Create mapping from corrected factset_id to universe ticker
        for row in id_map.to_dicts():
            if row["factset_id"] is not None:
                id_map_dict[row["factset_id"]] = row["ticker"]
        print(f"Loaded ID mapping with {len(id_map_dict)} entries")

    # Collect all rows
    rows = []
    factset_ids_seen = set()

    # Read all batch files except _test_batch.json
    batch_files = sorted(glob.glob(str(batch_dir / "fs_prices_batch_*.json")))
    batch_files = [f for f in batch_files if "_test_batch" not in f]

    print(f"\nProcessing {len(batch_files)} batch files...")

    for batch_file in batch_files:
        with open(batch_file) as f:
            batch_data = json.load(f)

        for factset_id, metrics in batch_data.items():
            factset_ids_seen.add(factset_id)

            # Only keep (ticker, factset_id) pairs that are in id_map — anything not
            # in id_map is junk (e.g. leftover wrong-id fix attempts like IHS-US,
            # VIA-US) and must be dropped, not kept under a guessed ticker name.
            if factset_id not in id_map_dict:
                continue
            ticker = id_map_dict[factset_id]

            dates = metrics.get("date")
            prices = metrics.get("price")
            volumes = metrics.get("volume")
            total_returns = metrics.get("total_return")

            # Skip if no data
            if not dates or dates is None:
                continue

            # Helper to safely extract value (handle null/@NA strings)
            def safe_get(arr, i):
                if arr is None or i >= len(arr):
                    return None
                val = arr[i]
                # Handle FactSet null markers
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
                    "price": safe_get(prices, i),
                    "volume": safe_get(volumes, i),
                    "total_return": safe_get(total_returns, i),
                })

    # Create DataFrame with explicit schema to handle nulls
    df = pl.DataFrame(rows, schema={
        "ticker": pl.Utf8,
        "factset_id": pl.Utf8,
        "date": pl.Int32,
        "price": pl.Float64,
        "volume": pl.Float64,
        "total_return": pl.Float64,
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
        pl.col("price"),
        pl.col("volume"),
        pl.col("total_return"),
    )

    # For tickers with ID mappings, prefer the mapped factset_id and exclude the original
    # Build reverse mapping (ticker -> correct_factset_id)
    mapped_tickers = {}
    if id_map_file.exists():
        for row in id_map.to_dicts():
            if row["factset_id"] is not None:
                mapped_tickers[row["ticker"]] = row["factset_id"]

    # Filter out rows with mapped tickers that use the wrong factset_id
    rows_to_keep = []
    for row in df.to_dicts():
        ticker = row["ticker"]
        factset_id = row["factset_id"]

        # If this ticker has a mapping, only keep rows with the correct factset_id
        if ticker in mapped_tickers:
            if factset_id == mapped_tickers[ticker]:
                rows_to_keep.append(row)
        else:
            rows_to_keep.append(row)

    df_filtered = pl.DataFrame(rows_to_keep, schema=df.schema)

    # Deduplicate on (ticker, date), preferring non-null prices
    df_sorted = df_filtered.with_columns(
        pl.col("price").is_null().cast(pl.Int32).alias("_is_null")
    ).sort(["ticker", "date", "_is_null"]).drop("_is_null")

    df_dedup = df_sorted.unique(subset=["ticker", "date"], keep="first")

    # Truncate prices/volume/total_return after delisting_date
    # (frozen prices from delisted ids like SIVBQ and FRCB must be null after delisting)
    delisting_dates = {}  # {ticker: delisting_date}
    for row in firm_df.to_dicts():
        if row["delisting_date"] is not None:
            delisting_dates[row["ticker"]] = row["delisting_date"]

    if delisting_dates:
        print(f"Truncating {len(delisting_dates)} tickers at delisting date...")
        # Iterate through rows and set nulls for dates after delisting
        rows_final = []
        for row in df_dedup.to_dicts():
            ticker = row["ticker"]
            date_val = row["date"]
            if ticker in delisting_dates:
                delisting_date = delisting_dates[ticker]
                # Convert datetime to date if needed
                if hasattr(delisting_date, 'date'):
                    delisting_date = delisting_date.date()
                if date_val > delisting_date:
                    # Set price, volume, total_return to null after delisting
                    row["price"] = None
                    row["volume"] = None
                    row["total_return"] = None
            rows_final.append(row)
        df_dedup = pl.DataFrame(rows_final, schema=df_dedup.schema)

    # Write parquet
    df_dedup.write_parquet(output_file)

    # Coverage stats
    unique_tickers = df_dedup["ticker"].n_unique()
    total_rows = len(df_dedup)
    date_min = df_dedup["date"].min()
    date_max = df_dedup["date"].max()

    print(f"\n=== Coverage ===")
    print(f"Unique tickers in data: {unique_tickers}")
    print(f"Total rows: {total_rows}")
    print(f"Date range: {date_min} to {date_max}")
    print(f"Unique factset_ids: {len(factset_ids_seen)}")

    # Find missing universe tickers
    tickers_in_data = set(df_dedup["ticker"].unique().to_list())
    missing_tickers = sorted(universe_tickers - tickers_in_data)

    print(f"\nUniverse tickers missing from data ({len(missing_tickers)}):")
    for t in missing_tickers:
        print(f"  - {t}")

    # Print ID mappings for fixed tickers
    fixed_ids = df_dedup.filter(
        pl.col("ticker").is_in(["BRK.B", "BF.B"])
    ).select(["ticker", "factset_id"]).unique()
    if len(fixed_ids) > 0:
        print(f"\nFixed ticker ID mappings:")
        for row in fixed_ids.to_dicts():
            print(f"  {row['ticker']} -> {row['factset_id']}")

    print(f"\nOutput: {output_file}")
    return df_dedup, missing_tickers

if __name__ == "__main__":
    parse_prices_batches()
