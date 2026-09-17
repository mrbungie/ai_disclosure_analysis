#!/usr/bin/env python
"""
FactSet benchmark parser: read raw benchmark JSON, output long-format parquet.
Usage: uv run python scripts/sources/fs/parse_benchmark.py
"""
import json
from pathlib import Path
from datetime import datetime
import polars as pl

def parse_benchmark():
    """Parse FactSet S&P 500 benchmark JSON into long-format parquet."""
    repo_root = Path(__file__).parent.parent.parent.parent
    input_file = repo_root / "data/raw/fs/benchmark/fs_benchmark_sp500.json"
    output_dir = repo_root / "data/raw/fs/benchmark"
    output_file = output_dir / "benchmark_sp500_daily.parquet"

    if not input_file.exists():
        print(f"Input file not found: {input_file}")
        return

    print(f"Parsing {input_file}...")

    with open(input_file) as f:
        benchmark_data = json.load(f)

    # Expected structure: {"SP50-SPX": {"date": [...], "price": [...], "total_return": [...]}}
    rows = []

    for symbol, metrics in benchmark_data.items():
        dates = metrics.get("date", [])
        prices = metrics.get("price", [])
        total_returns = metrics.get("total_return", [])

        print(f"  {symbol}: {len(dates)} observations")

        for i in range(len(dates)):
            date_int = dates[i] if i < len(dates) else None
            price = prices[i] if i < len(prices) else None
            total_return = total_returns[i] if i < len(total_returns) else None

            # Skip if no date
            if date_int is None:
                continue

            # Handle FactSet null markers
            if price == "@NA" or price == "":
                price = None
            if total_return == "@NA" or total_return == "":
                total_return = None

            rows.append({
                "date": date_int,
                "price": price,
                "total_return": total_return,
            })

    # Create DataFrame
    df = pl.DataFrame(rows, schema={
        "date": pl.Int32,
        "price": pl.Float64,
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
        pl.col("date_parsed").alias("date"),
        pl.col("price"),
        pl.col("total_return"),
    )

    # Sort by date descending (most recent first, per FactSet convention)
    df = df.sort("date", descending=True)

    # Write parquet
    df.write_parquet(output_file)

    # Coverage stats
    total_rows = len(df)
    date_min = df["date"].min()
    date_max = df["date"].max()

    print(f"\n=== Benchmark Coverage ===")
    print(f"Total rows: {total_rows}")
    print(f"Date range: {date_min} to {date_max}")
    print(f"Output: {output_file}")
    return df

if __name__ == "__main__":
    parse_benchmark()
