"""
scripts/bronze/patents.py — bronze.patents_firm_year, bronze.patents_preshock:
Google Patents / OECD (2025) AI-patent counts for S&P 500 assignees, as
fetched by scripts/raw_ingestion/patents/{run_oecd_ai_patents,
build_sec_preshock_patents}.py and sitting on disk under data/raw/patents/.
No universe filter here (that's silver's job): this only types and
deduplicates the source tables.
"""

from __future__ import annotations

import polars as pl
from _paths import L

PATENTS_RAW = L.DATA / "raw" / "patents"
FIRM_YEAR_SRC = PATENTS_RAW / "sp500_firm_year_ai_patents_oecd2025.parquet"
PRESHOCK_SRC = PATENTS_RAW / "sp500_firm_preshock_patent_capacity.parquet"

BUILDER = "scripts/bronze/patents.py"


def main() -> None:
    firm_year = pl.scan_parquet(FIRM_YEAR_SRC).with_columns(
        pl.col("ticker").str.to_uppercase(),
        pl.col("year").cast(pl.Int64),
    )
    L.write_table("bronze.patents_firm_year", firm_year, keys=["ticker", "year"],
                  inputs=[FIRM_YEAR_SRC], builder=BUILDER)

    preshock = pl.scan_parquet(PRESHOCK_SRC).with_columns(pl.col("ticker").str.to_uppercase())
    L.write_table("bronze.patents_preshock", preshock, keys=["ticker"],
                  inputs=[PRESHOCK_SRC], builder=BUILDER)


if __name__ == "__main__":
    main()
