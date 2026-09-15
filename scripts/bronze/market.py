"""
scripts/bronze/market.py — bronze.market_prices, bronze.market_factors_daily,
bronze.market_factors_monthly (daily prices per ticker and Fama-French 3
factors, as downloaded).
"""

from __future__ import annotations

import polars as pl
from _paths import FACTORS, PRICES, L, files

BUILDER = "scripts/bronze/market.py"


def main() -> None:
    price_files = files(PRICES, "*.parquet")
    prices = pl.concat([pl.scan_parquet(f).with_columns(pl.lit(str(f)).alias("filename")) for f in price_files],
                       how="diagonal_relaxed")
    L.write_table("bronze.market_prices", prices, keys=["ticker", "date"], inputs=price_files, builder=BUILDER)
    for freq in ("daily", "monthly"):
        src = FACTORS / f"ff3_{freq}.parquet"
        lf = pl.scan_parquet(src)
        L.write_table(f"bronze.market_factors_{freq}", lf, keys=None, sort_by=[lf.collect_schema().names()[0]],
                      inputs=[src], builder=BUILDER)


if __name__ == "__main__":
    main()
