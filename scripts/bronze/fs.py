"""
scripts/bronze/fs.py — one bronze table per fs (FactSet) raw dataset under
data/raw/fs/ (docs/sources/fs.md, docs/plans/fs_gold_replacement.md).

No cross-dataset joins here (bronze convention: one cleaned table per
source, typed and deduplicated to the current run, no analysis-universe
filter -- that happens in silver). `fundamentals_long.parquet` (ARPT+STND
long form) is not bronze-ified: only the STND wide extracts and the parsed
report-dates table are read downstream.
"""

from __future__ import annotations

import polars as pl
from _paths import L

BUILDER = "scripts/bronze/fs.py"
FS_RAW = L.DATA / "raw" / "fs"


def main() -> None:
    id_map = FS_RAW / "reference" / "id_map.parquet"
    L.write_table("bronze.fs_id_map", pl.scan_parquet(id_map), keys=["ticker"], inputs=[id_map], builder=BUILDER)

    prices = FS_RAW / "prices" / "daily" / "prices_daily.parquet"
    L.write_table("bronze.fs_prices_daily", pl.scan_parquet(prices), keys=["ticker", "date"],
                  inputs=[prices], builder=BUILDER)

    mcap = FS_RAW / "market_cap" / "market_cap_daily.parquet"
    L.write_table("bronze.fs_market_cap_daily", pl.scan_parquet(mcap), keys=["ticker", "date"],
                  inputs=[mcap], builder=BUILDER)

    bench = FS_RAW / "benchmark" / "benchmark_sp500_daily.parquet"
    L.write_table("bronze.fs_benchmark_daily", pl.scan_parquet(bench), keys=["date"],
                  inputs=[bench], builder=BUILDER)

    ann = FS_RAW / "fundamentals" / "fundamentals_stnd_wide_annual.parquet"
    L.write_table("bronze.fs_fundamentals_annual", pl.scan_parquet(ann), keys=["ticker", "period_end"],
                  inputs=[ann], builder=BUILDER)

    qtr = FS_RAW / "fundamentals" / "fundamentals_stnd_wide_quarterly.parquet"
    L.write_table("bronze.fs_fundamentals_quarterly", pl.scan_parquet(qtr), keys=["ticker", "period_end"],
                  inputs=[qtr], builder=BUILDER)

    profile_files = sorted((FS_RAW / "profile").glob("profile_*.parquet"))
    if profile_files:
        latest_profile = profile_files[-1]
        L.write_table("bronze.fs_profile", pl.scan_parquet(latest_profile), keys=["ticker"],
                      inputs=[latest_profile], builder=BUILDER)

    report_dates = FS_RAW / "report_dates" / "fs_report_dates_parsed.parquet"
    if report_dates.exists():
        L.write_table("bronze.fs_report_dates", pl.scan_parquet(report_dates),
                      keys=["ticker", "period", "period_end"], inputs=[report_dates], builder=BUILDER)
    else:
        print(f"SKIP bronze.fs_report_dates: {report_dates} not built yet "
              f"(run scripts/sources/fs/parse_report_dates.py first)")


if __name__ == "__main__":
    main()
