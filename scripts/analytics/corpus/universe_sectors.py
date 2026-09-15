"""SIC / sector classification of the study universe (thesis.qmd
`fig-sic-distribution`, lines ~376-500, and `tbl-a0-sic-sector-mapping`,
lines ~3891-3950).

Both chunks query the old DuckDB `firm_universe` view and report $N=499$,
matching `silver.firm_universe` (the analysis universe, S&P 500 at
2021-01-01) rather than `bronze.firm_universe` (547 rows, every ticker ever
fetched). One row per firm with its SIC code, 2-digit SIC group, broad SIC
division (fig-sic-distribution Panel A), aggregated economic sector
(fig-sic-distribution Panel C / tbl-a0-sic-sector-mapping), and
`active_status`/`industry_group` (used by the Chapter 3 archetype tables
and sector heatmap, which also need to know which universe firms were
delisted or acquired) -- a later migration pass computes the Panel B
top-12-plus-other breakdown and the per-sector SIC-2 code list directly
from `sic2`/`agg_sector` here.

Usage:
    .venv/bin/python scripts/analytics/corpus/universe_sectors.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import layers as L  # noqa: E402


def sic_division(sic_raw: str | None) -> str:
    if sic_raw is None:
        return "Other / Diversified"
    try:
        s = int(sic_raw)
    except (TypeError, ValueError):
        return "Other / Diversified"
    if s < 2000: return "Mining & Construction"
    elif s < 4000: return "Manufacturing"
    elif s < 5000: return "Transport & Utilities"
    elif s < 6000: return "Wholesale & Retail"
    elif s < 6800: return "Finance, Ins. & Real Estate"
    elif s < 9000: return "Services"
    else: return "Other / Diversified"


def map_sector(sic2: str | None) -> str:
    try:
        s = int(sic2)
    except (TypeError, ValueError):
        return "Other / Diversified"
    if s in (73, 35, 36): return "Technology"
    elif s == 48: return "Communications"
    elif s in (28, 38, 80): return "Healthcare & Pharma"
    elif 60 <= s <= 67: return "Financial Services"
    elif 20 <= s <= 39: return "Industrials & Mfg"
    elif 40 <= s <= 47: return "Transportation"
    elif s == 49: return "Utilities"
    elif 10 <= s <= 14 or s == 29: return "Energy & Mining"
    elif 50 <= s <= 59: return "Retail & Wholesale"
    elif 70 <= s <= 89: return "Business Services"
    else: return "Other / Diversified"


def main() -> None:
    fu = L.scan("silver.firm_universe").filter(pl.col("country_code") == "us").collect()
    fu = fu.with_columns(pl.col("sic").cast(pl.Utf8).str.zfill(4).str.slice(0, 2).alias("sic2"))

    rows = fu.select("sic2", "sic").iter_rows(named=True)
    divisions, sectors = [], []
    for r in rows:
        divisions.append(sic_division(r["sic"]))
        sectors.append(map_sector(r["sic2"]))

    out = fu.with_columns(
        pl.Series("division", divisions),
        pl.Series("agg_sector", sectors),
    ).select("ticker", "company_name", "sic", "sic2", "division", "agg_sector", "active_status", "industry_group")

    out_path = L.results_path("corpus", "universe_sectors.parquet")
    out.write_parquet(out_path)
    print(f"{out.height} firms -> {out_path}")
    print(out.group_by("division").len().sort("len"))
    print(out.group_by("agg_sector").len().sort("len"))


if __name__ == "__main__":
    main()
