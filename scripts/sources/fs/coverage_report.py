#!/usr/bin/env python
"""
scripts/sources/fs/coverage_report.py — before/after non-null coverage of
every column of covariates/targets {firm_year, firm_quarter, call} x
{financials, market}, comparing the archived pre-fs tables
(data/deprecated/fs_replacement_20260917T174800Z/gold/...) against the
current fs-backed ones (docs/plans/fs_gold_replacement.md).

Writes:
  data/results/fs_validation/gold_coverage_before_after.csv
    (grain, kind, family, column, rows_before, nonnull_before,
     nonnull_share_before, rows_after, nonnull_share_after)
  data/results/fs_validation/gold_coverage_by_year.csv
    (key columns only: revenue, total_assets, log_market_cap, beta/beta_pre,
     rd_expense, capex; by year 2021-2026, before vs after)

Read-only against data/deprecated and data/gold; writes only under
data/results/fs_validation/.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).parent.parent.parent.parent
BEFORE_ROOT = REPO_ROOT / "data" / "deprecated" / "fs_replacement_20260917T174800Z" / "gold"
AFTER_ROOT = REPO_ROOT / "data" / "gold"
OUT_DIR = REPO_ROOT / "data" / "results" / "fs_validation"

GRAINS_FAMILIES = [
    ("firm_year", "covariates", "financials"),
    ("firm_year", "covariates", "market"),
    ("firm_year", "targets", "financials"),
    ("firm_year", "targets", "market"),
    ("firm_quarter", "covariates", "financials"),
    ("firm_quarter", "covariates", "market"),
    ("firm_quarter", "targets", "financials"),
    ("firm_quarter", "targets", "market"),
    ("call", "covariates", "financials"),
    ("call", "covariates", "market"),
    ("call", "targets", "financials"),
    ("call", "targets", "market"),
]

DATE_COL = {"firm_year": "as_of_date", "firm_quarter": "as_of_date", "call": "fecha"}
KEY_COLS = {"firm_year": ["ticker", "year"], "firm_quarter": ["ticker", "quarter"], "call": ["ticker", "call_accession_number"]}
SPINE_COLS = {"firm_year": ["id", "ticker", "year", "as_of_date"],
             "firm_quarter": ["id", "ticker", "quarter", "as_of_date"],
             "call": ["id", "ticker", "call_accession_number", "fecha"]}

# key columns tracked by year for the by-year report (thesis field name ->
# (grain, kind, family, column) where it lives)
KEY_COLUMNS_BY_YEAR = [
    ("firm_year", "covariates", "financials", "revenue"),
    ("firm_year", "covariates", "financials", "rd_expense"),
    ("firm_year", "covariates", "financials", "capex"),
    ("firm_year", "covariates", "financials", "total_assets"),
    ("firm_year", "covariates", "market", "log_market_cap"),
    ("firm_year", "covariates", "market", "beta"),
    ("call", "covariates", "financials", "revenue"),
    ("call", "covariates", "financials", "rd_intensity_pre"),
    ("call", "covariates", "market", "log_market_cap"),
    ("call", "covariates", "market", "beta_pre"),
]


def load(root: Path, grain: str, kind: str, family: str) -> pl.DataFrame | None:
    p = root / kind / grain / f"{family}.parquet"
    if not p.exists():
        return None
    return pl.read_parquet(p)


def coverage_rows() -> list[dict]:
    rows = []
    for grain, kind, family, in [(g, k, f) for g, k, f in GRAINS_FAMILIES]:
        before = load(BEFORE_ROOT, grain, kind, family)
        after = load(AFTER_ROOT, grain, kind, family)
        if after is None:
            continue
        value_cols_after = [c for c in after.columns if c not in SPINE_COLS[grain]]
        n_after = after.height
        before_cols = set(before.columns) if before is not None else set()
        n_before = before.height if before is not None else 0
        for col in value_cols_after:
            share_after = float(after[col].is_not_null().mean()) if col in after.columns else None
            if before is not None and col in before_cols:
                share_before = float(before[col].is_not_null().mean())
            else:
                share_before = None
            rows.append({
                "grain": grain, "kind": kind, "family": family, "column": col,
                "rows_before": n_before, "nonnull_share_before": share_before,
                "rows_after": n_after, "nonnull_share_after": share_after,
                "delta": None if share_before is None else round(share_after - share_before, 4),
            })
        # columns that existed before but were dropped after
        for col in before_cols - set(value_cols_after) - set(SPINE_COLS[grain]):
            share_before = float(before[col].is_not_null().mean())
            rows.append({
                "grain": grain, "kind": kind, "family": family, "column": col,
                "rows_before": n_before, "nonnull_share_before": share_before,
                "rows_after": n_after, "nonnull_share_after": None, "delta": None,
            })
    return rows


def by_year_rows() -> list[dict]:
    rows = []
    for grain, kind, family, col in KEY_COLUMNS_BY_YEAR:
        before = load(BEFORE_ROOT, grain, kind, family)
        after = load(AFTER_ROOT, grain, kind, family)
        if after is None or col not in after.columns:
            continue
        date_col = DATE_COL[grain]
        after_y = after.with_columns(pl.col(date_col).dt.year().alias("year"))
        after_agg = (after_y.group_by("year").agg(pl.col(col).is_not_null().mean().alias("share"), pl.len().alias("n"))
                    .sort("year"))
        before_agg = None
        if before is not None and col in before.columns and date_col in before.columns:
            before_y = before.with_columns(pl.col(date_col).dt.year().alias("year"))
            before_agg = (before_y.group_by("year").agg(pl.col(col).is_not_null().mean().alias("share"), pl.len().alias("n"))
                         .sort("year"))
        years = range(2021, 2027)
        for y in years:
            a = after_agg.filter(pl.col("year") == y)
            b = before_agg.filter(pl.col("year") == y) if before_agg is not None else None
            rows.append({
                "grain": grain, "kind": kind, "family": family, "column": col, "year": y,
                "n_after": int(a["n"][0]) if a.height else 0,
                "share_after": round(float(a["share"][0]), 4) if a.height else None,
                "n_before": int(b["n"][0]) if b is not None and b.height else 0,
                "share_before": round(float(b["share"][0]), 4) if b is not None and b.height else None,
            })
    return rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cov = pl.DataFrame(coverage_rows())
    cov.write_csv(OUT_DIR / "gold_coverage_before_after.csv")
    print(f"Wrote {OUT_DIR / 'gold_coverage_before_after.csv'}: {cov.height} rows")

    by_year = pl.DataFrame(by_year_rows())
    by_year.write_csv(OUT_DIR / "gold_coverage_by_year.csv")
    print(f"Wrote {OUT_DIR / 'gold_coverage_by_year.csv'}: {by_year.height} rows")

    # sanity checks the orchestrator asked for
    print("\n=== sanity checks ===")
    fy_mkt = load(AFTER_ROOT, "firm_year", "covariates", "market")
    for t in ["NVDA", "AMZN", "GOOGL"]:
        rows = fy_mkt.filter(pl.col("ticker") == t).sort("year").select("year", "market_cap").to_dicts()
        print(f"{t} market_cap by year: {rows}")
    fy_fin = load(AFTER_ROOT, "firm_year", "covariates", "financials")
    aapl_2025 = fy_fin.filter((pl.col("ticker") == "AAPL") & (pl.col("year") == 2025)).select("revenue").to_dicts()
    print(f"AAPL FY2025-filed revenue: {aapl_2025} (known: ~416,161M)")
    rmd = fy_mkt.filter(pl.col("ticker") == "RMD").sort("year").select("year", "market_cap").to_dicts()
    print(f"RMD market_cap by year: {rmd}")


if __name__ == "__main__":
    main()
