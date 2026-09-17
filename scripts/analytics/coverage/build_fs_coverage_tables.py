#!/usr/bin/env python
"""scripts/analytics/coverage/build_fs_coverage_tables.py -- non-null coverage
of the fs-backed gold panel (docs/sources/fs.md), read straight from
data/gold (never re-derived), for Appendix A.5 of the thesis.

Four cuts, one CSV each, so the qmd chunk stays a pure `pd.read_csv` reader:

  data/results/coverage/coverage_by_grain_variable.csv
    grain, variable, family (market|financial), n_total, n_nonnull, pct_nonnull
    -- one row per tracked variable at each of firm_year/firm_quarter/call.

  data/results/coverage/coverage_by_year.csv
    grain, variable, year, n_total, n_nonnull, pct_nonnull
    -- the same variables, split by year 2021-2026 (year of the grain's own
    as_of_date/fecha).

  data/results/coverage/pit_source_shares.csv
    grain, year, pit_source (edgar|fs_release|null), n, pct
    -- measured from silver.fs_financials, restricted to the S&P 500
    2021-01-01 universe and period_end >= 2020-11-01 (the manifests' own
    coverage window, per docs/sources/fs.md), annual rows reported under
    "firm_year" and quarterly rows under "firm_quarter" (the call grain's
    pre-call fundamentals are matched from the same annual/quarterly join,
    so it is not tracked separately here).

  data/results/coverage/revenue_basis_shares.csv
    grain, year, revenue_basis (sales|bank_nii_plus_nonii|null), n, pct
    -- same restriction and grain mapping as pit_source_shares.

Usage:
  .venv/bin/python scripts/analytics/coverage/build_fs_coverage_tables.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "common"))
import layers as L  # noqa: E402

OUT_DIR = L.results_path("coverage", "coverage_by_grain_variable.csv").parent

YEARS = [2021, 2022, 2023, 2024, 2025, 2026]

# (grain, family, variable, kind) -- kind is "market" or "financial", used to
# split @tbl-a5-* into two blocks per grain. Only variables actually present
# in that grain's gold family are read; a variable absent for a grain is
# skipped rather than padded with a fabricated zero row.
VARS = [
    # firm_year
    ("firm_year", "covariates", "market", "beta", "market"),
    ("firm_year", "covariates", "market", "idio_vol_252d", "market"),
    ("firm_year", "covariates", "market", "momentum_12_1", "market"),
    ("firm_year", "covariates", "market", "ps_ratio", "market"),
    ("firm_year", "covariates", "market", "market_cap", "market"),
    ("firm_year", "covariates", "financials", "revenue", "financial"),
    ("firm_year", "covariates", "financials", "net_margin", "financial"),
    ("firm_year", "covariates", "financials", "capex", "financial"),
    ("firm_year", "covariates", "financials", "sga_expense", "financial"),
    ("firm_year", "covariates", "financials", "rd_expense", "financial"),
    ("firm_year", "covariates", "financials", "cash", "financial"),
    ("firm_year", "covariates", "financials", "interest_expense", "financial"),
    ("firm_year", "covariates", "financials", "roic_minus_wacc", "financial"),
    ("firm_year", "covariates", "financials", "revenue_yoy", "financial"),
    # firm_quarter
    ("firm_quarter", "covariates", "market", "beta_pre_252", "market"),
    ("firm_quarter", "covariates", "market", "idio_vol_pre_252", "market"),
    ("firm_quarter", "covariates", "market", "ps_ratio", "market"),
    ("firm_quarter", "covariates", "market", "log_market_cap", "market"),
    ("firm_quarter", "targets", "market", "beta_post_63", "market"),
    ("firm_quarter", "covariates", "financials", "revenue", "financial"),
    ("firm_quarter", "covariates", "financials", "sga_expense", "financial"),
    ("firm_quarter", "covariates", "financials", "rd_expense", "financial"),
    ("firm_quarter", "covariates", "financials", "capex", "financial"),
    ("firm_quarter", "targets", "financials", "next_revenue_yoy", "financial"),
    # call
    ("call", "covariates", "market", "beta_pre", "market"),
    ("call", "covariates", "market", "beta_shift_pre", "market"),
    ("call", "covariates", "market", "ncskew_wk_pre", "market"),
    ("call", "covariates", "market", "duvol_wk_pre", "market"),
    ("call", "covariates", "market", "log_market_cap", "market"),
    ("call", "covariates", "market", "ps_ratio_pre", "market"),
    ("call", "targets", "market", "beta_shift_delta", "market"),
    ("call", "targets", "market", "car_m1_p1", "market"),
    ("call", "targets", "market", "car_p2_p63", "market"),
    ("call", "targets", "market", "ncskew_wk_post", "market"),
    ("call", "targets", "market", "duvol_wk_post", "market"),
    ("call", "covariates", "financials", "revenue", "financial"),
    ("call", "covariates", "financials", "roa", "financial"),
    ("call", "covariates", "financials", "gross_margin_ttm_pre", "financial"),
    ("call", "covariates", "financials", "roic_minus_wacc_ttm_pre", "financial"),
    ("call", "covariates", "financials", "revenue_growth_ttm_pre", "financial"),
    ("call", "targets", "financials", "gross_margin_ttm_post", "financial"),
    ("call", "targets", "financials", "roic_minus_wacc_ttm_post", "financial"),
    ("call", "targets", "financials", "revenue_growth_ttm_post", "financial"),
]

DATE_COL = {"firm_year": "as_of_date", "firm_quarter": "as_of_date", "call": "fecha"}


def _cached_gold(cache: dict, grain: str, kind: str, family: str) -> pd.DataFrame:
    key = (grain, kind, family)
    if key not in cache:
        cache[key] = L.read_gold(grain, (kind, family))
    return cache[key]


def coverage_rows() -> tuple[list[dict], list[dict]]:
    by_grain_variable: list[dict] = []
    by_year: list[dict] = []
    cache: dict = {}
    for grain, kind, family, col, fam_kind in VARS:
        df = _cached_gold(cache, grain, kind, family)
        if col not in df.columns:
            continue
        n_total = len(df)
        nonnull = df[col].notna()
        n_nonnull = int(nonnull.sum())
        by_grain_variable.append({
            "grain": grain, "variable": col, "family": fam_kind,
            "n_total": n_total, "n_nonnull": n_nonnull,
            "pct_nonnull": round(100 * n_nonnull / n_total, 2) if n_total else 0.0,
        })
        date_col = DATE_COL[grain]
        years = pd.to_datetime(df[date_col]).dt.year
        for y in YEARS:
            mask = years == y
            n_y = int(mask.sum())
            if n_y == 0:
                continue
            n_nn_y = int((nonnull & mask).sum())
            by_year.append({
                "grain": grain, "variable": col, "year": y,
                "n_total": n_y, "n_nonnull": n_nn_y,
                "pct_nonnull": round(100 * n_nn_y / n_y, 2),
            })
    return by_grain_variable, by_year


def _pit_and_basis_rows() -> tuple[list[dict], list[dict]]:
    universe = pl.read_parquet(REPO_ROOT / "data" / "silver" / "firm_universe.parquet")
    tickers = set(universe.filter(pl.col("country_code") == "us")["ticker"].to_list())

    fin = pl.read_parquet(REPO_ROOT / "data" / "silver" / "fs_financials.parquet")
    fin = fin.filter(
        pl.col("ticker").is_in(tickers)
        & (pl.col("period_end") >= pl.date(2020, 11, 1))
    )
    fin = fin.with_columns(pl.col("period_end").dt.year().alias("year"))
    grain_of = {"annual": "firm_year", "quarterly": "firm_quarter"}

    pit_rows: list[dict] = []
    basis_rows: list[dict] = []
    for period, grain in grain_of.items():
        sub = fin.filter(pl.col("period") == period)
        for y in YEARS:
            sub_y = sub.filter(pl.col("year") == y)
            n_y = sub_y.height
            if n_y == 0:
                continue
            pit_counts = sub_y.with_columns(pl.col("pit_source").fill_null("null")) \
                .group_by("pit_source").len().to_dicts()
            for row in pit_counts:
                pit_rows.append({
                    "grain": grain, "year": y, "pit_source": row["pit_source"],
                    "n": row["len"], "pct": round(100 * row["len"] / n_y, 2),
                })
            basis_counts = sub_y.with_columns(pl.col("revenue_basis").fill_null("null")) \
                .group_by("revenue_basis").len().to_dicts()
            for row in basis_counts:
                basis_rows.append({
                    "grain": grain, "year": y, "revenue_basis": row["revenue_basis"],
                    "n": row["len"], "pct": round(100 * row["len"] / n_y, 2),
                })
    return pit_rows, basis_rows


def main() -> None:
    by_grain_variable, by_year = coverage_rows()
    pd.DataFrame(by_grain_variable).to_csv(OUT_DIR / "coverage_by_grain_variable.csv", index=False)
    pd.DataFrame(by_year).to_csv(OUT_DIR / "coverage_by_year.csv", index=False)

    pit_rows, basis_rows = _pit_and_basis_rows()
    pd.DataFrame(pit_rows).to_csv(OUT_DIR / "pit_source_shares.csv", index=False)
    pd.DataFrame(basis_rows).to_csv(OUT_DIR / "revenue_basis_shares.csv", index=False)

    print(f"wrote {len(by_grain_variable)} grain/variable rows, {len(by_year)} by-year rows, "
          f"{len(pit_rows)} pit_source rows, {len(basis_rows)} revenue_basis rows to {OUT_DIR}")


if __name__ == "__main__":
    main()
