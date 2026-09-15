"""Materializes data/gold/covariates/firm_year/market.parquet: everything
scripts/gold/financials/build_market_factors.py produces that isn't
already claimed as a target (see build_targets_yearly.py in
scripts/gold/consolidate/firm_year/) -- valuation multiples and
pre-period momentum, one row per (ticker, fiscal_year).

Source: data/gold/covariates/firm_year/market_raw.parquet
(scripts/gold/financials/build_market_factors.py -- this family's own
source builder) for every value column. The (ticker, year) SPINE is
data/gold/spines/firm_year/firm_year.parquet -- same source as
consolidate/financial_ratios/build_covariates_firm_year.py, see that
script's docstring.

Usage:
    uv run python scripts/gold/consolidate/market/build_covariates_firm_year.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
SRC = REPO_ROOT / "data" / "gold" / "covariates" / "firm_year" / "market_raw.parquet"
SPINE = REPO_ROOT / "data" / "gold" / "spines" / "firm_year" / "firm_year.parquet"
OUT = REPO_ROOT / "data" / "gold" / "covariates" / "firm_year" / "market.parquet"

COLUMNS = ["market_cap", "pe_ratio", "pb_ratio", "ev_revenue", "ev_ebitda", "momentum_12_1"]


def main() -> None:
    spine = pd.read_parquet(SPINE, columns=["ticker", "year"]).drop_duplicates()
    values = pd.read_parquet(SRC)[["ticker", "year"] + COLUMNS].copy()
    df = spine.merge(values, on=["ticker", "year"], how="left")
    df.insert(0, "id", df["ticker"] + "_" + df["year"].astype(int).astype(str))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f"-> {OUT} ({len(df):,} rows, {len(df.columns)} cols)")


if __name__ == "__main__":
    main()
