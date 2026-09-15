"""Materializes data/gold/covariates/firm_year/financial_ratios.parquet:
everything scripts/gold/financials/build_firm_financials.py produces --
margins, returns, liquidity/leverage/turnover ratios, plus the firm
reference fields (sic2, in_text_panel) that come out of the same build --
one row per (ticker, fiscal_year). Orthogonal to market/ (valuation +
market-priced data, a separate source: build_market_factors.py).

Also carries `has_10k` (whether this ticker-fiscal_year had its own 10-K
filed, from `silver.filing_manifest`) -- a silver-layer lookup, which is why it's
read here rather than by a predictions/ script downstream: covariate/
target builders are the only things allowed to read below data/gold/, see
docs/gold_pipeline.md.

Source: data/gold/covariates/firm_year/financial_ratios_raw.parquet
(scripts/gold/financials/build_firm_financials.py -- this family's own
source builder) for every value column; silver.filing_manifest (for
has_10k only). The (ticker, year) SPINE is
data/gold/spines/firm_year/firm_year.parquet: it carries 49 ticker-years
with disclosure/text-panel membership but no filed 10-K (so no financials
row), which financial_ratios_raw.parquet alone does not have keys for.

Usage:
    uv run python scripts/gold/consolidate/financial_ratios/build_covariates_firm_year.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "common"))
import layers as L  # noqa: E402

SRC = REPO_ROOT / "data" / "gold" / "covariates" / "firm_year" / "financial_ratios_raw.parquet"
SPINE = REPO_ROOT / "data" / "gold" / "spines" / "firm_year" / "firm_year.parquet"
OUT = REPO_ROOT / "data" / "gold" / "covariates" / "firm_year" / "financial_ratios.parquet"

COLUMNS = ["gross_margin", "operating_margin", "net_margin", "roa", "roe",
           "current_ratio", "debt_to_equity", "asset_turnover", "capex_intensity", "sic2"]


def load_has_10k() -> pd.DataFrame:
    return (L.scan("silver.filing_manifest")
            .filter((pl.col("country_code") == "us") & (pl.col("form_type") == "10-K")
                    & pl.col("filing_date").is_not_null())
            .select("ticker", pl.col("filing_date").dt.year().cast(pl.Int32).alias("year"),
                    pl.lit(True).alias("has_10k"))
            .unique(maintain_order=True)
            .collect().to_pandas())


def main() -> None:
    spine = pd.read_parquet(SPINE, columns=["ticker", "year"]).drop_duplicates()
    values = pd.read_parquet(SRC)[["ticker", "year"] + COLUMNS].copy()
    df = spine.merge(values, on=["ticker", "year"], how="left")
    # Every row from build_firm_financials.py has a filed 10-K by construction
    # (it aligns facts to filed 10-Ks); the spine's other ticker-years (text
    # panel membership without a filed 10-K) get NaN ratio values here, same
    # as the old firm_year_master_v2-sourced output.
    df["in_text_panel"] = True
    df = df.merge(load_has_10k(), on=["ticker", "year"], how="left")
    df["has_10k"] = df["has_10k"].fillna(False)
    df.insert(0, "id", df["ticker"] + "_" + df["year"].astype(int).astype(str))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f"-> {OUT} ({len(df):,} rows, {len(df.columns)} cols)")


if __name__ == "__main__":
    main()
