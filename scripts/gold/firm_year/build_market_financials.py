"""Firm-year accounting and market families, anchored at the 10-K filed in
the calendar year.

  covariates/firm_year/financials  the fiscal year the 10-K discloses
      (build_firm_financials.annual_panel: filing_date, accession_number,
      disclosed_period_end, income statement, balance sheet, cover-page
      shares, ratios, revenue_yoy (the disclosed fiscal year over the prior
      one), sic2), has_10k, and ROIC vs. WACC with its components
      (build_roic_wacc.value_creation)
  covariates/firm_year/market      valuation multiples at the last close before
      the filing, pre-filing beta (252 trading days, one factor), idiosyncratic
      and 60-day volatility, momentum 12-1 (build_market_factors)
  targets/firm_year/financials     the next fiscal year: next_period_end,
      next_gap_days, next_*_yoy growth
  targets/firm_year/market         volatility over the 60 trading days from the
      filing and raw return over [-1, +5]

Firm-years without a filed 10-K (or without prices) have null values;
`has_10k` is false there.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "gold" / "financials"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "common"))
from build_firm_financials import annual_panel  # noqa: E402
from build_market_factors import filing_market_panel  # noqa: E402
from build_roic_wacc import value_creation  # noqa: E402

import layers as L  # noqa: E402

BUILDER = "scripts/gold/firm_year/build_market_financials.py"
FINANCIALS = ["filing_date", "accession_number", "disclosed_period_end", "revenue", "cost_of_revenue", "rd_expense",
              "sga_expense", "capex", "operating_income", "net_income", "da", "interest_expense", "pretax_income",
              "tax_expense", "eps_diluted", "total_assets", "equity", "current_assets", "current_liabilities",
              "long_term_debt", "cash", "shares_out", "gross_margin", "operating_margin", "net_margin", "roa", "roe",
              "current_ratio", "debt_to_equity", "asset_turnover", "rd_intensity", "capex_intensity", "ebitda", "sic2",
              "revenue_yoy"]
NEXT_YEAR = ["next_period_end", "next_gap_days", "next_revenue_yoy", "next_rd_expense_yoy", "next_capex_yoy",
             "next_sga_expense_yoy"]
MARKET = ["market_cap", "pe_ratio", "ps_ratio", "pb_ratio", "ev_revenue", "ev_ebitda", "beta", "beta_n_obs",
          "idio_vol_252d", "vol_pre_60d", "momentum_12_1"]
POST_FILING = ["vol_post_60d", "ret_m1_p5"]


def load_10k_years() -> pd.DataFrame:
    """(ticker, calendar filing year) with at least one filed 10-K."""
    return (L.scan("silver.filing_manifest")
            .filter((pl.col("country_code") == "us") & (pl.col("form_type") == "10-K")
                    & pl.col("filing_date").is_not_null())
            .select("ticker", pl.col("filing_date").dt.year().cast(pl.Int32).alias("year"))
            .unique(maintain_order=True)
            .collect().to_pandas())


def main() -> None:
    spine = L.read_gold("firm_year")
    keys = spine[["ticker", "year"]]
    annual = keys.merge(annual_panel(), on=["ticker", "year"], how="inner")
    market = filing_market_panel(annual)
    value = value_creation(annual, market)

    financials = (spine.merge(annual[["ticker", "year"] + FINANCIALS], on=["ticker", "year"], how="left", validate="one_to_one")
                  .merge(load_10k_years().assign(has_10k=True), on=["ticker", "year"], how="left")
                  .merge(value, on=["ticker", "year"], how="left", validate="one_to_one"))
    financials["has_10k"] = financials["has_10k"].fillna(False).astype(bool)
    L.write_gold("covariates", "firm_year", "financials", financials, builder=BUILDER)
    L.write_gold("targets", "firm_year", "financials",
                 spine.merge(annual[["ticker", "year"] + NEXT_YEAR], on=["ticker", "year"], how="left"), builder=BUILDER)
    market = spine.merge(market, on=["ticker", "year"], how="left", validate="one_to_one")
    L.write_gold("covariates", "firm_year", "market", market[list(spine.columns) + MARKET], builder=BUILDER)
    L.write_gold("targets", "firm_year", "market", market[list(spine.columns) + POST_FILING], builder=BUILDER)


if __name__ == "__main__":
    main()
