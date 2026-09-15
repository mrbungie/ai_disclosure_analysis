"""Materializes data/gold/targets/firm_year/firm_year.parquet: every outcome
variable regressed on disclosure at the (ticker, fiscal_year) grain --
the RQ4 headline outcomes (`incremental_signal.py::OUTCOMES`/
`EXTRA_OUTCOMES`) plus the forward-looking growth/event-study targets
build_market_factors.py/build_firm_financials.py compute alongside them.

Column names carry their measurement window explicitly so a reader never
has to open the source script to know what "beta" means:
  beta_252d              pre-filing 252-trading-day market beta
  idio_vol_252d          already windowed (source name unchanged)
  vol_pre_60d            already windowed (source name unchanged, robustness outcome)
  vol_post_60d           already windowed (source name unchanged)
  car_m1_p5d             cumulative abnormal return, trading-day window [-1,+5] around filing
  price_to_sales_t       contemporaneous (as-of period t, not a lead/lag)
  rd_intensity_t         contemporaneous
  revenue_growth_lead1y  fiscal_year -> fiscal_year+1 YoY growth
  rd_expense_growth_lead1y
  capex_growth_lead1y
  sga_expense_growth_lead1y

Source: data/gold/spines/firm_year/firm_year.parquet for the (ticker, year)
grain; data/gold/covariates/firm_year/market_raw.parquet (beta, valuation,
windowed vol, filing-window CAR) and financial_ratios_raw.parquet
(rd_intensity, next-year growth) for values -- each family's own source
builder, one hop before the consolidated financial_ratios.py/market.py
covariate files.

Usage:
    uv run python scripts/gold/consolidate/firm_year/build_targets.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
SPINE = REPO_ROOT / "data" / "gold" / "spines" / "firm_year" / "firm_year.parquet"
MARKET_RAW = REPO_ROOT / "data" / "gold" / "covariates" / "firm_year" / "market_raw.parquet"
RATIOS_RAW = REPO_ROOT / "data" / "gold" / "covariates" / "firm_year" / "financial_ratios_raw.parquet"
OUT = REPO_ROOT / "data" / "gold" / "targets" / "firm_year" / "firm_year.parquet"

MARKET_RENAME = {
    "beta": "beta_252d",
    "ps_ratio": "price_to_sales_t",
    "car_m1_p5": "car_m1_p5d",
    # idio_vol_252d, vol_pre_60d, vol_post_60d already windowed -- unchanged.
}
RATIOS_RENAME = {
    "rd_intensity": "rd_intensity_t",
    "next_revenue_yoy": "revenue_growth_lead1y",
    "next_rd_expense_yoy": "rd_expense_growth_lead1y",
    "next_capex_yoy": "capex_growth_lead1y",
    "next_sga_expense_yoy": "sga_expense_growth_lead1y",
}


def main() -> None:
    spine = pd.read_parquet(SPINE, columns=["id", "ticker", "year"])
    market = pd.read_parquet(MARKET_RAW, columns=["ticker", "year"] + list(MARKET_RENAME) +
                              ["idio_vol_252d", "vol_pre_60d", "vol_post_60d"])
    ratios = pd.read_parquet(RATIOS_RAW, columns=["ticker", "year"] + list(RATIOS_RENAME))

    out = (spine
           .merge(market.rename(columns=MARKET_RENAME), on=["ticker", "year"], how="left")
           .merge(ratios.rename(columns=RATIOS_RENAME), on=["ticker", "year"], how="left"))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    print(f"-> {OUT} ({len(out):,} rows, {len(out.columns)} cols)")


if __name__ == "__main__":
    main()
