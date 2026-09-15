"""Materializes data/gold/covariates/call/market.parquet: pre-call
market data (beta, price, market cap, shares out) -- the call-grain
sibling of market/build_covariates_firm_year.py, same source family
(price/factor windows, same estimator as build_market_factors.py, anchored
to the call date instead of the filing date).

Source: data/gold/spines/call/call.parquet.

Usage:
    uv run python scripts/gold/consolidate/market/build_covariates_call.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
SRC = REPO_ROOT / "data" / "gold" / "spines" / "call" / "call.parquet"
OUT = REPO_ROOT / "data" / "gold" / "covariates" / "call" / "market.parquet"

COLUMNS = ["beta_pre", "price_pre", "log_market_cap", "shares_out"]


def main() -> None:
    df = pd.read_parquet(SRC)[["call_accession_number"] + COLUMNS].copy()
    df.insert(0, "id", df["call_accession_number"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f"-> {OUT} ({len(df):,} rows, {len(df.columns)} cols)")


if __name__ == "__main__":
    main()
