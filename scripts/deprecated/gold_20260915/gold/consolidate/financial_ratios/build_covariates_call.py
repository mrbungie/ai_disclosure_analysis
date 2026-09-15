"""Materializes data/gold/covariates/call/financial_ratios.parquet:
the most-recent-known-before-this-call accounting fundamentals (from
scripts/gold/financials/build_firm_financials.py, attached by merge_asof
-- same source as financial_ratios/build_covariates_firm_year.py, different
grain and anchor date), plus firm reference fields (sic, sic2, fe).

Source: data/gold/spines/call/call.parquet.

Usage:
    uv run python scripts/gold/consolidate/financial_ratios/build_covariates_call.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
SRC = REPO_ROOT / "data" / "gold" / "spines" / "call" / "call.parquet"
OUT = REPO_ROOT / "data" / "gold" / "covariates" / "call" / "financial_ratios.parquet"

COLUMNS = ["revenue", "operating_income", "total_assets", "operating_margin", "asset_turnover", "roa",
           "sic", "sic2", "fe", "filing_date_pt"]


def main() -> None:
    df = pd.read_parquet(SRC)[["call_accession_number"] + COLUMNS].copy()
    df.insert(0, "id", df["call_accession_number"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f"-> {OUT} ({len(df):,} rows, {len(df.columns)} cols)")


if __name__ == "__main__":
    main()
