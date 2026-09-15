"""Materializes data/gold/covariates/firm_year/posture.parquet: the 8
disclosure-posture dimensions (Archetypal Analysis fits on exactly these,
see models/posture_archetype_static/), one row per (ticker, fiscal_year).

Kept as its OWN covariate family, orthogonal to disclosure_volume/
financial_ratios/market/firm_reference, so any dataset that needs posture
but not, say, market data can join just this file.

Source: data/processed/clusters/firm_year_master_v2.parquet.

Usage:
    uv run python scripts/gold/consolidate/posture/build_covariates_firm_year.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
SRC = REPO_ROOT / "data" / "processed" / "clusters" / "firm_year_master_v2.parquet"
OUT = REPO_ROOT / "data" / "gold" / "covariates" / "firm_year" / "posture.parquet"

COLUMNS = ["promotional_posture", "hedging_posture", "risk_orientation", "governance_orientation",
           "temporal_posture", "ai_positioning", "specificity", "disclosure_intensity"]


def main() -> None:
    df = pd.read_parquet(SRC)[["ticker", "year"] + COLUMNS].copy()
    df.insert(0, "id", df["ticker"] + "_" + df["year"].astype(int).astype(str))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f"-> {OUT} ({len(df):,} rows, {len(df.columns)} cols)")


if __name__ == "__main__":
    main()
