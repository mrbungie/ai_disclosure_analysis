"""Materializes data/gold/covariates/firm_year/activities.parquet: disclosed-
activity counts and rates by family (deployment, infrastructure, named
provider, ...), one row per (ticker, fiscal_year) -- the disclosed-ACTION
signal, orthogonal to disclosure_volume (which measures how much a firm
talks, not what it says it does) and posture (how it talks). Source:
scripts/gold/posture/activity_profiles.py's output.

Source: data/processed/clusters/firm_year_activities.parquet.

Usage:
    uv run python scripts/gold/consolidate/activities/build_covariates_firm_year.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
SRC = REPO_ROOT / "data" / "processed" / "clusters" / "firm_year_activities.parquet"
OUT = REPO_ROOT / "data" / "gold" / "covariates" / "firm_year" / "activities.parquet"


def main() -> None:
    df = pd.read_parquet(SRC)
    df.insert(0, "id", df["ticker"] + "_" + df["year"].astype(int).astype(str))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f"-> {OUT} ({len(df):,} rows, {len(df.columns)} cols)")


if __name__ == "__main__":
    main()
