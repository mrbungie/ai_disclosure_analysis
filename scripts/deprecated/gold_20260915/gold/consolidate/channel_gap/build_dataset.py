"""Materializes data/gold/datasets/firm_year/channel_gap.parquet: one row
per (ticker, fiscal_year) comparing call-vs-filing disclosure intensity --
not a covariate/target pair, it's the comparison table
channel_gap_analysis.py itself regresses `post` on, so both sides of that
comparison stay in one dataset rather than being split artificially.

Source: data/gold/covariates/firm_year/channel_gap_cells_extensive.parquet
(built with --period fy, scripts/gold/channel_gap/build_channel_gap_cells.py's
default).

Usage:
    uv run python scripts/gold/consolidate/channel_gap/build_dataset.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
SRC = REPO_ROOT / "data" / "gold" / "covariates" / "firm_year" / "channel_gap_cells_extensive.parquet"
OUT = REPO_ROOT / "data" / "gold" / "datasets" / "firm_year" / "channel_gap.parquet"


def main() -> None:
    df = pd.read_parquet(SRC)
    df = df.rename(columns={"t": "year"})
    df.insert(0, "id", df["ticker"] + "_" + df["year"].astype(int).astype(str))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f"-> {OUT} ({len(df):,} rows, {len(df.columns)} cols)")


if __name__ == "__main__":
    main()
