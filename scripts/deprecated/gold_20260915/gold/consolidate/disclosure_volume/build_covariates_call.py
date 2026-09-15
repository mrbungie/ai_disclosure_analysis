"""Materializes data/gold/covariates/call/disclosure.parquet:
call-level AI disclosure/grounding/substance and their expanding
same-ticker history, one row per (ticker, call) -- the call-grain sibling
of disclosure_volume/build_covariates_firm_year.py, same source family
(ai_intensity.document_table() + activity_profiles.flags(), assembled by
scripts/gold/call_beta/build_call_beta_panel.py), different grain and
column set. `flags()` (used by build_call_beta_panel.py to compute
grounding/substance) now lives in scripts/gold/posture/build_firm_activities.py.

Source: data/gold/spines/call/call.parquet (the call_beta group's spine,
scripts/gold/call_beta/build_call_beta_panel.py).

Usage:
    uv run python scripts/gold/consolidate/disclosure_volume/build_covariates_call.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
SRC = REPO_ROOT / "data" / "gold" / "spines" / "call" / "call.parquet"
OUT = REPO_ROOT / "data" / "gold" / "covariates" / "call" / "disclosure.parquet"

COLUMNS = ["n_words", "n_frames", "disclosure", "n_activities", "grounding", "substance",
           "n_prior_calls", "hist_disclosure", "surprise_disclosure", "hist_substance", "surprise_substance"]


def main() -> None:
    df = pd.read_parquet(SRC)[["call_accession_number", "ticker", "fecha"] + COLUMNS].copy()
    df.insert(0, "id", df["call_accession_number"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f"-> {OUT} ({len(df):,} rows, {len(df.columns)} cols)")


if __name__ == "__main__":
    main()
