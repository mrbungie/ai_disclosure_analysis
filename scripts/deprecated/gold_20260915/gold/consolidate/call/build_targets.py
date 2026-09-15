"""Materializes data/gold/targets/call/call.parquet: the post-call
outcomes at the (ticker, call) grain -- market beta re-estimated over
three forward windows, and the 60-trading-day forward return.

Column names already carry their window (source names unchanged, this
script only selects and adds the canonical `id`):
  beta_post_63d   forward beta, 63 trading days (~1 quarter) post-call
  beta_post_126d  forward beta, 126 trading days (~2 quarters) post-call
  beta_post_252d  forward beta, 252 trading days (~1 year) post-call
  return_post_60d forward raw return, 60 trading days post-call

Source: data/gold/spines/call/call.parquet.

Usage:
    uv run python scripts/gold/consolidate/call/build_targets.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
SRC = REPO_ROOT / "data" / "gold" / "spines" / "call" / "call.parquet"
OUT = REPO_ROOT / "data" / "gold" / "targets" / "call" / "call.parquet"

RENAME = {
    "beta_post_63": "beta_post_63d",
    "beta_post_126": "beta_post_126d",
    "beta_post_252": "beta_post_252d",
    "return60": "return_post_60d",
}


def main() -> None:
    df = pd.read_parquet(SRC)
    out = df[["call_accession_number"] + list(RENAME.keys())].rename(columns=RENAME).copy()
    out.insert(0, "id", out["call_accession_number"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    print(f"-> {OUT} ({len(out):,} rows, {len(out.columns)} cols)")


if __name__ == "__main__":
    main()
