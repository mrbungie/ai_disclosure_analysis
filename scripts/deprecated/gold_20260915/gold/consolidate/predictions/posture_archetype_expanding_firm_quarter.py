"""Materializes data/gold/predictions/firm_quarter/posture_archetype_expanding.parquet
from data/gold/covariates/firm_quarter/archetype_weights_expanding.parquet.

**Not a separate "load model + apply" step, unlike posture_archetype_static.py**:
scripts/gold/posture/build_archetype_weights_quarterly_asof.py refits a FRESH
AA model at every quarterly cutoff on that cutoff's data only, then
immediately transforms that SAME cutoff's cross-section with it -- fit and
apply happen in the same walk-forward pass, by construction (there is no
later population to apply an already-fit cutoff's model to; next quarter
gets its own new fit). Each cutoff's model is still persisted to
models/posture_archetype_expanding/cutoff=<YYYYQN>/model.pkl for inspection
and reproducibility, even though this script only relabels the already-
computed weights rather than re-deriving them.

**Point-in-time safe** (unlike posture_archetype_static.parquet): each
row's weights use only frames known as of that cutoff quarter or before.

Usage:
    uv run python scripts/gold/consolidate/predictions/posture_archetype_expanding_firm_quarter.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

import sys
REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "common"))
import layers as L  # noqa: E402

SRC = L.gold_path("covariates", "firm_quarter", "archetype_weights_expanding")
OUT = REPO_ROOT / "data" / "gold" / "predictions" / "firm_quarter" / "posture_archetype_expanding.parquet"


def main() -> None:
    df = pd.read_parquet(SRC)
    df.insert(0, "id", df["ticker"] + "_" + df["cutoff_quarter"].astype(str))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f"-> {OUT} ({len(df):,} rows, {len(df.columns)} cols)")


if __name__ == "__main__":
    main()
