"""Materializes data/gold/predictions/firm_year/posture_archetype_expanding.parquet
from panel_expanding_archetypes.parquet.

Same walk-forward design as posture_archetype_expanding_firm_quarter.py, but
yearly cutoffs -- see that script's docstring for why fit and apply happen
in one pass rather than as separate steps here. Each cutoff's model is
still persisted to
models/posture_archetype_expanding_yearly/cutoff=<year>/model.pkl.

**Point-in-time safe** (unlike posture_archetype_static.parquet): each
firm-year's label uses only frames known as of that fiscal year or
before. Used as the archetype predictor in the crash-risk regressions
(scripts/analytics/crash_archetypes/call_archetype_full_battery.py),
where a future-informed label would make the outcome circular.

Usage:
    uv run python scripts/gold/consolidate/predictions/posture_archetype_expanding_firm_year.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
SRC = REPO_ROOT / "data" / "processed" / "clusters" / "panel_expanding_archetypes.parquet"
OUT = REPO_ROOT / "data" / "gold" / "predictions" / "firm_year" / "posture_archetype_expanding.parquet"


def main() -> None:
    df = pd.read_parquet(SRC)
    df.insert(0, "id", df["ticker"] + "_" + df["year"].astype(int).astype(str))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f"-> {OUT} ({len(df):,} rows, {len(df.columns)} cols)")


if __name__ == "__main__":
    main()
