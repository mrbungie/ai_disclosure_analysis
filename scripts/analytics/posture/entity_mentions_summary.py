"""
scripts/analytics/posture/entity_mentions_summary.py — geography/modality
distribution of named AI entity mentions, moved out of
`scripts/enrichment/ai_entity_mentions.py` (E-M5): the manifest there now
only records run_id, term count, output path and row count; this script
does the actual aggregation.

Reads `silver.ai_entity_mentions` (one row per paragraph INSTANCE x matched
term) and counts UNIQUE TEXTS per geography/modality — same "dedup as a
phase" unit ai_entity_mentions.py always used, not paragraph instances.

Output:
    data/results/posture/entity_mentions_by_geo.json
    data/results/posture/entity_mentions_by_modality.json

Usage:
    uv run python scripts/analytics/posture/entity_mentions_summary.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "common"))
import layers as L  # noqa: E402


def unique_text_counts(column: str) -> dict:
    df = (L.scan("silver.ai_entity_mentions")
          .group_by(column).agg(pl.col("text_hash").n_unique().alias("unique_texts"))
          .sort("unique_texts", descending=True).collect())
    return {row[column]: int(row["unique_texts"]) for row in df.to_dicts()}


def main() -> None:
    by_geo = unique_text_counts("geo")
    by_modality = unique_text_counts("modality")

    for name, counts in (("entity_mentions_by_geo.json", by_geo),
                         ("entity_mentions_by_modality.json", by_modality)):
        out_path = L.results_path("posture", name)
        out_path.write_text(json.dumps(
            {"counts": counts, "created_at": datetime.now(timezone.utc).isoformat()},
            indent=2, sort_keys=True) + "\n")
        print(f"-> {out_path}")
        for key, count in counts.items():
            print(f"  {key}: {count:,}")


if __name__ == "__main__":
    main()
