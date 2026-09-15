"""Disclosed AI activity schema field distributions (thesis.qmd
`tbl-activity-schema`, lines ~691-736).

The qmd chunk read the legacy `firm_activities.parquet` artifact (under the
now-deprecated processed-clusters tree); the correct source now is `silver.ai_activities` filtered to `has_activity`
(silver.ai_activities also carries `has_activity = false` rows, 5,311 of
63,032, which the qmd's data never included), deduplicated on
`(text_hash, frame_id, activity_id)` the same way frame_schema_distribution
dedups frames.

Reproduces action, maturity stage, technology sourcing, target beneficiary,
and evidence-attribute distributions, plus the named-entity citation count
-- all direct columns on `silver.ai_activities` (action, stage, source,
target, evidence_type, entities).

One column from the old table has no equivalent here: `function_family`,
the qmd's 6-value "Business Domain" taxonomy (Operate, Sell, Build, Control,
Serve, Enable). That taxonomy was a regex-based derivation
(`FUNCTION_FAMILIES` in `scripts/gold/activity/build_activity.py`,
16 fine-grained categories, not 6) applied to the free-text `function`
column -- a gold-layer business rule, out of scope for a bronze/silver
descriptive-stats script. `silver.ai_activities` instead carries `domain`,
a 3-value schema (customer_facing / internal / unspecified) already present
on the table; this script reports that field under `domain` instead of
reproducing the superseded 6-domain scheme, for a later migration pass to
reconcile against the current prose.

Usage:
    .venv/bin/python scripts/analytics/corpus/activity_schema_distribution.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import layers as L  # noqa: E402

FIELDS = ["action", "stage", "source", "target", "evidence_type", "domain"]


def main() -> None:
    a = L.scan("silver.ai_activities").filter(
        (pl.col("country_code") == "us") & pl.col("has_activity")
    ).collect()
    dedup = a.unique(subset=["text_hash", "frame_id", "activity_id"])
    tot = dedup.height

    rows: list[tuple[str, str, int, float]] = []
    for col in FIELDS:
        for value, n in dedup.group_by(col).len().sort("len", descending=True).iter_rows():
            rows.append((col, value, n, round(n / tot * 100, 1)))

    n_entities = int(dedup["entities"].list.len().sum())
    rows.append(("named_entities", "entity_citations", n_entities, round(n_entities / tot * 100, 1)))

    out = pl.DataFrame(rows, schema=["field", "value", "n", "pct"], orient="row")
    out_path = L.results_path("corpus", "activity_schema_distribution.parquet")
    out.write_parquet(out_path)
    print(f"n_act (dedup activities) = {tot:,} -> {out_path}")
    print(out)


if __name__ == "__main__":
    main()
