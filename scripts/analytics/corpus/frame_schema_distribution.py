"""Semantic frame schema field distributions (thesis.qmd `tbl-frame-schema`,
lines ~627-684).

The qmd chunk built a temp table from the old `gold_ai_frames` DuckDB view
(`WHERE country_code='us' AND has_frame AND {PANEL_DOCS}`); the equivalent
source is `silver.ai_frames` filtered to `has_frame` (silver.ai_frames also
carries `has_frame = false` rows, 6,631 of 92,866, which the qmd's SQL
excludes -- PANEL_DOCS itself is already implied by silver's analysis-universe
scoping, so no separate join is needed). Frames are deduplicated on
`(text_hash, frame_id)` first, matching the qmd's own dedup comment (a
paragraph broadcast to more than one accession_number would otherwise be
double-counted).

Reproduces the single-label fields (subject, ai_type, temporal: exhaustive,
percentages sum to 100 within each field) and the multi-label fields
(specificity, rhetoric: tag prevalence, percentages need not sum to 100)
plus the sentence-anchor summary (mean sentences per claim, % anchored).
The "Functional Concepts" row in the qmd table (Adoption/deploy 46.1%,
Governance/control 18.2%, Risk 22.4%) is a hardcoded string in the qmd, not
computed by any SQL there; this script instead reports the raw multi-label
`concepts` tag prevalence (the field those three buckets were manually
rolled up from), which a later migration pass can re-bucket if needed.

Usage:
    .venv/bin/python scripts/analytics/corpus/frame_schema_distribution.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import layers as L  # noqa: E402

SINGLE_LABEL_FIELDS = ["subject", "ai_type", "temporal"]
MULTI_LABEL_FIELDS = ["concepts", "specificity", "rhetoric"]


def main() -> None:
    f = L.scan("silver.ai_frames").filter(
        (pl.col("country_code") == "us") & pl.col("has_frame")
    ).collect()

    dedup = f.select(
        ["text_hash", "frame_id", "subject", "ai_type", "temporal",
         "concepts", "specificity", "rhetoric", "sentence_ids"]
    ).unique(subset=["text_hash", "frame_id"])
    tot = dedup.height

    rows: list[tuple[str, str, int, float]] = []
    for col in SINGLE_LABEL_FIELDS:
        for value, n in dedup.group_by(col).len().sort("len", descending=True).iter_rows():
            rows.append((col, value, n, round(n / tot * 100, 1)))

    for col in MULTI_LABEL_FIELDS:
        values = dedup.select(col).explode(col).drop_nulls()[col].unique(maintain_order=True).sort().to_list()
        for v in values:
            n = int(dedup[col].list.contains(v).sum())
            rows.append((col, v, n, round(n / tot * 100, 1)))

    sent_len = dedup["sentence_ids"].list.len()
    rows.append(("sentence_anchors", "mean_sentences_per_claim", tot, round(float(sent_len.mean()), 2)))
    rows.append(("sentence_anchors", "pct_anchored", tot, round(float((sent_len > 0).sum() / tot * 100), 1)))

    out = pl.DataFrame(rows, schema=["field", "value", "n", "pct"], orient="row")
    out_path = L.results_path("corpus", "frame_schema_distribution.parquet")
    out.write_parquet(out_path)
    print(f"n (dedup frames) = {tot:,} -> {out_path}")
    print(out)


if __name__ == "__main__":
    main()
