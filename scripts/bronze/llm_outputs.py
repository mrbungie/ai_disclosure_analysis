"""
scripts/bronze/llm_outputs.py — bronze.ai_frames, bronze.ai_activities,
bronze.ai_entity_mentions (one row per text_hash × frame / activity / term).

Reads the additive LLM outputs without touching them. Failed calls (error not
null) are skipped. Frames keep the latest judge CALL per text (session_id,
classified_at): all rows of that call come together, so frame 0 of one call
is never paired with frame 1 of another. Activities use the same call rule
restricted to POSITIVE calls when one exists: the judge is not deterministic
across runs, so an extraction found by any run is kept even if a later run
returned no activity; only texts no run found activity for keep their latest
negative call. Entity mentions keep
the latest run per (text_hash, term).

Rows here are per distinct text; silver broadcasts them to paragraph
instances and restricts them to the deployed prefilter population.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
from _paths import L, files

BUILDER = "scripts/bronze/llm_outputs.py"


def latest_call(parts: list[Path]) -> pl.DataFrame:
    lf = pl.concat([pl.scan_parquet(f) for f in parts], how="diagonal_relaxed").filter(pl.col("error").is_null())
    latest = (lf.select("text_hash", "session_id", "classified_at").unique()
              .sort(["session_id", "classified_at"], descending=True, maintain_order=True)
              .unique("text_hash", keep="first", maintain_order=True))
    return lf.join(latest, on=["text_hash", "session_id", "classified_at"], how="semi").collect()


def latest_positive_call(parts: list[Path]) -> pl.DataFrame:
    lf = pl.concat([pl.scan_parquet(f) for f in parts], how="diagonal_relaxed").filter(pl.col("error").is_null())
    lf = (lf.with_columns(pl.col("has_activity").any().over("text_hash").alias("_any_positive"))
          .filter(pl.col("has_activity") | ~pl.col("_any_positive"))
          .drop("_any_positive"))
    latest = (lf.select("text_hash", "session_id", "classified_at").unique()
              .sort(["session_id", "classified_at"], descending=True, maintain_order=True)
              .unique("text_hash", keep="first", maintain_order=True))
    return lf.join(latest, on=["text_hash", "session_id", "classified_at"], how="semi").collect()


def main() -> None:
    frames = files(L.INTERIM / "ai_classify", "ai_frames__session=*.parquet")
    L.write_table("bronze.ai_frames", latest_call(frames), keys=["text_hash", "frame_id"],
                  inputs=frames, builder=BUILDER)

    activities = files(L.INTERIM / "ai_activities", "ai_activities__session=*.parquet")
    L.write_table("bronze.ai_activities", latest_positive_call(activities), keys=["text_hash", "frame_id", "activity_id"],
                  inputs=activities, builder=BUILDER)

    mentions = files(L.INTERIM / "ai_entity_mentions", "ai_entity_mentions__run=*.parquet")
    lf = (pl.concat([pl.scan_parquet(f) for f in mentions], how="diagonal_relaxed")
          .sort("run_id", descending=True, maintain_order=True)
          .unique(["text_hash", "term"], keep="first", maintain_order=True))
    L.write_table("bronze.ai_entity_mentions", lf, keys=["text_hash", "term"], inputs=mentions, builder=BUILDER)


if __name__ == "__main__":
    main()
