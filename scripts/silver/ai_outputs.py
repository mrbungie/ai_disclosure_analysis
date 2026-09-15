"""
scripts/silver/ai_outputs.py — silver.ai_frames, silver.ai_activities,
silver.ai_entity_mentions: LLM and lexical outputs per paragraph INSTANCE.

Enrichment runs once per distinct text (bronze.unique_paragraphs); gold asks
"which filings said this". Instances are restricted to documents of the
analysis universe (silver.filing_manifest, silver.filing_manifest_10q). Each output row is broadcast via `text_hash` to
every paragraph instance carrying that text, with `duplicate_count` (how many
instances share it) kept for weighting.

Frames and activities are restricted to the population of the DEPLOYED
prefilter (bronze.prefilter_predictions, is_ai_prefiltered). The LLM outputs
are additive and keep every text any past deployment flagged; without this
restriction the tables would depend on the order of past deployments rather
than on the current model.

Lineage: (country_code, form, accession_number, item_key, paragraph_index)
-> bronze.paragraphs -> bronze.extraction_trace; (text_hash, session_id,
frame_id[, activity_id]) -> the interim LLM part file.
"""

from __future__ import annotations

import polars as pl
from _paths import PARAGRAPH_KEY, L, bronze_inputs, log

BUILDER = "scripts/silver/ai_outputs.py"

FRAME_COLUMNS = ["frame_id", "has_frame", "subject", "ai_type", "temporal", "concepts", "specificity",
                 "rhetoric", "valence", "sentence_ids", "judge_model", "prompt_version", "session_id",
                 "classified_at"]
ACTIVITY_COLUMNS = ["frame_id", "activity_id", "has_activity", "action", "object", "function", "target", "stage",
                    "source", "domain", "entities", "metrics", "evidence_type", "sentence_ids", "firm",
                    "judge_model", "prompt_version", "session_id", "classified_at"]


def panel_documents() -> pl.LazyFrame:
    """Documents of the analysis universe: filing accession numbers from the
    silver manifests, and the document_id that stands in for the accession
    number of an earnings call."""
    manifest = L.scan("silver.filing_manifest")
    return pl.concat([
        manifest.select(pl.col("accession_number")),
        manifest.filter(pl.col("accession_number").is_null()).select(pl.col("document_id").alias("accession_number")),
        L.scan("silver.filing_manifest_10q").select(pl.col("accession_number")),
    ]).drop_nulls().unique()


def instances() -> pl.LazyFrame:
    return (L.scan("bronze.paragraphs").select(*PARAGRAPH_KEY, "text_hash")
            .join(panel_documents(), on="accession_number", how="semi")
            .join(L.scan("bronze.unique_paragraphs").select("text_hash", "duplicate_count"), on="text_hash"))


def prefiltered() -> pl.LazyFrame:
    return L.scan("bronze.prefilter_predictions").filter(pl.col("is_ai_prefiltered")).select("text_hash")


def broadcast(outputs: pl.LazyFrame, columns: list[str]) -> pl.DataFrame:
    texts = outputs.select("text_hash").unique()
    return (instances().join(texts, on="text_hash", how="semi")
            .join(outputs, on="text_hash")
            .select(*PARAGRAPH_KEY, "text_hash", "duplicate_count", *columns)
            .collect(engine="streaming"))


def main() -> None:
    base = ("bronze.paragraphs", "bronze.unique_paragraphs")

    frames = L.scan("bronze.ai_frames").join(prefiltered(), on="text_hash", how="semi")
    out = broadcast(frames, FRAME_COLUMNS)
    L.write_table("silver.ai_frames", out, keys=[*PARAGRAPH_KEY, "frame_id"],
                  inputs=bronze_inputs(*base, "bronze.ai_frames", "bronze.prefilter_predictions"), builder=BUILDER)
    log(f"ai_frames: {out.height} rows")

    # `domain`: who USES or benefits from the AI activity (who performs it is
    # the frame's `subject`). internal = employees / internal processes;
    # customer_facing = customers / developers; unspecified otherwise.
    activities = (L.scan("bronze.ai_activities").join(prefiltered(), on="text_hash", how="semi")
                  .with_columns(pl.when(pl.col("target").is_in(["employees", "internal_process"]))
                                .then(pl.lit("internal"))
                                .when(pl.col("target").is_in(["customers", "developers"]))
                                .then(pl.lit("customer_facing"))
                                .otherwise(pl.lit("unspecified")).alias("domain")))
    out = broadcast(activities, ACTIVITY_COLUMNS)
    L.write_table("silver.ai_activities", out, keys=[*PARAGRAPH_KEY, "frame_id", "activity_id"],
                  inputs=bronze_inputs(*base, "bronze.ai_activities", "bronze.prefilter_predictions"),
                  builder=BUILDER)
    log(f"ai_activities: {out.height} rows")

    out = broadcast(L.scan("bronze.ai_entity_mentions"), ["term", "geo", "modality", "run_id"])
    L.write_table("silver.ai_entity_mentions", out, keys=[*PARAGRAPH_KEY, "term"],
                  inputs=bronze_inputs(*base, "bronze.ai_entity_mentions"), builder=BUILDER)
    log(f"ai_entity_mentions: {out.height} rows")


if __name__ == "__main__":
    main()
