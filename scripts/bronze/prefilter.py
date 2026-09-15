"""
scripts/bronze/prefilter.py — bronze.prefilter_scores, bronze.prefilter_predictions,
bronze.prefilter_anchors, bronze.prefilter_entity_terms.

Reads the additive prefilter outputs without touching them.

  scores:      scores over unique texts (interim/prefilter_scores_unique). Score
               parts accumulate several complete populations (after
               retuning anchors or switching precision). Only the newest run's
               (model, anchors_fingerprint, dtype) population is kept, and
               within it the newest score per text. A text scored twice in the
               same run (once per paragraph instance, equal up to float noise)
               keeps the instance with the smallest natural key.
  predictions: the prediction of the latest `model_version` per text — the
               deployed prefilter that defines which texts are AI-flagged. The
               same model_version can be deployed several times; the newest
               deployment file wins.
  anchors / entity terms: configs/ai_prefilter.yaml as tables.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
from _paths import COUNTRY, L, files, log

import ai_prefilter_anchors as anchors  # noqa: E402  (on sys.path via _paths)

BUILDER = "scripts/bronze/prefilter.py"


def build_scores() -> None:
    parts = files(L.INTERIM / "prefilter_scores_unique", "prefilter_scores__run=*__part=*.parquet")
    scores = pl.concat([pl.scan_parquet(f) for f in parts], how="diagonal_relaxed").with_columns(
        pl.col("model", "anchors_fingerprint", "dtype", "run_id").cast(pl.String))
    current = scores.sort("run_id", descending=True).select("model", "anchors_fingerprint", "dtype").head(1)
    out = (scores.join(current, on=["model", "anchors_fingerprint", "dtype"], how="semi")
           .filter(pl.col("country_code") == COUNTRY)
           .sort(["run_id", "accession_number", "item_key", "paragraph_index"],
                 descending=[True, False, False, False], maintain_order=True)
           .unique("text_hash", keep="first", maintain_order=True)
           .collect())
    L.write_table("bronze.prefilter_scores", out, keys=["text_hash"], inputs=parts, builder=BUILDER)
    log(f"prefilter_scores: {out.height} texts")


def build_predictions() -> None:
    parts = files(L.INTERIM / "prefilter_predictions_unique", "prefilter_predictions__run=*.parquet")
    preds = (pl.concat([pl.scan_parquet(f).with_columns(pl.lit(f.name).alias("source_file")) for f in parts],
                       how="diagonal_relaxed")
             .filter(pl.col("country_code") == COUNTRY))
    # Latest model_version per text; the newest run file breaks ties.
    latest_version = preds.group_by("text_hash").agg(pl.col("model_version").max())
    candidates = preds.join(latest_version, on=["text_hash", "model_version"], how="semi")
    latest_file = candidates.group_by("text_hash").agg(pl.col("source_file").max())
    out = candidates.join(latest_file, on=["text_hash", "source_file"], how="semi").collect(engine="streaming")
    L.write_table("bronze.prefilter_predictions", out, keys=["text_hash"], inputs=parts, builder=BUILDER)
    log(f"prefilter_predictions: {out.height} texts")


def build_config_tables() -> None:
    cfg = Path(anchors.DEFAULT_CONFIG)
    L.write_table("bronze.prefilter_anchors", pl.DataFrame(anchors.anchor_rows()), keys=["anchor_id"],
                  inputs=[cfg], builder=BUILDER)
    terms = pl.DataFrame([{"term": t, "geo": m["geo"], "modality": m["modality"]}
                          for t, m in anchors.entity_terms(cfg).items()])
    L.write_table("bronze.prefilter_entity_terms", terms, keys=["term"], inputs=[cfg], builder=BUILDER)


def main() -> None:
    build_scores()
    build_predictions()
    build_config_tables()


if __name__ == "__main__":
    main()
