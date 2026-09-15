"""
scripts/bronze/unique_paragraphs.py — bronze.unique_paragraphs.

One row per distinct `text_hash`, the dedup surface every compute-heavy
enrichment step (embeddings, prefilter, LLM calls) runs on. The instance with
the smallest (country_code, form, accession_number, item_key,
paragraph_index) represents the text; `duplicate_count` counts its instances.
"""

from __future__ import annotations

import polars as pl
from _paths import L
from text_split import KEY

BUILDER = "scripts/bronze/unique_paragraphs.py"
PARAGRAPH_KEY = [*KEY, "paragraph_index"]


def main() -> None:
    par = L.scan("bronze.paragraphs")
    rep = (par.select(*PARAGRAPH_KEY, "text_hash")
           .sort(PARAGRAPH_KEY)
           .group_by("text_hash", maintain_order=True)
           .agg(*(pl.col(k).first() for k in PARAGRAPH_KEY), pl.len().cast(pl.Int64).alias("duplicate_count"))
           .collect())
    texts = (par.join(rep.lazy().select(PARAGRAPH_KEY), on=PARAGRAPH_KEY, how="semi")
             .select(*PARAGRAPH_KEY, "text_hash", "paragraph_text", "content_type", "is_scorable")
             .collect(engine="streaming"))
    out = rep.join(texts, on=[*PARAGRAPH_KEY, "text_hash"]).select(
        *PARAGRAPH_KEY, "text_hash", "paragraph_text", "content_type", "is_scorable", "duplicate_count")
    inputs = sorted(L.path("bronze.paragraphs").rglob("*.parquet"))
    L.write_table("bronze.unique_paragraphs", out, keys=["text_hash"], inputs=inputs, builder=BUILDER)


if __name__ == "__main__":
    main()
