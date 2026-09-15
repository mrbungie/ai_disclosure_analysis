"""One-time reconciliation after the 2026-09-12 sentences segmentation fix
(see docs/judge_model_selection.md): decides, per text_hash already present
in an archived ai_classify.py run, whether its LLM-labeled frames are still
valid (sentence breakdown for that text didn't change) or must be redone (it
changed -- the frames were built from a different numbered-sentence prompt).

Never overwrites the archive. Writes a compacted "reusable" part into
--output-dir (default: the live data/interim/ai_classify/) containing ONLY
the rows for texts whose sentence breakdown is unchanged. Anything NOT
written there is implicitly "not yet classified" the next time
ai_classify.py's fetch_pending() runs -- no separate "delete" step needed.

`--old-sentences` is a parquet file or hive-partitioned directory with the
sentence table from before the fix (same schema as bronze.sentences); the
current breakdown is bronze.sentences.

Usage:
    uv run python scripts/enrichment/reconcile_ai_classify_after_resegment.py \\
        --archived-glob "data/archive/interim/ai_classify/20260912T174126Z/ai_frames__session=*.parquet" \\
        --old-sentences path/to/sentences_before_resegment \\
        --dry-run   # print counts only, write nothing
"""
from __future__ import annotations

import argparse
import glob
import sys
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "common"))
import layers as L  # noqa: E402

SENTENCE_KEY = ["accession_number", "item_key", "paragraph_index"]


def scan_sentences(path: Path) -> pl.LazyFrame:
    if path.is_dir():
        return pl.scan_parquet(path, hive_partitioning=True)
    return pl.scan_parquet(path)


def changed_text_hashes(old_sentences: Path) -> set[int]:
    """text_hash values, WITHIN the AI-positive population (the only ones
    ai_classify.py ever sends to the LLM), whose (accession_number,
    item_key, paragraph_index) now has a different sentence count than
    before the resegmentation fix. Scoped to that population on purpose --
    the same query without it also matches millions of irrelevant table
    paragraphs across the whole corpus, which would make "N changed" look
    alarming without meaning anything for what ai_classify.py needs redone."""
    positives = (L.scan("bronze.prefilter_predictions").filter(pl.col("is_ai_prefiltered"))
                 .select("text_hash"))
    old_n = scan_sentences(old_sentences).group_by(SENTENCE_KEY).agg(pl.len().alias("n_old"))
    new_n = L.scan("bronze.sentences").group_by(SENTENCE_KEY).agg(pl.len().alias("n_new"))
    changed = (L.scan("bronze.paragraphs").filter(pl.col("country_code") == "us")
               .select("text_hash", *SENTENCE_KEY)
               .join(positives, on="text_hash", how="semi")
               .join(old_n, on=SENTENCE_KEY).join(new_n, on=SENTENCE_KEY)
               .filter(pl.col("n_old") != pl.col("n_new"))
               .select("text_hash").unique().collect())
    return set(changed["text_hash"].to_list())


def reconcile(archived_glob: str, changed: set[int]) -> tuple[pl.DataFrame, int]:
    """Archived rows without error whose text is not in `changed`, and the
    number of distinct archived texts without error."""
    archived = (pl.concat([pl.scan_parquet(f) for f in sorted(glob.glob(archived_glob))],
                          how="diagonal_relaxed")
                .filter(pl.col("error").is_null()))
    changed_frame = pl.DataFrame({"text_hash": sorted(changed)}, schema={"text_hash": pl.UInt64})
    reusable = archived.join(changed_frame.lazy(), on="text_hash", how="anti").collect()
    total_archived = archived.select(pl.col("text_hash").n_unique()).collect().item()
    return reusable, total_archived


def to_archive_schema(frame: pl.DataFrame, archived_glob: str):
    """Arrow table with the column order and types of the archived parts."""
    schema = pq.read_schema(sorted(glob.glob(archived_glob))[0])
    table = frame.to_arrow(compat_level=pl.CompatLevel.oldest())
    if set(schema.names) != set(table.column_names):
        return table
    return table.select(schema.names).cast(schema)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--archived-glob", required=True)
    parser.add_argument("--old-sentences", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "data" / "interim" / "ai_classify")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    changed = changed_text_hashes(args.old_sentences)
    print(f"{len(changed):,} text_hash con segmentación distinta (se descartan, quedan pendientes)")

    reusable, total_archived = reconcile(args.archived_glob, changed)
    n_reusable_texts = reusable["text_hash"].n_unique()
    print(f"{total_archived:,} textos archivados ok -> {n_reusable_texts:,} reutilizables "
          f"({total_archived - n_reusable_texts:,} se descartan por cambio de segmentación)")
    print(f"{reusable.height:,} filas (frames) a copiar")

    if args.dry_run:
        print("--dry-run: no se escribió nada")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    session_id = "resegment-reconcile-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = args.output_dir / f"ai_frames__session={session_id}__part=00000.parquet"
    staging = destination.with_suffix(".parquet.partial")
    pq.write_table(to_archive_schema(reusable, args.archived_glob), staging, compression="zstd")
    staging.replace(destination)
    print(f"-> {destination}")


if __name__ == "__main__":
    main()
