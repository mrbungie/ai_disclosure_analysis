"""One-time reconciliation after the 2026-09-12 sentences segmentation fix
(see docs/judge_model_selection.md and git history for context): decides,
per text_hash already present in an archived ai_classify.py run, whether its
LLM-labeled frames are still valid (sentence breakdown for that text didn't
change) or must be redone (it changed -- the frames were built from a
different, broken numbered-sentence prompt).

Never overwrites the archive. Writes a compacted "reusable" part into
--output-dir (default: the live data/interim/ai_classify/) containing ONLY
the rows for texts whose sentence breakdown is unchanged. Anything NOT
written there is implicitly "not yet classified" the next time
ai_classify.py's fetch_pending() runs -- no separate "delete" step needed.

Usage:
    uv run python scripts/common/reconcile_ai_classify_after_resegment.py \\
        --archived-glob "data/archive/interim/ai_classify/20260912T174126Z/ai_frames__session=*.parquet" \\
        --old-db duckdb/thesis.duckdb.before_resegment_backup \\
        --new-db duckdb/thesis.duckdb \\
        --dry-run   # print counts only, write nothing
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[2]


def changed_text_hashes(con, old_db: Path, new_db: Path) -> set[int]:
    """text_hash values, WITHIN the AI-positive population (the only ones
    ai_classify.py ever sends to the LLM), whose (accession_number,
    item_key, paragraph_index) now has a different sentence count than
    before the resegmentation fix. Scoped to that population on purpose --
    the same query without it also matches millions of irrelevant table
    paragraphs across the whole corpus, which would make "N changed" look
    alarming without meaning anything for what ai_classify.py needs redone."""
    con.execute(f"ATTACH '{old_db}' AS olddb (READ_ONLY)")
    con.execute(f"ATTACH '{new_db}' AS newdb (READ_ONLY)")
    df = con.execute("""
        WITH positives AS (
            SELECT text_hash FROM (
                SELECT text_hash, is_ai_prefiltered FROM read_parquet(
                    'data/interim/prefilter_predictions_unique/prefilter_predictions__run=*.parquet',
                    union_by_name=True)
                QUALIFY row_number() OVER (PARTITION BY text_hash ORDER BY model_version DESC) = 1
            ) WHERE is_ai_prefiltered
        ), old_n AS (SELECT accession_number, item_key, paragraph_index, count(*) AS n_old
                     FROM olddb.sentences GROUP BY 1,2,3),
        new_n AS (SELECT accession_number, item_key, paragraph_index, count(*) AS n_new
                  FROM newdb.sentences GROUP BY 1,2,3),
        joined AS (
            SELECT p.text_hash, n_old, n_new
            FROM newdb.paragraphs p
            JOIN positives pos ON pos.text_hash = p.text_hash
            JOIN old_n USING (accession_number, item_key, paragraph_index)
            JOIN new_n USING (accession_number, item_key, paragraph_index)
            WHERE p.country_code = 'us'
        )
        SELECT DISTINCT text_hash FROM joined WHERE n_old <> n_new
    """).df()
    return set(df["text_hash"].tolist())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--archived-glob", required=True)
    parser.add_argument("--old-db", type=Path, required=True)
    parser.add_argument("--new-db", type=Path, default=REPO_ROOT / "duckdb" / "thesis.duckdb")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "data" / "interim" / "ai_classify")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    con = duckdb.connect()
    con.execute("SET enable_progress_bar=false")
    changed = changed_text_hashes(con, args.old_db, args.new_db)
    print(f"{len(changed):,} text_hash con segmentación distinta (se descartan, quedan pendientes)")

    con.register("_changed", __import__("pandas").DataFrame({"text_hash": sorted(changed)}, dtype="uint64"))
    reusable = con.execute(f"""
        SELECT a.* FROM read_parquet('{args.archived_glob}', union_by_name=True) a
        WHERE a.error IS NULL
          AND NOT EXISTS (SELECT 1 FROM _changed c WHERE c.text_hash = a.text_hash)
    """).to_arrow_table()
    total_archived = con.execute(f"""
        SELECT count(DISTINCT text_hash) FROM read_parquet('{args.archived_glob}', union_by_name=True) WHERE error IS NULL
    """).fetchone()[0]
    n_reusable_texts = len(set(reusable.column("text_hash").to_pylist()))
    print(f"{total_archived:,} textos archivados ok -> {n_reusable_texts:,} reutilizables "
          f"({total_archived - n_reusable_texts:,} se descartan por cambio de segmentación)")
    print(f"{reusable.num_rows:,} filas (frames) a copiar")

    if args.dry_run:
        print("--dry-run: no se escribió nada")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    session_id = "resegment-reconcile-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = args.output_dir / f"ai_frames__session={session_id}__part=00000.parquet"
    staging = destination.with_suffix(".parquet.partial")
    pq.write_table(reusable, staging, compression="zstd")
    staging.replace(destination)
    print(f"-> {destination}")


if __name__ == "__main__":
    main()
