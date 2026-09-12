"""Same reconciliation as reconcile_ai_classify_after_resegment.py, for
pass-2's archived output. A pass-2 row is reusable exactly when its
paragraph's sentence breakdown is unchanged: pass-1's frames for that
paragraph are then unchanged too (same prompt in, same frame_ids out), so
pass-2's frame_id references stay valid without a separate check -- no need
to re-verify frame_id against the reconciled pass-1 output specifically.

Usage:
    uv run python scripts/common/reconcile_ai_activities_after_resegment.py \\
        --archived-glob "data/archive/interim/ai_activities/20260912T174126Z/ai_activities__session=*.parquet" \\
        --old-db duckdb/thesis.duckdb.before_resegment_backup \\
        --new-db duckdb/thesis.duckdb \\
        --dry-run
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[2]


def changed_text_hashes(con, old_db: Path, new_db: Path) -> set[int]:
    """Same population-scoped definition of "changed" as
    reconcile_ai_classify_after_resegment.py -- kept in sync deliberately;
    pass-2 rows are keyed by the same text_hash as their source paragraph."""
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
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "data" / "interim" / "ai_activities")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    con = duckdb.connect()
    con.execute("SET enable_progress_bar=false")
    changed = changed_text_hashes(con, args.old_db, args.new_db)
    print(f"{len(changed):,} text_hash con segmentación distinta (se descartan, quedan pendientes)")

    con.register("_changed", pd.DataFrame({"text_hash": sorted(changed)}, dtype="uint64"))
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
    print(f"{reusable.num_rows:,} filas (actividades) a copiar")

    if args.dry_run:
        print("--dry-run: no se escribió nada")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    session_id = "resegment-reconcile-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = args.output_dir / f"ai_activities__session={session_id}__part=00000.parquet"
    staging = destination.with_suffix(".parquet.partial")
    pq.write_table(reusable, staging, compression="zstd")
    staging.replace(destination)
    print(f"-> {destination}")


if __name__ == "__main__":
    main()
