"""Same reconciliation as reconcile_ai_classify_after_resegment.py, for
pass-2's archived output. A pass-2 row is reusable exactly when its
paragraph's sentence breakdown is unchanged: pass-1's frames for that
paragraph are then unchanged too (same prompt in, same frame_ids out), so
pass-2's frame_id references stay valid without a separate check -- no need
to re-verify frame_id against the reconciled pass-1 output specifically.

Usage:
    uv run python scripts/enrichment/reconcile_ai_activities_after_resegment.py \\
        --archived-glob "data/archive/interim/ai_activities/20260912T174126Z/ai_activities__session=*.parquet" \\
        --old-sentences path/to/sentences_before_resegment \\
        --dry-run
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "enrichment"))
# Same population-scoped definition of "changed": pass-2 rows are keyed by
# the same text_hash as their source paragraph.
from reconcile_ai_classify_after_resegment import (  # noqa: E402
    changed_text_hashes, reconcile, to_archive_schema)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--archived-glob", required=True)
    parser.add_argument("--old-sentences", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "data" / "interim" / "ai_activities")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    changed = changed_text_hashes(args.old_sentences)
    print(f"{len(changed):,} text_hash con segmentación distinta (se descartan, quedan pendientes)")

    reusable, total_archived = reconcile(args.archived_glob, changed)
    n_reusable_texts = reusable["text_hash"].n_unique()
    print(f"{total_archived:,} textos archivados ok -> {n_reusable_texts:,} reutilizables "
          f"({total_archived - n_reusable_texts:,} se descartan por cambio de segmentación)")
    print(f"{reusable.height:,} filas (actividades) a copiar")

    if args.dry_run:
        print("--dry-run: no se escribió nada")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    session_id = "resegment-reconcile-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = args.output_dir / f"ai_activities__session={session_id}__part=00000.parquet"
    staging = destination.with_suffix(".parquet.partial")
    pq.write_table(to_archive_schema(reusable, args.archived_glob), staging, compression="zstd")
    staging.replace(destination)
    print(f"-> {destination}")


if __name__ == "__main__":
    main()
