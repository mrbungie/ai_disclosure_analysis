"""
scripts/it/manifest_store.py — append-only storage for Italy's filing
manifest.

WHY. `filing_manifest.parquet` was being REWRITTEN WHOLESALE on every
checkpoint: read the whole file, update a dict in memory, write the whole
file back. That is the one shape this project does not use anywhere else —
paragraphs, prefilter scores and embeddings are all append-only part files
matched by a glob, with the reader resolving which row wins. A whole-file
rewrite means any second process touching that file loses its writes, and
it means "re-run with a changed setting" has no answer except deleting
what is already there. Both of those actually happened here.

So: every run appends `filing_manifest__run=<id>__part=<n>.parquet` and
NOTHING ever rewrites a part. `read()` unions the parts and keeps the
newest row per `document_id`, which is the same rule
build_duckdb.py's `ai_prefilter_scores` view uses over its own append-only
score parts.

`filing_manifest.parquet` still exists, because build_duckdb.py's
per-country UNION reads it by that exact name. It is now a DERIVED
SNAPSHOT — recomputed from the parts, never the source of truth — so
overwriting it destroys nothing that the parts cannot reproduce.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PART_GLOB = "filing_manifest__run=*__part=*.parquet"
SNAPSHOT_NAME = "filing_manifest.parquet"


def part_paths(manifest_dir: Path) -> list[Path]:
    return sorted(manifest_dir.glob(PART_GLOB))


def read(manifest_dir: Path) -> pd.DataFrame:
    """Every filing known so far, newest row per document_id.

    The legacy single-file snapshot is read FIRST and treated as the oldest
    source, so a manifest written before this module existed is picked up
    rather than orphaned — and any part overrides it, which is what makes
    the migration a no-op instead of a re-fetch.
    """
    frames = []
    legacy = manifest_dir / SNAPSHOT_NAME
    if legacy.exists() and not part_paths(manifest_dir):
        frames.append(pd.read_parquet(legacy))
    for path in part_paths(manifest_dir):
        try:
            frames.append(pd.read_parquet(path))
        except Exception:  # noqa: BLE001 — a part mid-write is not fatal
            continue
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    if "updated_at" in combined.columns:
        combined = combined.sort_values("updated_at")
    return combined.drop_duplicates(subset="document_id", keep="last").reset_index(drop=True)


def append(rows: list[dict], manifest_dir: Path, run_id: str, part_num: int) -> Path | None:
    """Writes one new part. Never touches an existing one."""
    if not rows:
        return None
    manifest_dir.mkdir(parents=True, exist_ok=True)
    path = manifest_dir / f"filing_manifest__run={run_id}__part={part_num:04d}.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def write_snapshot(manifest_dir: Path) -> Path:
    """Recomputes `filing_manifest.parquet` from the parts, for
    build_duckdb.py to read by name. Derived, so this overwrite is safe:
    delete it and this function rebuilds it exactly."""
    snapshot = read(manifest_dir)
    path = manifest_dir / SNAPSHOT_NAME
    if not snapshot.empty:
        snapshot.to_parquet(path, index=False)
    return path
