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

There is no snapshot file and nothing to rewrite. An earlier version kept
`filing_manifest.parquet` as a "derived snapshot" recomputed on every
flush — which is the same whole-file rewrite wearing a different label, and
still something two processes could fight over. build_duckdb.py reads the
glob instead, and the pre-existing single file is simply MOVED once to
`filing_manifest__run=legacy__part=0000.parquet`: it becomes one more part,
never touched again.
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

    The single legacy file is ALWAYS unioned in when present, not only when
    no parts exist. A process started before this change is still writing
    it, and treating it as "superseded once a part appears" would silently
    drop whatever that process had fetched. Deduplication by
    `updated_at` sorts that out without either side needing to know about
    the other.
    """
    frames = []
    legacy = manifest_dir / SNAPSHOT_NAME
    if legacy.exists():
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


def adopt_legacy_file(manifest_dir: Path, run_id: str = "legacy") -> Path | None:
    """One-time MOVE of a pre-existing `filing_manifest.parquet` into the
    part naming, so it stops being a special case. A rename, never a
    rewrite: the bytes are untouched and no row is recomputed."""
    legacy = manifest_dir / SNAPSHOT_NAME
    if not legacy.exists():
        return None
    target = manifest_dir / f"filing_manifest__run={run_id}__part=0000.parquet"
    if target.exists():
        return target
    legacy.rename(target)
    return target
