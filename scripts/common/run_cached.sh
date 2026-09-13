#!/usr/bin/env bash
# Skip a command if none of its declared inputs changed since the last run
# that actually executed it. Used by the Makefile's analytics-* targets so
# `make analytics` (which render.py now runs before every render -- see
# thesis_document/render.py's refresh_analytics()) doesn't unconditionally
# re-run a 15-stage pipeline, including a several-minutes 200-replicate
# bootstrap, every single time.
#
# Cache key = sha256 of (path, size, mtime) for every file matched by the
# watch globs -- not file CONTENTS: hashing content would mean reading
# every byte of every watched file (some of these globs cover large XBRL/
# parquet trees) just to decide whether to skip work, which defeats the
# point. (path, size, mtime) is what `make` itself uses to decide staleness,
# just computed as one combined hash instead of per-file timestamp compares,
# so it also survives watch lists that mix a handful of scripts with a
# glob over hundreds of data files.
#
# Usage:
#   run_cached.sh <cache_key> <watch_glob> [<watch_glob> ...] -- <command...>
#
# A watch glob with no matches is silently skipped (e.g. an optional input
# that doesn't exist yet) -- same "skip cleanly" convention build_duckdb.py
# uses for a country whose files aren't built yet.
set -euo pipefail

if [[ $# -lt 3 ]]; then
    echo "Usage: run_cached.sh <cache_key> <watch_glob>... -- <command...>" >&2
    exit 2
fi

KEY="$1"; shift
WATCHES=()
while [[ "$1" != "--" ]]; do
    WATCHES+=("$1")
    shift
done
shift  # drop the -- separator

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CACHE_DIR="$REPO_ROOT/.make_cache"
mkdir -p "$CACHE_DIR"
HASH_FILE="$CACHE_DIR/$KEY.hash"

shopt -s nullglob globstar
ALL_FILES=()
for pattern in "${WATCHES[@]}"; do
    for f in $pattern; do
        [[ -f "$f" ]] && ALL_FILES+=("$f")
    done
done
shopt -u nullglob globstar

# One batched `stat` call (BSD stat; this project runs on darwin only, see
# CLAUDE.md) instead of one process per watched file -- a watch glob over
# the raw XBRL facts tree alone is thousands of files, and spawning `stat`
# once per file in a loop took over a minute just to decide whether there
# was anything to skip, most of it on this step alone.
NEW_HASH=""
if [[ ${#ALL_FILES[@]} -gt 0 ]]; then
    # xargs batches into argv-safe chunks -- a single `stat` call with
    # thousands of paths can exceed ARG_MAX depending on path lengths.
    NEW_HASH="$(printf '%s\n' "${ALL_FILES[@]}" | xargs -n 500 stat -f "%N %z %m" 2>/dev/null | sort | shasum -a 256 | cut -d' ' -f1)"
fi

OLD_HASH=""
[[ -f "$HASH_FILE" ]] && OLD_HASH="$(cat "$HASH_FILE")"

if [[ -n "$NEW_HASH" && "$NEW_HASH" == "$OLD_HASH" ]]; then
    echo "[cache hit]  $KEY -- inputs unchanged, skipping: $*"
    exit 0
fi

echo "[cache miss] $KEY -- running: $*"
"$@"
echo "$NEW_HASH" > "$HASH_FILE"
