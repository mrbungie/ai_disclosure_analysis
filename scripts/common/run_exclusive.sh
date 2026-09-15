#!/usr/bin/env bash
# Run a command while holding a machine-wide lock, so memory-heavy pipeline
# steps started by different sessions/agents never overlap on one machine.
# Waits (polling) for the lock instead of failing.
#
# Usage:
#   scripts/common/run_exclusive.sh <command...>
#   LOCK_NAME=heavy scripts/common/run_exclusive.sh make silver
set -euo pipefail

LOCK_DIR="${TMPDIR:-/tmp}/thesis-${LOCK_NAME:-heavy}.lock"
waited=0
until mkdir "$LOCK_DIR" 2>/dev/null; do
  if [[ -f "$LOCK_DIR/pid" ]] && ! kill -0 "$(cat "$LOCK_DIR/pid")" 2>/dev/null; then
    rm -rf "$LOCK_DIR"   # stale lock left by a killed holder
    continue
  fi
  (( waited % 60 == 0 )) && echo "waiting for $LOCK_DIR (held by pid $(cat "$LOCK_DIR/pid" 2>/dev/null || echo '?'))" >&2
  sleep 5; waited=$((waited + 5))
done
echo $$ > "$LOCK_DIR/pid"
trap 'rm -rf "$LOCK_DIR"' EXIT
"$@"
