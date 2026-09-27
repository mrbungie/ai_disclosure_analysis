#!/usr/bin/env bash
# Build every main_*.py chart in this directory into thesis_presentation/figures/.
set -euo pipefail
cd "$(dirname "$0")"
PY=../../.venv/bin/python
for f in main_*.py; do
    echo "=== $f ==="
    "$PY" "$f"
done
