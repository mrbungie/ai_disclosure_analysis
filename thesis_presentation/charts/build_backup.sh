#!/usr/bin/env bash
# Build every backup-slide chart (bk_*.py) into thesis_presentation/figures/.
set -euo pipefail
cd "$(dirname "$0")"

for f in bk_*.py; do
    echo "=== $f ==="
    ../../.venv/bin/python "$f"
done
