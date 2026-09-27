#!/bin/bash
# Rebuild every chart from data/gold + data/results, then render the deck.
# Output: thesis_presentation/presentation.html (single self-contained file).
set -euo pipefail
cd "$(dirname "$0")/charts"
PY=../../.venv/bin/python
for s in main_*.py bk_*.py cover_art.py; do $PY "$s" 2>&1 | grep -E "^wrote|Error|Traceback" || true; done
cd ..
quarto render presentation.qmd
