"""Shared setup for the silver builders: import path, analysis universe, logging."""

from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
_common = str(REPO_ROOT / "scripts" / "common")
if _common not in sys.path:
    sys.path.insert(0, _common)

import layers as L  # noqa: E402

# Firms downloaded for other reasons (later index additions, a curated core)
# stay in bronze only.
ANALYSIS_PANEL = L.ANALYSIS_PANEL
PARAGRAPH_KEY = ["country_code", "form", "accession_number", "item_key", "paragraph_index"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def bronze_inputs(*names: str) -> list[Path]:
    out: list[Path] = []
    for name in names:
        p = L.path(name)
        out += sorted(p.rglob("*.parquet")) if p.is_dir() else [p]
    return out
