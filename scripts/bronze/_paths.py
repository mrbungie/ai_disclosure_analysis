"""Shared setup for the bronze builders: import path, source locations, logging."""

from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for sub in ("common", "bronze", "enrichment"):
    p = str(REPO_ROOT / "scripts" / sub)
    if p not in sys.path:
        sys.path.insert(0, p)

import layers as L  # noqa: E402

COUNTRY = "us"
MANIFESTS = L.INTERIM / "manifests"
SECTIONS = L.INTERIM / "sections"
PRICES = L.DATA / "raw" / "market" / "prices"
FACTORS = L.DATA / "raw" / "market" / "factors"


def files(base: Path, pattern: str) -> list[Path]:
    return sorted(base.glob(pattern))


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
