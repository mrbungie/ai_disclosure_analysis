"""Persist the document-level panel behind every firm-year intensity measure.

One row per U.S. document of the analysis universe (10-K, 10-Q, DEF 14A, 8-K
and earnings calls) with its ticker, filing date, fiscal year, scorable word
count and the AI frame counts `ai_intensity.document_table()` computes. The
firm-year tables (`build_firm_panels.py`) are aggregates of exactly this
table; persisting it lets the thesis reproduce any firm-year measure inside a
calendar window -- in particular the same-window seasonal adjustment used to
estimate full-year 2026 values from the partial year (thesis.qmd, "Temporal
coverage").

Deterministic, no LLM. Reads duckdb/thesis.duckdb; writes
data/processed/clusters/document_panel.parquet.
"""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ai_intensity import document_table  # noqa: E402

DB = REPO_ROOT / "duckdb" / "thesis.duckdb"
OUT_DIR = REPO_ROOT / "data" / "processed" / "clusters"
OUT_PATH = OUT_DIR / "document_panel.parquet"


def main() -> None:
    con = duckdb.connect(str(DB), read_only=True)
    docs = document_table(con)
    con.close()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    docs.to_parquet(OUT_PATH, index=False)
    print(f"{len(docs):,} documents | {docs['ticker'].nunique():,} firms | "
          f"{docs['fecha'].min().date()} .. {docs['fecha'].max().date()}")
    print(f"-> {OUT_PATH}")


if __name__ == "__main__":
    main()
