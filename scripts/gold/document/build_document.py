"""Document spine and the per-document disclosure volume.

One row per U.S. document of the analysis universe (10-K, 10-Q, DEF 14A, 8-K
and earnings calls) with scorable paragraphs, from
`ai_intensity.document_table()`:

  spines/document/document               id (= accession_number; the synthetic
                                         `TICKER_YYYYQn` document_id for calls),
                                         ticker, accession_number, fecha (filing
                                         or call date), channel, form, cik,
                                         period_end, call_fy, fye_month, fy
                                         (fiscal year the document covers)
  covariates/document/disclosure_volume  n_paragraphs, n_words and the AI frame
                                         counts (zero for a document without
                                         frames)

Every firm-year, firm-quarter and call disclosure measure is an aggregate of
these two tables; `gold_document_table()` returns them joined in the shape of
`document_table()` so the other gold builders read gold instead of
recomputing it from bronze.

Deterministic, no LLM.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "gold" / "posture"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "common"))
from ai_intensity import COUNT_COLUMNS, document_table  # noqa: E402

import layers as L  # noqa: E402

BUILDER = "scripts/gold/document/build_document.py"
SPINE_ATTRIBUTES = ["channel", "form", "cik", "period_end", "call_fy", "fye_month", "fy"]
VOLUME_COLUMNS = ["n_paragraphs", "n_words"] + COUNT_COLUMNS


def gold_document_table() -> pd.DataFrame:
    """`ai_intensity.document_table()` rebuilt from the gold document tables
    (same columns and values, `quarter` = calendar quarter of `fecha`)."""
    docs = L.read_gold("document", ("covariates", "disclosure_volume"))
    docs["fecha"] = docs["fecha"].astype("datetime64[us]")
    docs["quarter"] = docs["fecha"].dt.to_period("Q")
    return docs.drop(columns="id")


def main() -> None:
    docs = document_table()
    docs.insert(0, "id", docs["accession_number"])
    docs["fecha"] = docs["fecha"].astype("datetime64[ns]")
    L.write_gold("spines", "document", "document", docs[L.GOLD_SPINE_COLUMNS["document"] + SPINE_ATTRIBUTES],
                 builder=BUILDER, extra={"grain": "document (US filing or earnings-call transcript with scorable paragraphs)"})
    L.write_gold("covariates", "document", "disclosure_volume", docs[L.GOLD_SPINE_COLUMNS["document"] + VOLUME_COLUMNS],
                 builder=BUILDER)
    print(f"{len(docs):,} documents | {docs['ticker'].nunique():,} firms | "
          f"{docs['fecha'].min().date()} .. {docs['fecha'].max().date()}")


if __name__ == "__main__":
    main()
