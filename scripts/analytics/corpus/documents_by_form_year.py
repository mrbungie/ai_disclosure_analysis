"""Documents by form (and filing year) for the processed U.S. corpus
(thesis.qmd `tbl-documents-by-form-year`, lines ~522-560).

The qmd chunk itself only groups by `form` (`GROUP BY 1` on `paragraphs`,
scoped to the panel documents via the PANEL_DOCS predicate over
`filing_manifest`/`filing_manifest_10q`); despite its label, it has no year
breakdown. This script reproduces that per-form total exactly (verified
below) and additionally joins each document back to its manifest
`filing_date` to add the `year` column the label promises, computed from
`silver.filing_manifest` (accession_number/document_id) and
`silver.filing_manifest_10q`, joined against `bronze.paragraphs` (no silver
counterpart cataloged for paragraphs). Summing `documents` over year within
each form reproduces the qmd's per-form total exactly.

Usage:
    .venv/bin/python scripts/analytics/corpus/documents_by_form_year.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import layers as L  # noqa: E402

FORMS = ["10-K", "10-Q", "20-F", "DEF 14A", "8-K", "Earnings call"]


def main() -> None:
    fm = L.scan("silver.filing_manifest").filter(pl.col("country_code") == "us")
    fmq = L.scan("silver.filing_manifest_10q").filter(pl.col("country_code") == "us")
    p = L.scan("bronze.paragraphs").filter(pl.col("country_code") == "us")

    panel_docs = pl.concat([
        fm.select(pl.col("document_id").alias("doc")),
        fmq.select(pl.col("accession_number").alias("doc")),
    ]).unique().collect()["doc"].implode()

    docs = (
        p.filter(pl.col("form").is_in(FORMS) & pl.col("accession_number").is_in(panel_docs))
        .select("form", "accession_number").unique().collect()
    )

    manifest_dates = pl.concat([
        fm.select(pl.col("document_id").alias("accession_number"), "filing_date"),
        fmq.select("accession_number", "filing_date"),
    ]).unique(subset=["accession_number"]).collect()

    joined = docs.join(manifest_dates, on="accession_number", how="left")
    unmatched = joined.filter(pl.col("filing_date").is_null()).height
    if unmatched:
        print(f"warning: {unmatched} documents with no manifest filing_date match")

    out = (
        joined.with_columns(pl.col("filing_date").dt.year().alias("year"))
        .group_by("form", "year").agg(pl.len().alias("documents"))
        .sort(["form", "year"])
    )

    out_path = L.results_path("corpus", "documents_by_form_year.parquet")
    out.write_parquet(out_path)
    print(f"{out.height} form-year rows -> {out_path}")
    by_form = out.group_by("form").agg(pl.col("documents").sum().alias("documents")).sort("documents", descending=True)
    print(by_form)
    print("total documents:", by_form["documents"].sum())


if __name__ == "__main__":
    main()
