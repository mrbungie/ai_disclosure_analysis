"""Corpus headline quantities quoted in the front matter and Methodology
chapter of thesis.qmd (setup chunk, `{python}` lines ~140-236, and the
firm-sample chunk ~338-367).

The old DuckDB views `firm_universe`, `filing_manifest`, `filing_manifest_10q`
that those chunks queried were already scoped to the analysis universe (S&P
500 at 2021-01-01) -- confirmed here by matching the thesis's own hardcoded
"499 unique issuers" figure (thesis.qmd line 369) against `silver.firm_universe`
(499 rows, 39 delisted) rather than `bronze.firm_universe` (547 rows, 41
delisted, i.e. every ticker ever fetched, pre-panel-filter). `bronze.paragraphs`
has no silver counterpart cataloged in layers.py, so it is used as-is; it
already carries no rows outside the panel documents referenced here.

Feeds the thesis.qmd setup chunk (hd_n_docs, hd_n_paragraphs, hd_n_universe,
hd_paragraphs_m) and the firm-sample chunk (uni_frame_total, uni_frame_delisted,
cov_*). Activity, document and firm-year counts come from gold, not from here.

Usage:
    .venv/bin/python scripts/analytics/corpus/headline_counts.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import layers as L  # noqa: E402

FORMS = ["10-K", "10-Q", "20-F", "DEF 14A", "8-K", "Earnings call"]


def main() -> None:
    fm = L.scan("silver.filing_manifest").filter(pl.col("country_code") == "us")
    fmq = L.scan("silver.filing_manifest_10q").filter(pl.col("country_code") == "us")
    fu = L.scan("silver.firm_universe").filter(pl.col("country_code") == "us")
    p = L.scan("bronze.paragraphs").filter(pl.col("country_code") == "us")

    panel_docs = pl.concat([
        fm.select(pl.col("document_id").alias("doc")),
        fmq.select(pl.col("accession_number").alias("doc")),
    ]).unique().collect()["doc"].implode()

    p_panel = p.filter(pl.col("form").is_in(FORMS) & pl.col("accession_number").is_in(panel_docs)).collect()
    hd_n_docs = p_panel["accession_number"].n_unique()
    hd_n_paragraphs = p_panel.height
    hd_n_universe = fu.select(pl.len()).collect().item()

    active = fu.group_by("active_status").agg(pl.len().alias("n")).collect()
    uni_frame_total = int(active["n"].sum())
    uni_frame_delisted = int(active.filter(pl.col("active_status") == "delisted")["n"].sum())

    cov = pl.concat([
        fm.select(["form_type", "ticker", "filing_date"]),
        fmq.select(pl.lit("10-Q").alias("form_type"), "ticker", "filing_date"),
    ]).filter(pl.col("filing_date").dt.year() == 2026)
    cov_agg = cov.group_by("form_type").agg(
        pl.col("ticker").n_unique().alias("firms"),
        pl.col("filing_date").max().alias("last_date"),
    ).collect()
    cov_by_form = {r["form_type"]: r["firms"] for r in cov_agg.iter_rows(named=True)}
    cov_last_date = cov_agg["last_date"].max()

    cov_10k_2025 = fm.filter(
        (pl.col("form_type") == "10-K") & (pl.col("filing_date").dt.year() == 2025)
    ).select(pl.col("ticker").n_unique()).collect().item()

    out = {
        "hd_n_docs": hd_n_docs,
        "hd_n_paragraphs": hd_n_paragraphs,
        "hd_n_universe": hd_n_universe,
        "hd_paragraphs_m": round(hd_n_paragraphs / 1e6, 1),
        "uni_frame_total": uni_frame_total,
        "uni_frame_delisted": uni_frame_delisted,
        "cov_last_date": str(cov_last_date),
        "cov_10k_2026": cov_by_form.get("10-K", 0),
        "cov_10k_2025": int(cov_10k_2025),
        "cov_10q_2026": cov_by_form.get("10-Q", 0),
        "cov_calls_2026": cov_by_form.get("Earnings call transcript", 0),
        "cov_months_2026": cov_last_date.month,
    }

    out_path = L.results_path("corpus", "headline_counts.json")
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
