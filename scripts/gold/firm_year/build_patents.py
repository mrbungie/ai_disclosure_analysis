"""Firm-year AI patents: Google Patents / OECD (2025) AI-patent counts per
(ticker, filing year) for universe firms matched to a patent assignee -- an
external, non-textual benchmark of AI technological capacity. A firm-year
without a matched patent filing that year has null counts (the source cannot
tell a firm with no patents from a firm with no matched assignee).

Source: silver.patents_firm_year (scripts/silver/patents.py).
Output: covariates/firm_year/patents.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "common"))
import layers as L  # noqa: E402

COLUMNS = ["total_patents", "ai_patents_oecd", "ai_patents_core", "ai_patents_related", "ai_patent_intensity"]


def main() -> None:
    spine = L.read_gold("firm_year")
    patents = L.scan("silver.patents_firm_year").select(["ticker", "year"] + COLUMNS).collect().to_pandas()
    patents["year"] = patents["year"].astype(spine["year"].dtype)
    out = spine.merge(patents, on=["ticker", "year"], how="left", validate="one_to_one")
    L.write_gold("covariates", "firm_year", "patents", out, builder="scripts/gold/firm_year/build_patents.py",
                 inputs=[L.path("silver.patents_firm_year")])


if __name__ == "__main__":
    main()
