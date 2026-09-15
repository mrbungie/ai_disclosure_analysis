"""
scripts/bronze/xbrl_facts.py — bronze.xbrl_facts: every inline-XBRL fact
extracted per filing (`data/raw/xbrl_facts/us_by_filing/*.parquet`,
`scripts/raw_processing/us/04_extract_inline_xbrl_facts.py`), unioned by
column name.

One source file has 0 rows and one has `period_start` typed Null (a filing
with no duration facts, only instants) -- `diagonal_relaxed` promotes both
to the common schema. No dedup, no universe filter: bronze keeps one
cleaned table per source, gold decides what to do with repeated facts
across filings.

`source_file` (the file's name, e.g. an accession number) and `source_row`
(0-based row index within that file, before any filtering) are added so
consumers can reproduce the old "sorted-file-name, in-file order" read
order explicitly, e.g. for a "first appearance in file order" tie-break:
sort by (filing_date, source_file, source_row).
"""

from __future__ import annotations

import polars as pl
from _paths import L, files

BUILDER = "scripts/bronze/xbrl_facts.py"
XBRL_DIR = L.DATA / "raw" / "xbrl_facts" / "us_by_filing"


def main() -> None:
    fact_files = files(XBRL_DIR, "*.parquet")
    facts = pl.concat(
        [pl.scan_parquet(f).with_row_index("source_row")
           .with_columns(pl.lit(f.name).alias("source_file"))
         for f in fact_files],
        how="diagonal_relaxed",
    )
    L.write_table("bronze.xbrl_facts", facts, keys=["source_file", "source_row"],
                  sort_by=["source_file", "source_row"], inputs=fact_files, builder=BUILDER)


if __name__ == "__main__":
    main()
