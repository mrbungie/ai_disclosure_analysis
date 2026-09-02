"""
scripts/common/build_duckdb.py — (re)creates duckdb/thesis.duckdb: SQL VIEWS
over the pipeline's parquet outputs, so any of them is one query away
instead of a pandas glob-and-concat every time.

Every view is a VIEW, not a materialized table — DuckDB re-evaluates the
underlying read_parquet(glob) on each query, so it's always current with
whatever's on disk; this script never copies data into the .duckdb file
itself, just (re)defines the views. Safe/cheap to rerun any time (e.g.
after a new extraction run adds part files, or the corpus grows).

Usage:
    uv run python scripts/common/build_duckdb.py
    duckdb duckdb/thesis.duckdb   # then: .tables / select * from filing_manifest limit 5;
"""

import sys
from pathlib import Path

import duckdb
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = REPO_ROOT / "duckdb" / "thesis.duckdb"


def main():
    with open(REPO_ROOT / "configs" / "us" / "config.yaml") as f:
        config = yaml.safe_load(f)

    manifests_dir = REPO_ROOT / config["storage"]["interim_manifests"]
    sections_dir = REPO_ROOT / config["storage"]["interim_sections"]
    market_prices_dir = REPO_ROOT / "data" / "raw" / "market" / "prices"
    market_factors_dir = REPO_ROOT / "data" / "raw" / "market" / "factors"

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB_PATH))

    views = {
        # --- scripts/us/10k: firm universe + filing manifest + extraction trace ---
        "firm_universe": f"""
            SELECT * FROM read_parquet('{manifests_dir}/firm_universe.parquet')
        """,
        "filing_manifest": f"""
            SELECT * FROM read_parquet('{manifests_dir}/filing_manifest.parquet')
        """,
        # --- scripts/us/10q: 10-Q shock series — a SEPARATE instrument, never
        # pooled with filing_manifest above (see the project's data-scope
        # decision) — hence its own manifest AND its own extraction_trace/
        # filing_sections views below, never a UNION with the 10-K ones.
        "filing_manifest_10q": f"""
            SELECT * FROM read_parquet('{manifests_dir}/filing_manifest_10q.parquet')
        """,
        # Full trace: one row per (filing, target item), found or not.
        # See scripts/us/section_segmenter.py:load_extraction_trace —
        # this view is that same contract in SQL. union_by_name=True
        # matters here: a part file whose "note" column is 100% NULL (every
        # row in that checkpoint's batch was `found=True`) gets that column
        # typed NULL rather than VARCHAR by Parquet's own type inference —
        # without union_by_name, read_parquet's glob takes its schema from
        # just the FIRST file it opens, so a query touching "note" on a glob
        # spanning many runs breaks the moment file ordering picks one of
        # those all-NULL files as the reference schema (hit exactly this:
        # ConversionException reading "note" as NULL after a rerun added a
        # new run_id's part files to the same glob).
        "extraction_trace": f"""
            SELECT *
            FROM read_parquet(
                '{sections_dir}/filing_sections__run=*__part=*.parquet', union_by_name=True
            )
            QUALIFY row_number() OVER (
                PARTITION BY accession_number, item_key ORDER BY run_date DESC
            ) = 1
        """,
        # Just the found sections — the text itself.
        "filing_sections": """
            SELECT * FROM extraction_trace WHERE found
        """,
        # 10-Q equivalent of the two views above — Item 2 (MD&A) only, see
        # scripts/us/10q/02_extract_sections.py for why.
        "extraction_trace_10q": f"""
            SELECT *
            FROM read_parquet(
                '{sections_dir}/filing_sections_10q__run=*__part=*.parquet', union_by_name=True
            )
            QUALIFY row_number() OVER (
                PARTITION BY accession_number, item_key ORDER BY run_date DESC
            ) = 1
        """,
        "filing_sections_10q": """
            SELECT * FROM extraction_trace_10q WHERE found
        """,
        # --- 03_market_data: prices + Fama-French factors ---
        "market_prices": f"""
            SELECT * FROM read_parquet('{market_prices_dir}/*.parquet', filename = true)
        """,
        "market_factors_daily": f"""
            SELECT * FROM read_parquet('{market_factors_dir}/ff3_daily.parquet')
        """,
        "market_factors_monthly": f"""
            SELECT * FROM read_parquet('{market_factors_dir}/ff3_monthly.parquet')
        """,
    }

    for name, query in views.items():
        con.execute(f"CREATE OR REPLACE VIEW {name} AS {query}")
        print(f"  view {name} OK")

    print(f"\nWrote -> {DB_PATH}")
    print("Open with: duckdb duckdb/thesis.duckdb   (then .tables, or SELECT * FROM <view> LIMIT 5;)")
    con.close()


if __name__ == "__main__":
    main()
