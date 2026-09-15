"""
scripts/us/10q/01_fetch_filings.py — builds the 10-Q manifest AND fetches
primary documents, via the shared edgar_fetch module (scripts/us/edgar_fetch.py)
— SEC EDGAR fetch logic, specific to this country/source.

10-Q only, into its OWN storage (data/raw/filings_html_10q/,
data/interim/manifests/filing_manifest_10q.parquet) — never pooled with
the 10-K panel (data-scope decision: 10-K panel core + 10-Q shock series,
kept as two separate instruments throughout).

Usage:
    uv run python scripts/us/10q/01_fetch_filings.py
"""

from pathlib import Path

import pandas as pd
import yaml

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/us/

import edgar_fetch


def main():
    with open("configs/us/config.yaml") as f:
        config = yaml.safe_load(f)

    universe_path = Path(config["storage"]["interim_manifests"]) / "firm_universe.parquet"
    if not universe_path.exists():
        print(f"Universe file not found at {universe_path}. Run scripts/us/00_build_firm_universe.py first.")
        return

    universe_df = pd.read_parquet(universe_path)

    edgar_fetch.configure(
        user_agent=config["sec"]["user_agent"],
        local_storage_dir=Path(".edgartools_data") / "10q",
    )

    filing_date = config["corpus"]["filings_10q"]["filing_date"]
    edgar_fetch.fetch_filings(
        universe_df=universe_df,
        form="10-Q",
        start_date=filing_date["from"],
        end_date=filing_date["to"],
        allow_amendments=config["corpus"]["filings_10q"].get("amendments", False),
        html_dir=Path("data/raw/filings_html_10q"),
        manifest_path=Path(config["storage"]["interim_manifests"]) / "filing_manifest_10q.parquet",
        universe_path=universe_path,
    )


if __name__ == "__main__":
    main()
