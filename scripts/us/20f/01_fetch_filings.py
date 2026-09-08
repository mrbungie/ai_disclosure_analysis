"""
scripts/us/20f/01_fetch_filings.py — builds the 20-F manifest AND fetches
primary documents, via the shared edgar_fetch module.

20-F only (foreign private issuers' annual report, the 10-K equivalent for
ASML, HMC, TM, TSM, UL in this universe), into its own storage
(data/raw/filings_20f/, data/interim/manifests/filing_manifest_20f.parquet)
— never pooled with the domestic 10-K panel, same "separate instrument"
pattern as proxy/8-K/10-Q.

Usage:
    uv run python scripts/us/20f/01_fetch_filings.py
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
        local_storage_dir=Path(".edgartools_data") / "20f",
    )

    filing_date = config["corpus"]["filings_20f"]["filing_date"]
    edgar_fetch.fetch_filings(
        universe_df=universe_df,
        form="20-F",
        start_date=filing_date["from"],
        end_date=filing_date["to"],
        allow_amendments=config["corpus"]["filings_20f"].get("amendments", False),
        html_dir=Path(config["storage"]["raw_html_20f"]),
        manifest_path=Path(config["storage"]["interim_manifests"]) / "filing_manifest_20f.parquet",
        universe_path=universe_path,
    )


if __name__ == "__main__":
    main()
