"""
scripts/us/6k/01_fetch_filings.py — builds the 6-K manifest AND fetches
primary documents, via the shared edgar_fetch module.

6-K only, into its own storage (data/raw/filings_6k/,
data/interim/manifests/filing_manifest_6k.parquet) — never pooled with the
domestic 8-K/10-Q series, same "separate instrument" pattern as 20-F/proxy.

6-K is the foreign private issuer's combined current-and-interim report —
the closest analogue to 8-K+10-Q for the same 5 tickers that file 20-F
instead of 10-K (ASML, HMC, TM, TSM, UL). Without it, those firms have
annual (20-F) but zero interim/current text coverage: 889 filings verified
missing across 2021-2026 before this script was added.

Usage:
    uv run python scripts/us/6k/01_fetch_filings.py
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
        local_storage_dir=Path(".edgartools_data") / "6k",
    )

    filing_date = config["corpus"]["filings_6k"]["filing_date"]
    edgar_fetch.fetch_filings(
        universe_df=universe_df,
        form="6-K",
        start_date=filing_date["from"],
        end_date=filing_date["to"],
        allow_amendments=config["corpus"]["filings_6k"].get("amendments", False),
        html_dir=Path(config["storage"]["raw_html_6k"]),
        manifest_path=Path(config["storage"]["interim_manifests"]) / "filing_manifest_6k.parquet",
        universe_path=universe_path,
    )


if __name__ == "__main__":
    main()
