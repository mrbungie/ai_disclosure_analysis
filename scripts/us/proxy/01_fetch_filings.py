"""
scripts/us/proxy/01_fetch_filings.py — builds the DEF 14A (proxy statement)
manifest AND fetches primary documents, via the shared edgar_fetch module.
See docs/document_expansion_plan.md Fase 3.

DEF 14A only, into its own storage (data/raw/filings_proxy/,
data/interim/manifests/filing_manifest_proxy.parquet) — a separate
instrument from 10-K/10-Q/8-K, same "never pooled" pattern as the 10-Q
shock series.

Usage:
    uv run python scripts/us/proxy/01_fetch_filings.py
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
        local_storage_dir=Path(".edgartools_data") / "proxy",
    )

    filing_date = config["corpus"]["filings_proxy"]["filing_date"]
    edgar_fetch.fetch_filings(
        universe_df=universe_df,
        form="DEF 14A",
        start_date=filing_date["from"],
        end_date=filing_date["to"],
        allow_amendments=config["corpus"]["filings_proxy"].get("amendments", False),
        html_dir=Path(config["storage"]["raw_html_proxy"]),
        manifest_path=Path(config["storage"]["interim_manifests"]) / "filing_manifest_proxy.parquet",
        universe_path=universe_path,
    )


if __name__ == "__main__":
    main()
