"""
scripts/us/8k/01_fetch_filings.py — builds the 8-K manifest AND fetches
primary documents, via the shared edgar_fetch module. See
docs/document_expansion_plan.md Fase 1.

8-K only, into its own storage (data/raw/filings_8k/,
data/interim/manifests/filing_manifest_8k.parquet) — highest-volume of
the three new document types (dozens per company/year, vs. one for
proxy), never pooled with 10-K/10-Q/proxy.

Usage:
    uv run python scripts/us/8k/01_fetch_filings.py
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
        local_storage_dir=Path(".edgartools_data") / "8k",
    )

    filing_date = config["corpus"]["filings_8k"]["filing_date"]
    edgar_fetch.fetch_filings(
        universe_df=universe_df,
        form="8-K",
        start_date=filing_date["from"],
        end_date=filing_date["to"],
        allow_amendments=config["corpus"]["filings_8k"].get("amendments", False),
        html_dir=Path(config["storage"]["raw_html_8k"]),
        manifest_path=Path(config["storage"]["interim_manifests"]) / "filing_manifest_8k.parquet",
        universe_path=universe_path,
    )


if __name__ == "__main__":
    main()
