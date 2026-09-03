"""
scripts/us/03_fetch_accounting_data.py — bulk XBRL "company facts" per firm
in configs/us/universe.csv, via SEC's own XBRL API
(data.sec.gov/api/xbrl/companyfacts/CIK##########.json), wrapped by
edgartools (already a project dependency, already used by
scripts/us/edgar_fetch.py for the narrative filings — same
EDGAR_IDENTITY/user-agent requirement, no new HTTP client needed).

This is the OUTCOME side of the panel (the "XBRL outcomes" instrument in
the project's data-scope decision), separate from the 10-K/10-Q narrative
text scripts/us/10k and scripts/us/10q already extract — see
docs/sources/accounting_data.md for why this needed its own script rather
than piggybacking on the filing fetch (one bulk pull per firm covering
every period/concept ever tagged, vs. the filing fetch's one call per
filing).

One row per (concept, period): concept, label, value, numeric_value,
unit, period_type, period_start, period_end, fiscal_year, fiscal_period
— edgartools' own EntityFacts.to_dataframe() shape, kept as-is rather
than reshaped, since collapsing it to a fixed set of "the concepts we
care about" would silently drop whatever the next research question
turns out to need.

Idempotent + resumable, same contract as scripts/us/edgar_fetch.py: a
ticker whose parquet already exists is skipped (use --refresh to
re-download).

Usage:
    uv run python scripts/us/03_fetch_accounting_data.py [--refresh] [--tickers AAPL,MSFT]
"""

import argparse
import sys
import time
from pathlib import Path

import edgar
import pandas as pd
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))  # scripts/common/
import pipeline_logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="Re-download even if a parquet already exists")
    parser.add_argument("--tickers", default=None, help="Comma-separated subset (smoke-testing)")
    args = parser.parse_args()

    with open("configs/us/config.yaml") as f:
        config = yaml.safe_load(f)
    edgar.set_identity(config["sec"]["user_agent"])

    universe = pd.read_csv("configs/us/universe.csv", dtype={"cik": str})
    if args.tickers:
        wanted = set(args.tickers.split(","))
        universe = universe[universe["ticker"].isin(wanted)]

    out_dir = Path(config["storage"]["raw_xbrl_facts"])
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir = Path(config["storage"]["interim_manifests"])

    pending = universe if args.refresh else universe[
        ~universe["ticker"].apply(lambda t: (out_dir / f"{t}.parquet").exists())
    ]
    if len(pending) == 0:
        print("Nothing to fetch — every ticker already has a parquet (use --refresh to redo).")
        return

    n_ok, n_missing = 0, 0
    for row in tqdm(pending.itertuples(), total=len(pending)):
        t0 = time.monotonic()
        try:
            company = edgar.Company(int(row.cik))
            facts = company.get_facts()
            df = facts.to_dataframe() if facts is not None else None
        except Exception as e:
            pipeline_logger.log_event(
                pipeline_step="us_xbrl_facts", level="ERROR",
                message=f"Exception fetching company facts: {e}", ticker=row.ticker, cik=row.cik,
                duration_seconds=time.monotonic() - t0, log_dir=manifest_dir,
            )
            n_missing += 1
            continue
        if df is None or df.empty:
            pipeline_logger.log_event(
                pipeline_step="us_xbrl_facts", level="WARNING",
                message="No XBRL facts returned", ticker=row.ticker, cik=row.cik, log_dir=manifest_dir,
            )
            n_missing += 1
            continue
        df.insert(0, "ticker", row.ticker)
        df.insert(1, "cik", row.cik)
        df.to_parquet(out_dir / f"{row.ticker}.parquet", index=False)
        pipeline_logger.log_event(
            pipeline_step="us_xbrl_facts", level="SUCCESS",
            message=f"Fetched {len(df)} XBRL fact rows", ticker=row.ticker, cik=row.cik,
            duration_seconds=time.monotonic() - t0, details={"rows": len(df)}, log_dir=manifest_dir,
        )
        n_ok += 1

    print(f"Done. {n_ok} tickers fetched, {n_missing} missing/failed -> {out_dir}")


if __name__ == "__main__":
    main()
