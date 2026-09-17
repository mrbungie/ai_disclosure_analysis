#!/usr/bin/env python
"""
FactSet company profile parser: read raw overview-profile JSON batches, output
one-row-per-ticker parquet.
Usage: uv run python scripts/sources/fs/parse_profile.py
"""
import json
import glob
from pathlib import Path
from datetime import date, datetime
import polars as pl


def _get(d, *path):
    cur = d
    for key in path:
        if cur is None:
            return None
        cur = cur.get(key)
    return cur


def parse_profile_batches(as_of: str = "2026-09-17"):
    repo_root = Path(__file__).parent.parent.parent.parent
    batch_dir = repo_root / "data/raw/fs/profile"
    output_file = batch_dir / f"profile_{as_of.replace('-', '')}.parquet"

    id_map = pl.read_parquet(repo_root / "data/raw/fs/reference/id_map.parquet")
    id_map_dict = {row["factset_id"]: row["ticker"] for row in id_map.to_dicts() if row["factset_id"]}

    batch_files = sorted(glob.glob(str(batch_dir / "fs_profile_batch_*.json")))
    print(f"Processing {len(batch_files)} batch files...")

    rows = []
    failed_ids = []
    seen_ids = set()

    for batch_file in batch_files:
        with open(batch_file) as f:
            batch_data = json.load(f)
        for factset_id, rec in batch_data.items():
            seen_ids.add(factset_id)
            if factset_id not in id_map_dict:
                continue
            ticker = id_map_dict[factset_id]

            if not rec.get("ok") or not rec.get("data"):
                failed_ids.append(factset_id)
                rows.append({
                    "ticker": ticker,
                    "factset_id": factset_id,
                    "as_of": date.fromisoformat(as_of),
                    "company_name": None,
                    "description": None,
                    "sector": None,
                    "sector_rbics_id": None,
                    "industry": None,
                    "industry_rbics_id": None,
                    "employee_number": None,
                    "founded_year": None,
                    "ipo_date": None,
                    "exchange_primary": None,
                    "dual_listed": None,
                })
                continue

            d = rec["data"]
            trade_start = _get(d, "stage", "tradeDateRange", "start", "value")
            ipo_date = None
            if trade_start:
                try:
                    ipo_date = datetime.strptime(trade_start, "%Y%m%d").date()
                except ValueError:
                    ipo_date = None

            founded = _get(d, "stage", "foundedYear", "value")
            try:
                founded_year = int(founded) if founded else None
            except (TypeError, ValueError):
                founded_year = None

            rows.append({
                "ticker": ticker,
                "factset_id": factset_id,
                "as_of": date.fromisoformat(as_of),
                "company_name": _get(d, "business", "name", "value"),
                "description": _get(d, "business", "description", "value"),
                "sector": _get(d, "business", "sector", "value"),
                "sector_rbics_id": _get(d, "business", "sector", "rbicsId"),
                "industry": _get(d, "business", "industry", "value"),
                "industry_rbics_id": _get(d, "business", "industry", "rbicsId"),
                "employee_number": _get(d, "size", "employeeNumber", "value"),
                "founded_year": founded_year,
                "ipo_date": ipo_date,
                "exchange_primary": _get(d, "stage", "exchangePrimary", "value"),
                "dual_listed": _get(d, "stage", "dualListed", "value"),
            })

    df = pl.DataFrame(rows, schema={
        "ticker": pl.Utf8,
        "factset_id": pl.Utf8,
        "as_of": pl.Date,
        "company_name": pl.Utf8,
        "description": pl.Utf8,
        "sector": pl.Utf8,
        "sector_rbics_id": pl.Utf8,
        "industry": pl.Utf8,
        "industry_rbics_id": pl.Utf8,
        "employee_number": pl.Int64,
        "founded_year": pl.Int32,
        "ipo_date": pl.Date,
        "exchange_primary": pl.Utf8,
        "dual_listed": pl.Int32,
    })

    df.write_parquet(output_file)

    print(f"\n=== Profile Coverage ===")
    print(f"Unique tickers: {df['ticker'].n_unique()}")
    print(f"Unique factset_ids seen in raw batches: {len(seen_ids)}")
    print(f"Ids not in id_map (dropped): {len(seen_ids - set(id_map_dict))}")
    print(f"Failed/empty ids (no profile data returned): {len(failed_ids)}")
    if failed_ids:
        print("  " + ", ".join(failed_ids))
    print(f"Output: {output_file}")
    return df, failed_ids


if __name__ == "__main__":
    parse_profile_batches()
