"""Extract point-in-time financial facts from cached US 10-K/10-Q HTML.

Company Facts collapses observations across filings, which lets a later
restatement overwrite the value that an investor could have known at an
earlier earnings call.  This extractor reads each filing already mirrored in
``data/raw/filings_html*`` and writes only its own facts, retaining
``accession_number`` and ``filing_date``.

The output retains every numerical inline-XBRL fact.  Concept selection happens
downstream: keeping custom tags here is essential for auditing and extending
the financial mappings without another extraction pass.  It is resumable: one
parquet per accession means an interrupted run restarts only missing filings.

Usage:
    uv run python scripts/us/04_extract_inline_xbrl_facts.py --workers 4
"""

from __future__ import annotations

import argparse
import gzip
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.analytics.filing_xbrl_facts import parse_inline_xbrl  # noqa: E402

OUT_DIR = REPO_ROOT / "data" / "raw" / "xbrl_facts" / "us_by_filing"


def output_path(accession_number: str) -> Path:
    return OUT_DIR / f"{accession_number}.parquet"


def has_valid_output(path: str) -> bool:
    try:
        return "accession_number" in pd.read_parquet(path).columns
    except Exception:
        return False


FACT_COLUMNS = [
    "ticker",
    "accession_number",
    "filing_date",
    "context_id",
    "has_dimensions",
    "concept",
    "numeric_value",
    "period_type",
    "period_start",
    "period_end",
]


def extract_one(record: dict[str, str]) -> tuple[str, int, str | None]:
    """Parse one local filing; worker-safe and side-effect free except its parquet."""
    destination = Path(record["destination"])
    try:
        with gzip.open(record["local_path"], "rt", encoding="utf-8", errors="replace") as source:
            html = source.read()
        facts = parse_inline_xbrl(
            html,
            ticker=record["ticker"],
            accession_number=record["accession_number"],
            filing_date=record["filing_date"],
        )
        table = pd.DataFrame(facts, columns=FACT_COLUMNS).drop_duplicates()
        destination.parent.mkdir(parents=True, exist_ok=True)
        table.to_parquet(destination, index=False)
        return record["accession_number"], len(table), None
    except Exception as exc:  # record and continue; the next run retries it
        return record["accession_number"], 0, f"{type(exc).__name__}: {exc}"


def filings() -> pd.DataFrame:
    manifests = [
        REPO_ROOT / "data" / "interim" / "manifests" / "filing_manifest.parquet",
        REPO_ROOT / "data" / "interim" / "manifests" / "filing_manifest_10q.parquet",
    ]
    records = pd.concat([pd.read_parquet(path) for path in manifests], ignore_index=True)
    records = records[(records["ticker"].notna()) & records["form_type"].isin(["10-K", "10-Q"])]
    records = records.dropna(subset=["accession_number", "filing_date", "local_path"])
    records["local_path"] = records["local_path"].map(lambda path: str(REPO_ROOT / path))
    records = records[records["local_path"].map(lambda path: Path(path).exists())].copy()
    records["destination"] = records["accession_number"].map(lambda accession: str(output_path(accession)))
    return records[["ticker", "accession_number", "filing_date", "local_path", "destination"]].drop_duplicates("accession_number")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true", help="Re-extract files already present")
    args = parser.parse_args()

    records = filings()
    if not args.force:
        records = records[~records["destination"].map(has_valid_output)]
    if args.limit is not None:
        records = records.head(args.limit)
    payloads = records.to_dict("records")
    print(f"extracting {len(payloads):,} cached filings into {OUT_DIR}")
    errors: list[tuple[str, str]] = []
    facts = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for accession, count, error in tqdm(executor.map(extract_one, payloads), total=len(payloads)):
            facts += count
            if error:
                errors.append((accession, error))
    print(f"wrote {facts:,} facts; {len(errors):,} filings failed")
    if errors:
        for accession, error in errors[:20]:
            print(f"  {accession}: {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
