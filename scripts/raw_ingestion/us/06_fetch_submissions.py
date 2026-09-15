"""
scripts/raw_ingestion/us/06_fetch_submissions.py — EDGAR submissions history
(data.sec.gov/submissions) of every analysis-universe CIK.

One JSON per CIK in data/raw/sec_submissions/ (`CIK##########.json`, the
`recent` block) plus the older pages it points to (`CIK##########-submissions-
NNN.json`) whose filings reach back to `--since`. The filing index carries
the 8-K item codes (Item 2.02 results releases) and the delisting forms
(Form 25 / 25-NSE, Form 15) that scripts/silver/universe.py reads.

Idempotent: a file already on disk is not fetched again (`--refresh` refetches
the `recent` block, e.g. to see a delisting filed after the last pull).

Usage:
    .venv/bin/python scripts/raw_ingestion/us/06_fetch_submissions.py [--since 2019-01-01] [--refresh]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import requests
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "common"))
import layers as L  # noqa: E402

BASE = "https://data.sec.gov/submissions/"


def get(session: requests.Session, name: str, target: Path) -> dict | None:
    r = session.get(BASE + name, timeout=30)
    time.sleep(0.12)  # SEC fair-access limit: 10 requests/second
    if r.status_code != 200:
        print(f"  {name}: HTTP {r.status_code}")
        return None
    target.write_bytes(r.content)
    return r.json()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2019-01-01")
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    config = yaml.safe_load((REPO_ROOT / "configs/us/config.yaml").read_text())
    out = REPO_ROOT / config["storage"]["raw_submissions"]
    out.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": config["sec"]["user_agent"]})

    ciks = L.read("bronze.firm_universe").get_column("cik").drop_nulls().unique().sort().to_list()
    fetched = 0
    for cik in ciks:
        name = f"CIK{int(cik):010d}.json"
        target = out / name
        if target.exists() and not args.refresh:
            data = json.loads(target.read_text())
        else:
            data = get(session, name, target)
            fetched += 1
            if data is None:
                continue
        for page in data.get("filings", {}).get("files", []):
            page_target = out / page["name"]
            if page["filingTo"] >= args.since and not page_target.exists():
                get(session, page["name"], page_target)
                fetched += 1
    print(f"{len(ciks)} CIKs, {fetched} files fetched -> {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
