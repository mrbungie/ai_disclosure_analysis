"""
scripts/it/00_build_firm_universe.py — the Italian issuer universe, derived
from filings.xbrl.org's own entity list rather than an index membership
file.

WHY NOT A FTSE MIB CONSTITUENT LIST. The US universe starts from an index
(S&P 500 frozen at 2021-12-31) because the population of SEC filers is far
larger than the study needs. Italy is the opposite case: every issuer that
files ESEF is already, by construction, a listed company with a mandatory
annual report, and there are 225 of them. Filtering that down to an index
would mean sourcing and versioning a constituent list — a second data
dependency, with its own point-in-time problems — to discard issuers this
project can afford to keep. Same call Chile's universe makes.

The LEI is the issuer key. Unlike a CIK or a RUT it is globally unique and
already present in the filings themselves, so nothing here has to resolve
identifiers across systems.

Writes configs/it/universe.csv, versioned in git the same way
configs/us/universe.csv and configs/cl/universe.csv are: the universe is a
research decision, not derived data, so it belongs next to the config that
declares it rather than under data/.

Usage:
    uv run python scripts/it/00_build_firm_universe.py
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from xbrl_filings_client import iter_filings


def main() -> None:
    config = yaml.safe_load((REPO_ROOT / "configs" / "it" / "config.yaml").read_text())
    country = config["source"]["country_filter"]
    window = config["corpus"]["filings"]["period_end"]

    issuers: dict[str, dict] = {}
    n_filings = 0
    for attributes, name, identifier in iter_filings(country):
        period_end = attributes.get("period_end") or ""
        if not (window["from"] <= period_end <= window["to"]):
            continue
        n_filings += 1
        # An issuer with no entity record still counts as an issuer; keying
        # it by LEI would drop it, so fall back to the filing's own id
        # prefix, which is the LEI in every ESEF filing seen so far.
        lei = identifier or (attributes.get("fxo_id") or "").split("-")[0]
        if not lei:
            continue
        row = issuers.setdefault(lei, {
            "lei": lei, "name": name, "country_code": country.lower(),
            "n_filings": 0, "first_period_end": period_end, "last_period_end": period_end,
        })
        row["n_filings"] += 1
        row["name"] = row["name"] or name
        row["first_period_end"] = min(row["first_period_end"], period_end)
        row["last_period_end"] = max(row["last_period_end"], period_end)

    universe = pd.DataFrame(sorted(issuers.values(), key=lambda r: r["lei"]))
    universe["source"] = config["source"]["name"]
    universe["built_at"] = datetime.now(timezone.utc).isoformat()

    out_path = REPO_ROOT / "configs" / "it" / "universe.csv"
    universe.to_csv(out_path, index=False)
    print(f"{len(universe)} issuers, {n_filings} filings in "
          f"[{window['from']}, {window['to']}] -> {out_path}")
    print(universe["n_filings"].describe().to_string())


if __name__ == "__main__":
    main()
