import csv
from collections import defaultdict
from pathlib import Path

import pandas as pd
import yaml

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))  # scripts/common/

import pipeline_logger

# Source of truth for the firm universe: configs/us/universe.csv (ticker, cik,
# company_name, active_status — see docs/universe_expansion_plan.md Phase A).
# configs/us/config.yaml's corpus.universe.tickers is kept only as a generated
# *mirror* (sorted tickers) for TUI/legacy code paths that still read it; it
# is never the source of truth and this script overwrites it every run to
# stay in sync with universe.csv.
#
# Membership/inclusion-reason tags (sp500_2021, core_manual, ...) live in a
# SEPARATE long-format dataset (configs/us/universe_membership.csv: ticker,
# group — one row per group a ticker belongs to) rather than a column on
# universe.csv itself, since a company can belong to more than one group at
# once (e.g. both sp500_2021 and a manually-added reason) — a single
# `inclusion_rule` column couldn't represent that without picking one
# arbitrarily. New groups are just new rows here, never a schema change.
UNIVERSE_CSV = Path("configs/us/universe.csv")
UNIVERSE_MEMBERSHIP_CSV = Path("configs/us/universe_membership.csv")


def load_universe_csv() -> list[dict]:
    with open(UNIVERSE_CSV, newline="") as f:
        return list(csv.DictReader(f))


def load_membership_groups() -> dict[str, list[str]]:
    """ticker -> sorted list of groups it belongs to (possibly several)."""
    groups = defaultdict(set)
    if UNIVERSE_MEMBERSHIP_CSV.exists():
        with open(UNIVERSE_MEMBERSHIP_CSV, newline="") as f:
            for row in csv.DictReader(f):
                ticker = (row.get("ticker") or "").strip().upper()
                group = (row.get("group") or "").strip()
                if ticker and group:
                    groups[ticker].add(group)
    return {ticker: sorted(g) for ticker, g in groups.items()}


def sync_config_ticker_mirror(config_path: Path, config: dict, rows: list[dict]) -> None:
    """Keep corpus.universe.tickers as a sorted mirror of universe.csv's tickers.

    universe.csv is the source of truth (see module docstring); this mirror
    exists only so config.yaml / tui_tickers.py keep working without
    reading the CSV themselves.
    """
    mirrored = sorted({row["ticker"].upper() for row in rows if row.get("ticker")})
    if config["corpus"]["universe"].get("tickers") == mirrored:
        return
    config["corpus"]["universe"]["tickers"] = mirrored
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False, default_flow_style=False, allow_unicode=True, width=1000)


def main():
    config_path = Path("configs/us/config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    output_dir = Path(config["storage"]["interim_manifests"])
    output_dir.mkdir(parents=True, exist_ok=True)

    if not UNIVERSE_CSV.exists():
        pipeline_logger.log_event(
            pipeline_step="universe_build",
            level="ERROR",
            message=f"{UNIVERSE_CSV} not found. It is the source of truth for the firm "
                    f"universe (see docs/universe_expansion_plan.md Phase A)."
        )
        return

    rows = load_universe_csv()
    membership_groups = load_membership_groups()
    pipeline_logger.log_event(
        pipeline_step="universe_build",
        level="INFO",
        message=f"Starting firm universe build from {UNIVERSE_CSV} ({len(rows)} rows)"
    )

    sync_config_ticker_mirror(config_path, config, rows)

    # Preserve any SIC/industry_group already resolved by 01 for firms whose
    # CIK doesn't change run to run, so a bare re-run of 00 doesn't blank out
    # data 01 already fetched (01 recomputes it anyway from the cached
    # submissions, but keeping it here avoids a spurious "Technology/Software"
    # placeholder in between).
    existing_sic = {}
    universe_path = output_dir / "firm_universe.parquet"
    if universe_path.exists():
        try:
            prev = pd.read_parquet(universe_path)
            for _, r in prev.iterrows():
                existing_sic[str(r["cik"])] = (r.get("sic", ""), r.get("industry_group", ""))
        except Exception as e:
            pipeline_logger.log_event(
                pipeline_step="universe_build",
                level="WARNING",
                message=f"Could not read existing firm_universe.parquet for SIC carry-over: {e}"
            )

    universe_data = []
    skipped = []
    for row in rows:
        ticker = (row.get("ticker") or "").strip().upper()
        cik = (row.get("cik") or "").strip()
        company_name = (row.get("company_name") or "").strip()
        active_status = (row.get("active_status") or "").strip()
        country = (row.get("country") or "US").strip()
        source = (row.get("source") or "SEC_EDGAR").strip()
        groups = membership_groups.get(ticker, [])

        if not cik:
            # A ticker with no resolvable CIK (e.g. an unresolved sp500_2021
            # row) contributes nothing to the manifest/download pipeline,
            # since 01 fetches by CIK. Report and skip rather than guess.
            # NOTE: this CIK requirement is SEC-EDGAR-specific — when a
            # non-US/non-EDGAR `source` is added, this check (and the
            # fetch scripts that key off `cik`) will need a per-source
            # identifier scheme, not just a looser version of this one.
            skipped.append(ticker or "<blank>")
            pipeline_logger.log_event(
                pipeline_step="universe_build",
                level="WARNING",
                message=f"Row has no CIK, skipping: ticker={ticker!r} groups={groups!r}",
                ticker=ticker
            )
            continue

        cik_padded = str(cik).zfill(10)
        sic, industry_group = existing_sic.get(cik_padded, ("", ""))
        universe_data.append({
            "ticker": ticker,
            "cik": cik_padded,
            "company_name": company_name,
            "country": country,
            "source": source,
            "sic": sic,
            "industry_group": industry_group or "",
            "membership_groups": groups,
            "active_status": active_status,
        })

    df = pd.DataFrame(universe_data)
    df.to_parquet(universe_path, index=False)

    pipeline_logger.log_event(
        pipeline_step="universe_build",
        level="SUCCESS",
        message=f"Successfully built firm universe with {len(df)} companies (skipped: {len(skipped)})",
        details={
            "company_count": len(df),
            "skipped_count": len(skipped),
            "output_file": str(universe_path)
        }
    )


if __name__ == "__main__":
    main()
