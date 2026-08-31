import csv
import json
from pathlib import Path

import pandas as pd

try:
    import pipeline_logger
except ImportError:
    from scripts import pipeline_logger

# Source of truth for the firm universe: configs/universe.csv (ticker, cik,
# company_name, inclusion_rule, active_status — see docs/universe_expansion_plan.md
# Phase A). configs/config.json's pipeline.tickers is kept only as a
# generated *mirror* (sorted tickers) for TUI/legacy code paths that still
# read it; it is never the source of truth and this script overwrites it
# every run to stay in sync with universe.csv.
UNIVERSE_CSV = Path("configs/universe.csv")


def load_universe_csv() -> list[dict]:
    with open(UNIVERSE_CSV, newline="") as f:
        return list(csv.DictReader(f))


def sync_config_ticker_mirror(config_path: Path, config: dict, rows: list[dict]) -> None:
    """Keep pipeline.tickers as a sorted mirror of universe.csv's tickers.

    universe.csv is the source of truth (see module docstring); this mirror
    exists only so config.json / tui_tickers.py keep working without
    reading the CSV themselves.
    """
    mirrored = sorted({row["ticker"].upper() for row in rows if row.get("ticker")})
    if config["pipeline"].get("tickers") == mirrored:
        return
    config["pipeline"]["tickers"] = mirrored
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
        f.write("\n")


def main():
    config_path = Path("configs/config.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    output_dir = Path(config["paths"]["interim_manifests"])
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
        inclusion_rule = (row.get("inclusion_rule") or "").strip()
        active_status = (row.get("active_status") or "").strip()

        if not cik:
            # A ticker with no resolvable CIK (e.g. an unresolved sp500_2021
            # row) contributes nothing to the manifest/download pipeline,
            # since 01 fetches by CIK. Report and skip rather than guess.
            skipped.append(ticker or "<blank>")
            pipeline_logger.log_event(
                pipeline_step="universe_build",
                level="WARNING",
                message=f"Row has no CIK, skipping: ticker={ticker!r} inclusion_rule={inclusion_rule!r}",
                ticker=ticker
            )
            continue

        cik_padded = str(cik).zfill(10)
        sic, industry_group = existing_sic.get(cik_padded, ("", ""))
        universe_data.append({
            "ticker": ticker,
            "cik": cik_padded,
            "company_name": company_name,
            "sic": sic,
            "industry_group": industry_group or "",
            "inclusion_rule": inclusion_rule,
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
