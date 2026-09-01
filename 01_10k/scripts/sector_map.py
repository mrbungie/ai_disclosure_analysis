"""
sector_map.py — Single resolver for "which aggregated sector is this firm in".

Two assignment levels, resolved in order:
  1. `pipeline.sector_overrides` (configs/config.json): {TICKER: sector} —
     per-company override, wins always. For firms whose SIC group lands them
     somewhere the thesis disagrees with (SHW's SIC is a retail group but the
     firm is a paint maker), or firms with no usable SIC (FRC's EDGAR profile
     is empty).
  2. `pipeline.sector_groups`: {sector: [SIC industry-group names]} — the
     rule layer; a firm inherits the sector of its SIC group.
Anything matching neither resolves to "unassigned".

Used by the TUI and by any analysis script that needs firm->sector — import
this instead of re-deriving from config, so both levels apply everywhere.
"""

import json
from pathlib import Path

CONFIG_PATH = Path("configs/config.json")
UNIVERSE_PATH = Path("data/interim/manifests/firm_universe.parquet")

UNASSIGNED = "unassigned"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return json.load(f)


def sic_rule_map(config: dict) -> dict[str, str]:
    """SIC industry-group name -> sector, from the rule layer."""
    return {sic: sector
            for sector, sics in config["pipeline"].get("sector_groups", {}).items()
            for sic in sics}


def resolve_sectors(config: dict | None = None) -> dict[str, str]:
    """ticker -> sector for every firm in firm_universe.parquet, applying
    overrides over SIC rules. Tickers in overrides but not (yet) in the
    universe parquet are included too, so an override never silently
    disappears."""
    import pandas as pd
    if config is None:
        config = load_config()
    overrides = {t.upper(): s for t, s in config["pipeline"].get("sector_overrides", {}).items()}
    rules = sic_rule_map(config)

    out: dict[str, str] = {}
    if UNIVERSE_PATH.exists():
        df = pd.read_parquet(UNIVERSE_PATH)
        for ticker, sic in zip(df["ticker"], df["industry_group"]):
            out[ticker] = rules.get(str(sic), UNASSIGNED)
    for ticker, sector in overrides.items():
        out[ticker] = sector
    return out


def sector_members(config: dict | None = None) -> dict[str, list[str]]:
    """sector -> sorted tickers (overrides applied). Includes 'unassigned'
    when any firm resolves there."""
    resolved = resolve_sectors(config)
    members: dict[str, list[str]] = {}
    for ticker, sector in resolved.items():
        members.setdefault(sector, []).append(ticker)
    return {s: sorted(ts) for s, ts in members.items()}
