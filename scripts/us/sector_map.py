"""
sector_map.py — Single resolver for "which aggregated sector is this firm in".

Two assignment levels, resolved in order:
  1. `classification.overrides` (configs/us/config.yaml): {TICKER: sector} —
     per-company override, wins always. For firms whose SIC group lands them
     somewhere the thesis disagrees with (SHW's SIC is a retail group but the
     firm is a paint maker), or firms with no usable SIC (FRC's EDGAR profile
     is empty).
  2. `classification.sector_groups`: {sector: [SIC major-group codes]} — the
     rule layer; a firm inherits the sector of its SIC major group (the first
     two digits of its SIC code).
Anything matching neither resolves to "unassigned".

Used by the TUI and by any analysis script that needs firm->sector — import
this instead of re-deriving from config, so both levels apply everywhere.
"""

import yaml
from pathlib import Path

CONFIG_PATH = Path("configs/us/config.yaml")
UNIVERSE_PATH = Path("data/interim/manifests/firm_universe.parquet")

UNASSIGNED = "unassigned"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def sic_rule_map(config: dict) -> dict[str, str]:
    """SIC major-group code -> sector, from the rule layer."""
    return {str(code).zfill(2): sector
            for sector, codes in config["classification"].get("sector_groups", {}).items()
            for code in codes}


def resolve_sectors(config: dict | None = None) -> dict[str, str]:
    """ticker -> sector for every firm in firm_universe.parquet, applying
    overrides over SIC rules. Tickers in overrides but not (yet) in the
    universe parquet are included too, so an override never silently
    disappears."""
    import pandas as pd
    if config is None:
        config = load_config()
    overrides = {t.upper(): s for t, s in config["classification"].get("overrides", {}).items()}
    rules = sic_rule_map(config)

    out: dict[str, str] = {}
    if UNIVERSE_PATH.exists():
        df = pd.read_parquet(UNIVERSE_PATH)
        for ticker, sic in zip(df["ticker"], df["sic"]):
            out[ticker] = rules.get(str(sic).zfill(4)[:2], UNASSIGNED)
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
