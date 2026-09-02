"""
tui_tickers.py — Minimal TUI for the firm universe: view the aggregated
sectors (tech, semis, defensa, ...), see which SIC industry groups make up
each one, and build new sectors by picking SIC groups.

Model — two assignment levels (resolved by scripts/sector_map.py, which
analysis code shares):
  1. `classification.overrides` {TICKER: sector} — per-company override,
     always wins (for firms whose SIC group misplaces them, or with no SIC).
  2. `classification.sector_groups` {sector: [SIC industry-group names]} — the
     rule layer; a firm inherits its SIC group's sector.

configs/us/universe.csv is the source of truth for the firm universe (ticker,
cik, company_name, active_status — see docs/universe_expansion_plan.md
Phase A). Inclusion-reason tags (sp500_2021, core_manual, ...) live
separately in configs/us/universe_membership.csv (ticker, group — long
format, a company can belong to more than one group). `corpus.universe.
tickers` in config.yaml is only a generated mirror (sorted tickers) that
script 00 rewrites every run for this TUI and other legacy code paths —
never edit it directly. The TUI appends new tickers to universe.csv and a
group="core_manual" row to universe_membership.csv; CIKs are resolved
against the local SEC ticker cache (data/sec_company_tickers.json) at
add-time, on a best-effort basis (they get a SIC group after the next
`make build-universe`).

First run seeds the 12 thesis sectors covering all current SIC groups; edit
freely afterwards — the seed never overwrites an existing sector_groups.

Usage:
    make tickers-tui        (or: uv run python scripts/us/tui_tickers.py)
"""

import csv
import json
from pathlib import Path

import yaml

import sector_map  # same dir, no path fix needed

CONFIG_PATH = Path("configs/us/config.yaml")
UNIVERSE_CSV_PATH = Path("configs/us/universe.csv")
UNIVERSE_MEMBERSHIP_CSV_PATH = Path("configs/us/universe_membership.csv")
UNIVERSE_PATH = Path("data/interim/manifests/firm_universe.parquet")
SEC_TICKERS_CACHE = Path("data/sec_company_tickers.json")

# Seed: the thesis' aggregated sectors, defined over the SIC industry_group
# names present in the current universe. Only used when config has no
# sector_groups yet.
SECTOR_SEED: dict[str, list[str]] = {
    "tech": [
        "Services-Prepackaged Software",
        "Services-Computer Programming, Data Processing, Etc.",
        "Services-Computer Processing & Data Preparation",
        "Services-Computer Programming Services",
        "Computer & office Equipment",
        "Computer Communications Equipment",
        "Computer Peripheral Equipment, NEC",
        "Electronic Computers",
        "Retail-Catalog & Mail-Order Houses",
        "Services-Video Tape Rental",
    ],
    "semis": [
        "Semiconductors & Related Devices",
        "Special Industry Machinery, NEC",
        "Optical Instruments & Lenses",
        "Radio & Tv Broadcasting & Communications Equipment",
    ],
    "defensa": [
        "Aircraft",
        "Aircraft Engines & Engine Parts",
        "Aircraft Parts & Auxiliary Equipment, NEC",
        "Guided Missiles & Space Vehicles & Parts",
        "Search, Detection, Navigation, Guidance, Aeronautical Sys",
        "Ship & Boat Building & Repairing",
        "Rolling Drawing & Extruding of  Nonferrous Metals",
    ],
    "industriales": [
        "Construction Machinery & Equip",
        "Farm Machinery & Equipment",
        "General Industrial Machinery & Equipment",
        "Electronic & Other Electrical Equipment (No Computer Equip)",
        "Surgical & Medical Instruments & Apparatus",
        "Railroads, Line-Haul Operating",
        "Trucking & Courier Services (No Air)",
        "Air Courier Services",
    ],
    "telecom": [
        "Telephone Communications (No Radiotelephone)",
        "Radiotelephone Communications",
        "Cable & Other Pay Television Services",
    ],
    "autos": [
        "Motor Vehicles & Passenger Car Bodies",
    ],
    "retail": [
        "Retail-Variety Stores",
        "Retail-Lumber & Other Building Materials Dealers",
        "Retail-Eating  Places",
        "Retail-Eating & Drinking Places",
    ],
    "consumo": [
        "Beverages",
        "Perfumes, Cosmetics & Other Toilet Preparations",
        "Soap, Detergents, Cleang Preparations, Perfumes, Cosmetics",
        "Rubber & Plastics Footwear",
    ],
    "energia": [
        "Petroleum Refining",
        "Crude Petroleum & Natural Gas",
        "Oil & Gas Field Services, NEC",
    ],
    "utilities": [
        "Electric Services",
        "Electric & Other Services Combined",
        "Gas & Other Services Combined",
    ],
    "salud": [
        "Pharmaceutical Preparations",
        "Biological Products, (No Diagnostic Substances)",
        "Orthopedic, Prosthetic & Surgical Appliances & Supplies",
        "Measuring & Controlling Devices, NEC",
    ],
    "financieras": [
        "National Commercial Banks",
        "Security Brokers, Dealers & Flotation Companies",
        "Services-Business Services, NEC",
        "Finance Services",
    ],
}


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def save_config(config: dict) -> None:
    with open(CONFIG_PATH, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False, default_flow_style=False, allow_unicode=True, width=1000)


def sic_to_tickers() -> dict[str, list[str]]:
    """SIC industry_group -> sorted tickers, from firm_universe.parquet."""
    if not UNIVERSE_PATH.exists():
        return {}
    import pandas as pd
    df = pd.read_parquet(UNIVERSE_PATH)
    out: dict[str, list[str]] = {}
    for sic, grp in df.groupby("industry_group"):
        out[str(sic)] = sorted(grp["ticker"])
    return out


def ensure_sectors(config: dict) -> dict[str, list[str]]:
    classification = config["classification"]
    classification.pop("ticker_groups", None)  # legacy SIC-copy seed, superseded by sector_groups
    if "sector_groups" not in classification:
        classification["sector_groups"] = {name: list(sics) for name, sics in SECTOR_SEED.items()}
        print(f"  Seeded {len(SECTOR_SEED)} aggregated sectors over the SIC groups.")
    return classification["sector_groups"]


def assigned_sics(sectors: dict[str, list[str]]) -> dict[str, str]:
    """SIC name -> sector it belongs to."""
    return {sic: sector for sector, sics in sectors.items() for sic in sics}


def unassigned(sectors: dict[str, list[str]], sic_map: dict[str, list[str]]) -> list[str]:
    taken = assigned_sics(sectors)
    return sorted(s for s in sic_map if s not in taken)


def sector_tickers(sics: list[str], sic_map: dict[str, list[str]]) -> list[str]:
    return sorted({t for s in sics for t in sic_map.get(s, [])})


def show_sectors(config: dict, sic_map: dict[str, list[str]], detail: bool = False) -> None:
    sectors = config["classification"]["sector_groups"]
    overrides = config["classification"].get("overrides", {})
    members = sector_map.sector_members(config)
    universe_tickers = {t for ts in sic_map.values() for t in ts}
    flat = set(config["corpus"]["universe"].get("tickers", []))
    print()
    for name, sics in sectors.items():
        tickers = members.get(name, [])
        n_over = sum(1 for s in overrides.values() if s == name)
        print(f"  [{name}]  {len(tickers)} tickers, {len(sics)} SIC group(s)"
              + (f", {n_over} override(s)" if n_over else ""))
        if detail:
            for s in sics:
                ts = " ".join(sic_map.get(s, [])) or "(no tickers yet)"
                print(f"      - {s}: {ts}")
            for t, s in sorted(overrides.items()):
                if s == name:
                    print(f"      * override: {t}")
    left_sics = unassigned(sectors, sic_map)
    if left_sics:
        print(f"  SIC groups not in any sector: {len(left_sics)} (option 2 to see them)")
    lost = sorted(members.get(sector_map.UNASSIGNED, []))
    if lost:
        print(f"  Firms resolving to no sector: {' '.join(lost)} (option 6 to override)")
    no_sic = sorted(flat - universe_tickers)
    if no_sic:
        print(f"  Tickers with no SIC yet (run `make build-universe`): {' '.join(no_sic)}")
    print()


def show_unassigned(sectors: dict[str, list[str]], sic_map: dict[str, list[str]]) -> list[str]:
    left = unassigned(sectors, sic_map)
    if not left:
        print("\n  Every SIC group is assigned to a sector.\n")
        return []
    print()
    for i, s in enumerate(left, 1):
        print(f"    {i:>2}. {s}  ({' '.join(sic_map[s])})")
    print()
    return left


def pick_sics(left: list[str]) -> list[str]:
    raw = input("  SIC group numbers (space/comma separated): ").replace(",", " ").split()
    picked = []
    for tok in raw:
        if tok.isdigit() and 1 <= int(tok) <= len(left):
            picked.append(left[int(tok) - 1])
    return picked


def create_sector(config: dict, sic_map: dict[str, list[str]]) -> None:
    sectors = config["classification"]["sector_groups"]
    left = show_unassigned(sectors, sic_map)
    if not left:
        return
    name = input("  New sector name: ").strip().lower().replace(" ", "_")
    if not name or name in sectors:
        print("  Cancelled (empty or existing name).\n")
        return
    picked = pick_sics(left)
    if not picked:
        print("  Cancelled (nothing picked).\n")
        return
    sectors[name] = picked
    save_config(config)
    print(f"  Created sector '{name}' with {len(picked)} SIC group(s) "
          f"({len(sector_tickers(picked, sic_map))} tickers).\n")


def extend_sector(config: dict, sic_map: dict[str, list[str]]) -> None:
    sectors = config["classification"]["sector_groups"]
    names = list(sectors)
    for i, n in enumerate(names, 1):
        print(f"    {i}. {n}")
    choice = input("  Sector number: ").strip()
    if not choice.isdigit() or not (1 <= int(choice) <= len(names)):
        print("  Cancelled.\n")
        return
    name = names[int(choice) - 1]
    left = show_unassigned(sectors, sic_map)
    if not left:
        return
    picked = pick_sics(left)
    if not picked:
        print("  Cancelled (nothing picked).\n")
        return
    sectors[name] = sorted(set(sectors[name]) | set(picked))
    save_config(config)
    print(f"  Added {len(picked)} SIC group(s) to '{name}'.\n")


def load_universe_rows() -> list[dict]:
    if not UNIVERSE_CSV_PATH.exists():
        return []
    with open(UNIVERSE_CSV_PATH, newline="") as f:
        return list(csv.DictReader(f))


def save_universe_rows(rows: list[dict]) -> None:
    fieldnames = ["ticker", "cik", "company_name", "active_status"]
    with open(UNIVERSE_CSV_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def add_membership(ticker: str, group: str) -> None:
    """Appends one (ticker, group) row to universe_membership.csv — the
    long-format side table for inclusion-reason tags (see
    00_build_firm_universe.py's module docstring for why it's separate
    from universe.csv: a company can belong to more than one group)."""
    is_new_file = not UNIVERSE_MEMBERSHIP_CSV_PATH.exists()
    with open(UNIVERSE_MEMBERSHIP_CSV_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["ticker", "group"])
        if is_new_file:
            w.writeheader()
        w.writerow({"ticker": ticker, "group": group})


def resolve_cik(ticker: str) -> tuple[str, str]:
    """Best-effort CIK/company_name lookup from the local SEC ticker cache."""
    if not SEC_TICKERS_CACHE.exists():
        return "", ""
    try:
        with open(SEC_TICKERS_CACHE) as f:
            sec_data = json.load(f)
    except Exception:
        return "", ""
    for entry in sec_data.values():
        if entry.get("ticker", "").upper() == ticker:
            return str(entry["cik_str"]).zfill(10), entry.get("title", "")
    return "", ""


def add_tickers() -> None:
    raw = input("  New tickers for the universe (space/comma separated): ")
    toks = [t.strip().upper() for t in raw.replace(",", " ").split() if t.strip()]
    if not toks:
        print("  Cancelled.\n")
        return
    rows = load_universe_rows()
    existing = {row["ticker"].upper() for row in rows}
    new = sorted(set(toks) - existing)
    if not new:
        print("  Cancelled (all already present in universe.csv).\n")
        return
    for t in new:
        cik, company_name = resolve_cik(t)
        active_status = "listed" if cik else "unresolved"
        rows.append({
            "ticker": t,
            "cik": cik,
            "company_name": company_name,
            "active_status": active_status,
        })
        add_membership(t, "core_manual")
        if not cik:
            print(f"  WARNING: could not resolve CIK for {t} from the local SEC ticker "
                  f"cache — added with active_status=unresolved; fill in cik by hand or "
                  f"refresh data/sec_company_tickers.json.")
    save_universe_rows(rows)
    print(f"  Added {len(new)} new ticker(s) to {UNIVERSE_CSV_PATH}: {' '.join(new)}")
    print("  Run `make build-universe` so they get a SIC group, then assign that group to a sector.\n")


def override_company(config: dict) -> None:
    """Assign one company to a sector directly, bypassing its SIC rule.
    Empty sector input clears an existing override."""
    overrides = config["classification"].setdefault("overrides", {})
    ticker = input("  Ticker to override: ").strip().upper()
    if not ticker:
        print("  Cancelled.\n")
        return
    resolved = sector_map.resolve_sectors(config)
    current = resolved.get(ticker)
    print(f"  {ticker}: currently -> {current if current else 'not in universe parquet'}"
          + (" (via override)" if ticker in overrides else " (via SIC rule)" if current else ""))
    names = list(config["classification"]["sector_groups"])
    for i, n in enumerate(names, 1):
        print(f"    {i}. {n}")
    choice = input("  Sector number (empty = clear override): ").strip()
    if not choice:
        if overrides.pop(ticker, None) is not None:
            save_config(config)
            print(f"  Cleared override for {ticker} — back to its SIC rule.\n")
        else:
            print("  Nothing to clear.\n")
        return
    if not choice.isdigit() or not (1 <= int(choice) <= len(names)):
        print("  Cancelled.\n")
        return
    overrides[ticker] = names[int(choice) - 1]
    save_config(config)
    print(f"  Override saved: {ticker} -> {overrides[ticker]}\n")


def main() -> None:
    config = load_config()
    ensure_sectors(config)
    save_config(config)
    sic_map = sic_to_tickers()
    if not sic_map:
        print("  NOTE: firm_universe.parquet not found — sectors will show 0 tickers "
              "until `make build-universe` runs.")

    print("=" * 60)
    print("  Firm universe — aggregated sectors over SIC groups")
    print("=" * 60)

    while True:
        show_sectors(config, sic_map)
        print("  1. View sectors in detail (SIC groups + tickers)")
        print("  2. View unassigned SIC groups")
        print("  3. Create a new sector from SIC groups")
        print("  4. Add SIC groups to an existing sector")
        print("  5. Add new tickers to the universe")
        print("  6. Override one company's sector (wins over its SIC rule)")
        print("  q. Quit")
        choice = input("\n  > ").strip().lower()
        if choice == "1":
            show_sectors(config, sic_map, detail=True)
        elif choice == "2":
            show_unassigned(config["classification"]["sector_groups"], sic_map)
        elif choice == "3":
            create_sector(config, sic_map)
        elif choice == "4":
            extend_sector(config, sic_map)
        elif choice == "5":
            add_tickers()
        elif choice == "6":
            override_company(config)
        elif choice in {"q", "quit", "exit"}:
            print("\n  Saved. New tickers enter the pipeline with: make collect-data\n")
            break
        else:
            print("  ?\n")


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\n  Bye.\n")
