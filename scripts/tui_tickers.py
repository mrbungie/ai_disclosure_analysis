"""
tui_tickers.py — Minimal TUI for the firm universe: view the aggregated
sectors (tech, semis, defensa, ...), see which SIC industry groups make up
each one, and build new sectors by picking SIC groups.

Model: `pipeline.sector_groups` in configs/config.json maps a sector name to
a list of SIC industry-group names (the `industry_group` values script 00
writes into firm_universe.parquet). A sector's tickers are DERIVED from that
mapping — assign a SIC group, and every ticker in it (current and future)
follows. `pipeline.tickers` stays the flat list scripts 00-03 read; the TUI
can also append new tickers to it (they get a SIC group after the next
`make build-universe`).

First run seeds the 12 thesis sectors covering all current SIC groups; edit
freely afterwards — the seed never overwrites an existing sector_groups.

Usage:
    make tickers-tui        (or: uv run python scripts/tui_tickers.py)
"""

import json
from pathlib import Path

CONFIG_PATH = Path("configs/config.json")
UNIVERSE_PATH = Path("data/interim/manifests/firm_universe.parquet")

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
        return json.load(f)


def save_config(config: dict) -> None:
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)
        f.write("\n")


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
    pipeline = config["pipeline"]
    pipeline.pop("ticker_groups", None)  # legacy SIC-copy seed, superseded by sector_groups
    if "sector_groups" not in pipeline:
        pipeline["sector_groups"] = {name: list(sics) for name, sics in SECTOR_SEED.items()}
        print(f"  Seeded {len(SECTOR_SEED)} aggregated sectors over the SIC groups.")
    return pipeline["sector_groups"]


def assigned_sics(sectors: dict[str, list[str]]) -> dict[str, str]:
    """SIC name -> sector it belongs to."""
    return {sic: sector for sector, sics in sectors.items() for sic in sics}


def unassigned(sectors: dict[str, list[str]], sic_map: dict[str, list[str]]) -> list[str]:
    taken = assigned_sics(sectors)
    return sorted(s for s in sic_map if s not in taken)


def sector_tickers(sics: list[str], sic_map: dict[str, list[str]]) -> list[str]:
    return sorted({t for s in sics for t in sic_map.get(s, [])})


def show_sectors(config: dict, sic_map: dict[str, list[str]], detail: bool = False) -> None:
    sectors = config["pipeline"]["sector_groups"]
    universe_tickers = {t for ts in sic_map.values() for t in ts}
    flat = set(config["pipeline"].get("tickers", []))
    print()
    for name, sics in sectors.items():
        tickers = sector_tickers(sics, sic_map)
        print(f"  [{name}]  {len(tickers)} tickers, {len(sics)} SIC group(s)")
        if detail:
            for s in sics:
                ts = " ".join(sic_map.get(s, [])) or "(no tickers yet)"
                print(f"      - {s}: {ts}")
    left_sics = unassigned(sectors, sic_map)
    if left_sics:
        print(f"  SIC groups not in any sector: {len(left_sics)} (option 2 to see them)")
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
    sectors = config["pipeline"]["sector_groups"]
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
    sectors = config["pipeline"]["sector_groups"]
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


def add_tickers(config: dict) -> None:
    raw = input("  New tickers for the universe (space/comma separated): ")
    toks = [t.strip().upper() for t in raw.replace(",", " ").split() if t.strip()]
    if not toks:
        print("  Cancelled.\n")
        return
    flat = set(config["pipeline"].get("tickers", []))
    new = sorted(set(toks) - flat)
    config["pipeline"]["tickers"] = sorted(flat | set(toks))
    save_config(config)
    print(f"  Added {len(new)} new ticker(s): {' '.join(new) if new else '(all already present)'}")
    print("  Run `make build-universe` so they get a SIC group, then assign that group to a sector.\n")


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
        print("  q. Quit")
        choice = input("\n  > ").strip().lower()
        if choice == "1":
            show_sectors(config, sic_map, detail=True)
        elif choice == "2":
            show_unassigned(config["pipeline"]["sector_groups"], sic_map)
        elif choice == "3":
            create_sector(config, sic_map)
        elif choice == "4":
            extend_sector(config, sic_map)
        elif choice == "5":
            add_tickers(config)
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
