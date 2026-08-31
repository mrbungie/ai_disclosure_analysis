"""
tui_tickers.py — Minimal TUI to view the firm universe and add ticker groups.

The config gains `pipeline.ticker_groups` ({group name: [tickers]}) as the
source of truth; `pipeline.tickers` is kept in sync as the sorted, deduped
union so scripts 00-03 keep working unchanged. On first run the groups are
seeded from the PRE-NAMED industry groups in firm_universe.parquet
(`industry_group`, the SIC names built by script 00); tickers not yet in the
universe land in "ungrouped" until `make build-universe` runs again.

After changing groups: make collect-data (00-04 are resumable — only new
tickers/filings are fetched).

Usage:
    make tickers-tui        (or: uv run python scripts/tui_tickers.py)
"""

import json
from pathlib import Path

CONFIG_PATH = Path("configs/config.json")
UNIVERSE_PATH = Path("data/interim/manifests/firm_universe.parquet")


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return json.load(f)


def save_config(config: dict) -> None:
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)
        f.write("\n")


def ensure_groups(config: dict) -> dict[str, list[str]]:
    """Seed ticker_groups from the pre-named industry groups (SIC names in
    firm_universe.parquet). Runs on first use, and replaces a legacy
    'initial_universe' blob seed. Tickers not (yet) in the universe parquet
    go to 'ungrouped'."""
    pipeline = config["pipeline"]
    groups = pipeline.get("ticker_groups")
    if groups and set(groups) != {"initial_universe"}:
        return groups

    flat = sorted(pipeline.get("tickers", []))
    industries = industry_map()
    if not industries:
        print("  NOTE: firm_universe.parquet not found — run `make build-universe` to "
              "seed groups from the pre-named industries. Using 'ungrouped' for now.")
        pipeline["ticker_groups"] = {"ungrouped": flat}
        return pipeline["ticker_groups"]

    seeded: dict[str, list[str]] = {}
    for t in flat:
        seeded.setdefault(industries.get(t, "ungrouped"), []).append(t)
    pipeline["ticker_groups"] = {name: sorted(members) for name, members in sorted(seeded.items())}
    print(f"  Seeded {len(pipeline['ticker_groups'])} group(s) from firm_universe industry names.")
    return pipeline["ticker_groups"]


def sync_flat_list(config: dict) -> None:
    """pipeline.tickers = sorted union of all groups (what 00-03 read)."""
    groups = config["pipeline"]["ticker_groups"]
    union = sorted({t for members in groups.values() for t in members})
    config["pipeline"]["tickers"] = union


def industry_map() -> dict[str, str]:
    if not UNIVERSE_PATH.exists():
        return {}
    import pandas as pd
    df = pd.read_parquet(UNIVERSE_PATH)
    return dict(zip(df["ticker"], df["industry_group"]))


def parse_tickers(raw: str) -> list[str]:
    toks = [t.strip().upper() for t in raw.replace(",", " ").split()]
    return [t for t in toks if t]


def show_groups(config: dict) -> None:
    groups = config["pipeline"]["ticker_groups"]
    total = len(config["pipeline"]["tickers"])
    print(f"\n  Universe: {total} tickers in {len(groups)} group(s)")
    for name, members in groups.items():
        print(f"    [{name}]  {len(members)} tickers")
    print()


def show_tickers(config: dict) -> None:
    groups = config["pipeline"]["ticker_groups"]
    industries = industry_map()
    for name, members in groups.items():
        print(f"\n  [{name}] ({len(members)})")
        for t in sorted(members):
            ind = industries.get(t)
            print(f"    {t:<8}{ind if ind else ''}")
    if not industries:
        print("\n  (industries appear after `make build-universe` populates firm_universe.parquet)")
    print()


def add_group(config: dict) -> None:
    groups = config["pipeline"]["ticker_groups"]
    name = input("  New group name: ").strip().lower().replace(" ", "_")
    if not name:
        print("  Cancelled (empty name).\n")
        return
    if name in groups:
        print(f"  Group '{name}' already exists — use option 4 to add tickers to it.\n")
        return
    raw = input("  Tickers (space/comma separated): ")
    tickers = parse_tickers(raw)
    if not tickers:
        print("  Cancelled (no tickers).\n")
        return
    existing = set(config["pipeline"]["tickers"])
    dupes = [t for t in tickers if t in existing]
    groups[name] = sorted(set(tickers))
    sync_flat_list(config)
    save_config(config)
    print(f"  Added group '{name}' with {len(groups[name])} ticker(s)."
          + (f" Already in universe via other groups: {', '.join(dupes)}." if dupes else "")
          + f" Universe is now {len(config['pipeline']['tickers'])}.\n")


def add_to_group(config: dict) -> None:
    groups = config["pipeline"]["ticker_groups"]
    names = list(groups)
    for i, name in enumerate(names, 1):
        print(f"    {i}. {name} ({len(groups[name])})")
    choice = input("  Group number: ").strip()
    if not choice.isdigit() or not (1 <= int(choice) <= len(names)):
        print("  Cancelled.\n")
        return
    name = names[int(choice) - 1]
    tickers = parse_tickers(input(f"  Tickers to add to [{name}]: "))
    if not tickers:
        print("  Cancelled (no tickers).\n")
        return
    before = set(groups[name])
    groups[name] = sorted(before | set(tickers))
    sync_flat_list(config)
    save_config(config)
    added = sorted(set(tickers) - before)
    print(f"  Added {len(added)} new ticker(s) to '{name}': {', '.join(added) if added else '(all were already there)'}\n")


def main() -> None:
    config = load_config()
    ensure_groups(config)
    sync_flat_list(config)
    save_config(config)

    print("=" * 56)
    print("  Firm universe — ticker groups (configs/config.json)")
    print("=" * 56)

    while True:
        show_groups(config)
        print("  1. View all tickers (by group, with industry)")
        print("  2. Add a new group")
        print("  3. Add tickers to an existing group")
        print("  q. Quit")
        choice = input("\n  > ").strip().lower()
        if choice == "1":
            show_tickers(config)
        elif choice == "2":
            add_group(config)
        elif choice == "3":
            add_to_group(config)
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
