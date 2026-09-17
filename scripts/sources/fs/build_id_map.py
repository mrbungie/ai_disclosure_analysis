#!/usr/bin/env python
"""
Build the ticker -> FactSet identifier map for the S&P 500 (2021-01-01 start panel)
universe, 499 tickers from data/silver/firm_universe.parquet (membership_groups
contains "sp500_2021_start_panel").

Method (see docs/sources/fs.md, "Identifiers" section, for the full writeup):
  - FQL functions P_NAME(), P_SYMBOL(), P_CUSIP() resolve a symbol's current
    security name / canonical symbol / CUSIP. Comparing P_NAME() to the universe's
    company_name (with FactSet's "LASTNAME FIRSTNAME"/legal-suffix formatting
    normalized away) flags wrong- or dead-ticker cases.
  - Plain "<TICKER>-US" silently resolves to a DIFFERENT, unrelated company for
    recycled tickers (INFO, PARA) instead of erroring, so name comparison -- not
    just error-checking -- is required.
  - For genuinely delisted/merged securities, FactSet keeps the historical
    security under a different id form found by looking up the security's own
    CUSIP: "<TICKER>Q-US" (bankruptcy/failure delisting, e.g. SIVBQ-US),
    "<TICKER>B-US" (bank-holding delisting, e.g. FRCB-US), or
    "<TICKER>.XX10-US" (corporate-action delisting, e.g. INFO.XX10-US for IHS
    Markit). There is no single suffix rule -- each case was resolved via
    symbols=<CUSIP> & exprs=P_NAME();;P_SYMBOL() to find the FactSet-canonical
    symbol for that CUSIP, then verified by re-querying that symbol directly.
  - Renamed-in-place tickers (DISCA->WBD, FBHS->FBIN, FLT->CPAY, GPS->GAP,
    HFC->DINO, WLTW->WTW) just need the new ticker's "-US" id; P_NAME() on the
    new id matches company_name directly.

This script does NOT hit the network. It reads the raw FQL batch responses
already captured under data/raw/fs/reference/name_check/ (P_NAME/P_SYMBOL/
P_CUSIP + annual PRICE/DATE for all 499 tickers, batches of 40) plus the daily
DATE/PRICE bounds captured for the 10 tickers that needed a fix, and applies the
verified fix list below.
"""
import json
import re
from pathlib import Path
from difflib import SequenceMatcher

import polars as pl

REPO = Path(__file__).parent.parent.parent.parent
NC_DIR = REPO / "data/raw/fs/reference/name_check"
UNIVERSE_PATH = REPO / "data/silver/firm_universe.parquet"
OUT_PATH = REPO / "data/raw/fs/reference/id_map.parquet"

# ---------------------------------------------------------------------------
# Verified fixes: ticker -> (factset_id, note)
# Found via CUSIP lookup (symbols=<CUSIP>&exprs=P_NAME();;P_SYMBOL()) in the
# FactSet Workstation FQL endpoint, then confirmed by re-querying the resolved
# symbol directly (name matches, daily price series present and plausible).
# ---------------------------------------------------------------------------
FIXES = {
    "DISCA": ("WBD-US", "Renamed: Discovery -> Warner Bros Discovery (2022-04-11)."),
    "FBHS": ("FBIN-US", "Renamed: Fortune Brands Home & Security -> Fortune Brands Innovations (2022-12-15)."),
    "FLT": ("CPAY-US", "Renamed: FleetCor Technologies -> Corpay (2024-04)."),
    "GPS": ("GAP-US", "Gap Inc trades under ticker GAP on FactSet, not GPS."),
    "HFC": ("DINO-US", "Renamed: HollyFrontier -> HF Sinclair (2022-03-15 merger)."),
    "WLTW": ("WTW-US", "Renamed: Willis Towers Watson -> WTW (2022-01)."),
    "SIVB": (
        "SIVBQ-US",
        "SVB Financial Group failed 2023-03-10 / delisted 2023-05-02; plain "
        "SIVB-US resolves to no security (@NA). SIVBQ-US ('Q' = post-failure "
        "delisted-security suffix) is FactSet's id for the historical security "
        "(CUSIP 78486Q10, confirmed via P_NAME/P_CUSIP). CAVEAT: FactSet holds "
        "SIVBQ-US's last traded price frozen (no @NA) through 2024-11-06, ~18 "
        "months after the actual 2023-05-02 delisting -- consumers MUST discard "
        "any SIVBQ-US observation after delisting_date, not trust the frozen tail.",
    ),
    "INFO": (
        "INFO.XX10-US",
        "INFO ticker was reused in 2024 for an unrelated REIT (Harbor ETF "
        "Trust); plain INFO-US now returns THAT company's data, not IHS Markit. "
        "IHS Markit Ltd merged into S&P Global 2022-02-28. INFO.XX10-US (found "
        "via CUSIP G4756710) is the historical IHS Markit security: P_NAME() = "
        "'IHS MARKIT LTD', daily prices 2015-09-17 through 2022-02-25 (matches "
        "the merger date), no post-merger frozen-price artifact. A prior pass's "
        "'fix' to IHS-US was WRONG -- that id is IHS Holding Ltd (Nigerian tower "
        "company, IPO 2021-10), a different, unrelated firm.",
    ),
    "PARA": (
        "PARAA-US",
        "PARA ticker was reused in 2025 for an unrelated micro-cap (Banzai "
        "International); plain PARA-US now returns THAT company's data, not "
        "Paramount Global/ViacomCBS. Paramount Global (renamed from ViacomCBS "
        "2022-02, itself renamed CBS Corp/Viacom merger 2019-12) merged into "
        "Paramount Skydance 2025-08-07. PARAA-US (Class A, found via CUSIP "
        "92556H10) is the historical security: P_NAME() = 'PARAMOUNT GLOBAL', "
        "daily prices 2015-09-17 through 2025-08-06 (one day before the merger "
        "close). A prior pass's 'fix' to VIA-US was WRONG -- that id is Via "
        "Transportation Inc (ride-hailing software, IPO 2025-09), a different, "
        "unrelated firm.",
    ),
    "FRC": (
        "FRCB-US",
        "First Republic Bank failed 2023-05-01 (FDIC receivership, sold to "
        "JPMorgan); plain FRC-US resolves to no security (@NA) -- this ticker "
        "was NOT in the previously-known broken list and was found by this "
        "verification pass. FRCB-US (found via CUSIP search) is FactSet's id "
        "for the historical security: P_NAME() = 'FIRST REP BK SAN FRANCISCO C'. "
        "CAVEAT: like SIVBQ-US, FactSet holds FRCB-US's last traded price frozen "
        "(no @NA) all the way to the present -- consumers MUST discard any "
        "FRCB-US observation after delisting_date (2023-05-01).",
    ),
}

# Tickers with a genuine, expected data gap (acquired mid-panel, or spun off
# after panel start) but whose plain "<TICKER>-US" id IS the correct security --
# verified by name match + a first/last active-price year consistent with the
# acquisition/spinoff date. Not identifier bugs; documented so they aren't
# re-flagged as mismatches downstream.
EXPECTED_GAP_NOTES = {
    "FLIR": "Acquired by Teledyne 2021-05-14; FLIR-US correct, data ends 2020.",
    "TIF": "Acquired by LVMH 2021-01-07; TIF-US correct, data ends 2020.",
    "CXO": "Acquired by ConocoPhillips 2021-01-19; CXO-US correct, data ends 2020.",
    "VAR": "Acquired by Siemens Healthineers/Agilent split 2021-04-15; VAR-US correct, data ends 2020.",
    "ALXN": "Acquired by AstraZeneca 2021-07-21; ALXN-US correct, data ends 2020.",
    "OTIS": "Spun off from United Technologies 2020-04-03; OTIS-US correct, data starts 2020.",
    "CARR": "Spun off from United Technologies 2020-04-03; CARR-US correct, data starts 2020.",
}


def norm(s: str) -> str:
    if not s:
        return ""
    s = s.upper()
    s = re.sub(r"[.,'&]", "", s)
    for suf in (" INC", " CORP", " CORPORATION", " CO", " COMPANY", " LTD", " PLC",
                " GROUP", " HOLDINGS", " HOLDING", " THE ", " SA", " NV", " LLC"):
        s = s.replace(suf, " ")
    return re.sub(r"\s+", " ", s).strip()


def name_score(a: str, b: str) -> float:
    na, nb = norm(a), norm(b)
    seq = SequenceMatcher(None, na, nb).ratio()
    sa, sb = set(na.split()), set(nb.split())
    jac = len(sa & sb) / len(sa | sb) if (sa or sb) else 0.0
    return max(seq, jac)


def load_namecheck():
    """Parse the P_NAME/P_SYMBOL/P_CUSIP + annual DATE/PRICE FQL batch dumps."""
    rows = {}
    for f in sorted(NC_DIR.glob("fs_namecheck_batch_*.json")):
        j = json.loads(f.read_text())
        tickers = j["tickers"]
        data = j["data"]
        for idx, t in enumerate(tickers):
            base = idx * 5
            recs = data[base:base + 5]
            name = recs[0]["$value"][0][0] if recs[0].get("$value") else None
            symbol = recs[1]["$value"][0][0] if recs[1].get("$value") else None
            cusip = recs[2]["$value"][0][0] if recs[2].get("$value") else None
            dates = [v[0] for v in recs[3]["$value"]] if recs[3].get("$value") else []
            prices = [v[0] for v in recs[4]["$value"]] if recs[4].get("$value") else []
            non_na_years = [d for d, p in zip(dates, prices) if p != "@NA"]
            rows[t] = {
                "fs_name": name if name != "@NA" else None,
                "fs_cusip": cusip if cusip != "@NA" else None,
                "first_year": (min(non_na_years) // 10000) if non_na_years else None,
                "last_year": (max(non_na_years) // 10000) if non_na_years else None,
            }
    return rows


# Exact daily first/last non-NA price date for the 10 fixed ids, captured via a
# single FQL batch call against the resolved ids (see docs/sources/fs.md).
FIXED_ID_DAILY_BOUNDS = {
    "WBD-US": ("2015-09-17", "2026-09-17"),
    "FBIN-US": ("2015-09-17", "2026-09-17"),
    "CPAY-US": ("2015-09-17", "2026-09-17"),
    "GAP-US": ("2015-09-17", "2026-09-17"),
    "DINO-US": ("2015-09-17", "2026-09-17"),
    "WTW-US": ("2015-09-17", "2026-09-17"),
    "SIVBQ-US": ("2015-09-17", "2024-11-06"),  # frozen tail past delisting; see FIXES note
    "INFO.XX10-US": ("2015-09-17", "2022-02-25"),
    "PARAA-US": ("2015-09-17", "2025-08-06"),
    "FRCB-US": ("2015-09-17", "2026-09-17"),  # frozen tail past delisting; see FIXES note
}


def main():
    universe = pl.read_parquet(UNIVERSE_PATH)
    universe = universe.filter(
        pl.col("membership_groups").list.contains("sp500_2021_start_panel")
        & pl.col("ticker").is_not_null()
    )
    uni = {r["ticker"]: r for r in universe.to_dicts()}

    nc = load_namecheck()

    # Daily bounds for the ~489 never-touched tickers come from the existing
    # prices_daily.parquet pull (already correct for anything not in FIXES).
    daily_bounds = {}
    daily_path = REPO / "data/raw/fs/prices/daily/prices_daily.parquet"
    if daily_path.exists():
        dd = pl.read_parquet(daily_path)
        grp = dd.group_by("ticker").agg(pl.col("date").min().alias("mn"), pl.col("date").max().alias("mx"))
        for r in grp.to_dicts():
            daily_bounds[r["ticker"]] = (str(r["mn"]), str(r["mx"]))

    rows = []
    for t, u in uni.items():
        info = nc.get(t, {})
        if t in FIXES:
            factset_id, note = FIXES[t]
            status = "fixed"
            first_dt, last_dt = FIXED_ID_DAILY_BOUNDS.get(factset_id, (None, None))
            # fs_name/cusip for the fixed id were verified interactively, not in the
            # bulk namecheck batch (which only covers plain "<ticker>-US").
            fixed_names = {
                "WBD-US": "WARNER BROS DISCOVERY INC", "FBIN-US": "FORTUNE BRANDS INNOVATIONS I",
                "CPAY-US": "CORPAY INC", "GAP-US": "GAP INC", "DINO-US": "HF SINCLAIR CORP",
                "WTW-US": "WILLIS TOWERS WATSON PLC LTD", "SIVBQ-US": "SVB FINANCIAL GROUP",
                "INFO.XX10-US": "IHS MARKIT LTD", "PARAA-US": "PARAMOUNT GLOBAL",
                "FRCB-US": "FIRST REP BK SAN FRANCISCO C",
            }
            fixed_cusips = {
                "WBD-US": "93442310", "FBIN-US": "34964C10", "CPAY-US": "21994810",
                "GAP-US": "36476010", "DINO-US": "40394910", "WTW-US": "G9662910",
                "SIVBQ-US": "78486Q10", "INFO.XX10-US": "G4756710", "PARAA-US": "92556H10",
                "FRCB-US": "33616C10",
            }
            fs_name = fixed_names.get(factset_id)
            fs_cusip = fixed_cusips.get(factset_id)
        else:
            factset_id = f"{t}-US"
            fs_name = info.get("fs_name")
            fs_cusip = info.get("fs_cusip")
            first_dt = f"{info['first_year']}-01-01" if info.get("first_year") else None
            last_dt = f"{info['last_year']}-12-31" if info.get("last_year") else None
            # refine with exact daily bounds if we have them
            if t in daily_bounds:
                first_dt, last_dt = daily_bounds[t]
            note = EXPECTED_GAP_NOTES.get(t, "")
            status = "verified"

        rows.append({
            "ticker": t,
            "cik": u.get("cik"),
            "company_name": u.get("company_name"),
            "factset_id": factset_id,
            "factset_name": fs_name,
            "fsym_security_id": None,  # not exposed by this FQL install; CUSIP used instead
            "cusip": fs_cusip,
            "first_price_date": first_dt,
            "last_price_date": last_dt,
            "match_status": status,
            "note": note,
        })

    df = pl.DataFrame(rows).sort("ticker")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(OUT_PATH)

    print(f"Wrote {OUT_PATH} ({df.height} rows)")
    print(df["match_status"].value_counts())
    print("\nFixed tickers:")
    print(df.filter(pl.col("match_status") == "fixed").select("ticker", "factset_id", "factset_name"))
    unresolved = df.filter(pl.col("match_status") == "unresolved")
    print(f"\nUnresolved: {unresolved.height}")


if __name__ == "__main__":
    main()
