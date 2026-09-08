"""
scripts/analytics/build_us_10q_financials_panel.py — firm-quarter financials
panel for the 10-Q shock series, combining two US XBRL sources:

1. Inline-XBRL per-filing facts (data/raw/xbrl_facts/us_by_filing/,
   scripts/us/04_extract_inline_xbrl_facts.py) — primary source, one value
   per (ticker, calendar quarter), most-recently-filed value wins.
2. SEC XBRL frames (data/raw/xbrl_frames_alt/us_10q_frames.parquet,
   scripts/us/05_fetch_xbrl_frames_alt.py) — fallback ONLY for
   (ticker, quarter) cells the inline-XBRL source has no value for.

Coverage bug fixed here (2026-09-08): the inline-XBRL extractor keeps every
duration fact tagged in a filing, which includes YTD and full-year
comparatives under the SAME tag as the single-quarter figure (e.g. a 10-Q
tags both the 3-month and 6-month-YTD Revenues under us-gaap:Revenues).
Counting any of those as "covering" the calendar quarter their period_end
falls in overstates true single-quarter coverage substantially.

Second fix, same day, same root cause: MANY filers never tag a standalone
discrete-quarter duration fact at all for cash-flow-statement lines —
capex (`PaymentsToAcquirePropertyPlantAndEquipment`) is the extreme case,
verified against AAPL, where every single observed duration fact is a
fiscal-year-to-date cumulative (90/181/272/363 days), never a bare
~90-day quarter. Filtering to 75-100-day facts alone therefore only ever
caught each firm's Q1 (which happens to be both YTD and discrete) and
silently dropped Q2-Q4 for any filer using this convention. `discrete_quarters()`
recovers them by differencing consecutive YTD facts that share the same
period_start (the fiscal-year anchor): Q2 = YTD_Q2 - YTD_Q1, Q3 = YTD_Q3 -
YTD_Q2, Q4 = FY - YTD_Q3. Differenced values were spot-checked against
AAPL's known quarterly capex and matched. This took capex from ~32% to
100% of the realistic coverage ceiling (see docs/sources/accounting_data.md
for the ceiling methodology and full per-metric numbers) and gave
double-digit gains to operating_income, sga_expense and cogs too.
`rd_expense` barely moved — most filers who omit a discrete R&D figure
also never disclose an R&D breakout in the YTD cash-flow/income statement
either, so there is nothing to difference.

The frames fallback recovers a further ~1pp per metric on top of that —
real but marginal next to the two fixes above; inline-XBRL stays the
primary source.

Output: one row per (ticker, year, quarter, metric), with `source`
(inline_xbrl | frames_fallback) and `source_ref` (accession number)
so provenance is always traceable back to a specific filing.

Usage:
    uv run python scripts/analytics/build_us_10q_financials_panel.py
"""

import glob
from pathlib import Path

import pandas as pd
import yaml

TAGS = {
    "revenue": ["us-gaap:Revenues", "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
                "us-gaap:RevenueFromContractWithCustomerIncludingAssessedTax", "us-gaap:SalesRevenueNet",
                "us-gaap:SalesRevenueGoodsNet"],
    "cogs": ["us-gaap:CostOfRevenue", "us-gaap:CostOfGoodsAndServicesSold", "us-gaap:CostOfGoodsSold",
             "us-gaap:CostOfServices"],
    "rd_expense": ["us-gaap:ResearchAndDevelopmentExpense",
                   "us-gaap:ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost",
                   "us-gaap:ResearchAndDevelopmentExpenseSoftwareExcludingAcquiredInProcessCost",
                   "ifrs-full:ResearchAndDevelopmentExpense"],
    "sga_expense": ["us-gaap:SellingGeneralAndAdministrativeExpense", "us-gaap:GeneralAndAdministrativeExpense"],
    "operating_income": ["us-gaap:OperatingIncomeLoss"],
    "net_income": ["us-gaap:NetIncomeLoss", "us-gaap:ProfitLoss"],
    "eps_diluted": ["us-gaap:EarningsPerShareDiluted"],
    "capex": ["us-gaap:PaymentsToAcquirePropertyPlantAndEquipment", "us-gaap:PaymentsToAcquireProductiveAssets"],
    "assets": ["us-gaap:Assets"],
    "current_assets": ["us-gaap:AssetsCurrent"],
    "current_liabilities": ["us-gaap:LiabilitiesCurrent"],
    "equity": ["us-gaap:StockholdersEquity", "us-gaap:StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "debt": ["us-gaap:LongTermDebtNoncurrent", "us-gaap:LongTermDebt",
             "us-gaap:LongTermDebtAndCapitalLeaseObligations", "us-gaap:NotesPayable"],
}
INSTANT = {"assets", "current_assets", "current_liabilities", "equity", "debt"}
ALL_TAGS = sorted({t for v in TAGS.values() for t in v})


def discrete_quarters(duration_facts: pd.DataFrame) -> pd.DataFrame:
    """One discrete-quarter value per (ticker, concept, period_end).

    Groups by (ticker, concept, period_start) — facts sharing a start are
    the same fiscal-year-to-date series. Within a group, sorted by
    period_end: the first entry is kept as-is if it already spans one
    quarter (75-100 days, true for Q1 since YTD-through-Q1 IS Q1); every
    later entry is replaced by its difference from the previous one,
    recovering the discrete quarter, but ONLY where the day-count gap
    between consecutive entries is itself 75-100 days (one real quarter,
    not a skipped filing or a fiscal-year restart).
    """
    duration_facts = (duration_facts
                       .sort_values("filing_date")
                       .drop_duplicates(["ticker", "concept", "start", "end"], keep="last"))
    out = []
    for (ticker, concept, _start), group in duration_facts.groupby(["ticker", "concept", "start"]):
        group = group.sort_values("end").reset_index(drop=True)
        for i in range(len(group)):
            gap = group.loc[i, "days"] if i == 0 else group.loc[i, "days"] - group.loc[i - 1, "days"]
            if not (75 <= gap <= 100):
                continue
            value = group.loc[i, "numeric_value"] if i == 0 else (
                group.loc[i, "numeric_value"] - group.loc[i - 1, "numeric_value"])
            out.append((ticker, concept, group.loc[i, "end"], value,
                        group.loc[i, "accession_number"], group.loc[i, "filing_date"]))
    return pd.DataFrame(
        out, columns=["ticker", "concept", "end", "numeric_value", "accession_number", "filing_date"])


def main():
    with open("configs/us/config.yaml") as f:
        config = yaml.safe_load(f)

    q_from = config["corpus"]["filings_10q"]["filing_date"]["from"]
    q_to = config["corpus"]["filings_10q"]["filing_date"]["to"]
    cal_q_min = int(q_from[:4]) * 10 + 1
    cal_q_max = int(q_to[:4]) * 10 + ((int(q_to[5:7]) - 1) // 3 + 1)

    files = glob.glob(f"{config['storage']['raw_xbrl_facts'].replace('/us', '/us_by_filing')}/*.parquet")
    chunks = []
    for f in files:
        d = pd.read_parquet(f, columns=["ticker", "accession_number", "filing_date", "has_dimensions",
                                         "concept", "numeric_value", "period_start", "period_end"])
        d = d[(~d["has_dimensions"]) & (d["concept"].isin(ALL_TAGS))]
        if len(d):
            chunks.append(d)
    base = pd.concat(chunks, ignore_index=True)
    base["end"] = pd.to_datetime(base["period_end"])
    base["start"] = pd.to_datetime(base["period_start"])
    base["days"] = (base["end"] - base["start"]).dt.days
    base["cal_q"] = base["end"].dt.year * 10 + base["end"].dt.quarter
    base = base[base["cal_q"].between(cal_q_min, cal_q_max)]
    base["filing_date"] = pd.to_datetime(base["filing_date"])

    frames = pd.read_parquet(f"{config['storage']['raw_xbrl_frames_alt']}/us_10q_frames.parquet")
    frames["cal_q"] = frames["year"] * 10 + frames["quarter"]
    frames = frames[frames["cal_q"].between(cal_q_min, cal_q_max)]

    duration_tags = sorted({t for m, tags in TAGS.items() if m not in INSTANT for t in tags})
    dq = discrete_quarters(base[base["concept"].isin(duration_tags) & base["start"].notna()])
    dq["cal_q"] = dq["end"].dt.year * 10 + dq["end"].dt.quarter
    dq = dq[dq["cal_q"].between(cal_q_min, cal_q_max)]

    rows_out = []
    for metric, tags in TAGS.items():
        sub = (base if metric in INSTANT else dq)
        sub = sub[sub["concept"].isin(tags)].copy()
        sub = sub.sort_values("filing_date").drop_duplicates(["ticker", "cal_q"], keep="last")
        sub = sub[["ticker", "cal_q", "numeric_value", "accession_number"]].rename(
            columns={"numeric_value": "value", "accession_number": "source_ref"})
        sub["source"] = "inline_xbrl"

        fsub = frames[frames["metric"] == metric][["ticker", "cal_q", "value", "accession_number"]].rename(
            columns={"accession_number": "source_ref"})
        fsub = fsub.drop_duplicates(["ticker", "cal_q"])

        have = set(map(tuple, sub[["ticker", "cal_q"]].values))
        fsub_gap = fsub[~fsub.apply(lambda r: (r["ticker"], r["cal_q"]) in have, axis=1)].copy()
        fsub_gap["source"] = "frames_fallback"

        combined = pd.concat([sub, fsub_gap], ignore_index=True)
        combined["metric"] = metric
        rows_out.append(combined)

    panel = pd.concat(rows_out, ignore_index=True)
    panel["year"] = panel["cal_q"] // 10
    panel["quarter"] = panel["cal_q"] % 10
    panel = panel.drop(columns=["cal_q"])

    out_path = Path("data/processed/us_10q_financials_panel.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(out_path, index=False)
    print(f"Done. {len(panel)} rows -> {out_path}")


if __name__ == "__main__":
    main()
