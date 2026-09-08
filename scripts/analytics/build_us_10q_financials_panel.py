"""
scripts/analytics/build_us_10q_financials_panel.py — firm-quarter financials
panel for the 10-Q shock series, combining two US XBRL sources:

1. Inline-XBRL per-filing facts (data/raw/xbrl_facts/us_by_filing/,
   scripts/us/04_extract_inline_xbrl_facts.py) — primary source, one value
   per (ticker, calendar quarter), EARLIEST-filed value wins (as-of safe:
   see `first_disclosed()`).
2. SEC XBRL frames (data/raw/xbrl_frames_alt/us_10q_frames.parquet,
   scripts/us/05_fetch_xbrl_frames_alt.py) — fallback ONLY for
   (ticker, quarter) cells the inline-XBRL source has no value for, and
   only rows filed within a normal quarterly-filing window (see the
   frames as-of leak fix below).

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

Fourth fix, same day, different failure mode: the frames fallback itself
wasn't as-of safe. SEC's frames API returns whatever value it currently
has cached for (tag, entity, period) - frequently a LATER filing's
comparative, not the period's own original disclosure. Joined against
filing_manifest: 54% of frames rows have a filing lag over 120 days past
period_end (median 387 days), versus the ~30-45 days a 10-Q actually has
to file. Now filtered to a 0-120-day lag before use, which shrank the
frames contribution from 0.7-3.2pp to 0.0-0.8pp per metric - most of what
it used to add was leaked future information, not genuine coverage.
inline-XBRL stays the primary source; frames is a small, now-safe
top-up.

Output: one row per (ticker, year, quarter, metric), with `source`
(inline_xbrl | frames_fallback) and `source_ref` (accession number)
so provenance is always traceable back to a specific filing.

Usage:
    uv run python scripts/analytics/build_us_10q_financials_panel.py
"""

import glob
from pathlib import Path

import duckdb
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
    "eps_diluted": ["us-gaap:EarningsPerShareDiluted", "us-gaap:EarningsPerShareBasicAndDiluted"],
    "capex": ["us-gaap:PaymentsToAcquirePropertyPlantAndEquipment", "us-gaap:PaymentsToAcquireProductiveAssets",
              "us-gaap:PaymentsToAcquireOtherPropertyPlantAndEquipment", "us-gaap:PaymentsForCapitalImprovements"],
    "assets": ["us-gaap:Assets"],
    "current_assets": ["us-gaap:AssetsCurrent"],
    "current_liabilities": ["us-gaap:LiabilitiesCurrent"],
    "equity": ["us-gaap:StockholdersEquity", "us-gaap:StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "debt": ["us-gaap:LongTermDebtNoncurrent", "us-gaap:LongTermDebt",
             "us-gaap:LongTermDebtAndCapitalLeaseObligations", "us-gaap:NotesPayable"],
}
INSTANT = {"assets", "current_assets", "current_liabilities", "equity", "debt"}
# Bank `revenue` OVERRIDE, ported from build_firm_financials.py: verified
# accounting identity, InterestIncomeExpenseNet + NoninterestIncome =
# Revenues at 0.0% difference for every bank that already tags a working
# combined Revenues concept (BAC/COF/JPM/C/PNC). FITB/ZION/CMA/SIVB/HBAN/
# RF/PBCT never tag that combined concept, only one scoped to ASC 606 fee
# income that structurally excludes net interest income, a bank's core
# revenue. NoninterestIncome is tagged by exactly 27 tickers in this
# universe, every one financial-sector — safe to apply unconditionally.
BANK_REVENUE_TAGS = ["us-gaap:InterestIncomeExpenseNet", "us-gaap:NoninterestIncome"]
ALL_TAGS = sorted({t for v in TAGS.values() for t in v} | set(BANK_REVENUE_TAGS))
# Balance-sheet concepts that can never legitimately be negative - unlike
# equity (real, from buybacks - McDonald's, Starbucks, ...), a negative
# value here is a filer sign-tagging error. Verified case ported from
# build_firm_financials.py: DuPont's 2021-02-12 10-K tags us-gaap:LongTermDebt
# as -$21.811B in the SAME filing where the sibling concept
# LongTermDebtAndCapitalLeaseObligations correctly shows +$21.806B.
NEVER_NEGATIVE_TAGS = {t for m in ("assets", "current_assets", "current_liabilities", "debt") for t in TAGS[m]}


def dimensional_singletons(dim_facts: pd.DataFrame) -> pd.DataFrame:
    """Last-resort candidates from dimensional contexts: kept ONLY where
    every dimensional fact for a (ticker, concept, start, end) cell agrees
    on one value AND that value is corroborated by >=2 observations — not
    a true multi-segment breakdown needing summation, just the
    whole-company figure filed under a dimensional context.
    Verified against GM's R&D expense (tagged with one dimensional member
    every period, 3+ corroborating observations each year, matching GM's
    actual reported R&D: $9.8B FY2022, $9.9B FY2023). The >=2-observation
    floor matters on its own: APA's RevenueFromContractWithCustomer
    ExcludingAssessedTax for FY2022 has exactly ONE dimensional
    observation ($18M, a single product/geography line, not the ~$11B
    total) — trivially "single distinct value" simply because there's
    nothing to disagree with it. A cell with 2+ DIFFERENT dimensional
    values (a real segment split, at any observation count) is still
    dropped rather than guessed at. Marked `is_dimensional=True` so it
    never outranks a non-dimensional fact for the same cell — see the
    priority sort in `main()`."""
    if dim_facts.empty:
        return dim_facts.assign(is_dimensional=True)
    keys = ["ticker", "concept", "start", "end"]
    grouped = dim_facts.groupby(keys)["numeric_value"]
    single = (grouped.transform("nunique") == 1) & (grouped.transform("size") >= 2)
    return dim_facts[single].assign(is_dimensional=True)


def first_disclosed(facts: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Collapse repeated (keys) observations across filings to the value
    from the EARLIEST filing_date — not the most recent, not the median.

    This panel feeds as-of joins: the value assigned to a period must be
    what a reader of that period's own filing could have known then, never
    a figure corrected by a later restatement. Median was tried first
    (robust to an isolated bad restatement, mirroring build_firm_financials.py's
    prior approach) and verified against DISH's 2024 post-EchoStar-merger
    10-K, which retags FY2021 revenue at ~10x what two prior, mutually
    consistent filings had already reported for the exact same period —
    median got this right only because the vote split (2 vs 1) happened to
    favor the original figure. "Earliest filing" is right regardless of
    the vote count, because it is the only rule that never looks at a
    filing later than the one whose value is being resolved.

    `is_dimensional` sorts first when present: a dimensional-singleton
    fallback (see `dimensional_singletons`) never outranks a real
    non-dimensional fact for the same cell, no matter which is older.
    """
    tiebreak = (["is_dimensional"] if "is_dimensional" in facts.columns else []) + ["filing_date"]
    return (facts
            .sort_values(keys + tiebreak)
            .drop_duplicates(keys, keep="first"))


def discrete_quarters(duration_facts: pd.DataFrame) -> pd.DataFrame:
    """One discrete-quarter value per (ticker, concept, period_end).

    Groups by (ticker, concept, period_start) — facts sharing a start are
    the same fiscal-year-to-date series. Within a group, sorted by
    period_end: the first entry is kept as-is if it already spans one
    quarter (75-100 days, true for Q1 since YTD-through-Q1 IS Q1); every
    later entry is replaced by its difference from the previous one,
    recovering the discrete quarter, but ONLY where the day-count gap
    between consecutive entries is itself 75-100 days (one real quarter,
    not a skipped filing or a fiscal-year restart). Each derived quarter
    keeps the filing_date/accession of its LATER endpoint — the moment the
    quarter's own value first became inferable, never earlier.
    """
    duration_facts = first_disclosed(duration_facts, ["ticker", "concept", "start", "end"])
    out = []
    for (ticker, concept, _start), group in duration_facts.groupby(["ticker", "concept", "start"]):
        group = group.sort_values("end").reset_index(drop=True)
        for i in range(len(group)):
            gap = group.loc[i, "days"] if i == 0 else group.loc[i, "days"] - group.loc[i - 1, "days"]
            if not (75 <= gap <= 100):
                continue
            value = group.loc[i, "numeric_value"] if i == 0 else (
                group.loc[i, "numeric_value"] - group.loc[i - 1, "numeric_value"])
            # A quarter derived by differencing a dimensional-singleton
            # endpoint (either side) is itself marked dimensional, so it
            # still loses a cross-concept priority tie to a fully
            # non-dimensional quarter.
            is_dim = bool(group.loc[i, "is_dimensional"]) or (
                i > 0 and bool(group.loc[i - 1, "is_dimensional"]))
            out.append((ticker, concept, group.loc[i, "end"], value,
                        group.loc[i, "accession_number"], group.loc[i, "filing_date"], is_dim))
    return pd.DataFrame(
        out, columns=["ticker", "concept", "end", "numeric_value", "accession_number", "filing_date",
                       "is_dimensional"])


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
        d = d[d["concept"].isin(ALL_TAGS)]
        if len(d):
            chunks.append(d)
    base = pd.concat(chunks, ignore_index=True)
    base = base[~(base["concept"].isin(NEVER_NEGATIVE_TAGS) & (base["numeric_value"] < 0))]
    base["end"] = pd.to_datetime(base["period_end"])
    base["start"] = pd.to_datetime(base["period_start"])
    base["days"] = (base["end"] - base["start"]).dt.days
    base["cal_q"] = base["end"].dt.year * 10 + base["end"].dt.quarter
    base = base[base["cal_q"].between(cal_q_min, cal_q_max)]
    base["filing_date"] = pd.to_datetime(base["filing_date"])
    base = pd.concat([base[~base["has_dimensions"]].assign(is_dimensional=False),
                       dimensional_singletons(base[base["has_dimensions"]])], ignore_index=True)

    # SEC's frames API returns whatever value it currently has cached for
    # (tag, entity, period) - which is frequently a LATER filing's
    # comparative, not the period's own original disclosure: joined against
    # filing_manifest, 54% of frames rows have a filing lag over 120 days
    # past period_end (median 387 days) versus the ~30-45 days a 10-Q
    # actually has to file. Keeping those would reintroduce exactly the
    # restatement leakage the rest of this script exists to avoid, for a
    # fallback source that only ever contributes ~1pp of coverage. Only
    # frames rows filed within a normal quarterly-filing window are kept.
    con = duckdb.connect("duckdb/thesis.duckdb", read_only=True)
    manifest = con.execute("""
        SELECT accession_number, filing_date FROM filing_manifest
        UNION ALL
        SELECT accession_number, filing_date FROM filing_manifest_10q
    """).fetchdf()
    con.close()
    frames = pd.read_parquet(f"{config['storage']['raw_xbrl_frames_alt']}/us_10q_frames.parquet")
    frames["cal_q"] = frames["year"] * 10 + frames["quarter"]
    frames = frames[frames["cal_q"].between(cal_q_min, cal_q_max)]
    frames = frames.merge(manifest.drop_duplicates("accession_number"), on="accession_number", how="left")
    frames["period_end"] = pd.to_datetime(frames["period_end"])
    frames["filing_date"] = pd.to_datetime(frames["filing_date"])
    lag_days = (frames["filing_date"] - frames["period_end"]).dt.days
    frames = frames[lag_days.between(0, 120)]

    duration_tags = sorted({t for m, tags in TAGS.items() if m not in INSTANT for t in tags}
                           | set(BANK_REVENUE_TAGS))
    dq = discrete_quarters(base[base["concept"].isin(duration_tags) & base["start"].notna()])
    dq["cal_q"] = dq["end"].dt.year * 10 + dq["end"].dt.quarter
    dq = dq[dq["cal_q"].between(cal_q_min, cal_q_max)]

    # Bank revenue OVERRIDE: one row per (ticker, cal_q) where BOTH
    # components exist, summed — see BANK_REVENUE_TAGS above. Provenance
    # (accession) taken from whichever component was filed later, the
    # moment the sum itself became knowable (same reasoning as
    # discrete_quarters's own endpoint choice).
    bank_parts = first_disclosed(dq[dq["concept"].isin(BANK_REVENUE_TAGS)], ["ticker", "concept", "cal_q"])
    bank_wide = bank_parts.pivot_table(index=["ticker", "cal_q"], columns="concept",
                                        values="numeric_value", aggfunc="first").dropna()
    bank_ref = (bank_parts.sort_values("filing_date")
                          .drop_duplicates(["ticker", "cal_q"], keep="last")
                          .set_index(["ticker", "cal_q"])["accession_number"])
    bank_revenue = bank_wide.sum(axis=1).rename("value").to_frame().join(bank_ref).reset_index()

    instant_tags = sorted({t for m, tags in TAGS.items() if m in INSTANT for t in tags})
    inst = first_disclosed(base[base["concept"].isin(instant_tags)], ["ticker", "concept", "end"])

    rows_out = []
    for metric, tags in TAGS.items():
        sub = (inst if metric in INSTANT else dq)
        sub = sub[sub["concept"].isin(tags)].copy()
        # One value per (ticker, cal_q): collapse repeats of the SAME
        # concept to its own earliest filing first (a concept can still
        # produce >1 candidate for the same cal_q — e.g. two different
        # YTD-anchor groups both landing a discrete quarter on the same
        # period_end), then resolve remaining cross-concept conflicts by
        # DATE FIRST, priority only as a same-date tie-break — never
        # priority regardless of date. Two distinct failure modes taught
        # this order:
        #   - Capital One tags BOTH us-gaap:Revenues (true total,
        #     ~$9-15B/quarter) and RevenueFromContractWithCustomer
        #     ExcludingAssessedTax (non-interest income subset,
        #     ~$1.2-1.6B/quarter) in the SAME original filing every
        #     quarter — priority correctly picks Revenues here because
        #     both dates tie.
        #   - Iron Mountain's original Q1 2022 10-Q tagged ONLY
        #     RevenueFromContractWithCustomerExcludingAssessedTax
        #     ($497M); us-gaap:Revenues for that same quarter appears for
        #     the first time over a YEAR later, in 2023, as a restated
        #     comparative at ~2.5x the original figure. Priority-first
        #     would have picked the higher-ranked "Revenues" concept and
        #     pulled in that later restatement even though it postdates
        #     the quarter by over a year. Date-first correctly keeps the
        #     $497M value that was actually knowable in 2022.
        # `is_dimensional` sorts before filing_date here too: a
        # dimensional-singleton fallback (see `dimensional_singletons`)
        # never outranks a real non-dimensional value from a DIFFERENT
        # concept just because it happens to have an earlier date.
        priority = {concept: rank for rank, concept in enumerate(tags)}
        sub["priority"] = sub["concept"].map(priority)
        sub = first_disclosed(sub, ["ticker", "concept", "cal_q"])
        sub = (sub.sort_values(["ticker", "cal_q", "is_dimensional", "filing_date", "priority"])
                  .drop_duplicates(["ticker", "cal_q"], keep="first"))
        sub = sub[["ticker", "cal_q", "numeric_value", "accession_number"]].rename(
            columns={"numeric_value": "value", "accession_number": "source_ref"})
        sub["source"] = "inline_xbrl"
        if metric == "revenue" and not bank_revenue.empty:
            # OVERRIDE, not fill: for banks that only tag the ASC-606
            # fee-income-scoped concept (understates revenue by ~10x —
            # e.g. FITB), the verified net-interest+noninterest sum
            # replaces it even though `sub` already has a (wrong) value.
            sub = sub.merge(bank_revenue, on=["ticker", "cal_q"], how="outer", suffixes=("", "_bank"))
            has_bank = sub["value_bank"].notna()
            sub["value"] = sub["value_bank"].where(has_bank, sub["value"])
            sub["source_ref"] = sub["accession_number"].where(has_bank, sub["source_ref"])
            sub["source"] = sub["source"].fillna("inline_xbrl")
            sub = sub.drop(columns=["value_bank", "accession_number"])

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
