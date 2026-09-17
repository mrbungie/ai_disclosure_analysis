#!/usr/bin/env python
"""
Build a thesis-ready WIDE extract of FactSet Standardized (STND) fundamentals: one row
per (factset_id, period_end, frequency), one column per thesis field, from the long-format
`fundamentals_long.parquet` produced by `parse_fundamentals.py`.

Only the STND basis is used (see docs/sources/fs.md "Field mapping" -- STND is the
common chart of accounts, comparable across all 499 tickers; ARPT row labels are
one-off per company and not used here).

Usage: uv run python scripts/sources/fs/build_fundamentals_wide.py
Rerunnable: re-reads fundamentals_long.parquet each time, so rerun parse_fundamentals.py
first if new raw files landed.

Output (one row per factset_id x period_end, i.e. per fiscal period):
  data/raw/fs/fundamentals/fundamentals_stnd_wide_annual.parquet
  data/raw/fs/fundamentals/fundamentals_stnd_wide_quarterly.parquet

Columns: ticker, factset_id, period_end (date), fiscal_label (e.g. "FY2025", "FY2025Q3"
-- derived below, NOT read off any FactSet label directly), period_status, then one
column per thesis field:
  revenue, cost_of_revenue, gross_profit, rd_expense, sga_expense, operating_income,
  ebitda, da, interest_expense, pretax_income, tax_expense, net_income, eps_diluted,
  total_assets, current_assets, current_liabilities, cash, total_debt, long_term_debt,
  equity, capex, shares_out
then revenue_basis ("sales" | "bank_nii_plus_nonii" | null): which STND
construct `revenue` was built from -- see build_revenue() and
docs/plans/fs_gold_coverage_fills.md, root cause 1.

`cash`, `sga_expense`, `interest_expense` are each coalesced across a small
set of industry-template STND label variants (see FIELD_MAP comments and
docs/plans/fs_gold_coverage_fills.md, root causes 2-3) -- same construct,
different label per industry chart of accounts, not a definitional change.
`current_assets`/`current_liabilities` are deliberately NOT extended with a
fallback: depositories, insurers, REITs, brokers and holding companies file
unclassified balance sheets with no current/non-current split to recover
(root cause 4) -- a null there for those industries is correct.

Units: all dollar-amount fields are in millions of USD (scale-normalized from FactSet's
per-row 'millions'/'billions' footnote -- see SCALE_MULT). eps_diluted is per-share (not
scaled). shares_out is in millions of shares. FactSet SIGN CONVENTIONS are preserved
as-is (not flipped to a thesis convention): capex and D&A/interest/tax rows on the CF/
INC statements come through FactSet's STND chart of accounts with whatever sign FactSet
assigns there -- see the "Sign conventions" printout below and docs/sources/fs.md.
total_debt is not a native STND row; it is computed here as
  ST Debt & Curr. Portion LT Debt (FF_DEBT_ST) + Long-Term Debt (FF_DEBT_LT)
(see docs/sources/fs.md "Field mapping").

fiscal_label derivation: FactSet's own column labels are calendar month/year only (e.g.
"27 SEP '25"), not a fiscal quarter number. This script infers each id's fiscal-year-end
month from its own STND annual periods (the mode of INC_ANN/BAL_ANN period_end months),
then labels every period (annual or quarterly) by how many months before that
fiscal-year-end month it falls: quarter = 4 - ((fye_month - month) % 12) // 3, fiscal_year
= period_end.year + (1 if month > fye_month else 0). Verified against AAPL (fye_month=9):
Sep'25->FY2025(Q4), Jun'25->FY2025Q3, Mar'25->FY2025Q2, Dec'24->FY2025Q1 -- matches AAPL's
known fiscal calendar.
"""

from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).parent.parent.parent.parent
FUND_LONG = REPO_ROOT / "data" / "raw" / "fs" / "fundamentals" / "fundamentals_long.parquet"
OUT_ANNUAL = REPO_ROOT / "data" / "raw" / "fs" / "fundamentals" / "fundamentals_stnd_wide_annual.parquet"
OUT_QUARTERLY = REPO_ROOT / "data" / "raw" / "fs" / "fundamentals" / "fundamentals_stnd_wide_quarterly.parquet"

SCALE_MULT = {"billions": 1e9, "millions": 1e6, None: 1.0}

# thesis field -> (report, exact STND row label, field_id filter or None, value kind)
# value kind: "amount" (scale-normalized to millions via the table's footnote scale word) |
# "per_share" (left as-is) | "shares" (left as-is -- see note on shares_out below)
FIELD_MAP = {
    # NOTE: "revenue" is NOT extracted through this generic loop -- see
    # BANK_REVENUE_LABELS / build_revenue() below. It's kept here (commented
    # in spirit by the dedicated function) purely so this dict stays the
    # single place documenting every thesis field's primary STND source.
    "revenue":             ("INC", "Sales", None, "amount"),
    "cost_of_revenue":     ("INC", "Cost of Goods Sold (COGS) incl. D&A", None, "amount"),
    "gross_profit":        ("INC", "Gross Income", None, "amount"),
    "rd_expense":          ("INC", "Research & Development", None, "amount"),
    # sga_expense: coalesced across industry-template label variants
    # (docs/plans/fs_gold_coverage_fills.md, root cause 3) -- REITs,
    # insurers, brokers and holding companies (sic2 62/63/67/70/73) report
    # "Selling, General & Admin. Expenses" or "...Expenses & Other" instead
    # of the plain "SG&A Expense" label most filers use. Same line item,
    # not a different construct -- safe to coalesce like shares_out's label
    # variants below. is_in()+mean is safe here because a single filer's
    # STND report only ever populates ONE of these three labels per period
    # (they are alternate labels for the same industry template, never
    # concurrent rows with independently differing values -- verified
    # against WRB/PSA/AMT/NDAQ raw STND rows), so the mean never averages
    # across two live values.
    "sga_expense":         ("INC", ["SG&A Expense", "Selling, General & Admin. Expenses",
                                     "Selling, General & Admin. Expenses & Other"], None, "amount"),
    "operating_income":    ("INC", "EBIT (Operating Income)", None, "amount"),
    "ebitda":              ("INC", "EBITDA", None, "amount"),
    "da":                  ("INC", "Depreciation & Amortization Expense", None, "amount"),
    # interest_expense: coalesced across industry-template label variants
    # (docs/plans/fs_gold_coverage_fills.md, root cause 2) -- depositories
    # (sic2 60/61) report "Total Interest Expense"; insurers/holding
    # companies (63/64/67) report "Interest Expense (excl. Interest
    # Capitalized)" or "...Net of Interest Capitalized" instead of the
    # plain "Interest Expense" label. Same is_in()+mean safety note as
    # sga_expense above: mutually exclusive per filer's template.
    "interest_expense":    ("INC", ["Interest Expense", "Total Interest Expense",
                                     "Interest Expense (excl. Interest Capitalized)",
                                     "Interest Expense, Net of Interest Capitalized"], None, "amount"),
    "pretax_income":       ("INC", "Pretax Income", None, "amount"),
    "tax_expense":         ("INC", "Income Taxes", None, "amount"),
    "net_income":          ("INC", "Net Income", None, "amount"),
    "eps_diluted":         ("INC", "EPS (diluted)", None, "per_share"),
    "total_assets":        ("BAL", "Total Assets", None, "amount"),
    # current_assets/current_liabilities: intentionally NOT extended with a
    # fallback label. Checked exhaustively against raw STND BAL rows for
    # depositories, insurers, REITs, brokers and holding companies (sic2
    # 60-67, 70): none of them has a "Total Current Assets"/"Total Current
    # Liabilities" row, because these industries file unclassified balance
    # sheets under GAAP (a current/non-current split isn't meaningful for
    # loan books, investment portfolios, or real estate held for
    # investment). A null here for those industries is CORRECT -- not a
    # coverage gap to fill (docs/plans/fs_gold_coverage_fills.md, root
    # cause 4). `current_ratio`, built from these two fields downstream, is
    # undefined-by-construction for the same industries.
    "current_assets":      ("BAL", "Total Current Assets", None, "amount"),
    "current_liabilities": ("BAL", "Total Current Liabilities", None, "amount"),
    # cash: coalesced across industry-template label variants
    # (docs/plans/fs_gold_coverage_fills.md, root cause 2) -- depositories
    # report "Cash & Due from Banks"; insurers/REITs/holding companies
    # report a bare "Cash" instead of "Cash & Short-Term Investments". Same
    # is_in()+mean safety note as sga_expense/interest_expense above.
    "cash":                ("BAL", ["Cash & Short-Term Investments", "Cash & Due from Banks",
                                     "Cash"], None, "amount"),
    "long_term_debt":      ("BAL", "Long-Term Debt", None, "amount"),
    "equity":              ("BAL", "Total Shareholders' Equity", None, "amount"),
    "capex":               ("CF", "Capital Expenditures", None, "amount"),
    # label varies: "Shs Outstanding (M)" when FactSet's own footnote scale is millions,
    # "Shs Outstanding" (no "(M)" suffix) when it's billions -- field_id FF_COM_SHS_OUT
    # is the same stable code either way and is what disambiguates from the "Shs
    # Outstanding" PCTCHG (year-over-year % change) row that shares the bare label.
    #
    # kind="shares", NOT "amount": this row's FactSet formula is
    # FF_COM_SHS_OUT(...,,'M') -- a HARD-CODED 'M' (millions) units parameter,
    # independent of the table's own dollar-scale footnote ("All figures in
    # billions/millions of U.S. Dollar EXCEPT PER SHARE AND LABELED ITEMS" --
    # shares outstanding is exactly such a labeled/excepted item). Every other
    # FIELD_MAP row's formula ends in `,'LOCAL','B')` -- 'B' means "use the
    # table's own bulk/base scale", which the footnote word correctly encodes,
    # so those rows are correctly left on kind="amount". shares_out was the
    # ONLY row found with a hard-coded units param (verified across every
    # FIELD_MAP row's raw formula string, AAPL/CMG/TSLA raw JSON, 2026-09-17);
    # applying the table's dollar-scale word to it as well corrupted every
    # mega-cap ticker whose footnote says "billions" by exactly 1000x (see
    # docs/plans/fs_gold_replacement.md S3.2a).
    "shares_out":          ("SHS", ["Shs Outstanding (M)", "Shs Outstanding"], "FF_COM_SHS_OUT", "shares"),
}
# extra BAL row needed only to compute total_debt (not a thesis field on its own)
TOTAL_DEBT_ST_LABEL = ("BAL", "ST Debt & Curr. Portion LT Debt", None, "amount")

# revenue: depositories and broker-dealers (sic2 60/61, and some 62) never
# report a "Sales" line on the STND INC report at all -- checked JPM, BAC,
# GS, MS, COF directly (docs/plans/fs_gold_coverage_fills.md, root cause 1).
# Their revenue-equivalent is "Net Interest Income" (or, when present, "Net
# Interest Income after Provision" -- the more complete of the two, so
# preferred when both exist) plus "Non-Interest Income". This is FactSet's
# own bank/broker "total revenue" convention, not a like-for-like swap for
# "Sales": a bank's net interest income is already net of interest expense,
# so this basis is not the gross revenue flow a manufacturer's "Sales" is.
# Used ONLY when "Sales" is absent for that (factset_id, period_end), never
# to override an existing "Sales" value, and always flagged via the
# `revenue_basis` column ("sales" vs "bank_nii_plus_nonii") rather than
# silently blended into one undifferentiated construct.
BANK_NII_LABELS = ["Net Interest Income after Provision", "Net Interest Income"]
BANK_NONINTEREST_LABEL = "Non-Interest Income"


def build_revenue(long: pl.DataFrame, frequency: str) -> pl.DataFrame:
    """revenue + revenue_basis, one row per (factset_id, period_end)."""
    sales = extract_field(long, frequency, "INC", "Sales", None, "amount").rename({"value": "sales"})
    nii = extract_field(long, frequency, "INC", BANK_NII_LABELS, None, "amount").rename({"value": "nii"})
    noninc = extract_field(long, frequency, "INC", BANK_NONINTEREST_LABEL, None, "amount").rename({"value": "noninc"})

    merged = sales.join(nii, on=["factset_id", "period_end"], how="full", coalesce=True) \
                  .join(noninc, on=["factset_id", "period_end"], how="full", coalesce=True)
    merged = merged.with_columns(
        pl.when(pl.col("nii").is_not_null() | pl.col("noninc").is_not_null())
          .then(pl.col("nii").fill_null(0) + pl.col("noninc").fill_null(0))
          .otherwise(None)
          .alias("bank_revenue")
    )
    merged = merged.with_columns(
        pl.when(pl.col("sales").is_not_null()).then(pl.col("sales"))
          .when(pl.col("bank_revenue").is_not_null()).then(pl.col("bank_revenue"))
          .otherwise(None).alias("revenue"),
        pl.when(pl.col("sales").is_not_null()).then(pl.lit("sales"))
          .when(pl.col("bank_revenue").is_not_null()).then(pl.lit("bank_nii_plus_nonii"))
          .otherwise(None).alias("revenue_basis"),
    )
    return merged.select(["factset_id", "period_end", "revenue", "revenue_basis"])


def extract_field(long: pl.DataFrame, frequency: str, report: str, label: str,
                   field_id: str | None, kind: str) -> pl.DataFrame:
    labels = [label] if isinstance(label, str) else label
    df = long.filter(
        (pl.col("basis") == "STND") & (pl.col("frequency") == frequency)
        & (pl.col("report") == report) & pl.col("field").is_in(labels)
    )
    if field_id is not None:
        df = df.filter(pl.col("field_id") == field_id)
    df = df.drop_nulls(["factset_id", "value", "period_end"])
    if df.height == 0:
        return df.select(["factset_id", "period_end", "value"])
    # collapse duplicate rows (same label surfacing from >1 raw column/formula path,
    # e.g. a field_id-missing formula-extraction miss alongside the resolved one) by
    # mean -- verified these duplicates carry identical values, not real duplicates
    df = df.group_by(["factset_id", "period_end"]).agg(
        pl.col("value").mean().alias("value"), pl.col("scale").first().alias("scale"),
    )
    if kind == "amount":
        df = df.with_columns(
            (pl.col("value") * pl.col("scale").map_elements(lambda s: SCALE_MULT.get(s, 1.0), return_dtype=pl.Float64) / 1e6)
            .alias("value")
        )
    # kind == "shares" (and "per_share"): left as-is -- already in the right unit
    # regardless of the table's dollar-scale footnote (see FIELD_MAP note on shares_out).
    return df.select(["factset_id", "period_end", "value"])


def build_frequency(long: pl.DataFrame, frequency: str) -> pl.DataFrame:
    wide = build_revenue(long, frequency)
    for construct, (report, label, field_id, kind) in FIELD_MAP.items():
        if construct == "revenue":
            continue  # handled by build_revenue() above (bank/broker fallback + revenue_basis)
        fld = extract_field(long, frequency, report, label, field_id, kind).rename({"value": construct})
        wide = wide.join(fld, on=["factset_id", "period_end"], how="full", coalesce=True)

    st_debt = extract_field(long, frequency, *TOTAL_DEBT_ST_LABEL).rename({"value": "st_debt"})
    wide = wide.join(st_debt, on=["factset_id", "period_end"], how="full", coalesce=True)
    wide = wide.with_columns(
        (pl.col("st_debt").fill_null(0) + pl.col("long_term_debt").fill_null(0)).alias("total_debt_sum")
    ).with_columns(
        pl.when(pl.col("st_debt").is_null() & pl.col("long_term_debt").is_null())
        .then(None).otherwise(pl.col("total_debt_sum")).alias("total_debt")
    ).drop(["st_debt", "total_debt_sum"])

    # ticker + period_status: pull from the long table (any STND row for that id/period/frequency)
    meta = (
        long.filter((pl.col("basis") == "STND") & (pl.col("frequency") == frequency))
        .select(["factset_id", "ticker", "period_end", "period_status"])
        .unique(subset=["factset_id", "period_end"], keep="first")
    )
    wide = wide.join(meta, on=["factset_id", "period_end"], how="left")

    # fiscal_label: infer each id's fiscal-year-end month from its own STND annual periods
    ann_months = (
        long.filter((pl.col("basis") == "STND") & (pl.col("frequency") == "annual") & pl.col("period_end").is_not_null())
        .with_columns(pl.col("period_end").dt.month().alias("m"))
        .group_by(["factset_id", "m"]).agg(pl.len().alias("n"))
        .sort(["factset_id", "n"], descending=[False, True])
        .group_by("factset_id", maintain_order=True).agg(pl.col("m").first().alias("fye_month"))
    )
    wide = wide.join(ann_months, on="factset_id", how="left")
    wide = wide.with_columns([
        pl.col("period_end").dt.month().alias("_m"),
        pl.col("period_end").dt.year().alias("_y"),
    ]).with_columns([
        (((pl.col("fye_month") - pl.col("_m")) % 12) // 3).alias("_months_to_fye_q"),
        pl.when(pl.col("_m") > pl.col("fye_month")).then(pl.col("_y") + 1).otherwise(pl.col("_y")).alias("fiscal_year"),
    ]).with_columns(
        (4 - pl.col("_months_to_fye_q")).alias("fiscal_quarter")
    )
    if frequency == "annual":
        wide = wide.with_columns(("FY" + pl.col("fiscal_year").cast(pl.Utf8)).alias("fiscal_label"))
    else:
        wide = wide.with_columns(
            ("FY" + pl.col("fiscal_year").cast(pl.Utf8) + "Q" + pl.col("fiscal_quarter").cast(pl.Utf8)).alias("fiscal_label")
        )
    wide = wide.drop(["_m", "_y", "_months_to_fye_q", "fye_month", "fiscal_year", "fiscal_quarter"])

    cols = (["ticker", "factset_id", "period_end", "fiscal_label", "period_status"]
            + list(FIELD_MAP.keys()) + ["revenue_basis", "total_debt"])
    wide = wide.select(cols).sort(["ticker", "period_end"])
    return wide


def main():
    long = pl.read_parquet(FUND_LONG)
    print(f"Loaded {FUND_LONG} ({len(long)} rows)")

    for frequency, out_path in (("annual", OUT_ANNUAL), ("quarterly", OUT_QUARTERLY)):
        wide = build_frequency(long, frequency)
        wide.write_parquet(out_path)
        n_ids = wide["factset_id"].n_unique()
        n_tickers = wide["ticker"].n_unique()
        print(f"\n=== {frequency} ===")
        print(f"Wrote {out_path}: {len(wide)} rows, {n_ids} factset_ids, {n_tickers} tickers")
        print(f"Rows per ticker: min={wide.group_by('ticker').len()['len'].min()}, "
              f"median={wide.group_by('ticker').len()['len'].median()}, "
              f"max={wide.group_by('ticker').len()['len'].max()}")
        fields = list(FIELD_MAP.keys()) + ["total_debt"]
        non_null = {f: float(wide[f].is_not_null().mean()) for f in fields}
        print("Share non-null per field:")
        for f, share in non_null.items():
            print(f"  {f:22s} {share:.3f}")
        basis_counts = wide["revenue_basis"].value_counts().sort("revenue_basis")
        print("revenue_basis breakdown:", basis_counts.to_dicts())

        if frequency == "annual":
            aapl = wide.filter((pl.col("ticker") == "AAPL") & (pl.col("fiscal_label") == "FY2025"))
            if aapl.height:
                r = aapl.row(0, named=True)
                print(f"\nAAPL FY2025 sanity: revenue={r['revenue']:.1f}M, net_income={r['net_income']:.1f}M "
                      f"(known: revenue ~$416,161M, net_income ~$112,010M per Apple's FY2025 10-K)")
            else:
                print("\nAAPL FY2025 sanity: NOT FOUND in wide extract")

    print("\nSign conventions (as returned by FactSet STND, not remapped):")
    print("  - capex (CF 'Capital Expenditures'): FactSet reports this as a cash OUTFLOW, "
          "i.e. negative, on the STND CF statement.")
    print("  - da, interest_expense, tax_expense, sga_expense, rd_expense, cost_of_revenue: "
          "FactSet STND reports these as POSITIVE magnitudes on the INC statement (expenses, "
          "not signed contra-revenue).")
    print("  - All figures are already in millions of USD after scale normalization here; "
          "eps_diluted is per-share (unscaled); shares_out is in millions of shares.")


if __name__ == "__main__":
    main()
