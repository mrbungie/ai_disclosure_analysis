"""
scripts/silver/universe.py — silver.firm_universe, silver.filing_manifest,
silver.filing_manifest_10q, silver.market_prices restricted to the analysis
universe.

Every gold join on ticker goes through these tables, so every panel inherits
the same universe without repeating the filter.

Delisted firms. silver.firm_universe carries `delisted`, `delisting_date` and
`delisting_source` for the universe firms whose common stock stopped trading
(firm-universe `active_status` delisted). The date is an EDGAR filing of the
firm's CIK (bronze.sec_filing_index): the Form 25 / 25-NSE (the exchange's
notice removing the stock) of the latest Form 25 filed since SAMPLE_START and
followed within DEREGISTRATION_DAYS by a Form 15-12B/15-12G (the stock's
deregistration);
otherwise the first Form 25 after the firm's last 10-K/10-Q; otherwise a
documented completion date (DELISTING_OVERRIDES). Never the last price date:
a delisted ticker can be reused by another company. A firm flagged delisted
without any such event (a ticker change, e.g. GPS -> GAP) is not delisted.
Documents a delisted firm filed after `delisting_date` and its prices after
that date are dropped from silver; everything before is kept.

Earnings-call transcripts enter silver.filing_manifest as checked in
bronze.call_transcripts (scripts/bronze/call_transcripts.py): transcripts with
an `exclude_reason` (another company's call, a non-results event, a duplicate
of a kept transcript, a break of the call sequence) are left out, and a kept
call's `filing_date` / `fiscal_period` are the checked `call_date` /
`fiscal_period`. The source values stay in `metadata_filing_date` /
`metadata_fiscal_period`, the rule that set each value in `call_date_source` /
`fiscal_period_source`, and the call's place in its ticker's sequence in
`call_sequence` (first | next | gap | label_break). Filing rows have nulls in
these columns.
"""

from __future__ import annotations

import pandas as pd
import polars as pl
from _paths import ANALYSIS_PANEL, L, bronze_inputs

BUILDER = "scripts/silver/universe.py"
CALL_FORM_TYPE = "Earnings call transcript"
CALL_COLUMNS = ["call_date_source", "fiscal_period_source", "call_sequence"]
DEREGISTRATION_DAYS = 30
SAMPLE_START = "2020-01-01"  # earlier Form 25 pairs are other securities (preferred stock, notes)
DELISTING_FORMS = ("25", "25-NSE")
DEREGISTRATION_FORMS = ("15-12B", "15-12G")
# Delisted universe firms without an EDGAR delisting filing.
DELISTING_OVERRIDES = {
    "FRC": ("2023-05-01", "FDIC receivership and sale to JPMorgan Chase completed 2023-05-01 "
                          "(First Republic Bank filed with the FDIC, not EDGAR)"),
}


def delistings(firms: pl.DataFrame) -> pl.DataFrame:
    """ticker, delisting_date, delisting_source for the delisted universe firms."""
    flagged = firms.filter(pl.col("active_status") == "delisted").select("ticker", "cik")
    filings = (L.scan("bronze.sec_filing_index").join(flagged.lazy(), on="cik").collect()
               .sort("filing_date").to_pandas())
    rows = []
    for ticker, cik in flagged.iter_rows():
        f = filings[filings["ticker"] == ticker]
        notices = f[f["form"].isin(DELISTING_FORMS)]
        deregistrations = f[f["form"].isin(DEREGISTRATION_FORMS)]
        recent = notices[notices["filing_date"] >= pd.Timestamp(SAMPLE_START)]
        paired = recent[[((deregistrations["filing_date"] >= d)
                           & ((deregistrations["filing_date"] - d).dt.days <= DEREGISTRATION_DAYS)).any()
                          for d in recent["filing_date"]]]
        last_periodic = f.loc[f["form"].isin(["10-K", "10-Q"]), "filing_date"].max()
        after = notices[notices["filing_date"] > last_periodic] if pd.notna(last_periodic) else notices
        if len(paired):
            hit, rule = paired.iloc[-1], "followed by Form 15-12B/15-12G"
        elif len(after):
            hit, rule = after.iloc[0], "first after the last 10-K/10-Q"
        elif ticker in DELISTING_OVERRIDES:
            date, note = DELISTING_OVERRIDES[ticker]
            rows.append((ticker, pd.Timestamp(date), note))
            continue
        else:
            continue
        rows.append((ticker, hit["filing_date"], f"EDGAR Form {hit['form']} {hit['accession_number']} ({rule})"))
    return pl.DataFrame(pd.DataFrame(rows, columns=["ticker", "delisting_date", "delisting_source"])).with_columns(
        pl.col("delisting_date").cast(pl.Date))


def before_delisting(lf: pl.LazyFrame, date_col: str, dates: pl.DataFrame) -> pl.LazyFrame:
    """Rows dated on or before the ticker's delisting date (all rows of a firm
    that is not delisted)."""
    return (lf.join(dates.lazy().select("ticker", "delisting_date"), on="ticker", how="left")
            .filter(pl.col("delisting_date").is_null() | (pl.col(date_col).cast(pl.Date) <= pl.col("delisting_date")))
            .drop("delisting_date"))


def main() -> None:
    firms = L.read("bronze.firm_universe").filter(pl.col("membership_groups").list.contains(ANALYSIS_PANEL))
    filings_index = L.scan("bronze.sec_filing_index").select("cik").unique().collect()
    dates = delistings(firms)
    firms = (firms.join(dates, on="ticker", how="left")
             .with_columns(pl.col("delisting_date").is_not_null().alias("delisted")))
    firms = L.write_table("silver.firm_universe", firms, keys=["ticker"],
                          inputs=bronze_inputs("bronze.firm_universe", "bronze.sec_filing_index"), builder=BUILDER,
                          extra={"delisted": [{"ticker": t, "delisting_date": str(d), "source": s}
                                              for t, d, s in dates.sort("delisting_date").iter_rows()],
                                 "ciks_with_edgar_index": filings_index.height})
    tickers = firms.select("ticker").lazy()

    checks = L.scan("bronze.call_transcripts").select(
        "document_id", "exclude_reason", pl.col("call_date").cast(pl.Date), pl.col("fiscal_period").alias("checked_fiscal_period"),
        *CALL_COLUMNS)
    manifest = (L.scan("bronze.filing_manifest").join(tickers, on="ticker", how="semi")
                .join(checks, on="document_id", how="left")
                .filter((pl.col("form_type") != CALL_FORM_TYPE) | pl.col("exclude_reason").is_null() & pl.col("call_date").is_not_null())
                .with_columns(pl.when(pl.col("form_type") == CALL_FORM_TYPE).then(pl.col("filing_date")).alias("metadata_filing_date"),
                              pl.when(pl.col("form_type") == CALL_FORM_TYPE).then(pl.col("fiscal_period")).alias("metadata_fiscal_period"))
                .with_columns(pl.coalesce("call_date", "filing_date").alias("filing_date"),
                              pl.coalesce("checked_fiscal_period", "fiscal_period").alias("fiscal_period"))
                .drop("exclude_reason", "call_date", "checked_fiscal_period"))
    L.write_table("silver.filing_manifest", before_delisting(manifest, "filing_date", dates),
                  keys=["form_type", "document_id", "ticker"],
                  inputs=bronze_inputs("bronze.firm_universe", "bronze.filing_manifest", "bronze.call_transcripts",
                                       "bronze.sec_filing_index"), builder=BUILDER)
    L.write_table("silver.filing_manifest_10q",
                  before_delisting(L.scan("bronze.filing_manifest_10q").join(tickers, on="ticker", how="semi"), "filing_date", dates),
                  keys=["document_id"],
                  inputs=bronze_inputs("bronze.firm_universe", "bronze.filing_manifest_10q", "bronze.sec_filing_index"),
                  builder=BUILDER)
    L.write_table("silver.market_prices",
                  before_delisting(L.scan("bronze.market_prices").join(tickers, on="ticker", how="semi"), "date", dates),
                  keys=["ticker", "date"],
                  inputs=bronze_inputs("bronze.firm_universe", "bronze.market_prices", "bronze.sec_filing_index"), builder=BUILDER)


if __name__ == "__main__":
    main()
