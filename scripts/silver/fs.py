"""
scripts/silver/fs.py — silver.market_prices (fs-backed replacement),
silver.fs_financials, silver.fs_market_factors_daily
(docs/plans/fs_gold_replacement.md).

silver.market_prices: fs daily price + market cap, universe-filtered and
delisting-truncated exactly like the yfinance-backed table it replaces
(silver.firm_universe's `delisted`/`delisting_date`, computed once in
scripts/silver/universe.py). `close`/`adj_close` both carry fs `price`
(already split-adjusted -- fs has no separate raw/adjusted close); `total_return`
is fs's own dividend-inclusive total-return index (used for every
return/beta/vol/NCSKEW/DUVOL/momentum formula per orchestrator decision);
`market_cap` is fs's own daily market cap (fixes the pre-split
understatement bug: silver close was split-adjusted but the old gold build
multiplied it by historical, pre-split cover-page shares -- fs market cap
needs no share count at all).

silver.fs_financials: bronze.fs_fundamentals_{annual,quarterly} unioned with
a `period` column, universe-filtered, `filing_date_pt`/`accession_number`/
`pit_source` attached (docs/plans/fs_gold_replacement.md S2):
  1. primary: EDGAR filing_date matched by (ticker, period_end) within +/-10
     calendar days, closest match -- annual matches silver.filing_manifest
     (10-K) only; quarterly matches silver.filing_manifest_10q AND
     silver.filing_manifest (10-K) combined, because a fiscal Q4 is
     disclosed in the 10-K, not a 10-Q.
  2. fallback: bronze.fs_report_dates.eps_rpt_date (fs's own earnings
     release date, 0-21 days before the SEC filing) when no EDGAR match,
     matched by the same +/-10-day nearest-asof tolerance as EDGAR
     (match_report_dates()) rather than exact equality on `period_end` --
     bronze.fs_report_dates keys `period_end` to the filer's actual close
     date (e.g. 2019-12-29), not the calendar-normalized STND period end
     (2019-12-31) fundamentals uses, so an exact join silently dropped this
     fallback for most periods (docs/plans/fs_gold_coverage_fills.md, root
     cause 5 -- it only surfaced as a coverage gap for periods with no
     EDGAR match either, e.g. 10-Ks filed before filing_manifest's
     ~2020-11-01 coverage start).
  3. pit_source records which of the two, or null when neither matched.
     revenue_basis ("sales"/"bank_nii_plus_nonii", from
     scripts/sources/fs/build_fundamentals_wide.py) passes through
     unchanged -- flags which STND construct `revenue` was built from.
shares_out is fs's own STND field (now scale-fixed, see
scripts/sources/fs/build_fundamentals_wide.py); it is the right concept for
a financials-family "shares outstanding as of the disclosed period" column.
Daily/point-in-time shares (a specific calendar date, not a fiscal
period-end) should instead be derived as market_cap/price from
silver.market_prices -- that derivation lives in the gold builders that need
it (firm_quarter market), not here.

silver.fs_market_factors_daily: (date, mktrf, rf). `mktrf` is fs's own S&P
500 benchmark (`SP50-SPX`) daily total-return pct-change minus `rf`
(orchestrator decision: betas/vol/NCSKEW/DUVOL/momentum regress on fs
total_return in excess of fs SP50-SPX total return). `rf` is kept unchanged
from bronze.market_factors_daily (Ken French daily one-month T-bill rate,
fs has no usable risk-free series).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for sub in ("common",):
    p = str(REPO_ROOT / "scripts" / sub)
    if p not in sys.path:
        sys.path.insert(0, p)

import polars as pl
import layers as L  # noqa: E402

BUILDER = "scripts/silver/fs.py"
PIT_TOLERANCE_DAYS = 10
# fs STND dollar fields are in millions USD; gold's dollar-unit convention
# (kept from the old XBRL build, docs/plans/fs_gold_replacement.md S3.1) is
# raw dollars -- scaled x1e6 in silver, once. shares_out is in millions of
# shares (scale-fixed, scripts/sources/fs/build_fundamentals_wide.py) and
# scales the same way to a raw share count. eps_diluted (per-share, already
# a small dollar amount) and every ratio/margin/ttm-growth column are left
# unscaled.
FS_MONEY_COLUMNS = ["revenue", "cost_of_revenue", "gross_profit", "rd_expense", "sga_expense", "operating_income",
                   "ebitda", "da", "interest_expense", "pretax_income", "tax_expense", "net_income",
                   "total_assets", "current_assets", "current_liabilities", "cash", "long_term_debt",
                   "equity", "capex", "total_debt"]
FS_SHARES_COLUMNS = ["shares_out"]


def universe_tickers() -> pl.LazyFrame:
    return L.scan("silver.firm_universe").select("ticker")


def delisting_dates() -> pl.DataFrame:
    return L.read("silver.firm_universe").select("ticker", "delisting_date")


def before_delisting(lf: pl.LazyFrame, date_col: str, dates: pl.DataFrame) -> pl.LazyFrame:
    return (lf.join(dates.lazy(), on="ticker", how="left")
            .filter(pl.col("delisting_date").is_null() | (pl.col(date_col).cast(pl.Date) <= pl.col("delisting_date")))
            .drop("delisting_date"))


def build_market_prices(tickers: pl.LazyFrame, dates: pl.DataFrame) -> None:
    prices = L.scan("bronze.fs_prices_daily").select("ticker", "date", "price", "volume", "total_return")
    # fs market_cap is in millions USD; gold's dollar-unit convention (kept
    # from the old XBRL/yfinance build, docs/plans/fs_gold_replacement.md
    # S3.1) is raw dollars -- scaled x1e6 here, once, so log_market_cap and
    # every ratio built on it downstream needs no further unit handling.
    # price/close/adj_close (dollars per share) and total_return (a
    # dividend-inclusive index level, not a dollar amount) are NOT scaled.
    mcap = L.scan("bronze.fs_market_cap_daily").select("ticker", "date", (pl.col("market_cap") * 1e6).alias("market_cap"))
    merged = (prices.join(mcap, on=["ticker", "date"], how="left")
              .join(tickers, on="ticker", how="semi")
              .with_columns(pl.col("price").alias("close"), pl.col("price").alias("adj_close"),
                            pl.lit("fs").alias("source"))
              .select("date", "ticker", "close", "adj_close", "total_return", "volume", "market_cap", "source"))
    merged = before_delisting(merged, "date", dates)
    L.write_table("silver.market_prices", merged, keys=["ticker", "date"],
                  inputs=[L.path("bronze.fs_prices_daily"), L.path("bronze.fs_market_cap_daily"),
                          L.path("silver.firm_universe")],
                  builder=BUILDER)


def match_edgar(fs_periods: pl.DataFrame, manifest: pl.DataFrame) -> pl.DataFrame:
    """For each (ticker, period_end) row, the closest manifest filing_date of
    the same ticker within +/-PIT_TOLERANCE_DAYS of period_end (nearest
    asof join, then a tolerance filter since polars' asof tolerance is
    one-directional per strategy)."""
    fs_periods = fs_periods.sort(["ticker", "period_end"])
    manifest = manifest.sort(["ticker", "period_end_date"])
    joined = fs_periods.join_asof(manifest, left_on="period_end", right_on="period_end_date", by="ticker",
                                  strategy="nearest")
    joined = joined.with_columns((pl.col("period_end_date") - pl.col("period_end")).dt.total_days().abs().alias("_gap"))
    joined = joined.with_columns(
        pl.when(pl.col("_gap") <= PIT_TOLERANCE_DAYS).then(pl.col("filing_date")).otherwise(None).alias("edgar_filing_date"),
        pl.when(pl.col("_gap") <= PIT_TOLERANCE_DAYS).then(pl.col("accession_number")).otherwise(None).alias("edgar_accession_number"),
    )
    return joined.select("ticker", "period_end", "edgar_filing_date", "edgar_accession_number")


def match_report_dates(fs_periods: pl.DataFrame, report_dates: pl.DataFrame) -> pl.DataFrame:
    """For each (ticker, period_end) row, the closest bronze.fs_report_dates
    row of the same ticker within +/-PIT_TOLERANCE_DAYS of period_end (same
    nearest-asof-then-tolerance pattern as match_edgar).

    Needed because bronze.fs_report_dates keys `period_end` to the company's
    actual fiscal close date (e.g. 2019-12-29 for HAS), while
    bronze.fs_fundamentals_{annual,quarterly}'s `period_end` is the
    calendar-normalized STND period end (2019-12-31) -- an exact-equality
    join on `period_end` (the previous behavior) silently missed the
    `eps_rpt_date` fallback for most periods, since a filer's actual close
    date is rarely exactly a calendar month/quarter end. This was masked for
    almost all rows because the EDGAR match (already tolerance-based) usually
    succeeds first; it surfaced as a systematic `filing_date_pt` gap only for
    periods with no EDGAR match at all -- concretely, 10-Ks filed before
    `silver.filing_manifest`'s coverage window starts (~2020-11-01), which
    drops a ticker's earliest disclosed fiscal year from the point-in-time
    panel and, downstream, kills `revenue_yoy` for the following year (no
    prior-year value to grow from) -- docs/plans/fs_gold_coverage_fills.md,
    root cause 5."""
    fs_periods = fs_periods.sort(["ticker", "period_end"])
    report_dates = report_dates.rename({"period_end": "rd_period_end"}).sort(["ticker", "rd_period_end"])
    joined = fs_periods.join_asof(report_dates, left_on="period_end", right_on="rd_period_end", by="ticker",
                                  strategy="nearest")
    joined = joined.with_columns((pl.col("rd_period_end") - pl.col("period_end")).dt.total_days().abs().alias("_gap"))
    joined = joined.with_columns(
        pl.when(pl.col("_gap") <= PIT_TOLERANCE_DAYS).then(pl.col("eps_rpt_date")).otherwise(None).alias("eps_rpt_date"),
        pl.when(pl.col("_gap") <= PIT_TOLERANCE_DAYS).then(pl.col("source_doc")).otherwise(None).alias("source_doc"),
    )
    return joined.select("ticker", "period_end", "eps_rpt_date", "source_doc")


def build_fs_financials(tickers: pl.LazyFrame) -> None:
    # "LTM" rows are a rolling trailing-twelve-month figure for the fiscal year
    # still in progress, not a period the company ever discloses as a discrete
    # filed fiscal year -- dropped, same reasoning as the repo's existing
    # frames-API leak guard (never keep a fabricated "period" with no real filing).
    ann = (L.scan("bronze.fs_fundamentals_annual")
           .filter(pl.col("period_status").ne_missing("LTM"))
           .with_columns(pl.lit("annual").alias("period")))
    qtr = L.scan("bronze.fs_fundamentals_quarterly").with_columns(pl.lit("quarterly").alias("period"))
    fin = pl.concat([ann, qtr], how="diagonal_relaxed").join(tickers, on="ticker", how="semi").collect()
    # unit fix (money millions -> dollars, shares millions -> raw count) --
    # see FS_MONEY_COLUMNS/FS_SHARES_COLUMNS above.
    fin = fin.with_columns([pl.col(c) * 1e6 for c in FS_MONEY_COLUMNS + FS_SHARES_COLUMNS])
    # fs (FactSet) STND reports capex as a signed cash OUTFLOW (negative),
    # documented as deliberately left as-is in
    # scripts/sources/fs/build_fundamentals_wide.py's sign-conventions
    # printout. Every existing gold covariate/target (capex, capex_intensity,
    # next_capex_yoy, the old XBRL-era PaymentsToAcquirePropertyPlantAndEquipment
    # figures) uses a positive dollar-magnitude convention -- flipped here,
    # once, so every downstream gold builder inherits the right sign without
    # repeating the fix. Found via next_capex_yoy silently going to 0% non-null
    # (a positive-base filter on an always-negative value never passes).
    fin = fin.with_columns(pl.col("capex") * -1)

    manifest_10k = L.scan("silver.filing_manifest").filter(pl.col("form_type") == "10-K").select(
        "ticker", "accession_number", "filing_date", "period_end_date").collect()
    manifest_10q = L.scan("silver.filing_manifest_10q").select(
        "ticker", "accession_number", "filing_date", "period_end_date").collect()
    manifest_10k = manifest_10k.with_columns(pl.col("filing_date").cast(pl.Date), pl.col("period_end_date").cast(pl.Date))
    manifest_10q = manifest_10q.with_columns(pl.col("filing_date").cast(pl.Date), pl.col("period_end_date").cast(pl.Date))
    manifest_combined = pl.concat([manifest_10q, manifest_10k])

    annual_periods = fin.filter(pl.col("period") == "annual").select("ticker", "period_end").unique()
    quarterly_periods = fin.filter(pl.col("period") == "quarterly").select("ticker", "period_end").unique()

    annual_match = match_edgar(annual_periods, manifest_10k).with_columns(pl.lit("annual").alias("period"))
    quarterly_match = match_edgar(quarterly_periods, manifest_combined).with_columns(pl.lit("quarterly").alias("period"))
    edgar_match = pl.concat([annual_match, quarterly_match])

    report_dates_all = L.read("bronze.fs_report_dates").select("ticker", "period", "period_end", "eps_rpt_date", "source_doc")
    report_dates_annual = report_dates_all.filter(pl.col("period") == "annual").select(
        "ticker", "period_end", "eps_rpt_date", "source_doc")
    report_dates_quarterly = report_dates_all.filter(pl.col("period") == "quarterly").select(
        "ticker", "period_end", "eps_rpt_date", "source_doc")
    # tolerance-matched (not exact-equality), see match_report_dates() docstring
    # -- root cause 5, docs/plans/fs_gold_coverage_fills.md.
    rd_annual_match = match_report_dates(annual_periods, report_dates_annual).with_columns(pl.lit("annual").alias("period"))
    rd_quarterly_match = match_report_dates(quarterly_periods, report_dates_quarterly).with_columns(pl.lit("quarterly").alias("period"))
    report_dates_match = pl.concat([rd_annual_match, rd_quarterly_match])

    fin = (fin.join(edgar_match, on=["ticker", "period", "period_end"], how="left")
           .join(report_dates_match, on=["ticker", "period", "period_end"], how="left"))
    fin = fin.with_columns(
        pl.coalesce("edgar_filing_date", "eps_rpt_date").alias("filing_date_pt"),
        pl.coalesce("edgar_accession_number").alias("accession_number"),
        pl.when(pl.col("edgar_filing_date").is_not_null()).then(pl.lit("edgar"))
          .when(pl.col("eps_rpt_date").is_not_null()).then(pl.lit("fs_release"))
          .otherwise(None).alias("pit_source"),
    ).drop(["edgar_filing_date", "edgar_accession_number", "eps_rpt_date", "source_doc"])

    L.write_table("silver.fs_financials", fin, keys=["ticker", "period", "period_end"],
                  inputs=[L.path("bronze.fs_fundamentals_annual"), L.path("bronze.fs_fundamentals_quarterly"),
                          L.path("bronze.fs_report_dates"), L.path("silver.filing_manifest"),
                          L.path("silver.filing_manifest_10q"), L.path("silver.firm_universe")],
                  builder=BUILDER,
                  extra={"pit_source_counts": {k: int(v) for k, v in
                         zip(*fin.group_by("pit_source").len().sort("pit_source")[["pit_source", "len"]])}})


def build_fs_market_factors() -> None:
    bench = (L.scan("bronze.fs_benchmark_daily").select("date", "total_return")
             .sort("date")
             .with_columns((pl.col("total_return") / pl.col("total_return").shift(1) - 1).alias("bench_ret"))
             .drop("total_return"))
    rf = L.scan("bronze.market_factors_daily").select(pl.col("date").cast(pl.Date), "rf")
    merged = bench.join(rf, on="date", how="inner").with_columns(
        (pl.col("bench_ret") - pl.col("rf")).alias("mktrf")).select("date", "mktrf", "rf")
    L.write_table("silver.fs_market_factors_daily", merged, keys=["date"],
                  inputs=[L.path("bronze.fs_benchmark_daily"), L.path("bronze.market_factors_daily")],
                  builder=BUILDER)


def main() -> None:
    tickers = universe_tickers()
    dates = delisting_dates()
    build_market_prices(tickers, dates)
    build_fs_financials(tickers)
    build_fs_market_factors()


if __name__ == "__main__":
    main()
