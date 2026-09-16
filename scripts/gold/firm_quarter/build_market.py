"""Firm-quarter market covariates and the forward-looking firm-quarter targets.

Every value of a covariate uses prices before `as_of_date` and filings filed
strictly before it; every target is measured strictly after it (trading days
from `as_of_date` on, or a fiscal quarter that ends after it). Prices come
from silver.market_prices (`market_windows`: idx = first trading day on or
after `as_of_date`), accounting values from the first-disclosed quarterly
XBRL panel (`build_financials.quarterly_panel`, one value per ticker and
calendar quarter of the fiscal period end) and the cover-page share count of
the 10-K/10-Q filings in silver.

  covariates/firm_quarter/market
    price_pre         close at idx-1 (within STALE_DAYS of as_of_date)
    shares_out        dei:EntityCommonStockSharesOutstanding of the latest
                      10-K/10-Q filed before as_of_date (non-dimensional value,
                      else the sum over share classes)
    log_market_cap    log(price_pre x shares_out)
    revenue_ttm       revenue of the latest four consecutive fiscal quarters
                      filed before as_of_date (null when one is missing)
    ps_ratio          price_pre x shares_out / revenue_ttm
    beta_pre_252, idio_vol_pre_252
                      one-factor beta over [idx-252, idx), at least 120 days,
                      and its residual s.d. x sqrt(252)

  targets/firm_quarter/market (the next quarter)
    beta_post_63, idio_vol_post_63
                      one-factor beta over [idx, idx+63), at least 60 days, and
                      its residual s.d. x sqrt(252)
    vol_post_63       s.d. of daily returns over [idx, idx+63) x sqrt(252)
    ps_ratio_post     ps_ratio at the next quarter's as_of_date: close on the
                      last trading day before it, shares_out and revenue_ttm
                      filed before it

  targets/firm_quarter/financials (the fiscal quarter ending in the next
  calendar quarter, `next_quarter`)
    next_revenue_yoy   revenue of that quarter / revenue of the same calendar
                       quarter a year earlier - 1
    next_rd_intensity  rd_expense / revenue of that quarter

Output: covariates/firm_quarter/market, targets/firm_quarter/market,
targets/firm_quarter/financials.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "gold" / "financials"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "gold" / "firm_quarter"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "common"))
import layers as L  # noqa: E402
import pit  # noqa: E402
from build_financials import quarterly_panel  # noqa: E402
from market_windows import event_index, load_factors, load_prices, one_factor_ols, with_factors  # noqa: E402

BUILDER = "scripts/gold/firm_quarter/build_market.py"
BETA_PRE_DAYS, BETA_PRE_MIN_OBS = 252, 120
POST_DAYS, POST_MIN_OBS = 63, 60
STALE_DAYS = 7  # price_pre needs a close within a week before as_of_date (no stale price after a delisting)
COV_MARKET = ["price_pre", "shares_out", "log_market_cap", "revenue_ttm", "ps_ratio", "beta_pre_252", "idio_vol_pre_252"]
TGT_MARKET = ["beta_post_63", "idio_vol_post_63", "vol_post_63", "ps_ratio_post"]
TGT_FINANCIALS = ["next_quarter", "next_revenue_yoy", "next_rd_intensity"]


def market_model(window: pd.DataFrame, min_obs: int) -> tuple[float, float]:
    """(beta, residual s.d. x sqrt(252)) of a one-factor regression, nulls
    below `min_obs` trading days."""
    if len(window) < min_obs:
        return np.nan, np.nan
    coef, _, residuals = one_factor_ols(window)
    return float(coef[1]), float(residuals.std(ddof=2) * np.sqrt(252))


def price_windows(spine: pd.DataFrame) -> pd.DataFrame:
    prices, factors = load_prices(), load_factors()
    rows = []
    for row in spine[["id", "ticker", "as_of_date", "next_as_of_date"]].itertuples(index=False):
        rec = {"id": row.id}
        pr = prices.get(row.ticker)
        if pr is not None:
            idx = event_index(pr, row.as_of_date)
            if 1 <= idx <= len(pr) and pr["date"].iloc[idx - 1] >= row.as_of_date - pd.Timedelta(days=STALE_DAYS):
                rec["price_pre"] = float(pr["close"].iloc[idx - 1])
            if idx >= BETA_PRE_DAYS:
                rec["beta_pre_252"], rec["idio_vol_pre_252"] = market_model(
                    with_factors(pr.iloc[idx - BETA_PRE_DAYS:idx], factors), BETA_PRE_MIN_OBS)
            if idx + POST_DAYS <= len(pr):
                post = pr.iloc[idx:idx + POST_DAYS]
                rec["beta_post_63"], rec["idio_vol_post_63"] = market_model(with_factors(post, factors), POST_MIN_OBS)
                if post["ret"].notna().sum() >= POST_MIN_OBS:
                    rec["vol_post_63"] = float(post["ret"].std() * np.sqrt(252))
            nxt = event_index(pr, row.next_as_of_date)
            if nxt < len(pr) and nxt - 1 >= idx:  # a trading day after the next as_of_date exists: that quarter closed
                rec["price_next"] = float(pr["close"].iloc[nxt - 1])
        rows.append(rec)
    return spine[["id"]].merge(pd.DataFrame(rows), on="id", how="left")


def filing_shares() -> pd.DataFrame:
    """ticker, filing_date, shares_out of every 10-K/10-Q in silver."""
    filings = pl.concat([
        L.scan("silver.filing_manifest").filter(pl.col("form_type") == "10-K").select("ticker", "accession_number"),
        L.scan("silver.filing_manifest_10q").select("ticker", "accession_number")])
    facts = (L.scan("bronze.xbrl_facts")
             .filter((pl.col("concept") == "dei:EntityCommonStockSharesOutstanding") & pl.col("numeric_value").is_not_null())
             .join(filings, on=["ticker", "accession_number"], how="semi")
             .group_by("ticker", "accession_number", "filing_date", "has_dimensions")
             .agg(pl.col("numeric_value").sum().alias("value"))
             .collect().to_pandas())
    # the non-dimensional total when the filing tags one, else the sum over classes
    facts = facts.sort_values(["ticker", "accession_number", "has_dimensions"])
    shares = facts.drop_duplicates(["ticker", "accession_number"], keep="first")
    shares = shares.rename(columns={"value": "shares_out"})[["ticker", "filing_date", "shares_out"]]
    shares["filing_date"] = pd.to_datetime(shares["filing_date"])
    return shares.sort_values(["ticker", "filing_date"]).drop_duplicates(["ticker", "filing_date"], keep="last")


def revenue_series() -> pd.DataFrame:
    """First-disclosed quarterly revenue and R&D: ticker, cal_q (year*4 +
    quarter-1 of the fiscal period end), filing_date, revenue, rd_expense."""
    panel = quarterly_panel().dropna(subset=["filing_date"])
    panel = panel[panel["metric"].isin(["revenue", "rd_expense"])]
    panel["cal_q"] = panel["year"] * 4 + panel["quarter"] - 1
    panel["filing_date"] = pd.to_datetime(panel["filing_date"])
    panel = panel.sort_values(["ticker", "metric", "cal_q", "filing_date"]).drop_duplicates(["ticker", "metric", "cal_q"])
    wide = panel.pivot_table(index=["ticker", "cal_q"], columns="metric", values="value", aggfunc="first").reset_index()
    revenue_filed = panel[panel["metric"] == "revenue"][["ticker", "cal_q", "filing_date"]]
    return wide.merge(revenue_filed, on=["ticker", "cal_q"], how="left")


def revenue_ttm_asof(series: pd.DataFrame, dates: pd.DataFrame, date_col: str) -> pd.Series:
    """Revenue over the latest four consecutive fiscal quarters filed before
    each (ticker, date) of `dates`."""
    out = pd.Series(np.nan, index=dates.index)
    rev = series.dropna(subset=["revenue", "filing_date"])
    by_ticker = {t: g for t, g in rev.groupby("ticker")}
    for ticker, rows in dates.groupby("ticker"):
        g = by_ticker.get(ticker)
        if g is None:
            continue
        for i, date in zip(rows.index, rows[date_col]):
            known = g[g["filing_date"] < date]
            if known.empty:
                continue
            latest = int(known["cal_q"].max())
            values = known.set_index("cal_q")["revenue"]
            quarters = [latest - k for k in range(4)]
            if all(q in values.index for q in quarters):
                out[i] = float(values.loc[quarters].sum())
    return out


def main() -> None:
    spine = L.read_gold("firm_quarter", spine_columns=L.GOLD_SPINE_COLUMNS["firm_quarter"])
    spine["next_as_of_date"] = spine["as_of_date"] + pd.DateOffset(months=3)
    panel = spine.merge(price_windows(spine), on="id", how="left")

    shares = filing_shares()
    for date_col, name in (("as_of_date", "shares_out"), ("next_as_of_date", "shares_next")):
        known = pit.asof_join(panel[["ticker", date_col]], shares, event_date=date_col, snapshot_date="filing_date",
                              strict=True)
        panel[name] = known["shares_out"].values
    series = revenue_series()
    panel["revenue_ttm"] = revenue_ttm_asof(series, panel, "as_of_date")
    panel["revenue_ttm_next"] = revenue_ttm_asof(series, panel, "next_as_of_date")
    positive = lambda s: s.where(s > 0)  # noqa: E731
    panel["log_market_cap"] = np.log(positive(panel["price_pre"] * panel["shares_out"]))
    panel["ps_ratio"] = panel["price_pre"] * panel["shares_out"] / positive(panel["revenue_ttm"])
    panel["ps_ratio_post"] = panel["price_next"] * panel["shares_next"] / positive(panel["revenue_ttm_next"])

    # the fiscal quarter ending in the calendar quarter that starts at as_of_date
    as_of = pd.to_datetime(panel["as_of_date"])
    panel["_next_q"] = as_of.dt.year * 4 + as_of.dt.quarter - 1
    panel["next_quarter"] = as_of.dt.year.astype(str) + "Q" + as_of.dt.quarter.astype(str)
    current = series.rename(columns={"cal_q": "_next_q", "revenue": "_rev", "rd_expense": "_rd"})[["ticker", "_next_q", "_rev", "_rd"]]
    year_ago = series.assign(_next_q=series["cal_q"] + 4).rename(columns={"revenue": "_rev_ly"})[["ticker", "_next_q", "_rev_ly"]]
    panel = panel.merge(current, on=["ticker", "_next_q"], how="left").merge(year_ago, on=["ticker", "_next_q"], how="left")
    panel["next_revenue_yoy"] = panel["_rev"] / positive(panel["_rev_ly"]) - 1
    panel["next_rd_intensity"] = panel["_rd"] / positive(panel["_rev"])

    cols = L.GOLD_SPINE_COLUMNS["firm_quarter"]
    extra = {"usable_from": "covariates: as_of_date; targets: measured after as_of_date"}
    L.write_gold("covariates", "firm_quarter", "market", panel[cols + COV_MARKET], builder=BUILDER, extra=extra)
    L.write_gold("targets", "firm_quarter", "market", panel[cols + TGT_MARKET], builder=BUILDER,
                 extra={**extra, "horizon": "the next quarter: [as_of_date, +63 trading days); ps_ratio_post at the next as_of_date"})
    L.write_gold("targets", "firm_quarter", "financials", panel[cols + TGT_FINANCIALS], builder=BUILDER,
                 extra={**extra, "horizon": "the fiscal quarter ending in the calendar quarter that starts at as_of_date"})


if __name__ == "__main__":
    main()
