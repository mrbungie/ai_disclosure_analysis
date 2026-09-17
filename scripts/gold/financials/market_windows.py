"""Daily price windows shared by the firm-year and call market builders.

Prices come from `silver.market_prices` (fs-backed, universe tickers,
nothing after a delisting); `ret` is the daily change of fs's own
dividend-inclusive `total_return` index, not a price return (orchestrator
decision, docs/plans/fs_gold_replacement.md S3.3). The market factor
(`mktrf`) comes from `silver.fs_market_factors_daily` -- fs's S&P 500
benchmark (`SP50-SPX`) total return in excess of `rf`; `rf` itself is kept
from the repo's existing Ken French source (fs has no usable risk-free
series). Every window is indexed in trading days of the firm's own price
series: `idx` = first trading day on or after the event date, `idx - 1` the
last trading day before it.

  one_factor_ols   OLS of excess return on `mktrf` (intercept, slope, residuals)
  ncskew           -[n (n-1)^1.5 sum w^3] / [(n-1)(n-2) (sum w^2)^1.5]
                   (Chen, Hong & Stein 2001), on market-model residuals
  duvol            ln{[(n_u - 1) sum_down w^2] / [(n_d - 1) sum_up w^2]}, with
                   down/up split at the residuals' own mean
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "common"))
import layers as L  # noqa: E402


def load_prices(tickers: set[str] | None = None) -> dict[str, pd.DataFrame]:
    """One frame per ticker (date, adj_close, close, market_cap, volume, ret),
    trading days with a total_return value, sorted by date; `ret` = daily
    change of fs's dividend-inclusive total_return index (not a price
    return). `volume` is fs's raw daily share volume (used by the weekly
    crash-risk family's DTURN control, docs/plans/cap5_classic_specs.md)."""
    all_prices = L.read("silver.market_prices").to_pandas()
    if tickers is not None:
        all_prices = all_prices[all_prices["ticker"].isin(tickers)]
    out = {}
    for ticker, prices in all_prices.groupby("ticker"):
        prices = prices[["date", "adj_close", "close", "total_return", "market_cap", "volume"]].copy()
        prices["date"] = pd.to_datetime(prices["date"])
        prices = prices.dropna(subset=["total_return"]).sort_values("date").reset_index(drop=True)
        prices["ret"] = prices["total_return"].pct_change()
        out[ticker] = prices
    return out


def load_factors() -> pd.DataFrame:
    factors = L.read("silver.fs_market_factors_daily").select(["date", "mktrf", "rf"]).to_pandas()
    factors["date"] = pd.to_datetime(factors["date"])
    return factors


def event_index(prices: pd.DataFrame, date) -> int:
    """First trading day on or after `date`."""
    return int(np.searchsorted(prices["date"].values, np.datetime64(date), side="left"))


def with_factors(window: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
    return window.merge(factors, on="date", how="inner").dropna(subset=["ret", "mktrf", "rf"])


def one_factor_ols(merged: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(coef [intercept, beta], excess returns, residuals) of excess ~ mktrf."""
    excess = merged["ret"].to_numpy() - merged["rf"].to_numpy()
    design = np.column_stack([np.ones(len(merged)), merged["mktrf"].to_numpy()])
    coef = np.linalg.lstsq(design, excess, rcond=None)[0]
    return coef, excess, excess - design @ coef


def beta(merged: pd.DataFrame, min_obs: int) -> float:
    if len(merged) < min_obs:
        return np.nan
    return float(one_factor_ols(merged)[0][1])


def ncskew(residuals: np.ndarray, min_obs: int) -> float:
    n = len(residuals)
    if n < min_obs:
        return np.nan
    s2 = np.sum(residuals ** 2)
    s3 = np.sum(residuals ** 3)
    if s2 <= 0:
        return np.nan
    denom = (n - 1) * (n - 2) * s2 ** 1.5
    if denom == 0:
        return np.nan
    return float(-(n * (n - 1) ** 1.5 * s3) / denom)


def duvol(residuals: np.ndarray, min_obs: int) -> float:
    n = len(residuals)
    if n < min_obs:
        return np.nan
    mean = residuals.mean()
    down = residuals[residuals < mean]
    up = residuals[residuals >= mean]
    n_d, n_u = len(down), len(up)
    if n_d < 2 or n_u < 2:
        return np.nan
    s2_down = np.sum(down ** 2)
    s2_up = np.sum(up ** 2)
    if s2_down <= 0 or s2_up <= 0:
        return np.nan
    ratio = ((n_u - 1) * s2_down) / ((n_d - 1) * s2_up)
    if ratio <= 0:
        return np.nan
    return float(np.log(ratio))


# ---- weekly panel + expanded (leads/lags) market model, Chen-Hong-Stein 2001
# / Kim-Li-Zhang 2011 crash-risk design -----------------------------------
#
# "Weeks" here are TRADING weeks: calendar weeks (pandas period "W", ending
# Sunday) that contain at least one trading day, kept in their traded order.
# A window of "26 weeks" is therefore 26 rows of this per-ticker weekly
# series, not strictly 26 elapsed calendar weeks -- the same convention the
# rest of this module uses for trading days (idx offsets are row positions,
# not calendar-day offsets). Documented in docs/plans/cap5_classic_specs.md.

def to_weekly(pr: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
    """Per-ticker weekly panel from daily prices+factors (`pr` must already
    carry `ret`; see market_windows.load_prices). `excess_w` compounds the
    firm's daily total return and the daily risk-free rate separately over
    the week and takes their difference; `mktrf_w` compounds fs's daily
    market-excess return directly (mktrf is already an excess return, so
    compounding it is the same first-order approximation the rest of this
    module's daily regressions already make at the daily frequency)."""
    m = pr.merge(factors, on="date", how="inner").dropna(subset=["ret", "mktrf", "rf"]).sort_values("date")
    if m.empty:
        return pd.DataFrame(columns=["week_ord", "excess_w", "mktrf_w", "volume_sum", "n_days"])
    m = m.assign(week=m["date"].dt.to_period("W"))
    g = m.groupby("week", sort=True)
    stock_gross = g["ret"].apply(lambda s: float(np.prod(1 + s.to_numpy())))
    rf_gross = g["rf"].apply(lambda s: float(np.prod(1 + s.to_numpy())))
    mkt_gross = g["mktrf"].apply(lambda s: float(np.prod(1 + s.to_numpy())))
    vol_col = "volume" if "volume" in m.columns else None
    out = pd.DataFrame({
        "week_ord": [p.ordinal for p in stock_gross.index],
        "excess_w": stock_gross.to_numpy() - rf_gross.to_numpy(),
        "mktrf_w": mkt_gross.to_numpy() - 1.0,
        "volume_sum": g[vol_col].sum().to_numpy() if vol_col else np.nan,
        "n_days": g.size().to_numpy(),
    }).sort_values("week_ord").reset_index(drop=True)
    return out


def week_index(weekly: pd.DataFrame, date) -> int:
    """First trading week (row of `weekly`) on or after `date`."""
    target = pd.Timestamp(date).to_period("W").ordinal
    return int(np.searchsorted(weekly["week_ord"].to_numpy(), target, side="left"))


def expanded_market_residuals(weekly: pd.DataFrame, start: int, end: int, min_obs: int,
                              leads_lags: int = 2) -> np.ndarray | None:
    """Residuals of the CHS 2001 expanded market model (excess_w on mktrf_w at
    lags/leads -`leads_lags`..+`leads_lags`) over rows [start, end) of `weekly`,
    using up to `leads_lags` extra rows on each side ONLY to build the
    lead/lag regressors so the window itself does not lose observations at
    its edges. Returns None if fewer than `min_obs` window rows have every
    lead/lag available (this can still happen at the very edge of a ticker's
    own weekly series)."""
    lo, hi = max(0, start - leads_lags), min(len(weekly), end + leads_lags)
    if lo >= hi or start < 0 or end > len(weekly) or end <= start:
        return None
    seg = weekly.iloc[lo:hi].reset_index(drop=True)
    cols = []
    for k in range(-leads_lags, leads_lags + 1):
        col = f"mkt_{k}"
        seg[col] = seg["mktrf_w"].shift(-k)
        cols.append(col)
    core_lo, core_hi = start - lo, end - lo
    core = seg.iloc[core_lo:core_hi].dropna(subset=[*cols, "excess_w"])
    if len(core) < min_obs:
        return None
    design = np.column_stack([np.ones(len(core)), *[core[c].to_numpy() for c in cols]])
    coef, *_ = np.linalg.lstsq(design, core["excess_w"].to_numpy(), rcond=None)
    return core["excess_w"].to_numpy() - design @ coef
