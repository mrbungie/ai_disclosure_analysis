"""Daily price windows shared by the firm-year and call market builders.

Prices come from `silver.market_prices` (universe tickers, nothing after a
delisting), the market factor and risk-free rate
from `bronze.market_factors_daily`. Every window is indexed in trading days
of the firm's own price series: `idx` = first trading day on or after the
event date, `idx - 1` the last trading day before it.

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
    """One frame per ticker (date, adj_close, close, ret), trading days with an
    adjusted close, sorted by date; `ret` = daily adjusted return."""
    all_prices = L.read("silver.market_prices").to_pandas()
    if tickers is not None:
        all_prices = all_prices[all_prices["ticker"].isin(tickers)]
    out = {}
    for ticker, prices in all_prices.groupby("ticker"):
        prices = prices[["date", "adj_close", "close"]].copy()
        prices["date"] = pd.to_datetime(prices["date"])
        prices = prices.dropna(subset=["adj_close"]).sort_values("date").reset_index(drop=True)
        prices["ret"] = prices["adj_close"].pct_change()
        out[ticker] = prices
    return out


def load_factors() -> pd.DataFrame:
    factors = L.read("bronze.market_factors_daily").select(["date", "mktrf", "rf"]).to_pandas()
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
