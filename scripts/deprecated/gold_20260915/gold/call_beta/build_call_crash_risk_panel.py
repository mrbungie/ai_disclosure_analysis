"""Builds NCSKEW/DUVOL pre- and post-call tail-risk measures, one row per
earnings call. No producer for this panel existed anywhere in the repo or
its git history before this script (it was reverse-engineered from the
methodology documented in thesis_document/thesis.qmd, Section 6.5.1: NCSKEW
and DUVOL estimated over [+21, +126] trading days post-call, with a
symmetric [-126, -21] lagged pre-call window as the control).

Method (same market-model residual construction `build_ncskew_63.py` uses
for its 63-day sibling window):
  - idiosyncratic daily returns = residuals from a one-factor market-model
    regression (excess return on `mktrf`), fit separately within each
    window, using `bronze.market_prices` / `bronze.market_factors_daily`
    (not raw CSVs -- gold reads bronze, not data/raw, per G-M6).
  - NCSKEW = -[n(n-1)^1.5 * sum(w^3)] / [(n-1)(n-2) * (sum(w^2))^1.5]
    (Chen, Hong & Stein 2001).
  - DUVOL = ln{ [(n_u - 1) * sum_down(w^2)] / [(n_d - 1) * sum_up(w^2)] },
    where "down"/"up" split residuals below/above their own window mean.
  - Pre window: trading days [-252, -21) before the call date -- the same
    trailing-year span `beta_pre` uses, shifted back by the 21-day gap so
    the pre-call crash-risk control never overlaps the call's immediate
    run-up. Reverse-engineered against the stale `call_crash_risk_panel`
    parquet (see below): candidate windows [-126,-21) and [-252,0) both
    correlate under 0.70 with the stale `ncskew_pre`/`duvol_pre`, while
    [-252,-21) correlates at 0.98.
  - Post window: trading days [+21, +126) after the call date, matching
    thesis_document/thesis.qmd Section 6.5.1 ("NCSKEW and DUVOL ... over
    [+21, +126] trading days post-call"). This window alone (independent of
    the pre-window search) correlates at 0.99 with the stale panel's
    `ncskew_post`/`duvol_post`.
  - MIN_OBS = 90 for both windows (~86% of the shorter, 105-observation
    post window; the same completeness ratio `build_ncskew_63.py` uses for
    its 63-day window at BETA_MIN_OBS_63=60).

Output: data/gold/covariates/call/crash_risk_pre.parquet (ncskew_pre,
duvol_pre -- observed as of the call, safe as a control) and
data/gold/targets/call/crash_risk_post.parquet (ncskew_post_105d,
duvol_post_105d -- the outcome).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "common"))
import layers as L  # noqa: E402

PRE_WINDOW = 231  # trading days: [-252,-21) is 231 sessions
POST_WINDOW = 105  # trading days: [+21,+126) is 105 sessions
GAP = 21
MIN_OBS = 90


def ncskew(residuals: np.ndarray) -> float:
    n = len(residuals)
    if n < MIN_OBS:
        return np.nan
    s2 = np.sum(residuals ** 2)
    s3 = np.sum(residuals ** 3)
    if s2 <= 0:
        return np.nan
    denom = (n - 1) * (n - 2) * s2 ** 1.5
    if denom == 0:
        return np.nan
    numer = -(n * (n - 1) ** 1.5 * s3)
    return float(numer / denom)


def duvol(residuals: np.ndarray) -> float:
    n = len(residuals)
    if n < MIN_OBS:
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


def load_prices() -> dict[str, pd.DataFrame]:
    prices = L.read("bronze.market_prices").to_pandas()
    out: dict[str, pd.DataFrame] = {}
    for ticker, group in prices.groupby("ticker"):
        d = group[["date", "adj_close"]].dropna(subset=["adj_close"]).sort_values("date").reset_index(drop=True)
        d["date"] = pd.to_datetime(d["date"])
        d["ret"] = d["adj_close"].pct_change()
        out[ticker] = d
    return out


def window_residuals(pr: pd.DataFrame, factors: pd.DataFrame, start: int, end: int) -> np.ndarray | None:
    win = pr.iloc[start:end].merge(factors, on="date", how="inner").dropna(subset=["ret", "mktrf", "rf"])
    if len(win) < MIN_OBS:
        return None
    y = win["ret"].to_numpy() - win["rf"].to_numpy()
    X = np.column_stack([np.ones(len(win)), win["mktrf"].to_numpy()])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ coef


def main() -> None:
    calls = pd.read_parquet(REPO_ROOT / "data/gold/datasets/call/call.parquet",
                            columns=["ticker", "fecha", "call_accession_number"])
    calls["fecha"] = pd.to_datetime(calls["fecha"])

    factors = L.read("bronze.market_factors_daily").select(["date", "mktrf", "rf"]).to_pandas()
    factors["date"] = pd.to_datetime(factors["date"])
    prices = load_prices()

    rows = []
    for row in calls.itertuples(index=False):
        pr = prices.get(row.ticker)
        rec = {"ticker": row.ticker, "fecha": row.fecha, "call_accession_number": row.call_accession_number,
               "ncskew_pre": np.nan, "duvol_pre": np.nan, "ncskew_post_105d": np.nan, "duvol_post_105d": np.nan}
        if pr is not None:
            dates = pr["date"].values
            idx = int(np.searchsorted(dates, np.datetime64(row.fecha), side="left"))
            pre_start, pre_end = idx - GAP - PRE_WINDOW, idx - GAP
            post_start, post_end = idx + GAP, idx + GAP + POST_WINDOW
            if pre_start >= 0:
                resid_pre = window_residuals(pr, factors, pre_start, pre_end)
                if resid_pre is not None:
                    rec["ncskew_pre"] = ncskew(resid_pre)
                    rec["duvol_pre"] = duvol(resid_pre)
            if post_end <= len(pr):
                resid_post = window_residuals(pr, factors, post_start, post_end)
                if resid_post is not None:
                    rec["ncskew_post_105d"] = ncskew(resid_post)
                    rec["duvol_post_105d"] = duvol(resid_post)
        rows.append(rec)

    out = pd.DataFrame(rows)
    out.insert(0, "id", out["call_accession_number"])

    pre = out[["id", "call_accession_number", "ticker", "fecha", "ncskew_pre", "duvol_pre"]]
    post = out[["id", "call_accession_number", "ticker", "fecha", "ncskew_post_105d", "duvol_post_105d"]]

    pre_path = L.gold_path("covariates", "call", "crash_risk_pre")
    post_path = L.gold_path("targets", "call", "crash_risk_post")
    pre.to_parquet(pre_path, index=False)
    post.to_parquet(post_path, index=False)
    print(f"-> {pre_path} ({len(pre):,} rows)")
    print(f"-> {post_path} ({len(post):,} rows)")
    print(f"ncskew_pre non-null: {pre.ncskew_pre.notna().mean()*100:.1f}%  "
          f"ncskew_post_105d non-null: {post.ncskew_post_105d.notna().mean()*100:.1f}%")


if __name__ == "__main__":
    main()
