"""Builds ncskew_pre_63 / ncskew_post_63 from scratch, standard Chen-Hong-
Stein (2001) construction: idiosyncratic daily returns = residuals from a
market-model regression (same factors/prices already used for beta), then

  NCSKEW = -[n(n-1)^1.5 * sum(W^3)] / [(n-1)(n-2) * (sum(W^2))^1.5]

over a 63-trading-day window, symmetric with beta_post_63/beta_pre's own
[-252,0) / [0,+63) windows and BETA_MIN_OBS_63=60 minimum-observation bar.
"""
from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
CLUSTERS = REPO_ROOT / "data/processed/clusters"
PRICES_DIR = REPO_ROOT / "data/raw/market/prices"
FACTORS_PATH = REPO_ROOT / "data/raw/market/factors/ff3_daily.parquet"

MIN_OBS_63 = 60

def ncskew(residuals: np.ndarray) -> float:
    n = len(residuals)
    if n < MIN_OBS_63:
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

panel = pd.read_parquet(CLUSTERS / "call_beta_main_panel_10k10q_asof.parquet")
panel["fecha"] = pd.to_datetime(panel["fecha"])
factors = pd.read_parquet(FACTORS_PATH, columns=["date", "mktrf", "rf"])
factors["date"] = pd.to_datetime(factors["date"])

price_cache = {}
def get_price(tkr):
    if tkr not in price_cache:
        fp = PRICES_DIR / f"{tkr}.parquet"
        if not fp.exists():
            price_cache[tkr] = None
        else:
            d = pd.read_parquet(fp, columns=["date", "adj_close"])
            d["date"] = pd.to_datetime(d["date"])
            d = d.dropna(subset=["adj_close"]).sort_values("date").reset_index(drop=True)
            d["ret"] = d["adj_close"].pct_change()
            price_cache[tkr] = d
    return price_cache[tkr]

def window_ncskew(pr, start, end):
    win = pr.iloc[start:end].merge(factors, on="date", how="inner").dropna(subset=["ret", "mktrf", "rf"])
    if len(win) < MIN_OBS_63:
        return np.nan
    y = win["ret"].to_numpy() - win["rf"].to_numpy()
    X = np.column_stack([np.ones(len(win)), win["mktrf"].to_numpy()])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    return ncskew(resid)

rows = []
for row in panel[["ticker", "fecha", "call_accession_number"]].itertuples(index=False):
    pr = get_price(row.ticker)
    rec = {"ticker": row.ticker, "fecha": row.fecha, "call_accession_number": row.call_accession_number,
           "ncskew_pre_63": np.nan, "ncskew_post_63": np.nan}
    if pr is not None:
        dates = pr["date"].values
        idx = int(np.searchsorted(dates, np.datetime64(row.fecha), side="left"))
        if idx - 63 >= 0:
            rec["ncskew_pre_63"] = window_ncskew(pr, idx - 63, idx)
        if idx + 63 < len(pr):
            rec["ncskew_post_63"] = window_ncskew(pr, idx, idx + 63)
    rows.append(rec)

out = pd.DataFrame(rows)
out.to_parquet(CLUSTERS / "ncskew_63.parquet", index=False)
print(f"Built ncskew_63 for {len(out)} calls")
print(f"ncskew_pre_63 non-null: {out.ncskew_pre_63.notna().sum()} ({out.ncskew_pre_63.notna().mean()*100:.1f}%)")
print(f"ncskew_post_63 non-null: {out.ncskew_post_63.notna().sum()} ({out.ncskew_post_63.notna().mean()*100:.1f}%)")
print(out[["ticker","fecha","ncskew_pre_63","ncskew_post_63"]].dropna().head(5))
