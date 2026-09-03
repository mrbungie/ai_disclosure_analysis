"""
scripts/cl/03_fetch_market_data.py — daily adjusted prices for Chile's firm
universe, parallel to scripts/03_market_data/01_collect_market_data.py (US).
Same tool (yfinance), same storage layout and schema — see that script's
own docstring for the full contract (per-ticker parquet, incremental,
CRSP-drop-in `source` column). Chile doesn't get a separate directory:
Yahoo Finance serves Bolsa de Santiago under `<NEMO>.SN`, so e.g.
`COPEC.SN.parquet` sits in the SAME data/raw/market/prices/ directory as
`AAPL.parquet` with zero collision risk — see docs/sources/market_prices.md.

Universe: configs/cl/universe.csv's `nemo` column (same nemo used
throughout the rest of the Chile pipeline — no separate ticker mapping).

Usage:
    uv run python scripts/cl/03_fetch_market_data.py [--refresh] [--nemo COPEC,CHILE]
"""

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))  # scripts/common/
import pipeline_logger

PRICES_DIR = Path("data/raw/market/prices")
DEFAULT_START = "2021-01-01"


def yahoo_symbol(nemo: str) -> str:
    return f"{nemo}.SN"


def fetch_prices(nemo: str, start: str) -> pd.DataFrame | None:
    import yfinance as yf
    symbol = yahoo_symbol(nemo)
    # tickers=[symbol], not a bare string: yf.download splits a plain string
    # on whitespace into MULTIPLE tickers — bit a real nemo with a space in
    # it ("AZUL AZUL" -> Yahoo "AZUL AZUL.SN"), which silently fetched two
    # tickers' OHLCV into one MultiIndex and produced duplicate columns
    # after flattening below.
    df = yf.download(tickers=[symbol], start=start, auto_adjust=False, progress=False)
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index().rename(columns={
        "Date": "date", "Adj Close": "adj_close", "Close": "close", "Volume": "volume",
    })
    df["ticker"] = nemo
    df["source"] = "yfinance"
    df["retrieved_at"] = datetime.now(timezone.utc)
    return df[["date", "adj_close", "close", "volume", "ticker", "source", "retrieved_at"]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--refresh", action="store_true", help="Re-download even if a parquet already exists")
    parser.add_argument("--nemo", default=None, help="Comma-separated subset (smoke-testing)")
    args = parser.parse_args()

    universe = pd.read_csv("configs/cl/universe.csv")
    nemos = universe["nemo"].tolist()
    if args.nemo:
        wanted = set(args.nemo.split(","))
        nemos = [n for n in nemos if n in wanted]

    PRICES_DIR.mkdir(parents=True, exist_ok=True)
    manifest_dir = Path("data/interim/manifests_cl")

    n_ok, n_missing = 0, 0
    for i, nemo in enumerate(nemos, 1):
        out_path = PRICES_DIR / f"{yahoo_symbol(nemo)}.parquet"
        if out_path.exists() and not args.refresh:
            continue
        t0 = time.monotonic()
        try:
            df = fetch_prices(nemo, args.start)
        except Exception as e:
            pipeline_logger.log_event(
                pipeline_step="cl_market_data", level="ERROR",
                message=f"Exception fetching prices: {e}", ticker=nemo,
                duration_seconds=time.monotonic() - t0, log_dir=manifest_dir,
            )
            n_missing += 1
            continue
        if df is None:
            pipeline_logger.log_event(
                pipeline_step="cl_market_data", level="WARNING",
                message="No price data returned by yfinance", ticker=nemo,
                duration_seconds=time.monotonic() - t0, log_dir=manifest_dir,
            )
            n_missing += 1
            continue
        df.to_parquet(out_path, index=False)
        pipeline_logger.log_event(
            pipeline_step="cl_market_data", level="SUCCESS",
            message=f"[{i}/{len(nemos)}] Fetched {len(df)} price rows", ticker=nemo,
            duration_seconds=time.monotonic() - t0, details={"rows": len(df)}, log_dir=manifest_dir,
        )
        n_ok += 1

    print(f"Done. {n_ok} tickers fetched, {n_missing} missing/failed -> {PRICES_DIR}")


if __name__ == "__main__":
    main()
