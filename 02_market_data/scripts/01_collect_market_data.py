"""
13_collect_market_data.py — Market data snapshot: daily adjusted prices per
ticker (yfinance) + Fama-French factors (Ken French data library), stored
partitioned so the snapshot is easy to EXTEND and hard to corrupt.

Storage layout (one file per unit — adding tickers later = just re-run; a
broken download never touches other tickers' files):

    data/raw/market/prices/{TICKER}.parquet    date, adj_close, close, volume,
                                               ticker, source, retrieved_at
    data/raw/market/factors/ff3_daily.parquet  date, mktrf, smb, hml, rf, ...
    data/raw/market/factors/ff3_monthly.parquet

Incremental: a ticker whose file already exists is skipped (use --refresh to
re-download). Run it again after adding tickers via the TUI and only the new
ones are fetched. Analysis derives returns from the stored prices — nothing
downstream re-hits the network.

CRSP drop-in (schema contract): every price file carries a `source` column
('yfinance' here). A CRSP/WRDS export loaded into the same per-ticker layout
with source='crsp' is a strict upgrade the analysis picks up without code
changes — that is also the only clean path to returns for DELISTED firms,
which yfinance does not serve (their EDGAR filings still flow through the
01_10k text pipeline; only their prices need CRSP). Survivorship note: the
firm universe itself (configs/config.json pipeline.tickers) was hand-
assembled from currently listed firms — a limitation of the universe, not
of this collector.

Usage:
    uv run python 02_market_data/scripts/01_collect_market_data.py [--start 2020-01-01] [--refresh] [--tickers AAPL,MSFT]
"""

import argparse
import io
import json
import os
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, for common/

try:
    import pipeline_logger
except ImportError:
    from common import pipeline_logger

PRICES_DIR = Path("data/raw/market/prices")
FACTORS_DIR = Path("data/raw/market/factors")

# Config ticker -> Yahoo symbol, for renames (the EDGAR/text pipeline keeps
# the config ticker; only the price fetch uses the alias).
YAHOO_ALIASES = {
    "SQ": "XYZ",  # Block renamed SQ -> XYZ (Jan 2025); Yahoo serves the full series under XYZ
}

FF_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp"
FF_FILES = {
    "ff3_monthly": ("F-F_Research_Data_Factors_CSV.zip", "%Y%m"),
    "ff3_daily": ("F-F_Research_Data_Factors_daily_CSV.zip", "%Y%m%d"),
}


def load_tickers() -> list[str]:
    with open("configs/config.json") as f:
        return json.load(f)["pipeline"]["tickers"]


def fetch_prices(ticker: str, start: str) -> pd.DataFrame | None:
    import yfinance as yf
    symbol = YAHOO_ALIASES.get(ticker, ticker)
    df = yf.download(symbol, start=start, auto_adjust=False, progress=False)
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    out = pd.DataFrame({
        "date": pd.to_datetime(df.index),
        "adj_close": df["Adj Close"].to_numpy(dtype=float),
        "close": df["Close"].to_numpy(dtype=float),
        "volume": df["Volume"].to_numpy(dtype=float),
    })
    out["ticker"] = ticker
    out["source"] = "yfinance" if symbol == ticker else f"yfinance:{symbol}"
    out["retrieved_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return out


def fetch_prices_stooq(ticker: str, start: str) -> pd.DataFrame | None:
    """Fallback for tickers Yahoo no longer serves (delisted, e.g. DFS after
    the Capital One acquisition). Stooq keeps delisted US histories, free, no
    key. Caveat carried in `source`: Stooq adjusts for splits but NOT
    dividends, so adj_close here is a split-adjusted close — fine for event
    windows, mildly understates long-horizon total returns."""
    url = f"https://stooq.com/q/d/l/?s={ticker.lower()}.us&i=d"
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    if not resp.text.startswith("Date,"):
        return None
    df = pd.read_csv(io.StringIO(resp.text), parse_dates=["Date"])
    df = df[df["Date"] >= pd.Timestamp(start)]
    if df.empty:
        return None
    out = pd.DataFrame({
        "date": df["Date"],
        "adj_close": df["Close"].to_numpy(dtype=float),  # split-adjusted only
        "close": df["Close"].to_numpy(dtype=float),
        "volume": df.get("Volume", pd.Series(dtype=float)).to_numpy(dtype=float),
    })
    out["ticker"] = ticker
    out["source"] = "stooq"
    out["retrieved_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return out


def fetch_prices_tiingo(ticker: str, start: str) -> pd.DataFrame | None:
    """Last-resort fallback for delisted tickers neither Yahoo nor Stooq
    serve (e.g. DFS after the Capital One acquisition). Tiingo keeps delisted
    histories with real dividend-adjusted closes; free API key at tiingo.com,
    set TIINGO_API_KEY in .env. Skipped silently when no key is set."""
    key = os.environ.get("TIINGO_API_KEY")
    if not key:
        return None
    url = f"https://api.tiingo.com/tiingo/daily/{ticker.lower()}/prices"
    resp = requests.get(url, params={"startDate": start, "token": key, "format": "json"}, timeout=60)
    if resp.status_code != 200:
        return None
    data = resp.json()
    if not data:
        return None
    df = pd.DataFrame(data)
    out = pd.DataFrame({
        "date": pd.to_datetime(df["date"]).dt.tz_localize(None),
        "adj_close": df["adjClose"].astype(float),
        "close": df["close"].astype(float),
        "volume": df["volume"].astype(float),
    })
    out["ticker"] = ticker
    out["source"] = "tiingo"
    out["retrieved_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return out


def collect_prices(tickers: list[str], start: str, refresh: bool, delay: float) -> tuple[int, int, list[str]]:
    PRICES_DIR.mkdir(parents=True, exist_ok=True)
    done, skipped, failed = 0, 0, []
    for ticker in tickers:
        path = PRICES_DIR / f"{ticker}.parquet"
        if path.exists() and not refresh:
            skipped += 1
            continue
        try:
            df = fetch_prices(ticker, start)
        except Exception as e:
            print(f"  {ticker}: yfinance error: {e}")
            df = None
        if df is None or df.empty:
            try:
                df = fetch_prices_stooq(ticker, start)
                if df is not None:
                    print(f"  {ticker}: not on Yahoo — using Stooq fallback (split-adjusted only)")
            except Exception as e:
                print(f"  {ticker}: stooq error: {e}")
                df = None
        if df is None or df.empty:
            try:
                df = fetch_prices_tiingo(ticker, start)
                if df is not None:
                    print(f"  {ticker}: using Tiingo fallback")
            except Exception as e:
                print(f"  {ticker}: tiingo error: {e}")
                df = None
        if df is None or df.empty:
            print(f"  {ticker}: no data from any source"
                  + ("" if os.environ.get("TIINGO_API_KEY")
                     else " (a free TIINGO_API_KEY in .env would try Tiingo, which keeps delisted histories)"))
            failed.append(ticker)
            continue
        df.to_parquet(path, index=False)
        done += 1
        print(f"  {ticker}: {len(df)} days ({df['date'].min().date()} -> {df['date'].max().date()})")
        if delay > 0:
            time.sleep(delay)
    return done, skipped, failed


def parse_ff_csv(raw_text: str, date_fmt: str) -> pd.DataFrame:
    """Parse a Ken French CSV: keep only rows whose first field is a date of
    the expected width (drops headers, the annual block, and copyright)."""
    width = len(datetime(2000, 1, 1).strftime(date_fmt))
    rows = []
    for line in raw_text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 5 and parts[0].isdigit() and len(parts[0]) == width:
            rows.append(parts[:5])
    df = pd.DataFrame(rows, columns=["date", "mktrf", "smb", "hml", "rf"])
    df["date"] = pd.to_datetime(df["date"], format=date_fmt)
    for col in ["mktrf", "smb", "hml", "rf"]:
        df[col] = pd.to_numeric(df[col], errors="coerce") / 100.0  # percent -> decimal
    df["source"] = "ken_french"
    df["retrieved_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return df.dropna()


def collect_factors(refresh: bool) -> int:
    FACTORS_DIR.mkdir(parents=True, exist_ok=True)
    n = 0
    for name, (zip_name, date_fmt) in FF_FILES.items():
        path = FACTORS_DIR / f"{name}.parquet"
        if path.exists() and not refresh:
            continue
        resp = requests.get(f"{FF_BASE}/{zip_name}", timeout=60)
        resp.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            raw = zf.read(zf.namelist()[0]).decode("latin-1")
        df = parse_ff_csv(raw, date_fmt)
        df.to_parquet(path, index=False)
        n += 1
        print(f"  {name}: {len(df)} rows ({df['date'].min().date()} -> {df['date'].max().date()}) -> {path}")
    return n


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2020-01-01", help="Price history start date")
    parser.add_argument("--refresh", action="store_true", help="Re-download files that already exist")
    parser.add_argument("--tickers", default=None, help="Comma-separated subset (default: all config tickers)")
    parser.add_argument("--delay", type=float, default=0.5, help="Seconds between ticker downloads")
    args = parser.parse_args()

    tickers = ([t.strip().upper() for t in args.tickers.split(",") if t.strip()]
               if args.tickers else load_tickers())

    print(f"Prices: {len(tickers)} tickers -> {PRICES_DIR}/ (one parquet per ticker)")
    done, skipped, failed = collect_prices(tickers, args.start, args.refresh, args.delay)
    print(f"Downloaded {done}, skipped {skipped} existing, failed {len(failed)}"
          + (f": {', '.join(failed)}" if failed else ""))

    print(f"\nFama-French factors -> {FACTORS_DIR}/")
    n_factors = collect_factors(args.refresh)
    if n_factors == 0:
        print("  (all factor files already present — use --refresh to update)")

    pipeline_logger.log_event(
        pipeline_step="market_data",
        level="SUCCESS" if not failed else "WARNING",
        message=f"Market data snapshot: {done} tickers downloaded, {skipped} skipped, {len(failed)} failed.",
        details={"failed": failed, "n_factor_files": n_factors},
    )


if __name__ == "__main__":
    main()
