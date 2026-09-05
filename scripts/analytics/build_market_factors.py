"""Construye el lado de MERCADO del panel empresa-año: retorno de ventana,
beta, volatilidad, momentum, CAR ajustado por mercado y múltiplos.

Produce los dos parquets que `docs/analytics/02_market_accounting_crosscheck.md`
y `04_ratios_factors_and_volatility.md` describen en prosa y que hasta ahora
no tenían código en el repo:

  firm_year_filing_returns.parquet   retorno CRUDO [-1, +5 días hábiles]
                                     alrededor de la fecha del 10-K  (doc 02)
  firm_year_market_factors.parquet   market cap, P/E, P/S, P/B, EV/Revenue,
                                     EV/EBITDA, beta, volatilidad pre/post,
                                     momentum 12-1 y CAR ajustado  (doc 04)

Definiciones (idénticas a las documentadas):

  beta          OLS de `retorno_exceso ~ mktrf` sobre los 252 días hábiles
                ANTERIORES al filing, mínimo 120 observaciones. Un factor
                (mercado), no Fama-French de 3 factores.
  vol_pre/post  desv. est. de retornos diarios x sqrt(252) en ventanas de 60
                días hábiles antes y después del filing.
  momentum_12_1 retorno acumulado de t-252 a t-21 — salta el último mes,
                convención estándar para no capturar reversión de corto plazo.
  ret_m1_p5     retorno CRUDO: precio ajustado en +5 / precio en -1 - 1.
  car_m1_p5     suma de (retorno_exceso - beta * mktrf) en [-1, +5]. Esta es
                la versión ajustada por mercado; el retorno crudo NO lo está,
                y con tech en alza 2024-2026 esa diferencia es material.

  market_cap    precio de cierre (sin ajustar) del último día hábil ANTES del
                filing x `shares_out` de la portada del 10-K.
  P/E           precio / EPS diluido.  EV = market_cap + deuda LP - caja.

El día 0 es el primer día hábil >= `filing_date`; la ventana empieza en el
día hábil anterior. Todo se ancla a `filing_date` real de punta a punta, sin
pasar por buckets de año calendario (docs/analytics/leakage-checking.md).

Limitaciones que quedan en los datos, no en el código:
  - Los precios empiezan en 2021-01-04, así que un 10-K de principios de 2021
    no tiene 252 días previos: beta sale de menos observaciones (mínimo 120) o
    queda nulo, y `momentum_12_1` queda nulo.
  - Los factores Fama-French del repo terminan antes que los precios, así que
    beta y CAR quedan nulos para los filings más recientes.

Determinístico, sin LLM, sin costo de API.

Uso:
    uv run python scripts/analytics/build_market_factors.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DB = REPO_ROOT / "duckdb" / "thesis.duckdb"
PRICES_DIR = REPO_ROOT / "data" / "raw" / "market" / "prices"
FACTORS = REPO_ROOT / "data" / "raw" / "market" / "factors" / "ff3_daily.parquet"
CLUSTERS = REPO_ROOT / "data" / "processed" / "clusters"

BETA_WINDOW = 252
BETA_MIN_OBS = 120
VOL_WINDOW = 60
MOMENTUM_START, MOMENTUM_END = 252, 21
EVENT_PRE, EVENT_POST = 1, 5


def load_prices(directory: Path, tickers: set[str]) -> dict[str, pd.DataFrame]:
    """Un DataFrame de precios por ticker, ordenado por fecha.

    Los archivos de Chile viven en el mismo directorio con sufijo `.SN`; se
    ignoran acá porque este panel es sólo EE.UU."""
    out = {}
    for path in sorted(directory.glob("*.parquet")):
        ticker = path.stem
        if ticker not in tickers:
            continue
        prices = pd.read_parquet(path, columns=["date", "adj_close", "close"])
        prices["date"] = pd.to_datetime(prices["date"])
        prices = prices.dropna(subset=["adj_close"]).sort_values("date").reset_index(drop=True)
        prices["ret"] = prices["adj_close"].pct_change()
        out[ticker] = prices
    return out


def window_metrics(prices: pd.DataFrame, factors: pd.DataFrame,
                   filing_date: pd.Timestamp) -> dict:
    """Todas las métricas de mercado de un (ticker, filing_date).

    `idx` es el primer día hábil >= filing_date. El evento se mide desde el
    día hábil ANTERIOR (idx-1) porque el 10-K puede presentarse después del
    cierre del mercado, así que el precio de referencia limpio es el previo."""
    dates = prices["date"].values
    idx = int(np.searchsorted(dates, np.datetime64(filing_date), side="left"))
    result = {"beta": np.nan, "vol_pre_60d": np.nan, "vol_post_60d": np.nan,
              "momentum_12_1": np.nan, "ret_m1_p5": np.nan, "car_m1_p5": np.nan,
              "price_pre": np.nan}
    if idx - EVENT_PRE < 0 or idx >= len(prices):
        return result
    result["price_pre"] = float(prices["close"].iloc[idx - EVENT_PRE])

    if idx + EVENT_POST < len(prices):
        p0 = prices["adj_close"].iloc[idx - EVENT_PRE]
        p1 = prices["adj_close"].iloc[idx + EVENT_POST]
        if p0 > 0:
            result["ret_m1_p5"] = float(p1 / p0 - 1)

    pre = prices.iloc[max(0, idx - BETA_WINDOW):idx]
    merged = pre.merge(factors, on="date", how="inner").dropna(subset=["ret", "mktrf", "rf"])
    if len(merged) >= BETA_MIN_OBS:
        excess = merged["ret"] - merged["rf"]
        design = np.column_stack([np.ones(len(merged)), merged["mktrf"].values])
        beta = float(np.linalg.lstsq(design, excess.values, rcond=None)[0][1])
        result["beta"] = beta
        event = prices.iloc[idx - EVENT_PRE:idx + EVENT_POST + 1].merge(
            factors, on="date", how="inner").dropna(subset=["ret", "mktrf", "rf"])
        if len(event) > 1:
            abnormal = (event["ret"] - event["rf"]) - beta * event["mktrf"]
            result["car_m1_p5"] = float(abnormal.iloc[1:].sum())

    vol_pre = prices["ret"].iloc[max(0, idx - VOL_WINDOW):idx]
    if vol_pre.notna().sum() >= VOL_WINDOW // 2:
        result["vol_pre_60d"] = float(vol_pre.std() * np.sqrt(252))
    vol_post = prices["ret"].iloc[idx:idx + VOL_WINDOW]
    if vol_post.notna().sum() >= VOL_WINDOW // 2:
        result["vol_post_60d"] = float(vol_post.std() * np.sqrt(252))

    if idx - MOMENTUM_START >= 0:
        start = prices["adj_close"].iloc[idx - MOMENTUM_START]
        end = prices["adj_close"].iloc[idx - MOMENTUM_END]
        if start > 0:
            result["momentum_12_1"] = float(end / start - 1)
    return result


def safe_div(numerator, denominator, positive_denominator: bool = True):
    den = pd.Series(denominator, dtype="float64")
    den = den.where(den > 0) if positive_denominator else den.where(den != 0)
    return pd.Series(numerator, dtype="float64").values / den.values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DB)
    parser.add_argument("--prices-dir", type=Path, default=PRICES_DIR)
    parser.add_argument("--factors", type=Path, default=FACTORS)
    parser.add_argument("--clusters-dir", type=Path, default=CLUSTERS)
    args = parser.parse_args()

    ratios = pd.read_parquet(args.clusters_dir / "firm_year_financials_ratios.parquet")
    ratios["filing_date"] = pd.to_datetime(ratios["filing_date"])
    factors = pd.read_parquet(args.factors)[["date", "mktrf", "rf"]]
    factors["date"] = pd.to_datetime(factors["date"])
    prices = load_prices(args.prices_dir, set(ratios["ticker"]))
    print(f"{len(ratios):,} filings con contables | precios para "
          f"{len(prices):,}/{ratios['ticker'].nunique():,} tickers | "
          f"factores hasta {factors['date'].max().date()}")

    rows = []
    for row in ratios.itertuples(index=False):
        series = prices.get(row.ticker)
        if series is None or series.empty:
            continue
        metrics = window_metrics(series, factors, row.filing_date)
        metrics.update({"ticker": row.ticker, "year": row.year,
                        "filing_date": row.filing_date})
        rows.append(metrics)
    market = pd.DataFrame(rows)
    print(f"{len(market):,} filas con ventana de precios "
          f"(beta no nulo: {market['beta'].notna().mean()*100:.0f}%, "
          f"CAR: {market['car_m1_p5'].notna().mean()*100:.0f}%)")

    returns = market[["ticker", "year", "filing_date", "ret_m1_p5"]].dropna(
        subset=["ret_m1_p5"]).copy()

    fundamentals = ratios[["ticker", "year", "shares_out", "eps_diluted", "revenue",
                           "equity", "ebitda", "long_term_debt", "cash"]]
    panel = market.merge(fundamentals, on=["ticker", "year"], how="left")
    panel["market_cap"] = panel["price_pre"] * panel["shares_out"]
    panel["pe_ratio"] = safe_div(panel["price_pre"], panel["eps_diluted"])
    panel["ps_ratio"] = safe_div(panel["market_cap"], panel["revenue"])
    panel["pb_ratio"] = safe_div(panel["market_cap"], panel["equity"])
    enterprise_value = (panel["market_cap"] + panel["long_term_debt"].fillna(0)
                        - panel["cash"].fillna(0))
    panel["ev_revenue"] = safe_div(enterprise_value, panel["revenue"])
    panel["ev_ebitda"] = safe_div(enterprise_value, panel["ebitda"])

    columns = ["ticker", "year", "market_cap", "pe_ratio", "ps_ratio", "pb_ratio",
               "ev_revenue", "ev_ebitda", "beta", "vol_pre_60d", "vol_post_60d",
               "momentum_12_1", "car_m1_p5"]
    market_factors = panel[columns].sort_values(["ticker", "year"])

    args.clusters_dir.mkdir(parents=True, exist_ok=True)
    for name, table in (("firm_year_filing_returns", returns),
                        ("firm_year_market_factors", market_factors)):
        destination = args.clusters_dir / f"{name}.parquet"
        table.to_parquet(destination, index=False)
        print(f"-> {destination} ({len(table):,} filas, {len(table.columns)} cols)")
    print("medianas:", {c: round(float(market_factors[c].median()), 3)
                        for c in ["beta", "vol_pre_60d", "momentum_12_1", "pe_ratio"]
                        if market_factors[c].notna().any()})


if __name__ == "__main__":
    main()
