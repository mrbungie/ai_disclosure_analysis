"""El lado de MERCADO del panel empresa-año (librería de
`scripts/gold/firm_year/build_market_financials.py`): beta, volatilidad,
momentum y retorno alrededor de cada 10-K, y múltiplos de valuación.

`filing_market_panel(annual)` devuelve una fila por 10-K con precios.

Definiciones (idénticas a las documentadas):

  idio_vol_252d desv. est. de los residuos de esa regresión x sqrt(252): la
                volatilidad que no explica el mercado
  beta          OLS de `retorno_exceso ~ mktrf` sobre los 252 días hábiles
                ANTERIORES al filing, mínimo 120 observaciones. Un factor
                (mercado), no Fama-French de 3 factores.
  vol_pre/post  desv. est. de retornos diarios x sqrt(252) en ventanas de 60
                días hábiles antes y después del filing.
  momentum_12_1 retorno acumulado de t-252 a t-21 — salta el último mes,
                convención estándar para no capturar reversión de corto plazo.
  ret_m1_p5     retorno CRUDO: precio ajustado en +5 / precio en -1 - 1.

  market_cap    fs's own daily market cap on the last trading day BEFORE the
                filing (direct read, not price x shares -- fixes the
                pre-split understatement documented in
                data/results/fs_validation/summary.md: the old build
                multiplied an already split-adjusted close by a historical,
                pre-split cover-page share count).
  P/E           precio / EPS diluido.  EV = market_cap + deuda LP - caja.

El día 0 es el primer día hábil >= `filing_date`; la ventana empieza en el
día hábil anterior. Todo se ancla a `filing_date` real de punta a punta, sin
pasar por buckets de año calendario (docs/analytics/leakage-checking.md).

Limitaciones que quedan en los datos, no en el código:
  - Los precios empiezan en 2021-01-04, así que un 10-K de principios de 2021
    no tiene 252 días previos: beta sale de menos observaciones (mínimo 120) o
    queda nulo, y `momentum_12_1` queda nulo.
  - Los factores Fama-French del repo terminan antes que los precios, así que
    beta queda nulo para los filings más recientes.

Determinístico, sin LLM, sin costo de API.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
import sys  # noqa: E402
sys.path.insert(0, str(REPO_ROOT / "scripts" / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from market_windows import load_factors, load_prices  # noqa: E402

BETA_WINDOW = 252
BETA_MIN_OBS = 120
VOL_WINDOW = 60
MOMENTUM_START, MOMENTUM_END = 252, 21
EVENT_PRE, EVENT_POST = 1, 5


def window_metrics(prices: pd.DataFrame, factors: pd.DataFrame,
                   filing_date: pd.Timestamp) -> dict:
    """Todas las métricas de mercado de un (ticker, filing_date).

    `idx` es el primer día hábil >= filing_date. El evento se mide desde el
    día hábil ANTERIOR (idx-1) porque el 10-K puede presentarse después del
    cierre del mercado, así que el precio de referencia limpio es el previo."""
    dates = prices["date"].values
    idx = int(np.searchsorted(dates, np.datetime64(filing_date), side="left"))
    result = {"beta": np.nan, "idio_vol_252d": np.nan, "vol_pre_60d": np.nan, "vol_post_60d": np.nan,
              "momentum_12_1": np.nan, "ret_m1_p5": np.nan,
              "price_pre": np.nan, "market_cap": np.nan, "beta_n_obs": np.nan}
    if idx - EVENT_PRE < 0 or idx >= len(prices):
        return result
    result["price_pre"] = float(prices["close"].iloc[idx - EVENT_PRE])
    mcap_pre = prices["market_cap"].iloc[idx - EVENT_PRE]
    if pd.notna(mcap_pre):
        result["market_cap"] = float(mcap_pre)

    if idx + EVENT_POST < len(prices):
        p0 = prices["adj_close"].iloc[idx - EVENT_PRE]
        p1 = prices["adj_close"].iloc[idx + EVENT_POST]
        if p0 > 0:
            result["ret_m1_p5"] = float(p1 / p0 - 1)

    pre = prices.iloc[max(0, idx - BETA_WINDOW):idx]
    merged = pre.merge(factors, on="date", how="inner").dropna(subset=["ret", "mktrf", "rf"])
    result["beta_n_obs"] = len(merged)
    if len(merged) >= BETA_MIN_OBS:
        excess = merged["ret"] - merged["rf"]
        design = np.column_stack([np.ones(len(merged)), merged["mktrf"].values])
        coef = np.linalg.lstsq(design, excess.values, rcond=None)[0]
        result["beta"] = float(coef[1])
        # volatilidad idiosincrática: desv. est. de los residuos del modelo de
        # mercado sobre la misma ventana de 252 días, anualizada
        residuals = excess.values - design @ coef
        result["idio_vol_252d"] = float(residuals.std(ddof=2) * np.sqrt(252))

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


def filing_market_panel(annual: pd.DataFrame) -> pd.DataFrame:
    """Market metrics and valuation multiples per 10-K of `annual`
    (`build_firm_financials.annual_panel()`), anchored at its filing date."""
    ratios = annual.dropna(subset=["filing_date"]).reset_index(drop=True)
    ratios["filing_date"] = pd.to_datetime(ratios["filing_date"])
    factors = load_factors()
    prices = load_prices(set(ratios["ticker"]))
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
          f"(beta no nulo: {market['beta'].notna().mean()*100:.0f}%)")

    fundamentals = ratios[["ticker", "year", "shares_out", "eps_diluted", "revenue",
                           "equity", "ebitda", "long_term_debt", "cash"]]
    panel = market.merge(fundamentals, on=["ticker", "year"], how="left")
    # market_cap comes straight from window_metrics (fs's own daily market cap,
    # not price_pre * shares_out -- see the module docstring's market_cap note).
    panel["pe_ratio"] = safe_div(panel["price_pre"], panel["eps_diluted"])
    panel["ps_ratio"] = safe_div(panel["market_cap"], panel["revenue"])
    panel["pb_ratio"] = safe_div(panel["market_cap"], panel["equity"])
    # Sin deuda LP o caja reportada el EV queda nulo: gold no imputa ceros.
    enterprise_value = panel["market_cap"] + panel["long_term_debt"] - panel["cash"]
    panel["ev_revenue"] = safe_div(enterprise_value, panel["revenue"])
    panel["ev_ebitda"] = safe_div(enterprise_value, panel["ebitda"])
    columns = ["ticker", "year", "market_cap", "pe_ratio", "ps_ratio", "pb_ratio",
               "ev_revenue", "ev_ebitda", "beta", "idio_vol_252d", "vol_pre_60d", "vol_post_60d",
               "momentum_12_1", "ret_m1_p5", "beta_n_obs"]
    print("medianas:", {c: round(float(panel[c].median()), 3)
                        for c in ["beta", "vol_pre_60d", "momentum_12_1", "pe_ratio"]
                        if panel[c].notna().any()})
    return panel[columns]
