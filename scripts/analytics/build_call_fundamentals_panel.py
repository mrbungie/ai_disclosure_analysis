"""Point-in-time pre/post fundamentals for the call-level generalization in
`call_beta_generalized_targets.py`, at the same precision the beta model
uses -- not a coarse fiscal-year bucket join.

For every call:
  PRE  = the last 10-K filed strictly BEFORE the call (backward merge_asof
         on `filing_date`, same join `build_call_beta_panel.py::
         attach_financials` already uses for its own pre-call controls).
  POST = the first 10-K filed strictly AFTER the call (forward merge_asof,
         same table, same key -- the natural post-call analogue).
Both come from `firm_year_financials_ratios.parquet`, so `gross_margin`,
`rd_intensity` and `next_revenue_yoy` are pulled as-is (already computed
per accession there), not recomputed. `ps_ratio` and `log_market_cap` use
the call's own pre-call price (`price_pre`, already in the call panel) and
a post-call price 125 trading days after the call (the last day of the
same [+21,+126]-style window `beta_post_126` is estimated over -- see
`build_call_beta_panel.py::attach_market`), against the matched filing's
shares outstanding. `roic_minus_wacc` reuses `build_roic_wacc.py`'s exact
formulas (ERP, risk-free rate, tax-rate and cost-of-debt bounds) applied to
these same point-in-time accounting snapshots and to the call panel's own
`beta_pre` / `beta_post_126` -- a finer beta than the annual one
`build_roic_wacc.py` uses for the firm-year economic profiles, since the
call panel already estimates beta at the exact pre/post windows this
generalization needs.

Determinístico, sin LLM. Requiere `call_beta_main_panel_10k10q_asof.parquet`,
`firm_year_financials_ratios.parquet`, precios en `data/raw/market/prices/`,
factores en `data/raw/market/factors/ff3_daily.parquet`.

Salida: `data/processed/clusters/call_fundamentals_panel.parquet`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_roic_wacc import annualized_equity_premium  # noqa: E402
# NOT importing build_roic_wacc.risk_free_at: it sorts its input by date and
# returns a plain RangeIndex aligned to that SORTED order, not to the
# caller's original row order -- fine where it's called on an
# already-appropriately-ordered frame, but silently misaligns here where
# `panel` is ordered by (ticker, fecha), not globally by date. `risk_free_at_ordered`
# below does the same backward-asof lookup but restores the caller's original
# row order explicitly.

REPO_ROOT = Path(__file__).resolve().parents[2]
CLUSTERS = REPO_ROOT / "data" / "processed" / "clusters"
PANEL_PATH = CLUSTERS / "call_beta_main_panel_10k10q_asof.parquet"
RATIOS_PATH = CLUSTERS / "firm_year_financials_ratios.parquet"
PRICES_DIR = REPO_ROOT / "data" / "raw" / "market" / "prices"
FACTORS_PATH = REPO_ROOT / "data" / "raw" / "market" / "factors" / "ff3_daily.parquet"
OUT_PATH = CLUSTERS / "call_fundamentals_panel.parquet"

RATIO_COLS = ["ticker", "filing_date", "accession_number", "revenue", "gross_margin", "rd_intensity",
              "next_revenue_yoy", "operating_income", "equity", "long_term_debt", "interest_expense",
              "pretax_income", "tax_expense", "shares_out"]
STATUTORY_TAX_RATE = 0.21
DEBT_SPREAD_FALLBACK = 0.02
MAX_COST_OF_DEBT = 0.30
ERP_SAMPLE_START = "2000-01-01"
POST_WINDOW_END_OFFSET = 125  # last trading day of beta_post_126's [idx, idx+126) window


def attach_ratio_snapshot(panel: pd.DataFrame, ratios: pd.DataFrame, direction: str, suffix: str) -> pd.DataFrame:
    left = panel[["ticker", "fecha"]].assign(fecha=lambda d: pd.to_datetime(d["fecha"]).astype("datetime64[ns]"))
    left = left.sort_values(["fecha", "ticker"])
    right = ratios.assign(filing_date=lambda d: pd.to_datetime(d["filing_date"]).astype("datetime64[ns]"))
    right = right.sort_values(["filing_date", "ticker"])
    merged = pd.merge_asof(left, right, left_on="fecha", right_on="filing_date", by="ticker",
                            direction=direction, allow_exact_matches=False)
    # `merge_asof` drops the index and returns rows in `left`'s (sorted) order,
    # not `panel`'s original row order -- restore it via `left.index` (sort_values
    # keeps the original labels attached to each row) before rejoining to `panel`.
    merged.index = left.index
    merged = merged.reindex(panel.index).drop(columns=["ticker", "fecha"]).add_suffix(f"_{suffix}")
    return pd.concat([panel, merged], axis=1)


def risk_free_at_ordered(factors: pd.DataFrame, dates: pd.Series) -> pd.Series:
    """Same backward-asof risk-free lookup as `build_roic_wacc.risk_free_at`,
    but returned in the caller's original row order (see the note on that
    import above -- `merge_asof` returns its result in sorted-by-date order,
    not in the order the input `dates` were given)."""
    daily = factors[["date", "rf"]].dropna().sort_values("date").reset_index(drop=True)
    dates = pd.to_datetime(dates)
    order = np.argsort(dates.to_numpy(), kind="stable")
    sorted_dates = dates.to_numpy()[order]
    idx = np.searchsorted(daily["date"].to_numpy(), sorted_dates, side="right") - 1
    idx = np.clip(idx, 0, len(daily) - 1)
    rf_sorted = daily["rf"].to_numpy()[idx]
    rf = np.empty(len(dates))
    rf[order] = rf_sorted
    return pd.Series((1 + rf) ** 252 - 1, index=dates.index)


def attach_post_price(panel: pd.DataFrame) -> pd.DataFrame:
    """`price_post`: close price `POST_WINDOW_END_OFFSET` trading days after
    the call -- the same day `beta_post_126`'s estimation window ends at."""
    rows = []
    tickers = set(panel["ticker"])
    prices = {}
    for t in tickers:
        p = PRICES_DIR / f"{t}.parquet"
        if not p.exists():
            continue
        d = pd.read_parquet(p, columns=["date", "close"])
        d["date"] = pd.to_datetime(d["date"])
        prices[t] = d.dropna(subset=["close"]).sort_values("date").reset_index(drop=True)
    for row in panel[["ticker", "fecha"]].itertuples(index=False):
        pr = prices.get(row.ticker)
        price_post = np.nan
        if pr is not None:
            dates = pr["date"].values
            idx = int(np.searchsorted(dates, np.datetime64(row.fecha), side="left"))
            end = idx + POST_WINDOW_END_OFFSET
            if 0 <= end < len(pr):
                price_post = float(pr["close"].iloc[end])
        rows.append({"ticker": row.ticker, "fecha": row.fecha, "price_post": price_post})
    return panel.merge(pd.DataFrame(rows), on=["ticker", "fecha"], how="left")


def roic_minus_wacc(rf: pd.Series, beta: pd.Series, erp: float, operating_income: pd.Series,
                     pretax_income: pd.Series, tax_expense: pd.Series, equity: pd.Series,
                     long_term_debt: pd.Series, interest_expense: pd.Series) -> pd.Series:
    """Exact formulas from `build_roic_wacc.py`, applied to a point-in-time
    snapshot instead of a firm-year one."""
    rate = tax_expense / pretax_income.where(pretax_income > 0)
    effective_tax_rate = rate.where(rate.between(0, 1)).fillna(STATUTORY_TAX_RATE)
    nopat = operating_income * (1 - effective_tax_rate)
    invested = long_term_debt.fillna(0) + equity
    roic = nopat / invested.where(invested > 0)
    cost_of_equity = rf + beta * erp
    debt = long_term_debt.where(long_term_debt > 0)
    cost_of_debt_raw = interest_expense / debt
    cost_of_debt = cost_of_debt_raw.where(cost_of_debt_raw.between(0, MAX_COST_OF_DEBT)).fillna(rf + DEBT_SPREAD_FALLBACK)
    debt_value = long_term_debt.fillna(0)
    total = equity + debt_value
    weight_equity = equity / total.where(total > 0)
    wacc = weight_equity * cost_of_equity + (1 - weight_equity) * cost_of_debt * (1 - effective_tax_rate)
    return roic - wacc


def main() -> None:
    panel = pd.read_parquet(PANEL_PATH)
    ratios = pd.read_parquet(RATIOS_PATH, columns=RATIO_COLS)

    panel = attach_ratio_snapshot(panel, ratios, "backward", "pre")
    panel = attach_ratio_snapshot(panel, ratios, "forward", "post")
    panel = attach_post_price(panel)

    panel["log_market_cap_pre"] = np.log(panel["price_pre"] * panel["shares_out_pre"])
    panel["log_market_cap_post"] = np.log(panel["price_post"] * panel["shares_out_post"])
    panel["ps_ratio_pre"] = panel["price_pre"] * panel["shares_out_pre"] / panel["revenue_pre"]
    panel["ps_ratio_post"] = panel["price_post"] * panel["shares_out_post"] / panel["revenue_post"]

    factors = pd.read_parquet(FACTORS_PATH, columns=["date", "mktrf", "rf"])
    factors["date"] = pd.to_datetime(factors["date"])
    _, erp = annualized_equity_premium(factors, ERP_SAMPLE_START)
    rf_pre = risk_free_at_ordered(factors, panel["filing_date_pre"])
    post_dates = panel["fecha"] + pd.to_timedelta(POST_WINDOW_END_OFFSET * 7 / 5, unit="D")
    rf_post = risk_free_at_ordered(factors, post_dates)

    panel["roic_minus_wacc_pre"] = roic_minus_wacc(
        rf_pre, panel["beta_pre"], erp, panel["operating_income_pre"], panel["pretax_income_pre"],
        panel["tax_expense_pre"], panel["equity_pre"], panel["long_term_debt_pre"], panel["interest_expense_pre"])
    panel["roic_minus_wacc_post"] = roic_minus_wacc(
        rf_post, panel["beta_post_126"], erp, panel["operating_income_post"], panel["pretax_income_post"],
        panel["tax_expense_post"], panel["equity_post"], panel["long_term_debt_post"], panel["interest_expense_post"])

    keep = ["ticker", "fecha"] + [f"{v}_{s}" for v in
            ["log_market_cap", "rd_intensity", "gross_margin", "ps_ratio", "next_revenue_yoy", "roic_minus_wacc"]
            for s in ("pre", "post")]
    out = panel[keep].copy()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, index=False)
    print(f"ERP geométrico usado: {erp:.4f}")
    for v in ["log_market_cap", "rd_intensity", "gross_margin", "ps_ratio", "next_revenue_yoy", "roic_minus_wacc"]:
        n = out[[f"{v}_pre", f"{v}_post"]].dropna().shape[0]
        print(f"  {v:18s} pre&post disponibles en {n:,} / {len(out):,} calls")
    print(f"-> {OUT_PATH}")


if __name__ == "__main__":
    main()
