"""Call-level CAR or post-beta regressions with point-in-time controls.

The main model is deliberately separate from ``call_beta_regressions.py``:
its outcome is market-model CAR in [-1,+5] trading days around an earnings
call, whereas that script studies a subsequent beta.  CAR uses a beta and
idio-volatility estimated only from [-252,-21] trading days before the call.

Earnings news is an as-of standardized unexpected earnings proxy (SUE),
``(EPS_q - EPS_{q-4}) / sd_prior(EPS_q - EPS_{q-4})``.  The standard
deviation is expanding and shifted, so no later quarter enters a call's SUE.
It is not analyst-consensus surprise.

Usage (the default estimates post-call beta without SUE):
  source .venv/bin/activate
  .venv/bin/python scripts/analytics/call_car_regressions.py --outcome beta_post_126
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

# This script is run directly in production but imported as a namespace module
# in tests; support both forms while sharing the point-in-time extractor.
try:
    from call_beta_regressions import attach_leverage
except ModuleNotFoundError:  # pragma: no cover - exercised by test import path
    from scripts.analytics.call_beta_regressions import attach_leverage

ROOT = Path(__file__).resolve().parents[2]
PANEL = ROOT / "data/processed/clusters/call_beta_main_panel_10k10q_asof.parquet"
Q_FINANCIALS = ROOT / "data/processed/us_10q_financials_panel.parquet"
Q_MANIFEST = ROOT / "data/interim/manifests/filing_manifest_10q.parquet"
PRICES = ROOT / "data/raw/market/prices"
FACTORS = ROOT / "data/raw/market/factors/ff3_daily.parquet"
OUT = ROOT / "data/processed/clusters"

AI = ["hist_disclosure", "hist_substance", "surprise_disclosure", "surprise_substance"]
CONTROLS = ["sue", "return60", "beta_pre_car", "idio_vol_pre", "log_market_cap"]
LABELS = {
    "hist_disclosure": "Historical disclosure intensity",
    "hist_substance": "Historical substantive activity",
    "surprise_disclosure": "Call-specific disclosure surprise",
    "surprise_substance": "Call-specific substance surprise",
    "sue": "Standardized unexpected earnings (as-of proxy)",
    "return60": "Pre-call return (60 trading days)",
    "beta_pre_car": "Pre-call beta",
    "idio_vol_pre": "Pre-call idiosyncratic volatility",
    "log_market_cap": "Log market capitalization",
    "roa": "Return on assets (as-of)",
    "liabilities_to_assets": "Liabilities / assets (as-of)",
}


def standardized_unexpected_earnings(eps: pd.DataFrame, min_history: int = 4) -> pd.DataFrame:
    """Compute SUE with an expanding *prior-only* firm-specific denominator."""
    out = eps.copy().sort_values(["ticker", "filing_date"]).reset_index(drop=True)
    out["eps_yoy_change"] = out.groupby("ticker")["eps_diluted"].diff(4)
    prior_sd = (out.groupby("ticker")["eps_yoy_change"]
                .transform(lambda x: x.shift().expanding(min_periods=min_history).std()))
    out["sue"] = out["eps_yoy_change"] / prior_sd.replace(0, np.nan)
    return out


def load_eps_events(q_financials: Path, manifest: Path) -> pd.DataFrame:
    q = pd.read_parquet(q_financials, filters=[("metric", "==", "eps_diluted")])
    m = pd.read_parquet(manifest, columns=["ticker", "accession_number", "filing_date"])
    m["filing_date"] = pd.to_datetime(m["filing_date"])
    q = q.rename(columns={"source_ref": "accession_number", "value": "eps_diluted"})
    eps = q.merge(m, on=["ticker", "accession_number"], how="inner")
    # A single accession can supply one EPS fact only; preserve provenance and
    # refuse to fabricate an average if a malformed input has duplicates.
    eps = eps.dropna(subset=["eps_diluted", "filing_date"]).drop_duplicates(
        ["ticker", "accession_number"], keep="first")
    return standardized_unexpected_earnings(eps[["ticker", "filing_date", "eps_diluted"]])


def load_price_series(prices_dir: Path, tickers: set[str]) -> dict[str, pd.DataFrame]:
    result: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        path = prices_dir / f"{ticker}.parquet"
        if not path.exists():
            continue
        d = pd.read_parquet(path, columns=["date", "adj_close"])
        d["date"] = pd.to_datetime(d["date"])
        d = d.dropna(subset=["adj_close"]).sort_values("date").reset_index(drop=True)
        d["ret"] = d["adj_close"].pct_change()
        result[ticker] = d
    return result


def market_metrics(prices: pd.DataFrame, factors: pd.DataFrame, event_date: pd.Timestamp) -> dict[str, float]:
    """Prior beta/idio-vol and [-1,+5] CAR for one exact call date."""
    result = {"beta_pre_car": np.nan, "idio_vol_pre": np.nan, "car_m1_p5": np.nan}
    idx = int(np.searchsorted(prices["date"].values, np.datetime64(event_date), side="left"))
    if idx < 252 or idx + 5 >= len(prices):
        return result
    pre = prices.iloc[idx - 252:idx - 20].merge(factors, on="date", how="inner")
    pre = pre.dropna(subset=["ret", "mktrf", "rf"])
    if len(pre) < 120:
        return result
    excess = pre["ret"].to_numpy() - pre["rf"].to_numpy()
    design = np.column_stack([np.ones(len(pre)), pre["mktrf"].to_numpy()])
    coef = np.linalg.lstsq(design, excess, rcond=None)[0]
    result["beta_pre_car"] = float(coef[1])
    result["idio_vol_pre"] = float((excess - design @ coef).std(ddof=2) * np.sqrt(252))
    event = prices.iloc[idx - 1:idx + 6].merge(factors, on="date", how="inner")
    event = event.dropna(subset=["ret", "mktrf", "rf"])
    if len(event) >= 6:
        abnormal = (event["ret"] - event["rf"]) - coef[1] * event["mktrf"]
        # As documented in the project market builder, day -1 provides the
        # clean reference price; CAR collects reaction days 0 through +5.
        result["car_m1_p5"] = float(abnormal.iloc[1:].sum())
    return result


def attach_market_metrics(panel: pd.DataFrame, prices_dir: Path, factors_path: Path) -> pd.DataFrame:
    factors = pd.read_parquet(factors_path, columns=["date", "mktrf", "rf"])
    factors["date"] = pd.to_datetime(factors["date"])
    series = load_price_series(prices_dir, set(panel["ticker"]))
    rows = []
    for row in panel[["ticker", "fecha"]].itertuples(index=False):
        price = series.get(row.ticker)
        if price is None:
            continue
        metrics = market_metrics(price, factors, pd.Timestamp(row.fecha))
        metrics.update({"ticker": row.ticker, "fecha": row.fecha})
        rows.append(metrics)
    return panel.merge(pd.DataFrame(rows), on=["ticker", "fecha"], how="left")


def attach_sue(panel: pd.DataFrame, eps: pd.DataFrame) -> pd.DataFrame:
    left = panel.copy()
    right = eps[["ticker", "filing_date", "sue", "eps_yoy_change"]].copy()
    left["fecha"] = pd.to_datetime(left["fecha"]).astype("datetime64[ns]")
    right = right.copy()
    right["filing_date"] = pd.to_datetime(right["filing_date"]).astype("datetime64[ns]")
    left = left.sort_values(["fecha", "ticker"])
    right = right.sort_values(["filing_date", "ticker"])
    return pd.merge_asof(left, right, left_on="fecha", right_on="filing_date", by="ticker",
                         direction="backward", allow_exact_matches=False)


def attach_roa(panel: pd.DataFrame) -> pd.DataFrame:
    """Attach the latest quarterly or annual ROA known strictly before call."""
    annual = pd.read_parquet(ROOT / "data/processed/clusters/firm_year_financials_ratios.parquet",
                             columns=["ticker", "filing_date", "roa"])
    annual["filing_date"] = pd.to_datetime(annual["filing_date"])
    q = pd.read_parquet(Q_FINANCIALS, filters=[("metric", "in", ["net_income", "assets"])])
    q = q.pivot_table(index=["ticker", "source_ref"], columns="metric", values="value", aggfunc="first").reset_index()
    q["roa"] = q["net_income"] / q["assets"].where(q["assets"] != 0)
    manifest = pd.read_parquet(Q_MANIFEST, columns=["ticker", "accession_number", "filing_date"])
    q = q.merge(manifest, left_on=["ticker", "source_ref"], right_on=["ticker", "accession_number"], how="inner")
    events = pd.concat([annual[["ticker", "filing_date", "roa"]], q[["ticker", "filing_date", "roa"]]])
    events["filing_date"] = pd.to_datetime(events["filing_date"]).astype("datetime64[ns]")
    events = events.dropna(subset=["roa"]).sort_values(["filing_date", "ticker"])
    left = panel.sort_values(["fecha", "ticker"]).copy()
    return pd.merge_asof(left, events, left_on="fecha", right_on="filing_date", by="ticker",
                         direction="backward", allow_exact_matches=False)


def fit(panel: pd.DataFrame, variables: list[str], outcome: str) -> tuple[sm.regression.linear_model.RegressionResultsWrapper, pd.DataFrame]:
    d = panel.dropna(subset=[outcome, "fe", *variables]).copy()
    d = d[d.groupby("fe")["ticker"].transform("size") >= 2].copy()
    y = (d[outcome] - d[outcome].mean()) / d[outcome].std(ddof=0)
    x = d[variables].apply(lambda c: (c - c.mean()) / c.std(ddof=0))
    fe = pd.get_dummies(d["fe"], prefix="fe", drop_first=True, dtype=float)
    result = sm.OLS(y, sm.add_constant(pd.concat([x, fe], axis=1))).fit(
        cov_type="cluster", cov_kwds={"groups": d["ticker"]})
    return result, d


def coefficient_table(result: sm.regression.linear_model.RegressionResultsWrapper,
                      variables: list[str], model: str) -> pd.DataFrame:
    rows = []
    for variable in variables:
        beta, se = result.params[variable], result.bse[variable]
        rows.append({"model": model, "variable": variable, "label": LABELS[variable],
                     "beta_std": beta, "ci95_low": beta - 1.96 * se,
                     "ci95_high": beta + 1.96 * se, "p": result.pvalues[variable]})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, default=PANEL)
    parser.add_argument("--q-financials", type=Path, default=Q_FINANCIALS)
    parser.add_argument("--q-manifest", type=Path, default=Q_MANIFEST)
    parser.add_argument("--prices-dir", type=Path, default=PRICES)
    parser.add_argument("--factors", type=Path, default=FACTORS)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--outcome", choices=["beta_post_126", "car_m1_p5"], default="beta_post_126")
    parser.add_argument("--include-sue", action="store_true",
                        help="Add the accounting SUE proxy (off by default).")
    args = parser.parse_args()

    panel = pd.read_parquet(args.panel)
    panel["fecha"] = pd.to_datetime(panel["fecha"])
    if args.include_sue:
        eps = load_eps_events(args.q_financials, args.q_manifest)
        panel = attach_sue(panel, eps)
    panel = attach_market_metrics(panel, args.prices_dir, args.factors)
    panel = attach_roa(panel)
    panel = attach_leverage(panel)
    controls = CONTROLS if args.include_sue else [v for v in CONTROLS if v != "sue"]
    baseline_variables = AI + controls
    result, sample = fit(panel, baseline_variables, args.outcome)
    accounting_variables = baseline_variables + ["roa", "liabilities_to_assets"]
    matched = panel.dropna(subset=["roa", "liabilities_to_assets"])
    matched_m0, matched_sample = fit(matched, baseline_variables, args.outcome)
    accounting_result, accounting_sample = fit(matched, accounting_variables, args.outcome)
    assert len(matched_sample) == len(accounting_sample)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(args.output_dir / "call_car_panel.parquet", index=False)
    table = pd.concat([
        coefficient_table(result, baseline_variables, f"{args.outcome}_baseline"),
        coefficient_table(matched_m0, baseline_variables, f"{args.outcome}_accounting_matched_M0"),
        coefficient_table(accounting_result, accounting_variables, f"{args.outcome}_accounting_M1"),
    ])
    prefix = ("call_beta_sue" if args.include_sue else "call_beta_clean") if args.outcome == "beta_post_126" else "call_car"
    table.to_csv(args.output_dir / f"{prefix}_regressions.csv", index=False)
    summary = pd.DataFrame([{
        "model": f"{args.outcome}_baseline", "n_calls": len(sample), "n_firms": sample.ticker.nunique(),
        "n_fe_cells": sample.fe.nunique(), "r2": result.rsquared,
        "median_prior_calls": sample.n_prior_calls.median(),
        "sue_included": args.include_sue, "car_coverage_calls": panel.car_m1_p5.notna().sum(),
    }, {
        "model": f"{args.outcome}_accounting_matched_M0", "n_calls": len(matched_sample),
        "n_firms": matched_sample.ticker.nunique(), "n_fe_cells": matched_sample.fe.nunique(),
        "r2": matched_m0.rsquared, "median_prior_calls": matched_sample.n_prior_calls.median(),
        "sue_included": args.include_sue, "car_coverage_calls": panel.car_m1_p5.notna().sum(),
    }, {
        "model": f"{args.outcome}_accounting_M1", "n_calls": len(accounting_sample),
        "n_firms": accounting_sample.ticker.nunique(), "n_fe_cells": accounting_sample.fe.nunique(),
        "r2": accounting_result.rsquared, "median_prior_calls": accounting_sample.n_prior_calls.median(),
        "sue_included": args.include_sue, "car_coverage_calls": panel.car_m1_p5.notna().sum(),
    }])
    summary.to_csv(args.output_dir / f"{prefix}_regression_samples.csv", index=False)
    print(summary.to_string(index=False))
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
