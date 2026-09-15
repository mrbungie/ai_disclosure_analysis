"""Call-level CAR or post-beta regressions with point-in-time controls.

The main model is deliberately separate from ``call_beta_regressions.py``:
its outcome is market-model CAR in [-1,+5] trading days around an earnings
call, whereas that script studies a subsequent beta.  CAR uses a beta and
idio-volatility estimated only from [-252,-21] trading days before the call.

Earnings news is an as-of standardized unexpected earnings proxy (SUE),
``(EPS_q - EPS_{q-4}) / sd_prior(EPS_q - EPS_{q-4})``.  The standard
deviation is expanding and shifted, so no later quarter enters a call's SUE.
It is not analyst-consensus surprise.

Every variable comes from the call gold families (built by
scripts/gold/call/): disclosure history and surprise
(covariates/call/disclosure), pre-call return, beta, idio-volatility and size
(covariates/call/market), SUE, ROA and leverage (covariates/call/financials),
CAR (targets/call/market) and post-call beta (targets/call/market).

Usage (the default estimates post-call beta without SUE):
  source .venv/bin/activate
  .venv/bin/python scripts/analytics/call_beta/call_car_regressions.py --outcome beta_post_126
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "common"))
import layers as L  # noqa: E402

OUT = ROOT / "data/results/call_beta"

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


def load_panel() -> pd.DataFrame:
    """One row per call with the disclosure, market, accounting and outcome columns."""
    return L.read_dataset(
        "call",
        ("covariates", "disclosure", AI + ["n_prior_calls"]),
        ("covariates", "market", ["return60", "beta_pre_car", "idio_vol_pre", "log_market_cap"]),
        ("covariates", "financials", ["roa", "sue", "debt_to_equity", "liabilities_to_assets"]),
        ("targets", "market", ["beta_post_126", "car_m1_p5"]),
    )


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
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--outcome", choices=["beta_post_126", "car_m1_p5"], default="beta_post_126")
    parser.add_argument("--include-sue", action="store_true",
                        help="Add the accounting SUE proxy (off by default).")
    args = parser.parse_args()

    panel = load_panel()
    controls = CONTROLS if args.include_sue else [v for v in CONTROLS if v != "sue"]
    baseline_variables = AI + controls
    result, sample = fit(panel, baseline_variables, args.outcome)
    accounting_variables = baseline_variables + ["roa", "liabilities_to_assets"]
    matched = panel.dropna(subset=["roa", "liabilities_to_assets"])
    matched_m0, matched_sample = fit(matched, baseline_variables, args.outcome)
    accounting_result, accounting_sample = fit(matched, accounting_variables, args.outcome)
    assert len(matched_sample) == len(accounting_sample)
    args.output_dir.mkdir(parents=True, exist_ok=True)
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
