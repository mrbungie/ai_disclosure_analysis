"""Robustness battery for the post-call beta regression.

Runs every block listed as pending in ``docs/analytics/call-beta-regressions.md``:
alternate beta windows, DeltaBeta, firm FE, AI sub-blocks, collinearity (VIF),
influence (winsorization), beta-estimation thresholds, market-model choice
(CAPM vs FF3; FF5 and an alternate broad benchmark are not available in this
repo's data and are skipped), leave-one-year/sector/decile-out, the placebo
test, the two disclosure-vs-substance Wald tests, joint tests, and the AI
block's incremental R^2.

Beta/idio-vol windows are recomputed directly from raw prices so every window
variant uses the same estimator (market-model OLS on trading-day offsets
relative to the exact call date), rather than trusting precomputed panel
columns that may differ in method.

Usage:
  source .venv/bin/activate
  .venv/bin/python scripts/analytics/call_beta_robustness.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

try:
    from call_car_regressions import load_price_series
except ModuleNotFoundError:  # pragma: no cover - exercised by test import path
    from scripts.analytics.call_car_regressions import load_price_series

ROOT = Path(__file__).resolve().parents[2]
PANEL = ROOT / "data/processed/clusters/call_beta_main_panel_10k10q_asof.parquet"
PRICES = ROOT / "data/raw/market/prices"
FACTORS = ROOT / "data/raw/market/factors/ff3_daily.parquet"
OUT = ROOT / "data/processed/clusters"

AI = ["hist_disclosure", "hist_substance", "surprise_disclosure", "surprise_substance"]
CONTROLS = ["return60", "log_market_cap", "idio_vol_pre_default"]
TECH_SIC2 = {"35", "36", "73"}


# --------------------------------------------------------------------------
# Window-level market-model estimation
# --------------------------------------------------------------------------

def window_metrics(prices: pd.DataFrame, mkt: pd.DataFrame, event_date: pd.Timestamp,
                    start: int, end: int, factor_cols: list[str], min_obs: int) -> dict[str, float]:
    """Market-model beta/idio-vol over trading-day offsets ``[start, end)``
    relative to the call date's price index, using ``factor_cols`` as the
    right-hand side (``["mktrf"]`` for CAPM, ``["mktrf", "smb", "hml"]`` for FF3)."""
    result = {"beta": np.nan, "idio_vol": np.nan, "n_obs": 0}
    idx = int(np.searchsorted(prices["date"].values, np.datetime64(event_date), side="left"))
    lo, hi = idx + start, idx + end
    if lo < 0 or hi > len(prices) or hi <= lo:
        return result
    window = prices.iloc[lo:hi].merge(mkt, on="date", how="inner")
    window = window.dropna(subset=["ret", *factor_cols, "rf"])
    if len(window) < min_obs:
        return result
    excess = window["ret"].to_numpy() - window["rf"].to_numpy()
    design = np.column_stack([np.ones(len(window)), *[window[c].to_numpy() for c in factor_cols]])
    coef, *_ = np.linalg.lstsq(design, excess, rcond=None)
    result["beta"] = float(coef[1])
    result["idio_vol"] = float((excess - design @ coef).std(ddof=len(factor_cols) + 1) * np.sqrt(252))
    result["n_obs"] = len(window)
    return result


def attach_windows(panel: pd.DataFrame, prices_dir: Path, factors_path: Path,
                    windows: dict[str, tuple[int, int]], factor_cols: list[str],
                    min_obs: int | dict[str, int] | None = None, min_frac: float = 0.7) -> pd.DataFrame:
    """``min_obs`` is either a fixed threshold applied to every window (only
    sensible for long windows), a per-window dict, or (default) ``None`` to
    scale the requirement to ``min_frac`` of each window's trading-day span
    so short windows (e.g. ``[+21,+63]``) are not required to clear a
    threshold only a long window could ever reach."""
    factors = pd.read_parquet(factors_path, columns=["date", "mktrf", "smb", "hml", "rf"])
    factors["date"] = pd.to_datetime(factors["date"])
    series = load_price_series(prices_dir, set(panel["ticker"]))
    thresholds = {}
    for name, (start, end) in windows.items():
        if isinstance(min_obs, dict):
            thresholds[name] = min_obs[name]
        elif isinstance(min_obs, int):
            thresholds[name] = min_obs
        else:
            thresholds[name] = max(15, int(min_frac * (end - start)))
    rows = []
    for row in panel[["ticker", "fecha"]].itertuples(index=False):
        price = series.get(row.ticker)
        out = {"ticker": row.ticker, "fecha": row.fecha}
        if price is not None:
            for name, (start, end) in windows.items():
                m = window_metrics(price, factors, pd.Timestamp(row.fecha), start, end,
                                    factor_cols, thresholds[name])
                out[f"beta_{name}"] = m["beta"]
                out[f"idio_vol_{name}"] = m["idio_vol"]
                out[f"n_obs_{name}"] = m["n_obs"]
        rows.append(out)
    keep = panel.drop(columns=[c for c in panel.columns if c.startswith("beta_") or c.startswith("idio_vol_")])
    return keep.merge(pd.DataFrame(rows), on=["ticker", "fecha"], how="left")


# --------------------------------------------------------------------------
# Regression helpers
# --------------------------------------------------------------------------

def standardize(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df[cols].apply(lambda c: (c - c.mean()) / c.std(ddof=0))


def fit_fe(panel: pd.DataFrame, variables: list[str], outcome: str, fe_col: str = "fe",
           min_fe_size: int = 2) -> tuple[sm.regression.linear_model.RegressionResultsWrapper, pd.DataFrame]:
    d = panel.dropna(subset=[outcome, fe_col, *variables]).copy()
    d = d[d.groupby(fe_col)["ticker"].transform("size") >= min_fe_size].copy()
    y = (d[outcome] - d[outcome].mean()) / d[outcome].std(ddof=0)
    x = standardize(d, variables)
    fe = pd.get_dummies(d[fe_col], prefix=fe_col, drop_first=True, dtype=float)
    design = sm.add_constant(pd.concat([x, fe], axis=1))
    result = sm.OLS(y, design).fit(cov_type="cluster", cov_kwds={"groups": d["ticker"]})
    return result, d


def coef_row(result, variable: str) -> dict:
    beta, se = result.params[variable], result.bse[variable]
    return {"variable": variable, "beta_std": beta, "ci95_low": beta - 1.96 * se,
            "ci95_high": beta + 1.96 * se, "p": result.pvalues[variable]}


def ai_coef_table(result, block: str, variables: list[str], n_calls: int | None = None) -> pd.DataFrame:
    rows = [{"block": block, **coef_row(result, v)} for v in variables if v in AI]
    if n_calls is not None:
        for row in rows:
            row["n_calls"] = n_calls
    return pd.DataFrame(rows)


def wald_diff(result, a: str, b: str) -> dict:
    t = result.t_test(f"{a} - {b} = 0")
    return {"contrast": f"{a} = {b}", "diff": float(t.effect[0]), "p": float(t.pvalue)}


def wald_joint(result, variables: list[str]) -> dict:
    f = result.f_test(", ".join(f"{v} = 0" for v in variables))
    return {"variables": variables, "F": float(f.fvalue), "p": float(f.pvalue)}


def partial_r2(panel: pd.DataFrame, variables: list[str], ai_vars: list[str], outcome: str) -> float:
    full, d = fit_fe(panel, variables, outcome)
    x_reduced = standardize(d, [v for v in variables if v not in ai_vars])
    fe = pd.get_dummies(d["fe"], prefix="fe", drop_first=True, dtype=float)
    y = (d[outcome] - d[outcome].mean()) / d[outcome].std(ddof=0)
    reduced_on_same = sm.OLS(y, sm.add_constant(pd.concat([x_reduced, fe], axis=1))).fit()
    return float(full.rsquared - reduced_on_same.rsquared)


# --------------------------------------------------------------------------
# Robustness blocks
# --------------------------------------------------------------------------

def block_windows(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for post in ["post_63", "post_126", "post_252"]:
        outcome = f"beta_{post}"
        variables = AI + ["beta_pre_default", *CONTROLS]
        result, d = fit_fe(panel, variables, outcome)
        rows.append(ai_coef_table(result, f"beta_window[{post}]", AI, n_calls=len(d)))
    for pre in ["pre_default", "pre_126_21", "pre_252_42"]:
        outcome = "beta_post_126"
        variables = AI + [f"beta_{pre}", *CONTROLS]
        result, d = fit_fe(panel, variables, outcome)
        rows.append(ai_coef_table(result, f"beta_pre_window[{pre}]", AI, n_calls=len(d)))
    return pd.concat(rows, ignore_index=True)


def block_delta_beta(panel: pd.DataFrame) -> pd.DataFrame:
    d = panel.copy()
    d["delta_beta"] = d["beta_post_126"] - d["beta_pre_default"]
    variables = AI + CONTROLS
    result, d2 = fit_fe(d, variables, "delta_beta")
    return ai_coef_table(result, "delta_beta", AI, n_calls=len(d2))


def block_firm_fe(panel: pd.DataFrame) -> pd.DataFrame:
    variables = AI + ["beta_pre_default", *CONTROLS]
    result, d = fit_fe(panel, variables, "beta_post_126", fe_col="ticker", min_fe_size=2)
    return ai_coef_table(result, "firm_fe_within", AI, n_calls=len(d))


def block_ai_subsets(panel: pd.DataFrame) -> pd.DataFrame:
    hist_only = ["hist_disclosure", "hist_substance"]
    surprise_only = ["surprise_disclosure", "surprise_substance"]
    rows = []
    for name, ai_vars in [("hist_only", hist_only), ("surprise_only", surprise_only), ("both", AI)]:
        variables = ai_vars + ["beta_pre_default", *CONTROLS]
        result, _ = fit_fe(panel, variables, "beta_post_126")
        for v in ai_vars:
            rows.append({"block": name, **coef_row(result, v)})
    return pd.DataFrame(rows)


def block_collinearity(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = panel.dropna(subset=AI).copy()
    corr = d[AI].corr()
    x = sm.add_constant(standardize(d, AI))
    vif = pd.DataFrame({"variable": ["const", *AI],
                        "vif": [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]})
    return corr, vif[vif.variable != "const"]


def block_influence(panel: pd.DataFrame) -> pd.DataFrame:
    variables = AI + ["beta_pre_default", *CONTROLS]
    winsor_cols = ["beta_post_126", "beta_pre_default", "idio_vol_pre_default", "log_market_cap", "return60"]
    d = panel.copy()
    for c in winsor_cols:
        if c not in d:
            continue
        lo, hi = d[c].quantile(0.01), d[c].quantile(0.99)
        d[c] = d[c].clip(lo, hi)
    result_w, _ = fit_fe(d, variables, "beta_post_126")
    d2 = panel.copy()
    for c in winsor_cols:
        if c not in d2:
            continue
        lo, hi = d2[c].quantile(0.01), d2[c].quantile(0.99)
        d2 = d2[(d2[c] >= lo) & (d2[c] <= hi)]
    result_e, _ = fit_fe(d2, variables, "beta_post_126")
    return pd.concat([ai_coef_table(result_w, "winsorized_1pct", AI),
                       ai_coef_table(result_e, "extremes_excluded_1pct", AI)], ignore_index=True)


def block_beta_threshold(panel_min60: pd.DataFrame, panel_min80: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, panel in [("min_obs_60", panel_min60), ("min_obs_80", panel_min80)]:
        variables = AI + ["beta_pre_default", *CONTROLS]
        result, d = fit_fe(panel, variables, "beta_post_126")
        rows.append(ai_coef_table(result, name, AI).assign(n_calls=len(d)))
    return pd.concat(rows, ignore_index=True)


def block_market_model(panel_capm: pd.DataFrame, panel_ff3: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, panel in [("capm", panel_capm), ("ff3", panel_ff3)]:
        variables = AI + ["beta_pre_default", *CONTROLS]
        result, d = fit_fe(panel, variables, "beta_post_126")
        rows.append(ai_coef_table(result, name, AI, n_calls=len(d)))
    return pd.concat(rows, ignore_index=True)


def block_leave_one_year_out(panel: pd.DataFrame) -> pd.DataFrame:
    variables = AI + ["beta_pre_default", *CONTROLS]
    rows = []
    for year in sorted(panel["fecha"].dt.year.unique()):
        d = panel[panel["fecha"].dt.year != year]
        result, dd = fit_fe(d, variables, "beta_post_126")
        rows.append(ai_coef_table(result, f"excl_{year}", AI).assign(n_calls=len(dd)))
    return pd.concat(rows, ignore_index=True)


def block_leave_one_sector_out(panel: pd.DataFrame) -> pd.DataFrame:
    variables = AI + ["beta_pre_default", *CONTROLS]
    rows = []
    for sic2 in sorted(panel["sic2"].dropna().unique()):
        d = panel[panel["sic2"] != sic2]
        if d["fe"].nunique() < 2:
            continue
        result, dd = fit_fe(d, variables, "beta_post_126")
        tag = f"excl_sic2_{sic2}" + ("_tech" if sic2 in TECH_SIC2 else "")
        rows.append(ai_coef_table(result, tag, AI).assign(n_calls=len(dd)))
    return pd.concat(rows, ignore_index=True)


def block_call_frequency(panel: pd.DataFrame) -> pd.DataFrame:
    freq = panel.groupby("ticker").size()
    threshold = freq.quantile(0.9)
    heavy = freq[freq > threshold].index
    d = panel[~panel["ticker"].isin(heavy)]
    variables = AI + ["beta_pre_default", *CONTROLS]
    result, dd = fit_fe(d, variables, "beta_post_126")
    return ai_coef_table(result, "excl_top_decile_call_freq", AI).assign(
        n_calls=len(dd), n_excluded_firms=len(heavy), threshold_calls=float(threshold))


def block_placebo(panel: pd.DataFrame) -> pd.DataFrame:
    variables = ["disclosure", "substance", *CONTROLS]
    result, _ = fit_fe(panel, variables, "beta_pre_default")
    return pd.DataFrame([{"block": "placebo_pre_beta", **coef_row(result, v)} for v in ["disclosure", "substance"]])


def block_formal_tests(panel: pd.DataFrame) -> dict:
    variables = AI + ["beta_pre_default", *CONTROLS]
    result, d = fit_fe(panel, variables, "beta_post_126")
    return {
        "wald_hist_disclosure_eq_hist_substance": wald_diff(result, "hist_disclosure", "hist_substance"),
        "wald_surprise_disclosure_eq_surprise_substance": wald_diff(result, "surprise_disclosure", "surprise_substance"),
        "joint_disclosure_terms": wald_joint(result, ["hist_disclosure", "surprise_disclosure"]),
        "joint_substance_terms": wald_joint(result, ["hist_substance", "surprise_substance"]),
        "partial_r2_ai_block": partial_r2(panel, variables, AI, "beta_post_126"),
        "n_calls": int(len(d)),
    }


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, default=PANEL)
    parser.add_argument("--prices-dir", type=Path, default=PRICES)
    parser.add_argument("--factors", type=Path, default=FACTORS)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()

    base = pd.read_parquet(args.panel)
    base["fecha"] = pd.to_datetime(base["fecha"])

    windows = {
        "pre_default": (-252, -21), "pre_126_21": (-126, -21), "pre_252_42": (-252, -42),
        "post_63": (21, 63), "post_126": (21, 126), "post_252": (21, 252),
    }
    default_windows = {"pre_default": (-252, -21), "post_126": (21, 126)}
    panel = attach_windows(base, args.prices_dir, args.factors, windows, ["mktrf"])
    panel_min60 = attach_windows(base, args.prices_dir, args.factors, default_windows, ["mktrf"], min_obs=60)
    panel_min80 = attach_windows(base, args.prices_dir, args.factors, default_windows, ["mktrf"], min_obs=80)
    panel_ff3 = attach_windows(base, args.prices_dir, args.factors, default_windows, ["mktrf", "smb", "hml"])

    out = {}
    out["windows"] = block_windows(panel)
    out["delta_beta"] = block_delta_beta(panel)
    out["firm_fe"] = block_firm_fe(panel)
    out["ai_subsets"] = block_ai_subsets(panel)
    corr, vif = block_collinearity(panel)
    out["ai_correlation"] = corr.reset_index().rename(columns={"index": "variable"})
    out["ai_vif"] = vif
    out["influence"] = block_influence(panel)
    out["beta_threshold"] = block_beta_threshold(panel_min60, panel_min80)
    out["market_model"] = block_market_model(panel_min60, panel_ff3)
    out["leave_one_year_out"] = block_leave_one_year_out(panel)
    out["leave_one_sector_out"] = block_leave_one_sector_out(panel)
    out["call_frequency"] = block_call_frequency(panel)
    out["placebo"] = block_placebo(panel)
    formal = block_formal_tests(panel)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, df in out.items():
        df.to_csv(args.output_dir / f"call_beta_robustness_{name}.csv", index=False)
    with open(args.output_dir / "call_beta_robustness_formal_tests.json", "w") as f:
        json.dump(formal, f, indent=2)

    print("Not implemented (data unavailable in this repo): FF5 factors, alternate broad benchmark.")
    print(json.dumps(formal, indent=2))
    for name, df in out.items():
        print(f"\n== {name} ==")
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
