"""Reproducible call-beta regressions with point-in-time 10-K/10-Q controls.

The input panel is the call-level panel constructed from exact call dates,
strictly prior expanding histories, and non-overlapping beta windows.  It
already carries the latest *filing accession* strictly before each call.
This module reads balance-sheet facts only from that accession, never from
the retrospective Company Facts aggregate.

Models:
  baseline              main specification, no leverage control
  debt_to_equity        matched-sample M0 / M1 with LT debt / equity
  liabilities_to_assets matched-sample M0 / M1 with liabilities / assets
  accounting            matched-sample M0 / M1 with ROA + liabilities/assets together
  all                   all four outputs

Usage:
  source .venv/bin/activate
  .venv/bin/python scripts/analytics/call_beta_regressions.py --model all
"""
from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[2]
PANEL = ROOT / "data/processed/clusters/call_beta_main_panel_10k10q_asof.parquet"
FACTS = ROOT / "data/raw/xbrl_facts/us_by_filing/*.parquet"
OUT = ROOT / "data/processed/clusters"
BASE = ["hist_disclosure", "hist_substance", "surprise_disclosure", "surprise_substance",
        "beta_pre", "log_market_cap", "return60", "roa"]
LABELS = {
    "hist_disclosure": "Historical disclosure intensity",
    "hist_substance": "Historical substantive activity",
    "surprise_disclosure": "Call-specific disclosure surprise",
    "surprise_substance": "Call-specific substance surprise",
    "beta_pre": "Pre-call beta", "log_market_cap": "Log market capitalization",
    "return60": "Pre-call return (60 trading days)", "operating_margin": "Operating margin",
    "asset_turnover": "Asset turnover", "debt_to_equity": "Debt / equity",
    "liabilities_to_assets": "Liabilities / assets", "roa": "Return on assets",
}
# The combined "accounting controls" model (ROA + liabilities/assets together,
# not run/M1 per single leverage measure like the M0/M1 pairs below) --
# thesis.qmd's prose specifically says "accounting controls (ROA and
# liabilities/assets)".
ACCOUNTING_VARS = ["liabilities_to_assets"]  # roa now lives in BASE itself
CONCEPTS = {
    "assets": ["us-gaap:Assets"],
    "liabilities": ["us-gaap:Liabilities"],
    "debt": ["us-gaap:LongTermDebtNoncurrent", "us-gaap:LongTermDebt"],
    "equity": ["us-gaap:StockholdersEquity", "us-gaap:StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
}


def select_instant_values(facts: pd.DataFrame, mappings: dict[str, list[str]]) -> pd.DataFrame:
    """One latest instant value per accession and metric, honoring tag priority."""
    rows: list[dict[str, object]] = []
    facts = facts.copy()
    facts["period_end"] = pd.to_datetime(facts["period_end"])
    for accession, group in facts.groupby("accession_number"):
        row: dict[str, object] = {"accession_number": accession}
        for metric, tags in mappings.items():
            eligible = group[group["concept"].isin(tags)].sort_values("period_end", ascending=False)
            for tag in tags:
                values = eligible[eligible["concept"] == tag]
                if not values.empty:
                    row[metric] = values.iloc[0]["numeric_value"]
                    break
        rows.append(row)
    return pd.DataFrame(rows)


def attach_leverage(panel: pd.DataFrame) -> pd.DataFrame:
    accessions = panel[["accession_number"]].dropna().drop_duplicates()
    con = duckdb.connect()
    try:
        con.register("selected", accessions)
        tags = ", ".join(repr(tag) for tags in CONCEPTS.values() for tag in tags)
        facts = con.execute(f"""
            SELECT f.accession_number, f.concept, f.numeric_value, f.period_end
            FROM read_parquet('{FACTS}', union_by_name=true) f
            JOIN selected USING (accession_number)
            WHERE NOT f.has_dimensions AND f.period_type = 'instant'
              AND f.concept IN ({tags})
        """).df()
    finally:
        con.close()
    values = select_instant_values(facts, CONCEPTS)
    out = panel.merge(values, on="accession_number", how="left")
    out["debt_to_equity"] = out["debt"] / out["equity"]
    out["liabilities_to_assets"] = out["liabilities"] / out["assets"]
    return out


ROA_PATH = ROOT / "data/processed/clusters/firm_year_financials_ratios.parquet"


def attach_roa(panel: pd.DataFrame) -> pd.DataFrame:
    """ROA per point-in-time filing accession, from the same ratios table
    `build_firm_panels.py`/economic-profile analyses use elsewhere -- not
    recomputed from raw XBRL facts here, since `firm_year_financials_ratios.
    parquet` already carries it keyed by `accession_number`, the same
    point-in-time filing key `attach_leverage` uses for debt/liabilities."""
    roa = pd.read_parquet(ROA_PATH, columns=["accession_number", "roa"]).drop_duplicates("accession_number")
    return panel.merge(roa, on="accession_number", how="left")


def fit(panel: pd.DataFrame, variables: list[str]) -> tuple[sm.regression.linear_model.RegressionResultsWrapper, pd.DataFrame]:
    d = panel.dropna(subset=["beta_post_63", "fe", *variables]).copy()
    d = d[d.groupby("fe")["ticker"].transform("size") >= 2].copy()
    y = (d["beta_post_63"] - d["beta_post_63"].mean()) / d["beta_post_63"].std(ddof=0)
    standardized = d[variables].apply(lambda col: (col - col.mean()) / col.std(ddof=0))
    fe = pd.get_dummies(d["fe"], prefix="fe", drop_first=True, dtype=float)
    design = sm.add_constant(pd.concat([standardized, fe], axis=1))
    return sm.OLS(y, design).fit(cov_type="cluster", cov_kwds={"groups": d["ticker"]}), d


def coefficient_table(result: sm.regression.linear_model.RegressionResultsWrapper, variables: list[str], model: str) -> pd.DataFrame:
    rows = []
    for var in variables:
        beta, se = result.params[var], result.bse[var]
        rows.append({"model": model, "variable": var, "label": LABELS[var], "beta_std": beta,
                     "ci95_low": beta - 1.96 * se, "ci95_high": beta + 1.96 * se, "p": result.pvalues[var]})
    return pd.DataFrame(rows)


def run(panel: pd.DataFrame, model: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if model == "baseline":
        result, sample = fit(panel, BASE)
        return coefficient_table(result, BASE, "baseline"), pd.DataFrame([{
            "model": "baseline", "n_calls": len(sample), "n_firms": sample.ticker.nunique(),
            "n_fe_cells": sample.fe.nunique(), "r2": result.rsquared,
        }])
    if model == "accounting":
        matched = panel.dropna(subset=ACCOUNTING_VARS).copy()
        m0, s0 = fit(matched, BASE)
        m1, s1 = fit(matched, BASE + ACCOUNTING_VARS)
        assert len(s0) == len(s1), "accounting M0/M1 must use the same ROA+leverage-observed sample"
        table = pd.concat([coefficient_table(m0, BASE, "accounting_M0"),
                           coefficient_table(m1, BASE + ACCOUNTING_VARS, "accounting_M1")])
        summary = pd.DataFrame([
            {"model": "accounting_M0", "n_calls": len(s0), "n_firms": s0.ticker.nunique(), "n_fe_cells": s0.fe.nunique(), "r2": m0.rsquared},
            {"model": "accounting_M1", "n_calls": len(s1), "n_firms": s1.ticker.nunique(), "n_fe_cells": s1.fe.nunique(), "r2": m1.rsquared},
        ])
        return table, summary
    leverage = model
    # M0 and M1 use exactly the same complete-case sample: only leverage differs.
    matched = panel.dropna(subset=[leverage]).copy()
    m0, s0 = fit(matched, BASE)
    m1, s1 = fit(matched, BASE + [leverage])
    assert len(s0) == len(s1), "M0/M1 must use the same leverage-observed sample"
    table = pd.concat([coefficient_table(m0, BASE, f"{model}_M0"),
                       coefficient_table(m1, BASE + [leverage], f"{model}_M1")])
    summary = pd.DataFrame([
        {"model": f"{model}_M0", "n_calls": len(s0), "n_firms": s0.ticker.nunique(), "n_fe_cells": s0.fe.nunique(), "r2": m0.rsquared},
        {"model": f"{model}_M1", "n_calls": len(s1), "n_firms": s1.ticker.nunique(), "n_fe_cells": s1.fe.nunique(), "r2": m1.rsquared},
    ])
    return table, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["baseline", "debt_to_equity", "liabilities_to_assets", "accounting", "all"], default="all")
    parser.add_argument("--panel", type=Path, default=PANEL)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    panel = attach_leverage(pd.read_parquet(args.panel))  # roa now comes pre-attached from build_call_beta_panel.py
    models = ["baseline", "debt_to_equity", "liabilities_to_assets", "accounting"] if args.model == "all" else [args.model]
    tables, summaries = zip(*(run(panel, model) for model in models))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(tables).to_csv(args.output_dir / "call_beta_regressions.csv", index=False)
    pd.concat(summaries).to_csv(args.output_dir / "call_beta_regression_samples.csv", index=False)
    print(pd.concat(summaries).to_string(index=False))


if __name__ == "__main__":
    main()
