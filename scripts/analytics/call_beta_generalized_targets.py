"""Generalizes the call-level beta ANCOVA (`call_beta_regressions.py`) to six
other firm fundamentals, so Chapter 6's "external characterization of
archetypes" section tests the same historical/surprise disclosure design
against a broader financial profile instead of a separate sector-year
archetype-dummy regression.

Same regressors, same fixed effects, same clustering as the beta model:
`hist_disclosure`, `hist_substance`, `surprise_disclosure`,
`surprise_substance` stay identical across every target. The only things
that change per target are the dependent variable (`{target}_post`, the
firm's fundamental from the first 10-K filed strictly AFTER the call) and
its own pre-call level (`{target}_pre`, from the last 10-K filed strictly
BEFORE the call), which slots into the regression exactly where `beta_pre`
sits for the beta model. Both come from `build_call_fundamentals_panel.py`
at filing-date / trading-day precision, not a fiscal-year bucket join.

Targets: log_market_cap, rd_intensity, gross_margin, ps_ratio,
next_revenue_yoy, roic_minus_wacc (the six non-beta outcomes already
reported in the old sector-year archetype regression). Beta itself is not
re-run here -- it is already the Chapter 6 baseline model, so the figure
this script feeds pulls that coefficient straight from
`call_beta_regressions.csv` instead of duplicating the estimation.

Usage:
  source .venv/bin/activate
  .venv/bin/python scripts/analytics/build_call_fundamentals_panel.py
  .venv/bin/python scripts/analytics/call_beta_generalized_targets.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[2]
CLUSTERS = ROOT / "data/processed/clusters"
PANEL_PATH = CLUSTERS / "call_beta_main_panel_10k10q_asof.parquet"
FUNDAMENTALS_PATH = CLUSTERS / "call_fundamentals_panel.parquet"
OUT = CLUSTERS / "call_beta_generalized_targets.csv"
OUT_SAMPLES = CLUSTERS / "call_beta_generalized_targets_samples.csv"

# AI regressors: identical across every target regression, including beta's.
AI_VARS = ["hist_disclosure", "hist_substance", "surprise_disclosure", "surprise_substance"]
# Fixed controls: identical across every target regression except the one
# that duplicates the target itself (dropped there, since {target}_pre
# already plays that role -- see `controls_for`).
FIXED_CONTROLS = ["log_market_cap", "return60", "operating_margin", "asset_turnover"]

TARGETS = {
    "log_market_cap": "Log market cap",
    "rd_intensity": "R&D / sales",
    "gross_margin": "Gross margin",
    "ps_ratio": "Price-to-sales",
    "next_revenue_yoy": "Revenue growth (t+1)",
    "roic_minus_wacc": "ROIC − WACC",
}
LABELS = {
    "hist_disclosure": "Historical disclosure intensity",
    "hist_substance": "Historical substantive activity",
    "surprise_disclosure": "Call-specific disclosure surprise",
    "surprise_substance": "Call-specific substance surprise",
}


def controls_for(target: str) -> tuple[list[str], list[str]]:
    """Same fixed controls for every target, minus the one that would
    duplicate `{target}_pre`. Returns (own_pre_regressor, other_controls)."""
    others = [c for c in FIXED_CONTROLS if c != target]
    return [f"{target}_pre"], others


def fit_target(panel: pd.DataFrame, target: str) -> tuple[pd.DataFrame, dict]:
    own_pre, others = controls_for(target)
    variables = AI_VARS + own_pre + others
    y_col = f"{target}_post"
    d = panel.dropna(subset=[y_col, "fe", *variables]).copy()
    d = d[d.groupby("fe")["ticker"].transform("size") >= 2].copy()
    y = (d[y_col] - d[y_col].mean()) / d[y_col].std(ddof=0)
    standardized = d[variables].apply(lambda col: (col - col.mean()) / col.std(ddof=0))
    fe = pd.get_dummies(d["fe"], prefix="fe", drop_first=True, dtype=float)
    design = sm.add_constant(pd.concat([standardized, fe], axis=1))
    result = sm.OLS(y, design).fit(cov_type="cluster", cov_kwds={"groups": d["ticker"]})
    rows = []
    for var in AI_VARS:
        beta, se = result.params[var], result.bse[var]
        rows.append({"target": target, "variable": var, "label": LABELS[var], "beta_std": beta,
                     "ci95_low": beta - 1.96 * se, "ci95_high": beta + 1.96 * se, "p": result.pvalues[var]})
    summary = {"target": target, "n_calls": len(d), "n_firms": d.ticker.nunique(),
               "n_fe_cells": d.fe.nunique(), "r2": result.rsquared}
    return pd.DataFrame(rows), summary


def main() -> None:
    panel = pd.read_parquet(PANEL_PATH)
    fundamentals_cols = ["ticker", "fecha"] + [f"{t}_{s}" for t in TARGETS for s in ("pre", "post")]
    fundamentals = pd.read_parquet(FUNDAMENTALS_PATH, columns=fundamentals_cols)
    panel = panel.merge(fundamentals, on=["ticker", "fecha"], how="left")

    tables, summaries = [], []
    for target in TARGETS:
        table, summary = fit_target(panel, target)
        tables.append(table)
        summaries.append(summary)
        print(f"{TARGETS[target]:22s} n={summary['n_calls']:5d} firms={summary['n_firms']:4d} | " +
              " | ".join(f"{r['variable']}: {r['beta_std']:+.3f} (p={r['p']:.3f})" for _, r in table.iterrows()))

    pd.concat(tables).to_csv(OUT, index=False)
    pd.DataFrame(summaries).to_csv(OUT_SAMPLES, index=False)
    print(f"\n-> {OUT}\n-> {OUT_SAMPLES}")


if __name__ == "__main__":
    main()
