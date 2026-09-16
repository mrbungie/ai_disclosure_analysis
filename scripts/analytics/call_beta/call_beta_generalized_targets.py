"""The post-call beta design applied to six firm fundamentals: log market cap,
R&D / sales, gross margin, price-to-sales, next-year revenue growth and
ROIC - WACC.

Design, panel and estimation: scripts/analytics/call_regression.py. The
outcome is `{target}_post` (first 10-K filed after the call) and its own
pre-call level (last 10-K filed before the call, call_regression.PRE_CONTROL;
for next-year revenue growth, the growth that 10-K reports) takes the place of
pre-call beta; the fixed control that duplicates the target is
dropped. ROIC - WACC reads unreported long-term debt as zero debt
(call_regression.attach_fundamentals).

Outputs data/results/call_beta/call_beta_generalized_targets.csv and
call_beta_generalized_targets_samples.csv.

Usage:
  .venv/bin/python scripts/analytics/call_beta/call_beta_generalized_targets.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "analytics"))
import call_regression as C  # noqa: E402

# Outcome column per target: a level keeps `_post`, a margin and a spread are
# read as the change they record, in percentage points (call_regression.
# CHANGE_TARGETS).
TARGETS = {
    "log_market_cap": ("log_market_cap_post", "Log market capitalization"),
    "gross_margin": ("gross_margin_change_pp", "Gross margin expansion (pp)"),
    "ps_ratio": ("ps_ratio_post", "Price-to-sales ratio"),
    "next_revenue_yoy": ("next_revenue_yoy_post", "Revenue growth (%)"),
    "roic_minus_wacc": ("roic_minus_wacc_change_pp", "Change in ROIC-WACC spread (pp)"),
}


def main() -> None:
    panel = C.attach_fundamentals(C.load_call_panel())
    tables, samples = [], []
    for target, (y_col, label) in TARGETS.items():
        res, d = C.fit(panel, y_col, C.AI_VARS + [C.PRE_CONTROL[target]] + C.controls_for(y_col))
        tables.append(C.coef_table(res, C.AI_VARS + C.REPORTED_WEIGHTS, target=target))
        samples.append(C.sample_row(res, d, target=target))
        print(f"{label:32s} n={len(d):5d} firms={d.ticker.nunique():4d} | "
              + " | ".join(f"{v}: {res.params[v]:+.3f} (p={res.pvalues[v]:.3f})" for v in C.AI_VARS + C.REPORTED_WEIGHTS))
    out = C.L.results_path("call_beta", "call_beta_generalized_targets.csv")
    pd.concat(tables, ignore_index=True).to_csv(out, index=False)
    pd.DataFrame(samples).to_csv(out.parent / "call_beta_generalized_targets_samples.csv", index=False)


if __name__ == "__main__":
    main()
