"""Posture configuration + decoupling across horizons: post-call beta (63d),
NCSKEW (105d) and NCSKEW (63d, matching the beta window).

Design, panel and estimation: scripts/analytics/call_regression.py. Controls:
the outcome's pre-call level, log market cap, 60-day return, ROA; SIC2 x year
fixed effects; errors clustered by firm.

Output: data/results/call_beta/config_decoupling_asof.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "analytics"))
import call_regression as C  # noqa: E402

OUTCOMES = [("beta_post_63", "beta_pre"), ("ncskew_post", "ncskew_pre"), ("ncskew_post_63", "ncskew_pre_63")]


def main() -> None:
    panel = C.attach_crash_risk(C.load_call_panel())
    panel = C.attach(panel, "covariates", "market", ["ncskew_pre_63"])
    panel = C.attach(panel, "targets", "market", ["ncskew_post_63"])
    tables = []
    for y_col, pre_col in OUTCOMES:
        res, d = C.fit(panel, y_col, C.AI_VARS + [pre_col] + C.controls_for(y_col))
        t = C.coef_table(res, C.AI_VARS + C.WEIGHTS + [pre_col], target=y_col)
        t["n_calls"], t["n_firms"], t["r2"] = len(d), d.ticker.nunique(), res.rsquared
        tables.append(t)
        print(f"\n=== {y_col}: N={len(d)}, firms={d.ticker.nunique()}, R2={res.rsquared:.4f}")
        print(t[["variable", "beta_std", "p"]].round(4).to_string(index=False))
    out = C.L.results_path("call_beta", "config_decoupling_asof.csv")
    pd.concat(tables, ignore_index=True).to_csv(out, index=False)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
