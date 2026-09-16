"""Post-call beta regressions (63 trading days).

Design, panel and estimation: scripts/analytics/call_regression.py (W, HistW,
cumulative disclosure intensity and the three posture archetype weights, all
point-in-time for the call). Controls: pre-call beta, log market cap, 60-day
return, ROA; SIC2 x year fixed effects; errors clustered by firm. Leverage
(`debt_to_equity`, `liabilities_to_assets`) comes from
covariates/call/financials (the last 10-K filed before the call).

Models:
  baseline              main specification
  debt_to_equity        matched-sample M0 / M1 with LT debt / equity
  liabilities_to_assets matched-sample M0 / M1 with liabilities / assets
  accounting            matched-sample M0 / M1 adding liabilities / assets to ROA

Outputs data/results/call_beta/call_beta_regressions.csv and
call_beta_regression_samples.csv.

Usage:
  .venv/bin/python scripts/analytics/call_beta/call_beta_regressions.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "analytics"))
import call_regression as C  # noqa: E402

OUTCOME = "beta_post_63"
BASE = C.AI_VARS + ["beta_pre"] + C.BASE_CTRLS
REPORTED = C.AI_VARS + C.REPORTED_WEIGHTS + ["beta_pre"] + C.BASE_CTRLS
LEVERAGE = {"debt_to_equity": ["debt_to_equity"], "liabilities_to_assets": ["liabilities_to_assets"],
            "accounting": ["liabilities_to_assets"]}


def run(panel: pd.DataFrame, model: str) -> tuple[pd.DataFrame, list[dict]]:
    if model == "baseline":
        res, d = C.fit(panel, OUTCOME, BASE)
        return C.coef_table(res, REPORTED, model="baseline", target="beta"), [C.sample_row(res, d, model="baseline")]
    extra = LEVERAGE[model]
    matched = panel.dropna(subset=extra)
    m0, s0 = C.fit(matched, OUTCOME, BASE)
    m1, s1 = C.fit(matched, OUTCOME, BASE + extra)
    assert len(s0) == len(s1), f"{model}: M0/M1 must use the same sample"
    table = pd.concat([C.coef_table(m0, REPORTED, model=f"{model}_M0", target="beta"),
                       C.coef_table(m1, REPORTED + extra, model=f"{model}_M1", target="beta")])
    return table, [C.sample_row(m0, s0, model=f"{model}_M0"), C.sample_row(m1, s1, model=f"{model}_M1")]


def main() -> None:
    panel = C.attach(C.load_call_panel(), "covariates", "financials", ["debt_to_equity", "liabilities_to_assets"])
    tables, samples = [], []
    for model in ["baseline", *LEVERAGE]:
        t, s = run(panel, model)
        tables.append(t)
        samples += s
    out = C.L.results_path("call_beta", "call_beta_regressions.csv")
    table = pd.concat(tables, ignore_index=True)
    table.to_csv(out, index=False)
    pd.DataFrame(samples).to_csv(out.parent / "call_beta_regression_samples.csv", index=False)
    print(pd.DataFrame(samples).to_string(index=False))
    print(table[(table.model == "baseline") & table.variable.isin(C.AI_VARS + C.REPORTED_WEIGHTS)][["variable", "beta_std", "p"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
