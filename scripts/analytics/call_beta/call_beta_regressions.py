"""Post-call beta regressions.

Design, panel and estimation: scripts/analytics/call_regression.py (W, HistW,
cumulative disclosure intensity and the three posture archetype weights, all
point-in-time for the call). Controls: pre-call beta, log market cap, 60-day
return, ROA; SIC2 x year fixed effects; errors clustered by firm. Leverage
(`debt_to_equity`, `liabilities_to_assets`) comes from
covariates/call/financials (the last 10-K filed before the call).

Headline: `symmetric_63` (row `model == "symmetric_63"`) is the primary
specification -- beta_post_63 (trading days [+21, +63) after the call) on
beta_pre_63 (trading days [-63, -21) before it), the SAME 42-day window
length and the same exclusion of the +-21 trading-day announcement period on
both sides, so the pre/post comparison reads as the effect of the call
rather than a difference of window length or period. `baseline` is the prior,
asymmetric design (beta_post_63 on beta_pre, a 252-day, no-gap pre window)
and is kept unchanged, as a reported comparison row -- and because
data/results/call_beta/call_beta_regression_samples.csv's `model=="baseline"`
row and call_beta_regressions.csv's `model=="baseline"` rows are read
elsewhere (thesis_document/thesis.qmd @tbl-call-beta-paper).

Models:
  baseline               prior asymmetric design: beta_post_63 ~ beta_pre (comparison row, unchanged)
  symmetric_63           HEADLINE: beta_post_63 ~ beta_pre_63 (42-day windows, same gap both sides)
  symmetric_252          robustness: beta_post_252 ~ beta_pre_252 (231-day windows, same gap both sides)
  delta_beta             robustness: beta_post_63 - beta_pre_63 ~ AI/weights + beta_pre_63 + controls
  delta_beta_nolevel     robustness: same outcome, without the beta_pre_63 control
  debt_to_equity         matched-sample M0 / M1 with LT debt / equity (on the symmetric_63 spec)
  liabilities_to_assets  matched-sample M0 / M1 with liabilities / assets (on the symmetric_63 spec)
  accounting             matched-sample M0 / M1 adding liabilities / assets to ROA (on the symmetric_63 spec)

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

# The prior, asymmetric design -- kept unchanged as a reported comparison row.
OUTCOME_OLD = "beta_post_63"
BASE_OLD = C.AI_VARS + ["beta_pre"] + C.BASE_CTRLS
REPORTED_OLD = C.AI_VARS + C.REPORTED_WEIGHTS + ["beta_pre"] + C.BASE_CTRLS

# Headline: same window length (42 trading days) and the same +-21-day
# announcement-period exclusion on both sides of the call.
OUTCOME_SYM63 = "beta_post_63"
BASE_SYM63 = C.AI_VARS + ["beta_pre_63"] + C.BASE_CTRLS
REPORTED_SYM63 = C.AI_VARS + C.REPORTED_WEIGHTS + ["beta_pre_63"] + C.BASE_CTRLS

# Robustness (a): the same symmetric design at the 231-day horizon.
OUTCOME_SYM252 = "beta_post_252"
BASE_SYM252 = C.AI_VARS + ["beta_pre_252"] + C.BASE_CTRLS
REPORTED_SYM252 = C.AI_VARS + C.REPORTED_WEIGHTS + ["beta_pre_252"] + C.BASE_CTRLS

# Robustness (b): the change itself as the outcome, with and without the pre level as a control.
BASE_DELTA = C.AI_VARS + ["beta_pre_63"] + C.BASE_CTRLS
REPORTED_DELTA = C.AI_VARS + C.REPORTED_WEIGHTS + ["beta_pre_63"] + C.BASE_CTRLS
BASE_DELTA_NOLEVEL = C.AI_VARS + C.BASE_CTRLS
REPORTED_DELTA_NOLEVEL = C.AI_VARS + C.REPORTED_WEIGHTS + C.BASE_CTRLS

LEVERAGE = {"debt_to_equity": ["debt_to_equity"], "liabilities_to_assets": ["liabilities_to_assets"],
            "accounting": ["liabilities_to_assets"]}


def run_leverage(panel: pd.DataFrame, model: str) -> tuple[pd.DataFrame, list[dict]]:
    """Matched-sample M0 (headline symmetric_63 controls) / M1 (+leverage)."""
    extra = LEVERAGE[model]
    matched = panel.dropna(subset=extra)
    m0, s0 = C.fit(matched, OUTCOME_SYM63, BASE_SYM63)
    m1, s1 = C.fit(matched, OUTCOME_SYM63, BASE_SYM63 + extra)
    assert len(s0) == len(s1), f"{model}: M0/M1 must use the same sample"
    table = pd.concat([C.coef_table(m0, REPORTED_SYM63, model=f"{model}_M0", target="beta"),
                       C.coef_table(m1, REPORTED_SYM63 + extra, model=f"{model}_M1", target="beta")])
    return table, [C.sample_row(m0, s0, model=f"{model}_M0"), C.sample_row(m1, s1, model=f"{model}_M1")]


def main() -> None:
    panel = C.load_call_panel()
    panel = C.attach(panel, "covariates", "financials", ["debt_to_equity", "liabilities_to_assets"])
    panel = C.attach(panel, "covariates", "market", ["beta_pre_63", "beta_pre_252"])
    panel["delta_beta_63"] = panel["beta_post_63"] - panel["beta_pre_63"]

    tables, samples = [], []

    res, d = C.fit(panel, OUTCOME_OLD, BASE_OLD)
    tables.append(C.coef_table(res, REPORTED_OLD, model="baseline", target="beta"))
    samples.append(C.sample_row(res, d, model="baseline"))

    res, d = C.fit(panel, OUTCOME_SYM63, BASE_SYM63)
    tables.append(C.coef_table(res, REPORTED_SYM63, model="symmetric_63", target="beta"))
    samples.append(C.sample_row(res, d, model="symmetric_63"))

    res, d = C.fit(panel, OUTCOME_SYM252, BASE_SYM252)
    tables.append(C.coef_table(res, REPORTED_SYM252, model="symmetric_252", target="beta"))
    samples.append(C.sample_row(res, d, model="symmetric_252"))

    res, d = C.fit(panel, "delta_beta_63", BASE_DELTA)
    tables.append(C.coef_table(res, REPORTED_DELTA, model="delta_beta", target="beta"))
    samples.append(C.sample_row(res, d, model="delta_beta"))

    res, d = C.fit(panel, "delta_beta_63", BASE_DELTA_NOLEVEL)
    tables.append(C.coef_table(res, REPORTED_DELTA_NOLEVEL, model="delta_beta_nolevel", target="beta"))
    samples.append(C.sample_row(res, d, model="delta_beta_nolevel"))

    for model in LEVERAGE:
        t, s = run_leverage(panel, model)
        tables.append(t)
        samples += s

    out = C.L.results_path("call_beta", "call_beta_regressions.csv")
    table = pd.concat(tables, ignore_index=True)
    table.to_csv(out, index=False)
    pd.DataFrame(samples).to_csv(out.parent / "call_beta_regression_samples.csv", index=False)
    print(pd.DataFrame(samples).to_string(index=False))
    for model in ["baseline", "symmetric_63", "symmetric_252", "delta_beta", "delta_beta_nolevel"]:
        sub = table[(table.model == model) & table.variable.isin(C.AI_VARS + C.REPORTED_WEIGHTS)]
        print(f"\n== {model} ==")
        print(sub[["variable", "beta_std", "p"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
