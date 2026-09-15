"""NCSKEW/DUVOL tail-risk regressions on earnings calls.

Design, panel and estimation: scripts/analytics/call_regression.py (W, HistW,
cumulative disclosure intensity and the three posture archetype weights,
all point-in-time for the call). Controls: pre-call level of the outcome,
log market cap, 60-day return, ROA; SIC2 x year fixed effects; errors
clustered by firm. `base` models omit the archetype weights.

Outputs data/results/crash_archetypes/call_crash_archetype_regressions.csv
(targets ncskew_base, ncskew_weights, duvol_base, duvol_weights) and
call_crash_archetype_regression_samples.csv.

Usage:
  .venv/bin/python scripts/analytics/crash_archetypes/call_crash_regressions.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "analytics"))
import call_regression as C  # noqa: E402

OUTCOMES = [("ncskew", "ncskew_post", "ncskew_pre"), ("duvol", "duvol_post", "duvol_pre")]


def main() -> None:
    panel = C.attach_crash_risk(C.load_call_panel())
    tables, samples = [], []
    for name, y_col, pre_col in OUTCOMES:
        regressors = C.AI_VARS + [pre_col] + C.controls_for(y_col)
        for weights in (False, True):
            target = f"{name}_{'weights' if weights else 'base'}"
            res, d = C.fit(panel, y_col, regressors, raw=C.WEIGHTS if weights else [])
            tables.append(C.coef_table(res, C.AI_VARS + (C.WEIGHTS if weights else []) + [pre_col], target=target))
            samples.append(C.sample_row(res, d, target=target))
            print(f"{target}: N={len(d)} firms={d.ticker.nunique()} R2={res.rsquared:.4f}")

    export = pd.concat(tables, ignore_index=True)
    export.to_csv(C.L.results_path("crash_archetypes", "call_crash_archetype_regressions.csv"), index=False)
    pd.DataFrame(samples).to_csv(C.L.results_path("crash_archetypes", "call_crash_archetype_regression_samples.csv"), index=False)
    print(export[export["variable"].isin(C.AI_VARS + C.WEIGHTS)][["target", "variable", "beta_std", "p"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
