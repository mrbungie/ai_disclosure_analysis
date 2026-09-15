"""Decomposing AI activity grounding (thesis.qmd `fig-ladder-substance`):
every disclosed AI activity is passed through a ladder of ever stricter
concreteness requirements (specific business function -> deployed/scaled
stage -> named product or process -> quantified outcome metric), plus five
independent (non-nested) component rates.

Reads the activity grain of gold (spines/activity/activity joined with
covariates/activity/{extraction, taxonomy}: one row per disclosed AI activity
instance). No LLM/model call: this only tabulates fields already produced by
scripts/gold/activity/build_activity.py.

Output: data/results/washing/ladder_substance.parquet, one row with the
five nested nested-nested counts (c1..c5), their percentages of the total
(pct_c1..pct_c5), and the five independent component rates
(p_func, p_stage, p_named, p_metric, p_vendor).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "common"))
import layers as L  # noqa: E402


def main() -> None:
    df = L.read_dataset("activity", ("covariates", "extraction"), ("covariates", "taxonomy"))
    n_total = len(df)

    # Nested counts: each step adds one more concreteness requirement.
    has_func = df["function_family"] != "unspecified"
    deployed = df["stage"].isin(["deployed", "scaled"])
    named_or_metric = df["evidence_type"].isin(["named", "metric"])
    metric = df["evidence_type"] == "metric"

    c1 = n_total
    c2 = int(has_func.sum())
    c3 = int((has_func & deployed).sum())
    c4 = int((has_func & deployed & named_or_metric).sum())
    c5 = int((has_func & deployed & metric).sum())
    counts = [c1, c2, c3, c4, c5]
    pcts_nested = [c / n_total * 100 for c in counts]

    # Independent (non-nested) component rates.
    p_func = float(has_func.mean() * 100)
    p_stage = float(deployed.mean() * 100)
    p_named = float(named_or_metric.mean() * 100)
    p_metric = float(metric.mean() * 100)
    p_vendor = float((df["source"] == "third_party").mean() * 100)

    row = {
        "n_total": n_total,
        "c1": c1, "c2": c2, "c3": c3, "c4": c4, "c5": c5,
        "pct_c1": pcts_nested[0], "pct_c2": pcts_nested[1], "pct_c3": pcts_nested[2],
        "pct_c4": pcts_nested[3], "pct_c5": pcts_nested[4],
        "p_func": p_func, "p_stage": p_stage, "p_named": p_named,
        "p_metric": p_metric, "p_vendor": p_vendor,
    }
    out_df = pd.DataFrame([row])
    out = L.results_path("washing", "ladder_substance.parquet")
    out_df.to_parquet(out, index=False)
    print(row)
    print(f"-> {out} ({len(out_df):,} row)")


if __name__ == "__main__":
    main()
