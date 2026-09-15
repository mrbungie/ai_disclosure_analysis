"""Appendix G cohort-grounding and rollup-sensitivity tables
(thesis.qmd `tbl-h5-cohort-grounding` and `tbl-h8-rollup-sensitivity`,
~4901-4958). Reproduces the two `docs/analytics/appendix_g_*.csv` files
the qmd chunks currently write inline via their own `_export()` helper,
reading gold:
  - covariates/firm_year/washing_score (scored firm-years: `w` not null)
  - spines/activity/activity + covariates/activity/{extraction, taxonomy}

No LLM/model call: pure groupby/mean reshaping of already-persisted gold.

Writes:
  - data/results/washing/appendix_g_cohort_grounding.csv
    (year, fixed_cohort_grounding, new_filer_year_grounding)
  - data/results/washing/appendix_g_rollup_sensitivity.csv
    (channel, activity_weighted_pct, firm_weighted_pct)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "common"))
import layers as L  # noqa: E402


def cohort_grounding(wy: pd.DataFrame) -> pd.DataFrame:
    first_year = wy.groupby("ticker")["year"].min()
    y0 = int(wy["year"].min())
    fixed_cohort = first_year[first_year == y0].index
    wy = wy.copy()
    wy["is_new_filer_year"] = wy["year"] == wy["ticker"].map(first_year)

    fixed_traj = wy[wy["ticker"].isin(fixed_cohort)].groupby("year")["grounding_index"].mean()
    new_filer_traj = wy[wy["is_new_filer_year"]].groupby("year")["grounding_index"].mean()
    years = sorted(wy["year"].unique())
    tbl = pd.DataFrame({"fixed_cohort_grounding": fixed_traj,
                        "new_filer_year_grounding": new_filer_traj}).reindex(years)
    tbl.index.name = "year"
    return tbl


def rollup_sensitivity(act: pd.DataFrame) -> pd.DataFrame:
    act = act.copy()
    act["grounded"] = act["evidence_type"].isin(["named", "metric"])
    activity_weighted = act.groupby("channel")["grounded"].mean() * 100
    firm_share = act.groupby(["ticker", "channel"])["grounded"].mean().reset_index()
    firm_weighted = firm_share.groupby("channel")["grounded"].mean() * 100
    tbl = pd.DataFrame({"activity_weighted_pct": activity_weighted, "firm_weighted_pct": firm_weighted})
    tbl.index.name = "channel"
    return tbl


def main() -> None:
    wy = L.read_gold("firm_year", ("covariates", "washing_score"))
    wy = wy[wy["w"].notna()].reset_index(drop=True)
    act = L.read_dataset("activity", ("covariates", "extraction"), ("covariates", "taxonomy"))

    cohort_tbl = cohort_grounding(wy)
    cohort_out = L.results_path("washing", "appendix_g_cohort_grounding.csv")
    cohort_tbl.to_csv(cohort_out)

    rollup_tbl = rollup_sensitivity(act)
    rollup_out = L.results_path("washing", "appendix_g_rollup_sensitivity.csv")
    rollup_tbl.to_csv(rollup_out)

    print(cohort_tbl)
    print(f"-> {cohort_out}")
    print(rollup_tbl)
    print(f"-> {rollup_out}")


if __name__ == "__main__":
    main()
