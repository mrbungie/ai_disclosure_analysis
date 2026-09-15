"""Materializes data/gold/datasets/firm_year/firm_year.parquet: the four
yearly covariate families (posture, disclosure_volume, financial_ratios,
market) JOINED with targets on `id` (= f"{ticker}_{fiscal_year}") -- the
analysis-ready panel thesis.qmd reads directly. This is the TASK-SPECIFIC
join; the covariate families themselves stay orthogonal and reusable (a
different dataset could join only posture + market, say, without
financial_ratios).

**Lead of each target, explicit** (see build_targets.py for the full
naming convention): every covariate is measured in fiscal_year `t` (the
SAME row's `id`). Targets fall into three groups relative to that `t`:
  - contemporaneous (t):      price_to_sales_t, rd_intensity_t
  - pre/post-filing windows:  beta_252d (pre), idio_vol_252d (pre),
                              vol_pre_60d (pre), vol_post_60d (post),
                              car_m1_p5d (event window around the filing)
  - forward one fiscal year (t -> t+1): revenue_growth_lead1y,
                              rd_expense_growth_lead1y, capex_growth_lead1y,
                              sga_expense_growth_lead1y
None of the *_lead1y columns use information from AFTER fiscal_year t+1 --
they are the single next fiscal year's growth, already computed upstream
in build_firm_financials.py from strictly-later filings.

Requires every scripts/gold/consolidate/{posture,disclosure_volume,
financial_ratios,market}/build_covariates_firm_year.py and this folder's
build_targets.py to have already run.

Usage:
    uv run python scripts/gold/consolidate/firm_year/build_dataset.py
"""
from __future__ import annotations

from functools import reduce
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
COVARIATES = REPO_ROOT / "data" / "gold" / "covariates" / "firm_year"
COVARIATE_FAMILIES = ["posture", "disclosure_volume", "financial_ratios", "market"]
TARGETS = REPO_ROOT / "data" / "gold" / "targets" / "firm_year" / "firm_year.parquet"
OUT = REPO_ROOT / "data" / "gold" / "datasets" / "firm_year" / "firm_year.parquet"


def main() -> None:
    families = [pd.read_parquet(COVARIATES / f"{name}.parquet") for name in COVARIATE_FAMILIES]
    # Every family after the first drops ticker/year -- already on `id` and
    # on the first family's columns, would otherwise collide on merge.
    # disclosure_volume also carries n_frames (identical values -- both are
    # ai_intensity.py document counts for the same (ticker, year)); keep
    # posture's copy only, so the join doesn't silently split it into
    # n_frames_x/n_frames_y.
    families = [families[0]] + [f.drop(columns=["ticker", "year"]).drop(columns=["n_frames"], errors="ignore")
                                for f in families[1:]]
    cov = reduce(lambda left, right: left.merge(right, on="id", how="left", validate="one_to_one"), families)

    tgt = pd.read_parquet(TARGETS).drop(columns=["ticker", "year"])
    out = cov.merge(tgt, on="id", how="left", validate="one_to_one")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    print(f"-> {OUT} ({len(out):,} rows, {len(out.columns)} cols)")


if __name__ == "__main__":
    main()
