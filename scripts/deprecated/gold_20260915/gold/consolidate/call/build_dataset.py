"""Materializes data/gold/datasets/call/call.parquet: the three
quarterly covariate families (disclosure, market, financial_ratios)
JOINED with targets on `id` (= call_accession_number, already
f"{ticker}_{fiscal_year}Q{n}") -- the analysis-ready call panel
thesis.qmd reads directly. This is the TASK-SPECIFIC join; the covariate
families themselves stay orthogonal and reusable.

**Lead of each target, explicit**: every covariate is observed strictly
BEFORE or AT the call (pre-call beta/price, disclosure history up to and
including this call). Every target is a FORWARD window measured from the
call date: beta_post_63d/126d/252d and return_post_60d are all
post-call -- none of them can leak into the covariates side, which is the
entire reason build_call_beta_panel.py keeps this a strict backward
merge_asof (see docs/analytics/analysis_spines.md, Spine A).

Requires every scripts/gold/consolidate/{disclosure_volume,market,
financial_ratios}/build_covariates_call.py and this folder's
build_targets.py to have already run.

Usage:
    uv run python scripts/gold/consolidate/call/build_dataset.py
"""
from __future__ import annotations

from functools import reduce
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
COVARIATES = REPO_ROOT / "data" / "gold" / "covariates" / "call"
COVARIATE_FILES = {"disclosure_volume": "disclosure.parquet", "market": "market.parquet",
                    "financial_ratios": "financial_ratios.parquet"}
TARGETS = REPO_ROOT / "data" / "gold" / "targets" / "call" / "call.parquet"
OUT = REPO_ROOT / "data" / "gold" / "datasets" / "call" / "call.parquet"


def main() -> None:
    families = [pd.read_parquet(COVARIATES / fname) for fname in COVARIATE_FILES.values()]
    # Only disclosure.parquet carries ticker/fecha -- the other two drop
    # call_accession_number too, already on `id` and on the first family.
    families = [families[0]] + [f.drop(columns=["call_accession_number"]) for f in families[1:]]
    cov = reduce(lambda left, right: left.merge(right, on="id", how="left", validate="one_to_one"), families)

    tgt = pd.read_parquet(TARGETS).drop(columns=["call_accession_number"])
    out = cov.merge(tgt, on="id", how="left", validate="one_to_one")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    print(f"-> {OUT} ({len(out):,} rows, {len(out.columns)} cols)")


if __name__ == "__main__":
    main()
