"""Materializes data/gold/predictions/firm_year/washing_score.parquet by
LOADING the persisted grounding-shrinkage priors
(models/washing_grounding_shrinkage/model.pkl, see
scripts/gold/washing/washing_score.py::load_activities_firm_year) and
applying them to raw activity/disclosure inputs -- the reusable half of
the index as a genuinely separate "model + data -> predictions" step.

**Why only the shrinkage prior is "the model", not the whole index**:
`W_it = pctrank(Disclosure_it) - pctrank(Substance_it)` is a percentile
rank, inherently defined relative to the current analytical population --
there's no fixed function you can apply to one new firm-year in isolation.
Only the grounding-component shrinkage (`Grounding_it`, an empirical-Bayes
mean) is a genuinely fitted, reusable parameter; the rank step is
recomputed over the population every run, same as the original
build_gold/washing/washing_score.py::build() does.

Requires models/washing_grounding_shrinkage/model.pkl to already exist
(built by scripts/gold/washing/washing_score.py) and the
`activities`/`disclosure_volume` firm_year covariate families (built by
scripts/gold/consolidate/{activities,disclosure_volume}/build_covariates_firm_year.py).

Usage:
    uv run python scripts/gold/consolidate/predictions/washing_score.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "common"))
import layers as L  # noqa: E402

MODEL_PATH = REPO_ROOT / "models" / "washing_grounding_shrinkage" / "model.pkl"
ACTIVITIES = REPO_ROOT / "data" / "gold" / "covariates" / "firm_year" / "activities.parquet"
DISCLOSURE_VOLUME = REPO_ROOT / "data" / "gold" / "covariates" / "firm_year" / "disclosure_volume.parquet"
OUT = REPO_ROOT / "data" / "gold" / "predictions" / "firm_year" / "washing_score.parquet"
TAIL_Q = 0.05


def _apply_shrink(numerator: pd.Series, denominator: pd.Series, alpha: float, beta_: float) -> pd.Series:
    raw = np.where(denominator > 0, numerator / denominator, np.nan)
    s = pd.Series(raw, index=numerator.index)
    return (s.fillna(0) * denominator + alpha) / (denominator + alpha + beta_)


def load_10k_years() -> pd.DataFrame:
    return (L.scan("silver.filing_manifest")
            .filter((pl.col("country_code") == "us") & (pl.col("form_type") == "10-K")
                    & pl.col("filing_date").is_not_null())
            .select("ticker", pl.col("filing_date").dt.year().cast(pl.Int32).alias("year"))
            .unique(maintain_order=True)
            .collect().to_pandas())


def main() -> None:
    bundle = joblib.load(MODEL_PATH)
    priors, components = bundle["grounding_priors"], bundle["grounding_components"]

    acts = pd.read_parquet(ACTIVITIES)[["ticker", "year", "n_activities"] + components]
    shrunk = pd.concat(
        [_apply_shrink(acts[c], acts["n_activities"], priors[c]["alpha"], priors[c]["beta"]) for c in components],
        axis=1)
    acts["grounding_index"] = shrunk.mean(axis=1)
    acts = acts[["ticker", "year", "n_activities", "grounding_index"]]

    disc = pd.read_parquet(DISCLOSURE_VOLUME)[["ticker", "year", "frames_per_1k", "any_ai", "n_promo", "n_frames"]]
    d = disc.merge(acts, on=["ticker", "year"], how="left")

    has_10k = load_10k_years()
    has_10k["has_10k"] = True
    d = d.merge(has_10k, on=["ticker", "year"], how="left")
    d["has_10k"] = d["has_10k"].fillna(False)
    # No imputation: null n_activities/grounding_index stay null (the as-of
    # below only carries values from fiscal years with their own 10-K).
    d = d.sort_values(["ticker", "year"])
    for col in ("n_activities", "grounding_index"):
        asof = d[col].where(d["has_10k"])
        d[col] = asof.groupby(d["ticker"]).ffill()
    d = d.dropna(subset=["n_activities", "grounding_index"])
    d = d.drop(columns="has_10k")
    d = d[d["any_ai"] > 0].reset_index(drop=True)

    d["substance"] = np.log1p(d["n_activities"]) * d["grounding_index"]
    d["pct_disclosure"] = d["frames_per_1k"].rank(pct=True)
    d["pct_substance"] = d["substance"].rank(pct=True)
    d["w"] = d["pct_disclosure"] - d["pct_substance"]
    q_hi, q_lo = d["w"].quantile(1 - TAIL_Q), d["w"].quantile(TAIL_Q)
    d["washing"] = d["w"] >= q_hi
    d["callada"] = d["w"] <= q_lo

    d.insert(0, "id", d["ticker"] + "_" + d["year"].astype(int).astype(str))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    d.to_parquet(OUT, index=False)
    print(f"-> {OUT} ({len(d):,} rows, {len(d.columns)} cols)")


if __name__ == "__main__":
    main()
