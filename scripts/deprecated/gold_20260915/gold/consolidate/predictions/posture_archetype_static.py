"""Materializes data/gold/predictions/firm_year/posture_archetype_static.parquet
by LOADING the already-fitted model (models/posture_archetype_static/model.pkl,
see scripts/gold/posture/build_strategy_dimensions.py) and applying it to
data/gold/covariates/firm_year/posture.parquet (the 8 posture dimensions,
plus `n_frames` for the active-population filter, by ticker-year) -- a
genuinely separate "model + dataset -> predictions" step, not a copy of
the fit script's in-process output.

**Not point-in-time safe**: this is a SINGLE fit over the whole panel, so
every year's label is informed by data from every other year (including
later ones). Descriptive/cross-sectional use only -- never as a predictor
in a regression whose outcome is measured at or after that year. For that,
use posture_archetype_expanding_yearly.parquet or
posture_archetype_expanding_quarterly.parquet instead, both walk-forward
(refit at each cutoff on strictly-prior-or-contemporaneous data only).

Usage:
    uv run python scripts/gold/consolidate/predictions/posture_archetype_static.py
"""
from __future__ import annotations

import os

# Pin BLAS to one thread before numpy/archetypes load: multi-threaded BLAS
# reduction order isn't deterministic run-to-run, which flips `.transform()`
# argmax ties near cluster boundaries (see build_strategy_dimensions.py).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
MODEL_PATH = REPO_ROOT / "models" / "posture_archetype_static" / "model.pkl"
COVARIATES = REPO_ROOT / "data" / "gold" / "covariates" / "firm_year" / "posture.parquet"
OUT = REPO_ROOT / "data" / "gold" / "predictions" / "firm_year" / "posture_archetype_static.parquet"


def main() -> None:
    bundle = joblib.load(MODEL_PATH)
    model, scaler = bundle["model"], bundle["scaler"]
    feature_names, cluster_names = bundle["feature_names"], bundle["cluster_names"]

    df = pd.read_parquet(COVARIATES)
    # NOTE: 3, not the fit's own MIN_FRAMES=5 -- build_strategy_dimensions.py
    # projects the firm-YEAR panel at a lower floor than the cross-sectional
    # fit population uses (see its `panel_active` vs `active`).
    active = df["n_frames"] >= 3
    X = scaler.transform(df.loc[active, feature_names].values)
    W = model.transform(X)
    cluster = pd.Series(np.nan, index=df.index)
    cluster.loc[active] = W.argmax(axis=1)

    out = df[["id", "ticker", "year"]].copy()
    out["cluster"] = cluster
    out["archetype"] = cluster.map(cluster_names).fillna("No AI")
    out.loc[~active, "cluster"] = -1

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    print(f"-> {OUT} ({len(out):,} rows, {len(out.columns)} cols)")


if __name__ == "__main__":
    main()
