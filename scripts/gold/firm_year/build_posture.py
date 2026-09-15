"""Firm-year posture: the eight posture dimensions and the static archetype
projected on the panel.

  covariates/firm_year/posture                   the seven posture rates over
      the year's posture-form frames (10-K, 8-K, DEF 14A) and
      disclosure_intensity. Firm-years with at least 3 filing frames get the
      rates shrunk to the prior of that year's firm-years and the percentile
      rank of frames per 1,000 words within that year; below that the rates
      are the raw rates and disclosure_intensity is null. A firm-year without
      posture-form frames has null rates. Prior and rank use the information
      of the year's cross-section only.
  covariates/firm_year/posture_archetype_static  cluster and archetype of the
      static full-sample model (models/posture_archetype_static, fit by
      scripts/gold/firm/build_posture_archetype_static.py); "No AI" (cluster
      -1) below 3 filing frames. The model is applied to its own pooled
      construction of the eight dimensions (prior and intensity rank over
      every active firm-year of the panel), not to the within-year covariate:
      descriptive, not point in time.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "gold" / "posture"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "gold" / "firm"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "common"))
from posture_features import CLUSTER_FEATURES, INTENSITY, POSTURE, build_posture, load_frames, shrink_to_prior  # noqa: E402

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import layers as L  # noqa: E402
from build_posture_archetype_static import MODEL_PATH  # noqa: E402

BUILDER = "scripts/gold/firm_year/build_posture.py"
MIN_PANEL_FRAMES = 3


def posture_dimensions(panel: pd.DataFrame) -> pd.DataFrame:
    """The eight dimensions of `panel`'s rows: rates shrunk to the rows' prior
    and the percentile rank of frames per 1,000 words among the rows."""
    # Shrinkage input only: a firm-year without posture-form frames
    # contributes zero rates to the prior and gets the shrunk value of a zero
    # rate, which the static model classifies like any other active firm-year.
    shrunk = shrink_to_prior(panel[POSTURE].fillna(0.0), panel["n_frames"])
    shrunk[INTENSITY] = panel["frames_per_1k"].rank(pct=True)
    return shrunk[CLUSTER_FEATURES]


def main() -> None:
    frames = load_frames()
    panel = L.read_gold("firm_year", ("covariates", "disclosure_volume", ["n_frames", "frames_per_1k"]))
    panel = panel.merge(build_posture(frames, ["ticker", "year"]).drop(columns="n_posture_frames"),
                        on=["ticker", "year"], how="left", validate="one_to_one")
    active = panel["n_frames"] >= MIN_PANEL_FRAMES
    pooled = posture_dimensions(panel[active])  # static projection input only, never written
    within_year = pd.concat([posture_dimensions(g) for _, g in panel[active].groupby("year", sort=True)])
    panel[INTENSITY] = np.nan
    for col in CLUSTER_FEATURES:
        panel.loc[active, col] = within_year[col]
    spine = L.GOLD_SPINE_COLUMNS["firm_year"]
    L.write_gold("covariates", "firm_year", "posture", panel[spine + CLUSTER_FEATURES], builder=BUILDER,
                 extra={"population_fit": "shrinkage prior and intensity rank within each calendar year"})

    bundle = joblib.load(MODEL_PATH)
    X = bundle["scaler"].transform(pooled[bundle["feature_names"]].values)
    cluster = pd.Series(np.nan, index=panel.index)
    cluster.loc[active] = bundle["model"].transform(X).argmax(axis=1)
    out = panel[spine].copy()
    out["cluster"] = cluster
    out["archetype"] = cluster.map(bundle["cluster_names"]).fillna("No AI")
    out.loc[~active, "cluster"] = -1
    L.write_gold("covariates", "firm_year", "posture_archetype_static", out, builder=BUILDER, inputs=[MODEL_PATH],
                 extra={"point_in_time": False,
                        "use": "descriptive: static full-sample model on dimensions pooled over the whole panel"})


if __name__ == "__main__":
    main()
