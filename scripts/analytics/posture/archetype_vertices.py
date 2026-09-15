"""Standardized archetypal vertices (matrix A, z-scores) of the k=3 posture
Archetypal Analysis fit (thesis.qmd `fig-archetype-heatmap`, ~1255-1311).

Loads the already-fitted model bundle (models/posture_archetype_static/model.pkl,
see scripts/gold/firm/build_posture_archetype_static.py) instead of refitting AA
inline: `bundle["model"].archetypes_` is the (k, n_features) matrix of extreme
points in the standardized feature space the model was fit on, and
`bundle["cluster_names"]` is the same archetype naming the gold prediction
(covariates/firm/posture_archetype_static.parquet) uses, so the vertex
labels and the firm-level archetype labels never diverge.

Output: data/results/posture/archetype_vertices.parquet, long-format
(archetype, feature, value), one row per (archetype, feature) cell of A.

Usage:
    uv run python scripts/analytics/posture/archetype_vertices.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "common"))
import layers as L  # noqa: E402

MODEL_PATH = REPO_ROOT / "models" / "posture_archetype_static" / "model.pkl"


def main() -> None:
    bundle = joblib.load(MODEL_PATH)
    A = bundle["model"].archetypes_
    feature_names = bundle["feature_names"]
    cluster_names = bundle["cluster_names"]

    rows = []
    for cluster_id in range(A.shape[0]):
        archetype = cluster_names[cluster_id]
        for j, feat in enumerate(feature_names):
            rows.append({"archetype": archetype, "feature": feat, "value": float(A[cluster_id, j])})
    out_df = pd.DataFrame(rows)
    out = L.results_path("posture", "archetype_vertices.parquet")
    out_df.to_parquet(out, index=False)
    print(f"-> {out} ({len(out_df):,} rows)")


if __name__ == "__main__":
    main()
