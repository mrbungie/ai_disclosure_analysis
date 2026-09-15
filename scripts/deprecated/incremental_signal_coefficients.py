"""DEPRECATED (G-H4/A-H1): gold must not read a model that analytics fits
and must not hold a coefficient/p-value table. The coefficient table now
comes straight out of scripts/analytics/shock/incremental_signal.py's own
`coefficients()` return value, written to
data/results/shock/incremental_signal_coefficients.csv. Not run by
anything; kept for history.

Materializes data/gold/predictions/firm_year/incremental_signal_coefficients.parquet
by LOADING each outcome's persisted M2 OLS
(models/incremental_signal/<outcome>/model.pkl, see
scripts/analytics/shock/incremental_signal.py::coefficients) and reading
its standardized coefficients, CIs, and p-values straight off the fitted
object -- a genuine "model -> predictions" step, not a reformat of a
report some other script already wrote.

**Why there is no incremental_signal.parquet summary file here**: R²/
ΔR²/Wald/permutation-test results compare FIVE nested models (M0..M4) per
outcome, only one of which (M2) is a persisted, re-appliable fit -- M0/M1/
M3/M4 are computed with a fast closed-form R² helper, never fit as full
model objects, and the Wald/bootstrap/permutation numbers come from
resampling, not from any single model. That comparison is an analytics
REPORT (scripts/analytics/shock/incremental_signal.py's own JSON output,
data/processed/clusters/incremental_signal.json), not something derivable
from one persisted model -- it doesn't belong under predictions/, which is
specifically for model-applied-to-dataset output.

Discovers outcomes by listing models/incremental_signal/ -- one
subdirectory per outcome, no manifest needed.

Usage:
    uv run python scripts/gold/consolidate/predictions/incremental_signal_coefficients.py
"""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
MODELS_DIR = REPO_ROOT / "models" / "incremental_signal"
OUT = REPO_ROOT / "data" / "gold" / "predictions" / "firm_year" / "incremental_signal_coefficients.parquet"

# Mirrors SEMANTIC (+ "volumen") in scripts/analytics/shock/incremental_signal.py --
# the M2 model also carries fundamentals/archetype dummies as controls,
# not reported here since they aren't the semantic block being tested.
SEMANTIC_FEATURES = {"realizado", "despliegue", "capacidad", "riesgo", "gobernanza", "promocional", "especificidad", "volumen"}


def main() -> None:
    rows = []
    for model_dir in sorted(MODELS_DIR.iterdir()):
        if not model_dir.is_dir():
            continue
        outcome = model_dir.name
        bundle = joblib.load(model_dir / "model.pkl")
        res, feature_names = bundle["model"], bundle["feature_names"]
        intervals = res.conf_int(alpha=0.05)
        for name, beta, p, (lo, hi) in zip(feature_names, res.params, res.pvalues, intervals):
            if name not in SEMANTIC_FEATURES:
                continue
            rows.append({
                "id": f"{outcome}__{name}", "outcome": outcome, "feature": name,
                "beta_std": float(beta), "ci95_low": float(lo), "ci95_high": float(hi), "p": float(p),
            })

    out = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    print(f"-> {OUT} ({len(out):,} rows, {len(out.columns)} cols)")


if __name__ == "__main__":
    main()
