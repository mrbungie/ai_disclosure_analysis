"""k=2 Archetypal Analysis fit on the pooled posture population (thesis.qmd,
Appendix C, "Archetype count: PCA, k=2 comparison, fixed-N, and K-means
robustness"): the substantive-rejection paragraph there needs the k=2
vertex coordinates on 4 of the 8 posture dimensions (promotional, risk,
governance, specificity) to show that collapsing to two poles erases the
governance-orientation distinction that the k=3 taxonomy (Chapter 3) keeps.

This is NOT the k=3 fit (that lives in
`scripts/gold/firm/build_posture_archetype_static.py` and is gold, not
analytics) -- it is a one-off robustness comparison against a different k,
run here rather than inline in the qmd so no AA().fit() remains in the
document.

Same population, same 8 standardized posture dimensions, and the same seed
as the k=3 gold fit: reads the k=3 fit's own firm-grain output
(`data/gold/covariates/firm/posture_archetype_static.parquet`), restricts
to firms with an actual AI archetype (drops "No AI"), re-standardizes the 8
posture dimensions, and refits Archetypal Analysis with n_archetypes=2,
random_state=42, max_iter=500 -- verified to reproduce, to full float
precision, the numbers the thesis.qmd chunk it replaces computed on the
pre-migration `firm_strategy_dimensions.parquet`.

Output: data/results/posture/archetype_k2_fit.json.

Usage:
    uv run python scripts/analytics/posture/archetype_k2_fit.py
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "common"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "gold" / "posture"))
from posture_features import CLUSTER_FEATURES, fit_aa  # noqa: E402
import layers as L  # noqa: E402

with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    pass

# fitted features come from the one place that defines them: intensity is a
# covariate, not a posture, so it does not shape the vertices
FEATS = list(CLUSTER_FEATURES)


def main() -> None:
    df = L.read_gold("firm", ("covariates", "posture_archetype_static"))
    fit = df[df["archetype"] != "No AI"]
    X = fit[FEATS].values
    X_std = (X - X.mean(axis=0)) / X.std(axis=0, ddof=0)

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        aa, _ = fit_aa(X_std, 2)
    A = aa.archetypes_

    def dim(name):
        return A[:, FEATS.index(name)].tolist()

    result = {
        "n_firms": int(len(fit)),
        "k2_promo": dim("promotional_posture"),
        "k2_risk": dim("risk_orientation"),
        "k2_gov": dim("governance_orientation"),
        "k2_spec": dim("specificity"),
    }
    out = L.results_path("posture", "archetype_k2_fit.json")
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
