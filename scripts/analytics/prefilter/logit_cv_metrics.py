"""
scripts/analytics/prefilter/logit_cv_metrics.py — out-of-fold cross-validation
metrics for the logistic prefilter (`scripts/enrichment/ai_prefilter_classify.py`),
moved out of that script (E-M2): fitting/threshold selection is enrichment's
job, reporting how well the fit does is analytics'.

Reruns the SAME nested GroupKFold(5) procedure `ai_prefilter_classify.py`
uses to pick its deployment C/threshold
(`ai_prefilter_classify.cv_threshold_and_metrics`) against the current golden
set (read-only, additive), and reports two honest out-of-fold metric sets:
the model alone, and the model OR'd with `named_entity_match` in the test
fold (the comparison that used to justify the `use_named_entity` override —
see that script's `main()` for why the override is kept regardless of this
comparison's verdict).

No LLM call, no refit of the deployed model: this is a read-only report over
labels and bronze scores already on disk.

Output: data/results/prefilter/logit_cv_metrics.json

Usage:
    uv run python scripts/analytics/prefilter/logit_cv_metrics.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import GroupKFold

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "common"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "enrichment"))
import layers as L  # noqa: E402
import ai_prefilter_classify as pc  # noqa: E402


def combined_with_named_entity(X: np.ndarray, y: np.ndarray, weights: np.ndarray,
                               groups: np.ndarray, named_entity: np.ndarray) -> dict:
    """Same nested procedure as `pc.cv_threshold_and_metrics`, but OR-combining
    the test-fold prediction with `named_entity_match` before scoring."""
    fit_weights = np.sqrt(weights)
    gkf = GroupKFold(n_splits=5)
    combined_oof = np.zeros(len(y), dtype=int)
    for train_idx, test_idx in gkf.split(X, y, groups):
        best_c, best_t, best_f1 = pc.C_GRID[0], 0.5, -1.0
        for c in pc.C_GRID:
            clf = LogisticRegression(max_iter=2000, C=c)
            clf.fit(X[train_idx], y[train_idx], sample_weight=fit_weights[train_idx])
            proba_train = clf.predict_proba(X[train_idx])[:, 1]
            for t in np.arange(0.05, 0.96, 0.02):
                f1w = f1_score(y[train_idx], (proba_train >= t).astype(int),
                               sample_weight=weights[train_idx], zero_division=0)
                if f1w > best_f1:
                    best_c, best_t, best_f1 = c, float(t), f1w
        clf = LogisticRegression(max_iter=2000, C=best_c)
        clf.fit(X[train_idx], y[train_idx], sample_weight=fit_weights[train_idx])
        proba_test = clf.predict_proba(X[test_idx])[:, 1]
        combined_oof[test_idx] = ((proba_test >= best_t) | named_entity[test_idx]).astype(int)
    return {
        "f1_estrato": float(f1_score(y, combined_oof)),
        "prec_pond": float(precision_score(y, combined_oof, sample_weight=weights)),
        "recall_pond": float(recall_score(y, combined_oof, sample_weight=weights)),
        "f1_pond": float(f1_score(y, combined_oof, sample_weight=weights)),
    }


def main(judge_model: str | None = pc.DEFAULT_JUDGE_MODEL) -> None:
    golden = pc.load_golden(judge_model)
    y = golden["is_ai_mention"].astype(int).values
    weights = golden["inclusion_weight"].astype(float).values
    groups = golden["accession_number"].values
    X = golden[pc.ALL_SIGNAL_COLUMNS].astype(float).values
    named_entity = golden["named_entity_match"].astype(bool).values

    print(f"{len(golden):,} etiquetas del golden set (juez: {judge_model or 'MEZCLA'})")
    deploy_c, threshold, oof_metrics = pc.cv_threshold_and_metrics(X, y, weights, groups)
    print(f"Modelo solo: F1 pond.={oof_metrics['f1_pond']:.3f} prec pond.={oof_metrics['prec_pond']:.3f} "
          f"recall pond.={oof_metrics['recall_pond']:.3f} F1 estrato={oof_metrics['f1_estrato']:.3f}")

    combined_metrics = combined_with_named_entity(X, y, weights, groups, named_entity)
    print(f"Modelo OR named_entity: F1 pond.={combined_metrics['f1_pond']:.3f} "
          f"prec pond.={combined_metrics['prec_pond']:.3f} recall pond.={combined_metrics['recall_pond']:.3f}")

    result = {
        "judge_model": judge_model or "MEZCLA (todos los jueces del golden set)",
        "golden_set_labels": int(len(golden)),
        "deploy_c": deploy_c, "deploy_threshold": threshold,
        "named_entity_matches_in_golden_set": int(named_entity.sum()),
        "cv_metrics": oof_metrics,
        "cv_metrics_with_named_entity": combined_metrics,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    out_path = L.results_path("prefilter", "logit_cv_metrics.json")
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"\n-> {out_path}")


if __name__ == "__main__":
    main()
