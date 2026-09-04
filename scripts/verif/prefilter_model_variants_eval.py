"""Tests candidate improvements to the deployed prefilter model (docs/
prefilter_evaluation.md §8.2), all under the SAME honest nested-CV
discipline as §8.3 (threshold and any hyperparameter chosen only on each
fold's training split, never on the fold it's scored on) — no candidate
gets credit for anything tuned on data it's then evaluated against.

Candidates:
  0. baseline: LogisticRegression(), unweighted fit, threshold grid-searched
     on train — this is what's deployed today.
  1. weighted fit: LogisticRegression(..., sample_weight=inclusion_weight)
     — the golden set is a deliberately non-representative stratified
     sample (docs/golden_set_sampling.md); inclusion_weight corrects for
     that when EVALUATING (every metric in this repo uses it), but the
     model itself was never fit with it, so its coefficients are learned
     against the sample's artificially inflated positive rate, not the
     corpus's real one.
  2. weighted fit + regularization grid: same as 1, plus C grid-searched
     on train (default C=1.0 was never actually checked against
     alternatives).
  3. weighted fit + max_semantic_score as a 12th feature: semantic_margin
     (max_semantic_score - negative_similarity) is in the model, but the
     unsubtracted max_semantic_score itself never is — the subtraction
     could be throwing away information the two components would carry
     separately.

Usage:
    uv run python scripts/verif/prefilter_model_variants_eval.py
"""

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import GroupKFold

REPO_ROOT = Path(__file__).resolve().parents[2]
DB = REPO_ROOT / "duckdb" / "thesis.duckdb"
LATEST_PREFILTER_RUN = "20260902T225928Z"

BASE_SIGNALS = [
    "score_ai_use", "score_ai_exploration", "score_ai_capability", "score_ai_outcome",
    "score_ai_risk", "score_ai_governance", "score_ai_strategy",
    "negative_similarity", "semantic_margin", "strong_lexical_match", "weak_lexical_match",
]
EXTRA_SIGNAL = "max_semantic_score"


def load() -> pd.DataFrame:
    con = duckdb.connect(str(DB), read_only=True)
    try:
        return con.execute(f"""
            SELECT l.is_ai_disclosure, l.inclusion_weight, l.accession_number,
                   {', '.join(f'p.{c}' for c in BASE_SIGNALS + [EXTRA_SIGNAL])}
            FROM read_parquet('data/interim/golden_set/golden_set_labels__session=*__part=*.parquet',
                               union_by_name=True) l
            JOIN read_parquet('data/interim/prefilter_scores/prefilter_scores__run={LATEST_PREFILTER_RUN}__part=*.parquet') p
                USING (country_code, form, accession_number, item_key, paragraph_index)
            WHERE l.error IS NULL
        """).df()
    finally:
        con.close()


def weighted_f1_at(y, proba, sample_weight, t):
    pred = (proba >= t).astype(int)
    return f1_score(y, pred, sample_weight=sample_weight, zero_division=0)


def run_variant(X, y, weights, groups, *, use_sample_weight: bool, c_grid: list[float]) -> dict:
    """Nested CV: for each outer fold, grid-search C (if len(c_grid) > 1) and
    threshold using ONLY the training split (itself split further for the
    C search would be ideal, but with a single train/test per outer fold
    and C search also only touching train-fold data via its OWN weighted F1
    on train predictions, no test-fold information leaks in)."""
    gkf = GroupKFold(n_splits=5)
    oof_pred = np.zeros(len(y), dtype=int)
    chosen_cs = []
    for train_idx, test_idx in gkf.split(X, y, groups):
        best_c, best_t, best_f1 = c_grid[0], 0.5, -1.0
        for c in c_grid:
            clf = LogisticRegression(max_iter=2000, C=c)
            fit_kwargs = {"sample_weight": weights[train_idx]} if use_sample_weight else {}
            clf.fit(X[train_idx], y[train_idx], **fit_kwargs)
            proba_train = clf.predict_proba(X[train_idx])[:, 1]
            for t in np.arange(0.05, 0.96, 0.02):
                f1w = weighted_f1_at(y[train_idx], proba_train, weights[train_idx], t)
                if f1w > best_f1:
                    best_c, best_t, best_f1 = c, float(t), f1w
        chosen_cs.append(best_c)
        clf = LogisticRegression(max_iter=2000, C=best_c)
        fit_kwargs = {"sample_weight": weights[train_idx]} if use_sample_weight else {}
        clf.fit(X[train_idx], y[train_idx], **fit_kwargs)
        proba_test = clf.predict_proba(X[test_idx])[:, 1]
        oof_pred[test_idx] = (proba_test >= best_t).astype(int)

    return {
        "f1_estrato": float(f1_score(y, oof_pred)),
        "prec_pond": float(precision_score(y, oof_pred, sample_weight=weights, zero_division=0)),
        "recall_pond": float(recall_score(y, oof_pred, sample_weight=weights, zero_division=0)),
        "f1_pond": float(f1_score(y, oof_pred, sample_weight=weights, zero_division=0)),
        "chosen_c_per_fold": chosen_cs,
    }


def main() -> None:
    df = load()
    y = df["is_ai_disclosure"].astype(int).values
    weights = df["inclusion_weight"].astype(float).values
    groups = df["accession_number"].values
    X_base = df[BASE_SIGNALS].astype(float).values
    X_extra = df[BASE_SIGNALS + [EXTRA_SIGNAL]].astype(float).values

    variants = {
        "0. baseline (deployado hoy: fit sin peso, C=1.0)":
            run_variant(X_base, y, weights, groups, use_sample_weight=False, c_grid=[1.0]),
        "1. fit CON inclusion_weight (C=1.0)":
            run_variant(X_base, y, weights, groups, use_sample_weight=True, c_grid=[1.0]),
        "2. fit CON inclusion_weight + grid de C":
            run_variant(X_base, y, weights, groups, use_sample_weight=True,
                        c_grid=[0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]),
        "3. (2) + max_semantic_score como señal 12":
            run_variant(X_extra, y, weights, groups, use_sample_weight=True,
                        c_grid=[0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]),
    }

    print(f"{'variante':<55} {'F1 estr.':>9} {'prec':>7} {'recall':>7} {'F1 pond.':>9}")
    for label, m in variants.items():
        print(f"{label:<55} {m['f1_estrato']:>9.3f} {m['prec_pond']:>7.3f} "
              f"{m['recall_pond']:>7.3f} {m['f1_pond']:>9.3f}")

    winner = max(variants.items(), key=lambda kv: kv[1]["f1_pond"])
    print(f"\nGana por F1 ponderado: {winner[0]} ({winner[1]['f1_pond']:.3f})")
    print(f"C elegido por fold en esa variante: {winner[1]['chosen_c_per_fold']}")


if __name__ == "__main__":
    main()
