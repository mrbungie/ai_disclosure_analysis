"""Does the deployed prefilter model (docs/prefilter_evaluation.md §8.2) lose
real positives it could catch? Verified: yes — it NEVER predicts positive
when strong_lexical_match=False (coefficient +5.31 dominates the linear
score), and 39/1,731 golden-set true positives (2.25%, ~0.46% of weighted
positive mass) fall in that group. This script tests two ways to rescue
some of them, evaluated with the SAME nested-CV discipline as the main
model (no threshold or rule is ever chosen using a fold it's then scored
on), and compares by weighted F1 against the baseline (main model alone).

Candidates:
  A. Rule: within strong_lexical_match=False rows, flag positive if
     weak_lexical_match AND semantic_margin >= a threshold (grid-searched).
  B. Small model: LogisticRegression on the semantic signals + weak_lexical_match,
     fit ONLY on strong_lexical_match=False training rows.
Both are combined (OR) with the main model's own out-of-fold decision, so
the reported metrics are for the COMBINED system, not the rescue branch in
isolation.

Usage:
    uv run python scripts/verif/prefilter_rescue_eval.py
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

MAIN_SIGNALS = [
    "score_ai_use", "score_ai_exploration", "score_ai_capability", "score_ai_outcome",
    "score_ai_risk", "score_ai_governance", "score_ai_strategy",
    "negative_similarity", "semantic_margin", "strong_lexical_match", "weak_lexical_match",
]
RESCUE_SIGNALS = [
    "score_ai_use", "score_ai_exploration", "score_ai_capability", "score_ai_outcome",
    "score_ai_risk", "score_ai_governance", "score_ai_strategy",
    "negative_similarity", "semantic_margin", "weak_lexical_match",
]


def load() -> pd.DataFrame:
    con = duckdb.connect(str(DB), read_only=True)
    try:
        return con.execute(f"""
            SELECT l.is_ai_disclosure, l.inclusion_weight, l.accession_number,
                   {', '.join(f'p.{c}' for c in set(MAIN_SIGNALS + RESCUE_SIGNALS))}
            FROM read_parquet('data/interim/golden_set/golden_set_labels__session=*__part=*.parquet',
                               union_by_name=True) l
            JOIN read_parquet('data/interim/prefilter_scores/prefilter_scores__run={LATEST_PREFILTER_RUN}__part=*.parquet') p
                USING (country_code, form, accession_number, item_key, paragraph_index)
            -- Un solo juez. El golden set tiene etiquetas de gemini-3.8-flash y de
            -- qwen3.7-flash sobre los MISMOS párrafos (el re-etiquetado dejó las
            -- viejas en disco a propósito, para poder medir acuerdo). Sin este
            -- filtro cada párrafo re-etiquetado entra DOS veces, con dos targets
            -- posiblemente distintos, y el CV agrupado por filing ni siquiera los
            -- separa. Ver scripts/verif/judge_agreement.py.
            WHERE l.error IS NULL AND l.judge_model = 'qwen/qwen3.7-flash'
        """).df()
    finally:
        con.close()


def weighted_f1_at(y, proba, weights, t):
    pred = (proba >= t).astype(int)
    return f1_score(y, pred, sample_weight=weights, zero_division=0)


def evaluate(oof_pred: np.ndarray, y: np.ndarray, weights: np.ndarray, label: str) -> dict:
    metrics = {
        "label": label,
        "f1_estrato": float(f1_score(y, oof_pred)),
        "prec_pond": float(precision_score(y, oof_pred, sample_weight=weights, zero_division=0)),
        "recall_pond": float(recall_score(y, oof_pred, sample_weight=weights, zero_division=0)),
        "f1_pond": float(f1_score(y, oof_pred, sample_weight=weights, zero_division=0)),
        "n_positive": int(oof_pred.sum()),
    }
    print(f"[{label}] F1 estrato={metrics['f1_estrato']:.3f} prec pond.={metrics['prec_pond']:.3f} "
          f"recall pond.={metrics['recall_pond']:.3f} F1 pond.={metrics['f1_pond']:.3f} "
          f"(n positivos oof={metrics['n_positive']})")
    return metrics


def main() -> None:
    df = load()
    y = df["is_ai_disclosure"].astype(int).values
    weights = df["inclusion_weight"].astype(float).values
    groups = df["accession_number"].values
    no_strong = ~df["strong_lexical_match"].astype(bool).values

    Xmain = df[MAIN_SIGNALS].astype(float).values
    Xrescue = df[RESCUE_SIGNALS].astype(float).values

    gkf = GroupKFold(n_splits=5)
    baseline_pred = np.zeros(len(y), dtype=int)
    ruleA_pred = np.zeros(len(y), dtype=int)
    modelB_pred = np.zeros(len(y), dtype=int)

    for train_idx, test_idx in gkf.split(Xmain, y, groups):
        # --- main model, same as ai_prefilter_classify.py ---
        main_clf = LogisticRegression(max_iter=2000)
        main_clf.fit(Xmain[train_idx], y[train_idx])
        main_proba_train = main_clf.predict_proba(Xmain[train_idx])[:, 1]
        main_proba_test = main_clf.predict_proba(Xmain[test_idx])[:, 1]

        # threshold chosen on TRAIN fold's own predictions only
        best_t, best_f1 = 0.5, -1.0
        for t in np.arange(0.05, 0.96, 0.01):
            f1w = weighted_f1_at(y[train_idx], main_proba_train, weights[train_idx], t)
            if f1w > best_f1:
                best_t, best_f1 = float(t), f1w
        main_test_pred = (main_proba_test >= best_t).astype(int)
        baseline_pred[test_idx] = main_test_pred
        ruleA_pred[test_idx] = main_test_pred
        modelB_pred[test_idx] = main_test_pred

        # --- candidate A: threshold on semantic_margin, no-strong-lexical rows only ---
        train_ns = train_idx[no_strong[train_idx]]
        test_ns = test_idx[no_strong[test_idx]]
        if len(train_ns) and y[train_ns].sum() > 0:
            margin_train = df["semantic_margin"].values[train_ns]
            weak_train = df["weak_lexical_match"].values[train_ns].astype(bool)
            best_tm, best_f1m = None, -1.0
            for tm in np.linspace(margin_train.min(), margin_train.max(), 50):
                pred = (weak_train & (margin_train >= tm)).astype(int)
                f1w = weighted_f1_at(y[train_ns], pred, weights[train_ns], 0.5)
                if f1w > best_f1m:
                    best_tm, best_f1m = float(tm), f1w
            margin_test = df["semantic_margin"].values[test_ns]
            weak_test = df["weak_lexical_match"].values[test_ns].astype(bool)
            rescue_pred = (weak_test & (margin_test >= best_tm)).astype(int)
            ruleA_pred[test_ns] = np.maximum(ruleA_pred[test_ns], rescue_pred)

        # --- candidate B: small logit on no-strong-lexical rows only ---
        if len(train_ns) and y[train_ns].sum() >= 5:
            rescue_clf = LogisticRegression(max_iter=2000)
            rescue_clf.fit(Xrescue[train_ns], y[train_ns])
            proba_train_ns = rescue_clf.predict_proba(Xrescue[train_ns])[:, 1]
            best_tb, best_f1b = 0.5, -1.0
            for t in np.arange(0.05, 0.96, 0.01):
                f1w = weighted_f1_at(y[train_ns], proba_train_ns, weights[train_ns], t)
                if f1w > best_f1b:
                    best_tb, best_f1b = float(t), f1w
            proba_test_ns = rescue_clf.predict_proba(Xrescue[test_ns])[:, 1]
            rescue_pred_b = (proba_test_ns >= best_tb).astype(int)
            modelB_pred[test_ns] = np.maximum(modelB_pred[test_ns], rescue_pred_b)

    print()
    results = [
        evaluate(baseline_pred, y, weights, "baseline (solo modelo principal)"),
        evaluate(ruleA_pred, y, weights, "A: + regla (weak_lexical Y margen>=t) en no-lexico-fuerte"),
        evaluate(modelB_pred, y, weights, "B: + modelo chico en no-lexico-fuerte"),
    ]
    winner = max(results, key=lambda r: r["f1_pond"])
    print(f"\nGana por F1 ponderado: {winner['label']} ({winner['f1_pond']:.3f})")


if __name__ == "__main__":
    main()
