"""Re-fits docs/prefilter_evaluation.md §8's logistic regression ("logit sobre
las 11 señales del prefiltro") now that the golden set is complete (9,900
labels, not the 6,038 it was fit on when that doc was written — see §9 point 1,
now resolved). No re-embedding, no new scoring: reuses the already-computed
prefilter_scores (latest anchors run) as-is. Same 11 signals, same GroupKFold-
by-filing methodology, same weighted metrics — only the label count changed.

Usage:
    uv run python scripts/verif/prefilter_logit_refit.py
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
LATEST_PREFILTER_RUN = "20260902T225928Z"  # newest anchors_fingerprint, per manifest

SIGNAL_COLUMNS = [
    "score_ai_use", "score_ai_exploration", "score_ai_capability", "score_ai_outcome",
    "score_ai_risk", "score_ai_governance", "score_ai_strategy",
    "negative_similarity", "semantic_margin", "strong_lexical_match", "weak_lexical_match",
]


def load() -> pd.DataFrame:
    con = duckdb.connect(str(DB), read_only=True)
    try:
        df = con.execute(f"""
            SELECT l.is_ai_disclosure, l.inclusion_weight, l.accession_number,
                   {', '.join(f'p.{c}' for c in SIGNAL_COLUMNS)}
            FROM read_parquet('data/interim/golden_set/golden_set_labels__session=*__part=*.parquet',
                               union_by_name=True) l
            JOIN read_parquet('data/interim/prefilter_scores/prefilter_scores__run={LATEST_PREFILTER_RUN}__part=*.parquet') p
                USING (country_code, form, accession_number, item_key, paragraph_index)
            WHERE l.error IS NULL
        """).df()
    finally:
        con.close()
    return df


def main() -> None:
    df = load()
    print(f"{len(df):,} etiquetas (vs 6,038 cuando se escribió prefilter_evaluation.md §8)")

    y = df["is_ai_disclosure"].astype(int).values
    weights = df["inclusion_weight"].astype(float).values
    groups = df["accession_number"].values
    X = df[SIGNAL_COLUMNS].astype(float).values

    gkf = GroupKFold(n_splits=5)
    all_pred, all_true, all_w = [], [], []
    for train_idx, test_idx in gkf.split(X, y, groups):
        # class_weight=None (sklearn default), NOT "balanced": verificado que
        # "balanced" bajo el peso real del corpus (muy sesgado hacia negativos)
        # empuja el modelo a colapsar en una copia casi exacta de
        # strong_lexical_match solo (F1 pond. idéntico a 3 decimales a la regla
        # léxica pura) — pierde toda la señal semántica en vez de sumarla.
        clf = LogisticRegression(max_iter=2000)
        clf.fit(X[train_idx], y[train_idx])
        all_pred.extend(clf.predict(X[test_idx]))
        all_true.extend(y[test_idx])
        all_w.extend(weights[test_idx])
    all_pred, all_true, all_w = np.array(all_pred), np.array(all_true), np.array(all_w)

    f1_strat = f1_score(all_true, all_pred)
    precw = precision_score(all_true, all_pred, sample_weight=all_w)
    recw = recall_score(all_true, all_pred, sample_weight=all_w)
    f1w = f1_score(all_true, all_pred, sample_weight=all_w)

    print(f"\nlogit sobre las 11 señales, {len(df):,} etiquetas (golden set completo):")
    print(f"  F1 estrato   = {f1_strat:.3f}")
    print(f"  prec pond.   = {precw:.3f}")
    print(f"  recall pond. = {recw:.3f}")
    print(f"  F1 pond.     = {f1w:.3f}")
    print(f"\nversus lo documentado en docs/prefilter_evaluation.md §8 (6,038 etiquetas):")
    print(f"  F1 estrato = 0.736 | prec pond. = 0.717 | recall pond. = 0.802 | F1 pond. = 0.757")


if __name__ == "__main__":
    main()
