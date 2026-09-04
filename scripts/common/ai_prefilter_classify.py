"""Trains the final AI-relevance classifier (logistic regression on the
prefilter's 11 signals — see docs/prefilter_evaluation.md §8/§8.1) and
applies it to the FULL scored corpus to produce the AI-candidate set that
scripts/common/ai_classify.py's frame extraction should actually run over.

Methodology (see §8.2 in the doc for the full writeup):
1. GroupKFold(5) by accession_number over the complete golden set (9,884
   labels) — same as §8.1's refit, but this time the out-of-fold predicted
   PROBABILITIES are kept (not just the default 0.5-cutoff predictions), so
   a decision threshold can be chosen properly instead of assumed.
2. Threshold: the value (scanned in 0.01 steps) that maximizes weighted F1
   (`inclusion_weight`) on the pooled out-of-fold probabilities. This is
   the honest, unbiased performance estimate — every prediction used to
   pick both the model's weights AND the threshold came from a fold that
   never saw that row during training.
3. Only THEN: refit on all 9,884 labeled rows (standard practice — CV is
   for validating/tuning, the deployed model uses every label it has) and
   apply that final model + threshold to the full 3,281,038-paragraph
   corpus (latest anchors run only — a different anchors_fingerprint is a
   different, incompatible score population, see ai_prefilter.py's own
   anchors_fingerprint() docstring).

Output: data/interim/prefilter_predictions/prefilter_predictions__run=<id>.parquet
(one row per paragraph, every paragraph — not just the positives — so a
downstream reader never has to treat "absent" as "negative" by assumption)
plus a manifest JSON with the CV metrics, threshold, coefficients (plain
JSON, not a pickle — this is an 11-feature linear model, no reason to carry
a full sklearn object with its version-pinning risk), and the funnel counts
at each stage (total corpus -> lexical gate -> this model's positives).

Usage:
    uv run python scripts/common/ai_prefilter_classify.py
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import GroupKFold

REPO_ROOT = Path(__file__).resolve().parents[2]
DB = REPO_ROOT / "duckdb" / "thesis.duckdb"
LATEST_PREFILTER_RUN = "20260902T225928Z"  # newest anchors_fingerprint, per manifest
OUT_DIR = REPO_ROOT / "data" / "interim" / "prefilter_predictions"

SIGNAL_COLUMNS = [
    "score_ai_use", "score_ai_exploration", "score_ai_capability", "score_ai_outcome",
    "score_ai_risk", "score_ai_governance", "score_ai_strategy",
    "negative_similarity", "semantic_margin", "strong_lexical_match", "weak_lexical_match",
]
PARAGRAPH_KEY = ("country_code", "form", "accession_number", "item_key", "paragraph_index")


def load_golden() -> pd.DataFrame:
    con = duckdb.connect(str(DB), read_only=True)
    try:
        return con.execute(f"""
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


def cv_threshold_and_metrics(X: np.ndarray, y: np.ndarray, weights: np.ndarray,
                              groups: np.ndarray) -> tuple[float, dict]:
    gkf = GroupKFold(n_splits=5)
    oof_proba = np.zeros(len(y))
    for train_idx, test_idx in gkf.split(X, y, groups):
        clf = LogisticRegression(max_iter=2000)
        clf.fit(X[train_idx], y[train_idx])
        oof_proba[test_idx] = clf.predict_proba(X[test_idx])[:, 1]

    best_threshold, best_f1w = 0.5, -1.0
    for t in np.arange(0.05, 0.96, 0.01):
        pred = (oof_proba >= t).astype(int)
        f1w = f1_score(y, pred, sample_weight=weights, zero_division=0)
        if f1w > best_f1w:
            best_threshold, best_f1w = float(t), f1w

    pred = (oof_proba >= best_threshold).astype(int)
    metrics = {
        "threshold": best_threshold,
        "f1_estrato": float(f1_score(y, pred)),
        "prec_pond": float(precision_score(y, pred, sample_weight=weights)),
        "recall_pond": float(recall_score(y, pred, sample_weight=weights)),
        "f1_pond": float(f1_score(y, pred, sample_weight=weights)),
    }
    return best_threshold, metrics


def funnel_counts(con) -> dict:
    total = con.execute("SELECT COUNT(*) FROM paragraphs").fetchone()[0]
    lexical = con.execute(f"""
        SELECT COUNT(*) FROM read_parquet(
            'data/interim/prefilter_scores/prefilter_scores__run={LATEST_PREFILTER_RUN}__part=*.parquet')
        WHERE strong_lexical_match OR weak_lexical_match
    """).fetchone()[0]
    strong_only = con.execute(f"""
        SELECT COUNT(*) FROM read_parquet(
            'data/interim/prefilter_scores/prefilter_scores__run={LATEST_PREFILTER_RUN}__part=*.parquet')
        WHERE strong_lexical_match
    """).fetchone()[0]
    return {"total_paragraphs": total, "lexical_strong_or_weak": lexical, "lexical_strong_only": strong_only}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Cargando golden set completo...")
    golden = load_golden()
    print(f"{len(golden):,} etiquetas")

    y = golden["is_ai_disclosure"].astype(int).values
    weights = golden["inclusion_weight"].astype(float).values
    groups = golden["accession_number"].values
    X = golden[SIGNAL_COLUMNS].astype(float).values

    print("GroupKFold(5) para elegir threshold y estimar performance out-of-fold...")
    threshold, cv_metrics = cv_threshold_and_metrics(X, y, weights, groups)
    print(f"Threshold elegido: {threshold:.2f}")
    print(f"CV: F1 estrato={cv_metrics['f1_estrato']:.3f} prec pond.={cv_metrics['prec_pond']:.3f} "
          f"recall pond.={cv_metrics['recall_pond']:.3f} F1 pond.={cv_metrics['f1_pond']:.3f}")

    print("Reajustando el modelo final sobre TODO el golden set...")
    final_model = LogisticRegression(max_iter=2000)
    final_model.fit(X, y)

    con = duckdb.connect(str(DB), read_only=True)
    print("Cargando el corpus completo (última corrida de anchors)...")
    corpus = con.execute(f"""
        SELECT {', '.join(PARAGRAPH_KEY)}, {', '.join(SIGNAL_COLUMNS)}
        FROM read_parquet('data/interim/prefilter_scores/prefilter_scores__run={LATEST_PREFILTER_RUN}__part=*.parquet')
    """).df()
    print(f"{len(corpus):,} párrafos")

    print("Aplicando el modelo final a todo el corpus...")
    Xc = corpus[SIGNAL_COLUMNS].astype(float).values
    proba = final_model.predict_proba(Xc)[:, 1]
    is_positive = proba >= threshold
    print(f"Marcados como IA-relevantes: {int(is_positive.sum()):,} / {len(corpus):,} "
          f"({100 * is_positive.mean():.2f}%)")

    print("Calculando el funnel...")
    funnel = funnel_counts(con)
    funnel["prefilter_model_positive"] = int(is_positive.sum())
    con.close()

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = corpus[list(PARAGRAPH_KEY)].copy()
    out["predicted_proba"] = proba.astype("float32")
    out["is_ai_prefiltered"] = is_positive
    out["threshold"] = threshold
    out["model_version"] = run_id
    out["anchors_run"] = LATEST_PREFILTER_RUN

    out_path = OUT_DIR / f"prefilter_predictions__run={run_id}.parquet"
    pq.write_table(pa.Table.from_pandas(out, preserve_index=False), out_path, compression="zstd")

    manifest = {
        "run_id": run_id, "anchors_run": LATEST_PREFILTER_RUN,
        "golden_set_labels": int(len(golden)),
        "threshold": threshold, "cv_metrics": cv_metrics,
        "coefficients": dict(zip(SIGNAL_COLUMNS, final_model.coef_[0].tolist())),
        "intercept": float(final_model.intercept_[0]),
        "funnel": funnel,
        "output": str(out_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = OUT_DIR / f"prefilter_predictions_manifest__run={run_id}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    print(f"\nParquet -> {out_path}")
    print(f"Manifiesto -> {manifest_path}")
    print(f"\nFunnel:")
    for stage, count in funnel.items():
        print(f"  {stage}: {count:,}")


if __name__ == "__main__":
    main()
