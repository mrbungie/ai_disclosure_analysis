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

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import GroupKFold

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "common"))
import layers as L  # noqa: E402

GOLDEN_DIR = REPO_ROOT / "data" / "interim" / "golden_set"

BASE_SIGNALS = [
    "score_ai_use", "score_ai_exploration", "score_ai_capability", "score_ai_outcome",
    "score_ai_risk", "score_ai_governance", "score_ai_strategy",
    "negative_similarity", "semantic_margin", "strong_lexical_match", "weak_lexical_match",
]
EXTRA_SIGNAL = "max_semantic_score"


SANITY_CASES = [
    ("0000051143-23-000032", "2", 913, "IBM watsonx"),
    ("0001564590-22-026876", "1", 33, "Microsoft AI-backed tools"),
    ("0001013237-24-000141", "1", 34, "FactSet AI Blueprint"),
]


def load() -> pd.DataFrame:
    """Golden set (un solo juez, sin error) unido por llave de instancia a
    `bronze.prefilter_scores`."""
    # Un solo juez. El golden set tiene etiquetas de gemini-3.8-flash y de
    # qwen3.7-flash sobre los MISMOS párrafos (el re-etiquetado dejó las
    # viejas en disco a propósito, para poder medir acuerdo). Sin este
    # filtro cada párrafo re-etiquetado entra DOS veces, con dos targets
    # posiblemente distintos, y el CV agrupado por filing ni siquiera los
    # separa. Ver scripts/verif/judge_agreement.py.
    keys = ["country_code", "form", "accession_number", "item_key", "paragraph_index"]
    labels = (pl.concat([pl.scan_parquet(f) for f in sorted(GOLDEN_DIR.glob(
                  "golden_set_labels__session=*__part=*.parquet"))], how="diagonal_relaxed")
              .filter(pl.col("error").is_null() & (pl.col("judge_model") == "qwen/qwen3.7-flash")))
    scores = L.scan("bronze.prefilter_scores")
    return (labels.select(*keys, "is_ai_disclosure", "inclusion_weight")
            .join(scores.select(*keys, *(BASE_SIGNALS + [EXTRA_SIGNAL])), on=keys)
            .sort(*keys)
            .select(["is_ai_disclosure", "inclusion_weight", "accession_number", "item_key", "paragraph_index", *(BASE_SIGNALS + [EXTRA_SIGNAL])])
            .collect().to_pandas())


def weighted_f1_at(y, proba, sample_weight, t):
    pred = (proba >= t).astype(int)
    return f1_score(y, pred, sample_weight=sample_weight, zero_division=0)


def run_variant(X, y, weights, groups, *, weight_fn, c_grid: list[float]) -> dict:
    """Nested CV: for each outer fold, grid-search C (if len(c_grid) > 1) and
    threshold using ONLY the training split. `weight_fn(raw_weights) ->
    fit_weights` transforms `inclusion_weight` before it's passed to
    `.fit()` (None = no sample_weight at all); evaluation always uses the
    RAW weights, since that's what makes a metric here actually mean
    "corpus-representative" — only the training weight is a design choice."""
    gkf = GroupKFold(n_splits=5)
    oof_pred = np.zeros(len(y), dtype=int)
    chosen_cs = []
    for train_idx, test_idx in gkf.split(X, y, groups):
        fit_w = weight_fn(weights[train_idx]) if weight_fn else None
        best_c, best_t, best_f1 = c_grid[0], 0.5, -1.0
        for c in c_grid:
            clf = LogisticRegression(max_iter=2000, C=c)
            clf.fit(X[train_idx], y[train_idx], sample_weight=fit_w)
            proba_train = clf.predict_proba(X[train_idx])[:, 1]
            for t in np.arange(0.05, 0.96, 0.02):
                f1w = weighted_f1_at(y[train_idx], proba_train, weights[train_idx], t)
                if f1w > best_f1:
                    best_c, best_t, best_f1 = c, float(t), f1w
        chosen_cs.append(best_c)
        clf = LogisticRegression(max_iter=2000, C=best_c)
        clf.fit(X[train_idx], y[train_idx], sample_weight=fit_w)
        proba_test = clf.predict_proba(X[test_idx])[:, 1]
        oof_pred[test_idx] = (proba_test >= best_t).astype(int)

    return {
        "f1_estrato": float(f1_score(y, oof_pred)),
        "prec_pond": float(precision_score(y, oof_pred, sample_weight=weights, zero_division=0)),
        "recall_pond": float(recall_score(y, oof_pred, sample_weight=weights, zero_division=0)),
        "f1_pond": float(f1_score(y, oof_pred, sample_weight=weights, zero_division=0)),
        "chosen_c_per_fold": chosen_cs,
    }


def sanity_check(df: pd.DataFrame, X: np.ndarray, y: np.ndarray, weights: np.ndarray,
                  *, weight_fn, c: float) -> dict:
    """Fits ONE final model on ALL golden-set data (same as deployment) and
    reports predicted_proba for 3 real, manually-verified substantive AI
    disclosures (see docs/prefilter_evaluation.md §8.6) — a metric can look
    better in aggregate while the model quietly stops recognizing exactly
    this kind of clear-cut case, which is what caught the sample_weight bug
    this script exists to fix. Never trust the aggregate number alone again."""
    fit_w = weight_fn(weights) if weight_fn else None
    clf = LogisticRegression(max_iter=2000, C=c)
    clf.fit(X, y, sample_weight=fit_w)
    out = {}
    for acc, item, pidx, label in SANITY_CASES:
        mask = ((df["accession_number"] == acc) & (df["item_key"] == item)
                & (df["paragraph_index"] == pidx))
        if not mask.any():
            out[label] = None
            continue
        row = df.loc[mask, BASE_SIGNALS].astype(float).values
        out[label] = float(clf.predict_proba(row)[0, 1])
    return out


def main() -> None:
    df = load()
    y = df["is_ai_disclosure"].astype(int).values
    weights = df["inclusion_weight"].astype(float).values
    groups = df["accession_number"].values
    X_base = df[BASE_SIGNALS].astype(float).values

    grid = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
    variants = {
        "0. baseline (deployado en §8.3: sin peso, C=1.0)":
            (None, run_variant(X_base, y, weights, groups, weight_fn=None, c_grid=[1.0])),
        "2. §8.5: fit CON inclusion_weight crudo + grid de C":
            (lambda w: w, run_variant(X_base, y, weights, groups, weight_fn=lambda w: w, c_grid=grid)),
        "4. fit con sqrt(inclusion_weight) + grid de C":
            (lambda w: np.sqrt(w), run_variant(X_base, y, weights, groups,
                                                weight_fn=lambda w: np.sqrt(w), c_grid=grid)),
        "5. fit con log1p(inclusion_weight) + grid de C":
            (lambda w: np.log1p(w), run_variant(X_base, y, weights, groups,
                                                 weight_fn=lambda w: np.log1p(w), c_grid=grid)),
        "6. fit con inclusion_weight recortado a percentil 95 + grid de C":
            (lambda w: np.clip(w, None, np.percentile(w, 95)),
             run_variant(X_base, y, weights, groups,
                         weight_fn=lambda w: np.clip(w, None, np.percentile(w, 95)), c_grid=grid)),
    }

    print(f"{'variante':<55} {'F1 estr.':>9} {'prec':>7} {'recall':>7} {'F1 pond.':>9}")
    for label, (_, m) in variants.items():
        print(f"{label:<55} {m['f1_estrato']:>9.3f} {m['prec_pond']:>7.3f} "
              f"{m['recall_pond']:>7.3f} {m['f1_pond']:>9.3f}")

    print(f"\n{'variante':<55} " + " | ".join(f"{c[3]:<22}" for c in SANITY_CASES))
    for label, (weight_fn, m) in variants.items():
        # C para el chequeo de sanidad: la moda de lo elegido por fold.
        cs = m["chosen_c_per_fold"]
        c_final = max(set(cs), key=cs.count)
        probs = sanity_check(df, X_base, y, weights, weight_fn=weight_fn, c=c_final)
        print(f"{label:<55} " + " | ".join(f"{probs[c[3]]:<22.3f}" for c in SANITY_CASES))

    winner = max(variants.items(), key=lambda kv: kv[1][1]["f1_pond"])
    print(f"\nGana por F1 ponderado: {winner[0]} ({winner[1][1]['f1_pond']:.3f})")


if __name__ == "__main__":
    main()
