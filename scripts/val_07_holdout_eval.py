"""
val_07_holdout_eval.py — Apply the frozen boolean formulas (val_06's output)
to the disjoint 350-chunk holdout (val_05's labels) ONCE and lock the report.

HARD RULE, same as the prefilter holdout (val_11): if
reports/boolean_search_holdout_eval__rule_based.txt already exists, this
script refuses to run again. A disappointing result is reported as-is; further
formula changes require a fresh, never-seen third batch — not re-running this
script or re-examining this holdout.

Usage:
    uv run python scripts/val_07_holdout_eval.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from val_06_boolean_search_harness import production_formula, build_meta_atoms, DIMENSIONS
except ImportError:
    from scripts.val_06_boolean_search_harness import production_formula, build_meta_atoms, DIMENSIONS

HOLDOUT_LABELED_PATH = Path("data/processed/variant_rule_based/validation/holdout_labeled__rule_based.parquet")
BOW_PATH = Path("data/interim/candidate_chunks/ai_disclosure_bow_features.parquet")
FROZEN_FORMULAS_PATH = Path("data/interim/validation/frozen_boolean_formulas__rule_based.json")
REPORT_PATH = Path("reports/boolean_search_holdout_eval__rule_based.txt")


def apply_spec(df: pd.DataFrame, spec: dict, dimension: str) -> np.ndarray:
    """Re-apply a frozen candidate spec (from val_06) to any dataframe with the
    same BoW feature columns — the holdout here, but the same function works
    on any future re-scoring pass."""
    op = spec["op"]
    if op == "production":
        return production_formula(df, dimension).to_numpy()
    meta_atoms = build_meta_atoms(df)

    def get(name: str) -> pd.Series:
        return meta_atoms[name] if name in meta_atoms else df[name].astype(bool)

    if op == "single":
        return get(spec["feature"]).to_numpy()
    if op == "not":
        return (~get(spec["feature"])).to_numpy()
    if op == "and":
        a, b = spec["features"]
        return (get(a) & get(b)).to_numpy()
    if op == "or":
        a, b = spec["features"]
        return (get(a) | get(b)).to_numpy()
    if op == "and_not":
        a, b = spec["features"]
        return (get(a) & ~get(b)).to_numpy()
    raise ValueError(f"Unknown spec op: {op}")


def f1_precision_recall(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    tp = int(np.sum(y_true & y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return f1, precision, recall


def main() -> None:
    if REPORT_PATH.exists():
        print(f"Error: {REPORT_PATH} already exists. This holdout has been evaluated ONCE — "
              f"per the single-look rule, this script refuses to run again. If the result needs "
              f"revisiting, sample and judge a fresh, never-seen third batch instead.")
        return
    if not HOLDOUT_LABELED_PATH.exists():
        print(f"Error: {HOLDOUT_LABELED_PATH} not found. Run val_05_label_holdout.py first.")
        return
    if not FROZEN_FORMULAS_PATH.exists():
        print(f"Error: {FROZEN_FORMULAS_PATH} not found. Run val_06_boolean_search_harness.py first.")
        return

    holdout = pd.read_parquet(HOLDOUT_LABELED_PATH)
    bow = pd.read_parquet(BOW_PATH).drop_duplicates(subset="chunk_id")
    df = holdout.merge(bow, on="chunk_id", how="inner")
    dropped = len(holdout) - len(df)
    if dropped:
        print(f"Dropped {dropped} holdout chunk_id(s) not present in current BoW features.")

    with open(FROZEN_FORMULAS_PATH) as f:
        frozen = json.load(f)

    lines = []
    lines.append("Boolean search — LOCKED HOLDOUT evaluation (single look)")
    lines.append("=" * 78)
    lines.append(f"Holdout size: {len(df)} chunks (never used in val_06's search)")
    lines.append("")

    for dimension in DIMENSIONS:
        y = df[f"llm_{dimension}"].astype(bool).to_numpy()
        winner = frozen[dimension]
        pred_winner = apply_spec(df, winner["spec"], dimension)
        pred_prod = production_formula(df, dimension).to_numpy()

        f1_w, p_w, r_w = f1_precision_recall(y, pred_winner)
        f1_p, p_p, r_p = f1_precision_recall(y, pred_prod)

        lines.append(f"Dimension: {dimension}  (holdout prevalence {y.mean()*100:.1f}%, n={len(y)})")
        lines.append(f"  Frozen winner: {winner['name']}")
        lines.append(f"    F1={f1_w:.3f}  P={p_w:.3f}  R={r_w:.3f}")
        lines.append(f"  Production formula (script 09, for comparison):")
        lines.append(f"    F1={f1_p:.3f}  P={p_p:.3f}  R={r_p:.3f}")
        lines.append("")

    lines.append("Rule: if any of these numbers disappoint, do not re-run val_06 against this holdout")
    lines.append("or re-examine it further. Report as-is. Improving a dimension requires sampling and")
    lines.append("judging a fresh, never-seen batch, then re-running the search harness against a search")
    lines.append("pool that includes it — not touching this holdout again.")

    report = "\n".join(lines)
    print(report)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report + "\n")
    print(f"\nLocked report written -> {REPORT_PATH}")


if __name__ == "__main__":
    main()
