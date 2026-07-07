"""
val_06_boolean_search_harness.py — CV boolean/threshold search for the 4 fine
dimensions (is_substantive, is_promotional, is_risk_related, is_governance_related)
over the SEARCH POOL ONLY (236 original + 130 top-up = up to 366 chunks).

This is a harness, not model training: for each dimension it evaluates a bounded
set of candidate boolean formulas built from existing BoW feature columns via
k-fold stratified CV, selecting the formula with the best mean F1 across folds
while penalizing high variance (a formula that looks great on one fold and bad
on another is fitting that fold's noise, not generalizing).

The current production formulas (build_proxy_sql() in script 09) are always
included as a baseline candidate, so the harness either confirms them or
justifies replacing them with something that generalizes better within the
search pool.

Output: reports/boolean_search_harness__rule_based.txt (frozen formula per
dimension, its CV mean/std F1, the production baseline's CV mean/std F1, and
the total number of candidates evaluated — for the thesis limitations section,
to make the amount of "search until it works" explicit).

Does NOT touch the holdout. val_07 evaluates the frozen formulas there, once.

Usage:
    uv run python scripts/val_06_boolean_search_harness.py [--folds 5] [--seed 42]
"""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

LABELED_SAMPLE_PATH = Path("data/processed/variant_rule_based/validation/llm_labeled_sample__rule_based.parquet")
TOPUP_LABELED_PATH = Path("data/processed/variant_rule_based/validation/search_pool_topup_labeled__rule_based.parquet")
BOW_PATH = Path("data/interim/candidate_chunks/ai_disclosure_bow_features.parquet")
FROZEN_FORMULAS_PATH = Path("data/interim/validation/frozen_boolean_formulas__rule_based.json")
REPORT_PATH = Path("reports/boolean_search_harness__rule_based.txt")

DIMENSIONS = ["is_substantive", "is_promotional", "is_risk_related", "is_governance_related"]

# Feature columns relevant to each dimension's search space — narrowed from the
# full ~335 BoW columns to the ones plausibly related to that dimension's
# semantics, so "top-K single-atom" pre-filtering isn't just picking up noise
# correlated by chance in a ~360-row sample.
DIMENSION_FEATURE_POOLS = {
    "is_substantive": [
        "has_model_training", "has_workforce_talent", "has_ai_hedge", "has_specific_product",
        "has_proprietary_data", "has_compute_infra", "has_word_ai", "has_word_gpu", "has_word_gpus",
        "has_realized_language", "has_deployment_verb", "has_ai_demand_context", "has_ai_quantified_claim",
        "has_ethics_policy", "has_risk_factor", "has_classical_ml_mention", "has_internal_productivity",
        "has_customer_facing", "has_ai_own_use", "has_competitor_mention", "has_classical_ml_operational",
        "has_ai_acquisition", "has_named_deployment", "has_dated_milestone", "has_ai_use_case_specific",
        "has_product_integration", "has_academic_research", "has_open_source",
    ],
    "is_promotional": [
        "has_word_revolutionize", "has_word_revolutionizing", "has_word_revolutionized", "has_word_transform",
        "has_word_transforming", "has_word_transformative", "has_word_cutting_edge", "has_word_next_generation",
        "has_word_leader", "has_word_leadership", "has_word_empower", "has_word_empowering",
        "has_risk_factor", "has_deployment_verb", "has_specific_product", "has_competitor_ai_mention",
    ],
    "is_risk_related": [
        "has_risk_factor", "has_regulatory_risk", "has_ethics_bias_risk", "has_supply_infra_risk",
        "has_data_licensing", "has_cyber_privacy_risk", "has_word_ai", "has_ip_copyright_risk",
        "has_labor_displacement_risk", "has_safety_critical", "has_competitor_mention", "has_ai_hedge",
    ],
    "is_governance_related": [
        "has_board_oversight", "has_audit_committee", "has_ethics_policy", "has_compliance", "has_word_ai",
        "has_us_regulation", "has_eu_regulation", "has_risk_factor",
    ],
}


def load_search_pool() -> pd.DataFrame:
    labeled = pd.read_parquet(LABELED_SAMPLE_PATH)
    parts = [labeled]
    if TOPUP_LABELED_PATH.exists():
        topup = pd.read_parquet(TOPUP_LABELED_PATH)
        parts.append(topup)
    else:
        print(f"Warning: {TOPUP_LABELED_PATH} not found — searching on the 236-chunk sample only. "
              f"Run val_04_label_search_topup.py first for the full search pool.")

    pool = pd.concat(parts, ignore_index=True)
    pool = pool.drop_duplicates(subset="chunk_id")

    bow = pd.read_parquet(BOW_PATH)
    bow = bow.drop_duplicates(subset="chunk_id")

    merged = pool.merge(bow, on="chunk_id", how="inner")
    dropped = len(pool) - len(merged)
    if dropped:
        print(f"Dropped {dropped} search-pool chunk_id(s) not present in current BoW features "
              f"(orphaned by the aip/aiops re-chunk).")
    return merged


def build_meta_atoms(df: pd.DataFrame) -> dict[str, pd.Series]:
    """A few derived (non-has_*) boolean atoms mirroring the ones already used
    in production's build_proxy_sql(), so the search space includes the
    formulas actually shipped, not just raw BoW presence flags."""
    return {
        "vague_gt_negative": df["count_vague_words"] > df["count_negative_words"],
        "sentiment_nonneg": df["bow_sentiment_score"] >= 0,
        "has_metrics": df["has_metric_percentage"] | df["has_metric_dollar"],
    }


def production_formula(df: pd.DataFrame, dimension: str) -> pd.Series:
    """Python re-implementation of build_proxy_sql() in script 09, for the
    dimension being searched — included as the baseline candidate every
    harness run is measured against."""
    if dimension == "is_substantive":
        return (
            (df["has_model_training"] & ~df["has_workforce_talent"] & ~df["has_ai_hedge"])
            | df["has_specific_product"]
            | df["has_proprietary_data"]
            | (df["has_compute_infra"] & df["has_word_ai"]
               & (df["has_word_gpu"] | df["has_word_gpus"] | df["has_realized_language"] | df["has_deployment_verb"])
               & ~df["has_ai_demand_context"])
            | (df["has_compute_infra"] & df["has_word_ai"] & df["has_ai_quantified_claim"] & ~df["has_ai_demand_context"])
            | (df["has_ethics_policy"] & df["has_word_ai"] & ~df["has_risk_factor"])
            | (df["has_classical_ml_mention"] & (df["has_internal_productivity"] | df["has_customer_facing"]) & ~df["has_workforce_talent"])
            | (df["has_ai_own_use"] & ~df["has_ai_hedge"] & ~df["has_competitor_mention"] & ~df["has_ai_demand_context"])
            | (df["has_classical_ml_operational"] & ~df["has_risk_factor"])
            | df["has_ai_acquisition"]
        )
    if dimension == "is_governance_related":
        return (
            df["has_board_oversight"] | df["has_audit_committee"]
            | (df["has_ethics_policy"] & df["has_word_ai"])
            | (df["has_compliance"] & df["has_word_ai"])
        )
    if dimension == "is_risk_related":
        return (
            df["has_risk_factor"] | df["has_regulatory_risk"] | df["has_ethics_bias_risk"]
            | df["has_supply_infra_risk"] | df["has_data_licensing"]
            | (df["has_cyber_privacy_risk"] & df["has_word_ai"])
            | (df["has_ip_copyright_risk"] & df["has_word_ai"])
        )
    if dimension == "is_promotional":
        meta = build_meta_atoms(df)
        return meta["vague_gt_negative"] & ~df["has_risk_factor"] & meta["sentiment_nonneg"]
    raise ValueError(dimension)


def f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = int(np.sum(y_true & y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def cv_score(y_true: np.ndarray, y_pred: np.ndarray, folds: list[tuple[np.ndarray, np.ndarray]]) -> tuple[float, float]:
    """Mean and std F1 across the SAME folds for every candidate (not re-split
    per candidate), so scores are comparable to each other."""
    scores = []
    for _, test_idx in folds:
        scores.append(f1(y_true[test_idx], y_pred[test_idx]))
    return float(np.mean(scores)), float(np.std(scores))


def search_dimension(df: pd.DataFrame, dimension: str, folds: list[tuple[np.ndarray, np.ndarray]], variance_penalty: float) -> dict:
    y = df[f"llm_{dimension}"].astype(bool).to_numpy()
    feature_pool = DIMENSION_FEATURE_POOLS[dimension]
    meta_atoms = build_meta_atoms(df)

    atoms: dict[str, pd.Series] = {name: df[name].astype(bool) for name in feature_pool if name in df.columns}
    if dimension == "is_promotional":
        atoms.update(meta_atoms)

    # Drop degenerate atoms (constant True/False in this search pool) BEFORE
    # building candidates. A constant atom's negation is a constant predictor —
    # e.g. has_ai_demand_context at 0% prevalence made "NOT has_ai_demand_context"
    # equivalent to "always predict True", which trivially scores F1=prevalence-
    # driven 0.73 on a majority-positive dimension by exploiting class imbalance,
    # not by encoding any real pattern. Caught by comparing the first harness run
    # against the majority-class baseline below — this exact case is why that
    # baseline is now always reported alongside the winner.
    degenerate = [name for name, s in atoms.items() if s.nunique() < 2]
    for name in degenerate:
        del atoms[name]

    # candidates: name -> (spec, prediction array). spec is JSON-serializable and
    # re-appliable to any dataframe with the same columns via apply_spec() —
    # val_07 uses it on the holdout, so the frozen winner isn't just a label.
    candidates: dict[str, tuple[dict, np.ndarray]] = {}
    candidates[f"PRODUCTION[{dimension}]"] = ({"op": "production", "dimension": dimension}, production_formula(df, dimension).to_numpy())

    for name, series in atoms.items():
        candidates[name] = ({"op": "single", "feature": name}, series.to_numpy())
        candidates[f"NOT {name}"] = ({"op": "not", "feature": name}, (~series).to_numpy())

    n_singles = len(candidates)

    single_scores = {name: cv_score(y, pred, folds)[0] for name, (_, pred) in candidates.items()}
    top_atoms = sorted(atoms.items(), key=lambda kv: single_scores.get(kv[0], 0.0), reverse=True)[:15]

    for (name_a, a), (name_b, b) in itertools.combinations(top_atoms, 2):
        candidates[f"{name_a} AND {name_b}"] = ({"op": "and", "features": [name_a, name_b]}, (a & b).to_numpy())
        candidates[f"{name_a} OR {name_b}"] = ({"op": "or", "features": [name_a, name_b]}, (a | b).to_numpy())
        candidates[f"{name_a} AND NOT {name_b}"] = ({"op": "and_not", "features": [name_a, name_b]}, (a & ~b).to_numpy())

    n_total = len(candidates)

    results = []
    for name, (spec, pred) in candidates.items():
        mean_f1, std_f1 = cv_score(y, pred, folds)
        penalized = mean_f1 - variance_penalty * std_f1
        results.append({"name": name, "spec": spec, "mean_f1": mean_f1, "std_f1": std_f1, "penalized": penalized})

    results.sort(key=lambda r: r["penalized"], reverse=True)
    prod_mean, prod_std = cv_score(y, candidates[f"PRODUCTION[{dimension}]"][1], folds)
    always_true_mean, always_true_std = cv_score(y, np.ones_like(y, dtype=bool), folds)

    return {
        "dimension": dimension,
        "n_candidates_singles": n_singles,
        "n_dropped_degenerate": len(degenerate),
        "n_candidates_total": n_total,
        "top10": results[:10],
        "winner": results[0],
        "production": {"mean_f1": prod_mean, "std_f1": prod_std},
        "always_true_baseline": {"mean_f1": always_true_mean, "std_f1": always_true_std},
        "prevalence": float(y.mean()),
        "n": len(y),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--variance-penalty", type=float, default=1.0,
                         help="Subtract variance_penalty * std_f1 from mean_f1 when ranking — discourages picking a formula that only wins on one lucky fold")
    args = parser.parse_args()

    df = load_search_pool()
    print(f"Search pool: {len(df)} chunks (236 original + top-up, deduped, BoW-joined).")

    frozen = {}
    all_results = {}
    lines = []
    lines.append("Boolean/threshold search harness — SEARCH POOL ONLY (not the final number)")
    lines.append("=" * 78)
    lines.append(f"Search pool size: {len(df)} chunks")
    lines.append(f"CV folds: {args.folds} (stratified per dimension), variance penalty: {args.variance_penalty}")
    lines.append("")

    for dimension in DIMENSIONS:
        y = df[f"llm_{dimension}"].astype(bool).to_numpy()
        skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        folds = list(skf.split(np.zeros(len(y)), y))

        result = search_dimension(df, dimension, folds, args.variance_penalty)
        all_results[dimension] = result
        frozen[dimension] = {"name": result["winner"]["name"], "spec": result["winner"]["spec"]}

        lines.append(f"Dimension: {dimension}  (prevalence {result['prevalence']*100:.1f}%, n={result['n']})")
        lines.append(f"  Candidates evaluated: {result['n_candidates_singles']} singles "
                      f"({result['n_dropped_degenerate']} dropped as degenerate/constant) + "
                      f"{result['n_candidates_total'] - result['n_candidates_singles']} pairs "
                      f"= {result['n_candidates_total']} total")
        lines.append(f"  Always-predict-True baseline: mean F1={result['always_true_baseline']['mean_f1']:.3f} "
                      f"(exploits prevalence alone — any winner must clear this to mean anything)")
        lines.append(f"  Production formula CV: mean F1={result['production']['mean_f1']:.3f}, "
                      f"std={result['production']['std_f1']:.3f}")
        lines.append(f"  Winner: {result['winner']['name']}")
        lines.append(f"    mean F1={result['winner']['mean_f1']:.3f}, std={result['winner']['std_f1']:.3f}, "
                      f"penalized score={result['winner']['penalized']:.3f}")
        lines.append(f"  Top 10 candidates (by penalized score):")
        for r in result["top10"]:
            lines.append(f"    {r['penalized']:.3f}  (mean={r['mean_f1']:.3f}, std={r['std_f1']:.3f})  {r['name']}")
        lines.append("")

    total_evaluated = sum(r["n_candidates_total"] for r in all_results.values())
    lines.append(f"Total candidate formulas evaluated across all 4 dimensions: {total_evaluated}")
    lines.append("")
    lines.append("Frozen formulas (winner per dimension, by CV mean F1 minus variance penalty):")
    for dim, winner in frozen.items():
        lines.append(f"  {dim}: {winner['name']}")
    lines.append("")
    lines.append("These are frozen now. val_07_holdout_eval.py evaluates them on the disjoint holdout ONCE.")
    lines.append("If the winner is just PRODUCTION[dimension], the existing script 09 formula survives CV")
    lines.append("and no code change is needed for that dimension.")

    report = "\n".join(lines)
    print(report)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report + "\n")

    FROZEN_FORMULAS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FROZEN_FORMULAS_PATH, "w") as f:
        json.dump(frozen, f, indent=2)
    print(f"\nReport -> {REPORT_PATH}")
    print(f"Frozen formulas -> {FROZEN_FORMULAS_PATH}")


if __name__ == "__main__":
    main()
