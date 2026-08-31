"""
11_fit_tag_harness.py — Cycle 2, the fit: split the LLM-judged chunk sample
(10) into dev/holdout, search boolean keyword formulas per dimension against
dev with a fold-stability-penalized score, then evaluate the frozen formulas
ONCE on the disjoint holdout and write the winners into configs/config.json.

    10 (sample + LLM judge)  ->  11 (this script)  ->  configs/config.json
                                                        (tagging.formulas)
                                                    ->  12 (tag the corpus)

Second instantiation of the shared harness-optimization cycle
(scripts/harness_fit.py, docs/meta_harness_plan.md). Same discipline as 08:
dev is spent, the fold machinery is a stability check (not CV), the holdout
is touched exactly once and then locked, metrics are reported raw and
population-weighted, the trace records every candidate evaluated, and config
is only written for dimensions whose winner is actually trustworthy on the
holdout (beats the always-True prevalence exploit; the legacy production
formula, where one exists, is always in the candidate set so it either wins
or is beaten on the merits).

The search space per dimension: single atoms, negated atoms, and
AND / OR / AND-NOT pairs of the top-15 single atoms — bounded and fully
enumerable, so every candidate's score lands in the trace (no greedy path
dependence, unlike 08's open-ended keyword mining).

Usage:
    uv run python scripts/11_fit_tag_harness.py [--folds 5] [--seed 42] [--variance-penalty 1.0]
"""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import agreement_check
    import harness_fit
    import pipeline_logger
    import tag_harness_defs as defs
except ImportError:
    from scripts import agreement_check, harness_fit, pipeline_logger
    from scripts import tag_harness_defs as defs

LABELED_PATH = Path("data/interim/tag_fit/labeled.parquet")
DEV_PATH = Path("data/interim/tag_fit/dev_split.parquet")
HOLDOUT_PATH = Path("data/interim/tag_fit/holdout_split.parquet")
FIT_REPORT_PATH = Path("reports/tag_fit_harness.txt")
HOLDOUT_REPORT_PATH = Path("reports/tag_fit_holdout_eval.txt")
SEARCH_TRACE_PATH = Path("reports/tag_fit_search_trace.json")
CONFIG_PATH = Path("configs/config.json")
ATOMS_PATH_KEY = "keyword_atom_features.parquet"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return json.load(f)


def save_config(config: dict) -> None:
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)
        f.write("\n")


def load_labeled_with_atoms(config: dict) -> pd.DataFrame:
    labeled = pd.read_parquet(LABELED_PATH)
    atoms_path = Path(config["paths"]["candidate_chunks"]) / ATOMS_PATH_KEY
    if not atoms_path.exists():
        raise FileNotFoundError(f"{atoms_path} not found. Run scripts/09_extract_keyword_atoms.py first.")
    atoms = pd.read_parquet(atoms_path).drop_duplicates(subset="chunk_id")
    merged = labeled.merge(atoms, on="chunk_id", how="inner")
    dropped = len(labeled) - len(merged)
    if dropped:
        print(f"Dropped {dropped} labeled chunk_id(s) not present in current atom features "
              f"(re-run 09 after any re-chunk).")
    return merged


def build_candidates(df: pd.DataFrame, dimension: str, pool: list[str],
                     folds: list[np.ndarray], y: np.ndarray) -> dict[str, tuple[dict, np.ndarray]]:
    """Bounded candidate set: legacy formula (if any), single atoms and their
    negations, and AND/OR/AND-NOT pairs of the top-15 singles by fold-stability
    mean. Degenerate (constant) atoms are dropped first — a constant atom's
    negation is an always-True predictor that trivially exploits prevalence.

    `pool` is the dimension's ACTIVE atom pool from config
    (tagging.atom_pools) — the harness state the meta-optimizer grows; see
    the journal in .claude/skills/meta-harness-opt/journals/. Pool names may be atom columns or meta-atom
    names (tag_harness_defs.build_meta_atoms)."""
    meta = defs.build_meta_atoms(df)
    atoms: dict[str, pd.Series] = {}
    for name in pool:
        if name in meta:
            atoms[name] = meta[name].astype(bool)
        elif name in df.columns:
            atoms[name] = df[name].astype(bool)
        else:
            print(f"  WARNING [{dimension}]: pool atom '{name}' not found in features or meta-atoms — skipped.")

    degenerate = [name for name, s in atoms.items() if s.nunique() < 2]
    for name in degenerate:
        del atoms[name]

    candidates: dict[str, tuple[dict, np.ndarray]] = {}
    if defs.legacy_formula(df, dimension) is not None:
        spec = {"op": "legacy", "dimension": dimension}
        candidates[f"LEGACY[{dimension}]"] = (spec, defs.apply_formula_spec(spec, df))

    for name, series in atoms.items():
        candidates[name] = ({"op": "single", "feature": name}, series.to_numpy())
        candidates[f"NOT {name}"] = ({"op": "not", "feature": name}, (~series).to_numpy())

    single_scores = {name: harness_fit.fold_stability_score(y, series.to_numpy().astype(bool), folds)[0]
                     for name, series in atoms.items()}
    top_atoms = sorted(atoms.items(), key=lambda kv: single_scores.get(kv[0], 0.0), reverse=True)[:15]

    for (name_a, a), (name_b, b) in itertools.combinations(top_atoms, 2):
        av, bv = a.to_numpy().astype(bool), b.to_numpy().astype(bool)
        candidates[f"{name_a} AND {name_b}"] = ({"op": "and", "features": [name_a, name_b]}, av & bv)
        candidates[f"{name_a} OR {name_b}"] = ({"op": "or", "features": [name_a, name_b]}, av | bv)
        candidates[f"{name_a} AND NOT {name_b}"] = ({"op": "and_not", "features": [name_a, name_b]}, av & ~bv)

    return candidates


def search_dimension(dev: pd.DataFrame, dimension: str, pool: list[str],
                     n_folds: int, seed: int, variance_penalty: float) -> dict:
    y = dev[f"llm_{dimension}"].astype(bool).to_numpy()
    folds = harness_fit.stability_folds(y, n_folds, seed)
    candidates = build_candidates(dev, dimension, pool, folds, y)

    results: list[dict] = []
    for name, (spec, pred) in candidates.items():
        mean_f1, std_f1 = harness_fit.fold_stability_score(y, pred.astype(bool), folds)
        results.append({"name": name, "spec": spec, "mean_f1": mean_f1, "std_f1": std_f1,
                        "penalized": mean_f1 - variance_penalty * std_f1})
    results.sort(key=lambda r: r["penalized"], reverse=True)

    always_true_mean, always_true_std = harness_fit.fold_stability_score(
        y, np.ones_like(y, dtype=bool), folds)
    legacy_key = f"LEGACY[{dimension}]"
    legacy = next((r for r in results if r["name"] == legacy_key), None)

    return {
        "dimension": dimension,
        "n_candidates": len(candidates),
        "prevalence": float(y.mean()),
        "n": len(y),
        "winner": results[0],
        "legacy": legacy,
        "always_true": {"mean_f1": always_true_mean, "std_f1": always_true_std},
        "all_candidates": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dev-frac", type=float, default=0.7)
    parser.add_argument("--variance-penalty", type=float, default=1.0,
                        help="Subtract penalty*std from mean F1 when ranking — a formula that only wins on one lucky fold doesn't get picked")
    parser.add_argument("--dev-only", action="store_true",
                        help="Proposer mode: run the dev search and report only — no holdout look, no config write. "
                             "Safe to run any number of times while iterating on atom pools / cycle code.")
    args = parser.parse_args()

    if not LABELED_PATH.exists():
        print(f"Error: {LABELED_PATH} not found. Run scripts/10_sample_and_label_tags.py first.")
        return
    if not args.dev_only and harness_fit.refuse_if_holdout_spent(
            HOLDOUT_REPORT_PATH,
            "Sample a fresh batch (10) for another iteration instead of re-running this, "
            "or iterate with --dev-only (no holdout spent)."):
        return

    config = load_config()
    atom_pools = config.get("tagging", {}).get("atom_pools")
    if not atom_pools:
        print("Error: no tagging.atom_pools in config — the harness has no active atom space.")
        return
    df = load_labeled_with_atoms(config)
    print(f"Loaded {len(df)} labeled chunks with atom features.")

    # One dev/holdout split for all dimensions (stratified on the rarest
    # dimension so its holdout isn't starved of positives); per-dimension
    # stability folds are drawn on dev.
    prevalences = {d: df[f"llm_{d}"].astype(bool).mean() for d in defs.DIMENSIONS}
    rarest = min(prevalences, key=lambda d: min(prevalences[d], 1 - prevalences[d]))
    print("Dimension prevalences: " + ", ".join(f"{d}={p*100:.0f}%" for d, p in prevalences.items()))
    print(f"Splitting stratified on rarest dimension: {rarest}")
    df[f"llm_{rarest}"] = df[f"llm_{rarest}"].astype(bool)
    dev, holdout = harness_fit.stratified_split(df, f"llm_{rarest}", args.dev_frac, args.seed)
    DEV_PATH.parent.mkdir(parents=True, exist_ok=True)
    dev.to_parquet(DEV_PATH, index=False)
    holdout.to_parquet(HOLDOUT_PATH, index=False)
    print(f"Dev: {len(dev)}  Holdout: {len(holdout)}")

    trace_rounds: list[dict] = []
    meta = {"n_folds": args.folds, "variance_penalty": args.variance_penalty,
            "dimensions": defs.DIMENSIONS,
            "score": "fold-stability penalized (mean F1 - penalty*std), not CV"}

    fit_lines = [
        "Tag harness fit — DEV search (spent, not the final number)",
        "=" * 74,
        f"Dev set: {len(dev)} chunks, {args.folds}-fold stability score per dimension",
        "(Fold-stability score, not cross-validation — the holdout below is the honest number.)",
        f"Full per-candidate trace: {SEARCH_TRACE_PATH}",
        "",
    ]

    results = {}
    for dimension in defs.DIMENSIONS:
        pool = atom_pools.get(dimension, [])
        result = search_dimension(dev, dimension, pool, args.folds, args.seed, args.variance_penalty)
        results[dimension] = result
        trace_rounds.append({
            "dimension": dimension,
            "n_candidates_evaluated": result["n_candidates"],
            "always_true": result["always_true"],
            "candidates": [{k: v for k, v in r.items()} for r in result["all_candidates"]],
            "chosen": result["winner"]["name"],
        })
        harness_fit.flush_trace(SEARCH_TRACE_PATH, meta, trace_rounds)

        fit_lines.append(f"Dimension: {dimension}  (dev prevalence {result['prevalence']*100:.1f}%, n={result['n']})")
        fit_lines.append(f"  Candidates evaluated: {result['n_candidates']}")
        fit_lines.append(f"  Always-predict-True baseline: mean F1={result['always_true']['mean_f1']:.3f} "
                         f"(prevalence exploit — a winner must clear this to mean anything)")
        if result["legacy"]:
            fit_lines.append(f"  Legacy production formula: mean F1={result['legacy']['mean_f1']:.3f}, "
                             f"std={result['legacy']['std_f1']:.3f}")
        fit_lines.append(f"  Winner: {result['winner']['name']}  "
                         f"(mean F1={result['winner']['mean_f1']:.3f}, std={result['winner']['std_f1']:.3f}, "
                         f"penalized={result['winner']['penalized']:.3f})")
        for r in result["all_candidates"][:8]:
            fit_lines.append(f"    {r['penalized']:.3f}  (mean={r['mean_f1']:.3f}, std={r['std_f1']:.3f})  {r['name']}")
        fit_lines.append("")
        print(f"[{dimension}] winner: {result['winner']['name']} "
              f"(dev stability mean F1={result['winner']['mean_f1']:.3f}) "
              f"vs always-True {result['always_true']['mean_f1']:.3f}")

    report = "\n".join(fit_lines)
    FIT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIT_REPORT_PATH.write_text(report + "\n")
    print(f"\nDev report -> {FIT_REPORT_PATH}")

    if args.dev_only:
        print("\n--dev-only: stopping before the holdout look. No holdout spent, no config written.")
        return

    # ---- Single-look holdout evaluation of the frozen winners ----
    w_holdout = harness_fit.sampling_weights(holdout)
    holdout_lines = [
        "Tag harness fit — LOCKED HOLDOUT evaluation (single look)",
        "=" * 74,
        f"Holdout set: {len(holdout)} chunks, never used in the dev search",
        "Raw metrics score the sample as drawn; weighted metrics are inverse-probability",
        "weighted back to the chunk population (industry-proportional draw, so the two",
        "are close by design here — reported anyway for consistency with cycle 1).",
        "",
    ]
    frozen: dict[str, dict] = {}
    for dimension in defs.DIMENSIONS:
        result = results[dimension]
        y_h = holdout[f"llm_{dimension}"].astype(bool).to_numpy()
        pred_h = defs.apply_formula_spec(result["winner"]["spec"], holdout).astype(bool)
        f1_winner = harness_fit.f1_score(y_h, pred_h)
        f1_always = harness_fit.f1_score(y_h, np.ones_like(y_h, dtype=bool))

        holdout_lines.append(f"Dimension: {dimension}  (holdout prevalence {y_h.mean()*100:.1f}%)")
        holdout_lines.append(f"  Winner: {result['winner']['name']}")
        holdout_lines.extend("  " + line for line in harness_fit.metrics_block("winner  ", y_h, pred_h, w_holdout))
        holdout_lines.append(f"  Always-predict-True holdout F1={f1_always:.3f}")
        if result["legacy"]:
            pred_legacy = defs.apply_formula_spec({"op": "legacy", "dimension": dimension}, holdout).astype(bool)
            holdout_lines.extend("  " + line for line in harness_fit.metrics_block("legacy  ", y_h, pred_legacy, w_holdout))

        # Config gate: a winner that can't beat the always-True prevalence
        # exploit on holdout is not a classifier — refuse to freeze it.
        if f1_winner > f1_always:
            frozen[dimension] = {"name": result["winner"]["name"], "spec": result["winner"]["spec"],
                                 "holdout_f1": round(f1_winner, 4)}
            holdout_lines.append("  -> FROZEN into config.")
        else:
            holdout_lines.append("  -> NOT frozen: does not beat the always-True baseline on holdout. "
                                 "This dimension needs better atoms or a fresh labeled batch (10).")
        holdout_lines.append("")

    holdout_lines.append("Judge validation (human anchor, from the auto-generated workbook):")
    holdout_lines.extend("  " + line for line in agreement_check.report_lines("tags"))
    holdout_lines.append("")
    holdout_lines.append("Rule: if this disappoints, do not re-run this script against this holdout.")
    holdout_lines.append("Sample and label a fresh batch (10) for the next fit iteration instead.")
    holdout_report = "\n".join(holdout_lines)
    print("\n" + holdout_report)
    HOLDOUT_REPORT_PATH.write_text(holdout_report + "\n")
    print(f"\nLocked holdout report -> {HOLDOUT_REPORT_PATH}")

    config.setdefault("tagging", {})["formulas"] = frozen
    save_config(config)
    print(f"\nUpdated {CONFIG_PATH} tagging.formulas with {len(frozen)}/{len(defs.DIMENSIONS)} dimensions. "
          f"Run scripts/12_tag_chunks.py to apply them to the corpus.")

    pipeline_logger.log_event(
        pipeline_step="tag_fit",
        level="SUCCESS",
        message=f"Tag harness fit complete: {len(frozen)}/{len(defs.DIMENSIONS)} dimensions frozen.",
        details={d: results[d]["winner"]["name"] for d in defs.DIMENSIONS},
    )


if __name__ == "__main__":
    main()
