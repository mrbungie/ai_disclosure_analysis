"""
08_fit_prefilter_harness.py — Semi-reproducible fit cycle for the AI-keyword
prefilter (05-06): split the LLM-labeled sample (07) into dev/holdout, greedily
search candidate keywords against dev via k-fold CV until a target F1 is hit
(or no candidate helps, or a max-iteration budget runs out), then evaluate the
frozen keyword list ONCE on the disjoint holdout.

    07 (sample + LLM label)  ->  08 (this script)  ->  configs/config.json
                                                        (ai_keywords updated)

Design principles (same discipline used to validate the earlier scoring
proxies, just automated instead of ad hoc):
  - Dev is allowed to be searched exhaustively — it's spent, never reused as
    a clean estimate.
  - Holdout is touched exactly once. A disappointing holdout result is
    reported as-is; re-running this script against the same holdout to "try
    again" is refused (reports/prefilter_fit_holdout_eval.txt existing blocks
    a second run) — improving further requires a fresh labeled batch from 07.
  - Every accepted candidate keyword must survive k-fold CV (mean F1 minus a
    variance penalty), not just a single train/test split — a keyword that
    only helps one fold is fitting that fold's noise.
  - config.json's ai_keywords is only overwritten after the dev-side search
    converges, not per-candidate, so a crashed/interrupted run never leaves
    the config in a half-fit state.

Usage:
    uv run python scripts/08_fit_prefilter_harness.py [--target-f1 0.85] [--folds 5] [--max-iterations 20] [--seed 42]
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import pipeline_logger
except ImportError:
    from scripts import pipeline_logger

LABELED_PATH = Path("data/interim/prefilter_fit/labeled.parquet")
DEV_PATH = Path("data/interim/prefilter_fit/dev_split.parquet")
HOLDOUT_PATH = Path("data/interim/prefilter_fit/holdout_split.parquet")
FIT_REPORT_PATH = Path("reports/prefilter_fit_harness.txt")
HOLDOUT_REPORT_PATH = Path("reports/prefilter_fit_holdout_eval.txt")
SEARCH_TRACE_PATH = Path("reports/prefilter_fit_search_trace.json")
CONFIG_PATH = Path("configs/config.json")

STOPWORDS = {
    "the", "and", "for", "are", "our", "with", "that", "this", "have", "has",
    "will", "may", "not", "from", "these", "such", "which", "its", "their",
    "any", "all", "can", "also", "been", "were", "was", "but", "than", "then",
    "into", "under", "over", "each", "other", "more", "most", "some", "use",
    "used", "using", "new", "based", "including", "include", "includes",
}


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return json.load(f)


def save_config(config: dict) -> None:
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)
        f.write("\n")


def build_regex(keywords: list[str]) -> re.Pattern:
    patterns = [r"\b" + re.escape(kw).replace(r"\ ", r"\s+") + r"\b" for kw in keywords]
    return re.compile("|".join(patterns), re.IGNORECASE)


def clean_false_positives(text: str, false_positives: list[str]) -> str:
    for fp in false_positives:
        text = re.sub(r"\b" + re.escape(fp) + r"\b", "", text, flags=re.IGNORECASE)
    return text


def predict(texts: pd.Series, keywords: list[str], false_positives: list[str]) -> np.ndarray:
    regex = build_regex(keywords)
    cleaned = texts.apply(lambda t: clean_false_positives(t, false_positives))
    return cleaned.apply(lambda t: bool(regex.search(t))).to_numpy()


def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = int(np.sum(y_true & y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0


def precision_recall(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    tp = int(np.sum(y_true & y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return precision, recall


def stratified_kfold(y: np.ndarray, n_folds: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Hand-rolled stratified k-fold (no sklearn dependency) — splits positive
    and negative indices separately into n_folds roughly-equal chunks, then
    combines chunk i from each class into fold i's test set."""
    rng = np.random.default_rng(seed)
    folds_test_idx: list[list[int]] = [[] for _ in range(n_folds)]
    for label in (True, False):
        idx = np.where(y == label)[0]
        rng.shuffle(idx)
        chunks = np.array_split(idx, n_folds)
        for i, chunk in enumerate(chunks):
            folds_test_idx[i].extend(chunk.tolist())

    all_idx = np.arange(len(y))
    folds = []
    for test_idx in folds_test_idx:
        test_idx = np.array(sorted(test_idx))
        train_idx = np.setdiff1d(all_idx, test_idx)
        folds.append((train_idx, test_idx))
    return folds


def stratified_split(df: pd.DataFrame, label_col: str, dev_frac: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Proportional dev/holdout split, stratified by label so both sides have
    a representative positive rate — not by industry (that's already handled
    upstream in 07's sampling)."""
    rng = np.random.default_rng(seed)
    dev_parts, holdout_parts = [], []
    for _, group in df.groupby(label_col):
        idx = group.index.to_numpy().copy()
        rng.shuffle(idx)
        n_dev = round(len(idx) * dev_frac)
        dev_parts.append(group.loc[idx[:n_dev]])
        holdout_parts.append(group.loc[idx[n_dev:]])
    return pd.concat(dev_parts).sample(frac=1, random_state=seed), pd.concat(holdout_parts).sample(frac=1, random_state=seed)


def mine_candidates(dev: pd.DataFrame, keywords: list[str], false_positives: list[str]) -> list[str]:
    """Extract candidate keyword tokens (1-2 word n-grams) from dev's false
    negatives (LLM says AI-related, current regex misses it) — the same kind
    of gap "AIP"/"AIOps" were, just found automatically via frequency instead
    of manual inspection."""
    y = dev["llm_is_ai_related"].to_numpy()
    pred = predict(dev["paragraph_text"], keywords, false_positives)
    false_negatives = dev[y & ~pred]

    existing_lower = {kw.lower() for kw in keywords}
    counts: Counter[str] = Counter()
    token_re = re.compile(r"\b[a-zA-Z][a-zA-Z\-]{2,20}\b")

    for text in false_negatives["paragraph_text"]:
        words = [w.lower() for w in token_re.findall(text)]
        words = [w for w in words if w not in STOPWORDS and len(w) > 2]
        unigrams = set(words)
        bigrams = {f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)}
        for tok in unigrams | bigrams:
            if tok not in existing_lower:
                counts[tok] += 1

    # require the candidate to show up in at least 2 distinct false-negative
    # paragraphs — singletons are almost certainly noise in an ~100-400 row dev set.
    candidates = [tok for tok, n in counts.most_common(200) if n >= 2]
    return candidates


def greedy_search(dev: pd.DataFrame, base_keywords: list[str], false_positives: list[str],
                   folds: list[tuple[np.ndarray, np.ndarray]], target_f1: float,
                   max_iterations: int, variance_penalty: float, min_delta: float,
                   trace_path: Path) -> dict:
    """Greedy forward-selection search. Every round's FULL candidate evaluation
    (not just the winner) is appended to `trace` and flushed to `trace_path`
    after each iteration — the actual "how did the search get here" record,
    not just the final report's prose summary. If the process is killed
    mid-run, the trace file still shows every round completed so far."""
    y = dev["llm_is_ai_related"].to_numpy()
    texts = dev["paragraph_text"]
    keywords = list(base_keywords)

    def cv_score(kw_list: list[str]) -> tuple[float, float]:
        pred = predict(texts, kw_list, false_positives)
        scores = [f1_score(y[test_idx], pred[test_idx]) for _, test_idx in folds]
        return float(np.mean(scores)), float(np.std(scores))

    def flush_trace() -> None:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        with open(trace_path, "w") as f:
            json.dump({"target_f1": target_f1, "variance_penalty": variance_penalty,
                       "min_delta": min_delta, "n_folds": len(folds), "rounds": trace}, f, indent=2)

    baseline_mean, baseline_std = cv_score(keywords)
    history = [{"iteration": 0, "action": "baseline", "keyword": None, "mean_f1": baseline_mean, "std_f1": baseline_std}]
    trace: list[dict] = [{
        "iteration": 0, "keywords_before": list(base_keywords), "n_candidates_evaluated": 0,
        "candidates": [], "chosen": None,
        "baseline_mean_f1": baseline_mean, "baseline_std_f1": baseline_std,
    }]
    flush_trace()
    print(f"Baseline (current config keywords): CV mean F1={baseline_mean:.3f}, std={baseline_std:.3f}")

    if baseline_mean >= target_f1:
        print(f"Baseline already meets target F1 >= {target_f1}. No search needed.")
        return {"final_keywords": keywords, "history": history, "n_candidates_tried": 0,
                "final_mean_f1": baseline_mean, "final_std_f1": baseline_std, "hit_target": True}

    n_tried = 0
    current_mean, current_std = baseline_mean, baseline_std
    for iteration in range(1, max_iterations + 1):
        candidates = mine_candidates(dev.assign(paragraph_text=texts), keywords, false_positives)
        candidates = [c for c in candidates if c not in keywords]
        if not candidates:
            print(f"[iter {iteration}] No more candidates to try. Stopping.")
            trace.append({"iteration": iteration, "keywords_before": list(keywords), "n_candidates_evaluated": 0,
                          "candidates": [], "chosen": None, "stop_reason": "no_candidates_mined"})
            flush_trace()
            break

        round_candidates = []
        best_candidate, best_mean, best_std, best_penalized = None, current_mean, current_std, current_mean - variance_penalty * current_std
        for cand in candidates:
            n_tried += 1
            mean_f1, std_f1 = cv_score(keywords + [cand])
            penalized = mean_f1 - variance_penalty * std_f1
            round_candidates.append({"keyword": cand, "mean_f1": mean_f1, "std_f1": std_f1, "penalized": penalized})
            if penalized > best_penalized + min_delta:
                best_candidate, best_mean, best_std, best_penalized = cand, mean_f1, std_f1, penalized

        # Full round trace (every candidate tried, sorted best-first) — not just the winner.
        round_candidates.sort(key=lambda r: r["penalized"], reverse=True)

        if best_candidate is None:
            print(f"[iter {iteration}] No candidate improved the penalized CV score by >= {min_delta}. Stopping "
                  f"(tried {len(candidates)} candidates this round).")
            trace.append({"iteration": iteration, "keywords_before": list(keywords),
                          "n_candidates_evaluated": len(candidates), "candidates": round_candidates,
                          "chosen": None, "stop_reason": "no_candidate_improved"})
            flush_trace()
            break

        keywords.append(best_candidate)
        current_mean, current_std = best_mean, best_std
        history.append({"iteration": iteration, "action": "add", "keyword": best_candidate,
                         "mean_f1": best_mean, "std_f1": best_std})
        trace.append({"iteration": iteration, "keywords_before": list(keywords[:-1]),
                      "n_candidates_evaluated": len(candidates), "candidates": round_candidates,
                      "chosen": best_candidate, "chosen_mean_f1": best_mean, "chosen_std_f1": best_std})
        flush_trace()
        print(f"[iter {iteration}] +'{best_candidate}' -> CV mean F1={best_mean:.3f}, std={best_std:.3f} "
              f"({len(candidates)} candidates evaluated this round)")

        if current_mean >= target_f1:
            print(f"Target F1 >= {target_f1} reached after {iteration} iteration(s).")
            return {"final_keywords": keywords, "history": history, "n_candidates_tried": n_tried,
                    "final_mean_f1": current_mean, "final_std_f1": current_std, "hit_target": True}

    return {"final_keywords": keywords, "history": history, "n_candidates_tried": n_tried,
            "final_mean_f1": current_mean, "final_std_f1": current_std, "hit_target": current_mean >= target_f1}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-f1", type=float, default=0.85)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-iterations", type=int, default=20)
    parser.add_argument("--dev-frac", type=float, default=0.7)
    parser.add_argument("--variance-penalty", type=float, default=1.0)
    parser.add_argument("--min-delta", type=float, default=0.005, help="Minimum penalized-score improvement to accept a candidate")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not LABELED_PATH.exists():
        print(f"Error: {LABELED_PATH} not found. Run scripts/07_sample_and_label_prefilter.py first.")
        return

    if HOLDOUT_REPORT_PATH.exists():
        print(f"Error: {HOLDOUT_REPORT_PATH} already exists — this fit cycle's holdout has already been "
              f"evaluated once. Sample a fresh batch (07) for another iteration instead of re-running this.")
        return

    config = load_config()
    base_keywords = config["prefiltering"]["ai_keywords"]
    false_positives = config["prefiltering"]["false_positives"]

    labeled = pd.read_parquet(LABELED_PATH)
    labeled["llm_is_ai_related"] = labeled["llm_is_ai_related"].astype(bool)
    print(f"Loaded {len(labeled)} labeled paragraphs "
          f"({labeled['llm_is_ai_related'].mean()*100:.1f}% positive).")

    dev, holdout = stratified_split(labeled, "llm_is_ai_related", args.dev_frac, args.seed)
    DEV_PATH.parent.mkdir(parents=True, exist_ok=True)
    dev.to_parquet(DEV_PATH, index=False)
    holdout.to_parquet(HOLDOUT_PATH, index=False)
    print(f"Dev: {len(dev)} ({dev['llm_is_ai_related'].mean()*100:.1f}% positive)  "
          f"Holdout: {len(holdout)} ({holdout['llm_is_ai_related'].mean()*100:.1f}% positive)")

    folds = stratified_kfold(dev["llm_is_ai_related"].to_numpy(), args.folds, args.seed)

    result = greedy_search(dev, base_keywords, false_positives, folds,
                            args.target_f1, args.max_iterations, args.variance_penalty, args.min_delta,
                            SEARCH_TRACE_PATH)

    lines = []
    lines.append("Prefilter keyword fit — DEV search (spent, not the final number)")
    lines.append("=" * 70)
    lines.append(f"Dev set: {len(dev)} paragraphs, {args.folds}-fold CV, target F1 >= {args.target_f1}")
    lines.append(f"Candidates evaluated across all iterations: {result['n_candidates_tried']}")
    lines.append(f"Target reached: {result['hit_target']}")
    lines.append(f"Full per-round candidate trace: {SEARCH_TRACE_PATH}")
    lines.append("")
    lines.append("Search history:")
    for h in result["history"]:
        if h["action"] == "baseline":
            lines.append(f"  [baseline] mean F1={h['mean_f1']:.3f}, std={h['std_f1']:.3f}")
        else:
            lines.append(f"  [iter {h['iteration']}] +'{h['keyword']}' -> mean F1={h['mean_f1']:.3f}, std={h['std_f1']:.3f}")
    lines.append("")
    lines.append(f"Final keyword list ({len(result['final_keywords'])} terms): {result['final_keywords']}")

    report = "\n".join(lines)
    print("\n" + report)
    FIT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIT_REPORT_PATH.write_text(report + "\n")
    print(f"\nDev report -> {FIT_REPORT_PATH}")

    # Single-look holdout evaluation of the frozen keyword list.
    final_keywords = result["final_keywords"]
    y_holdout = holdout["llm_is_ai_related"].to_numpy()
    pred_final = predict(holdout["paragraph_text"], final_keywords, false_positives)
    pred_baseline = predict(holdout["paragraph_text"], base_keywords, false_positives)

    f1_final = f1_score(y_holdout, pred_final)
    p_final, r_final = precision_recall(y_holdout, pred_final)
    f1_base = f1_score(y_holdout, pred_baseline)
    p_base, r_base = precision_recall(y_holdout, pred_baseline)

    holdout_lines = [
        "Prefilter keyword fit — LOCKED HOLDOUT evaluation (single look)",
        "=" * 70,
        f"Holdout set: {len(holdout)} paragraphs ({y_holdout.mean()*100:.1f}% positive), never used in the dev search",
        "",
        f"Fitted keyword list ({len(final_keywords)} terms):",
        f"  F1={f1_final:.3f}  P={p_final:.3f}  R={r_final:.3f}",
        "",
        f"Original config keywords ({len(base_keywords)} terms, for comparison):",
        f"  F1={f1_base:.3f}  P={p_base:.3f}  R={r_base:.3f}",
        "",
        "Rule: if this disappoints, do not re-run this script against this holdout.",
        "Sample and label a fresh batch (07) for the next fit iteration instead.",
    ]
    holdout_report = "\n".join(holdout_lines)
    print("\n" + holdout_report)
    HOLDOUT_REPORT_PATH.write_text(holdout_report + "\n")
    print(f"\nLocked holdout report -> {HOLDOUT_REPORT_PATH}")

    # Only overwrite config once the full dev search has converged — never
    # leave ai_keywords in a half-fit state if this script is interrupted mid-search.
    config["prefiltering"]["ai_keywords"] = final_keywords
    save_config(config)
    print(f"\nUpdated {CONFIG_PATH} with the fitted keyword list ({len(final_keywords)} terms). "
          f"Re-run scripts 05-06 to apply it to the corpus.")

    pipeline_logger.log_event(
        pipeline_step="prefilter_fit",
        level="SUCCESS",
        message=f"Prefilter fit complete. Dev mean F1={result['final_mean_f1']:.3f}, holdout F1={f1_final:.3f}.",
        details={
            "n_keywords_final": len(final_keywords),
            "n_keywords_baseline": len(base_keywords),
            "dev_mean_f1": result["final_mean_f1"],
            "holdout_f1": f1_final,
            "holdout_precision": p_final,
            "holdout_recall": r_final,
        },
    )


if __name__ == "__main__":
    main()
