"""
08_fit_prefilter_harness.py — Cycle 1, the fit: split the LLM-judged sample
(07) into dev/holdout, greedily search candidate keywords against dev with a
fold-stability-penalized score until a target F1 is hit (or no candidate
helps, or the iteration budget runs out), then evaluate the frozen keyword
list ONCE on the disjoint holdout.

    07 (sample + LLM judge)  ->  08 (this script)  ->  configs/config.json
                                                        (ai_keywords updated)

This is one instantiation of the shared harness-optimization cycle
(scripts/harness_fit.py, docs/meta_harness_plan.md). Discipline:
  - Dev is spent freely by the search — never reused as a clean estimate.
  - The per-fold machinery is a FOLD-STABILITY check, not cross-validation:
    nothing is trained per fold and candidates are mined from all of dev, so
    the dev score is optimistic by construction. The holdout is the honest
    number.
  - Holdout is touched exactly once; a second run against the same holdout is
    refused. Improving further requires a fresh labeled batch from 07.
  - Metrics are reported raw AND inverse-probability weighted (07 records
    sampling_weight), so the balanced stratum draw doesn't overstate
    recall/F1 relative to the population.
  - config.json's ai_keywords is only overwritten after the dev search
    converges AND the fitted list does not underperform the baseline on the
    holdout — a crashed run or a losing search never degrades the config.
  - Every round's full candidate evaluation is flushed to the search trace
    after each iteration.

Usage:
    uv run python scripts/08_fit_prefilter_harness.py [--target-f1 0.85] [--folds 5] [--max-iterations 20] [--seed 42]
    uv run python scripts/08_fit_prefilter_harness.py --self-check          # synthetic search validation, no holdout spent
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import harness_fit
    import pipeline_logger
except ImportError:
    from scripts import harness_fit, pipeline_logger

LABELED_PATH = Path("data/interim/prefilter_fit/labeled.parquet")
DEV_PATH = Path("data/interim/prefilter_fit/dev_split.parquet")
HOLDOUT_PATH = Path("data/interim/prefilter_fit/holdout_split.parquet")
FIT_REPORT_PATH = Path("reports/prefilter_fit_harness.txt")
HOLDOUT_REPORT_PATH = Path("reports/prefilter_fit_holdout_eval.txt")
SEARCH_TRACE_PATH = Path("reports/prefilter_fit_search_trace.json")
SELFCHECK_REPORT_PATH = Path("reports/prefilter_fit_selfcheck.txt")
SELFCHECK_TRACE_PATH = Path("reports/prefilter_fit_selfcheck_trace.json")
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


def predict(texts: pd.Series, keywords: list[str], false_positives: list[str]) -> np.ndarray:
    regex = harness_fit.build_regex(keywords)
    cleaned = texts.apply(lambda t: harness_fit.clean_false_positives(t, false_positives))
    return cleaned.apply(lambda t: bool(regex.search(t))).to_numpy()


def mine_candidates(dev: pd.DataFrame, keywords: list[str], false_positives: list[str]) -> list[str]:
    """Extract candidate keyword tokens (1-2 word n-grams) from dev's false
    negatives (judge says AI-related, current regex misses it) — the same kind
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
                  folds: list[np.ndarray], target_f1: float,
                  max_iterations: int, variance_penalty: float, min_delta: float,
                  trace_path: Path) -> dict:
    """Greedy forward-selection search on dev. Every round's FULL candidate
    evaluation (not just the winner) is appended to the trace and flushed to
    `trace_path` after each iteration — the actual "how did the search get
    here" record. If the process is killed mid-run, the trace still shows
    every round completed so far.

    Note on ties: within a round each accepted improvement raises the bar for
    later candidates by min_delta, so among near-tied candidates the winner
    depends on evaluation order (frequency-ranked from mine_candidates). This
    is deliberate — earlier candidates appear in more false negatives — but it
    means the winner is not order-invariant."""
    y = dev["llm_is_ai_related"].to_numpy()
    texts = dev["paragraph_text"]
    keywords = list(base_keywords)

    def stability(kw_list: list[str]) -> tuple[float, float]:
        pred = predict(texts, kw_list, false_positives)
        return harness_fit.fold_stability_score(y, pred, folds)

    trace: list[dict] = []
    meta = {"target_f1": target_f1, "variance_penalty": variance_penalty,
            "min_delta": min_delta, "n_folds": len(folds),
            "score": "fold-stability penalized (mean F1 - penalty*std), not CV"}

    baseline_mean, baseline_std = stability(keywords)
    history = [{"iteration": 0, "action": "baseline", "keyword": None, "mean_f1": baseline_mean, "std_f1": baseline_std}]
    trace.append({
        "iteration": 0, "keywords_before": list(base_keywords), "n_candidates_evaluated": 0,
        "candidates": [], "chosen": None,
        "baseline_mean_f1": baseline_mean, "baseline_std_f1": baseline_std,
    })
    harness_fit.flush_trace(trace_path, meta, trace)
    print(f"Baseline (current config keywords): dev fold-stability mean F1={baseline_mean:.3f}, std={baseline_std:.3f}")

    if baseline_mean >= target_f1:
        print(f"Baseline already meets target F1 >= {target_f1}. No search needed.")
        return {"final_keywords": keywords, "history": history, "n_candidates_tried": 0,
                "final_mean_f1": baseline_mean, "final_std_f1": baseline_std, "hit_target": True}

    n_tried = 0
    current_mean, current_std = baseline_mean, baseline_std
    for iteration in range(1, max_iterations + 1):
        candidates = mine_candidates(dev, keywords, false_positives)
        candidates = [c for c in candidates if c not in keywords]
        if not candidates:
            print(f"[iter {iteration}] No more candidates to try. Stopping.")
            trace.append({"iteration": iteration, "keywords_before": list(keywords), "n_candidates_evaluated": 0,
                          "candidates": [], "chosen": None, "stop_reason": "no_candidates_mined"})
            harness_fit.flush_trace(trace_path, meta, trace)
            break

        round_candidates = []
        best_candidate, best_mean, best_std = None, current_mean, current_std
        best_penalized = current_mean - variance_penalty * current_std
        for cand in candidates:
            n_tried += 1
            mean_f1, std_f1 = stability(keywords + [cand])
            penalized = mean_f1 - variance_penalty * std_f1
            round_candidates.append({"keyword": cand, "mean_f1": mean_f1, "std_f1": std_f1, "penalized": penalized})
            if penalized > best_penalized + min_delta:
                best_candidate, best_mean, best_std, best_penalized = cand, mean_f1, std_f1, penalized

        # Full round trace (every candidate tried, sorted best-first) — not just the winner.
        round_candidates.sort(key=lambda r: r["penalized"], reverse=True)

        if best_candidate is None:
            print(f"[iter {iteration}] No candidate improved the penalized stability score by >= {min_delta}. Stopping "
                  f"(tried {len(candidates)} candidates this round).")
            trace.append({"iteration": iteration, "keywords_before": list(keywords),
                          "n_candidates_evaluated": len(candidates), "candidates": round_candidates,
                          "chosen": None, "stop_reason": "no_candidate_improved"})
            harness_fit.flush_trace(trace_path, meta, trace)
            break

        keywords.append(best_candidate)
        current_mean, current_std = best_mean, best_std
        history.append({"iteration": iteration, "action": "add", "keyword": best_candidate,
                        "mean_f1": best_mean, "std_f1": best_std})
        trace.append({"iteration": iteration, "keywords_before": list(keywords[:-1]),
                      "n_candidates_evaluated": len(candidates), "candidates": round_candidates,
                      "chosen": best_candidate, "chosen_mean_f1": best_mean, "chosen_std_f1": best_std})
        harness_fit.flush_trace(trace_path, meta, trace)
        print(f"[iter {iteration}] +'{best_candidate}' -> dev stability mean F1={best_mean:.3f}, std={best_std:.3f} "
              f"({len(candidates)} candidates evaluated this round)")

        if current_mean >= target_f1:
            print(f"Target F1 >= {target_f1} reached after {iteration} iteration(s).")
            return {"final_keywords": keywords, "history": history, "n_candidates_tried": n_tried,
                    "final_mean_f1": current_mean, "final_std_f1": current_std, "hit_target": True}

    return {"final_keywords": keywords, "history": history, "n_candidates_tried": n_tried,
            "final_mean_f1": current_mean, "final_std_f1": current_std, "hit_target": current_mean >= target_f1}


def run_self_check(labeled: pd.DataFrame, config: dict, args: argparse.Namespace) -> None:
    """Synthetic validation of the search machinery itself: remove keywords
    from the baseline list, run the dev search, and check it recovers the lost
    F1. Uses dev only — the real holdout is neither read nor created, no
    config is written, and the real trace/report files are untouched, so this
    can run at any time without spending anything."""
    base_keywords = config["prefiltering"]["ai_keywords"]
    false_positives = config["prefiltering"]["false_positives"]

    if args.self_check_drop:
        dropped = [k.strip() for k in args.self_check_drop.split(",")]
        missing = [k for k in dropped if k not in base_keywords]
        if missing:
            print(f"Error: not in ai_keywords: {missing}")
            return
    else:
        rng = np.random.default_rng(args.seed)
        # Drop 2 keywords that actually fire on this sample, so the handicap is real.
        firing = [kw for kw in base_keywords
                  if predict(labeled["paragraph_text"], [kw], false_positives).any()]
        dropped = [str(k) for k in rng.choice(firing, size=min(2, len(firing)), replace=False)]
    crippled = [kw for kw in base_keywords if kw not in dropped]
    print(f"Self-check: dropped {dropped} from the keyword list ({len(base_keywords)} -> {len(crippled)}).")

    dev, _ = harness_fit.stratified_split(labeled, "llm_is_ai_related", args.dev_frac, args.seed)
    folds = harness_fit.stability_folds(dev["llm_is_ai_related"].to_numpy(), args.folds, args.seed)

    y = dev["llm_is_ai_related"].to_numpy()
    full_mean, _ = harness_fit.fold_stability_score(y, predict(dev["paragraph_text"], base_keywords, false_positives), folds)
    result = greedy_search(dev, crippled, false_positives, folds,
                           target_f1=full_mean, max_iterations=args.max_iterations,
                           variance_penalty=args.variance_penalty, min_delta=args.min_delta,
                           trace_path=SELFCHECK_TRACE_PATH)

    recovered = result["final_mean_f1"] >= full_mean - args.min_delta
    lines = [
        "Prefilter fit harness — SELF-CHECK (synthetic; no holdout spent, no config written)",
        "=" * 78,
        f"Dropped keywords: {dropped}",
        f"Full-list dev stability mean F1 (recovery target): {full_mean:.3f}",
        f"Crippled-list baseline: {result['history'][0]['mean_f1']:.3f}",
        f"After search: {result['final_mean_f1']:.3f} "
        f"({len(result['final_keywords']) - len(crippled)} keyword(s) added: "
        f"{result['final_keywords'][len(crippled):]})",
        f"Candidates evaluated: {result['n_candidates_tried']}  (trace: {SELFCHECK_TRACE_PATH})",
        "",
        f"RESULT: {'PASS' if recovered else 'FAIL'} — the search machinery "
        f"{'recovered' if recovered else 'did NOT recover'} the removed keywords' F1 within min_delta.",
    ]
    report = "\n".join(lines)
    print("\n" + report)
    SELFCHECK_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SELFCHECK_REPORT_PATH.write_text(report + "\n")
    print(f"\nSelf-check report -> {SELFCHECK_REPORT_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-f1", type=float, default=0.85)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-iterations", type=int, default=20)
    parser.add_argument("--dev-frac", type=float, default=0.7)
    parser.add_argument("--variance-penalty", type=float, default=1.0)
    parser.add_argument("--min-delta", type=float, default=0.005, help="Minimum penalized-score improvement to accept a candidate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--self-check", action="store_true",
                        help="Synthetic search validation: drop keywords, verify the search recovers them (dev only)")
    parser.add_argument("--self-check-drop", type=str, default=None,
                        help="Comma-separated keywords to drop in --self-check (default: 2 random firing keywords)")
    parser.add_argument("--dev-only", action="store_true",
                        help="Proposer mode: run the dev search and report only — no holdout look, no config write. "
                             "Safe to run any number of times while iterating on the cycle code.")
    args = parser.parse_args()

    if not LABELED_PATH.exists():
        print(f"Error: {LABELED_PATH} not found. Run scripts/07_sample_and_label_prefilter.py first.")
        return

    config = load_config()
    labeled = pd.read_parquet(LABELED_PATH)
    labeled["llm_is_ai_related"] = labeled["llm_is_ai_related"].astype(bool)

    if args.self_check:
        run_self_check(labeled, config, args)
        return

    if not args.dev_only and harness_fit.refuse_if_holdout_spent(
            HOLDOUT_REPORT_PATH,
            "Sample a fresh batch (07) for another iteration instead of re-running this, "
            "or iterate with --dev-only (no holdout spent)."):
        return

    base_keywords = config["prefiltering"]["ai_keywords"]
    false_positives = config["prefiltering"]["false_positives"]

    print(f"Loaded {len(labeled)} labeled paragraphs "
          f"({labeled['llm_is_ai_related'].mean()*100:.1f}% positive).")

    dev, holdout = harness_fit.stratified_split(labeled, "llm_is_ai_related", args.dev_frac, args.seed)
    DEV_PATH.parent.mkdir(parents=True, exist_ok=True)
    dev.to_parquet(DEV_PATH, index=False)
    holdout.to_parquet(HOLDOUT_PATH, index=False)
    print(f"Dev: {len(dev)} ({dev['llm_is_ai_related'].mean()*100:.1f}% positive)  "
          f"Holdout: {len(holdout)} ({holdout['llm_is_ai_related'].mean()*100:.1f}% positive)")

    folds = harness_fit.stability_folds(dev["llm_is_ai_related"].to_numpy(), args.folds, args.seed)

    result = greedy_search(dev, base_keywords, false_positives, folds,
                           args.target_f1, args.max_iterations, args.variance_penalty, args.min_delta,
                           SEARCH_TRACE_PATH)

    lines = []
    lines.append("Prefilter keyword fit — DEV search (spent, not the final number)")
    lines.append("=" * 70)
    lines.append(f"Dev set: {len(dev)} paragraphs, {args.folds}-fold stability score, target F1 >= {args.target_f1}")
    lines.append("(Fold-stability score, not cross-validation: nothing is trained per fold and")
    lines.append(" candidates are mined from all of dev — the holdout below is the honest number.)")
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

    if args.dev_only:
        print("\n--dev-only: stopping before the holdout look. No holdout spent, no config written.")
        return

    # Single-look holdout evaluation of the frozen keyword list — raw and
    # inverse-probability-weighted (population) metrics side by side.
    final_keywords = result["final_keywords"]
    y_holdout = holdout["llm_is_ai_related"].to_numpy()
    w_holdout = harness_fit.sampling_weights(holdout)
    pred_final = predict(holdout["paragraph_text"], final_keywords, false_positives)
    pred_baseline = predict(holdout["paragraph_text"], base_keywords, false_positives)

    f1_final = harness_fit.f1_score(y_holdout, pred_final)
    f1_base = harness_fit.f1_score(y_holdout, pred_baseline)

    holdout_lines = [
        "Prefilter keyword fit — LOCKED HOLDOUT evaluation (single look)",
        "=" * 70,
        f"Holdout set: {len(holdout)} paragraphs ({y_holdout.mean()*100:.1f}% positive), never used in the dev search",
        "Raw metrics score the sample as drawn (50/50 stratum draw — overstates recall);",
        "weighted metrics are inverse-probability weighted back to the paragraph population.",
        "",
        f"Fitted keyword list ({len(final_keywords)} terms):",
        *("  " + line for line in harness_fit.metrics_block("fitted  ", y_holdout, pred_final, w_holdout)),
        "",
        f"Original config keywords ({len(base_keywords)} terms, for comparison):",
        *("  " + line for line in harness_fit.metrics_block("baseline", y_holdout, pred_baseline, w_holdout)),
        "",
        "Rule: if this disappoints, do not re-run this script against this holdout.",
        "Sample and label a fresh batch (07) for the next fit iteration instead.",
    ]
    holdout_report = "\n".join(holdout_lines)
    print("\n" + holdout_report)
    HOLDOUT_REPORT_PATH.write_text(holdout_report + "\n")
    print(f"\nLocked holdout report -> {HOLDOUT_REPORT_PATH}")

    # Config write gate: only after a converged search (never mid-run), and
    # only if the fitted list does not underperform the baseline on holdout —
    # at that point the baseline was the safer choice.
    if final_keywords == base_keywords:
        print(f"\nKeyword list unchanged — {CONFIG_PATH} left as is.")
    elif f1_final < f1_base:
        print(f"\nNOT updating {CONFIG_PATH}: fitted list underperforms the baseline on holdout "
              f"(F1 {f1_final:.3f} < {f1_base:.3f}). Baseline keywords kept. "
              f"Sample a fresh batch (07) before trying again.")
    else:
        config["prefiltering"]["ai_keywords"] = final_keywords
        save_config(config)
        print(f"\nUpdated {CONFIG_PATH} with the fitted keyword list ({len(final_keywords)} terms). "
              f"Re-run scripts 05-06 to apply it to the corpus.")

    pipeline_logger.log_event(
        pipeline_step="prefilter_fit",
        level="SUCCESS",
        message=f"Prefilter fit complete. Dev stability mean F1={result['final_mean_f1']:.3f}, holdout F1={f1_final:.3f}.",
        details={
            "n_keywords_final": len(final_keywords),
            "n_keywords_baseline": len(base_keywords),
            "dev_stability_mean_f1": result["final_mean_f1"],
            "holdout_f1": f1_final,
        },
    )


if __name__ == "__main__":
    main()
