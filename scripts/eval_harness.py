"""
eval_harness.py — Evaluate(H, X) per stage, per docs/distillation_map.html.

    uv run python scripts/eval_harness.py --task detection --candidate 001_x [--split search]
    uv run python scripts/eval_harness.py --task detection --candidate 001_x --split test   # THE one look
    uv run python scripts/eval_harness.py --leaderboard [--task detection]

Three tasks; detection and phase0 share one eval set, classification has
its own; three selection metrics (map §0, §3, §4):

  detection      Population = seed-screen-stratified paragraphs
                 (data/interim/eval/eval_set_detection.parquet). Objective:
                 minimize candidate volume subject to weighted search recall
                 >= recall_floor (r0, configs/config.json:
                 seed_screen.recall_floor) — a floor, not a maximization, so
                 the trivial "admit everything" optimum is excluded. Search
                 split reports raw recall/precision/volume for selection;
                 test split reports the weighted holdout estimate of corpus
                 recall, precision, and corpus reduction — the headline
                 numbers — and gates the freeze on the recorded search
                 recall clearing the floor.

  phase0         Same population/labels as detection (same underlying
                 question — is this text AI-related — different candidate
                 family: embedding/ConceptSeed-backed rather than regex).
                 Objective: maximize UNIQUE recall gain over the current
                 detection ACTIVE (text this candidate catches that
                 detection's frozen candidate misses), subject to added
                 volume <= phase0.max_added_volume_frac — a ceiling, not a
                 floor, since here recall is what's maximized. Reads
                 detection's ACTIVE at scoring time, so results are only
                 meaningful relative to whichever detection candidate is
                 frozen right now.

  classification Population = chunks built around detection's admitted
                 paragraphs (data/interim/eval/eval_set_classification.parquet).
                 Objective: macro-averaged balanced accuracy (mean of
                 sensitivity and specificity) per label — not F1, so a
                 skewed base rate can't be won by predicting the majority
                 class. Search split is raw per-label balanced accuracy;
                 test split is the weighted holdout estimate.

Contract:
  - `--split search` (default): free, repeatable, for any candidate. Writes
    harnesses/<task>/<name>/eval_search.json and trace_search.parquet (per-
    instance predictions vs labels with a `wrong` column) — the filesystem
    history the proposer greps and reads.
  - `--split test`: ONE look per task per eval-set batch, for the candidate
    being frozen. Guarded by harnesses/<task>/TEST_LOOK (records the eval
    set fingerprint; a fresh labeled batch resets it). Detection refuses to
    freeze a candidate whose recorded search recall is below the floor;
    phase0 refuses to freeze one whose recorded added volume is above the
    ceiling. On success the candidate becomes harnesses/<task>/ACTIVE.
"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import agreement_check
    import harness_fit
    import pipeline_logger
except ImportError:
    from scripts import agreement_check, harness_fit, pipeline_logger

EVAL_SET_PATHS = {
    "detection": Path("data/interim/eval/eval_set_detection.parquet"),
    "classification": Path("data/interim/eval/eval_set_classification.parquet"),
    "phase0": Path("data/interim/eval/eval_set_detection.parquet"),  # same construct/labels as detection
}
ID_COLS = {"detection": "paragraph_id", "classification": "chunk_id", "phase0": "paragraph_id"}
TEXT_COLS = {"detection": "paragraph_text", "classification": "chunk_text", "phase0": "paragraph_text"}


def load_config() -> dict:
    with open("configs/config.json") as f:
        return json.load(f)


def eval_set_fingerprint(df: pd.DataFrame, id_col: str) -> str:
    ids = "|".join(sorted(df[id_col]))
    return hashlib.sha256(ids.encode()).hexdigest()[:16]


def detection_predictions(classify, texts: pd.Series) -> np.ndarray:
    return np.array([bool(classify(t)) for t in texts])


def classification_predictions(classify, texts: pd.Series) -> pd.DataFrame:
    return pd.DataFrame([classify(t) for t in texts])


def score_detection(name: str, split: str, df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    classify = harness_fit.load_candidate("detection", name)
    p = detection_predictions(classify, df["paragraph_text"])
    y = df["llm_is_ai_related"].astype(bool).to_numpy()

    trace = df[["paragraph_id", "ticker", "section_name", "paragraph_text"]].copy()
    trace["pred_is_ai_related"] = p
    trace["label_is_ai_related"] = y
    trace["wrong"] = trace["pred_is_ai_related"] != trace["label_is_ai_related"]

    if split == "search":
        precision, recall, f1 = harness_fit.precision_recall_f1(y, p)
        scores = {
            "search": {"precision": round(precision, 4), "recall": round(recall, 4),
                       "f1": round(f1, 4), "volume_frac": round(float(p.mean()) if len(p) else 0.0, 4)},
            "reward": round(recall, 4),  # selection ranks eligible candidates; recall is the gate
            "n_instances": len(df),
        }
    else:
        weights = harness_fit.sampling_weights(df)
        assert weights is not None, "the test/holdout split must carry sampling_weight"
        precision, recall, f1 = harness_fit.precision_recall_f1(y, p, weights)
        total_est = float(weights.sum())
        admitted_est = float(weights[p].sum())
        corpus_reduction = 1 - admitted_est / total_est if total_est else 0.0
        scores = {
            "holdout": {"precision_weighted": round(precision, 4), "recall_weighted": round(recall, 4),
                       "f1_weighted": round(f1, 4), "corpus_reduction": round(corpus_reduction, 4)},
            "reward": round(recall, 4),
            "n_instances": len(df),
        }
    return scores, trace


def score_phase0(name: str, split: str, df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """phase0 candidates aren't scored on their own detection quality — they're
    scored on what they add ON TOP of the current detection ACTIVE, since
    catching what a lexical candidate already catches is zero marginal value
    for the seed screen's semantic_hit signal (harnesses/phase0/, consumed by
    scripts/seed_screen.py)."""
    phase0_classify = harness_fit.load_candidate("phase0", name)
    lexical_classify = harness_fit.load_candidate("detection", harness_fit.active_candidate("detection"))
    p_phase0 = detection_predictions(phase0_classify, df["paragraph_text"])
    p_lexical = detection_predictions(lexical_classify, df["paragraph_text"])
    y = df["llm_is_ai_related"].astype(bool).to_numpy()
    added = p_phase0 & ~p_lexical

    trace = df[["paragraph_id", "ticker", "section_name", "paragraph_text"]].copy()
    trace["pred_phase0"] = p_phase0
    trace["pred_lexical_baseline"] = p_lexical
    trace["added_hit"] = added
    trace["label_is_ai_related"] = y
    trace["wrong"] = trace["added_hit"] & ~trace["label_is_ai_related"]

    weights = harness_fit.sampling_weights(df) if split == "test" else None
    w = weights if weights is not None else np.ones(len(df))
    y_f, added_f = y.astype(float), added.astype(float)
    unique_recall_gain = float((w * added_f * y_f).sum() / max((w * y_f).sum(), 1e-9))
    added_volume_frac = float((w * added_f).sum() / w.sum())
    added_precision = (float((w * added_f * y_f).sum() / (w * added_f).sum())
                       if added.any() else None)

    key = "search" if split == "search" else "holdout"
    scores = {
        key: {"unique_recall_gain": round(unique_recall_gain, 4),
              "added_volume_frac": round(added_volume_frac, 4),
              "added_precision": round(added_precision, 4) if added_precision is not None else None},
        "reward": round(unique_recall_gain, 4),
        "n_instances": len(df),
    }
    return scores, trace


def score_classification(name: str, split: str, df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    classify = harness_fit.load_candidate("classification", name)
    preds = classification_predictions(classify, df["chunk_text"])
    labels = harness_fit.TASK_LABELS["classification"]
    weights = harness_fit.sampling_weights(df) if split == "test" else None

    per_label: dict[str, dict] = {}
    for field in labels:
        y = df[f"llm_{field}"].astype(bool).to_numpy()
        p = preds[field].to_numpy()
        sens, spec, bal_acc = harness_fit.balanced_accuracy(y, p, weights)
        per_label[field] = {"sensitivity": round(sens, 4), "specificity": round(spec, 4),
                            "balanced_accuracy": round(bal_acc, 4),
                            "prevalence": round(float(np.mean(y)), 4)}

    reward = round(sum(v["balanced_accuracy"] for v in per_label.values()) / len(per_label), 4)
    scores = {"per_label": per_label, "reward": reward, "n_instances": len(df)}

    trace = df[["chunk_id", "ticker", "section_name", "chunk_text"]].copy()
    for field in labels:
        trace[f"pred_{field}"] = preds[field].to_numpy()
        trace[f"label_{field}"] = df[f"llm_{field}"].astype(bool).to_numpy()
    trace["wrong"] = [
        ",".join(f for f in labels if row[f"pred_{f}"] != row[f"label_{f}"]) or ""
        for _, row in trace.iterrows()
    ]
    return scores, trace


def score_candidate(task: str, name: str, split: str, df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    if task == "detection":
        return score_detection(name, split, df)
    if task == "phase0":
        return score_phase0(name, split, df)
    return score_classification(name, split, df)


def print_scores(task: str, name: str, split: str, scores: dict) -> None:
    print(f"\n[{task}] {name} on {split} (n={scores['n_instances']})")
    if task in ("detection", "phase0"):
        block = scores.get("search") or scores.get("holdout")
        label = "search" if "search" in scores else "holdout"
        print(f"  [{label}] " + "  ".join(f"{k}={v}" for k, v in block.items()))
        if task == "detection":
            print(f"  reward (recall, for the recall-floor gate): {scores['reward']}")
        else:
            print(f"  reward (unique recall gain over detection's ACTIVE, for the volume-ceiling gate): {scores['reward']}")
    else:
        for field, v in scores["per_label"].items():
            print(f"    {field:<24} sens={v['sensitivity']:.3f} spec={v['specificity']:.3f} "
                  f"bal_acc={v['balanced_accuracy']:.3f}  prev={v['prevalence']:.2f}")
        print(f"  reward (macro balanced accuracy): {scores['reward']}")


def leaderboard(task_filter: str | None) -> None:
    for task in harness_fit.TASK_LABELS:
        if task_filter and task != task_filter:
            continue
        task_dir = harness_fit.HARNESSES_DIR / task
        active = harness_fit.active_candidate(task)
        print(f"\n[{task}] candidates on search (ACTIVE = {active}):")

        if task == "detection":
            recall_floor = load_config()["seed_screen"]["recall_floor"]
            candidate_scores = {}
            for d in sorted(task_dir.iterdir()):
                f = d / "eval_search.json"
                if d.is_dir() and f.exists():
                    s = json.loads(f.read_text())["scores"]["search"]
                    candidate_scores[d.name] = {"recall_weighted": s["recall"], "volume_weighted": s["volume_frac"]}
            if not candidate_scores:
                print("  (none evaluated yet)")
                continue
            selected = harness_fit.select_stage1_candidate(candidate_scores, recall_floor)
            print(f"  recall floor = {recall_floor}; selection = min volume among candidates clearing it")
            for name, v in sorted(candidate_scores.items(), key=lambda kv: (kv[1]["recall_weighted"] < recall_floor, kv[1]["volume_weighted"])):
                mark = " <- selects" if name == selected else ""
                eligible = "eligible " if v["recall_weighted"] >= recall_floor else "below floor"
                print(f"  {name:<24} recall={v['recall_weighted']}  volume_frac={v['volume_weighted']}  ({eligible}){mark}")
            if selected is None:
                print("  No candidate clears the recall floor yet.")
        elif task == "phase0":
            ceiling = load_config().get("phase0", {}).get("max_added_volume_frac")
            rows = []
            for d in sorted(task_dir.iterdir()):
                f = d / "eval_search.json"
                if d.is_dir() and f.exists():
                    s = json.loads(f.read_text())["scores"]["search"]
                    rows.append((d.name, s["unique_recall_gain"], s["added_volume_frac"]))
            if not rows:
                print("  (none evaluated yet)")
                continue
            print(f"  max_added_volume_frac ceiling = {ceiling}; selection = max unique recall gain within it")
            for name, gain, vol in sorted(rows, key=lambda r: -r[1]):
                eligible = "eligible " if ceiling is None or vol <= ceiling else "over ceiling"
                print(f"  {name:<24} unique_recall_gain={gain}  added_volume_frac={vol}  ({eligible})")
        else:
            rows = []
            for d in sorted(task_dir.iterdir()):
                f = d / "eval_search.json"
                if d.is_dir() and f.exists():
                    s = json.loads(f.read_text())
                    rows.append((d.name, s["scores"]["reward"], s["evaluated_at"]))
            if not rows:
                print("  (none evaluated yet)")
                continue
            for name, reward, at in sorted(rows, key=lambda r: -r[1]):
                print(f"  {name:<24} macro balanced accuracy={reward}  ({at})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=sorted(harness_fit.TASK_LABELS), default=None)
    parser.add_argument("--candidate", default=None, help="Candidate dir name (default: the task's ACTIVE)")
    parser.add_argument("--split", choices=["search", "test"], default="search")
    parser.add_argument("--leaderboard", action="store_true")
    args = parser.parse_args()

    if args.leaderboard:
        leaderboard(args.task)
        return
    if not args.task:
        parser.error("--task is required (detection | phase0 | classification)")

    task = args.task
    eval_set_path = EVAL_SET_PATHS[task]
    id_col = ID_COLS[task]
    if not eval_set_path.exists():
        print(f"Error: {eval_set_path} not found. Run scripts/build_eval_set.py --stage {task} first.")
        return

    name = args.candidate or harness_fit.active_candidate(task)
    full = pd.read_parquet(eval_set_path)
    fingerprint = eval_set_fingerprint(full, id_col)
    df = full[full["split"] == args.split].reset_index(drop=True)
    test_look_path = harness_fit.HARNESSES_DIR / task / "TEST_LOOK"

    if args.split == "test" and test_look_path.exists():
        look = json.loads(test_look_path.read_text())
        if look.get("eval_set_fingerprint") == fingerprint:
            print(f"Error: [{task}] this eval set's test split was already looked at once "
                  f"({look['candidate']} on {look['looked_at']}). It is spent — build and label a "
                  f"fresh eval set before the next freeze.")
            return

    out_dir = harness_fit.HARNESSES_DIR / task / name

    if task == "detection" and args.split == "test":
        recall_floor = load_config()["seed_screen"]["recall_floor"]
        search_path = out_dir / "eval_search.json"
        if not search_path.exists():
            print(f"Error: run --split search for {name} first — the freeze gate checks its "
                  f"recorded search recall against the floor ({recall_floor}).")
            return
        search_recall = json.loads(search_path.read_text())["scores"]["search"]["recall"]
        if search_recall < recall_floor:
            print(f"Error: [{task}] {name} search recall {search_recall} is below the recall "
                  f"floor {recall_floor} — refusing to freeze. Unconstrained recall has a trivial "
                  f"optimum (admit everything); the floor is what makes selection meaningful.")
            return

    if task == "phase0" and args.split == "test":
        ceiling = load_config()["phase0"]["max_added_volume_frac"]
        search_path = out_dir / "eval_search.json"
        if not search_path.exists():
            print(f"Error: run --split search for {name} first — the freeze gate checks its "
                  f"recorded added volume against the ceiling ({ceiling}).")
            return
        added_volume = json.loads(search_path.read_text())["scores"]["search"]["added_volume_frac"]
        if added_volume > ceiling:
            print(f"Error: [{task}] {name} would add {added_volume} volume_frac, above the ceiling "
                  f"{ceiling} — refusing to freeze. Unconstrained gain has a trivial optimum (admit "
                  f"everything); the ceiling is what makes selection meaningful.")
            return

    scores, trace = score_candidate(task, name, args.split, df)
    print_scores(task, name, args.split, scores)

    payload = {"task": task, "candidate": name, "split": args.split,
               "eval_set_fingerprint": fingerprint,
               "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "scores": scores}
    (out_dir / f"eval_{args.split}.json").write_text(json.dumps(payload, indent=2) + "\n")
    trace.to_parquet(out_dir / f"trace_{args.split}.parquet", index=False)
    print(f"\nScores -> {out_dir}/eval_{args.split}.json   trace -> {out_dir}/trace_{args.split}.parquet")

    if args.split == "test":
        test_look_path.write_text(json.dumps({
            "candidate": name, "eval_set_fingerprint": fingerprint,
            "looked_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}, indent=2) + "\n")
        harness_fit.set_active_candidate(task, name)
        print(f"FROZEN: [{task}] {name} is now ACTIVE. This eval set's test split is spent for {task}.")
        print("Reward-label audit (human workbook):")
        for line in agreement_check.report_lines(f"eval_{task}"):
            print("  " + line)

    pipeline_logger.log_event(pipeline_step="eval_harness", level="SUCCESS",
                              message=f"Evaluated [{task}] {name} on {args.split}: reward={scores['reward']}.",
                              details={"task": task, "candidate": name, "split": args.split})


if __name__ == "__main__":
    main()
