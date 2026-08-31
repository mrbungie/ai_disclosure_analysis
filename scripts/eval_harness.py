"""
eval_harness.py — Evaluate(H, X): score one candidate of one task on the
labeled eval set and log everything to the candidate's directory.

    uv run python scripts/eval_harness.py --task detection --candidate 001_x [--split search]
    uv run python scripts/eval_harness.py --task detection --candidate 001_x --split test   # THE one look
    uv run python scripts/eval_harness.py --leaderboard [--task detection]

Two tasks, one pattern:
  - detection      evaluates on ALL eval rows against label is_ai_related;
                   reward = F1 (raw + population-weighted via sampling_weight).
  - classification evaluates on the rows whose reward label says AI-related
                   (the task is only defined there), against the six
                   dimension labels; reward = macro-F1 across the six.

Contract:
  - `--split search` (default): free, repeatable, for any candidate. Writes
    harnesses/<task>/<name>/eval_search.json and trace_search.parquet
    (per-instance predictions vs labels with a `wrong` column) — the
    filesystem history the proposer greps and reads.
  - `--split test`: ONE look per task per eval-set batch, for the candidate
    being frozen. Guarded by harnesses/<task>/TEST_LOOK (records the eval
    set fingerprint; a fresh labeled batch resets it). On success the
    candidate becomes harnesses/<task>/ACTIVE — that is the freeze.
  - Raw metrics score the sample as drawn (balanced strata — optimistic);
    weighted metrics are the population estimate. Both logged; weighted is
    the honest number.
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

EVAL_SET_PATH = Path("data/interim/eval/eval_set.parquet")


def eval_set_fingerprint(df: pd.DataFrame) -> str:
    ids = "|".join(sorted(df["paragraph_id"]))
    return hashlib.sha256(ids.encode()).hexdigest()[:16]


def task_rows(task: str, df: pd.DataFrame) -> pd.DataFrame:
    if task == "classification":
        return df[df["llm_is_ai_related"].astype(bool)].reset_index(drop=True)
    return df


def predictions(task: str, classify, texts: pd.Series) -> pd.DataFrame:
    if task == "detection":
        return pd.DataFrame({"is_ai_related": [bool(classify(t)) for t in texts]})
    return pd.DataFrame([classify(t) for t in texts])


def score_candidate(task: str, name: str, df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    classify = harness_fit.load_candidate(task, name)
    preds = predictions(task, classify, df["paragraph_text"])
    labels = harness_fit.TASK_LABELS[task]
    weights = harness_fit.sampling_weights(df)

    per_label: dict[str, dict] = {}
    for field in labels:
        y = df[f"llm_{field}"].astype(bool).to_numpy()
        p = preds[field].to_numpy()
        rp, rr, rf1 = harness_fit.precision_recall_f1(y, p)
        entry = {"raw": {"precision": round(rp, 4), "recall": round(rr, 4), "f1": round(rf1, 4)},
                 "prevalence": round(float(np.mean(y)), 4)}
        if weights is not None:
            wp, wr, wf1 = harness_fit.precision_recall_f1(y, p, weights)
            entry["weighted"] = {"precision": round(wp, 4), "recall": round(wr, 4), "f1": round(wf1, 4)}
        per_label[field] = entry

    scores = {
        "reward_raw": round(sum(v["raw"]["f1"] for v in per_label.values()) / len(per_label), 4),
        "reward_weighted": (round(sum(v["weighted"]["f1"] for v in per_label.values()) / len(per_label), 4)
                            if weights is not None else None),
        "per_label": per_label,
        "n_instances": len(df),
    }

    trace = df[["paragraph_id", "ticker", "section_name", "paragraph_text"]].copy()
    for field in labels:
        trace[f"pred_{field}"] = preds[field].to_numpy()
        trace[f"label_{field}"] = df[f"llm_{field}"].astype(bool).to_numpy()
    trace["wrong"] = [
        ",".join(f for f in labels if row[f"pred_{f}"] != row[f"label_{f}"]) or ""
        for _, row in trace.iterrows()
    ]
    return scores, trace


def print_scores(task: str, name: str, split: str, scores: dict) -> None:
    print(f"\n[{task}] {name} on {split} (n={scores['n_instances']})")
    print(f"  reward: raw={scores['reward_raw']}"
          + (f"  weighted={scores['reward_weighted']}" if scores["reward_weighted"] is not None else ""))
    for field, v in scores["per_label"].items():
        w = v.get("weighted")
        print(f"    {field:<24} raw F1={v['raw']['f1']:.3f} (P={v['raw']['precision']:.3f} R={v['raw']['recall']:.3f})"
              + (f"  wF1={w['f1']:.3f}" if w else "") + f"  prev={v['prevalence']:.2f}")


def leaderboard(task_filter: str | None) -> None:
    for task in harness_fit.TASK_LABELS:
        if task_filter and task != task_filter:
            continue
        task_dir = harness_fit.HARNESSES_DIR / task
        active = harness_fit.active_candidate(task)
        rows = []
        for d in sorted(task_dir.iterdir()):
            f = d / "eval_search.json"
            if d.is_dir() and f.exists():
                s = json.loads(f.read_text())
                rows.append((d.name, s["scores"]["reward_raw"], s["scores"].get("reward_weighted"),
                             s["evaluated_at"]))
        print(f"\n[{task}] candidates on search (ACTIVE = {active}):")
        if not rows:
            print("  (none evaluated yet)")
        for name, raw, wt, at in sorted(rows, key=lambda r: -(r[2] if r[2] is not None else r[1])):
            print(f"  {name:<24} raw={raw}  weighted={wt}  ({at})")


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
        parser.error("--task is required (detection | classification)")
    if not EVAL_SET_PATH.exists():
        print(f"Error: {EVAL_SET_PATH} not found. Run scripts/build_eval_set.py first.")
        return

    task = args.task
    name = args.candidate or harness_fit.active_candidate(task)
    full = pd.read_parquet(EVAL_SET_PATH)
    fingerprint = eval_set_fingerprint(full)
    df = task_rows(task, full[full["split"] == args.split].reset_index(drop=True))
    test_look_path = harness_fit.HARNESSES_DIR / task / "TEST_LOOK"

    if args.split == "test" and test_look_path.exists():
        look = json.loads(test_look_path.read_text())
        if look.get("eval_set_fingerprint") == fingerprint:
            print(f"Error: [{task}] this eval set's test split was already looked at once "
                  f"({look['candidate']} on {look['looked_at']}). It is spent — build and label a "
                  f"fresh eval set before the next freeze.")
            return

    scores, trace = score_candidate(task, name, df)
    print_scores(task, name, args.split, scores)

    out_dir = harness_fit.HARNESSES_DIR / task / name
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
        for line in agreement_check.report_lines("eval"):
            print("  " + line)

    pipeline_logger.log_event(pipeline_step="eval_harness", level="SUCCESS",
                              message=f"Evaluated [{task}] {name} on {args.split}: reward raw={scores['reward_raw']}.",
                              details={"task": task, "candidate": name, "split": args.split})


if __name__ == "__main__":
    main()
