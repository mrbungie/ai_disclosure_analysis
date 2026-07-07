"""
val_05_label_holdout.py — LLM-judge the 350-chunk holdout (val_03's
holdout_candidates__rule_based.parquet) for the 4 fine dimensions.

HARD RULE: run this only AFTER val_06's boolean-search harness has frozen a
formula per dimension. Labeling the holdout before the search is frozen doesn't
violate anything by itself, but there's no reason to do it early, and doing it
late enforces the discipline of "the holdout is judged once, after the formula
is locked, and the resulting number is never used to re-tune."

This script does NOT compute any recall/F1 report — it only produces the labels.
val_07_holdout_eval.py applies the frozen formulas and writes the locked report,
refusing to re-run once it has.

Usage:
    uv run python scripts/val_05_label_holdout.py [--concurrency 4] [--delay 0.3]
"""

import argparse
import asyncio
from pathlib import Path

import pandas as pd

try:
    import pipeline_logger
    from val_01_sample_and_label import build_agent, label_chunk_async
except ImportError:
    from scripts import pipeline_logger
    from scripts.val_01_sample_and_label import build_agent, label_chunk_async

CANDIDATES_PATH = Path("data/processed/variant_rule_based/validation/holdout_candidates__rule_based.parquet")
OUT_PATH = Path("data/processed/variant_rule_based/validation/holdout_labeled__rule_based.parquet")
FROZEN_FORMULAS_PATH = Path("data/interim/validation/frozen_boolean_formulas__rule_based.json")


async def run(args: argparse.Namespace) -> None:
    if not CANDIDATES_PATH.exists():
        print(f"Error: {CANDIDATES_PATH} not found. Run val_03_dedup_split.py first.")
        return
    if not FROZEN_FORMULAS_PATH.exists():
        print(f"Error: {FROZEN_FORMULAS_PATH} not found. Run val_06_boolean_search_harness.py first — "
              f"the formula must be frozen before the holdout is labeled.")
        return

    candidates = pd.read_parquet(CANDIDATES_PATH)

    if OUT_PATH.exists():
        existing = pd.read_parquet(OUT_PATH)
        done_ids = set(existing["chunk_id"].tolist())
        records = existing.to_dict("records")
        to_label = candidates[~candidates["chunk_id"].isin(done_ids)].reset_index(drop=True)
        print(f"Resuming: {len(done_ids)} already labeled, {len(to_label)} remaining.")
        if len(to_label) == 0:
            print("All chunks already labeled.")
            return
    else:
        to_label = candidates
        records = []

    agent = build_agent()
    write_lock = asyncio.Lock()
    queue: asyncio.Queue = asyncio.Queue()
    for _, row in to_label.iterrows():
        queue.put_nowait(row)

    progress = {"done": 0, "total": len(to_label)}

    async def worker():
        while True:
            try:
                row = queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            label = await label_chunk_async(agent, row["chunk_text"])
            async with write_lock:
                if label is not None:
                    record = {
                        "chunk_id": row["chunk_id"],
                        "ticker": row.get("ticker"),
                        "filing_date": row.get("filing_date"),
                        "section_name": row.get("section_name"),
                        "chunk_text": row["chunk_text"],
                    }
                    record.update({f"llm_{k}": v for k, v in label.items()})
                    records.append(record)
                progress["done"] += 1
                print(f"[{progress['done']}/{progress['total']}] Labeled chunk {row['chunk_id']}")
                if len(records) % 20 == 0:
                    pd.DataFrame(records).to_parquet(OUT_PATH, index=False)
            if args.delay > 0:
                await asyncio.sleep(args.delay)
            queue.task_done()

    workers = [asyncio.create_task(worker()) for _ in range(max(1, args.concurrency))]
    await asyncio.gather(*workers)

    labeled_df = pd.DataFrame(records)
    labeled_df.to_parquet(OUT_PATH, index=False)
    print(f"\nDone. {len(labeled_df)} labeled holdout chunks -> {OUT_PATH}")
    print("Not evaluated yet. Run val_07_holdout_eval.py to apply the frozen formulas ONCE.")

    pipeline_logger.log_event(
        pipeline_step="validation_hardening",
        level="SUCCESS",
        message=f"Labeled {len(labeled_df)} holdout chunks for the 4 fine dimensions.",
        details={"n_labeled": len(labeled_df)},
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--delay", type=float, default=0.3)
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
