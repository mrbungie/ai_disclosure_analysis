"""
val_04_label_search_topup.py — LLM-judge the 130-chunk search-pool top-up
(val_03's search_pool_topup_candidates__rule_based.parquet) and append it to
the search pool used by the val_06 boolean-search harness.

This is search-pool material — it's allowed to be biased (val_03 oversampled
toward is_governance_related_proxy) and gets used to freeze the 4 fine-dimension
formulas. It never becomes holdout data. Contrast with val_05, which labels the
disjoint, proportional holdout and is hard-locked to a single run.

Reuses the same independent judge/schema as val_01 (LLM_JUDGE_* env vars,
ValidationLabel).

Usage:
    uv run python scripts/val_04_label_search_topup.py [--concurrency 4] [--delay 0.3]
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

CANDIDATES_PATH = Path("data/processed/variant_rule_based/validation/search_pool_topup_candidates__rule_based.parquet")
OUT_PATH = Path("data/processed/variant_rule_based/validation/search_pool_topup_labeled__rule_based.parquet")


async def run(args: argparse.Namespace) -> None:
    if not CANDIDATES_PATH.exists():
        print(f"Error: {CANDIDATES_PATH} not found. Run val_03_dedup_split.py first.")
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
                        "stratum": row.get("stratum"),
                        "chunk_text": row["chunk_text"],
                    }
                    record.update({f"llm_{k}": v for k, v in label.items()})
                    records.append(record)
                progress["done"] += 1
                print(f"[{progress['done']}/{progress['total']}] Labeled chunk {row['chunk_id']}")
                if len(records) % 10 == 0:
                    pd.DataFrame(records).to_parquet(OUT_PATH, index=False)
            if args.delay > 0:
                await asyncio.sleep(args.delay)
            queue.task_done()

    workers = [asyncio.create_task(worker()) for _ in range(max(1, args.concurrency))]
    await asyncio.gather(*workers)

    labeled_df = pd.DataFrame(records)
    labeled_df.to_parquet(OUT_PATH, index=False)
    print(f"\nDone. {len(labeled_df)} labeled chunks -> {OUT_PATH}")

    pipeline_logger.log_event(
        pipeline_step="validation_hardening",
        level="SUCCESS",
        message=f"Labeled {len(labeled_df)} search-pool top-up chunks.",
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
