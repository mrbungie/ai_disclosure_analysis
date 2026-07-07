"""
val_11_prefilter_holdout_label.py — Judge the FRESH prefilter-recall holdout
(val_10) ONCE and write the locked recall report for the frozen keyword list
(default terms + aip/aiops).

Hard rule, same as the fine-dimension holdout: if reports/prefilter_recall_holdout_validation.txt
already exists, this script refuses to run again. If the number disappoints,
the fix is a THIRD fresh batch never seen before — not re-running this one.

Usage:
    uv run python scripts/val_11_prefilter_holdout_label.py [--concurrency 4] [--delay 0.3]
"""

import argparse
import asyncio
import json
from pathlib import Path

import pandas as pd

try:
    from val_09_prefilter_recall_label import (
        build_agent, label_paragraph_async, wilson_interval, LABELED_SAMPLE_PATH,
    )
    from val_08_prefilter_recall import build_regex, clean_false_positives, load_config
except ImportError:
    from scripts.val_09_prefilter_recall_label import (
        build_agent, label_paragraph_async, wilson_interval, LABELED_SAMPLE_PATH,
    )
    from scripts.val_08_prefilter_recall import build_regex, clean_false_positives, load_config

try:
    import pipeline_logger
except ImportError:
    from scripts import pipeline_logger

CANDIDATES_PATH = Path("data/interim/validation/prefilter_recall_holdout_candidates.parquet")
POPULATIONS_PATH = Path("data/interim/validation/prefilter_recall_holdout_populations.json")
OUT_PATH = Path("data/interim/validation/prefilter_recall_holdout_labeled.parquet")
REPORT_PATH = Path("reports/prefilter_recall_holdout_validation.txt")


async def label_all(candidates: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    records = []
    agent = build_agent()
    write_lock = asyncio.Lock()
    queue: asyncio.Queue = asyncio.Queue()
    for _, row in candidates.iterrows():
        queue.put_nowait(row)

    progress = {"done": 0, "total": len(candidates)}

    async def worker():
        while True:
            try:
                row = queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            label = await label_paragraph_async(agent, row["paragraph_text"])
            async with write_lock:
                if label is not None:
                    records.append({
                        "paragraph_id": row["paragraph_id"],
                        "accession_number": row["accession_number"],
                        "ticker": row["ticker"],
                        "industry_group": row["industry_group"],
                        "filing_date": row["filing_date"],
                        "section_name": row["section_name"],
                        "stratum": row["stratum"],
                        "paragraph_text": row["paragraph_text"],
                        "llm_is_ai_related": label["is_ai_related"],
                        "llm_rationale": label["rationale"],
                    })
                progress["done"] += 1
                print(f"[{progress['done']}/{progress['total']}] Labeled paragraph {row['paragraph_id']} ({row['stratum']})")
                if len(records) % 20 == 0:
                    pd.DataFrame(records).to_parquet(OUT_PATH, index=False)
            if args.delay > 0:
                await asyncio.sleep(args.delay)
            queue.task_done()

    workers = [asyncio.create_task(worker()) for _ in range(max(1, args.concurrency))]
    await asyncio.gather(*workers)

    labeled_df = pd.DataFrame(records)
    labeled_df.to_parquet(OUT_PATH, index=False)
    return labeled_df


def compute_covered_paragraphs() -> int:
    """Raw paragraph count actually captured by script 06's +/-1 merged windows —
    the TP basis. NOT the chunk count: each chunk merges ~3.5 raw paragraphs on
    average (14,815 covered paragraphs / 4,228 chunks in this corpus), so
    TP = precision * n_chunks would undercount TP relative to FN (which is
    naturally in paragraph units from the sample), making recall look far worse
    than it is. Both sides of the recall fraction must be in the same unit."""
    config = load_config()
    keywords = config["prefiltering"]["ai_keywords"]
    false_positives = config["prefiltering"]["false_positives"]
    ai_regex = build_regex(keywords)

    manifest = pd.read_parquet(Path(config["paths"]["interim_manifests"]) / "filing_manifest.parquet")
    sections = pd.read_parquet(Path(config["paths"]["interim_sections"]) / "filing_sections.parquet")
    matched_acc = set(manifest.loc[manifest["prefilter_status"] == "matched", "accession_number"])

    total_paras = 0
    excluded_paras = 0
    for _, row in sections[sections["accession_number"].isin(matched_acc)].iterrows():
        paragraphs = [p.strip() for p in row["section_text"].split("\n") if p.strip()]
        total_paras += len(paragraphs)
        matched_idx = [i for i, p in enumerate(paragraphs) if ai_regex.search(clean_false_positives(p, false_positives))]
        if not matched_idx:
            excluded_paras += len(paragraphs)
            continue
        windows = [(max(0, i - 1), min(len(paragraphs) - 1, i + 1)) for i in matched_idx]
        merged = []
        for s, e in sorted(windows):
            if not merged or merged[-1][1] < s:
                merged.append([s, e])
            else:
                merged[-1][1] = max(merged[-1][1], e)
        covered = set()
        for s, e in merged:
            covered.update(range(s, e + 1))
        excluded_paras += len(paragraphs) - len(covered)
    return total_paras - excluded_paras


def compute_report(labeled_df: pd.DataFrame) -> str:
    with open(POPULATIONS_PATH) as f:
        populations = json.load(f)

    precision_sample = pd.read_parquet(LABELED_SAMPLE_PATH, columns=["chunk_id", "llm_is_ai_related"])
    precision_sample = precision_sample.drop_duplicates(subset="chunk_id") if "chunk_id" in precision_sample.columns else precision_sample
    precision = float(precision_sample["llm_is_ai_related"].astype(bool).mean())
    covered_paragraphs = compute_covered_paragraphs()
    tp_est = precision * covered_paragraphs

    lines = []
    lines.append("Prefilter/chunking recall — LOCKED HOLDOUT (post-fix keyword list: default + aip/aiops)")
    lines.append("=" * 78)
    lines.append("")
    lines.append("This is the single-look holdout number. The 1,468-paragraph search-pool sample")
    lines.append("(prefilter_recall_validation.txt) is what surfaced the aip/aiops fix and must not")
    lines.append("be cited as the recall estimate — this file is the clean one.")
    lines.append("")
    lines.append(f"Precision on candidate chunks (from {len(precision_sample)}-chunk judged sample): {precision:.3f}")
    lines.append(f"Covered (in-window) paragraphs: {covered_paragraphs:,} — TP basis, NOT the chunk count")
    lines.append(f"  (each chunk merges ~{covered_paragraphs / 4228:.1f} raw paragraphs on average; using the chunk")
    lines.append(f"  count as TP would understate it relative to FN, which is naturally in paragraph units)")
    lines.append(f"Estimated true positives among {covered_paragraphs:,} covered paragraphs: {tp_est:,.0f}")
    lines.append("")

    total_fn_est = 0.0
    for stratum, pop_key in [
        ("no_matches_filing", "no_matches_paragraph_population"),
        ("excluded_within_matched", "excluded_within_matched_paragraph_population"),
    ]:
        strat_df = labeled_df[labeled_df["stratum"] == stratum]
        n = len(strat_df)
        k = int(strat_df["llm_is_ai_related"].sum())
        rate = k / n if n else 0.0
        lo, hi = wilson_interval(k, n)
        population = populations[pop_key]
        fn_est = rate * population
        fn_lo, fn_hi = lo * population, hi * population
        total_fn_est += fn_est

        lines.append(f"Stratum: {stratum}")
        lines.append(f"  Sampled: {n} paragraphs ({strat_df['industry_group'].nunique()} industries)")
        lines.append(f"  Judge-positive (is_ai_related): {k}/{n} = {rate*100:.1f}% "
                     f"(95% Wilson CI: {lo*100:.1f}%-{hi*100:.1f}%)")
        lines.append(f"  Population (paragraphs): {population:,}")
        lines.append(f"  Estimated false negatives: {fn_est:,.0f} (CI: {fn_lo:,.0f}-{fn_hi:,.0f})")
        lines.append("")
        if k > 0:
            for _, r in strat_df[strat_df["llm_is_ai_related"]].iterrows():
                lines.append(f"  [positive] {r['ticker']} ({r['industry_group']}): {r['llm_rationale']}")
            lines.append("")

    recall_est = tp_est / (tp_est + total_fn_est) if (tp_est + total_fn_est) > 0 else float("nan")
    lines.append(f"Estimated overall is_ai_related recall (post-fix, holdout-based): {recall_est:.3f}")
    lines.append(f"  (TP~={tp_est:,.0f}, FN~={total_fn_est:,.0f})")
    lines.append("")
    lines.append("Rule: if this number disappoints, do not re-run this script or re-examine this")
    lines.append("holdout. A disappointing result here should be reported as-is; further keyword-list")
    lines.append("changes require a third, never-seen batch, sampled and judged fresh.")

    return "\n".join(lines)


async def run(args: argparse.Namespace) -> None:
    if REPORT_PATH.exists():
        print(f"Error: {REPORT_PATH} already exists. This holdout has already been looked at ONCE — "
              f"per the single-look rule, this script refuses to run again. If the result needs "
              f"revisiting, sample a third, never-seen batch instead.")
        return
    if not CANDIDATES_PATH.exists():
        print(f"Error: {CANDIDATES_PATH} not found. Run val_10_prefilter_holdout.py first.")
        return

    candidates = pd.read_parquet(CANDIDATES_PATH)
    print(f"Loaded {len(candidates)} fresh holdout candidates to judge (single look).")

    labeled_df = await label_all(candidates, args)
    print(f"\nLabeled {len(labeled_df)} paragraphs total.")

    report = compute_report(labeled_df)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report + "\n")
    print(f"\n{report}")
    print(f"\nLocked report written -> {REPORT_PATH}")

    pipeline_logger.log_event(
        pipeline_step="validation_hardening",
        level="SUCCESS",
        message=f"Prefilter recall HOLDOUT validation complete (locked). Labeled {len(labeled_df)} paragraphs.",
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
