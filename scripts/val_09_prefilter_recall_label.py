"""
val_09_prefilter_recall_label.py — LLM-judge the prefilter-recall candidates
(val_08's 534 paragraphs, industry-stratified across the two exclusion failure
modes) and estimate the prefilter/chunking funnel's recall on is_ai_related.

Reuses the same independent judge configured via LLM_JUDGE_* (see val_01) so the
judge is never the same model grading its own homework.

Recall estimate:
    TP  ~= precision_on_candidates * n_candidate_chunks
    FN  ~= sum over strata of (observed judge-positive rate in stratum sample)
                              * (stratum's true population size)
    recall = TP / (TP + FN)

precision_on_candidates comes from the existing 236-chunk labeled sample
(llm_labeled_sample__rule_based.parquet, is_ai_related field) — the pipeline
predicts is_ai_related=True for every candidate chunk by construction, so
precision = P(judge agrees | predicted positive).

This is an order-of-magnitude estimate, not an exact count: paragraphs are the
sampling unit here, chunks (merged +/-1 paragraph windows) are the unit precision
was measured on, so FN counts paragraphs, not the windows they'd form if chunked.
Documented explicitly in the output report rather than papered over.

Supports resuming (like val_01): chunks already in the output file are skipped.

Usage:
    uv run python scripts/val_09_prefilter_recall_label.py [--concurrency 1] [--delay 1.0]
"""

import argparse
import asyncio
import json
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

try:
    import pipeline_logger
except ImportError:
    from scripts import pipeline_logger

load_dotenv()

SYSTEM_PROMPT = (
    "You are a financial disclosure analyst specializing in SEC 10-K filings. "
    "Determine whether a short excerpt discusses artificial intelligence, machine "
    "learning, LLMs, or closely related technology in any capacity (adoption, risk, "
    "governance, or promotional mention). Be precise and evidence-based."
)

CANDIDATES_PATH = Path("data/interim/validation/prefilter_recall_candidates.parquet")
POPULATIONS_PATH = Path("data/interim/validation/prefilter_recall_populations.json")
LABELED_SAMPLE_PATH = Path("data/processed/variant_rule_based/validation/llm_labeled_sample__rule_based.parquet")
OUT_PATH = Path("data/interim/validation/prefilter_recall_labeled.parquet")
REPORT_PATH = Path("reports/prefilter_recall_validation.txt")


class RecallLabel(BaseModel):
    is_ai_related: bool = Field(description="Discusses AI, ML, LLMs, or related technology in any capacity.")
    rationale: str = Field(description="One sentence explaining the classification.")


def build_prompt(text: str) -> str:
    return f"Analyze this excerpt from a corporate 10-K SEC filing:\n\n---\n{text[:2500]}\n---"


def build_agent() -> Agent:
    api_key = os.environ.get("LLM_JUDGE_API_KEY")
    if not api_key:
        raise ValueError(
            "LLM_JUDGE_API_KEY is not set. Please configure it in your .env file "
            "(judge model is independent from the main pipeline's NVIDIA_* classifier)."
        )
    base_url = os.environ.get("LLM_JUDGE_BASE_URL", "https://api.openai.com/v1")
    model_name = os.environ.get("LLM_JUDGE_MODEL", "gpt-4o-mini")

    provider = OpenAIProvider(base_url=base_url, api_key=api_key)
    model = OpenAIChatModel(model_name, provider=provider)
    return Agent(
        model,
        output_type=RecallLabel,
        retries=3,
        system_prompt=SYSTEM_PROMPT,
    )


async def label_paragraph_async(agent: Agent, text: str, max_retries: int = 5) -> dict | None:
    retry_delay = 5.0
    for attempt in range(max_retries):
        try:
            result = await agent.run(build_prompt(text))
            return result.output.model_dump(mode="json")
        except Exception as e:
            error_str = str(e)
            is_rate_limit = (
                "429" in error_str
                or "Too Many Requests" in error_str
                or "rate limit" in error_str.lower()
                or getattr(e, "status_code", None) == 429
            )
            if is_rate_limit and attempt < max_retries - 1:
                print(f"  Rate limited. Retrying in {retry_delay}s... (attempt {attempt + 1}/{max_retries})")
                await asyncio.sleep(retry_delay)
                retry_delay *= 2.0
                continue
            print(f"  Judge error: {e}")
            return None
    return None


async def label_all(candidates: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if OUT_PATH.exists():
        existing = pd.read_parquet(OUT_PATH)
        done_ids = set(existing["paragraph_id"].tolist())
        records = existing.to_dict("records")
        to_label = candidates[~candidates["paragraph_id"].isin(done_ids)].reset_index(drop=True)
        print(f"Resuming: {len(done_ids)} already labeled, {len(to_label)} remaining.")
        if len(to_label) == 0:
            return existing
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
            label = await label_paragraph_async(agent, row["paragraph_text"])
            async with write_lock:
                if label is not None:
                    record = {
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
                    }
                    records.append(record)
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


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion — stable at small n/rare k,
    unlike the normal approximation, which matters here since some strata may have
    very few (or zero) observed positives."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half_width = (z * ((p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5)) / denom
    return (max(0.0, center - half_width), min(1.0, center + half_width))


def compute_recall_report(labeled_df: pd.DataFrame) -> str:
    with open(POPULATIONS_PATH) as f:
        populations = json.load(f)

    precision_sample = pd.read_parquet(LABELED_SAMPLE_PATH, columns=["chunk_id", "llm_is_ai_related"])
    precision_sample = precision_sample.drop_duplicates(subset="chunk_id") if "chunk_id" in precision_sample.columns else precision_sample
    precision = float(precision_sample["llm_is_ai_related"].astype(bool).mean())
    # TP basis must be in the SAME unit as FN (paragraphs), not chunks — each chunk
    # merges ~3.5 raw paragraphs on average, so precision * n_chunks understates TP
    # relative to a paragraph-level FN estimate and makes recall look far worse than
    # it is. See val_11_prefilter_holdout_label.py's compute_covered_paragraphs.
    covered_paragraphs = populations["covered_paragraphs"]
    tp_est = precision * covered_paragraphs

    lines = []
    lines.append("Prefilter/chunking recall validation (scripts 05-06) — SEARCH POOL, NOT the final number")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"Precision on candidate chunks (from {len(precision_sample)}-chunk judged sample): {precision:.3f}")
    lines.append(f"Covered (in-window) paragraphs (TP basis, not chunk count): {covered_paragraphs:,}")
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

    recall_est = tp_est / (tp_est + total_fn_est) if (tp_est + total_fn_est) > 0 else float("nan")
    lines.append(f"Estimated overall is_ai_related recall: {recall_est:.3f}")
    lines.append(f"  (TP~={tp_est:,.0f}, FN~={total_fn_est:,.0f})")
    lines.append("")
    lines.append("Caveats:")
    lines.append("- FN is extrapolated from a paragraph-level sample to the full paragraph population;")
    lines.append("  the candidate corpus is measured in chunks (merged +/-1 paragraph windows), so this")
    lines.append("  is an order-of-magnitude estimate, not an exact missed-chunk count.")
    lines.append("- no_matches_filing stratum is a census over all 142 filings (2 paragraphs each) — exact")
    lines.append("  industry coverage, no sampling error on which filings/industries are represented.")
    lines.append("- excluded_within_matched stratum is proportional-by-industry_group sampling (no")
    lines.append("  oversampling of rare industries), matching the holdout design principle used elsewhere")
    lines.append("  in this validation hardening pass.")

    return "\n".join(lines)


async def run(args: argparse.Namespace) -> None:
    if not CANDIDATES_PATH.exists():
        print(f"Error: {CANDIDATES_PATH} not found. Run val_08_prefilter_recall.py first.")
        return
    if not POPULATIONS_PATH.exists():
        print(f"Error: {POPULATIONS_PATH} not found. Run val_08_prefilter_recall.py first.")
        return
    if not LABELED_SAMPLE_PATH.exists():
        print(f"Error: {LABELED_SAMPLE_PATH} not found (needed for candidate-corpus precision).")
        return

    candidates = pd.read_parquet(CANDIDATES_PATH)
    print(f"Loaded {len(candidates)} candidates to judge.")

    labeled_df = await label_all(candidates, args)
    print(f"\nLabeled {len(labeled_df)} paragraphs total.")

    report = compute_recall_report(labeled_df)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report + "\n")
    print(f"\n{report}")
    print(f"\nReport written -> {REPORT_PATH}")

    pipeline_logger.log_event(
        pipeline_step="validation_hardening",
        level="SUCCESS",
        message=f"Prefilter recall validation complete. Labeled {len(labeled_df)} paragraphs.",
        details={"n_labeled": len(labeled_df)},
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", type=int, default=1, help="Number of concurrent judge requests")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay in seconds after each request")
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
