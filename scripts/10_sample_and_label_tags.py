"""
10_sample_and_label_tags.py — Cycle 2, sampling + judging: draw an
industry-stratified sample of AI candidate chunks and get the independent LLM
judge's label on all six classification dimensions for each, as the reward
signal for 11_fit_tag_harness.py.

Second instantiation of the shared harness-optimization cycle
(scripts/harness_fit.py, docs/meta_harness_plan.md):

    10 (sample + LLM judge)  ->  11 (dev-only formula search, single holdout
    look, frozen formulas into config)  ->  12 (tag the corpus)

Sampling design: proportional by industry_group over the chunk population,
with each row's `sampling_weight` (population count / sampled count per
industry cell) recorded at sample time so 11 can report population-weighted
metrics. The judge sees the FULL chunk — no truncation.

Usage:
    uv run python scripts/10_sample_and_label_tags.py --sample [--n 400] [--seed 42]
    uv run python scripts/10_sample_and_label_tags.py --label [--concurrency 3] [--delay 0.3]
"""

import argparse
import asyncio
import json
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field

try:
    import harness_fit
    import pipeline_logger
except ImportError:
    from scripts import harness_fit, pipeline_logger

load_dotenv()

SAMPLE_PATH = Path("data/interim/tag_fit/sample.parquet")
LABELED_PATH = Path("data/interim/tag_fit/labeled.parquet")

SYSTEM_PROMPT = (
    "You are a financial disclosure analyst specializing in SEC 10-K filings. "
    "You will read an excerpt that discusses artificial intelligence and classify it on six "
    "independent dimensions. Judge only what the text itself says, sentence by sentence — "
    "not what you know about the company. Be precise and evidence-based; a chunk can be "
    "true on several dimensions at once, or on none."
)


class TagLabel(BaseModel):
    is_substantive: bool = Field(description=(
        "Describes concrete, operational AI activity the firm itself does or owns: deployed "
        "products/features, model training on own data, AI infrastructure in use, completed "
        "acquisitions. Vague intentions, market commentary, or generic capability claims are NOT substantive."))
    is_promotional: bool = Field(description=(
        "Uses promotional, hype-oriented language about AI: transformative/revolutionary/leader-style "
        "claims, benefits asserted without evidence. Sober risk or factual operational text is NOT promotional."))
    is_risk_related: bool = Field(description=(
        "Discusses risks, harms, uncertainties, or adverse effects connected to AI: regulatory, "
        "competitive, security/privacy, IP, operational, or reputational."))
    is_governance_related: bool = Field(description=(
        "Discusses oversight or control of AI: board/committee oversight, policies, responsible-AI "
        "frameworks, compliance processes, internal controls."))
    is_use_case_specific: bool = Field(description=(
        "Names a specific application or use case for AI (e.g. fraud detection, customer support, "
        "a named product/feature), rather than mentioning AI generically."))
    is_quantified: bool = Field(description=(
        "Attaches numbers to AI claims: dollar amounts, percentages, dated milestones, headcounts, "
        "or other measurable quantities tied to the AI discussion."))
    rationale: str = Field(description="One or two sentences citing the text evidence behind the labels.")


LABEL_FIELDS = ["is_substantive", "is_promotional", "is_risk_related",
                "is_governance_related", "is_use_case_specific", "is_quantified", "rationale"]


def load_config() -> dict:
    with open("configs/config.json") as f:
        return json.load(f)


def do_sample(args: argparse.Namespace) -> None:
    config = load_config()
    chunks = pd.read_parquet(Path(config["paths"]["candidate_chunks"]) / "ai_candidate_chunks.parquet")
    firm_universe = pd.read_parquet(Path(config["paths"]["interim_manifests"]) / "firm_universe.parquet")
    ticker_industry = dict(zip(firm_universe["ticker"], firm_universe["industry_group"]))
    chunks = chunks.copy()
    chunks["industry_group"] = chunks["ticker"].map(ticker_industry)

    print(f"Chunk population: {len(chunks)} candidate chunks across "
          f"{chunks['industry_group'].nunique()} industries.")

    sample = harness_fit.proportional_by_group(chunks, "industry_group", args.n, args.seed)
    sample = harness_fit.attach_sampling_weights(sample, chunks, ["industry_group"])

    print(f"Sample: {len(sample)} chunks, {sample['industry_group'].nunique()} industries "
          f"(mean sampling weight {sample['sampling_weight'].mean():.1f}).")

    SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    sample.to_parquet(SAMPLE_PATH, index=False)
    print(f"\nWrote sample -> {SAMPLE_PATH}")
    print("Not labeled yet. Run with --label to judge these with the LLM.")

    pipeline_logger.log_event(
        pipeline_step="tag_fit",
        level="SUCCESS",
        message=f"Sampled {len(sample)} chunks for the tag fit cycle.",
        details={"n_sampled": len(sample)},
    )


def make_prompt(row: pd.Series) -> str:
    # Full chunk, no truncation.
    return (f"Classify this AI-related excerpt from a corporate 10-K SEC filing "
            f"on all six dimensions:\n\n---\n{row['chunk_text']}\n---")


async def do_label(args: argparse.Namespace) -> None:
    if not SAMPLE_PATH.exists():
        print(f"Error: {SAMPLE_PATH} not found. Run with --sample first.")
        return

    candidates = pd.read_parquet(SAMPLE_PATH)

    if LABELED_PATH.exists():
        existing = pd.read_parquet(LABELED_PATH)
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

    agent = harness_fit.build_judge(TagLabel, SYSTEM_PROMPT)
    labeled_df = await harness_fit.judge_batch(
        agent, to_label, make_prompt, LABEL_FIELDS,
        records, LABELED_PATH, args.concurrency, args.delay,
    )
    print(f"\nDone. {len(labeled_df)} labeled chunks -> {LABELED_PATH}")
    for f in LABEL_FIELDS[:-1]:
        print(f"  {f}: {labeled_df[f'llm_{f}'].astype(bool).mean()*100:.1f}% positive")
    print("Next: scripts/11_fit_tag_harness.py")

    pipeline_logger.log_event(
        pipeline_step="tag_fit",
        level="SUCCESS",
        message=f"Labeled {len(labeled_df)} chunks for the tag fit cycle.",
        details={"n_labeled": len(labeled_df)},
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", action="store_true", help="Build and write the sample (no LLM calls)")
    parser.add_argument("--label", action="store_true", help="Label the sample with the LLM judge")
    parser.add_argument("--n", type=int, default=400, help="Total sample size (--sample)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--delay", type=float, default=0.3)
    args = parser.parse_args()

    if not args.sample and not args.label:
        parser.error("Pass --sample and/or --label")

    if args.sample:
        do_sample(args)
    if args.label:
        asyncio.run(do_label(args))


if __name__ == "__main__":
    main()
