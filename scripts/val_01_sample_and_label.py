"""
val_01_sample_and_label.py — Sample chunks and label them via an independent LLM judge

Draws a stratified sample of ~150 chunks from ai_scored_chunks__{variant}.parquet
and classifies each using a pydantic_ai Agent against an OpenAI-compatible endpoint.
Labels are saved to data/processed/variant_{variant}/validation/llm_labeled_sample__{variant}.parquet.

Only supports --variant rule_based (validating llm_full with another LLM judge
would be circular). See variant_utils.require_variant.

The judge model is configured independently from the main pipeline classifier
(script 08) via its own environment variables, so the same model isn't grading its
own homework:

    LLM_JUDGE_API_KEY    (required)
    LLM_JUDGE_BASE_URL   (default: https://api.openai.com/v1)
    LLM_JUDGE_MODEL      (default: gpt-4o-mini)

Stratification: year × section group × is_substantive, so the sample covers
the full feature space rather than being dominated by any single stratum.

Supports resuming: chunks already in the output file are skipped.

Usage:
    uv run python scripts/val_01_sample_and_label.py --variant rule_based [--n 150] [--seed 42] [--concurrency 1] [--delay 1.0]
"""

import argparse
import asyncio
import json
import os
from enum import Enum
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.profiles.openai import OpenAIModelProfile
from pydantic_ai.providers.openai import OpenAIProvider

try:
    import pipeline_logger
    import variant_utils
except ImportError:
    from scripts import pipeline_logger
    from scripts import variant_utils

load_dotenv()

SYSTEM_PROMPT = (
    "You are a financial disclosure analyst specializing in SEC 10-K filings. "
    "Classify text excerpts from corporate annual reports on their AI disclosure "
    "characteristics. Be precise, evidence-based, and consistent."
)


class SpecificityLevel(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


class ValidationLabel(BaseModel):
    is_ai_related: bool = Field(description="Discusses AI, ML, LLMs, or related technology.")
    is_substantive: bool = Field(
        description=(
            "Describes a SPECIFIC operational use, implementation, named tool, or "
            "quantified outcome — NOT a vague mention like 'we use AI to improve operations'."
        )
    )
    is_promotional: bool = Field(
        description="Tone is clearly boosterish or marketing-oriented rather than neutral/factual."
    )
    is_risk_related: bool = Field(
        description="Discusses AI-related risks, uncertainties, or regulatory concerns."
    )
    is_governance_related: bool = Field(
        description="Mentions AI governance, oversight, ethics policy, board involvement, or compliance framework."
    )
    specificity: SpecificityLevel = Field(
        description=(
            "low = vague/generic, medium = some concrete detail, "
            "high = detailed implementation with named tools/metrics/outcomes."
        )
    )
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
    # Many NIM-hosted OpenAI-"compatible" models 400 on OpenAI's strict tool-definition
    # mode (extra_forbidden on tools.0.function.strict) — disable it so the judge isn't
    # locked to the handful of models that happen to support it.
    profile = OpenAIModelProfile(openai_supports_strict_tool_definition=False)
    model = OpenAIChatModel(model_name, provider=provider, profile=profile)
    return Agent(
        model,
        output_type=ValidationLabel,
        retries=3,
        system_prompt=SYSTEM_PROMPT,
    )


async def label_chunk_async(agent: Agent, text: str, max_retries: int = 5) -> dict | None:
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


def stratified_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    df = df.copy()
    df["year"] = pd.to_datetime(df["filing_date"], errors="coerce").dt.year

    def section_group(s: str) -> str:
        s = str(s).lower()
        if "risk" in s:
            return "risk"
        if "7" in s or "mda" in s or "discussion" in s:
            return "mda"
        return "other"

    df["_section_g"] = df["section_name"].apply(section_group)
    df["_subst_g"] = df["is_substantive"].astype(bool)
    df["_stratum"] = (
        df["year"].astype(str) + "_" + df["_section_g"] + "_" + df["_subst_g"].astype(str)
    )

    n_strata = df["_stratum"].nunique()
    per_stratum = max(1, n // n_strata)

    # pandas >=3.0 drops the grouping column from apply()'s result by default
    # (include_groups=False); recover the sampled rows via their original
    # index instead of relying on the grouping column surviving the apply.
    sampled_idx = (
        df.groupby("_stratum", group_keys=False)
        .apply(lambda g: g.sample(min(len(g), per_stratum), random_state=seed), include_groups=False)
        .index
    )
    sampled = df.loc[sampled_idx]

    if len(sampled) < n:
        remaining = df[~df["chunk_id"].isin(sampled["chunk_id"])]
        extra = remaining.sample(min(n - len(sampled), len(remaining)), random_state=seed)
        sampled = pd.concat([sampled, extra])

    result = sampled.sample(min(n, len(sampled)), random_state=seed).reset_index(drop=True)
    result["stratum"] = result["_stratum"]
    return result.drop(columns=["_section_g", "_subst_g", "_stratum"])


def load_config() -> dict:
    with open("configs/config.json") as f:
        return json.load(f)


# Pipeline dimensions to guarantee positive-class coverage for. is_ai_related
# is excluded — it's ~100% prevalent by construction (these are AI candidate
# chunks), so it never needs a top-up.
DIMENSIONS_TO_COVER = ["is_substantive", "is_promotional", "is_risk_related", "is_governance_related"]
MIN_POSITIVES_PER_DIM = 40


def ensure_dimension_coverage(
    df: pd.DataFrame, sample: pd.DataFrame, dims: list[str], min_positives: int, seed: int
) -> pd.DataFrame:
    """Top up `sample` so each dimension in `dims` has at least `min_positives`
    pipeline-flagged positive chunks.

    A single stratified draw over year x section x is_substantive can leave a
    rare dimension (e.g. governance at ~1.4% prevalence pipeline-wide) with too
    few true positives to estimate recall on, even at a few hundred chunks.
    Rather than special-casing one dimension, check every binary dimension and
    force in whatever's missing, so validation coverage isn't blind to
    whichever rule happens to be rarest.
    """
    sample = sample.copy()
    topups = []
    covered_ids = set(sample["chunk_id"])
    for dim in dims:
        if dim not in df.columns:
            continue
        have = int(sample[dim].astype(bool).sum()) if dim in sample.columns else 0
        need = min_positives - have
        if need <= 0:
            continue
        pool = df[df[dim].astype(bool) & ~df["chunk_id"].isin(covered_ids)]
        extra = pool.sample(min(need, len(pool)), random_state=seed).copy()
        if len(extra) == 0:
            continue
        extra["stratum"] = f"{dim}_topup"
        topups.append(extra)
        covered_ids |= set(extra["chunk_id"])
        print(f"  +{len(extra)} forced positive chunks for '{dim}' (had {have}/{min_positives})")

    if topups:
        sample = pd.concat([sample] + topups, ignore_index=True)
    return sample


async def run(args: argparse.Namespace) -> None:
    config = load_config()
    variant = variant_utils.resolve_variant(args.variant, config)
    variant_utils.require_variant(variant, allowed=("rule_based",), script_name="val_01")
    output_root = config.get("variants", {}).get("output_root", "data/processed")

    scored_path = variant_utils.variant_path(variant, "ai_scored_chunks", "parquet", output_root=output_root)
    out_dir = variant_utils.variant_dir(variant, output_root=output_root) / "validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"llm_labeled_sample__{variant}.parquet"

    if not scored_path.exists():
        print(f"Error: {scored_path} not found. Run scripts 07–09 first.")
        return

    load_cols = ["chunk_id", "ticker", "filing_date", "section_name", "chunk_text",
                 "is_substantive"] + [d for d in DIMENSIONS_TO_COVER if d != "is_substantive"]
    df = pd.read_parquet(scored_path, columns=load_cols)
    print(f"Loaded {len(df)} scored chunks.")

    sample = stratified_sample(df, args.n, args.seed)
    sample = ensure_dimension_coverage(df, sample, DIMENSIONS_TO_COVER, MIN_POSITIVES_PER_DIM, args.seed)
    print(f"Sample: {len(sample)} chunks across {sample['stratum'].nunique()} strata.")

    if out_path.exists():
        existing = pd.read_parquet(out_path)
        done_ids = set(existing["chunk_id"].tolist())
        records = existing.to_dict("records")
        to_label = sample[~sample["chunk_id"].isin(done_ids)].reset_index(drop=True)
        print(f"Resuming: {len(done_ids)} already labeled, {len(to_label)} remaining.")
        if len(to_label) == 0:
            print("All chunks already labeled.")
            return
    else:
        to_label = sample
        records = []

    agent = build_agent()
    write_lock = asyncio.Lock()

    async def label_and_record(row) -> dict | None:
        label = await label_chunk_async(agent, row["chunk_text"])
        if label is None:
            return None
        record = {
            "chunk_id": row["chunk_id"],
            "ticker": row.get("ticker"),
            "filing_date": row.get("filing_date"),
            "section_name": row.get("section_name"),
            "stratum": row.get("stratum"),
            "chunk_text": row["chunk_text"],
        }
        record.update({f"llm_{k}": v for k, v in label.items()})
        return record

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
            record = await label_and_record(row)
            async with write_lock:
                if record is not None:
                    records.append(record)
                progress["done"] += 1
                print(f"[{progress['done']}/{progress['total']}] Labeled chunk {row['chunk_id']}")
                if len(records) % 10 == 0:
                    pd.DataFrame(records).to_parquet(out_path, index=False)
            if args.delay > 0:
                await asyncio.sleep(args.delay)
            queue.task_done()

    workers = [asyncio.create_task(worker()) for _ in range(max(1, args.concurrency))]
    await asyncio.gather(*workers)

    pd.DataFrame(records).to_parquet(out_path, index=False)
    pipeline_logger.log_event(
        pipeline_step="validation_labeling",
        level="SUCCESS",
        message=f"Labeled {len(records)} chunks. Saved to {out_path}",
        details={"variant": variant},
    )
    print(f"\nDone. {len(records)} labeled chunks → {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=150, help="Target sample size (default: 150)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--concurrency", type=int, default=1, help="Number of concurrent judge requests")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay in seconds after each request")
    variant_utils.add_variant_arg(parser)
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
