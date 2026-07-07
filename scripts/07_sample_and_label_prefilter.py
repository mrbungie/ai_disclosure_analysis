"""
07_sample_and_label_prefilter.py — Sample paragraphs across the full filing
universe (both currently AI-flagged and currently excluded) and get an
independent LLM judge's is_ai_related label for each, as ground truth for
08_fit_prefilter_harness.py.

This is step 1 of the semi-reproducible prefilter fit cycle:

    07 (sample + LLM label)  ->  08 (CV harness fits ai_keywords against a dev
    split, evaluates once on a held-out split, writes the fitted keyword list)

Population sampled, industry-stratified (proportional, no oversampling — the
harness needs an unbiased estimate of both precision and recall):
  - "candidate" stratum: paragraphs already inside a candidate chunk window
    (current prefilter says AI-related) — measures precision.
  - "excluded" stratum: paragraphs the current prefilter does NOT flag,
    whether because their filing never matched any keyword at all, or because
    they fall outside every keyword hit's +/-1 window — measures recall / the
    search space for new candidate keywords.

Usage:
    uv run python scripts/07_sample_and_label_prefilter.py --sample [--n 800] [--seed 42]
    uv run python scripts/07_sample_and_label_prefilter.py --label [--concurrency 3] [--delay 0.3]
"""

import argparse
import asyncio
import hashlib
import json
import os
import re
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
except ImportError:
    from scripts import pipeline_logger

load_dotenv()

SAMPLE_PATH = Path("data/interim/prefilter_fit/sample.parquet")
LABELED_PATH = Path("data/interim/prefilter_fit/labeled.parquet")

SYSTEM_PROMPT = (
    "You are a financial disclosure analyst specializing in SEC 10-K filings. "
    "Determine whether a short excerpt discusses artificial intelligence, machine "
    "learning, LLMs, or closely related technology in any capacity (adoption, risk, "
    "governance, or promotional mention). Be precise and evidence-based."
)


class PrefilterLabel(BaseModel):
    is_ai_related: bool = Field(description="Discusses AI, ML, LLMs, or related technology in any capacity.")
    rationale: str = Field(description="One sentence explaining the classification.")


def load_config() -> dict:
    with open("configs/config.json") as f:
        return json.load(f)


def build_regex(keywords: list[str]) -> re.Pattern:
    patterns = [r"\b" + re.escape(kw).replace(r"\ ", r"\s+") + r"\b" for kw in keywords]
    return re.compile("|".join(patterns), re.IGNORECASE)


def clean_false_positives(text: str, false_positives: list[str]) -> str:
    for fp in false_positives:
        text = re.sub(r"\b" + re.escape(fp) + r"\b", "", text, flags=re.IGNORECASE)
    return text


def paragraph_id(accession_number: str, section_name: str, para_idx: int, text: str) -> str:
    h = hashlib.sha256(f"{accession_number}|{section_name}|{para_idx}|{text}".encode("utf-8")).hexdigest()
    return h[:16]


def matched_window(paragraphs: list[str], ai_regex: re.Pattern, false_positives: list[str]) -> set[int]:
    """Paragraph indices covered by script 06's +/-1 merged windows around keyword hits."""
    matched_idx = [i for i, p in enumerate(paragraphs) if ai_regex.search(clean_false_positives(p, false_positives))]
    if not matched_idx:
        return set()
    windows = [(max(0, i - 1), min(len(paragraphs) - 1, i + 1)) for i in matched_idx]
    merged: list[list[int]] = []
    for s, e in sorted(windows):
        if not merged or merged[-1][1] < s:
            merged.append([s, e])
        else:
            merged[-1][1] = max(merged[-1][1], e)
    covered: set[int] = set()
    for s, e in merged:
        covered.update(range(s, e + 1))
    return covered


def build_population(config: dict) -> pd.DataFrame:
    """One row per paragraph across the ENTIRE filing universe (not just
    candidate chunks), tagged with the current prefilter's predicted label
    (in_candidate_window) and industry_group, for stratified sampling."""
    keywords = config["prefiltering"]["ai_keywords"]
    false_positives = config["prefiltering"]["false_positives"]
    ai_regex = build_regex(keywords)

    manifest = pd.read_parquet(Path(config["paths"]["interim_manifests"]) / "filing_manifest.parquet")
    sections = pd.read_parquet(Path(config["paths"]["interim_sections"]) / "filing_sections.parquet")
    firm_universe = pd.read_parquet(Path(config["paths"]["interim_manifests"]) / "firm_universe.parquet")
    ticker_industry = dict(zip(firm_universe["ticker"], firm_universe["industry_group"]))

    completed_acc = set(manifest.loc[manifest["parse_status"] == "completed", "accession_number"])
    sections = sections[sections["accession_number"].isin(completed_acc)]

    rows = []
    for _, row in sections.iterrows():
        acc = row["accession_number"]
        ticker = row["ticker"]
        filing_date = row["filing_date"]
        sec_name = row["section_name"]
        paragraphs = [p.strip() for p in row["section_text"].split("\n") if p.strip()]
        if not paragraphs:
            continue
        covered = matched_window(paragraphs, ai_regex, false_positives)
        for i, p in enumerate(paragraphs):
            rows.append({
                "paragraph_id": paragraph_id(acc, sec_name, i, p),
                "accession_number": acc,
                "ticker": ticker,
                "industry_group": ticker_industry.get(ticker),
                "filing_date": filing_date,
                "section_name": sec_name,
                "paragraph_text": p,
                "in_candidate_window": i in covered,
            })
    return pd.DataFrame(rows)


def proportional_by_group(df: pd.DataFrame, group_col: str, n: int, seed: int, min_per_group: int = 1) -> pd.DataFrame:
    """Proportional allocation by group_col (frac = n/N within each group), with
    a floor so small groups aren't zeroed out. No equal-count stratification —
    that biases rare groups the way ensure_dimension_coverage()-style top-ups do."""
    df = df.copy()
    frac = min(1.0, n / len(df)) if len(df) else 0.0

    def sample_group(g):
        target = max(min_per_group, round(len(g) * frac))
        return g.sample(min(target, len(g)), random_state=seed)

    sampled = df.groupby(group_col, group_keys=False).apply(sample_group, include_groups=False)
    sampled = df.loc[sampled.index]
    if len(sampled) > n:
        sampled = sampled.sample(n, random_state=seed)
    return sampled


def do_sample(args: argparse.Namespace) -> None:
    config = load_config()
    population = build_population(config)
    print(f"Full paragraph universe: {len(population)} paragraphs across "
          f"{population['industry_group'].nunique()} industries.")

    candidate_pop = population[population["in_candidate_window"]]
    excluded_pop = population[~population["in_candidate_window"]]
    print(f"  in_candidate_window=True (current prefilter says AI-related): {len(candidate_pop)}")
    print(f"  in_candidate_window=False (excluded by current prefilter): {len(excluded_pop)}")

    n_candidate = args.n // 2
    n_excluded = args.n - n_candidate
    candidate_sample = proportional_by_group(candidate_pop, "industry_group", n_candidate, args.seed)
    excluded_sample = proportional_by_group(excluded_pop, "industry_group", n_excluded, args.seed)

    sample = pd.concat([candidate_sample, excluded_sample], ignore_index=True)
    print(f"\nSample: {len(candidate_sample)} candidate + {len(excluded_sample)} excluded = {len(sample)} total")
    print(f"Industries covered: {sample['industry_group'].nunique()}")

    SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    sample.to_parquet(SAMPLE_PATH, index=False)
    print(f"\nWrote sample -> {SAMPLE_PATH}")
    print("Not labeled yet. Run with --label to judge these with the LLM.")

    pipeline_logger.log_event(
        pipeline_step="prefilter_fit",
        level="SUCCESS",
        message=f"Sampled {len(sample)} paragraphs for prefilter fit (candidate={len(candidate_sample)}, excluded={len(excluded_sample)}).",
        details={"n_candidate": len(candidate_sample), "n_excluded": len(excluded_sample)},
    )


def build_agent() -> Agent:
    api_key = os.environ.get("LLM_JUDGE_API_KEY")
    if not api_key:
        raise ValueError("LLM_JUDGE_API_KEY is not set. Configure it in .env.")
    base_url = os.environ.get("LLM_JUDGE_BASE_URL", "https://api.openai.com/v1")
    model_name = os.environ.get("LLM_JUDGE_MODEL", "gpt-4o-mini")

    provider = OpenAIProvider(base_url=base_url, api_key=api_key)
    # Several OpenAI-"compatible" NIM-hosted models 400 on strict tool-definition
    # mode (extra_forbidden on tools.0.function.strict) — disable it so the judge
    # isn't locked to the handful of models that happen to support it.
    profile = OpenAIModelProfile(openai_supports_strict_tool_definition=False)
    model = OpenAIChatModel(model_name, provider=provider, profile=profile)
    return Agent(model, output_type=PrefilterLabel, retries=3, system_prompt=SYSTEM_PROMPT)


async def label_one(agent: Agent, text: str, max_retries: int = 5) -> dict | None:
    retry_delay = 5.0
    for attempt in range(max_retries):
        try:
            result = await agent.run(f"Analyze this excerpt from a corporate 10-K SEC filing:\n\n---\n{text[:2500]}\n---")
            return result.output.model_dump(mode="json")
        except Exception as e:
            error_str = str(e)
            is_rate_limit = "429" in error_str or "rate limit" in error_str.lower() or getattr(e, "status_code", None) == 429
            if is_rate_limit and attempt < max_retries - 1:
                print(f"  Rate limited. Retrying in {retry_delay}s...")
                await asyncio.sleep(retry_delay)
                retry_delay *= 2.0
                continue
            print(f"  Judge error: {e}")
            return None
    return None


async def do_label(args: argparse.Namespace) -> None:
    if not SAMPLE_PATH.exists():
        print(f"Error: {SAMPLE_PATH} not found. Run with --sample first.")
        return

    candidates = pd.read_parquet(SAMPLE_PATH)

    if LABELED_PATH.exists():
        existing = pd.read_parquet(LABELED_PATH)
        done_ids = set(existing["paragraph_id"].tolist())
        records = existing.to_dict("records")
        to_label = candidates[~candidates["paragraph_id"].isin(done_ids)].reset_index(drop=True)
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
            label = await label_one(agent, row["paragraph_text"])
            async with write_lock:
                if label is not None:
                    record = row.to_dict()
                    record["llm_is_ai_related"] = label["is_ai_related"]
                    record["llm_rationale"] = label["rationale"]
                    records.append(record)
                progress["done"] += 1
                print(f"[{progress['done']}/{progress['total']}] Labeled {row['paragraph_id']} "
                      f"({'candidate' if row['in_candidate_window'] else 'excluded'})")
                if len(records) % 20 == 0:
                    pd.DataFrame(records).to_parquet(LABELED_PATH, index=False)
            if args.delay > 0:
                await asyncio.sleep(args.delay)
            queue.task_done()

    workers = [asyncio.create_task(worker()) for _ in range(max(1, args.concurrency))]
    await asyncio.gather(*workers)

    labeled_df = pd.DataFrame(records)
    labeled_df.to_parquet(LABELED_PATH, index=False)
    print(f"\nDone. {len(labeled_df)} labeled paragraphs -> {LABELED_PATH}")
    print("Next: scripts/08_fit_prefilter_harness.py")

    pipeline_logger.log_event(
        pipeline_step="prefilter_fit",
        level="SUCCESS",
        message=f"Labeled {len(labeled_df)} paragraphs for prefilter fit.",
        details={"n_labeled": len(labeled_df)},
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", action="store_true", help="Build and write the sample (no LLM calls)")
    parser.add_argument("--label", action="store_true", help="Label the sample with the LLM judge")
    parser.add_argument("--n", type=int, default=800, help="Total sample size (--sample)")
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
