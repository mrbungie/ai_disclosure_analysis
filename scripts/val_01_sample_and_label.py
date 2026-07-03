"""
val_01_sample_and_label.py — Sample chunks and label them via Claude Code Agent SDK

Draws a stratified sample of ~150 chunks from ai_scored_chunks.parquet and classifies
each using the Claude Code Agent SDK (claude-code-sdk). Labels are saved to
data/processed/validation/llm_labeled_sample.parquet.

Stratification: year × section group × is_substantive, so the sample covers
the full feature space rather than being dominated by any single stratum.

Supports resuming: chunks already in the output file are skipped.

Usage:
    uv run python scripts/val_01_sample_and_label.py [--n 150] [--seed 42]
"""

import argparse
import asyncio
import json
import re
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
from claude_code_sdk import query, ClaudeCodeOptions
from claude_code_sdk.types import AssistantMessage, TextBlock, ResultMessage

try:
    import pipeline_logger
except ImportError:
    from scripts import pipeline_logger

load_dotenv()

SYSTEM_PROMPT = (
    "You are a financial disclosure analyst specializing in SEC 10-K filings. "
    "Classify text excerpts from corporate annual reports on their AI disclosure "
    "characteristics. Be precise, evidence-based, and consistent. "
    "Respond only with a valid JSON object — no markdown fences, no explanation outside the JSON."
)


def build_prompt(text: str) -> str:
    return f"""{SYSTEM_PROMPT}

Analyze this excerpt from a corporate 10-K SEC filing:

---
{text[:2500]}
---

Return a JSON object with exactly these keys:

{{
  "is_ai_related": true/false,
  "is_substantive": true/false,
  "is_promotional": true/false,
  "is_risk_related": true/false,
  "is_governance_related": true/false,
  "specificity": "low" | "medium" | "high",
  "rationale": "<one sentence>"
}}

Definitions:
- is_ai_related: discusses AI, ML, LLMs, or related technology
- is_substantive: describes a SPECIFIC operational use, implementation, named tool, or quantified outcome — NOT a vague mention like "we use AI to improve operations"
- is_promotional: tone is clearly boosterish or marketing-oriented rather than neutral/factual
- is_risk_related: discusses AI-related risks, uncertainties, or regulatory concerns
- is_governance_related: mentions AI governance, oversight, ethics policy, board involvement, or compliance framework
- specificity: low = vague/generic, medium = some concrete detail, high = detailed implementation with named tools/metrics/outcomes"""


_QUERY_OPTIONS = ClaudeCodeOptions(
    model="claude-sonnet-4-6",
    max_turns=1,
    allowed_tools=[],  # pure text response, no tool use
)


async def label_chunk_async(text: str) -> dict | None:
    collected = []
    try:
        async for message in query(
            prompt=build_prompt(text),
            options=_QUERY_OPTIONS,
        ):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        collected.append(block.text)
            elif isinstance(message, ResultMessage) and message.result:
                collected.append(message.result)
    except Exception as e:
        err = str(e)
        # rate_limit_event is an unrecognised event type in this SDK version — not fatal
        if "rate_limit_event" in err or "Unknown message type" in err:
            pass
        else:
            print(f"  Agent SDK error: {e}")

    if not collected:
        return None

    raw = "".join(collected).strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"  JSON parse error: {e} — raw: {raw[:150]}")
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

    sampled = (
        df.groupby("_stratum", group_keys=False)
        .apply(lambda g: g.sample(min(len(g), per_stratum), random_state=seed))
    )

    if len(sampled) < n:
        remaining = df[~df["chunk_id"].isin(sampled["chunk_id"])]
        extra = remaining.sample(min(n - len(sampled), len(remaining)), random_state=seed)
        sampled = pd.concat([sampled, extra])

    result = sampled.sample(min(n, len(sampled)), random_state=seed).reset_index(drop=True)
    result["stratum"] = result["_stratum"]
    return result.drop(columns=["_section_g", "_subst_g", "_stratum"])


async def run(args: argparse.Namespace) -> None:
    scored_path = Path("data/interim/candidate_chunks/ai_scored_chunks.parquet")
    out_dir = Path("data/processed/validation")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "llm_labeled_sample.parquet"

    if not scored_path.exists():
        print(f"Error: {scored_path} not found. Run scripts 07–09 first.")
        return

    load_cols = ["chunk_id", "ticker", "filing_date", "section_name", "chunk_text", "is_substantive"]
    df = pd.read_parquet(scored_path, columns=load_cols)
    print(f"Loaded {len(df)} scored chunks.")

    sample = stratified_sample(df, args.n, args.seed)
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

    for i, (_, row) in enumerate(tqdm(to_label.iterrows(), total=len(to_label), desc="Labeling")):
        label = await label_chunk_async(row["chunk_text"])
        if label is None:
            continue
        # small pause every 5 chunks to stay within rate limits
        if i > 0 and i % 5 == 0:
            await asyncio.sleep(1)

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

        if len(records) % 10 == 0:
            pd.DataFrame(records).to_parquet(out_path, index=False)

    pd.DataFrame(records).to_parquet(out_path, index=False)
    pipeline_logger.log_event(
        pipeline_step="validation_labeling",
        level="SUCCESS",
        message=f"Labeled {len(records)} chunks. Saved to {out_path}",
    )
    print(f"\nDone. {len(records)} labeled chunks → {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=150, help="Target sample size (default: 150)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
