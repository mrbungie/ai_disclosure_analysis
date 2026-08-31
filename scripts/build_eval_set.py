"""
build_eval_set.py — Build one labeled evaluation set for the classification
harness (the task instances X and their reward labels).

    build_eval_set --sample   draw paragraphs from the full corpus
    build_eval_set --label    obtain reward labels (LLM labeler) + export the
                              human audit workbook
    -> data/interim/eval/eval_set.parquet  (one row per instance: text,
       7 llm_* label columns, sampling_weight, split ∈ {search, test})

Design, mirroring the meta-harness setup:
  - Instances are PARAGRAPHS drawn from the entire corpus. Stratified by the
    ACTIVE candidate's own is_ai_related prediction (half predicted-positive,
    half predicted-negative, industry-proportional within each) so both error
    directions get labeled rows; every row records its inverse sampling
    probability (`sampling_weight`) so population-weighted metrics stay
    honest despite the balanced draw.
  - The search/test split is assigned HERE, once, at labeling time, and
    stored in the parquet. The proposer spends the search rows freely;
    eval_harness.py enforces the single look at the test rows.
  - Reward labels come from an LLM labeler (LLM_JUDGE_* env vars) over the
    FULL text, all seven fields in one call; a balanced human audit workbook
    is exported automatically at the end (agreement_check.py scores it).

Usage:
    uv run python scripts/build_eval_set.py --sample [--n 800] [--seed 42]
    uv run python scripts/build_eval_set.py --label [--concurrency 3] [--delay 0.3]
"""

import argparse
import asyncio
import hashlib
import json
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field

try:
    import agreement_check
    import harness_fit
    import pipeline_logger
except ImportError:
    from scripts import agreement_check, harness_fit, pipeline_logger

load_dotenv()

SAMPLE_PATH = Path("data/interim/eval/sample.parquet")
EVAL_SET_PATH = Path("data/interim/eval/eval_set.parquet")
HARNESSES_DIR = Path("harnesses")

LABEL_FIELDS = ["is_ai_related", "is_substantive", "is_promotional", "is_risk_related",
                "is_governance_related", "is_use_case_specific", "is_quantified"]

SYSTEM_PROMPT = (
    "You are a financial disclosure analyst specializing in SEC 10-K filings. "
    "Classify a short excerpt on seven independent binary dimensions about artificial "
    "intelligence and related technology. Judge only what the text itself says. "
    "If the excerpt does not discuss AI/ML/LLMs or closely related technology at all, "
    "is_ai_related is false and every other dimension is false too."
)


class RewardLabel(BaseModel):
    is_ai_related: bool = Field(description="Discusses AI, ML, LLMs, or closely related technology in any capacity.")
    is_substantive: bool = Field(description="Concrete, operational AI activity the firm itself does or owns (deployed products, training on own data, infrastructure in use, completed acquisitions). Vague intentions or market commentary are NOT substantive.")
    is_promotional: bool = Field(description="Promotional, hype-oriented AI language: transformative/revolutionary/leader-style claims, benefits asserted without evidence.")
    is_risk_related: bool = Field(description="Risks, harms, uncertainties, or adverse effects connected to AI.")
    is_governance_related: bool = Field(description="Oversight or control of AI: board/committee oversight, policies, responsible-AI frameworks, compliance.")
    is_use_case_specific: bool = Field(description="Names a specific application or use case for AI rather than mentioning it generically.")
    is_quantified: bool = Field(description="Numbers attached to the AI discussion: dollar amounts, percentages, dated milestones, measurable quantities.")
    rationale: str = Field(description="One sentence citing the text evidence.")


def load_config() -> dict:
    with open("configs/config.json") as f:
        return json.load(f)


def load_active_harness():
    """The ACTIVE detection candidate stratifies the draw (its own
    predicted-positive/negative split)."""
    name = harness_fit.active_candidate("detection")
    print(f"Active detection candidate for stratification: {name}")
    return harness_fit.load_candidate("detection", name)


def paragraph_id(accession_number: str, section_name: str, para_idx: int, text: str) -> str:
    h = hashlib.sha256(f"{accession_number}|{section_name}|{para_idx}|{text}".encode("utf-8")).hexdigest()
    return h[:16]


def build_population(config: dict, classify) -> pd.DataFrame:
    """One row per paragraph across the entire parsed corpus, tagged with the
    active candidate's is_ai_related prediction and the firm's industry."""
    manifest = pd.read_parquet(Path(config["paths"]["interim_manifests"]) / "filing_manifest.parquet")
    sections = pd.read_parquet(Path(config["paths"]["interim_sections"]) / "filing_sections.parquet")
    firm_universe = pd.read_parquet(Path(config["paths"]["interim_manifests"]) / "firm_universe.parquet")
    ticker_industry = dict(zip(firm_universe["ticker"], firm_universe["industry_group"]))

    completed_acc = set(manifest.loc[manifest["parse_status"] == "completed", "accession_number"])
    sections = sections[sections["accession_number"].isin(completed_acc)]

    rows = []
    for _, row in sections.iterrows():
        paragraphs = [p.strip() for p in row["section_text"].split("\n") if p.strip()]
        for i, p in enumerate(paragraphs):
            rows.append({
                "paragraph_id": paragraph_id(row["accession_number"], row["section_name"], i, p),
                "accession_number": row["accession_number"],
                "ticker": row["ticker"],
                "industry_group": ticker_industry.get(row["ticker"]),
                "filing_date": row["filing_date"],
                "section_name": row["section_name"],
                "paragraph_text": p,
            })
    df = pd.DataFrame(rows)
    df["stratum"] = ["pred_positive" if classify(t) else "pred_negative"
                     for t in df["paragraph_text"]]
    return df


def do_sample(args: argparse.Namespace) -> None:
    config = load_config()
    classify = load_active_harness()
    population = build_population(config, classify)
    pos = population[population["stratum"] == "pred_positive"]
    neg = population[population["stratum"] == "pred_negative"]
    print(f"Corpus: {len(population)} paragraphs — active candidate predicts "
          f"{len(pos)} AI-related ({len(pos)/len(population)*100:.2f}%).")

    n_pos = args.n // 2
    pos_sample = harness_fit.proportional_by_group(pos, "industry_group", n_pos, args.seed)
    neg_sample = harness_fit.proportional_by_group(neg, "industry_group", args.n - n_pos, args.seed)
    sample = pd.concat([pos_sample, neg_sample], ignore_index=True)
    sample = harness_fit.attach_sampling_weights(sample, population, ["stratum", "industry_group"])

    # The search/test split is fixed NOW, before any labels exist, stratified
    # by the sampling stratum. eval_harness.py enforces the one test look.
    _, test = harness_fit.stratified_split(sample, "stratum", args.search_frac, args.seed)
    sample["split"] = "search"
    sample.loc[sample["paragraph_id"].isin(test["paragraph_id"]), "split"] = "test"

    print(f"Sample: {len(pos_sample)} pred-positive + {len(neg_sample)} pred-negative = {len(sample)} "
          f"({(sample['split'] == 'search').sum()} search / {(sample['split'] == 'test').sum()} test)")
    SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    sample.to_parquet(SAMPLE_PATH, index=False)
    print(f"Wrote sample -> {SAMPLE_PATH}\nNot labeled yet. Run with --label.")

    pipeline_logger.log_event(pipeline_step="eval_set", level="SUCCESS",
                              message=f"Sampled {len(sample)} paragraphs for the harness eval set.",
                              details={"n": len(sample)})


def make_prompt(row: pd.Series) -> str:
    return (f"Classify this excerpt from a corporate 10-K SEC filing on all seven "
            f"dimensions:\n\n---\n{row['paragraph_text']}\n---")


async def do_label(args: argparse.Namespace) -> None:
    if not SAMPLE_PATH.exists():
        print(f"Error: {SAMPLE_PATH} not found. Run with --sample first.")
        return
    candidates = pd.read_parquet(SAMPLE_PATH)
    if EVAL_SET_PATH.exists():
        existing = pd.read_parquet(EVAL_SET_PATH)
        done = set(existing["paragraph_id"])
        records = existing.to_dict("records")
        to_label = candidates[~candidates["paragraph_id"].isin(done)].reset_index(drop=True)
        print(f"Resuming: {len(done)} labeled, {len(to_label)} remaining.")
        if not len(to_label):
            print("Eval set already fully labeled.")
            return
    else:
        to_label, records = candidates, []

    agent = harness_fit.build_judge(RewardLabel, SYSTEM_PROMPT)
    labeled = await harness_fit.judge_batch(
        agent, to_label, make_prompt, LABEL_FIELDS + ["rationale"],
        records, EVAL_SET_PATH, args.concurrency, args.delay)
    print(f"\nDone. {len(labeled)} labeled instances -> {EVAL_SET_PATH}")
    for f in LABEL_FIELDS:
        print(f"  {f}: {labeled[f'llm_{f}'].astype(bool).mean()*100:.1f}% positive")

    agreement_check.auto_make("eval")
    print("Next: scripts/eval_harness.py --task detection --candidate 000_seed")

    pipeline_logger.log_event(pipeline_step="eval_set", level="SUCCESS",
                              message=f"Labeled {len(labeled)} eval instances.",
                              details={"n": len(labeled)})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--label", action="store_true")
    parser.add_argument("--n", type=int, default=800)
    parser.add_argument("--search-frac", type=float, default=0.7)
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
