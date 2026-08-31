"""
07_sample_and_label_prefilter.py — Cycle 1, sampling + judging: draw paragraphs
across the full filing universe (both currently AI-flagged and currently
excluded) and get an independent LLM judge's is_ai_related label for each, as
the reward signal for 08_fit_prefilter_harness.py.

This is the first half of harness-optimization cycle 1 (detection keywords —
see docs/meta_harness_plan.md and scripts/harness_fit.py):

    07 (sample + LLM judge)  ->  08 (dev-only search fits ai_keywords,
    evaluates once on holdout, writes the fitted list back to config)

Sampling design, and why weights are recorded:
  - "candidate" stratum: paragraphs already inside a candidate chunk window
    (current prefilter says AI-related) — measures precision.
  - "excluded" stratum: paragraphs the current prefilter does NOT flag —
    measures recall / the search space for new candidate keywords.
  - The draw is HALF/HALF across those strata (so both precision and recall
    get enough labeled rows), proportional by industry within each stratum.
    That balanced draw is deliberate oversampling of the (tiny) candidate
    stratum, so every row records its `sampling_weight` (population count /
    sampled count for its stratum x industry cell); 08 reports
    inverse-probability-weighted metrics next to the raw ones. Without the
    weights, recall and F1 would be overstated relative to the population.
  - The judge sees the FULL paragraph — no truncation.

Usage:
    uv run python scripts/07_sample_and_label_prefilter.py --sample [--n 800] [--seed 42]
    uv run python scripts/07_sample_and_label_prefilter.py --label [--concurrency 3] [--delay 0.3]
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


def paragraph_id(accession_number: str, section_name: str, para_idx: int, text: str) -> str:
    h = hashlib.sha256(f"{accession_number}|{section_name}|{para_idx}|{text}".encode("utf-8")).hexdigest()
    return h[:16]


def matched_window(paragraphs: list[str], ai_regex, false_positives: list[str]) -> set[int]:
    """Paragraph indices covered by script 06's +/-1 merged windows around keyword hits."""
    matched_idx = [i for i, p in enumerate(paragraphs)
                   if ai_regex.search(harness_fit.clean_false_positives(p, false_positives))]
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
    ai_regex = harness_fit.build_regex(keywords)

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
    df = pd.DataFrame(rows)
    df["stratum"] = df["in_candidate_window"].map({True: "candidate", False: "excluded"})
    return df


def do_sample(args: argparse.Namespace) -> None:
    config = load_config()
    population = build_population(config)
    print(f"Full paragraph universe: {len(population)} paragraphs across "
          f"{population['industry_group'].nunique()} industries.")

    candidate_pop = population[population["stratum"] == "candidate"]
    excluded_pop = population[population["stratum"] == "excluded"]
    print(f"  in_candidate_window=True (current prefilter says AI-related): {len(candidate_pop)}")
    print(f"  in_candidate_window=False (excluded by current prefilter): {len(excluded_pop)}")

    n_candidate = args.n // 2
    n_excluded = args.n - n_candidate
    candidate_sample = harness_fit.proportional_by_group(candidate_pop, "industry_group", n_candidate, args.seed)
    excluded_sample = harness_fit.proportional_by_group(excluded_pop, "industry_group", n_excluded, args.seed)

    sample = pd.concat([candidate_sample, excluded_sample], ignore_index=True)
    # The 50/50 stratum draw oversamples the candidate stratum by design;
    # record each row's inverse sampling probability so 08 can weight metrics
    # back to the population.
    sample = harness_fit.attach_sampling_weights(sample, population, ["stratum", "industry_group"])

    print(f"\nSample: {len(candidate_sample)} candidate + {len(excluded_sample)} excluded = {len(sample)} total")
    print(f"Industries covered: {sample['industry_group'].nunique()}")
    print(f"Sampling weights: candidate stratum mean={sample.loc[sample['stratum'] == 'candidate', 'sampling_weight'].mean():.1f}, "
          f"excluded stratum mean={sample.loc[sample['stratum'] == 'excluded', 'sampling_weight'].mean():.1f} "
          f"(population paragraphs represented per labeled row)")

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


def make_prompt(row: pd.Series) -> str:
    # Full paragraph, no truncation — a label for a truncated excerpt is a
    # label for a different document than the harness will see.
    return f"Analyze this excerpt from a corporate 10-K SEC filing:\n\n---\n{row['paragraph_text']}\n---"


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
            print("All paragraphs already labeled.")
            return
    else:
        to_label = candidates
        records = []

    agent = harness_fit.build_judge(PrefilterLabel, SYSTEM_PROMPT)
    labeled_df = await harness_fit.judge_batch(
        agent, to_label, make_prompt, ["is_ai_related", "rationale"],
        records, LABELED_PATH, args.concurrency, args.delay,
    )
    print(f"\nDone. {len(labeled_df)} labeled paragraphs -> {LABELED_PATH}")

    # Auto-generate the human validation workbook from this batch — the
    # to-be-validated sample always exists as soon as labeling finishes.
    agreement_check.auto_make("prefilter")
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
