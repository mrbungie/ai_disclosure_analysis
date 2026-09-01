"""
build_eval_set.py — Build one stage's labeled evaluation set (the task
instances X and their reward labels), per docs/distillation_map.html.

    build_eval_set --stage detection --sample
    build_eval_set --stage detection --label
    build_eval_set --stage classification --sample   # only after detection is frozen
    build_eval_set --stage classification --label

Two stages, two populations, two reference judges — never fused (map §1,
§3, §6):

  detection (stage 1)
    Population: every corpus paragraph, stratified by the lexical seed
    screen (scripts/seed_screen.py) into hit / no_hit_filing_hits /
    no_hit_filing_clean (map §2). Search sample is weighted toward weak
    hits and no-hit-in-a-hitting-filing; holdout is a probability sample,
    stratified by year and sector, with guaranteed coverage of all three
    strata (including the no-hit-anywhere stratum). Reference judge scores
    ONLY the scope question: is_ai_related.
    -> data/interim/eval/eval_set_detection.parquet (one row per paragraph)

  classification (stage 2)
    Cannot start until harnesses/detection/ACTIVE has produced the
    candidate frame (data/processed/candidate_frame.parquet via
    scripts/apply_harness.py). Population: chunks built around admitted
    paragraphs (harness_fit.build_chunks) so the reference judge sees
    enough context to apply the codebook. Reference judge scores ONLY the
    six dimensions, per chunk.
    -> data/interim/eval/eval_set_classification.parquet (one row per
       chunk, with member_paragraph_ids for the paragraph-level rollout)

Both stages: the search/test split is assigned HERE, once, at sampling
time, and stored in the parquet — the proposer spends search rows freely;
eval_harness.py enforces the single test look per stage.
"""

import argparse
import asyncio
import json
from pathlib import Path

import numpy as np
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

SEED_SCREEN_PATH = Path("data/interim/seed_screen/seed_screen.parquet")
CANDIDATE_FRAME_PATH = Path("data/processed/candidate_frame.parquet")
SAMPLE_PATHS = {
    "detection": Path("data/interim/eval/sample_detection.parquet"),
    "classification": Path("data/interim/eval/sample_classification.parquet"),
}
EVAL_SET_PATHS = {
    "detection": Path("data/interim/eval/eval_set_detection.parquet"),
    "classification": Path("data/interim/eval/eval_set_classification.parquet"),
}

DIMENSION_FIELDS = ["is_substantive", "is_promotional", "is_risk_related",
                    "is_governance_related", "is_use_case_specific", "is_quantified"]


class ScopeLabel(BaseModel):
    is_ai_related: bool = Field(description="Discusses AI, ML, LLMs, or closely related technology in any capacity.")
    rationale: str = Field(description="One sentence citing the text evidence.")


class DimensionLabel(BaseModel):
    is_substantive: bool = Field(description="Concrete, operational AI activity the firm itself does or owns (deployed products, training on own data, infrastructure in use, completed acquisitions). Vague intentions or market commentary are NOT substantive.")
    is_promotional: bool = Field(description="Promotional, hype-oriented AI language: transformative/revolutionary/leader-style claims, benefits asserted without evidence.")
    is_risk_related: bool = Field(description="Risks, harms, uncertainties, or adverse effects connected to AI.")
    is_governance_related: bool = Field(description="Oversight or control of AI: board/committee oversight, policies, responsible-AI frameworks, compliance.")
    is_use_case_specific: bool = Field(description="Names a specific application or use case for AI rather than mentioning it generically.")
    is_quantified: bool = Field(description="Numbers attached to the AI discussion: dollar amounts, percentages, dated milestones, measurable quantities.")
    rationale: str = Field(description="One sentence citing the text evidence.")


SCOPE_SYSTEM_PROMPT = (
    "You are a financial disclosure analyst specializing in SEC 10-K filings. "
    "Decide whether a short excerpt discusses AI, ML, LLMs, or closely related technology "
    "in any capacity. Judge only what the text itself says."
)

DIMENSION_SYSTEM_PROMPT = (
    "You are a financial disclosure analyst specializing in SEC 10-K filings. This excerpt "
    "has already been judged to discuss AI in some capacity. Classify it on six independent "
    "binary dimensions about that AI discussion. Judge only what the text itself says."
)


def load_config() -> dict:
    with open("configs/config.json") as f:
        return json.load(f)


def year_of(filing_date) -> str:
    return str(pd.Timestamp(filing_date).year)


# ---------------------------------------------------------------------------
# Stage 1: detection
# ---------------------------------------------------------------------------


SEARCH_STRATUM_FRACS = {"hit": 0.40, "no_hit_filing_hits": 0.40, "no_hit_filing_clean": 0.20}

# Disagreement quadrants (scripts/seed_screen.py: agreement_quadrant, only
# populated when phase0 is enabled) are the most informative rows for the
# proposer to see — lex_high_emb_low are probable lexical false positives,
# lex_low_emb_high are probable lexical false negatives (exactly what
# phase0 exists to surface). Open/tunable, not load-bearing science.
DISAGREEMENT_QUADRANT_BOOST = 3.0
DISAGREEMENT_QUADRANTS = ["lex_high_emb_low", "lex_low_emb_high"]


def _search_weights(pool: pd.DataFrame, stratum: str) -> pd.Series | None:
    """Sampling weights for the SEARCH draw only — never the holdout, which
    must stay an unweighted probability sample. Generalizes the existing
    hit_count reweighting (favor weak lexical hits) with an orthogonal
    boost for the two lexical x semantic disagreement quadrants, when that
    signal is available (configs/config.json: phase0.enabled)."""
    w = 1.0 / pool["hit_count"].clip(lower=1) if stratum == "hit" else pd.Series(1.0, index=pool.index)
    if "agreement_quadrant" in pool.columns and pool["agreement_quadrant"].notna().any():
        w = w * np.where(pool["agreement_quadrant"].isin(DISAGREEMENT_QUADRANTS),
                         DISAGREEMENT_QUADRANT_BOOST, 1.0)
    return w if (w != 1.0).any() else None


def do_sample_detection(args: argparse.Namespace) -> None:
    """Accumulative and index-based: never redraws or discards a row already
    in the sample. The search split is topped up to meet its per-stratum
    quotas (SEARCH_STRATUM_FRACS) by adding only the shortfall as NEW rows —
    already-sampled (and possibly already-labeled) search rows are kept
    as-is even if their stratum already exceeds its quota, so a re-run never
    throws away paid labeling work. The holdout is drawn once and never
    touched again after that: it's the probability sample opened a single
    time at freeze, so redrawing it after some of it may already be labeled
    would break that guarantee."""
    if not SEED_SCREEN_PATH.exists():
        print(f"Error: {SEED_SCREEN_PATH} not found. Run scripts/seed_screen.py first.")
        return
    population = pd.read_parquet(SEED_SCREEN_PATH)
    counts = population["stratum"].value_counts()
    print(f"Seed-screen population: {len(population)} paragraphs — " +
          ", ".join(f"{s}={n}" for s, n in counts.items()))

    sample_path = SAMPLE_PATHS["detection"]
    existing = pd.read_parquet(sample_path) if sample_path.exists() else None
    existing_ids = set(existing["paragraph_id"]) if existing is not None else set()
    existing_search = existing[existing["split"] == "search"] if existing is not None else population.iloc[0:0]
    existing_holdout = existing[existing["split"] == "test"] if existing is not None else population.iloc[0:0]

    new_search_parts = []
    for stratum, frac in SEARCH_STRATUM_FRACS.items():
        target = round(args.n_search * frac)
        have = int((existing_search["stratum"] == stratum).sum())
        need = max(0, target - have)
        pool = population[(population["stratum"] == stratum) & (~population["paragraph_id"].isin(existing_ids))]
        need = min(need, len(pool))
        if need == 0:
            continue
        weights = _search_weights(pool, stratum)
        draw = pool.sample(n=need, weights=weights, random_state=args.seed).copy()
        draw["sampling_weight"] = pd.NA
        draw["split"] = "search"
        new_search_parts.append(draw)
        existing_ids |= set(draw["paragraph_id"])

    if new_search_parts:
        added_search = pd.concat(new_search_parts, ignore_index=True)
        print(f"Topping up search sample: +{len(added_search)} new rows "
              f"({dict(added_search['stratum'].value_counts())}); "
              f"{len(existing_search)} existing search rows kept untouched.")
    else:
        added_search = population.iloc[0:0]
        print(f"Search sample already meets its per-stratum quotas ({len(existing_search)} rows) — nothing to add.")

    if len(existing_holdout):
        holdout = existing_holdout
        print(f"Holdout already drawn ({len(holdout)} rows) — never redrawn once it exists "
              f"(it's the single-look probability sample).")
    else:
        population["_year"] = population["filing_date"].map(year_of)
        holdout_pool = population[~population["paragraph_id"].isin(existing_ids)]
        holdout = harness_fit.stratified_holdout_with_coverage(
            holdout_pool, population, "stratum", ["_year", "industry_group"],
            args.n_holdout, args.seed)
        holdout = holdout.drop(columns="_year", errors="ignore")
        holdout["split"] = "test"
        missing_strata = set(population["stratum"].unique()) - set(holdout["stratum"].unique())
        if missing_strata:
            print(f"WARNING: holdout is missing strata {missing_strata} — "
                  f"increase --n-holdout for full coverage.")
        print(f"Drew holdout: {len(holdout)} rows ({dict(holdout['stratum'].value_counts())}).")

    sample = pd.concat([existing_search, added_search, holdout], ignore_index=True)
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_parquet(sample_path, index=False)
    n_search_total = int((sample["split"] == "search").sum())
    n_holdout_total = int((sample["split"] == "test").sum())
    print(f"Sample: {n_search_total} search + {n_holdout_total} holdout = {len(sample)} -> {sample_path}")
    print("Run with --stage detection --label to label any unlabeled rows.")

    pipeline_logger.log_event(pipeline_step="eval_set_detection", level="SUCCESS",
                              message=f"Sampled {len(sample)} paragraphs for the detection eval set "
                                      f"(+{len(added_search)} new search rows).",
                              details={"n": len(sample), "added": len(added_search)})


async def do_label_detection(args: argparse.Namespace) -> None:
    await _label_stage(
        stage="detection", sample_path=SAMPLE_PATHS["detection"], eval_set_path=EVAL_SET_PATHS["detection"],
        label_fields=["is_ai_related"], output_type=ScopeLabel, system_prompt=SCOPE_SYSTEM_PROMPT,
        text_col="paragraph_text", args=args)


# ---------------------------------------------------------------------------
# Stage 2: classification
# ---------------------------------------------------------------------------


def do_sample_classification(args: argparse.Namespace) -> None:
    if not CANDIDATE_FRAME_PATH.exists():
        print(f"Error: {CANDIDATE_FRAME_PATH} not found. Stage 1 must be frozen and applied "
              f"(scripts/apply_harness.py) before stage 2 can sample its candidate frame.")
        return
    config = load_config()
    all_paragraphs = harness_fit.flatten_corpus_paragraphs(config)
    admitted = pd.read_parquet(CANDIDATE_FRAME_PATH)
    all_paragraphs["is_ai_related"] = all_paragraphs["paragraph_id"].isin(admitted["paragraph_id"])
    n_admitted = int(all_paragraphs["is_ai_related"].sum())
    print(f"Candidate frame: {n_admitted} admitted paragraphs of {len(all_paragraphs)} "
          f"({n_admitted / len(all_paragraphs) * 100:.2f}%).")

    chunks, membership = harness_fit.build_chunks(
        all_paragraphs, admit_col="is_ai_related", window=args.chunk_window)
    print(f"Built {len(chunks)} chunks from {membership['paragraph_id'].nunique()} admitted paragraphs.")

    firm_universe = pd.read_parquet(Path(config["paths"]["interim_manifests"]) / "firm_universe.parquet")
    ticker_industry = dict(zip(firm_universe["ticker"], firm_universe["industry_group"]))
    chunks["industry_group"] = chunks["ticker"].map(ticker_industry)
    chunks["_year"] = chunks["filing_date"].map(year_of)

    n = min(args.n, len(chunks))
    search_n = round(n * args.search_frac)
    search = chunks.sample(n=min(search_n, len(chunks)), random_state=args.seed)
    holdout_pool = chunks[~chunks["chunk_id"].isin(search["chunk_id"])]
    holdout = harness_fit.proportional_by_group(
        holdout_pool, "industry_group", min(n - len(search), len(holdout_pool)), args.seed)
    holdout = harness_fit.attach_sampling_weights(holdout, chunks, ["industry_group"])

    search = search.copy()
    search["sampling_weight"] = pd.NA
    search["split"] = "search"
    holdout["split"] = "test"
    sample = pd.concat([search, holdout], ignore_index=True).drop(columns="_year", errors="ignore")

    print(f"Sample: {len(search)} search + {len(holdout)} holdout chunks = {len(sample)}")
    SAMPLE_PATHS["classification"].parent.mkdir(parents=True, exist_ok=True)
    sample.to_parquet(SAMPLE_PATHS["classification"], index=False)
    print(f"Wrote sample -> {SAMPLE_PATHS['classification']}\nNot labeled yet. Run with --stage classification --label.")

    pipeline_logger.log_event(pipeline_step="eval_set_classification", level="SUCCESS",
                              message=f"Sampled {len(sample)} chunks for the classification eval set.",
                              details={"n": len(sample)})


async def do_label_classification(args: argparse.Namespace) -> None:
    await _label_stage(
        stage="classification", sample_path=SAMPLE_PATHS["classification"],
        eval_set_path=EVAL_SET_PATHS["classification"], label_fields=DIMENSION_FIELDS,
        output_type=DimensionLabel, system_prompt=DIMENSION_SYSTEM_PROMPT,
        text_col="chunk_text", args=args)


# ---------------------------------------------------------------------------
# Shared labeling loop
# ---------------------------------------------------------------------------


async def _label_stage(stage: str, sample_path: Path, eval_set_path: Path, label_fields: list[str],
                       output_type, system_prompt: str, text_col: str, args: argparse.Namespace) -> None:
    """The sample is fixed once by --sample; --label only ever labels what's
    still missing from eval_set_path, so a failed/interrupted run and a
    later re-run accumulate onto the same fixed sample instead of starting
    over or drawing a new one."""
    if not sample_path.exists():
        print(f"Error: {sample_path} not found. Run with --stage {stage} --sample first.")
        return
    id_col = "paragraph_id" if stage == "detection" else "chunk_id"
    candidates = pd.read_parquet(sample_path)
    if eval_set_path.exists():
        existing = pd.read_parquet(eval_set_path)
        done = set(existing[id_col])
        records = existing.to_dict("records")
        to_label = candidates[~candidates[id_col].isin(done)].reset_index(drop=True)
        print(f"Resuming: {len(done)} labeled, {len(to_label)} remaining.")
        if not len(to_label):
            print("Eval set already fully labeled.")
            return
    else:
        to_label, records = candidates, []

    agent = harness_fit.build_judge(output_type, system_prompt)
    labeled = await harness_fit.judge_batch(
        agent, to_label, text_col, label_fields + ["rationale"],
        records, eval_set_path, args.batch_size, args.delay)
    print(f"\nDone. {len(labeled)} labeled instances -> {eval_set_path}")
    for f in label_fields:
        print(f"  {f}: {labeled[f'llm_{f}'].astype(bool).mean() * 100:.1f}% positive")

    agreement_check.auto_make(f"eval_{stage}")
    print(f"Next: scripts/eval_harness.py --task {stage} --candidate 000_seed")

    pipeline_logger.log_event(pipeline_step=f"eval_set_{stage}", level="SUCCESS",
                              message=f"Labeled {len(labeled)} {stage} eval instances.",
                              details={"n": len(labeled)})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["detection", "classification"], required=True)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--label", action="store_true")
    parser.add_argument("--n", type=int, default=800, help="classification: total chunks to sample")
    parser.add_argument("--n-search", type=int, default=560, help="detection: search sample size")
    parser.add_argument("--n-holdout", type=int, default=240, help="detection: holdout sample size")
    parser.add_argument("--chunk-window", type=int, default=None,
                        help="classification: paragraphs of context each side (default: configs/config.json)")
    parser.add_argument("--search-frac", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=5,
                        help="excerpts numbered into one judge call (benchmarked sweet spot: 5)")
    parser.add_argument("--delay", type=float, default=0.3, help="pause between sequential batch calls")
    args = parser.parse_args()
    if not args.sample and not args.label:
        parser.error("Pass --sample and/or --label")
    if args.chunk_window is None:
        args.chunk_window = load_config()["seed_screen"]["chunk_window"]

    if args.stage == "detection":
        if args.sample:
            do_sample_detection(args)
        if args.label:
            asyncio.run(do_label_detection(args))
    else:
        if args.sample:
            do_sample_classification(args)
        if args.label:
            asyncio.run(do_label_classification(args))


if __name__ == "__main__":
    main()
