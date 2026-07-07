"""
val_10_prefilter_holdout.py — Draw a FRESH, never-seen holdout sample for the
prefilter/chunking keyword-list dimension (is_ai_related at the funnel level).

val_08/val_09's 1,468-paragraph sample is BURNED: it's what surfaced the single
"AIP" (Palantir) miss that got turned into a keyword-list fix (aip/aiops added
to configs/config.json, scripts 05-06 re-run). Using that same sample to also
report "the recall of the fixed keyword list" would be exactly the mistake the
236-chunk sample made for the fine dimensions (tuning and reporting on the same
data) — just one phase earlier in the pipeline (the keyword list is the boolean
search space here, not a proxy formula in script 09).

This script excludes every paragraph_id already seen in
prefilter_recall_labeled.parquet from the sampling population, then draws a
fresh, industry-stratified sample against the CURRENT (already-updated)
keyword list — so the holdout evaluates the frozen post-fix funnel, not the
pre-fix one.

Usage:
    uv run python scripts/val_10_prefilter_holdout.py [--nm-per-filing 2] [--excluded-n 250] [--seed 43]
"""

import argparse
import json
from pathlib import Path

import pandas as pd

try:
    import pipeline_logger
    from val_08_prefilter_recall import (
        build_regex, clean_false_positives, sample_no_matches_stratum,
        collect_excluded_within_matched, proportional_by_industry, load_config,
    )
except ImportError:
    from scripts import pipeline_logger
    from scripts.val_08_prefilter_recall import (
        build_regex, clean_false_positives, sample_no_matches_stratum,
        collect_excluded_within_matched, proportional_by_industry, load_config,
    )

OUT_DIR = Path("data/interim/validation")
BURNED_LABELED_PATH = OUT_DIR / "prefilter_recall_labeled.parquet"
OUT_PATH = OUT_DIR / "prefilter_recall_holdout_candidates.parquet"
POPULATIONS_PATH = OUT_DIR / "prefilter_recall_holdout_populations.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nm-per-filing", type=int, default=2, help="Paragraphs sampled per no_matches filing (census)")
    parser.add_argument("--excluded-n", type=int, default=250, help="Target sample size for excluded_within_matched stratum")
    parser.add_argument("--seed", type=int, default=43, help="Different seed from val_08 by default, on top of the burned-id exclusion")
    args = parser.parse_args()

    if OUT_PATH.exists():
        print(f"Error: {OUT_PATH} already exists. This is holdout material — regenerating it after "
              f"having potentially seen it would defeat the point. Delete it manually if you're certain "
              f"it was never inspected, or draw a third fresh batch instead.")
        return

    config = load_config()
    keywords = config["prefiltering"]["ai_keywords"]
    false_positives = config["prefiltering"]["false_positives"]
    ai_regex = build_regex(keywords)
    print(f"Using CURRENT keyword list ({len(keywords)} terms, includes any post-search-pool additions).")

    manifest = pd.read_parquet(Path(config["paths"]["interim_manifests"]) / "filing_manifest.parquet")
    sections = pd.read_parquet(Path(config["paths"]["interim_sections"]) / "filing_sections.parquet")
    firm_universe = pd.read_parquet(Path(config["paths"]["interim_manifests"]) / "firm_universe.parquet")
    ticker_industry = dict(zip(firm_universe["ticker"], firm_universe["industry_group"]))

    burned_ids = set()
    if BURNED_LABELED_PATH.exists():
        burned = pd.read_parquet(BURNED_LABELED_PATH, columns=["paragraph_id"])
        burned_ids = set(burned["paragraph_id"].tolist())
    print(f"Excluding {len(burned_ids)} already-seen (burned search-pool) paragraph_ids from resampling.")

    no_matches_acc = set(manifest.loc[manifest["prefilter_status"] == "no_matches", "accession_number"])
    matched_acc = set(manifest.loc[manifest["prefilter_status"] == "matched", "accession_number"])
    print(f"no_matches filings: {len(no_matches_acc)} | matched filings: {len(matched_acc)}")

    nm_sections = sections[sections["accession_number"].isin(no_matches_acc)]
    no_matches_population = sum(
        len([p for p in t.split("\n") if p.strip()]) for t in nm_sections["section_text"]
    )

    # Oversample the per-filing draw, then drop burned ids and trim back to target —
    # simplest way to keep the census-over-142-filings property while still avoiding
    # re-serving previously seen paragraphs.
    nm_pool = sample_no_matches_stratum(sections, no_matches_acc, ticker_industry, args.nm_per_filing * 3, args.seed)
    nm_pool = nm_pool[~nm_pool["paragraph_id"].isin(burned_ids)]
    nm_sample = (
        nm_pool.groupby("accession_number", group_keys=False)
        .apply(lambda g: g.sample(min(args.nm_per_filing, len(g)), random_state=args.seed), include_groups=False)
    )
    nm_sample = nm_pool.loc[nm_sample.index]
    print(f"no_matches_filing holdout: {len(nm_sample)} paragraphs, {nm_sample['industry_group'].nunique()} industries.")

    excluded_pool = collect_excluded_within_matched(sections, matched_acc, ticker_industry, ai_regex, false_positives)
    excluded_pool = excluded_pool[~excluded_pool["paragraph_id"].isin(burned_ids)]
    print(f"excluded_within_matched population (post-fix, minus burned): {len(excluded_pool)} paragraphs.")

    excluded_sample = proportional_by_industry(excluded_pool, args.excluded_n, args.seed)
    print(f"excluded_within_matched holdout: {len(excluded_sample)} paragraphs, {excluded_sample['industry_group'].nunique()} industries.")

    overlap = set(nm_sample["paragraph_id"]) & burned_ids | set(excluded_sample["paragraph_id"]) & burned_ids
    assert not overlap, f"Holdout overlaps burned search-pool ids: {overlap}"
    print("Disjointness check passed: holdout shares no paragraph_id with the burned search-pool sample.")

    combined = pd.concat([nm_sample, excluded_sample], ignore_index=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(OUT_PATH, index=False)
    print(f"\nWrote {len(combined)} fresh holdout candidates -> {OUT_PATH}")

    populations = {
        "no_matches_filings": len(no_matches_acc),
        "matched_filings": len(matched_acc),
        "no_matches_paragraph_population": int(no_matches_population),
        "excluded_within_matched_paragraph_population": len(excluded_pool),
    }
    with open(POPULATIONS_PATH, "w") as f:
        json.dump(populations, f, indent=2)
    print(f"Wrote population sizes -> {POPULATIONS_PATH}")
    print("\nNot labeled yet. val_11_prefilter_holdout_label.py judges this ONCE and locks the report.")

    pipeline_logger.log_event(
        pipeline_step="validation_hardening",
        level="SUCCESS",
        message="Built fresh, never-seen prefilter-recall holdout sample against the post-fix keyword list.",
        details={
            "n_no_matches_filing_holdout": len(nm_sample),
            "n_excluded_within_matched_holdout": len(excluded_sample),
            "n_burned_excluded": len(burned_ids),
            **populations,
        },
    )


if __name__ == "__main__":
    main()
