"""
val_03_dedup_split.py — Text-level dedup map + disjoint search-pool/holdout candidate split

Part of the rule_based validation hardening (see docs/methodology_iteration.md):
the 236-chunk sample already labeled by the judge (llm_labeled_sample__rule_based.parquet)
was used to hand-tune the promotional/governance BoW proxies, so it cannot serve as a
clean holdout. This script:

  1. Builds a corpus-wide dedup map keyed on chunk_id (= sha256(chunk_text)[:16], so
     identical text always shares an id) recording how many raw rows each unique text
     represents and where they occur, for the panel-level duplicate-inflation check.
  2. Excludes the already-labeled 236 chunk_ids from the sampling population.
  3. Draws a search-pool top-up sample (oversampled toward is_governance_related_proxy,
     since that's the rarest dimension and the 236 alone under-covers it for k-fold CV)
     from the remaining pool.
  4. Draws a holdout sample, proportional to true proxy prevalence (no oversampling —
     the holdout must reflect real-world class balance, not be inflated the way the
     236's ensure_dimension_coverage() top-up was), disjoint from both the 236 and the
     search-pool top-up.

Neither candidate set is labeled here — that's val_04 (search-pool top-up labeling,
appends to the existing 236) and val_05 (holdout labeling, single-look, refuses to
re-run once written).

Usage:
    uv run python scripts/val_03_dedup_split.py [--topup-n 130] [--holdout-n 350] [--seed 42]
"""

import argparse
import json
from pathlib import Path

import pandas as pd

try:
    import pipeline_logger
    import variant_utils
except ImportError:
    from scripts import pipeline_logger
    from scripts import variant_utils

DIMENSIONS = ["is_substantive", "is_promotional", "is_risk_related", "is_governance_related"]
RAREST_DIM = "is_governance_related"

LABELED_SAMPLE_PATH = Path("data/processed/variant_rule_based/validation/llm_labeled_sample__rule_based.parquet")
DEDUP_MAP_PATH = Path("data/interim/chunk_dedup_map.parquet")
OUT_DIR = Path("data/processed/variant_rule_based/validation")
TOPUP_CANDIDATES_PATH = OUT_DIR / "search_pool_topup_candidates__rule_based.parquet"
HOLDOUT_CANDIDATES_PATH = OUT_DIR / "holdout_candidates__rule_based.parquet"


def load_config() -> dict:
    with open("configs/config.json") as f:
        return json.load(f)


def build_dedup_map(scored_df: pd.DataFrame) -> pd.DataFrame:
    """One row per unique chunk_id/text: n_rows (raw duplicate count), and the list
    of (accession_number, ticker, filing_date, section_name) each row occurred at.
    Feeds the panel-level duplicate-inflation check (val step 7), not just validation
    sampling — inflated D1/D9 firm-year features from repeated boilerplate would show
    up as high n_rows concentrated in specific ticker/year combinations."""
    agg = (
        scored_df.groupby("chunk_id")
        .agg(
            n_rows=("chunk_id", "size"),
            tickers=("ticker", lambda s: sorted(set(s))),
            filing_dates=("filing_date", lambda s: sorted(set(str(d) for d in s))),
            section_names=("section_name", lambda s: sorted(set(s))),
        )
        .reset_index()
    )
    agg["is_duplicated"] = agg["n_rows"] > 1
    return agg


def stratified_by_dim(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Proportional-allocation sample stratified by year only (for temporal spread),
    with each year's draw sized to its share of the population — NOT by the boolean
    dims. Stratifying by dim combo with equal counts per stratum would inflate rare
    combos (exactly the ensure_dimension_coverage()-style bias this holdout must
    avoid); by only conditioning on year and using proportional allocation, the
    4 boolean dims' marginals fall out close to their true population prevalence."""
    df = df.copy()
    df["year"] = pd.to_datetime(df["filing_date"], errors="coerce").dt.year
    frac = min(1.0, n / len(df))

    sampled = (
        df.groupby("year", group_keys=False)
        .apply(lambda g: g.sample(max(1, round(len(g) * frac)), random_state=seed), include_groups=False)
    )
    sampled = df.loc[sampled.index]
    if len(sampled) > n:
        sampled = sampled.sample(n, random_state=seed)
    elif len(sampled) < n:
        remaining = df[~df.index.isin(sampled.index)]
        extra = remaining.sample(min(n - len(sampled), len(remaining)), random_state=seed)
        sampled = pd.concat([sampled, extra])
    return sampled.drop(columns=[c for c in ["year"] if c in sampled.columns])


def oversampled_topup(df: pd.DataFrame, rarest_dim: str, n: int, seed: int) -> pd.DataFrame:
    """Search-pool top-up: half the budget targeted at rarest_dim positives (the
    dimension under-covered by the existing 236 for stable k-fold CV), the rest a
    plain random draw across the remaining pool. This set is allowed to be biased —
    unlike the holdout, it only feeds the CV search, never the final reported metric."""
    n_rare = n // 2
    rare_pool = df[df[rarest_dim].astype(bool)]
    rare_sample = rare_pool.sample(min(n_rare, len(rare_pool)), random_state=seed)

    remaining = df[~df.index.isin(rare_sample.index)]
    n_rest = n - len(rare_sample)
    rest_sample = remaining.sample(min(n_rest, len(remaining)), random_state=seed)

    out = pd.concat([rare_sample, rest_sample])
    out["stratum"] = rarest_dim + "_topup"
    out.loc[rest_sample.index, "stratum"] = "random_topup"
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topup-n", type=int, default=130, help="Search-pool top-up target size")
    parser.add_argument("--holdout-n", type=int, default=350, help="Holdout target size")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = load_config()
    output_root = config.get("variants", {}).get("output_root", "data/processed")
    scored_path = variant_utils.variant_path("rule_based", "ai_scored_chunks", "parquet", output_root=output_root)

    if not scored_path.exists():
        print(f"Error: {scored_path} not found. Run scripts 07-09 (--variant rule_based) first.")
        return
    if not LABELED_SAMPLE_PATH.exists():
        print(f"Error: {LABELED_SAMPLE_PATH} not found. This script excludes it from resampling; it must exist.")
        return

    load_cols = ["chunk_id", "accession_number", "ticker", "filing_date", "section_name", "chunk_text"] + DIMENSIONS
    scored_df = pd.read_parquet(scored_path, columns=load_cols)
    print(f"Loaded {len(scored_df)} scored rows ({scored_df['chunk_id'].nunique()} unique chunk_id/text).")

    dedup_map = build_dedup_map(scored_df)
    DEDUP_MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    dedup_map.to_parquet(DEDUP_MAP_PATH, index=False)
    n_dup_groups = int(dedup_map["is_duplicated"].sum())
    n_dup_rows = int(dedup_map.loc[dedup_map["is_duplicated"], "n_rows"].sum())
    print(f"Dedup map: {len(dedup_map)} unique texts, {n_dup_groups} duplicated groups covering {n_dup_rows} raw rows.")
    print(f"  -> {DEDUP_MAP_PATH}")

    # unique-text population (one representative row per chunk_id)
    unique_pop = scored_df.drop_duplicates(subset="chunk_id", keep="first").reset_index(drop=True)

    already_labeled = pd.read_parquet(LABELED_SAMPLE_PATH, columns=["chunk_id"])
    labeled_ids = set(already_labeled["chunk_id"].tolist())
    print(f"Already labeled (excluded from resampling): {len(labeled_ids)} chunk_ids.")

    pool = unique_pop[~unique_pop["chunk_id"].isin(labeled_ids)].reset_index(drop=True)
    print(f"Remaining unlabeled unique-text pool: {len(pool)}.")

    topup = oversampled_topup(pool, RAREST_DIM, args.topup_n, args.seed)
    print(f"Search-pool top-up candidates: {len(topup)} "
          f"({int(topup[RAREST_DIM].sum())} governance-positive by proxy).")

    holdout_pool = pool[~pool["chunk_id"].isin(topup["chunk_id"])].reset_index(drop=True)
    holdout = stratified_by_dim(holdout_pool, args.holdout_n, args.seed)
    print(f"Holdout candidates: {len(holdout)} (proportional, no oversampling). Proxy positives per dimension:")
    for dim in DIMENSIONS:
        n_pos = int(holdout[dim].sum())
        print(f"  {dim}: {n_pos} ({n_pos / len(holdout) * 100:.1f}%)")

    overlap_topup_labeled = set(topup["chunk_id"]) & labeled_ids
    overlap_holdout_labeled = set(holdout["chunk_id"]) & labeled_ids
    overlap_topup_holdout = set(topup["chunk_id"]) & set(holdout["chunk_id"])
    assert not overlap_topup_labeled, f"Top-up overlaps labeled 236: {overlap_topup_labeled}"
    assert not overlap_holdout_labeled, f"Holdout overlaps labeled 236: {overlap_holdout_labeled}"
    assert not overlap_topup_holdout, f"Top-up overlaps holdout: {overlap_topup_holdout}"
    print("Disjointness check passed: 236 labeled / top-up / holdout share no chunk_id.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    topup_out = topup[["chunk_id", "ticker", "filing_date", "section_name", "chunk_text", "stratum"] + DIMENSIONS]
    holdout_out = holdout[["chunk_id", "ticker", "filing_date", "section_name", "chunk_text"] + DIMENSIONS]
    topup_out.to_parquet(TOPUP_CANDIDATES_PATH, index=False)
    holdout_out.to_parquet(HOLDOUT_CANDIDATES_PATH, index=False)
    print(f"\nWrote {len(topup_out)} search-pool top-up candidates -> {TOPUP_CANDIDATES_PATH}")
    print(f"Wrote {len(holdout_out)} holdout candidates -> {HOLDOUT_CANDIDATES_PATH}")
    print("\nNeither file is labeled yet. val_04 labels the top-up (appends to the search pool);"
          " val_05 labels the holdout once, ever.")

    pipeline_logger.log_event(
        pipeline_step="validation_hardening",
        level="SUCCESS",
        message="Built chunk dedup map and disjoint search-pool-topup/holdout candidate sets.",
        details={
            "n_unique_texts": len(dedup_map),
            "n_duplicated_groups": n_dup_groups,
            "n_topup_candidates": len(topup_out),
            "n_holdout_candidates": len(holdout_out),
        },
    )


if __name__ == "__main__":
    main()
