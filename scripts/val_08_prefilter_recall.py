"""
val_08_prefilter_recall.py — Sample the universe EXCLUDED by the prefilter/chunking
funnel (scripts 05-06), stratified by industry, for an LLM-judge recall check.

The 82% precision reported in validation_summary__rule_based.txt only measures
agreement on the 4,204 chunks that already made it into ai_candidate_chunks.parquet.
It says nothing about recall: paragraphs the funnel never surfaced as candidates in
the first place. Two distinct failure modes, both upstream of the rule_based/llm_full
variant split (05-06 run once, before variant selection applies):

  1. Filing-level: 142/658 filings (21.6%) never matched the keyword regex at all
     (prefilter_status == "no_matches") and were never chunked — the whole filing
     is invisible to the pipeline.
  2. Paragraph-level: within the 513 "matched" filings, only paragraphs within a
     +/-1 window of a keyword hit became candidate chunks. ~619K of ~634K paragraphs
     in matched filings (97.7%) fall outside any window and were never chunked either.

This script only builds the two sampling populations and draws industry-stratified
candidates — it does NOT call the judge. That's val_09_prefilter_recall_label.py.

Usage:
    uv run python scripts/val_08_prefilter_recall.py [--nm-per-filing 2] [--excluded-n 250] [--seed 42]
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd

try:
    import pipeline_logger
except ImportError:
    from scripts import pipeline_logger

OUT_DIR = Path("data/interim/validation")
OUT_PATH = OUT_DIR / "prefilter_recall_candidates.parquet"
POPULATIONS_PATH = OUT_DIR / "prefilter_recall_populations.json"


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


def sample_no_matches_stratum(sections_df, no_matches_acc, ticker_industry, per_filing_n, seed) -> pd.DataFrame:
    """Census over all 142 no_matches filings: per_filing_n paragraphs each, so
    industry coverage is exact (every filing represented), not sampled."""
    rows = []
    for acc, grp in sections_df[sections_df["accession_number"].isin(no_matches_acc)].groupby("accession_number"):
        paragraphs = []
        for _, sec_row in grp.iterrows():
            paras = [p.strip() for p in sec_row["section_text"].split("\n") if p.strip()]
            for i, p in enumerate(paras):
                paragraphs.append((sec_row["section_name"], i, p))
        if not paragraphs:
            continue
        n = min(per_filing_n, len(paragraphs))
        idx = pd.Series(range(len(paragraphs))).sample(n, random_state=seed).tolist()
        ticker = grp["ticker"].iloc[0]
        filing_date = grp["filing_date"].iloc[0]
        for i in idx:
            sec_name, para_idx, text = paragraphs[i]
            rows.append({
                "paragraph_id": paragraph_id(acc, sec_name, para_idx, text),
                "accession_number": acc,
                "ticker": ticker,
                "industry_group": ticker_industry.get(ticker),
                "filing_date": filing_date,
                "section_name": sec_name,
                "paragraph_text": text,
                "stratum": "no_matches_filing",
            })
    return pd.DataFrame(rows)


def collect_excluded_within_matched(sections_df, matched_acc, ticker_industry, ai_regex, false_positives) -> pd.DataFrame:
    """All paragraphs in matched filings that fall outside every keyword-hit's
    +/-1 merged window — the population script 06 silently drops."""
    rows = []
    for _, row in sections_df[sections_df["accession_number"].isin(matched_acc)].iterrows():
        acc = row["accession_number"]
        ticker = row["ticker"]
        filing_date = row["filing_date"]
        sec_name = row["section_name"]
        paragraphs = [p.strip() for p in row["section_text"].split("\n") if p.strip()]
        if not paragraphs:
            continue
        matched_idx = [i for i, p in enumerate(paragraphs) if ai_regex.search(clean_false_positives(p, false_positives))]
        covered = set()
        if matched_idx:
            windows = [(max(0, i - 1), min(len(paragraphs) - 1, i + 1)) for i in matched_idx]
            merged = []
            for s, e in sorted(windows):
                if not merged or merged[-1][1] < s:
                    merged.append([s, e])
                else:
                    merged[-1][1] = max(merged[-1][1], e)
            for s, e in merged:
                covered.update(range(s, e + 1))
        for i, p in enumerate(paragraphs):
            if i in covered:
                continue
            rows.append({
                "paragraph_id": paragraph_id(acc, sec_name, i, p),
                "accession_number": acc,
                "ticker": ticker,
                "industry_group": ticker_industry.get(ticker),
                "filing_date": filing_date,
                "section_name": sec_name,
                "paragraph_text": p,
                "stratum": "excluded_within_matched",
            })
    return pd.DataFrame(rows)


def proportional_by_industry(df: pd.DataFrame, n: int, seed: int, min_per_group: int = 2) -> pd.DataFrame:
    """Proportional allocation by industry_group (frac = n/N within each group),
    with a floor so small industries aren't zeroed out by rounding. No equal-count
    stratification — that would over-represent thin industries, the same bias we
    corrected for in the holdout split (val_03)."""
    df = df.copy()
    df["industry_group"] = df["industry_group"].fillna("Unknown")
    frac = min(1.0, n / len(df))

    def sample_group(g):
        target = max(min_per_group, round(len(g) * frac))
        return g.sample(min(target, len(g)), random_state=seed)

    sampled = df.groupby("industry_group", group_keys=False).apply(sample_group, include_groups=False)
    sampled = df.loc[sampled.index]
    if len(sampled) > n:
        sampled = sampled.sample(n, random_state=seed)
    return sampled


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nm-per-filing", type=int, default=2, help="Paragraphs sampled per no_matches filing (census)")
    parser.add_argument("--excluded-n", type=int, default=250, help="Target sample size for excluded_within_matched stratum")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = load_config()
    keywords = config["prefiltering"]["ai_keywords"]
    false_positives = config["prefiltering"]["false_positives"]
    ai_regex = build_regex(keywords)

    manifest = pd.read_parquet(Path(config["paths"]["interim_manifests"]) / "filing_manifest.parquet")
    sections = pd.read_parquet(Path(config["paths"]["interim_sections"]) / "filing_sections.parquet")
    firm_universe = pd.read_parquet(Path(config["paths"]["interim_manifests"]) / "firm_universe.parquet")
    ticker_industry = dict(zip(firm_universe["ticker"], firm_universe["industry_group"]))

    no_matches_acc = set(manifest.loc[manifest["prefilter_status"] == "no_matches", "accession_number"])
    matched_acc = set(manifest.loc[manifest["prefilter_status"] == "matched", "accession_number"])
    print(f"no_matches filings: {len(no_matches_acc)} | matched filings: {len(matched_acc)}")

    nm_sections = sections[sections["accession_number"].isin(no_matches_acc)]
    no_matches_population = sum(
        len([p for p in t.split("\n") if p.strip()]) for t in nm_sections["section_text"]
    )

    nm_tickers = set(manifest.loc[manifest["accession_number"].isin(no_matches_acc), "ticker"])
    nm_industries_present = {ticker_industry.get(t) for t in nm_tickers}
    nm_sample = sample_no_matches_stratum(sections, no_matches_acc, ticker_industry, args.nm_per_filing, args.seed)
    print(f"no_matches_filing stratum: {len(nm_sample)} paragraphs, "
          f"{nm_sample['industry_group'].nunique()} industries covered "
          f"(of {len(nm_industries_present)} present in that population).")

    excluded_pool = collect_excluded_within_matched(sections, matched_acc, ticker_industry, ai_regex, false_positives)
    print(f"excluded_within_matched population: {len(excluded_pool)} paragraphs across {excluded_pool['industry_group'].nunique()} industries.")

    excluded_sample = proportional_by_industry(excluded_pool, args.excluded_n, args.seed)
    print(f"excluded_within_matched stratum sample: {len(excluded_sample)} paragraphs, "
          f"{excluded_sample['industry_group'].nunique()} industries covered.")

    combined = pd.concat([nm_sample, excluded_sample], ignore_index=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(OUT_PATH, index=False)
    print(f"\nWrote {len(combined)} candidates ({len(nm_sample)} no_matches_filing + {len(excluded_sample)} excluded_within_matched)")
    print(f"  -> {OUT_PATH}")
    print("\nNot labeled yet — run val_09_prefilter_recall_label.py to judge these and compute the recall estimate.")

    candidate_chunks_path = Path(config["paths"]["candidate_chunks"]) / "ai_candidate_chunks.parquet"
    n_candidate_chunks = len(pd.read_parquet(candidate_chunks_path, columns=["chunk_id"])) if candidate_chunks_path.exists() else None

    m_sections_all = sections[sections["accession_number"].isin(matched_acc)]
    matched_total_paragraphs = sum(
        len([p for p in t.split("\n") if p.strip()]) for t in m_sections_all["section_text"]
    )
    covered_paragraphs = matched_total_paragraphs - len(excluded_pool)

    populations = {
        "no_matches_filings": len(no_matches_acc),
        "matched_filings": len(matched_acc),
        "no_matches_paragraph_population": int(no_matches_population),
        "excluded_within_matched_paragraph_population": len(excluded_pool),
        "n_candidate_chunks": n_candidate_chunks,
        "covered_paragraphs": int(covered_paragraphs),
    }
    with open(POPULATIONS_PATH, "w") as f:
        json.dump(populations, f, indent=2)
    print(f"Wrote population sizes -> {POPULATIONS_PATH}")

    pipeline_logger.log_event(
        pipeline_step="validation_hardening",
        level="SUCCESS",
        message="Built prefilter-recall candidate sample, stratified by industry_group across both exclusion failure modes.",
        details={
            "n_no_matches_filing_sample": len(nm_sample),
            "n_excluded_within_matched_sample": len(excluded_sample),
            **populations,
        },
    )


if __name__ == "__main__":
    main()
