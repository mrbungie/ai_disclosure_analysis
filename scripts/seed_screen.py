"""
seed_screen.py — The lexical seed screen (docs/distillation_map.html §2, §7
step 3): a fixed, deliberately over-inclusive keyword screen over the whole
corpus. It is NOT a harness candidate and is never scored or optimized — its
only job is to define the three sampling strata stage 1 draws from:

    hit                  paragraph matches a seed keyword
    no_hit_filing_hits   paragraph doesn't match, but its filing has a hit
                         elsewhere
    no_hit_filing_clean  paragraph doesn't match, and its filing has none

All three retain positive inclusion probability in downstream sampling —
the screen's own accuracy is not assumed. Recording hit counts per paragraph
(not just per filing) is what lets the search sample later be weighted
toward weak hits.

Usage:
    uv run python scripts/seed_screen.py
"""

import json
import re
from pathlib import Path

import pandas as pd

try:
    import harness_fit
    import pipeline_logger
except ImportError:
    from scripts import harness_fit, pipeline_logger

OUTPUT_PATH = Path("data/interim/seed_screen/seed_screen.parquet")


def build_regex(keywords: list[str]) -> re.Pattern:
    patterns = [r"\b" + re.escape(kw).replace(r"\ ", r"\s+") + r"\b" for kw in keywords]
    return re.compile("|".join(patterns), re.IGNORECASE)


def clean_false_positives(text: str, false_positives: list[str]) -> str:
    for fp in false_positives:
        text = re.sub(r"\b" + re.escape(fp) + r"\b", "", text, flags=re.IGNORECASE)
    return text


def main() -> None:
    with open("configs/config.json") as f:
        config = json.load(f)
    screen_cfg = config["seed_screen"]
    ai_regex = build_regex(screen_cfg["ai_keywords"])
    false_positives = screen_cfg["false_positives"]

    paragraphs = harness_fit.flatten_corpus_paragraphs(config)

    def hit_count(text: str) -> int:
        cleaned = clean_false_positives(text, false_positives)
        return len(ai_regex.findall(cleaned))

    paragraphs["hit_count"] = paragraphs["paragraph_text"].map(hit_count)
    paragraphs["hit"] = paragraphs["hit_count"] > 0

    filing_has_hit = paragraphs.groupby("accession_number")["hit"].any().rename("filing_has_hit")
    paragraphs = paragraphs.merge(filing_has_hit, on="accession_number", how="left")

    paragraphs["stratum"] = "no_hit_filing_clean"
    paragraphs.loc[~paragraphs["hit"] & paragraphs["filing_has_hit"], "stratum"] = "no_hit_filing_hits"
    paragraphs.loc[paragraphs["hit"], "stratum"] = "hit"

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    paragraphs.to_parquet(OUTPUT_PATH, index=False)

    counts = paragraphs["stratum"].value_counts()
    print(f"Seed screen: {len(paragraphs)} paragraphs across "
          f"{paragraphs['accession_number'].nunique()} filings.")
    for stratum, n in counts.items():
        print(f"  {stratum:<22} {n} ({n / len(paragraphs) * 100:.2f}%)")
    print(f"Wrote -> {OUTPUT_PATH}")

    pipeline_logger.log_event(pipeline_step="seed_screen", level="SUCCESS",
                              message=f"Seed-screened {len(paragraphs)} paragraphs.",
                              details={"n": len(paragraphs), **counts.to_dict()})


if __name__ == "__main__":
    main()
