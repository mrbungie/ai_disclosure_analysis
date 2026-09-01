"""
seed_screen.py — The DUAL-ROUTE seed screen (docs/distillation_map.html
§2, §7 step 3): a fixed, deliberately over-inclusive screen over the whole
corpus. It is NOT a harness candidate and is never scored or optimized —
its only job is to define the three sampling strata stage 1 draws from:

    hit                  paragraph matches a seed keyword OR the semantic screen
    no_hit_filing_hits   paragraph doesn't match, but its filing has a hit
                         elsewhere
    no_hit_filing_clean  paragraph doesn't match, and its filing has none

All three retain positive inclusion probability in downstream sampling —
the screen's own accuracy is not assumed. Recording hit counts per paragraph
(not just per filing) is what lets the search sample later be weighted
toward weak hits.

Two ROUTES to a hit, kept as separate columns (`lexical_hit`,
`semantic_hit`) and only UNIONED — never required to agree — for
stratification: `Seed(x) = [lexical_hit] OR [semantic_hit]`. This is
deliberate, not an implementation shortcut: lexical matching and
embedding similarity solve different problems (lexical is a partially
independent check that doesn't inherit whatever bias a phase0
ConceptSeed's LLM-generated anchors might carry; embeddings catch
paraphrases lexical matching structurally cannot), so the screen is
over-inclusive along BOTH routes rather than picking one or requiring
consensus. If configs/config.json: phase0.enabled is true AND a phase0
harness candidate is frozen (harnesses/phase0/ACTIVE — see
.claude/skills/phase0-opt/SKILL.md and scripts/eval_harness.py
--task phase0), `semantic_hit` applies that candidate's classify() at
corpus scale exactly the way scripts/apply_harness.py applies detection/
classification candidates. With phase0.enabled false (the default),
`semantic_hit` is always False and behavior is identical to before Phase
0 existed.

An `agreement_quadrant` column (also phase0-gated, else None) records
where each paragraph falls in the lexical x semantic 2x2 — lex_high_emb_
high (easy positives), lex_high_emb_low (probable lexical false
positives), lex_low_emb_high (probable lexical false negatives — exactly
what phase0 exists to surface), lex_low_emb_low (background). This does
NOT affect `hit`/`stratum` at all; it's read by
scripts/build_eval_set.py to make the SEARCH sample (never the holdout)
oversample the two disagreement quadrants, since they're the most
informative rows for a proposer to see.

Usage:
    uv run python scripts/seed_screen.py
"""

import json
import re
from pathlib import Path

import numpy as np

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
    paragraphs["lexical_hit"] = paragraphs["hit_count"] > 0

    phase0_cfg = config.get("phase0", {})
    phase0_active_path = harness_fit.HARNESSES_DIR / "phase0" / "ACTIVE"
    paragraphs["agreement_quadrant"] = None
    if phase0_cfg.get("enabled") and phase0_active_path.exists():
        phase0_name = harness_fit.active_candidate("phase0")
        print(f"Phase 0 enabled: applying harnesses/phase0/{phase0_name} at corpus scale "
              f"— this embeds every corpus paragraph and can take a while.")
        phase0_classify = harness_fit.load_candidate("phase0", phase0_name)
        paragraphs["semantic_hit"] = paragraphs["paragraph_text"].map(phase0_classify)
        # lexical x semantic 2x2 — read by build_eval_set.py to oversample the
        # two disagreement quadrants in the SEARCH sample (never the holdout);
        # does not affect hit/stratum at all.
        paragraphs["agreement_quadrant"] = np.select(
            [paragraphs["lexical_hit"] & paragraphs["semantic_hit"],
             paragraphs["lexical_hit"] & ~paragraphs["semantic_hit"],
             ~paragraphs["lexical_hit"] & paragraphs["semantic_hit"]],
            ["lex_high_emb_high", "lex_high_emb_low", "lex_low_emb_high"],
            default="lex_low_emb_low")
    else:
        paragraphs["semantic_hit"] = False

    paragraphs["hit"] = paragraphs["lexical_hit"] | paragraphs["semantic_hit"]

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
    if phase0_cfg.get("enabled"):
        semantic_only = int((paragraphs["semantic_hit"] & ~paragraphs["lexical_hit"]).sum())
        print(f"  semantic-only hits (caught by Phase 0, missed by lexical): {semantic_only}")
    print(f"Wrote -> {OUTPUT_PATH}")

    pipeline_logger.log_event(pipeline_step="seed_screen", level="SUCCESS",
                              message=f"Seed-screened {len(paragraphs)} paragraphs.",
                              details={"n": len(paragraphs), **counts.to_dict()})


if __name__ == "__main__":
    main()
