"""
phase0_discovery.py — Phase 0: Concept & Rule Discovery
(docs/distillation_map.html §0), upstream of Cycle 1 / Distillation 1 —
tooling only. Producing and freezing a phase0 harness candidate is the
proposer's job (Claude Code, via .claude/skills/phase0-opt/SKILL.md —
same pattern as detection-opt/classification-opt), not this script's: it
draws a discovery sample and asks an LLM to suggest a ConceptSeed, but
the ConceptSeed that actually gets scored and frozen lives inside a
versioned, immutable `harnesses/phase0/<name>/concept_seed.json` the
proposer writes, evaluated via `scripts/eval_harness.py --task phase0`
exactly like any other candidate — never auto-generated or auto-frozen
by this script.

Induces a ConceptSeed SUGGESTION — named concepts, positive/negative
semantic anchors, candidate lexical terms, and candidate inclusion/
exclusion rules — from an LLM reading a small DISCOVERY sample of the
corpus, never the fitting/estimation sample the harnesses search
against. This widens the seed screen's own construct representation
beyond literal keyword matching (configs/config.json:
seed_screen.ai_keywords), without touching that keyword list directly —
lexical_candidates are surfaced as suggestions for a human/proposer to
fold into seed_screen.ai_keywords later, not auto-merged.

Usage:
    uv run python scripts/phase0_discovery.py --sample
    uv run python scripts/phase0_discovery.py --induce
    uv run python scripts/phase0_discovery.py --embed-anchors
    uv run python scripts/phase0_discovery.py --embed-discovery-sample
"""

import argparse
import asyncio
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field

try:
    import embeddings
    import harness_fit
    import pipeline_logger
except ImportError:
    from scripts import embeddings, harness_fit, pipeline_logger

load_dotenv()

DISCOVERY_SAMPLE_NAME = "discovery_sample.parquet"
SUGGESTIONS_SUBDIR = "concept_seed_suggestions"


def load_config() -> dict:
    with open("configs/config.json") as f:
        return json.load(f)


def phase0_dir(config: dict) -> Path:
    d = Path(config["paths"]["interim_phase0"])
    d.mkdir(parents=True, exist_ok=True)
    return d


def suggestions_dir(config: dict) -> Path:
    d = phase0_dir(config) / SUGGESTIONS_SUBDIR
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# ConceptSeed schema — the payload a harnesses/phase0/<name>/concept_seed.json
# candidate file holds
# ---------------------------------------------------------------------------


class Concept(BaseModel):
    name: str = Field(description="Short slug for the concept, e.g. internal_ai_adoption.")
    description: str = Field(description="One or two sentences defining the concept.")
    positive_anchors: list[str] = Field(
        description="3-8 short example sentences/phrases of real 10-K text that clearly express this "
                    "concept, drawn from or closely paraphrasing the discovery sample.")
    negative_anchors: list[str] = Field(
        description="2-5 short example sentences/phrases that LOOK related (generic tech/business "
                    "language) but do NOT express this concept — contrastive examples that should NOT "
                    "match, e.g. boilerplate 'new technologies' risk-factor language with no AI content.")


class ConceptSeed(BaseModel):
    concepts: list[Concept]
    lexical_candidates: list[str] = Field(
        description="Candidate terms or short phrases (not already in the existing keyword list given "
                    "to you) worth considering for the literal seed-screen keyword list. Suggestions "
                    "only, low false-positive risk preferred.")
    inclusion_rules: list[str] = Field(
        description="Short natural-language notes on what should count as in-scope AI disclosure, "
                    "beyond literal keyword presence.")
    exclusion_rules: list[str] = Field(
        description="Short natural-language notes on known false-positive patterns (acronym "
                    "collisions, generic-tech boilerplate that resembles AI language but isn't).")


INDUCTION_SYSTEM_PROMPT = (
    "You are a financial-disclosure research assistant helping build a search space for detecting "
    "AI/ML/LLM-related content in SEC 10-K filings. You will read a sample of real 10-K paragraphs "
    "(NOT all AI-related — most are not) and induce a structured ConceptSeed: a small set of named "
    "concepts covering the different ways AI shows up in 10-Ks (e.g. internal AI adoption, AI "
    "products/features, generative AI, AI as a risk factor, AI governance/oversight), each with "
    "positive example anchors (real or closely paraphrased text expressing the concept) and negative "
    "anchors (text that superficially resembles AI language but is generic tech/business boilerplate "
    "with no real AI content — this contrast is important for later precision). Also list lexical "
    "candidates (terms not already in the existing keyword list) and short inclusion/exclusion rules. "
    "Do not invent content beyond what plausibly matches the style of the sample; ground anchors in it."
)


# ---------------------------------------------------------------------------
# --sample: discovery sample (independent of, and never checked against,
# the eval search/holdout samples — a separate purpose-built draw)
# ---------------------------------------------------------------------------


def do_sample(args: argparse.Namespace) -> None:
    config = load_config()
    path = phase0_dir(config) / DISCOVERY_SAMPLE_NAME
    n = args.n or config["phase0"]["discovery_sample"]["n"]
    seed = args.seed if args.seed is not None else config["phase0"]["discovery_sample"]["seed"]

    population = harness_fit.flatten_corpus_paragraphs(config)
    existing = pd.read_parquet(path) if path.exists() else None
    existing_ids = set(existing["paragraph_id"]) if existing is not None else set()

    pool = population[~population["paragraph_id"].isin(existing_ids)]
    need = max(0, n - len(existing_ids))
    if need == 0:
        print(f"Discovery sample already has {len(existing_ids)} rows (>= requested {n}) — nothing to add.")
        return
    draw = pool.sample(n=min(need, len(pool)), random_state=seed)
    sample = pd.concat([existing, draw], ignore_index=True) if existing is not None else draw
    sample.to_parquet(path, index=False)
    print(f"Discovery sample: {len(sample)} paragraphs -> {path}")
    pipeline_logger.log_event(pipeline_step="phase0_sample", level="SUCCESS",
                              message=f"Drew {len(draw)} new discovery-sample paragraphs.",
                              details={"n_total": len(sample), "n_new": len(draw)})


# ---------------------------------------------------------------------------
# --induce: one LLM call over a subset of the discovery sample -> a
# ConceptSeed SUGGESTION (not a candidate — see module docstring)
# ---------------------------------------------------------------------------


def _version_of(path: Path) -> int:
    m = re.search(r"v(\d+)\.json$", path.name)
    assert m is not None, f"not a suggestion_v*.json path: {path}"
    return int(m.group(1))


def _next_suggestion_version(config: dict) -> int:
    existing = list(suggestions_dir(config).glob("suggestion_v*.json"))
    versions = [_version_of(p) for p in existing]
    return max(versions, default=0) + 1


def harness_fit_scope_prompt() -> str:
    try:
        import build_eval_set
    except ImportError:
        from scripts import build_eval_set
    return build_eval_set.SCOPE_SYSTEM_PROMPT


async def do_induce(args: argparse.Namespace) -> None:
    config = load_config()
    sample_path = phase0_dir(config) / DISCOVERY_SAMPLE_NAME
    if not sample_path.exists():
        print(f"Error: {sample_path} not found. Run with --sample first.")
        return
    sample = pd.read_parquet(sample_path)
    subset = sample.sample(n=min(args.induction_n, len(sample)), random_state=args.seed)
    excerpts = "\n\n".join(f"[{i}] {t[:args.excerpt_chars]}" for i, t in enumerate(subset["paragraph_text"]))

    keywords = ", ".join(config["seed_screen"]["ai_keywords"])
    prompt = (
        f"Existing scope rule: {harness_fit_scope_prompt()}\n\n"
        f"Existing seed-screen keywords (do not just repeat these as lexical_candidates): {keywords}\n\n"
        f"Discovery sample ({len(subset)} 10-K paragraphs, mixed AI-related and not):\n\n{excerpts}"
    )

    agent = harness_fit.build_judge(ConceptSeed, INDUCTION_SYSTEM_PROMPT)
    # Single unbatched call: build_judge's batching wraps output_type into a
    # list-of-items schema for per-row labeling; induction wants ONE
    # aggregate ConceptSeed, so we call the underlying model directly rather
    # than going through judge_one/judge_batch's per-item plumbing.
    from pydantic_ai import Agent
    induction_agent = Agent(agent.model, output_type=ConceptSeed, retries=3,
                            system_prompt=INDUCTION_SYSTEM_PROMPT)
    result = await induction_agent.run(prompt)
    concept_seed = result.output

    version = _next_suggestion_version(config)
    out_path = suggestions_dir(config) / f"suggestion_v{version}.json"
    out_path.write_text(json.dumps(concept_seed.model_dump(), indent=2))
    print(f"ConceptSeed suggestion v{version}: {len(concept_seed.concepts)} concepts, "
          f"{len(concept_seed.lexical_candidates)} lexical candidates -> {out_path}")
    for c in concept_seed.concepts:
        print(f"  - {c.name}: {len(c.positive_anchors)} positive / {len(c.negative_anchors)} negative anchors")
    print("This is raw material, not a candidate — copy/edit it into a new "
          "harnesses/phase0/<name>/concept_seed.json to actually propose and score it "
          "(scripts/eval_harness.py --task phase0 --candidate <name>).")
    pipeline_logger.log_event(pipeline_step="phase0_induce", level="SUCCESS",
                              message=f"Induced ConceptSeed suggestion v{version}.",
                              details={"version": version, "n_concepts": len(concept_seed.concepts)})


# ---------------------------------------------------------------------------
# --embed-anchors / --embed-discovery-sample, and the scoring machinery
# harnesses/phase0/<name>/harness.py candidates import
# ---------------------------------------------------------------------------


def latest_suggestion_path(config: dict, version: int | None = None) -> Path:
    if version is not None:
        return suggestions_dir(config) / f"suggestion_v{version}.json"
    candidates = list(suggestions_dir(config).glob("suggestion_v*.json"))
    if not candidates:
        raise FileNotFoundError("No suggestion_v*.json found — run --induce first.")
    return max(candidates, key=_version_of)


def anchor_dataframe(concept_seed: ConceptSeed) -> pd.DataFrame:
    """One row per anchor string across all concepts, both polarities.
    id is a stable hash of (concept name, polarity, text) so re-embedding
    the same anchor twice hits the cache."""
    rows = []
    for c in concept_seed.concepts:
        for polarity, anchors in (("positive", c.positive_anchors), ("negative", c.negative_anchors)):
            for text in anchors:
                rows.append({
                    "paragraph_id": harness_fit.paragraph_id(c.name, polarity, len(rows), text),
                    "paragraph_text": text, "concept": c.name, "polarity": polarity,
                })
    return pd.DataFrame(rows)


def do_embed_anchors(args: argparse.Namespace) -> None:
    config = load_config()
    concept_seed = ConceptSeed.model_validate_json(
        latest_suggestion_path(config, args.suggestion_version).read_text())
    anchors = anchor_dataframe(concept_seed)
    model_key = config["phase0"]["active_embedding_model"]
    vecs = embeddings.get_or_compute_embeddings(anchors, model_key, config)
    print(f"Embedded {len(vecs)} anchors with '{model_key}' -> "
          f"{Path(config['paths']['interim_embeddings']) / (model_key + '.parquet')}")


def do_embed_discovery_sample(args: argparse.Namespace) -> None:
    config = load_config()
    sample_path = phase0_dir(config) / DISCOVERY_SAMPLE_NAME
    if not sample_path.exists():
        print(f"Error: {sample_path} not found. Run with --sample first.")
        return
    sample = pd.read_parquet(sample_path)
    model_key = config["phase0"]["active_embedding_model"]
    vecs = embeddings.get_or_compute_embeddings(sample, model_key, config)
    print(f"Embedded {len(vecs)} discovery-sample paragraphs with '{model_key}' -> "
          f"{Path(config['paths']['interim_embeddings']) / (model_key + '.parquet')}")


def concept_anchor_similarities(texts: pd.DataFrame, concept_seed: ConceptSeed, model_key: str,
                                config: dict, text_col: str = "paragraph_text",
                                id_col: str = "paragraph_id") -> pd.DataFrame:
    """For each row in `texts`, the max cosine similarity to any POSITIVE
    anchor per concept, minus the max similarity to any NEGATIVE anchor of
    that same concept (a simple contrastive score — high only when a text
    is close to what a concept looks like and far from what it is
    confusable with). Used by `semantic_hit()` below (the single-text
    entry point harness.py candidates call) and any ad hoc verification.
    Anchors are embedded (and cached) on demand via `embeddings.py`."""
    anchors = anchor_dataframe(concept_seed)
    anchor_vecs = embeddings.get_or_compute_embeddings(anchors, model_key, config)
    anchor_vecs = anchors.merge(anchor_vecs, on="paragraph_id")

    text_vecs = embeddings.get_or_compute_embeddings(texts, model_key, config, id_col=id_col, text_col=text_col)
    text_matrix = np.stack(text_vecs["embedding"].to_numpy())

    out = pd.DataFrame({id_col: text_vecs[id_col]})
    for concept in concept_seed.concepts:
        pos = anchor_vecs[(anchor_vecs["concept"] == concept.name) & (anchor_vecs["polarity"] == "positive")]
        neg = anchor_vecs[(anchor_vecs["concept"] == concept.name) & (anchor_vecs["polarity"] == "negative")]
        pos_matrix = np.stack(pos["embedding"].to_numpy())
        pos_sim = embeddings.cosine_similarity(text_matrix, pos_matrix).max(axis=1)
        if len(neg):
            neg_matrix = np.stack(neg["embedding"].to_numpy())
            neg_sim = embeddings.cosine_similarity(text_matrix, neg_matrix).max(axis=1)
        else:
            neg_sim = np.zeros(len(text_matrix))
        out[f"sim_{concept.name}"] = pos_sim - neg_sim
    out["semantic_score"] = out.filter(like="sim_").max(axis=1)
    return out


def semantic_hit(text: str, concept_seed: ConceptSeed, model_key: str, threshold: float,
                 config: dict | None = None) -> bool:
    """Single-text entry point for a harnesses/phase0/<name>/harness.py
    candidate's classify(text) -> bool. Keyed by a text hash rather than a
    real paragraph_id — same constraint the classify(text) contract
    already has everywhere else (no id is passed in)."""
    config = config or load_config()
    df = pd.DataFrame({"paragraph_id": [embeddings.text_hash(text)], "paragraph_text": [text]})
    sims = concept_anchor_similarities(df, concept_seed, model_key, config)
    return bool(sims["semantic_score"].iloc[0] >= threshold)


# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--induce", action="store_true")
    parser.add_argument("--embed-anchors", action="store_true")
    parser.add_argument("--embed-discovery-sample", action="store_true")
    parser.add_argument("--n", type=int, default=None, help="--sample: override configs/config.json: phase0.discovery_sample.n")
    parser.add_argument("--seed", type=int, default=None, help="--sample/--induce: override configs/config.json: phase0.discovery_sample.seed")
    parser.add_argument("--induction-n", type=int, default=300, help="--induce: how many discovery-sample rows to feed the LLM in one call")
    parser.add_argument("--excerpt-chars", type=int, default=500, help="--induce: chars per excerpt in the induction prompt")
    parser.add_argument("--suggestion-version", type=int, default=None, help="--embed-anchors: default is the latest suggestion")
    args = parser.parse_args()

    if not any([args.sample, args.induce, args.embed_anchors, args.embed_discovery_sample]):
        parser.error("Pass at least one of --sample/--induce/--embed-anchors/--embed-discovery-sample")

    if args.sample:
        do_sample(args)
    if args.induce:
        asyncio.run(do_induce(args))
    if args.embed_anchors:
        do_embed_anchors(args)
    if args.embed_discovery_sample:
        do_embed_discovery_sample(args)


if __name__ == "__main__":
    main()
