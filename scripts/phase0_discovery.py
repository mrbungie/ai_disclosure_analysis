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
exclusion rules — via an EMBEDDINGS-FIRST discovery pipeline
(`--discover`) over a small DISCOVERY sample of the corpus, never the
fitting/estimation sample the harnesses search against:

    seed anchors (1 LLM call, names/descriptions only)
      -> embed anchors + the full discovery sample
      -> nearest-neighbor NEIGHBORHOOD per anchor (pure computation)
      -> discriminative lexical terms per neighborhood (Counter-based,
         no new ML dependency — this is what turns an opaque
         sim(x, concept)=0.81 into "characterized by machine learning,
         AI-enabled, copilot")
      -> grounded refinement (1 LLM call, batched): given each
         neighborhood's REAL excerpts + its discriminative terms, the
         LLM confirms/discards the concept and writes anchors grounded
         in that evidence rather than an ungrounded first guess

Two roles for the two signal types, deliberately kept separate rather
than fused into one score (see docs/distillation_map.html §0): embeddings
are the DISCOVERY/coverage mechanism (find the concept even when the
literal wording differs); lexical presence/counts are the
INTERPRETABILITY mechanism (what a neighborhood is actually about, in
words a reader can audit). This widens the seed screen's own construct
representation beyond literal keyword matching (configs/config.json:
seed_screen.ai_keywords), without touching that keyword list directly —
lexical_candidates are surfaced as suggestions for a human/proposer to
fold into seed_screen.ai_keywords later, not auto-merged.

Usage:
    uv run python scripts/phase0_discovery.py --sample
    uv run python scripts/phase0_discovery.py --discover
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


class SeedAnchor(BaseModel):
    name: str = Field(description="Short slug, e.g. internal_ai_adoption.")
    description: str = Field(description="One sentence naming a way AI might show up in a 10-K.")


class SeedAnchorSet(BaseModel):
    anchors: list[SeedAnchor]


SEED_ANCHOR_SYSTEM_PROMPT = (
    "You are a financial-disclosure research assistant. Propose 5-8 short, DISTINCT candidate concepts "
    "for the different ways AI/ML/LLM content might show up in a SEC 10-K filing (e.g. internal AI "
    "adoption, AI products/features, generative AI, AI as a risk factor, AI governance/oversight, "
    "AI-related vendor/partnership disclosures, workforce/hiring for AI). Each concept: a short slug "
    "name and a ONE-SENTENCE description only — no examples, no anchors yet. These are just starting "
    "points that will be checked against a real corpus sample next, so cast a reasonably wide net."
)

REFINEMENT_SYSTEM_PROMPT = (
    "You are a financial-disclosure research assistant. For each numbered candidate concept below, you "
    "are shown: its name/description, a NEIGHBORHOOD of real 10-K excerpts that turned out to be "
    "semantically closest to it in a corpus sample, and a list of terms/phrases that are statistically "
    "over-represented in that neighborhood versus the rest of the sample. Using ONLY this evidence: "
    "(1) decide whether the neighborhood is actually about the candidate concept (AI/ML/LLM content) — "
    "if the excerpts are NOT really AI-related, DROP this concept entirely (return it with an empty "
    "concepts entry is wrong; simply omit it from your output); (2) if kept, write positive_anchors as "
    "short excerpts or close paraphrases DRAWN FROM the neighborhood shown (not invented) and "
    "negative_anchors as short excerpts of generic tech/business language that would be confusable but "
    "isn't really this concept; (3) from the over-represented terms shown, select the ones that are "
    "low false-positive-risk as lexical_candidates, and note which ones you rejected and why in "
    "exclusion_rules. Also fill inclusion_rules with any general pattern you notice across concepts."
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
# --discover: seed anchors -> embed -> nearest-neighbor neighborhoods ->
# discriminative terms -> grounded LLM refinement -> a ConceptSeed
# SUGGESTION (not a candidate — see module docstring)
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


_TOKEN_RE = re.compile(r"[a-z]+(?:-[a-z]+)*")
_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "is", "are",
    "this", "that", "as", "by", "from", "at", "our", "we", "its", "be", "will", "not",
    "may", "any", "such", "which", "these", "those", "has", "have", "had", "was", "were",
}


def _ngrams(text: str, n: int) -> list[str]:
    tokens = _TOKEN_RE.findall(text.lower())
    return [" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def discriminative_terms(neighborhood_texts: list[str], background_texts: list[str],
                         top_k: int = 15) -> list[str]:
    """Unigram/bigram/trigram frequency-ratio scoring: a term's rate inside
    the neighborhood over its (Laplace-smoothed) rate in the background,
    ranked descending. stdlib Counter only — no new ML dependency. This is
    the artifact that turns an opaque sim(x, concept)=0.81 into
    'characterized by machine learning, AI-enabled, copilot' — the
    interpretability layer for embeddings' discovery layer."""
    from collections import Counter

    def counts(texts: list[str], n: int) -> Counter:
        c = Counter()
        for t in texts:
            c.update(_ngrams(t, n))
        return c

    scored: list[tuple[str, float, int]] = []
    for n in (1, 2, 3):
        nb, bg = counts(neighborhood_texts, n), counts(background_texts, n)
        nb_total, bg_total = sum(nb.values()) or 1, sum(bg.values()) or 1
        for term, c in nb.items():
            if n == 1 and (term in _STOPWORDS or len(term) < 3):
                continue
            if c < 2:
                continue
            rate_nb = c / nb_total
            rate_bg = (bg.get(term, 0) + 0.5) / bg_total
            scored.append((term, rate_nb / rate_bg, c))
    scored.sort(key=lambda x: -x[1])
    return [term for term, _, _ in scored[:top_k]]


async def _propose_seed_anchors(config: dict) -> SeedAnchorSet:
    agent = harness_fit.build_judge(SeedAnchorSet, SEED_ANCHOR_SYSTEM_PROMPT)
    keywords = ", ".join(config["seed_screen"]["ai_keywords"])
    prompt = (
        f"Existing scope rule: {harness_fit_scope_prompt()}\n\n"
        f"Existing seed-screen keywords (propose concepts beyond these, not restating them): {keywords}"
    )
    from pydantic_ai import Agent
    seed_agent = Agent(agent.model, output_type=SeedAnchorSet, retries=3,
                       system_prompt=SEED_ANCHOR_SYSTEM_PROMPT)
    result = await seed_agent.run(prompt)
    return result.output


def _build_neighborhoods(seed_anchors: SeedAnchorSet, sample: pd.DataFrame, model_key: str,
                         config: dict, top_n: int) -> list[dict]:
    anchor_df = pd.DataFrame({
        "paragraph_id": [harness_fit.paragraph_id(a.name, "seed", i, a.description)
                        for i, a in enumerate(seed_anchors.anchors)],
        "paragraph_text": [a.description for a in seed_anchors.anchors],
    })
    anchor_vecs = embeddings.get_or_compute_embeddings(anchor_df, model_key, config)
    anchor_matrix = np.stack(anchor_vecs["embedding"].to_numpy())

    sample_vecs = embeddings.get_or_compute_embeddings(sample, model_key, config)
    sample_vecs = sample.merge(sample_vecs, on="paragraph_id")
    sample_matrix = np.stack(sample_vecs["embedding"].to_numpy())

    sims = embeddings.cosine_similarity(anchor_matrix, sample_matrix)  # (n_anchors, n_sample)
    neighborhoods = []
    for i, anchor in enumerate(seed_anchors.anchors):
        order = np.argsort(-sims[i])
        top_idx = order[:top_n]
        neighborhood_texts = sample_vecs.iloc[top_idx]["paragraph_text"].tolist()
        background_idx = order[top_n:]
        background_texts = sample_vecs.iloc[background_idx]["paragraph_text"].sample(
            n=min(len(background_idx), top_n * 4), random_state=42).tolist() if len(background_idx) else []
        neighborhoods.append({
            "anchor": anchor, "texts": neighborhood_texts,
            "terms": discriminative_terms(neighborhood_texts, background_texts),
        })
    return neighborhoods


async def do_discover(args: argparse.Namespace) -> None:
    config = load_config()
    sample_path = phase0_dir(config) / DISCOVERY_SAMPLE_NAME
    if not sample_path.exists():
        print(f"Error: {sample_path} not found. Run with --sample first.")
        return
    sample = pd.read_parquet(sample_path)
    model_key = config["phase0"]["active_embedding_model"]

    print("Step A: proposing seed anchors...")
    seed_anchors = await _propose_seed_anchors(config)
    for a in seed_anchors.anchors:
        print(f"  - {a.name}: {a.description}")

    print(f"Step B: embedding {len(sample)} discovery-sample rows + anchors, "
          f"building top-{args.neighborhood_size} neighborhoods...")
    neighborhoods = _build_neighborhoods(seed_anchors, sample, model_key, config, args.neighborhood_size)
    for nb in neighborhoods:
        print(f"  - {nb['anchor'].name}: top terms = {', '.join(nb['terms'][:8])}")

    print("Step C: grounded refinement (1 LLM call, all neighborhoods batched)...")
    blocks = []
    for i, nb in enumerate(neighborhoods):
        excerpts = "\n".join(f"    - {t[:300]}" for t in nb["texts"])
        blocks.append(
            f"[{i}] Candidate concept: {nb['anchor'].name} — {nb['anchor'].description}\n"
            f"  Neighborhood excerpts:\n{excerpts}\n"
            f"  Over-represented terms: {', '.join(nb['terms'])}"
        )
    prompt = "\n\n".join(blocks)

    agent = harness_fit.build_judge(ConceptSeed, REFINEMENT_SYSTEM_PROMPT)
    from pydantic_ai import Agent
    refine_agent = Agent(agent.model, output_type=ConceptSeed, retries=3,
                         system_prompt=REFINEMENT_SYSTEM_PROMPT)
    result = await refine_agent.run(prompt)
    concept_seed = result.output

    version = _next_suggestion_version(config)
    out_path = suggestions_dir(config) / f"suggestion_v{version}.json"
    out_path.write_text(json.dumps(concept_seed.model_dump(), indent=2))
    print(f"\nConceptSeed suggestion v{version}: {len(concept_seed.concepts)}/{len(seed_anchors.anchors)} "
          f"seed anchors kept after grounding, {len(concept_seed.lexical_candidates)} lexical "
          f"candidates -> {out_path}")
    for c in concept_seed.concepts:
        print(f"  - {c.name}: {len(c.positive_anchors)} positive / {len(c.negative_anchors)} negative anchors")
    dropped = {a.name for a in seed_anchors.anchors} - {c.name for c in concept_seed.concepts}
    if dropped:
        print(f"  Dropped as not grounded in real neighborhood evidence: {', '.join(sorted(dropped))}")
    print("This is raw material, not a candidate — copy/edit it into a new "
          "harnesses/phase0/<name>/concept_seed.json to actually propose and score it "
          "(scripts/eval_harness.py --task phase0 --candidate <name>).")
    pipeline_logger.log_event(pipeline_step="phase0_discover", level="SUCCESS",
                              message=f"Discovered ConceptSeed suggestion v{version}.",
                              details={"version": version, "n_concepts": len(concept_seed.concepts),
                                      "n_seed_anchors": len(seed_anchors.anchors)})


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
    parser.add_argument("--discover", action="store_true")
    parser.add_argument("--embed-anchors", action="store_true")
    parser.add_argument("--embed-discovery-sample", action="store_true")
    parser.add_argument("--n", type=int, default=None, help="--sample: override configs/config.json: phase0.discovery_sample.n")
    parser.add_argument("--seed", type=int, default=None, help="--sample: override configs/config.json: phase0.discovery_sample.seed")
    parser.add_argument("--neighborhood-size", type=int, default=15, help="--discover: nearest-neighbor rows per seed anchor")
    parser.add_argument("--suggestion-version", type=int, default=None, help="--embed-anchors: default is the latest suggestion")
    args = parser.parse_args()

    if not any([args.sample, args.discover, args.embed_anchors, args.embed_discovery_sample]):
        parser.error("Pass at least one of --sample/--discover/--embed-anchors/--embed-discovery-sample")

    if args.sample:
        do_sample(args)
    if args.discover:
        asyncio.run(do_discover(args))
    if args.embed_anchors:
        do_embed_anchors(args)
    if args.embed_discovery_sample:
        do_embed_discovery_sample(args)


if __name__ == "__main__":
    main()
