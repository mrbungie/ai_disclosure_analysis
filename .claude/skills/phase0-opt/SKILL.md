---
name: phase0-opt
description: >
  Run ONE meta-harness optimization iteration as the agentic proposer for
  the phase0 task (concept discovery — embedding/ConceptSeed-backed
  candidates that widen the detection task's seed screen beyond literal
  keyword matching, harnesses/phase0/ — see docs/distillation_map.html
  §0): classify(text: str) -> bool, scored as unique recall gain over the
  current detection ACTIVE. The proposer draws a discovery sample, runs
  the embeddings-first discovery pipeline (nearest-neighbor neighborhoods
  + discriminative lexical terms + grounded LLM refinement) to curate a
  ConceptSeed, writes a NEW candidate program, and evaluates it on the
  search split. Use when the user asks to optimize, improve,
  iterate, or run an opt cycle on the phase0 / concept-discovery harness.
  Only one optimization at a time (lock file). No argument needed.
---

# phase0-opt: meta-harness optimization iteration (agentic proposer)

You (Claude Code) are the proposer, exactly as the Meta-Harness paper
describes it (`docs/arXiv-2603.28052v1`, Section 3: "the proposer P is
Claude Code") — a full coding agent with **unrestricted** filesystem access
via your own tools (Bash, Read, Grep, Edit, Write), not a separate
constrained agent with a pre-picked toolset. This skill is the paper's
"minimal domain-specific skill that describes where to write new harnesses,
how to inspect previous harnesses and their execution traces, and what
files it can and cannot modify" — nothing more. It states the objective, the
constraints, and the levers available to you. It deliberately does NOT
prescribe a method: no bespoke inspection functions, no worked techniques,
no maintained analysis scripts. Deciding what evidence to gather, what to
try, and how, is the job — that's what makes you the search operator and
not a scripted loop.

Sibling skills, same pattern, different task: `detection-opt` (the lexical
seed-screen-driven `classify(text) -> bool` task this one measures itself
against) and `classification-opt` (six-dimension tagging, blocked on
detection, not on this task).

## Objective

Phase 0's job is NOT "be a good AI detector" — that's `detection-opt`'s
job. It's **"catch AI-disclosure text the current detection ACTIVE
structurally misses"** — real content phrased without any keyword,
however broad the seed screen's list gets (e.g. "Rivian Autonomy
Platform," "HR virtual assistant," an "automated platform" description
that never says "AI" or "machine learning"). Given everything currently
in `harnesses/phase0/`, `harnesses/detection/ACTIVE`,
`data/interim/eval/eval_set_detection.parquet` (same file detection uses —
same underlying construct), and `configs/config.json: phase0`, find the
candidate with the highest unique recall gain that still clears the
volume ceiling. Iterate for as many turns as it takes, until the evidence
says you've hit a real ceiling.

`classify(text: str) -> bool` — same contract and eval set as detection
(only the implementation family differs: embedding similarity against a
`concept_seed.json`, typically, via `scripts/phase0_discovery.
semantic_hit()` — see `harnesses/phase0/000_no_concepts` for the minimal
shape). Selection metric: maximize UNIQUE recall gain over the CURRENT
`harnesses/detection/ACTIVE` (`added = phase0_pred & ~detection_pred`,
`unique_recall_gain = P(added | label=True)`), subject to
`added_volume_frac <= configs/config.json: phase0.max_added_volume_frac`
— a ceiling, not a floor, since here recall is what's maximized and
volume is what's kept in check (mirrors detection's "constrained, not
maximized" framing, inverted).

**No hard block on this task** (unlike classification's dependency on a
frozen detection) — `harnesses/detection/ACTIVE` always exists (at least
`000_seed`). But results are only meaningful RELATIVE to whichever
detection candidate is currently ACTIVE: if detection's ACTIVE changes
later, an earlier phase0 journal entry's numbers are stale evidence, not
wrong — note the detection candidate name/commit a given phase0 result
was measured against.

## Constraints

- **The holdout is never read or reasoned from during iteration.** All
  iteration is search-split only (`eval_harness.py` default). The test
  look happens once, at freeze, with explicit user agreement, and is
  gated on the volume ceiling.
- **History is immutable.** One new candidate directory per iteration;
  never edit a previously evaluated candidate's `harness.py` or
  `concept_seed.json` — copy to a new name and change that.
- **A candidate exposes exactly `classify(text: str) -> bool`.** Real
  embedding-model inference at corpus scale is the accepted cost of this
  task (same tradeoff already made for LLM-backed detection candidates).
- **Phase 0 never sees the fitting/estimation sample.** The discovery
  sample (`scripts/phase0_discovery.py --sample`) is independent of, and
  never checked against, `data/interim/eval/eval_set_detection.parquet`'s
  search/holdout rows — inducing a ConceptSeed from what a downstream run
  failed to catch is exactly the leakage this task exists to prevent
  ("saw what failed, patched an anchor, re-measured").
- **Lexical candidates surfaced during induction are suggestions, not
  edits.** A ConceptSeed's `lexical_candidates` field is material for the
  `detection-opt` proposer to consider folding into
  `seed_screen.ai_keywords` — this skill never edits that config itself.
- **One iteration, one journal entry, one commit.** Lock file
  (`.claude/skills/phase0-opt/journals/OPT_LOCK`) held for the duration;
  refuse and report if one already exists.

## Levers

Everything here can be exercised more than once, in any order, across
iterations — nothing is a one-shot special case:
- **Draw a discovery sample and run the embeddings-first discovery
  pipeline.** `scripts/phase0_discovery.py --sample` (accumulative, tops
  up toward `configs/config.json: phase0.discovery_sample.n`) then
  `--discover`: proposes seed anchors (1 LLM call, names/descriptions
  only), embeds them + the full discovery sample, builds a
  nearest-neighbor NEIGHBORHOOD per anchor, extracts discriminative
  lexical terms per neighborhood (frequency-ratio, no LLM call — this is
  the interpretable trace: "characterized by machine learning,
  AI-enabled, copilot," not just a bare similarity number), then grounds
  a second LLM call in each neighborhood's real excerpts + terms to
  confirm/discard the concept and write anchors from that evidence —
  writes `data/interim/phase0/concept_seed_suggestions/suggestion_v{N}.
  json` (raw material, not itself a candidate). Read the suggestion,
  curate further if the grounding still looks off — don't rubber-stamp
  it — and write the result into a NEW `harnesses/phase0/<name>/
  concept_seed.json` + `harness.py`.
- **Switch the active embedding model.**
  `configs/config.json: phase0.active_embedding_model` picks from the
  registry in `phase0.embedding_models` (BGE-M3, EmbeddingGemma-300M,
  gte-multilingual-base, multilingual-e5-large-instruct — swap freely).
  Embeddings are cached per model key
  (`data/interim/embeddings/<model_key>.parquet`) and NEVER deleted —
  switching models costs disk, not lost work, so trying several to
  compare is cheap. `--embed-anchors`/`--embed-discovery-sample` populate
  a new model's cache before building a candidate against it.
- **Tune the threshold baked into a candidate.** Each `harness.py` owns
  its own similarity threshold (no global config value) — a candidate
  copy with a different threshold, evaluated, is a legitimate new
  candidate, same as a regex tweak is for detection.
- **Curate/rewrite the concept anchors directly.** You are not bound to
  what `--discover` produced — read the search trace's false negatives,
  write better positive/negative anchors by hand, same as a detection
  proposer hand-writes regex exclusions from trace evidence. Anchors,
  discriminative terms, and neighborhood evidence are the discovery/
  interpretability layers respectively — final judgment on whether a
  concept is real stays yours, not the LLM's.
- **Use whatever exploration method actually answers the question in
  front of you** — read the search trace, grep the corpus for phrasing
  patterns, read a handful of `no_hit_filing_hits`-stratum paragraphs
  directly (already-judged data, no new LLM calls). Pick the method to
  fit the question.

## Before proposing

Read `harnesses/phase0/`: `ACTIVE`, every candidate's `harness.py`,
`concept_seed.json`, `notes.md`, `eval_search.json`, `trace_search.
parquet` — the actual texts this candidate added or false-added, not just
the score. Read `harnesses/detection/ACTIVE`'s current `harness.py` (the
baseline everything here is measured against). `scripts/eval_harness.py
--leaderboard --task phase0` for standings. `git log` and the journal for
the story so far. Check `data/interim/phase0/concept_seed_suggestions/`
for unused discovery output before running `--discover` again. If
`configs/config.json: phase0.enabled` was true for the latest
`scripts/seed_screen.py` run, also check whether the search sample
carries an `agreement_quadrant` column (`scripts/build_eval_set.py`
oversamples the two disagreement quadrants there) — that's the most
informative trace evidence this task has.

## Freezing

Only with explicit user agreement, when iteration plateaus:
`eval_harness.py --task phase0 --candidate <best> --split test` — one
look, auto-promotes to ACTIVE, spends the test split for this batch
(shared with detection's eval set — check `detection-opt`'s recent
activity before assuming the split is still fresh). Report the result as-
is. After freezing, if `configs/config.json: phase0.enabled` is true,
re-run `scripts/seed_screen.py` so the semantic hit signal actually takes
effect in sampling.

A freeze is not verified by the search-split score alone. Before treating
it as settled, read a sample of the ADDED hits directly — are they real
AI content the detection baseline missed, or is the candidate quietly
admitting generic boilerplate the negative anchors should have excluded?
`added_precision` in the trace is the number to distrust until you've
read a few of the texts behind it.

## Close

1. Append the journal entry
   (`.claude/skills/phase0-opt/journals/phase0.md`; create with a short
   header + a `000` entry describing current state if missing). Note
   which detection candidate this iteration's numbers were measured
   against. Template:
   ```
   ## NNN — YYYY-MM-DD — <one-line summary>
   - Candidates before: <ACTIVE + best-on-search>
   - Measured against: <detection ACTIVE name/commit at time of this run>
   - Evidence read: <what you looked at, what it showed>
   - New candidate / concept seed change: <what and why>
   - Result: <metric before -> after>
   - Test split: <untouched | spent this iteration: result>
   - Commit: <hash>
   ```
2. Commit everything (`[meta-opt/phase0] ...` + Co-Authored-By trailer for
   your model), push if the session has been pushing.
3. Delete `OPT_LOCK`.
4. Report: diagnosis, what changed, the metric delta, what's next.
