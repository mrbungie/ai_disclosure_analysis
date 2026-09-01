---
name: detection-opt
description: >
  Run ONE meta-harness optimization iteration as the agentic proposer for
  the detection task (pre-classification of AI text, harnesses/detection/):
  classify(text: str) -> bool on individual paragraphs. The proposer reads
  prior candidates' code, scores, and traces, writes a NEW candidate
  program, and evaluates it on the search split. Use when the user asks to
  optimize, improve, iterate, or run an opt cycle on the detection harness.
  Only one optimization at a time (lock file). No argument needed.
---

# detection-opt: meta-harness optimization iteration (agentic proposer)

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
not a scripted loop. If you find yourself building a reusable tool for one
specific analysis technique, stop: write the analysis inline for this
iteration, or just do the reasoning directly, and let the technique itself
live in your reasoning and the journal, not in a maintained script.

Sibling skills, same pattern, different task: `phase0-opt` (embedding/
ConceptSeed candidates that widen this task's seed screen, `harnesses/
phase0/`) and `classification-opt` (six-dimension tagging,
`harnesses/classification/` — cannot start until THIS task is frozen,
since detection's ACTIVE candidate produces the candidate frame
classification samples from).

## Objective

Find the harness (candidate program) that does best on detection, under
its selection metric, given everything currently in `harnesses/detection/`,
`data/interim/eval/`, and `configs/config.json`. "Best" is not a single
pass — iterate for as many turns as it takes, using whichever of the
levers below actually moves the metric, until the evidence says you've
hit a real ceiling (not just that this particular technique ran out of
ideas). Levers exist so a ceiling from one of them (a candidate's pattern
vocabulary, the seed screen's keyword coverage, the eval set's size or
coverage) doesn't get mistaken for a ceiling on the achievable classifier.

`classify(text: str) -> bool` on individual paragraphs — is this filing
text AI-related? Selection metric: minimize candidate volume subject to
weighted search recall >= the recall floor (`configs/config.json:
seed_screen.recall_floor`) — recall is a gate, not something to maximize
past the floor. Sampled from `data/interim/eval/eval_set_detection.parquet`
(seed-screen-stratified paragraphs — see `scripts/seed_screen.py`, whose
`hit` column is lexical-only unless the `phase0-opt` skill's frozen
candidate is enabled, in which case it's `lexical_hit | semantic_hit` —
either way this task's own candidates stay purely a function of `text`).

## Constraints

- **The holdout is never read or reasoned from during iteration.** All
  iteration is search-split only (`eval_harness.py` default). The test
  look happens once, at freeze, with explicit user agreement, and is
  gated on the recall floor.
- **History is immutable.** One new candidate directory per iteration;
  never edit a previously evaluated candidate's `harness.py` — copy to a
  new name and change that.
- **A candidate exposes exactly `classify(text: str) -> bool`.** How it's
  implemented — rules, regex, an LLM call, anything — is open; whatever
  form it takes, it's scored the same way by `eval_harness.py`, and an
  LLM-backed candidate means real API calls at corpus scale (accepted
  tradeoff, already made).
- **The construct's boundary is the seed screen's own keyword list**
  (`configs/config.json: seed_screen.ai_keywords`) — not a narrower reading
  you introduce on your own. Whether a TERM belongs in the construct at all
  is a human call if a disagreement surfaces; whether an in-scope term's
  PATTERN needs tightening (a qualifier, an exclusion, a phrase instead of
  a bare word) so it stops firing on unrelated text is an ordinary
  engineering decision — yours to make and evaluate, not to hand back.
- **One iteration, one journal entry, one commit.** Lock file
  (`.claude/skills/detection-opt/journals/OPT_LOCK`) held for the
  duration; refuse and report if one already exists.

## Levers

Everything here can be exercised more than once, in any order, across
iterations — nothing is a one-shot special case:
- **Rewrite the candidate program.** The default lever.
- **Edit the seed screen itself and regenerate it** (`scripts/seed_screen.py`)
  if you judge its LEXICAL keyword coverage inadequate — this is bigger than
  a candidate edit (it reshapes the population every future sample draws
  from), so do it deliberately with the evidence in the journal and commit
  message, and know already-drawn rows' cached `stratum` becomes stale
  relative to the regenerated screen (labels stay valid; only future
  sampling's bucketing shifts). Verification here is post-hoc — act, don't
  wait for a go-ahead on this specific action. If the gap looks semantic
  rather than lexical (real AI content with no matching keyword, however
  broad the list), that's the `phase0-opt` skill's job, not this one's.
- **Get more data.** More search-sample rows, more labeling, a bigger
  holdout, a human-audit pass — the search sample is meant to be topped up
  as needed (`build_eval_set.py`'s sampling is additive/accumulative,
  never discards prior work). Labeling costs money: decide you want it,
  then get the user's go-ahead for that specific spend before running it.
- **Use whatever exploration method actually answers the question in front
  of you** — read traces, grep the corpus, write a one-off script, reason
  it through by hand, ask an LLM to help design a candidate. Pick the
  method to fit the question; don't let a technique you used once become
  the thing you always reach for.

## Before proposing

Read `harnesses/detection/`: `ACTIVE`, every candidate's `harness.py`,
`notes.md`, `eval_search.json`, `trace_search.parquet` — the actual
misclassified texts, not just the score. `scripts/eval_harness.py
--leaderboard --task detection` for standings. `git log` and the journal
for the story so far. If the eval set doesn't exist yet
(`data/interim/eval/eval_set_detection.parquet`), that's the first thing
to get (see Levers) — `--label` costs money, get a go-ahead first.

## Freezing

Only with explicit user agreement, when iteration plateaus:
`eval_harness.py --task detection --candidate <best> --split test` — one
look, auto-promotes to ACTIVE, spends the test split for this batch.
Report the result as-is. A spent test split means the next freeze needs a
fresh labeled eval set. After freezing, run `apply_harness.py
--detection-only` to (re)write `data/processed/candidate_frame.parquet` —
this is what unblocks `classification-opt`. If `configs/config.json:
phase0.enabled` is true, also re-run `scripts/seed_screen.py` so the
sampling strata reflect the new lexical baseline.

A freeze is not verified by the search-split score alone. Before treating
it as settled, account for: the holdout's WEIGHTED metrics (not just raw —
they can diverge sharply from search, and if they do, that divergence is
the finding); what share of the corpus the candidate actually admits,
broken down by a meaningful grouping (sector, filing period, whatever the
task suggests) to sanity-check the pattern makes sense rather than being
flat or inverted; and a handful of real examples read for a specific
insight, not as a substitute for the actual scoring.

## Close

1. Append the journal entry
   (`.claude/skills/detection-opt/journals/detection.md`; create with a
   short header + a `000` entry describing current state if missing).
   Template:
   ```
   ## NNN — YYYY-MM-DD — <one-line summary>
   - Candidates before: <ACTIVE + best-on-search>
   - Evidence read: <what you looked at, what it showed>
   - New candidate / seed change / data obtained: <what and why>
   - Result: <metric before -> after>
   - Test split: <untouched | spent this iteration: result>
   - Commit: <hash>
   ```
2. Commit everything (`[meta-opt/detection] ...` + Co-Authored-By trailer
   for your model), push if the session has been pushing.
3. Delete `OPT_LOCK`.
4. Report: diagnosis, what changed, the metric delta, what's next.
