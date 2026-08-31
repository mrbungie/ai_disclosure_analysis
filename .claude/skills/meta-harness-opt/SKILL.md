---
name: meta-harness-opt
description: >
  Run ONE meta-harness optimization iteration as the agentic proposer for one
  of the two tasks: "detection" (pre-classification of AI text,
  harnesses/detection/) or "classification" (dimension tagging of AI text,
  harnesses/classification/). The proposer reads prior candidates' code,
  scores, and traces, writes a NEW candidate program, and evaluates it on the
  search split. Use when the user asks to optimize, improve, iterate, or run
  an opt cycle on a harness. Only one optimization at a time (lock file).
  Args: detection | classification.
---

# Meta-harness optimization iteration (agentic proposer)

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

**Argument (required):** `detection` or `classification`. If unspecified, ask.

## Objective

Find the harness (candidate program) that does best on the stage's task,
under its selection metric, given everything currently in
`harnesses/<task>/`, `data/interim/eval/`, and `configs/config.json`.
"Best" is not a single pass — iterate for as many turns as it takes,
using whichever of the levers below actually moves the metric, until the
evidence says you've hit a real ceiling (not just that this particular
technique ran out of ideas). Levers exist so a ceiling from one of them
(a candidate's pattern vocabulary, the seed screen's keyword coverage, the
eval set's size or coverage) doesn't get mistaken for a ceiling on the
achievable classifier.

The two stages are NOT interchangeable — classification cannot be worked on
until detection is frozen, because detection's ACTIVE candidate produces the
candidate frame classification samples from:
- `detection` (stage 1): `classify(text: str) -> bool` on individual
  paragraphs — is this filing text AI-related? Selection metric: minimize
  candidate volume subject to weighted search recall >= the recall floor
  (`configs/config.json: seed_screen.recall_floor`) — recall is a gate, not
  something to maximize past the floor. Sampled from
  `data/interim/eval/eval_set_detection.parquet` (seed-screen-stratified
  paragraphs — see `scripts/seed_screen.py`).
- `classification` (stage 2): `classify(text: str) -> dict` with the six
  dimension bools (substantive, promotional, risk, governance, use-case,
  quantified), operating on CHUNK text (a merged +/-1 paragraph window
  around admitted paragraphs — `harness_fit.build_chunks`), not raw
  paragraphs. Selection metric: macro-averaged balanced accuracy (mean of
  sensitivity/specificity) per label, not F1. Sampled from
  `data/interim/eval/eval_set_classification.parquet`. Requires
  `data/processed/candidate_frame.parquet` to exist (written by
  `scripts/apply_harness.py --detection-only` after a detection freeze).

## Constraints

- **The holdout is never read or reasoned from during iteration.** All
  iteration is search-split only (`eval_harness.py` default). The test
  look happens once, at freeze, with explicit user agreement, and is
  gated on the recall floor for detection.
- **History is immutable.** One new candidate directory per iteration;
  never edit a previously evaluated candidate's `harness.py` — copy to a
  new name and change that.
- **A candidate exposes exactly the stage's `classify` contract.** How it's
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
  (`.claude/skills/meta-harness-opt/journals/OPT_LOCK`) held for the
  duration; refuse and report if one already exists.

## Levers

Everything here can be exercised more than once, in any order, across
iterations — nothing is a one-shot special case:
- **Rewrite the candidate program.** The default lever.
- **Edit the seed screen itself and regenerate it** (`scripts/seed_screen.py`)
  if you judge its keyword coverage inadequate — this is bigger than a
  candidate edit (it reshapes the population every future sample draws
  from), so do it deliberately with the evidence in the journal and commit
  message, and know already-drawn rows' cached `stratum` becomes stale
  relative to the regenerated screen (labels stay valid; only future
  sampling's bucketing shifts). Verification here is post-hoc — act, don't
  wait for a go-ahead on this specific action.
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

Read `harnesses/<task>/`: `ACTIVE`, every candidate's `harness.py`,
`notes.md`, `eval_search.json`, `trace_search.parquet` — the actual
misclassified texts, not just the score. `scripts/eval_harness.py
--leaderboard --task <task>` for standings. `git log` and the journal for
the story so far. If the stage's eval set doesn't exist yet
(`data/interim/eval/eval_set_<task>.parquet`), that's the first thing to
get (see Levers) — `--label` costs money, get a go-ahead first. For
classification, also check `data/processed/candidate_frame.parquet` exists.

## Freezing

Only with explicit user agreement, when iteration plateaus:
`eval_harness.py --task <task> --candidate <best> --split test` — one look,
auto-promotes to ACTIVE, spends the test split for this batch. Report the
result as-is. A spent test split means the next freeze needs a fresh
labeled eval set. After a detection freeze, run `apply_harness.py
--detection-only` to (re)write `data/processed/candidate_frame.parquet`.

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
   (`.claude/skills/meta-harness-opt/journals/<task>.md`; create with a
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
2. Commit everything (`[meta-opt/<task>] ...` + Co-Authored-By trailer for
   your model), push if the session has been pushing.
3. Delete `OPT_LOCK`.
4. Report: diagnosis, what changed, the metric delta, what's next.
