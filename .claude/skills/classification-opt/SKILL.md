---
name: classification-opt
description: >
  Run ONE meta-harness optimization iteration as the agentic proposer for
  the classification task (six-dimension tagging of AI text,
  harnesses/classification/): classify(text: str) -> dict of six binary
  labels, operating on chunk text. The proposer reads prior candidates'
  code, scores, and traces, writes a NEW candidate program, and evaluates
  it on the search split. Cannot start until the detection task is
  frozen. Use when the user asks to optimize, improve, iterate, or run an
  opt cycle on the classification harness. Only one optimization at a
  time (lock file). No argument needed.
---

# classification-opt: meta-harness optimization iteration (agentic proposer)

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

Sibling skills, same pattern, different task: `detection-opt` (must be
frozen before this one can start — its ACTIVE candidate produces the
candidate frame this task samples from) and `phase0-opt` (widens
detection's seed screen, no direct dependency on this task).

## Objective

Find the harness (candidate program) that does best on classification,
under its selection metric, given everything currently in
`harnesses/classification/`, `data/interim/eval/`, and
`configs/config.json`. "Best" is not a single pass — iterate for as many
turns as it takes, using whichever of the levers below actually moves the
metric, until the evidence says you've hit a real ceiling.

**Cannot start until `detection-opt` has frozen a candidate and
`data/processed/candidate_frame.parquet` exists** (written by
`scripts/apply_harness.py --detection-only`) — check this first; if it's
missing, this skill's task isn't ready yet.

`classify(text: str) -> dict` with the six dimension bools (substantive,
promotional, risk, governance, use-case, quantified), operating on CHUNK
text (a merged +/-1 paragraph window around admitted paragraphs —
`harness_fit.build_chunks`), not raw paragraphs. Selection metric:
macro-averaged balanced accuracy (mean of sensitivity/specificity) per
label, not F1 — a skewed base rate can't be won by predicting the
majority class. Sampled from
`data/interim/eval/eval_set_classification.parquet`.

## Constraints

- **The holdout is never read or reasoned from during iteration.** All
  iteration is search-split only (`eval_harness.py` default). The test
  look happens once, at freeze, with explicit user agreement.
- **History is immutable.** One new candidate directory per iteration;
  never edit a previously evaluated candidate's `harness.py` — copy to a
  new name and change that.
- **A candidate exposes exactly the `classify(text) -> dict` contract.**
  How it's implemented — rules, regex, an LLM call, anything — is open;
  whatever form it takes, it's scored the same way by `eval_harness.py`,
  and an LLM-backed candidate means real API calls at corpus scale
  (accepted tradeoff, already made).
- **The six dimensions are the codebook's definition** (see
  `docs/distillation_map.html` and `build_eval_set.py`'s
  `DIMENSION_SYSTEM_PROMPT`), not a narrower reading you introduce on
  your own. Whether a dimension's DEFINITION needs revisiting is a human
  call if a disagreement surfaces; whether a candidate's PATTERN for an
  in-scope dimension needs tightening is an ordinary engineering decision
  — yours to make and evaluate, not to hand back.
- **One iteration, one journal entry, one commit.** Lock file
  (`.claude/skills/classification-opt/journals/OPT_LOCK`) held for the
  duration; refuse and report if one already exists.

## Levers

Everything here can be exercised more than once, in any order, across
iterations — nothing is a one-shot special case:
- **Rewrite the candidate program.** The default lever.
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

Confirm `data/processed/candidate_frame.parquet` exists (see Objective).
Read `harnesses/classification/`: `ACTIVE`, every candidate's
`harness.py`, `notes.md`, `eval_search.json`, `trace_search.parquet` —
per-label misclassifications, not just the aggregate reward.
`scripts/eval_harness.py --leaderboard --task classification` for
standings. `git log` and the journal for the story so far. If the eval
set doesn't exist yet
(`data/interim/eval/eval_set_classification.parquet`), that's the first
thing to get (see Levers) — `--label` costs money, get a go-ahead first.

## Freezing

Only with explicit user agreement, when iteration plateaus:
`eval_harness.py --task classification --candidate <best> --split test` —
one look, auto-promotes to ACTIVE, spends the test split for this batch.
Report the result as-is. A spent test split means the next freeze needs a
fresh labeled eval set.

A freeze is not verified by the search-split score alone. Before treating
it as settled, account for: the holdout's WEIGHTED per-label metrics (not
just raw); each label's prevalence, so a high balanced accuracy on a rare
label isn't mistaken for something stronger than it is; and a handful of
real examples per label read for a specific insight, not as a substitute
for the actual scoring.

## Close

1. Append the journal entry
   (`.claude/skills/classification-opt/journals/classification.md`;
   create with a short header + a `000` entry describing current state if
   missing). Template:
   ```
   ## NNN — YYYY-MM-DD — <one-line summary>
   - Candidates before: <ACTIVE + best-on-search>
   - Evidence read: <what you looked at, what it showed>
   - New candidate / data obtained: <what and why>
   - Result: <per-label metric before -> after>
   - Test split: <untouched | spent this iteration: result>
   - Commit: <hash>
   ```
2. Commit everything (`[meta-opt/classification] ...` + Co-Authored-By
   trailer for your model), push if the session has been pushing.
3. Delete `OPT_LOCK`.
4. Report: diagnosis, what changed, the metric delta, what's next.
