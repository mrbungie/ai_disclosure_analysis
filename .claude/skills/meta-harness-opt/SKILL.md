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

You are the proposer from the Meta-Harness pattern applied as a two-stage
distillation cascade (docs/distillation_map.html): a coding agent with
filesystem access to every prior candidate's source code, scores, and
per-instance traces. One iteration = inspect history → write one new
candidate program → evaluate it on the search split → journal → commit.

**Argument (required):** `detection` or `classification`. If unspecified, ask.

The two stages (fixed contracts; everything inside a candidate is yours to
design). They are NOT interchangeable — classification cannot be worked on
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

## 0. Journals and lock (this directory, gitignored)

Audit trail at `.claude/skills/meta-harness-opt/journals/`: one journal per
task (`detection.md`, `classification.md`) plus the lock `OPT_LOCK`.
- Lock: if `OPT_LOCK` exists, REFUSE and report its contents — one
  optimization at a time; the user must finish or explicitly abandon it.
  Otherwise create it (`<task> <ISO timestamp>`) first and delete it last.
- Missing journal (fresh clone): create it with the standard header
  (purpose, append-only rule, "committed evidence lives in harnesses/ and
  the [meta-opt/*] commits") and an entry `000 — initial state observed
  (reconstructed)` describing the task's current ACTIVE candidate.

Journal entry template (append-only, one per iteration):

```
## NNN — YYYY-MM-DD — <one-line summary>
- Candidates before: <ACTIVE + best-on-search>
- Evidence read: <traces/scores/code inspected; what the errors showed>
- New candidate: <name — what it changes and why>
- Search result: <reward raw/weighted vs predecessor, per-label deltas that matter>
- Test split: <untouched | spent this iteration: result>
- Commit: <hash>
```

## 1. Read the history (before proposing)

All under `harnesses/<task>/`:
1. `ACTIVE` — the frozen candidate; every `<candidate>/` dir: `harness.py`
   (source), `notes.md` (prior proposer rationale), `eval_search.json`
   (scores), `trace_search.parquet` (per-instance predictions vs labels,
   `wrong` column). Grep and read the ACTUAL misclassified texts — that is
   the point of the filesystem: diagnose WHY, not just how much.
2. `scripts/eval_harness.py --leaderboard --task <task>` for the standings.
3. `git log --oneline -15` for recent iterations; the journal for the story.
4. The stage's eval set exists? (`data/interim/eval/eval_set_<task>.parquet`).
   If not, stop: the user must run
   `build_eval_set.py --stage <task> --sample/--label` (`--label` costs
   money — never run it without their go-ahead in this conversation). For
   `classification`, also check `data/processed/candidate_frame.parquet`
   exists — it can't sample without it.

## 2. Iron rules

- **NEVER run `--split test`, read a test trace, or reason from test rows'
  labels during iteration.** All iteration is search-split only
  (`eval_harness.py` default). The test look happens once, at freeze, with
  explicit user agreement.
- One iteration = ONE new candidate directory with a coherent idea, its
  evaluation, one journal entry, one commit. Never edit a previously
  evaluated candidate's harness.py — copy to a new numbered candidate
  (`NNN_shortname`) and change that; history stays immutable.
- A candidate is one self-contained stdlib-only harness.py. Deleted legacy
  machinery (keyword miners, atom libraries — see git history around
  9e6138c and the pre-redesign scripts 05-12) is yours to crib patterns
  from, inside the candidate file.
- Propose from evidence AND domain knowledge: read the false
  negatives/positives in the trace, then write the patterns yourself. You
  are the search operator — there is no other miner.

## 3. Run ONE iteration

1. Pick the diagnosis from the traces (e.g. detection misses paragraphs
   mentioning ChatGPT/LLMs without the word "AI"; classification confuses
   promotional with substantive on realized-language sentences).
2. `mkdir harnesses/<task>/NNN_shortname/`, write `harness.py` (start from a
   copy of the best candidate) and `notes.md` (the rationale — what you saw,
   what you changed, what you expect).
3. `uv run python scripts/eval_harness.py --task <task> --candidate NNN_shortname`
   — repeat read→edit within THIS candidate only until its idea is cleanly
   expressed (avoid overfitting spirals: if you are tweaking thresholds to
   chase the search score, stop and journal that).
4. Compare against the leaderboard; for `detection` the number that matters
   is search recall against the floor (volume only breaks ties among
   eligible candidates); for `classification` it's macro balanced accuracy.

**Freezing** (only with explicit user agreement, when iterations plateau):
`eval_harness.py --task <task> --candidate <best> --split test` — one look,
auto-promotes to ACTIVE, spends the test split for this stage's eval-set
batch. Report the result as-is, favorable or not. Detection's freeze is
additionally gated: it refuses if the candidate's recorded search recall is
below the floor. A spent test split means the next freeze needs a fresh
labeled eval set. After a detection freeze, run
`scripts/apply_harness.py --detection-only` to (re)write
`data/processed/candidate_frame.parquet` — classification's eval set
samples from that frame, so it shifts whenever detection's ACTIVE changes.

## 4. Close

1. Append the journal entry. 2. Commit everything (`[meta-opt/<task>] ...`
   + Co-Authored-By trailer for your model) and push if the session has been
   pushing. 3. Delete `OPT_LOCK`. 4. Report: diagnosis, new candidate, search
   deltas, what the next iteration should look at.
