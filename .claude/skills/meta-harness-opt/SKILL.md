---
name: meta-harness-opt
description: >
  Run ONE meta-harness optimization iteration as the agentic proposer (level 2)
  for one of the two harnesses: "detection" (ai_keywords, scripts 07-08) or
  "classification" (tagging atom pools/formulas, scripts 09-12). Use when the
  user asks to optimize, improve, iterate, or run an opt cycle on a harness,
  the prefilter keywords, or the tag formulas. Only one optimization may run
  at a time, enforced via a lock file. Args: detection | classification.
---

# Meta-harness optimization iteration (agentic proposer)

You are acting as the **level-2 proposer** from `docs/meta_harness_methodology.md`:
the coding agent that reads the full history (traces, reports, journals, git
log) and improves one harness — its level-0 state in `configs/config.json`
and/or its level-1 cycle code — one journaled iteration at a time.

**Argument (required):** `detection` or `classification`. If the user didn't
specify which harness, ask before doing anything else.

## 0. Acquire the lock (one optimization at a time)

- Lock file: `docs/journals/OPT_LOCK`.
- If it exists: REFUSE to proceed. Report its contents (which harness, when)
  and stop — tell the user to finish or explicitly abandon that iteration
  first (abandoning = they ask you to delete the lock).
- If not: create it with one line — `<harness> <ISO timestamp> <session>` —
  before touching anything else. Delete it as the last step of this skill,
  including on failure paths you control.

## 1. Read the state (before proposing anything)

Read, in this order:
1. The harness's journal — `docs/journals/harness1_detection.md` or
   `docs/journals/harness2_classification.md` — the full history of what has
   been tried and why.
2. The current level-0 state in `configs/config.json` (`prefiltering.*` for
   detection; `tagging.atom_pools` + `tagging.formulas` for classification).
3. The latest evidence: `reports/prefilter_fit_*` or `reports/tag_fit_*`
   (dev report, search trace JSON, self-check report). For weak spots, read
   the actual misclassified texts from the dev split parquet
   (`data/interim/{prefilter_fit,tag_fit}/dev_split.parquet`), not just scores.
4. `git log --oneline -15` for recent proposer iterations.

## 2. The iron rules (proposer discipline)

- **NEVER read, evaluate against, or reason from holdout labels or a holdout
  report produced this iteration.** All iteration happens against dev and the
  self-check, using the `--dev-only` flags. The holdout look happens at most
  once, at the END, only when you and the user agree the state is frozen.
- If the cycle's holdout is already spent (`reports/*_holdout_eval.txt`
  exists), a freeze this iteration requires a FRESH labeled batch first (07
  or 10) — never re-use the spent holdout.
- One iteration = one coherent change with measurable dev evidence, one
  journal entry, one commit. Do not batch unrelated changes.
- Labeling costs money (LLM judge): never run `--label` steps without the
  user's go-ahead in this conversation.

## 3. Run ONE iteration

For **detection**:
1. Baseline: `.venv/bin/python scripts/08_fit_prefilter_harness.py --dev-only`
   (and `--self-check` if the search code changed since last run).
2. Diagnose from the dev report + `reports/prefilter_fit_search_trace.json` +
   false negatives/positives in the dev split.
3. Propose ONE change: accept the search's keyword additions, adjust the
   level-1 code (mining, scoring, sampling), or conclude a fresh 07 batch is
   needed. Re-run `--dev-only` to measure it.

For **classification**:
1. Ensure atoms exist (`scripts/09_extract_keyword_atoms.py` ran after any
   re-chunk) and a labeled batch exists (else stop and tell the user to run
   10 with their judge key).
2. Baseline: `.venv/bin/python scripts/11_fit_tag_harness.py --dev-only`.
3. Diagnose per dimension against the always-True baseline in the dev report
   and the trace. For weak dimensions, read misclassified chunk texts.
4. Propose ONE change: add atoms from the reference menu
   (`tag_harness_defs.DIMENSION_FEATURE_POOLS`) into that dimension's
   `tagging.atom_pools`, or write a NEW atom regex in script 09 (it's code —
   keep it auditable and add it to the menu too), justified by the texts you
   read. Re-run `--dev-only` to measure the delta.

Freezing (either harness): only with explicit user agreement, on a fresh
holdout, by running the fit script WITHOUT `--dev-only`. Report the holdout
result as-is, favorable or not.

## 4. Close the iteration

1. Append a journal entry using the journal's template (state before,
   evidence read, change, dev validation numbers, holdout status, commit).
   Never rewrite past entries.
2. Commit everything from this iteration with message prefix
   `[meta-opt/detection]` or `[meta-opt/classification]`, and put the commit
   hash into the journal entry (amend or note "(this commit)").
3. Delete `docs/journals/OPT_LOCK`.
4. Report to the user: what changed, dev delta, what the next iteration
   should probably look at.
