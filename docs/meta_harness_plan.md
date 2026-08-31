# Meta-harness restructuring plan

Approved 2026-08-31. Companion documents:
[design_assessment_2026-08-31.md](design_assessment_2026-08-31.md) (why),
[meta_harness_map.html](meta_harness_map.html) (system diagram, open in a browser).

## Framing

Modeled on *Meta-Harness: End-to-End Optimization of Model Harnesses*
(`docs/arXiv-2603.28052v1/`), adapted, not copied. The paper's core object is a
**harness** (deterministic code around a frozen model) optimized by an **outer
loop** that evaluates candidates on a search set, keeps full traces of every
candidate rather than compressed summaries, and reports a held-out result once.

This thesis inverts one detail but keeps the structure: here the harness
*replaces* the LLM at corpus scale — keyword regexes and boolean formulas,
interpretable and replicable, which suits an economics/finance paper better
than black-box classification — while the LLM sits in the **outer loop as the
judge** producing the reward signal. Same objective form:

    H* = argmax_H  E[ r(H, x) ]

where `r` is agreement (F1) with the LLM judge on labeled samples, searched on
a dev split only, with the holdout touched exactly once and every candidate's
evaluation logged to a search-trace file (the paper's "filesystem of prior
candidates" role).

The **same optimization cycle** — sample → LLM judge → dev-only search →
freeze → single holdout look → write back to config — is instantiated twice:

- **Cycle 1 — detection harness** (live): `ai_keywords` answering *"is this
  text AI-related?"* Fit by scripts 07–08, feeding the 05–06 prefilter.
- **Cycle 2 — classification harness** (to rebuild): per-dimension boolean
  keyword formulas answering *"what does the AI text say?"* over **six
  dimensions**: substantive, promotional, risk, governance, use-case
  specificity, quantification. Rebuilt from the stripped `val_06` machinery
  (commit `9e6138c^`) on the shared cycle template.

The shared cycle module is the meta-harness claim made concrete: one reusable
optimization loop, two application sites.

## Work items

### 1. Shared cycle module — `scripts/harness_fit.py`

Extracted machinery used by both cycles: regex building / false-positive
cleaning, plain and **inverse-probability-weighted** precision/recall/F1,
stratified dev/holdout split, stratified k-fold indices, the
**fold-stability-penalized dev score** (renamed from "CV" — nothing is trained
per fold; candidates are mined from all of dev; the holdout is the honest
number), search-trace flushing, the single-look holdout lock, and the
agreement-sample export/scoring helpers.

### 2. Cycle-1 fixes (scripts 07–08)

1. **Stratum reweighting** — 07 records each (stratum × industry) sampling
   fraction into the sample parquet as `sampling_weight`; 08 reports weighted
   (population-estimate) P/R/F1 alongside raw (balanced-sample) metrics on both
   dev and holdout, clearly labeled. The search itself still optimizes the
   unweighted score (weighted recall hinges on a handful of high-weight
   excluded-stratum positives — too noisy to steer a greedy search).
2. **Remove the 2,500-char truncation** — the judge sees the full paragraph.
3. **Rename "CV"** to fold-stability score in code, reports, README.
4. **Gate the config write** — `ai_keywords` is only overwritten when the
   fitted list does not underperform the baseline on holdout; otherwise warn
   and keep the baseline.
5. **`--self-check` mode** — remove a known keyword, confirm the search
   recovers it on dev only; never touches the real holdout, never writes
   config, writes `reports/prefilter_fit_selfcheck.txt`.
6. **Agreement sample** — export ~60 stratified already-LLM-labeled paragraphs
   to an Excel workbook for hand-labeling (human sheet has no LLM labels, to
   avoid anchoring; LLM labels live in a separate sheet), plus a scoring mode
   computing Cohen's kappa once filled in. Generated from the existing 700
   labeled rows — no new LLM calls needed.

### 3. Cycle-2 rebuild (scripts 09–12)

- **09 — keyword-atom features**: per-chunk boolean `has_*` regex atoms over
  `ai_candidate_chunks.parquet`, adapted from the stripped BoW extraction.
- **10 — sample + judge-label chunks**: shared cycle machinery; the judge
  labels each sampled chunk on all six dimensions (+ rationale); sampling
  weights recorded as in 07.
- **11 — fit tag harness**: per-dimension boolean-formula search over the
  atoms (dimension-scoped atom pools, as in old `val_06`), fold-stability
  penalty, per-dimension trace, one holdout look, frozen formulas written to
  config gated on holdout (baseline = best single atom).
- **12 — apply tags**: tag every candidate chunk with the frozen formulas →
  `data/processed/tagged_chunks.parquet` (chunk-level; firm-year aggregation is
  analysis, out of scope here).
- Agreement workbook for the six dimension labels too (same helper).

### 4. Docs / repo hygiene

- `docs/meta_harness_methodology.md` — the framing above, written for the
  methodology chapter; measurement-error chain (human → judge kappa → harness
  F1 → corpus) made explicit.
- README pipeline section and Makefile targets updated for 09–12 and the new
  modes; memory of decisions kept in the project docs, not just chat.

## Constraints carried over

- Holdout discipline: a spent holdout is never reused; disappointing results
  force a fresh labeled batch (07/10).
- Config writes only after a converged search, never mid-run.
- Search traces flushed after every round so interrupted runs stay auditable.
- The existing cycle-1 holdout lock (`reports/prefilter_fit_holdout_eval.txt`)
  stays honored: code fixes take effect on the *next* labeled batch; nothing
  re-rolls the already-spent holdout.
