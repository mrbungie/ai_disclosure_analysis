# Meta-harness methodology

How this pipeline's measurement layer is built and validated, written for the
thesis methodology chapter. Companion documents:
[meta_harness_plan.md](meta_harness_plan.md) (the engineering plan),
[meta_harness_map.html](meta_harness_map.html) (system diagram),
[design_assessment_2026-08-31.md](design_assessment_2026-08-31.md) (design review).

## The framing

The methodology adapts the structure of *Meta-Harness: End-to-End Optimization
of Model Harnesses* (`docs/arXiv-2603.28052v1/`). In that paper, a **harness**
is the deterministic code around a frozen language model, and an **outer
loop** searches over harness candidates: it evaluates them on a search set,
keeps the full record of every candidate tried (code, scores, traces — not
compressed summaries), and reports held-out performance once.

This thesis keeps the structure and inverts one role. Here the harness
*replaces* the LLM at corpus scale: keyword regexes and boolean formulas —
fully interpretable, exactly replicable, and auditable line by line, which
suits an economics/finance measurement exercise better than a black-box
classifier. The LLM sits in the **outer loop as the judge** producing the
reward signal the search optimizes against. The objective has the same form
as the paper's:

    H* = argmax_H  E[ r(H, x) ]

where `H` is a harness artifact (a keyword list; a boolean formula), `x` is a
unit of filing text, and `r` is agreement (F1) with the LLM judge's label.

## The optimization cycle

One reusable cycle (implemented once, in `scripts/harness_fit.py`) fits every
harness:

1. **Stratified sample** of the relevant text population, with each row's
   inverse sampling probability (`sampling_weight`) recorded at draw time.
2. **LLM judge labels** the sample (full text, no truncation). The judge is a
   different model family from anything the harness approximates.
3. **Dev-only search** over harness candidates, scored by a
   **fold-stability-penalized** dev F1 (mean across stratified folds minus a
   variance penalty). This is deliberately *not* called cross-validation:
   nothing is trained per fold and candidates are proposed from all of dev,
   so the dev score is optimistic by construction. Every candidate evaluated
   in every round is flushed to a JSON **search trace** — the cycle's
   equivalent of the paper's filesystem of prior candidates.
4. **Freeze** the winning artifact.
5. **Single holdout look**: the frozen artifact is evaluated once on a
   disjoint holdout; the report file then locks — re-running the fit against
   the same holdout is refused in code. A disappointing holdout stands as
   reported; improving further requires a fresh labeled batch.
6. **Gated config write**: the artifact enters `configs/config.json` only if
   the holdout supports it (cycle 1: not underperforming the incumbent
   baseline; cycle 2: beating the always-True prevalence exploit).

### Instantiation 1 — detection ("is this text AI-related?")

Scripts 07–08 fit `ai_keywords`, the regex list scripts 05–06 use to flag AI
paragraphs and cut candidate chunks. Sampling is half from the currently
flagged stratum (precision), half from the currently excluded stratum
(recall), industry-proportional within each; candidates are 1–2-gram tokens
mined from dev false negatives and added greedily. The search machinery is
itself validated by a synthetic **self-check** (`08 --self-check`): drop
known keywords, confirm the search recovers their F1 — run on dev only, so
nothing is spent.

### Instantiation 2 — classification ("what does the AI text say?")

Scripts 09–12 fit one boolean formula per disclosure dimension over ~345
keyword/regex **atoms** extracted per chunk (09). The six dimensions answer
the proposal's facets directly: **substantive, promotional, risk-related,
governance-related, use-case-specific, quantified**. The candidate space per
dimension is bounded and fully enumerated — single atoms, negations, and
AND/OR/AND-NOT pairs of the top-15 singles — so the trace contains every
candidate's score with no path dependence. The pre-strip production formulas
are kept in the candidate set as baselines: the search either beats them on
the merits or confirms them. Frozen winners are applied to the corpus by 12.

## The measurement-error chain

Every link between "what a human would say" and "what the corpus numbers say"
is measured, and each estimate names its link:

    human  <-- Cohen's kappa -->  LLM judge  <-- holdout F1 -->  keyword harness  -->  corpus measurement
           (agreement_check.py)              (08 / 11 reports)                     (05-06 / 12)

- **Human ↔ judge**: `scripts/agreement_check.py` exports a balanced,
  hand-labelable Excel subsample per cycle (judge labels on a separate sheet
  so they can't anchor the rater) and computes raw agreement and Cohen's
  kappa once filled in. Kappa is the reported number; raw agreement on the
  deliberately balanced draw is descriptive only.
- **Judge ↔ harness**: the single-look holdout F1, reported **raw** (on the
  sample as drawn) and **inverse-probability weighted** (population
  estimate) side by side. The weighted number matters wherever the sampling
  design oversampled a stratum — cycle 1's 50/50 candidate/excluded draw
  overstates raw recall by design, and the weights undo exactly that.
- **Harness → corpus**: deterministic, zero additional error — this is the
  point of a keyword harness. Anyone with the config can reproduce every tag.

## Honest-reporting rules (limitations section, pre-committed)

- The dev search score is optimistic and is never cited as a result; only
  holdout numbers are.
- Holdouts are spent after one look. Search traces record how much "search
  until it works" happened, making the multiplicity explicit rather than
  hidden.
- The judge is one LLM, not ground truth; kappa against a human is the bound
  on what "judge agreement" can mean. Weak kappa on a dimension weakens every
  downstream number on that dimension — that propagation is acknowledged, not
  patched.
- A dimension whose fitted formula cannot beat the always-True baseline on
  holdout is dropped from the corpus tagging (11 refuses to freeze it), not
  quietly shipped.
