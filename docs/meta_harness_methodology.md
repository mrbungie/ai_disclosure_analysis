# Meta-harness methodology

How this pipeline's measurement layer is built and validated, written for the
thesis methodology chapter. Companion documents:
[meta_harness_plan.md](meta_harness_plan.md) (original engineering plan),
[meta_harness_map.html](meta_harness_map.html) (system diagram),
[design_assessment_2026-08-31.md](design_assessment_2026-08-31.md) (design review),
[universe_expansion_plan.md](universe_expansion_plan.md) (sampling frame).

## The pattern

The methodology instantiates the structure of *Meta-Harness: End-to-End
Optimization of Model Harnesses* (`docs/arXiv-2603.28052v1/`): a **harness**
is a self-contained program performing a task; an **agentic proposer** with
filesystem access to every prior candidate's source code, scores, and
execution traces proposes new candidates; each candidate is evaluated on a
**search set** (spent freely) and the final one on a held-out **test set**
(looked at once). The objective is the paper's:

    H* = argmax_H  E[ r(H, x) ]

Two tasks in this pipeline follow the pattern, each with its own harness:

1. **Detection** — pre-classification of filing text:
   `classify(text) -> bool` (is this AI-related at all?).
2. **Classification** — dimension tagging of AI text:
   `classify(text) -> {substantive, promotional, risk, governance,
   use-case-specific, quantified}` — defined on task 1's positives.

At corpus scale the two frozen harnesses compose exactly in that order
(`scripts/apply_harness.py`): detection pre-classifies every paragraph;
classification tags the ones that pass. Both are deterministic keyword/regex
programs — interpretable, exactly replicable, auditable line by line, which
suits an economics/finance measurement exercise better than a black-box
classifier.

## The pieces

- **Candidates** (`harnesses/<task>/<candidate>/harness.py`): one
  self-contained stdlib-only file per candidate. Keywords, patterns, staging
  — everything is internal to the candidate; the fixed part is only the
  task contract above. Candidates are immutable once evaluated: a new idea
  is a new numbered directory with a `notes.md` stating the proposer's
  rationale. `harnesses/<task>/ACTIVE` names the frozen candidate. Both
  tasks start from a deliberately minimal `000_seed` (detection: three
  keywords; classification: one obvious pattern per dimension) — the floor
  every later candidate is measured against, so the growth trajectory is
  itself a documented result.

- **Reward** (`scripts/build_eval_set.py`): one labeled evaluation set
  serves both tasks — paragraphs sampled from the whole corpus, stratified
  by the ACTIVE detection candidate's own prediction (half predicted
  positive, half negative, industry-proportional within each), each row
  carrying its inverse sampling probability (`sampling_weight`). All seven
  labels per row come from an LLM labeler in one call, over the full text.
  The **search/test split is fixed at sampling time** and stored in the
  data; nobody re-splits.

- **Evaluate(H, X)** (`scripts/eval_harness.py`): runs a candidate on the
  eval rows of its task (detection: all rows; classification: the
  AI-labeled rows) and writes to the candidate's directory its scores
  (per-label P/R/F1 and (macro-)F1 reward, raw AND weighted) plus a
  per-instance trace — predictions vs labels with a `wrong` column. Raw
  scores the balanced sample as drawn and is optimistic by design; the
  weighted number is the honest population estimate.

- **The proposer** (the `meta-harness-opt` skill — a coding agent): reads
  the leaderboards, the prior candidates' code, and the actual misclassified
  texts in the traces; writes ONE new candidate per iteration; evaluates it
  on the search split; appends a journal entry; commits. One optimization at
  a time (lock file). The journals
  (`.claude/skills/meta-harness-opt/journals/`, local) plus the committed
  candidates and `[meta-opt/*]` commit history are the proposer's trace.

- **The freeze**: `eval_harness --split test` — one look per task per
  labeled batch, guarded in code (`TEST_LOOK` records the eval set's
  fingerprint). It promotes the candidate to ACTIVE and reports the result
  as-is, favorable or not. A spent test split forces a fresh labeled batch
  before the next freeze.

## The measurement-error chain

    human  <-- Cohen's kappa -->  reward labels  <-- test F1 -->  harness  -->  corpus
          (agreement_check.py)                   (eval_harness)          (apply_harness, deterministic)

- **Human ↔ reward labels**: labeling auto-exports a balanced audit workbook
  (stored labels on a separate sheet so they can't anchor the rater);
  `agreement_check.py` scores raw agreement and Cohen's kappa per label.
  Kappa is the reported number; the freeze output carries it or an explicit
  UNVALIDATED warning.
- **Harness ↔ reward labels**: the single-look test F1 per task, raw and
  weighted side by side.
- **Harness → corpus**: deterministic, zero additional error — anyone with
  the candidate file reproduces every classification. In the corpus output,
  dimension tags on non-AI paragraphs are `NA`, not `False`: "not evaluated"
  stays distinguishable from "evaluated negative".

## Honest-reporting rules (limitations section, pre-committed)

- Search-split scores are never cited as results; only test-split numbers
  are, and only their weighted variants as population claims.
- Test splits are spent after one look; candidate directories record every
  evaluation, making the amount of "search until it works" explicit.
- The LLM labeler is not ground truth; kappa against a human bounds what
  "reward agreement" can mean, and weak kappa on a dimension weakens every
  downstream number on that dimension — acknowledged, not patched.
- Candidates are immutable and the ACTIVE pointers are the only state:
  every corpus number is traceable to two named candidate files.
