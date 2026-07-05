# Critique: BoW calibration → LLM-judge → descriptive/DiD methodology

Stated process being evaluated:

1. Apply a hypothesized `rule_based` (BoW rules + clustering + factor
   analysis) version, then judge it with an LLM-judge harness (Claude Code +
   LLM runs). Measure, iterate, improve until a threshold is crossed.
2. Only then run descriptive analysis + DiD.

## What's genuinely strong

- Separating `rule_based`/`llm_full` and blocking the LLM-judge from
  validating `llm_full` (val_01/val_02 circularity guard) avoids the common
  mistake of judging an LLM classifier with another LLM and calling it
  validation.
- Reporting F1/precision/recall per dimension, a confusion matrix, and
  cutoff sensitivity (±10% shift → 4-6% archetype churn) instead of just
  "accuracy looks fine" — unusually rigorous for a dictionary/BoW method.
- DiD power check gates cell sizes *before* the regression is run.
- Calibrating the measurement instrument before running the descriptive/DiD
  stage is the right order — it avoids tuning the instrument on the outcome
  you're trying to prove.

## Where it's exposed to criticism

1. **The LLM judge is not ground truth, it's another instrument.** The
   validation report literally calls it "LLM as ground truth," but there is
   no human-coded gold standard anywhere in the chain. Obvious reviewer
   question: how do you know the judge itself is right? Currently no answer.

2. **"Improved until a threshold is crossed" overstates what happened.**
   Actual numbers: `is_ai_related` F1=0.86 (crossed, strong), but
   `is_promotional` F1=0.57 and `is_governance_related` F1=0.50 / R=0.33
   never crossed a comparable bar. What really happened: iterate on some
   dimensions, hit a wall on governance, downgrade to a documented caveat
   instead of continuing to iterate. Defensible, but a different claim than
   "converged" — should be described as such or a reviewer will spot the
   mismatch.

3. **Validation sample size, especially for rare classes.** 167 of 4,204
   chunks. For a rare class like governance, recall=0.333 may rest on very
   few true positives — could swing a lot on a different random draw.
   Report raw TP/FP/FN counts (not just the ratio), ideally with a
   confidence interval, before treating it as stable.

4. **Researcher degrees of freedom in the word lists themselves.**
   Sensitivity analysis exists for the numeric cutoffs (`d5_governance_min`,
   etc.) but not for the hand-curated word lists / governance regexes
   (`VAGUE_WORDS`, `SPECIFIC_WORDS`, `ETHICS_POLICY_PAT`, ...). No check for
   "what if a different reasonable list had been chosen" — the more common
   garden-of-forking-paths critique for dictionary methods, currently
   unaddressed.

5. **Reproducibility of the calibration itself.** The threshold-crossing
   decision was made against one LLM-judge snapshot at one point in time.
   Unclear whether the same dimensions would still pass on a rerun or a
   different model version. Worth recording which judge model/version and
   when, since that's now part of the measurement instrument.

## Recommended framing

Don't claim uniform convergence. State explicitly: some dimensions cleared
a validation bar and are trusted at face value (AI-relatedness, risk);
others didn't (governance, promotional tone) and are reported with an
explicit sensitivity analysis/caveat rather than trusted outright. That is
more defensible than "measured and improved until threshold crossed," and is
consistent with what `governance_sensitivity__rule_based.txt` and
`validation_summary__rule_based.txt` actually show.

See also [[methodology_iteration]] for the run/check/fix-or-caveat loop this
critique assumes.
