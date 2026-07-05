# Result iteration methodology

The pipeline's later stages (09–14) aren't run once and trusted — each run is
checked against its own generated reports, and issues found there feed back
into either a code fix or a documented methodological caveat, before the next
run. This file records that loop so it isn't implicit tribal knowledge.

## The loop

1. Run the variant chain end to end (`make run-scoring` → `run-features` →
   `run-clustering` → `run-event-study` → `run-factor-analysis` →
   `run-governance-sensitivity`, all with the same `--variant`).
2. Read every report the run produces (`reports/*__{variant}.txt`,
   validation summaries, console warnings) — not just "did it exit 0".
3. Classify each anomaly found:
   - **Bug** (code doesn't do what it claims to do) → fix in code, rerun the
     affected stage(s) downstream, re-check the report.
   - **Methodological limitation** (code does what it claims, the limitation
     is inherent to the design, e.g. BoW proxy recall) → do not silently
     patch the logic; document it as a caveat in README/thesis_plan instead.
4. Repeat from step 1 until a run's reports show no unexplained anomalies —
   only known, documented caveats.

## Worked example (2026-07-05, `rule_based` variant)

Running the chain surfaced two anomalies in `reports/did_power_check__rule_based.txt`:
`deepseek DiD` and `technical x tech-sector` specs were skipped for "missing
flags" (`high_defensive_pre2025`, `high_technical_pre2024`), even though the
underlying `D3 Technical` / `D8 Defensive` dimension scores exist in the
clusters table.

- **Root cause**: `12_event_study_panel.py` merged only
  `["ticker", "year", "cluster", "cluster_name"]` from the clusters parquet,
  silently dropping the `D1`–`D9` columns that `TREATMENT_DEFS` reads from —
  a bug, not a data gap.
- **Fix**: include the `D*` columns in the merge.
- **Re-run**: `run-event-study` re-executed for the affected variant; the
  previously-skipped specs should now populate in the power check.

The governance recall gap seen in the same run
(`is_governance_related` F1=0.50, R=0.33 in validation; 22.2% archetype churn
in `governance_sensitivity`) was **not** patched the same way in this first
pass — treated as a known limitation of the BoW proxy and documented as a
caveat instead. See the next worked example: once the *measurement* of that
gap was itself fixed, the gap turned out to be partly a bug, not purely a
limitation.

## Worked example 2 (2026-07-05, `rule_based` variant — validation coverage + proxy rules)

Revisiting the governance/promotional caveat above, in order:

1. **Coverage gap in the measurement itself.** The validation sample only had
   ~3 true governance-positive chunks (1.8% of 167) — nowhere near enough to
   trust a recall estimate on. Fix: `ensure_dimension_coverage()` in
   `val_01_sample_and_label.py` checks every binary dimension's positive count
   in the stratified draw and force-tops-up any short of 40 positives, rather
   than special-casing governance with a CLI flag (rejected in review — a
   generic check catches whichever dimension is rarest, not just the one
   already known to be a problem).
2. **Judge model was unconfigured.** `LLM_JUDGE_API_KEY` (OpenAI-compatible)
   wasn't set anymore; only the pipeline's own `NVIDIA_*` classifier vars
   were. Fix: pointed `LLM_JUDGE_*` at NVIDIA NIM too, but pinned to
   `qwen/qwen3-next-80b-a3b-instruct` — a different model family from the
   classifier's `meta/llama-3.3-70b-instruct` — so the judge isn't grading a
   close relative of itself. (First model choice, `qwen/qwen2.5-72b-instruct`,
   404'd — not in NIM's current catalog — caught by a 5-chunk smoke test
   before committing to the full run.)
3. **A second bug surfaced by the larger, coverage-guaranteed run:**
   `chunk_id = sha256(chunk_text)[:16]` (script 06) has no
   ticker/section/filing component, so repeated boilerplate within one filing
   collides on id (197 distinct ids affected). This fanned one labeled chunk
   out into several identical evaluation rows in `val_02`'s merge, inflating
   `n` and silently double-weighting some chunks. Fix: dedupe both sides on
   `chunk_id` before merging.
4. **With coverage and duplication fixed, the "limitation" from example 1
   turned out to be partly a genuine rule bug.** `is_promotional_proxy` fired
   on named-product mentions and deployment verbs ("launched", "copilot",
   "subscription") — i.e. "ships an AI product," not promotional *tone* —
   scoring F1=0.277 against the now-trustworthy judge labels. Grid-searching
   existing BoW features (`count_vague_words`, `count_negative_words`,
   `bow_sentiment_score`, `has_risk_factor`) against the 236-chunk validation
   set found `count_vague_words > count_negative_words AND NOT
   has_risk_factor AND bow_sentiment_score >= 0` at F1=0.400. Adding `has_compliance
   AND has_word_ai` to `is_governance_related_proxy` moved its F1 0.500→0.641
   (recall 0.333→0.723).
5. **Re-ran the full chain** (scoring → features → clustering → event study →
   governance sensitivity → factor analysis) so every downstream artifact
   reflects the corrected proxies, not just the validation report.

**New caveat this introduces**: both proxy formulas in step 4 were selected
by testing candidates directly against the validation sample used to report
their F1 — an in-sample tuning exercise, flagged explicitly in the README
rather than presented as an out-of-sample generalization estimate. See
[[bow_validation_methodology_critique]] point 4 (researcher degrees of
freedom in word-list/rule tuning) — this is exactly that risk, now realized
and disclosed rather than silently absorbed into "the model just does
better."
