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

## Worked example 3 (2026-07-05/07, `rule_based` variant — search/holdout discipline + prefilter recall)

Follow-up to example 2's in-sample-tuning caveat: before locking `rule_based`
in as the final approach, the 4 fine-dimension proxy formulas and the
prefilter's keyword list both needed a proper search-pool/holdout split, not
just "grid-search against the one validation sample and report that number."

1. **Fine-dimension split infrastructure** (`val_03_dedup_split.py`): the
   existing 236-chunk labeled sample is burned (it's what produced example
   2's formulas) — it stays as search-pool material forever, never becomes a
   holdout. Built a corpus-wide text dedup map, then drew a disjoint,
   governance-oversampled search-pool top-up (130) and a disjoint,
   **proportional** (no oversampling) holdout (350). First version of the
   holdout sampler stratified by year × dimension-combo with equal counts per
   stratum — that inflated rare combos exactly like the bug it was supposed
   to avoid (governance came out at 28.6% vs. 13% true prevalence). Fixed to
   stratify by year only, with size-proportional allocation. The CV search
   harness itself (steps 3–4 of the plan) is still not built — this session
   only produced the clean split.
2. **`is_ai_related` is a different kind of tunable.** It's not a boolean
   formula over BoW features like the other 4 — the proxy is `TRUE` constant
   within `ai_candidate_chunks.parquet` (no negative class exists inside the
   candidate corpus by construction). The actual tunable surface for this
   dimension is the **keyword list in scripts 05–06**, one phase upstream.
   Scoped explicitly: no negative-side sampling outside the candidate corpus
   (rejected as scope creep with no bearing on thesis conclusions), but the
   keyword list itself gets the same search/holdout discipline as the fine
   dimensions, one phase earlier.
3. **Prefilter recall check** (`val_08`/`val_09`): industry-stratified sample
   (1,468 paragraphs, 30/30 and 55/55 industry coverage) of the universe the
   prefilter excludes — 142/658 filings that never match any keyword at all,
   and ~618K/634K paragraphs inside matched filings that fall outside every
   keyword hit's ±1 window. Found one confirmed miss: Palantir's "AIP"
   product name (`\bai\b` doesn't match inside "AIP"). Verified narrowly
   before shipping — `aip`/`aiops` checked against the *entire* corpus first
   (not just the sample) to confirm zero false-positive risk, unlike a
   broader `AI[A-Z]+` acronym pattern which mostly matched noise (AIDS,
   AICPA, AIRCRAFT). Keywords added, scripts 06→14 re-run for `rule_based`
   (4,204 → 4,228 chunks).
4. **That fix itself broke the search/holdout rule in miniature**: the
   1,468-paragraph sample that surfaced the "AIP" miss was then used to
   decide the fix, and would have been reused to *report* the resulting
   recall — the same mistake as citing F1 on the sample used to tune the
   formula. Caught and corrected: that sample is now documented as burned
   search-pool material (`prefilter_recall_validation.txt`, recall≈0.95, not
   citable); a **fresh, never-seen holdout** (534 paragraphs, `val_10`/`val_11`,
   hard-locked against re-running) gives the real number.
5. **Two more bugs found on self-audit, both in the recall arithmetic, not
   the sampling:**
   - **Unit mismatch**: TP was `precision × n_chunks` (4,228 chunks) while FN
     was `rate × population` in raw paragraphs. Each chunk merges ~3.5
     paragraphs on average, so this understated TP relative to FN and made
     recall look far worse than it was (0.584 instead of the corrected
     0.831). Fixed by computing TP as `precision × covered_paragraphs`
     (14,815) — both sides of the fraction in the same unit.
   - **Duplicate rows in the precision sample**: `llm_labeled_sample__rule_based.parquet`
     has 239 rows but only 236 unique `chunk_id`s (3 duplicates, same label
     each time) — the precision mean was counting them twice. Effect was
     tiny (0.824→0.822) since the duplicates agreed, but fixed anyway:
     dedupe on `chunk_id` before averaging, matching `val_02`'s existing
     practice.
   - Both fixes were re-derivable from already-collected data — no new judge
     calls, no re-look at the holdout, just corrected aggregation.
6. **Re-chunking side effect, checked and handled**: adding `aip`/`aiops`
   changed paragraph-window boundaries for the 4 affected tickers
   (PLTR/PANW/DDOG/AVGO), orphaning 5 of the 236 labeled `chunk_id`s and 1 of
   the new holdout's. `val_02` already inner-joins on `chunk_id`, so this
   silently (and correctly) drops orphans rather than corrupting the merge —
   verified rather than assumed. `val_03`'s split was regenerated post-fix to
   avoid carrying a stale holdout entry forward.

**Final, holdout-based (not search-pool) prefilter recall for `rule_based`:
≈0.83** (`reports/prefilter_recall_holdout_validation.txt`). One further gap
(Snowflake using "ML" as a standalone acronym, missed by the same
full-phrase-only keyword pattern) was found in that same holdout and is
deliberately left **unfixed** — acting on it would burn this holdout the same
way `AIP` burned the search-pool sample. A third, never-seen batch is
required before touching the keyword list again.
