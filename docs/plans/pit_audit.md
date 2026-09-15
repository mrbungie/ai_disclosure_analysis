# Point-in-time (look-ahead) audit: scripts/gold/** and scripts/analytics/**

Scope: every script under `scripts/gold` and `scripts/analytics` except `scripts/deprecated`. The audit was read-only. Empirical checks ran on the gold parquets already on disk (built 2026-09-15 17:00–18:29) using `scratchpad/checks.py` and `checks2.py`; no builders were run.

Per the user's instruction, the difference-in-differences analyses (channel gap DiD, shock DiD and event studies) are out of scope and appear only briefly in section 5. The audit concentrates on what feeds the remaining predictive results: call beta, crash risk and the archetype battery, the incremental signal, and the washing score used as a predictor.

PIT convention used for the fixes (`scripts/common/pit.py`):
- every event row has an `available_date`;
- accumulating constructs are quarterly snapshots (`as_of_date` = 1 Jan/Apr/Jul/Oct), built only from rows with `available_date < as_of_date`;
- events receive snapshots through `pit.asof_join(event_date=...)`, a backward join with exact matches allowed.

Severity scale:
- **HIGH**: a predictive regression coefficient can be contaminated by future information.
- **MEDIUM**: timing ambiguity or overlap, a full-sample parameter in a predictor, or a construct bug that changes timing.
- **LOW**: descriptive only, or negligible.

---

## 0. Summary

| # | Sev | Where | Problem | Empirical check |
|---|-----|-------|---------|-----------------|
| H1 | HIGH | `analytics/crash_archetypes/call_crash_regressions.py:78-82`, `call_archetype_full_battery.py:102-106, 197-200` + builder `gold/crash_archetypes/build_call_crash_and_archetypes.py` | Yearly expanding archetype of the call's own calendar year (known issue). The builder also has further leaks, listed in H1 | 97.4% of calls get a label. Median look-ahead is 194 days; 77% of calls have more than 90 days. The label differs from the prior-year (lagged) label in 32% of calls. It matches the new as-of snapshot label in only 52% |
| H2 | HIGH | `analytics/call_beta/call_beta_config_decoupling_asof.py:49-52, 67-68` | Quarterly archetype weights are matched on the fiscal quarter label in `call_accession_number`, compared against a calendar cutoff quarter built from filing dates. Off-calendar fiscal years push the label ahead of the real date | 18.3% of matched calls get a cutoff quarter that ends on or after the call date (median 109 days, maximum 230) |
| H3 | HIGH | `gold/crash_archetypes/build_call_fundamentals_panel.py:60-61,151` → `next_revenue_yoy_pre`, used in `call_beta_generalized_targets.py:56` and `call_archetype_full_battery.py:65` | The "pre" control is next-fiscal-year revenue growth from the last 10-K before the call. That fiscal year usually ends after the call | 75.7% of non-null `next_revenue_yoy_pre` values cover a fiscal year that ends after the call (median 144 days later) |
| H4 | HIGH | `analytics/shock/incremental_signal.py:74-76, 96-150` + `gold/consolidate/firm_year/*` + `disclosure_volume/build_covariates_firm_year.py:45` | Firm-year join on calendar filing year. AI text covers every filing of year t. Outcomes are anchored to the 10-K filing date (beta, idiosyncratic volatility, vol_pre_60d and P/S use windows before that date). `next_revenue_yoy` covers the fiscal year that the year-t 10-Qs already report | 72% of year-t filings (34% of AI frames) are filed after that year's 10-K |
| H5 | HIGH | `incremental_signal.py:70,100-101,165-167` (M3) | Static full-sample archetype (`posture_archetype_static`) used as a regressor. Its result, the archetype ΔR², is in `data/results/shock/incremental_signal.json` | By construction: one AA fit and scaler over all years, applied to every year |
| M1 | MED | `gold/washing/washing_score.py:140,195,210-214`; `consolidate/predictions/washing_score.py:80,90-95` → `incremental_signal.py:141` (M4) and `sec_comment_letter_cases.py` | W uses percentile ranks pooled over 2021-2026. The grounding priors and the median fill are also fitted on all years | Correlation between pooled W and within-year W is 0.886. Mean pooled W drifts from −0.14 (2021) to +0.09 (2026). Mean |ΔW| is 0.087 |
| M2 | MED | `gold/posture/build_strategy_dimensions.py:163-164` → `covariates/firm_year/posture.parquet` | The firm-year posture covariate uses a shrinkage prior and a `disclosure_intensity` percentile rank pooled over all years. It feeds the H1 cross-section and `oos_validation.py` | Mean `disclosure_intensity` by year: 0.36 (2021) → 0.64 (2026) |
| M3 | MED | `gold/posture/posture_features.py:226-266` (`global_posture_pca`, `name_vertices`) used by `build_call_crash_and_archetypes.py:122` and `build_archetype_weights_quarterly_asof.py:112` | Walk-forward vertices are named on the PCA of the static full-sample fit | Structural |
| M4 | MED | `gold/call_beta/build_call_beta_panel.py:91-98,154-156` | The empirical-Bayes prior for per-call grounding is fitted on all calls 2005-2026. It feeds `substance`, `hist_substance` and `surprise_substance` in every call regression | Structural; small, bounded effect |
| M5 | MED | `analytics/crash_archetypes/call_crash_regressions.py:84-86` | `w_call` = percentile rank within the call's calendar year, so later calls of the same year enter the rank | Structural |
| M6 | MED | `gold/call_beta/build_call_crash_risk_panel.py:51-53,130-131`; `build_ncskew_63.py:74-77`; `build_call_beta_panel.py:228-235` | Post-call windows contain the next earnings call. Consecutive calls have overlapping outcomes, and the next call's pre-window overlaps them | The next call falls inside [+21,+126) for 94.9% of calls, and inside ≤88 calendar days for 20.4% (relevant to the 63-day windows) |
| M7 | MED | `gold/call_beta/build_call_car_panel.py:56-76` (SUE) | Not a leak but a broken construct: `diff(4)` runs on a series with Q4 missing, and SUE is one quarter stale at the call | Only 0.5% of `diff(4)` pairs are 4 calendar quarters apart (66% are 5 apart, 31% are 6). The source 10-Q is filed a median 91 days before the call |
| M8 | MED | `analytics/crash_archetypes/call_archetype_full_battery.py:215-217` | The "posture switchers" robustness sample is selected on the full-sample number of distinct labels, which uses future status | Structural |
| M9 | MED | `analytics/appendix/report_crosscheck_stats.py:214, 253` | "Talk vs walk" and "does talking about AI predict…" correlations have the same timing problem as H4 | Same as H4 |
| M10 | MED | `analytics/washing/build_strategy_economic_profiles.py:54-55` | Static archetype × outcomes that include `next_revenue_yoy` (Panel B gammas) | Structural |
| M11 | MED | `analytics/posture/oos_validation.py:86-89` | The "out-of-sample" test uses features whose prior and rank already include the test years | Structural |
| L* | LOW | see section 4 | FE and standardization on the estimation sample, full-sample ERP, return60 mislabeled as a forward target, `pit.asof_join` dtype bug, and others | — |

Calendar-year-only merges (item 1 of the brief) also appear in `build_roic_wacc.py:119`, `build_market_factors.py:182` and `consolidate/firm_year/*`. Those join two quantities from the same 10-K (same filing), so they are PIT-safe: no `(ticker, year)` duplicates exist in `financial_ratios_raw` or `market_raw`. The problem arises only where text aggregated over a calendar year meets outcomes anchored to the 10-K (H4).

---

## 1. HIGH

### H1. Crash-risk and archetype battery use the same-year expanding archetype

**Consumers**
- `scripts/analytics/crash_archetypes/call_crash_regressions.py:78-82`: `panel["year"] = fecha.dt.year`, then `merge(exp, on=["ticker","year"])`.
- `scripts/analytics/crash_archetypes/call_archetype_full_battery.py:102-106` (fundamentals battery) and `:197-200` (sector-exclusion and switcher robustness).
- These feed `data/results/crash_archetypes/call_crash_archetype_regressions.csv`, `call_archetype_full_battery_{targets,robustness,bh}.csv`.

**Misalignment:** label built from frames with `filing_date.year <= year(call)` vs call date.

**Additional leaks inside the builder** `scripts/gold/crash_archetypes/build_call_crash_and_archetypes.py`:
1. `:63, :104`: the training set for cutoff Y includes every frame filed through 31 Dec Y. That is up to 12 months after a Jan–Nov call in year Y.
2. `:86, :109`: `document_panel` totals (`fecha.year <= Y`) include earnings calls later in year Y, some of them the firm's next calls.
3. `:83-84, :117-120`: the cross-section that is actually scored and labelled is `posture.parquet` (firm-year, full calendar year). Its `disclosure_intensity` is a percentile rank pooled over 2021-2026 and its prior is pooled (M2). This scale also differs from the one the model was trained on (a within-cutoff rank at line 112).
4. `:83, :118`: "No AI" comes from `posture_archetype_static` (`n_frames >= 3` in year Y, full calendar year).
5. `:122`: vertices are named with `name_vertices`, which rests on the static global PCA (M3).

**Checks**
- 10,234 of 10,507 calls (97.4%) get a label.
- Look-ahead from the call to 31 Dec: median 194 days, mean 197 days, over 90 days for 77.3% of calls.
- The label differs from the prior-year label (a minimal PIT lag) in 32.4% of 8,691 calls.
- Agreement with the new `posture_archetype_asof` snapshot through `asof_join` is 52.1%. This partly reflects method differences (12-month window, warm start, profile-based naming), so it bounds rather than measures the leak.

**Fix**
- Drop `posture_archetype_expanding` (firm_year) from every predictive path.
- In both analytics scripts, replace the `(ticker, year)` merge with `pit.asof_join(calls, posture_archetype_asof, event_date="fecha")` on the columns `archetype`, `w_voc`, `w_gov`, `w_def`, `n_frames_window`.
- Build the dummies from the snapshot `archetype`. "No AI" means `n_frames_window == 0`, not the static label.
- Keep the `models/posture_archetype_expanding_yearly` bundles only for descriptive composition (`archetype_composition_annual.py`).
- Re-run the Benjamini-Hochberg family. The hard-coded p-values at `call_archetype_full_battery.py:240-245` (0.0460, 0.9390) must come from the re-estimated headline model, not be typed in.

### H2. `call_beta_config_decoupling_asof.py` matches weights on the fiscal label, not the call date

**Where**
- `scripts/analytics/call_beta/call_beta_config_decoupling_asof.py:49-52` builds `call_period` from `call_accession_number` (`TICKER_YYYYQn`, the fiscal quarter discussed).
- `:67-68` runs `merge_asof(left_on="call_period", right_on="cutoff_quarter", backward, allow_exact_matches=False)`.
- `cutoff_quarter` in `build_archetype_weights_quarterly_asof.py:50,76` is the calendar quarter of `filing_date` and includes every frame filed through the end of that quarter.

**Misalignment:** fiscal-label quarter vs calendar filing quarter.
- For firms whose fiscal year ends off-calendar, the label runs ahead of the real date. Example: NVDA FY2024Q1 is discussed in a call held in May 2023, but the label maps to cutoff 2023Q4, which contains frames filed through December 2023.
- For calendar fiscal years, the call happens in the quarter after the labelled one, so the join is merely stale by one quarter.

**Checks**
- 4,076 calls matched a cutoff (the weights table covers only part of the panel).
- For 18.3% of them, the matched cutoff quarter ends on or after the call date: median 109 days of look-ahead, maximum 230.
- Across the whole spine, `fecha − end of labelled quarter` has a 5th percentile of −235 days, and 22.7% of calls fall before the end of their labelled quarter.
- The script's own comment calls `fecha` "buggy". The spine has only 6 duplicated `(ticker, fecha)` rows and the rest of the pipeline (market windows) already trusts `fecha`, so the label is the less reliable key.

**Other issues**
- `:81-82`: missing weights are set to 0, which merges "no history" with "zero posture weight".
- `name_vertices` applies the global PCA (M3).
- Results are only printed; nothing is written to `data/results`. They can still reach thesis prose (`docs/analytics/12_config_decoupling_asof.md`).

**Fix**
- Join on the call date: `pit.asof_join(panel, posture_archetype_asof[["ticker","as_of_date","w_voc","w_gov","w_def","n_frames_window"]], event_date="fecha")`.
- Retire `archetype_weights_expanding` and `predictions/firm_quarter/posture_archetype_expanding` (archive, do not delete).
- If `fecha` is suspect for specific tickers, fix `fecha` in the spine. Never use the fiscal label as a date.

### H3. `next_revenue_yoy_pre` is future revenue growth

**Where**
- `build_firm_financials.py:560-570` defines `next_revenue_yoy` as growth from the disclosed fiscal year to the following one.
- `build_call_fundamentals_panel.py:60-61, 151` attaches it through the backward 10-K snapshot as `next_revenue_yoy_pre`.
- It is used as the "pre" control for the target `next_revenue_yoy_post` in `call_beta_generalized_targets.py:56` (results in `data/results/call_beta/call_beta_generalized_targets.csv`) and in `call_archetype_full_battery.py:65`.

**Misalignment:** the latest 10-K filed before the call describes fiscal year t−1. Its `next_period_end` is the end of fiscal year t, which normally lies after the call. The control therefore measures revenue growth over a period that has not finished at the call date, and it overlaps the outcome. It is also published only with the next 10-K, which comes after the call.

**Check:** 8,967 non-null values; 75.7% cover a fiscal year ending after the call, a median 144 days later.

**Fix**
- Define the pre control as the growth the pre 10-K discloses: `revenue_yoy` = revenue of the disclosed fiscal year / revenue of the prior fiscal year − 1.
- Give it `available_date = filing_date` of that 10-K and attach it with the existing backward join (strict).
- Keep `next_revenue_yoy` only as a target.
- Audit other `next_*` columns of `financial_ratios_raw` that enter a `_pre` or control slot. Currently only revenue is used.

### H4. Firm-year panel: calendar-year text vs outcomes anchored to the 10-K

**Where**
- `disclosure_volume/build_covariates_firm_year.py:45` and `build_spine.py:41`: `year = fecha.year`, summed over every 10-K, 10-Q, DEF 14A and 8-K of that calendar year.
- `consolidate/firm_year/build_targets.py:43-55`: `beta_252d`, `idio_vol_252d` and `vol_pre_60d` are windows before the 10-K filing date. `price_to_sales_t` uses the price just before the filing. `revenue_growth_lead1y` is the fiscal year after the one the year-t 10-K discloses.
- `consolidate/firm_year/build_dataset.py:9-21` claims every covariate precedes its target. That claim is false for these columns.
- `incremental_signal.py:74-76` uses exactly these outcomes (RQ4 headline, `data/results/shock/incremental_signal.json`, `incremental_signal_coefficients.csv`).

**Misalignment**
- Most 10-Ks are filed in February or March, so most of year t's text (10-Qs, proxies, 8-Ks) is published after the outcome's measurement window has closed. For beta, idiosyncratic volatility, vol_pre_60d and P/S, the predictor postdates the outcome.
- For `next_revenue_yoy`, the outcome fiscal year is roughly calendar year t, and the year-t 10-Qs in the text already report its quarterly revenue, plus `revenue_outcome` frames. This is contemporaneous information in the predictor, not a prediction.
- The M2A/M4 activity block (`activities.parquet`, filing year) has the same issue.

**Check:** 72.3% of filing documents in firm-years with a 10-K, and 33.8% of their AI frames, are filed after that year's 10-K.

**Fix, for a predictive reading**
- Build the events as 10-K filings (`available_date = filing_date`), one row per 10-K.
- Take the text predictor from a posture or disclosure-volume snapshot joined as-of: `pit.asof_join(tenk_events, disclosure_snapshots, event_date="filing_date")`. Each snapshot is a trailing 12-month aggregate of documents with `available_date < as_of_date`.
- Outcomes must start at or after `filing_date`: post-filing beta, idiosyncratic volatility and volatility (e.g. [+1, +252] or [+21, +126]), and P/S at a date after `filing_date`.
- For revenue growth, the target is the first fiscal year ending after the snapshot date.
- If the analysis stays contemporaneous (association, not prediction), the thesis text must say so and `build_dataset.py`'s docstring must change. The pre-filing beta and volatility outcomes still postdate the predictors in that case.

### H5. Static full-sample archetype in the incremental-signal M3 block

**Where:** `incremental_signal.py:70, 100-101` merge `posture_archetype_static` on `(ticker, year)`; `:165-167` build the M3 dummies. The archetype ΔR² is reported in `incremental_signal.json`. The static model (`build_strategy_dimensions.py:109-111, 148`) is one AA fit plus a `StandardScaler` over the pooled firm population. The firm-year panel is projected with `posture.parquet`, which has a pooled prior and pooled rank (M2).

**Fix:** use `posture_archetype_asof` snapshots joined as-of on the event date (the 10-K `filing_date` under the H4 redesign), or drop M3 from the predictive table and keep the static archetype for description only.

---

## 2. MEDIUM

### M1. Washing score W: pooled ranks and priors

**Status (2026-09-15): fixed.** `scripts/gold/firm_year/build_washing_score.py` fits the grounding priors, percentile ranks and 5% tails within each calendar year's cross-section, as the firm-quarter score does (priors persisted per year). corr(old pooled W, new W) = 0.885, mean |ΔW| = 0.087; mean W by year −0.141, −0.149, −0.104, +0.023, +0.063, +0.093 → 0.000 in every year.

**Where**
- `gold/washing/washing_score.py:140`: grounding priors are fitted on every firm-year.
- `:195`: missing `grounding_index` is filled with the pooled median.
- `:210-214`: `pctrank` of disclosure and substance over the pooled 2021-2026 sample, plus pooled tail quantiles.
- The same logic is duplicated in `consolidate/predictions/washing_score.py:80, 90-95`.

**Consumers:** `incremental_signal.py:141` (M4, `washing_z`); `sec_comment_letter_cases.py:102-104` (descriptive matching, LOW); `channel_gap_did.py` and the shock scripts (out of scope).

**Misalignment:** each firm-year's W depends on the distribution in later years. The percentile rank is globally monotone, so within one year the ordering matches a within-year rank; only the spacing changes. The sector×year fixed effects absorb the level drift. The priors and median fill also leak, mildly.

**Check:** corr(pooled W, within-year W) = 0.886. Mean pooled W by year: −0.141, −0.149, −0.104, +0.023, +0.063, +0.093 (2021→2026). Mean absolute difference 0.087.

**Fix:** as a predictor, compute W on snapshots. At each `as_of_date`, rank `frames_per_1k` and `substance` over the trailing 12-month firm cross-section (documents with `available_date < as_of_date`), with priors fitted on the same history (as `build_posture_asof.shrink_params` already does). Join by `asof_join`. The pooled W can stay as the descriptive construct, but it keeps one name and one formula, so the predictive variant must be a snapshot of the same W, not a new construct.

### M2. `covariates/firm_year/posture.parquet`: pooled prior and rank

**Status (2026-09-15): fixed.** `scripts/gold/firm_year/build_posture.py` fits the shrinkage prior and the `disclosure_intensity` rank within each calendar year (mean intensity 0.505 … 0.501 by year; rate correlations with the pooled version 0.96-0.998, intensity 0.917). The static archetype projection (`covariates/firm_year/posture_archetype_static`) keeps its own pooled construction of the eight dimensions inside the builder and is unchanged; the firm-grain static fit does not read this covariate.

**Where:** `build_strategy_dimensions.py:163-164`.
**Consumers:** H1 builder (cross-section), `posture_archetype_static` (firm_year), `oos_validation.py`.
**Check:** mean `disclosure_intensity` by year is 0.361, 0.355, 0.365, 0.452, 0.519, 0.642.
**Fix:** the predictive paths should use `posture_archetype_asof` (which already fits priors and the intensity scale on training rows `< as_of`). Keep `posture.parquet` descriptive only and document that.

### M3. Vertex naming on the static global PCA

**Where:** `posture_features.py:226-240` loads `models/posture_archetype_static/model.pkl` and `predictions/firm/posture_archetype_static.parquet`; `:243-266` names each vertex by correlation with those PCA scores.
**Used by:** `build_call_crash_and_archetypes.py:122` and `build_archetype_weights_quarterly_asof.py:112`.
**Misalignment:** which walk-forward vertex is called "Governance-Led Disclosers" (the `dum_gov` headline) is decided with loadings estimated on the full sample.
**Fix:** already solved in `build_posture_asof.py:92-98`, which names vertices by their own standardized profile. Retire the other two builders from predictive use; keep `global_posture_pca` for descriptive figures only.

### M4. Per-call grounding prior fitted on all calls

**Where:** `build_call_beta_panel.py:91-98, 154-156`. `_shrink` fits the Beta prior on every call from 2005 to 2026, and the docstring acknowledges that values move when 2026 is added. It feeds `substance`, `hist_substance` and `surprise_substance`, which are core regressors in `call_beta_regressions.py`, `call_car_regressions.py`, `call_beta_robustness.py`, the crash scripts, generalized targets and config-decoupling.
**Severity:** MEDIUM. A shared prior shifts calls with few activities toward a global mean; the leak is small but real.
**Fix:** an expanding prior. Fit `(alpha, beta)` on calls with `fecha < as_of_date` at each quarterly snapshot, and apply the snapshot prior to calls in `[as_of, next as_of)`. `hist_*` and `surprise_*` then keep their `expanding().shift(1)` construction on the PIT `substance`.

### M5. `w_call` ranked within the calendar year

**Where:** `call_crash_regressions.py:84-86` (`ncskew_w` and `duvol_w` rows of the results CSV).
**Fix:** rank against the trailing cross-section of calls with `fecha < call date`. Either compute it from quarterly snapshot distributions (as-of join of the snapshot's empirical CDF) or use an expanding rank like the expanding z-score in `config_decoupling_asof.py:94-102`.

### M6. Post-call windows overlap the next call

**Where**
- `build_call_crash_risk_panel.py`: post window [+21, +126) and pre window [−252, −21).
- `build_ncskew_63.py:74-77`: pre [−63, 0), post [0, +63).
- `build_call_beta_panel.py:219-235`: `beta_post_63/126/252` = [0, +h), pre [−252, 0).
- `call_beta_robustness.py:354-357`: post windows [+21, h).

**Misalignment**
- None of these windows mixes past and future within a row: pre windows end before `idx`, and `price_pre` and `return60` stop at `idx−1`.
- However, the outcome for call q contains call q+1's earnings announcement (and for 252-day windows, several later calls), plus the market reaction to later disclosures.
- Call q+1's pre-window control overlaps call q's outcome. Outcomes of consecutive calls overlap by about 40–60%.
- Windows that start at day 0 include the announcement-day reaction when a call happens before the market opens.

**Check:** median gap between calls is 91 days. The next call falls inside [+21, +126) for 94.9% of calls, and within 88 calendar days (about 63 trading days) for 20.4%.

**Fix**
- End post windows at the trading day before the next call. Add `next_call_date` per ticker (the spine already has it, via `shift(−1)`), or truncate at `min(idx + h, idx_next_call − 1)` with a minimum-observations rule.
- Report overlap-robust inference: firm clustering is already there; add Driscoll-Kraay or date clustering.
- For the 63-day windows, start at +2 to exclude the announcement reaction.

### M7. SUE is stale and its lag is wrong (construct bug)

**Status (2026-09-15): skipped.** SUE and CAR are no longer used in the thesis. A rebuild keyed on the fiscal quarter (seasonal difference over the s.d. of the 8 prior quarters, Q4 = annual − Q1..Q3, filings before the call) exists as an uncommitted change to `scripts/gold/call/build_market_financials.py`; with it SUE coverage is 4,690 → 5,699 calls.

**Where:** `build_call_car_panel.py:66-76`. The eps panel is inner-joined to `silver.filing_manifest_10q`, which drops Q4 values sourced from 10-Ks (7,836 of 10,947 kept; Q4 = 512). `:56-63` then runs `diff(4)` over filing order.
**Check:** only 0.5% of `diff(4)` pairs are 4 calendar quarters apart (66.4% are 5, 30.8% are 6). The merge is backward and strict on the 10-Q `filing_date`, so SUE reflects the previous quarter; the median source 10-Q is filed 91 days before the call. There is no look-ahead, but the "surprise" is neither year-over-year nor current.
**Severity:** MEDIUM; SUE is used only with `--include-sue` (`call_beta_sue_regressions.csv`).
**Fix:** key eps by `(ticker, fiscal period_end)`; compute year-over-year change against the same fiscal quarter one year earlier (by `period_end` minus about 365 days); set `available_date` to the filing date of the document that first reports the quarter (10-Q or 10-K); prior-only standard deviation as now; backward as-of join on `fecha`. If an earnings press release (8-K item 2.02) date is available, use it as `available_date` to get the current quarter.

### M8. "Posture switchers" selected on the full sample

**Status (2026-09-15): fixed.** `call_archetype_full_battery.py` selects calls at which the firm has already shown two different dominant archetypes (argmax of the as-of `posture_ttm` weights) at that or an earlier call, plus its later calls. 5,046 calls / 274 firms → 2,591 / 246; Defensive weight NCSKEW +0.118 (p 0.077) → +0.259 (p 0.049), DUVOL +0.173 (p 0.013) → +0.339 (p 0.008); Governance-Led and Vocal weights null in both.

**Where:** `call_archetype_full_battery.py:215-217`: `nunique(arch_exp) > 1` over the whole panel. Firms are selected on labels observed after each call.
**Fix:** define switching as-of, e.g. calls where the snapshot archetype differs from the snapshot four quarters earlier (both joined as-of). Alternatively, report the full-sample definition as descriptive only.

### M9. Appendix cross-check correlations

**Status (2026-09-15): labeled.** `data/results/appendix/crosscheck_stats.json` carries a `timing` note (descriptive, contemporaneous, not point in time); no redesign.

**Where:** `report_crosscheck_stats.py:214` (`revenue_outcome_per_1k` vs `next_revenue_yoy`, "talk vs walk") and `:253` ("does talking about AI predict …" `next_*_yoy`, `ret_m1_p5`, `car_m1_p5`). Same firm-year timing as H4. `ret_m1_p5` and `car_m1_p5` are windows around the 10-K filing date and precede most of year t's text.
**Output:** `data/results/appendix/crosscheck_stats.json`.
**Fix:** as in H4; alternatively, rename the section as contemporaneous association.

### M10. Economic profiles by static archetype

**Status (2026-09-15): labeled.** `data/results/washing/strategy_economic_profiles.json` carries a `timing` note (descriptive, contemporaneous static archetype vs same-year outcomes); no redesign.

**Where:** `build_strategy_economic_profiles.py:54-55` (static archetype) regressed on `next_revenue_yoy`, beta and vol_pre_60d, among others (Panel B, `data/results/washing/strategy_economic_profiles.json`).
**Severity:** MEDIUM if Panel B is read as "archetype predicts"; LOW if it is strictly a profile. The same H4 timing mismatch applies.
**Fix:** state it as descriptive, or rebuild on as-of snapshots with post-event outcomes.

### M11. Temporal out-of-sample validation is not out of sample

**Status (2026-09-15): resolved by M2.** `oos_validation.py` reads `covariates/firm_year/posture`, now fitted within each year, so training-year features carry no test-year information; standardization is on training rows. Cluster names still come from the static archetype label (naming only). ARI 2021-24→2025 0.480 → 0.321; 2021-23→2025 0.328 → 0.026; 2021-25→2026 0.705 → 0.552.

**Where:** `oos_validation.py:86-89`. Features come from `posture.parquet` (M2: prior and rank pooled over every year, including 2025-2026 test years). Cluster names come from the static archetype. The per-window standardization at `:61-64` is correct.
**Fix:** recompute features with priors and ranks fitted on training years only (or from `posture_archetype_asof` snapshots dated ≤ 2025-01-01 for training).

---

## 3. Checked and found PIT-safe

- **`hist_*` / `surprise_*`** (`build_call_beta_panel.py:170-175`): `expanding().mean().shift(1)` per ticker, sorted by `(ticker, fecha, call_accession_number)`, so the history excludes the current call. There are 6 rows with duplicated `(ticker, fecha)`. The tie is broken by the accession label, so for those pairs one sibling's same-day value enters the other's history. Negligible (LOW). A stricter rule would group by `(ticker, fecha)` before shifting.
- **Accounting controls** (`build_call_beta_panel.py:250-251`, `build_call_fundamentals_panel.py:75-76`, `build_call_car_panel.py:86-87`): backward `merge_asof` on the 10-K/10-Q `filing_date`, `allow_exact_matches=False`, `by="ticker"`, no tolerance. Median staleness is 173 days (p95 364); consider a 400-day tolerance so a firm that stops filing does not carry year-old fundamentals indefinitely (LOW).
- **Leverage** (`build_call_car_panel.py:155-168`): taken from the as-of 10-K's own accession ✓.
- **XBRL value selection** (`build_firm_financials.py:299-308`, `pivot_metrics:452`; `build_us_10q_financials_panel.py:141-164`): the first-reported value wins, with date first and priority second; the frames fallback is restricted to a 0–120-day filing lag ✓. Residual risk (LOW): if a 10-K did not tag an annual fact, the earliest report is the next year's comparative, filed after that 10-K. Suggested check: for each `financial_ratios_raw` row, confirm the chosen fact's `filing_date` ≤ the row's `filing_date`. This needs keeping `filing_date` through the pivot.
- **Pre-call market windows:** `beta_pre` [−252, 0), `beta_pre_car` / `idio_vol_pre` [−252, −21), `return60` [−60, −1), `price_pre` = close at −1; `ncskew_pre` / `duvol_pre` [−252, −21) ✓. `idx` = first trading day ≥ `fecha`.
- **CAR** (`build_call_car_panel.py:115-119`): `abnormal.iloc[1:]` actually sums [0, +5], not [−1, +5]. This is a label issue with no leak (LOW). Beta comes from [−252, −21) ✓.
- **`build_posture_asof.py`** (new): frames `available_date < as_of` within a 12-month window (`:82`); documents `fecha < as_of` (`:85`); priors, standardization and intensity scale fitted on training snapshots ≤ q (`:118-128`); warm start from the previous snapshot; profile-based naming ✓. The existing output has 0% of rows with `as_of_date > fecha` after the join. Notes in section 4 (L8–L10).
- **Universe:** `silver.filing_manifest` is filtered to the S&P 500 as of 2021-01-01 (ex-ante ✓). No script filters on future index status. Full-sample conditions that do appear: `call_beta_robustness.py:309-317` (top decile of total call count, LOW) and the balanced-panel requirements in the DiD scripts (out of scope).

---

## 4. LOW

- **L1.** In-sample standardization of y and X (`fit` in every call regression) and sector×year fixed effects (`fe = sic2_callyear`, `build_call_beta_panel.py:261`; `sector_year` in `incremental_signal.py:121`). These are standard in-sample panel inference: scaling does not change t-statistics, and the fixed effects demean with same-year peers. For a genuine out-of-sample forecast, replace them with snapshot-based scaling.
- **L2.** Pooled winsorization: `incremental_signal.py:106-108`, `call_beta_robustness.py:247-265`, `build_strategy_economic_profiles.py:58-59`. Pooled shrinkage priors on semantic shares: `incremental_signal.py:110-120`.
- **L3.** The ERP is a geometric mean over 2000-2026 (`build_roic_wacc.py:78-88`, `build_call_fundamentals_panel.py:162`) and enters `roic_minus_wacc_pre`. It is a single scalar that rescales `beta_pre`'s contribution. Fix: use the ERP as of `filing_date` (expanding mean of `mktrf` through `available_date`).
- **L4.** `consolidate/call/build_targets.py:31` renames `return60` (the pre-call return over [−60, −1)) to `return_post_60d` and calls it a forward target; the docstring of `consolidate/call/build_dataset.py` repeats this. Check: the two columns are identical. No script reads `return_post_60d` today, but anyone using it as an outcome would regress a pre-call return. Fix: compute a true post return [0, +60) with `available_date = idx + 60`, or drop the column.
- **L5.** `build_firm_financials.py:667-670`: `shares_out` is attached by `direction="nearest"` within ±75 days of `filing_date`, so a cover-page count dated slightly after the filing can be chosen. Fix: backward only, ≤ `filing_date + 5 days`.
- **L6.** `build_call_fundamentals_panel.py:152, 164-165`: the "post" snapshot is the first 10-K after the call, which can be 1–12 months later, so outcome horizons differ across calls. `rf_post` is taken at `fecha + 176` calendar days. It is an outcome, so this is not a leak; make the horizon explicit (e.g. the first 10-K filed ≥ 126 trading days after the call).
- **L7.** Duplicates: `fundamentals_pre` / `fundamentals_post` / `car_pre` each have 6 duplicated `call_accession_number` rows, handled downstream with `drop_duplicates`. Fix at the builder: the `(ticker, fecha)` merge at `build_call_fundamentals_panel.py:123` and `build_call_car_panel.py:135` fans out the 6 same-date pairs.
- **L8.** `scripts/common/pit.py:39-44`: `asof_join` fails with `MergeError: incompatible merge keys dtype('<M8[ns]') and dtype('<M8[us]')` when the spine (ns) meets the gold parquet (us). Reproduced with `call.parquet` × `posture_archetype_asof.parquet`. Fix: cast both keys with `.astype("datetime64[ns]")` inside `asof_join`.
- **L9.** `build_posture_asof.py`:
  - Scoring requires `n_frames_window >= 1` (`:138`) while training requires ≥ 5. Firms with 1–4 frames get noisy weights; not a leak.
  - A fixed 12-month window can drop a 10-K when filing dates shift across the Apr 1 or Jan 1 boundary (a 10-K filed 30 Mar one year and 2 Apr the next leaves the Apr 1 snapshot with neither). Consider 15 months, or "last 10-K plus trailing 12 months".
  - The universe comes from `silver.firm_universe` (`:105`). Confirm it is the 502-ticker 2021 panel and not the 543 tickers with filings; the memory notes the restriction is enforced at the DuckDB view layer.
- **L10.** `posture_features.load_frames` (working-tree change):
  - The join to `documents` dropped `country_code`, which risks accession collisions once Chile and Italy data arrive.
  - `unique(subset=["accession_number","ticker"])` keeps a filing under both tickers of an aliased dual-class issuer, so its frames count twice. Posture is a rate, so the effect is minor.
- **L11.** Descriptive uses of the static archetype (`activity_profiles`, `archetype_*`, `representative_firms`, `geo_provenance`, `firm_comparisons`, `plot_archetypal_simplex`, `check_archetype_document_channels`, `bootstrap_archetype_stability`, `archetype_10k_refit`, `strategy_dimensions_diagnostics`) are fine as descriptions. `patents_controls.py:94` uses the static table only to list the universe.
- **L12.** `validate_washing_score.py`: persistence and coherence correlations on pooled W are descriptive. The persistence estimate is slightly inflated by the shared pooled ranks.
- **L13.** `call_beta_robustness.py:309-317`: call-frequency exclusion based on full-sample counts. It also reads `data/raw` prices (a layer issue, not PIT).
- **L14.** `silver.ai_frames` and the prefilter/judge labels come from classifiers developed on data from every year. This measurement-model dependence is common to every analysis and cannot be removed without re-labelling. Mention it as a limitation.

---

## 5. Out of scope (DiD no longer in the thesis), LOW

- `channel_gap/build_channel_gap_cells.py` and `channel_gap_did.py`:
  - `post = fy >= 2024` against a 2023-12-05 / 2024-03-18 event.
  - The call fiscal year comes from `document_id` labels that are known to disagree across sources for off-calendar retailers.
  - `fye_month` is the full-sample mode.
  - The `firm × W` cross at `channel_gap_did.py:211` merges on ticker only, fanning out firm-years.
- `shock/shock_analysis.py:104-111` and `shock_did_simple.py:79-84`: treatment and group are pre-2024 means of pooled W (M1). Balanced-panel filters require ≥ 2 post-event quarters. `shock_did_simple.py` says "Post = 2024Q2" in its docstring but uses `EVENT = 2024Q1` in code.
- `evolution_figures.py`: descriptive.

---

## 6. Suggested migration order

1. Fix `pit.asof_join` dtype casting (L8), then finish `build_posture_asof`.
2. H1 and H2: switch both crash scripts and `config_decoupling_asof` to `asof_join` on `fecha`. Re-run the crash CSVs and the Benjamini-Hochberg family with no hard-coded p-values. Archive the yearly and quarterly expanding outputs.
3. H3: replace `next_revenue_yoy_pre` with the disclosed `revenue_yoy`. Re-run generalized targets and the battery.
4. M4 and M5: expanding grounding prior and trailing rank for `w_call`. M6: truncate post windows at the next call.
5. H4, H5 and M1: redesign the incremental signal around 10-K events with snapshot predictors (disclosure volume, semantic block, activities, snapshot W, snapshot archetype) and post-filing outcomes, or explicitly reframe it as a contemporaneous association. Update `consolidate/firm_year/build_dataset.py`'s timing claims.
6. M7: rebuild SUE on `period_end`. L4: fix or drop `return_post_60d`.
7. Following CLAUDE.md rule 9, search the thesis for "expanding archetype", "posture_archetype_expanding", "cutoff year", "next_revenue_yoy_pre" and the pooled-W wording, and update every reference.
