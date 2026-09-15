# Stage-rule violation audit — task list

Read-only audit of `scripts/` (excluding `scripts/deprecated/`, `tests/`), `Makefile`, and
`thesis_document/thesis.qmd`. Line numbers are from the working tree at audit time. About 31
files were being edited by other agents at that moment (mostly migrating DuckDB reads to
`layers.scan`), so re-check line numbers before editing. None of those edits change a
file's stage placement or output paths.

Severity: **HIGH**: data flows the wrong way, or a downstream consumer depends on the
misplaced/stale artifact. **MEDIUM**: a report or statistic is produced in the wrong stage, or
lives in the wrong place. **LOW**: cosmetic or organizational; print-only logs are acceptable.

---

## 0. Prerequisites (infra, not violations; do first)

- **P1** Add to `scripts/common/layers.py`: `GOLD = DATA/"gold"`, `GOLD_INTERMEDIATE = GOLD/"intermediate"`
  (topic-level panels that today live in `data/processed/clusters/`), `RESULTS = DATA/"results"`, plus a
  `results_dir(topic) -> Path` helper. All tasks below that say "write to data/gold/intermediate/<topic>/" or
  "data/results/<topic>/" use these constants.
- **P2** (last) Once no script reads `data/processed/clusters/*`, move it to `data/deprecated/processed/clusters/`
  locally and in B2 (memory rule: archive before restructuring). thesis.qmd is the only reader allowed to keep
  pointing there, and only temporarily.

### Producer → consumer map for `data/processed/clusters` (drives tasks G-H6/G-H8/A-M6)

| file | producer (stage) | script consumers | thesis.qmd |
|---|---|---|---|
| firm_year_financials(_ratios).parquet | gold/financials/build_firm_financials.py:84,685 | gold build_market_factors:155, build_roic_wacc:105, build_call_beta_panel:86, build_call_fundamentals_panel:54; analytics call_beta_regressions:95, call_car_regressions:149 | — |
| firm_year_market_factors / firm_year_filing_returns | gold/financials/build_market_factors.py:201 | gold build_roic_wacc:106, build_firm_panels:98-99 | `_p("firm_year_market_factors.parquet")` |
| firm_year_roic_wacc.parquet | gold/financials/build_roic_wacc.py:153 | analytics report_crosscheck_stats:190, build_strategy_economic_profiles:45 | — |
| us_10q_financials_panel.parquet (data/processed/) | gold/financials/build_us_10q_financials_panel.py:357 | analytics call_car_regressions:35 | — |
| firm_strategy_dimensions / firm_year_strategy_dimensions | gold/posture/build_strategy_dimensions.py:476-478 | gold activity_profiles:332, build_geo_provenance:52, build_firm_panels:93,103; analytics check_archetype_document_channels:35, plot_archetypal_simplex:47, build_strategy_economic_profiles:50 | 12× `_p("firm_strategy_dimensions")`, 1× year |
| strategy_dimensions_manifest.json | build_strategy_dimensions.py:492 | — | 211-216, 2182, 4342 |
| firm_activities / firm_activity_profiles / firm_year_activities / channel_activity_cells | gold/posture/activity_profiles.py:334-338 | gold build_strategy_dimensions:378, build_call_beta_panel:149, washing_score:135,154, consolidate/activities:20, consolidate/predictions/washing_score:39; analytics incremental_signal:111, activity_grounding:47 | 11× `_p("firm_activities")`, 700, 2373 |
| activity_profiles.json | activity_profiles.py:466 | — | 222-226, 4367, 4416 |
| document_panel.parquet | gold/posture/build_document_panel.py:31 | gold crash:69, archetype_weights_quarterly:66; analytics sec_comment_letter_cases:88 | 170, 192, 3973 |
| panel_expanding_archetype_weights_quarterly | gold/posture/build_archetype_weights_quarterly_asof.py:139 | gold consolidate/predictions/posture_archetype_expanding_quarterly:28; analytics call_beta_config_decoupling_asof:48 | — |
| firm_year_master_v2 / firm_year_full_crosscheck / cohort_2021_crosscheck / segment_financials | gold/washing/build_firm_panels.py:165-173 | gold activity_profiles:290, washing_score:165,203,251, crash:68, consolidate {disclosure_volume,financial_ratios,market,posture}/build_covariates_yearly, firm_year/build_targets; analytics incremental_signal:90, evolution_figures:31, build_strategy_economic_profiles:43, firm_year_aggregation_robustness, sec_comment_letter_cases:102, report_crosscheck_stats:188-189 | 160, 345, 796 + 4× `_p` |
| firm_year_washing_score / firm_washing_score (+manifest) | gold/washing/washing_score.py:298,313,316 | gold channel_gap_analysis:424; analytics incremental_signal:132, shock_analysis:106, shock_did_simple:75, sec_comment_letter_cases:100 | 5× `_p` |
| call_beta_main_panel_10k10q_asof.parquet | gold/call_beta/build_call_beta_panel.py:304 | gold build_ncskew_63:35, build_call_fundamentals_panel:53, crash:25, consolidate call/build_targets:24, {disclosure_volume,financial_ratios,market}/build_covariates_quarterly; analytics call_beta_{regressions,robustness,generalized_targets,config_decoupling_asof}, call_car_regressions, call_archetype_full_battery | `_p(...)` |
| ncskew_63.parquet | gold/call_beta/build_ncskew_63.py:79 | analytics call_beta_config_decoupling_asof:130 | — |
| call_fundamentals_panel.parquet | gold/crash_archetypes/build_call_fundamentals_panel.py:180 | gold crash:26; analytics call_beta_generalized_targets:39, call_archetype_full_battery | — |
| panel_expanding_archetypes.parquet | gold/crash_archetypes/build_call_crash_and_archetypes.py:130 | gold consolidate/predictions/posture_archetype_expanding_yearly:26; analytics call_archetype_full_battery | 3783 |
| channel_gap_cells / channel_gap_firm | gold/channel_gap/channel_gap_analysis.py:439-440 | gold consolidate/channel_gap/build_dataset:20 | 3758 |
| **call_crash_risk_panel.parquet** | **no producer** (not in scripts/, scripts/deprecated/ or git history) | gold crash:27,135; analytics call_beta_config_decoupling_asof:122, call_archetype_full_battery:35 | — |
| **firm_segments.parquet** | **scripts/deprecated/build_segments.py** (stale) | gold channel_gap_analysis:430; analytics shock_analysis:120 | — |
| gbdt_prefilter_importance.json | **no producer** | — | — |

---

## 1. verif

### V-H1 HIGH: verif builds training labels that enrichment consumes
- **Where:** `scripts/verif/prefilter_form_validation.py:56-62, 105-155 (cmd_sample), 158-~200 (cmd_label via golden_set.label_rows)`,
  writing `data/interim/golden_set_forms/{form_validation_sample,form_train_sample,calls_validation_sample}.parquet` and
  LLM labels under `golden_set_forms/`, `train/`, `calls/`.
- **Consumer:** `scripts/enrichment/ai_prefilter_deploy.py:67, 109-123, 207-209` (training labels) and `:235` (holdout).
  `common/layers.py:71-72` lists `golden_set.py` as the producer, which is wrong.
- **Rule broken:** verif must not produce data that another stage consumes. Sampling and LLM labeling are enrichment work.
- **Task:** Move the `sample` and `label` subcommands into a new `scripts/enrichment/golden_set_forms.py`. Keep the same
  output paths, and do not move or delete any existing labels. Update `ADDITIVE_SOURCES["golden_set_forms"]["producer"]`.
  Leave only `evaluate` in verif for now (V-M2 moves it).
- **Deps:** do before E-M3.

### V-M2 MEDIUM: evaluation metrics written into an append-only LLM directory
- **Where:** `prefilter_form_validation.py:203-265 (cmd_evaluate)`, which writes `form_validation_metrics.json` into
  `data/interim/golden_set_forms/`. It reads the predictions glob in interim at `:220`.
- **Task:** Create `scripts/analytics/prefilter/form_validation_metrics.py`, reading `bronze.prefilter_predictions` and the
  golden_set_forms labels. Output: `data/results/prefilter/form_validation_metrics.json`. Then delete `cmd_evaluate`.
  Deps: V-H1.

### V-M3 MEDIUM: report written into the prefilter predictions directory
- **Where:** `scripts/verif/prefilter_judge_comparison.py:67-70, 128` writes `judge_comparison.json` into
  `data/interim/prefilter_predictions_unique/`.
- **Task:** Change the default `--output` to `data/interim/audits/prefilter/judge_comparison.json`. If the thesis cites it,
  move it to `scripts/analytics/prefilter/judge_comparison.py` with output under `data/results/prefilter/`.

### V-M4 MEDIUM: report written into the golden_set LLM directory
- **Where:** `scripts/verif/judge_agreement.py:117-118` writes `judge_agreement.json` into `data/interim/golden_set/`.
- **Task:** Write to `data/interim/audits/golden_set/judge_agreement.json` instead.

### V-M5 MEDIUM: report written into golden_set_forms
- **Where:** `scripts/verif/prefilter_feature_eval.py:225, 290` writes `feature_eval.json` into `data/interim/golden_set_forms/`.
- **Task:** Write to `data/interim/audits/prefilter/feature_eval.json` instead.

### V-L6 LOW: print-only evaluations read the legacy DB and a hardcoded interim run
- **Where:** `prefilter_logit_refit.py:22,38-40`, `prefilter_model_variants_eval.py:40,65-67`, `prefilter_rescue_eval.py:33,54-56`.
  They read `duckdb/thesis.duckdb` and `prefilter_scores__run={LATEST_PREFILTER_RUN}`.
- **Task:** Switch reads to `L.scan("bronze.prefilter_scores")`. No output change. Bundle with C-M1.

---

## 2. enrichment

### E-M1a MEDIUM: funnel counts computed inside enrichment (new analytics script)
- **Where:** `scripts/enrichment/ai_prefilter_classify.py:563-600 (funnel_counts)`, `:695-708` (apply_only funnel and
  per-form `by_form` group_by), `:870-875`; manifests `:736-738, :903` (`funnel`, `funnel_by_form`).
  Also `ai_prefilter_deploy.py:279, 296-300, 312, 316-317` and `ai_prefilter_apply_frozen.py:86, 104-108, 115`.
- **Rule broken:** enrichment must do zero aggregation and must not produce funnels.
- **Task:** Create `scripts/analytics/prefilter/funnel.py`, reading `bronze.paragraphs` (raw and scorable),
  `bronze.unique_paragraphs`, `bronze.prefilter_scores` (lexical gate), `bronze.prefilter_predictions` (model-only,
  named-entity rescue, positives, instances), `silver.ai_frames` and `silver.ai_activities`. Outputs:
  `data/results/prefilter/funnel.json` and `funnel_by_form.csv`.
- **thesis.qmd:** `tbl-funnel-summary` (569-614) and `tbl-a1-funnel-full` currently compute the funnel inline from
  `duckdb/thesis.duckdb` (85, 525-640). Point them at `data/results/prefilter/funnel*.{json,csv}`.

### E-M1b MEDIUM: strip the funnel from the three prefilter scripts
- **Task:** Delete `funnel_counts()` and every `funnel*` key or print in classify, deploy and apply_frozen. The manifest
  keeps only: run_id, anchors_run, model path or coefficients, threshold, features, label counts, output, created_at.
  Keep `coefficients`/`intercept`/`threshold` in the classify manifest, because `load_deployed_model` (`:610-658`) reads them.
- **Deps:** E-M1a first, so the thesis never loses the numbers.

### E-M2 MEDIUM: classify persists cross-validation metrics as a report
- **Where:** `ai_prefilter_classify.py:488-560` returns `oof_metrics`; `:772-809` prints and computes `combined_metrics`;
  manifest `:731-732, :898` (`cv_metrics`, `cv_metrics_with_named_entity`), echoed by `apply_only` at `:653-654`.
- **Task:** Keep threshold/C selection as fitting. Persist the decisions (`deploy_c`, `threshold`,
  `named_entity_used_in_deployment`), not the metric report. Move OOF metric reporting to
  `scripts/analytics/prefilter/logit_cv_metrics.py`, output `data/results/prefilter/logit_cv_metrics.json`.
  The docstring says `ai_prefilter_deploy.py` supersedes this training path, so consider deprecating the training path
  entirely and keeping only `--apply-only`.

### E-M3 MEDIUM: deploy runs a holdout evaluation and a decision curve, and copies the report into models/
- **Where:** `ai_prefilter_deploy.py:137-155` (weighted_f1), `:223-225` (CV metric prints), `:235-261` (holdout per-form
  metrics and decision curve), manifest `:302-315` (`cv_metrics`, `cv_folds`, `holdout_form_metrics`). `:315` copies it to
  `models/ai_classification/run=<id>/manifest.json`.
- **Task:** Keep `nested_cv` for picking depth and threshold, and persist `depth`, `threshold`, `threshold_source`,
  `features` and label counts. Move the holdout evaluation and decision curve to
  `scripts/analytics/prefilter/holdout_form_metrics.py`. It loads `models/ai_classification/run=<id>/model.joblib`, reads the
  golden_set_forms validation labels plus bronze features, and writes `data/results/prefilter/holdout_form_metrics.json`
  and `decision_curve.csv`. Remove the `cv_metrics` and `holdout_form_metrics` keys from both manifests.
- **Deps:** V-H1, E-M1b.

### E-M4 MEDIUM: apply_frozen computes a run-over-run diff metric
- **Where:** `ai_prefilter_apply_frozen.py:73-84` reads the latest prior predictions parquet and counts
  `new_positive_since_prior_run`; persisted at `:108, :115`.
- **Task:** Remove it. If needed, create `scripts/analytics/prefilter/run_diff.py`, comparing two
  `data/interim/prefilter_predictions_unique/prefilter_predictions__run=*.parquet` files and writing
  `data/results/prefilter/run_diff.json`.

### E-M5 MEDIUM: entity-mention manifest carries summary statistics
- **Where:** `scripts/enrichment/ai_entity_mentions.py:98-106` (`by_geo`/`by_modality` group_by in the manifest) plus prints.
- **Task:** The manifest keeps run_id, term count, output and row count. Move the distributions to
  `scripts/analytics/posture/entity_mentions_summary.py`, reading `silver.ai_entity_mentions` and writing
  `data/results/posture/entity_mentions_by_geo.json`.

### E-L6 LOW: sentence scores aggregated to paragraph level
- **Where:** `ai_prefilter_sentences.py:180-199` computes max/mean of sentence scores per `text_hash` before writing.
- **Task:** Optionally write one row per sentence to `data/interim/prefilter_sentence_scores/` and compute the
  `sent_max_*`/`sent_mean_*` features where they are joined (`ai_prefilter_classify.join_sentence_scores`). Acceptable as-is,
  because the output is one feature row per atomic text.

### E-L7 LOW: enrichment writes tables into the shared DuckDB
- **Where:** `ai_entity_mentions.py:57-59` (`read_only=False` plus `register_entity_terms` DROP/CREATE), `ai_prefilter.py:272-290`
  (`ensure_anchor_table`), `ai_prefilter_anchors.py:114-140`.
- **Task:** Use in-memory or TEMP tables, or read `bronze.prefilter_entity_terms` and `bronze.prefilter_anchors`. Drop the
  persistent DB writes.

### E-L8 LOW: golden_set sampling reads silver and imports raw_ingestion code
- **Where:** `golden_set.py:233-236` reads `silver.filing_manifest*` and `silver.firm_universe`; `:189` imports
  `raw_ingestion/us/sector_map`.
- **Why it matters:** silver is built after enrichment. There is no cycle today, because the silver universe tables do not
  depend on LLM outputs.
- **Task:** Filter `bronze.firm_universe` with the `ANALYSIS_PANEL` membership constant, moved from `silver/_paths.py` to
  `common/layers.py`. Move `sector_map.py` to `scripts/common/` (see R-L2).

### E-L9 LOW: enrichment reads the legacy DuckDB and interim prediction globs instead of bronze
- **Where:** `ai_classify.py:434-481` (and `:457-461` interim predictions glob), `ai_activities_from_frames.py:407-481`,
  `ai_embed.py:310`, `ai_prefilter.py:353`, `ai_prefilter_sentences.py:76-145`, `ai_entity_mentions.py:49,57-83`,
  `reconcile_ai_{classify,activities}_after_resegment.py:37-58/47-64`.
- **Task:** Switch to `L.scan("bronze.paragraphs" | "bronze.unique_paragraphs" | "bronze.sentences" | "bronze.prefilter_predictions")`
  and `silver.filing_manifest*`. Part of C-M1.

### E-L10 LOW (no action): run manifests with run counts
- `ai_classify.py:572-578`, `ai_activities_from_frames.py:533-539`, `golden_set.py:649-661`, `ai_embed.py:193, 447`,
  `ai_prefilter.py:265-267` and `ai_prefilter_sentences.py:202-206` record requested, done, failed and parts written. These
  are run logs and are acceptable; keep them.

---

## 3. gold

### G-H1 HIGH: dependency cycle among strategy_dimensions, activity_profiles and build_firm_panels
- **The cycle:**
  - `build_strategy_dimensions.py:377-379` reads `firm_activities.parquet` (from `activity_profiles.py:335`).
  - `activity_profiles.py:332` reads `firm_strategy_dimensions.parquet`, and `:290` reads `firm_year_master_v2.parquet`.
  - `build_firm_panels.py:93,103` builds `firm_year_master_v2` from strategy dimensions.
  - The Makefile runs `build_strategy_dimensions.py`, then `activity_profiles.py`, then `build_strategy_dimensions.py`
    again (Makefile ~170-173) to converge.
- **Effect:** results depend on run order and stale files. If `firm_activities` is missing, `promotional_excess` is NaN and
  `n_activities` is 0 (`:434-435`).
- **Task:**
  - (a) Create `scripts/gold/posture/build_firm_activities.py`: `silver.ai_activities` + `silver.ai_frames` +
    `ai_intensity.document_table()` (for `n_words`, replacing master_v2) → `data/gold/intermediate/posture/`
    `{firm_activities, firm_year_activities, channel_activity_cells}.parquet`, with no archetype join.
  - (b) `build_strategy_dimensions.py` reads (a) as a declared input.
  - (c) Archetype-joined activity profiles go to analytics (G-M3).
  - (d) Remove the double run from the Makefile.
- **Deps:** P1. G-M1, G-M3 and G-H8b follow.

### G-H2 HIGH: channel_gap_analysis.py is a DiD analysis living in gold, feeding a gold dataset
- **Where:** `scripts/gold/channel_gap/channel_gap_analysis.py`:
  - `:67` statsmodels
  - `:263-325` FE DiD with clustered SEs, event study, pretrend F
  - `:344-355` t-stats vs zero
  - `:366-373` robustness
  - `:399-418` notorious cases
  - `:421-435` Spearman against the washing score and gap by segment, reading **stale `firm_segments.parquet`** from
    `scripts/deprecated/build_segments.py`
  - `:441-451` writes `channel_gap_analysis.json`
- **Consumers:** the same script writes `channel_gap_cells.parquet` (`:439`), which `gold/consolidate/channel_gap/build_dataset.py:20`
  and thesis.qmd `3758` read. `analytics/channel_gap/channel_gap_words_robustness.py:26-27` imports `load_frames` and
  `document_counts` from it.
- **Task:** Split into:
  - `scripts/gold/channel_gap/build_channel_gap_cells.py` (`load_frames`, `load_documents`, `cells`, `cells_extensive`,
    firm gap averages) → `data/gold/intermediate/channel_gap/{cells_extensive,cells_paired,firm}.parquet`.
  - `scripts/analytics/channel_gap/channel_gap_did.py` (estimate, event study, robustness, levels, notorious, cross)
    → `data/results/channel_gap/channel_gap_did.json`.
  - Drop the `firm_segments` cross, which reads a deprecated construct, or replace it with the archetype from
    `data/gold/predictions/yearly/posture_archetype_static.parquet`.
- **Also repoint:** consolidate `build_dataset.py` SRC, the words_robustness import, and qmd 3758.

### G-H3 HIGH: crash_archetypes script mixes a model fit with regressions and reads a file nobody builds
- **Where:** `scripts/gold/crash_archetypes/build_call_crash_and_archetypes.py`
  - (a) Expanding archetype fit `:30-131` → `panel_expanding_archetypes.parquet`, plus models. Consumed by
    `consolidate/predictions/posture_archetype_expanding_yearly.py:26`, `analytics/crash_archetypes/call_archetype_full_battery.py:83`
    and qmd 3783.
  - (b) OLS NCSKEW/DUVOL with clustered SEs `:162-244`, prints of beta/se/p, and `call_crash_archetype_regressions.csv`
    plus `_samples.csv` (qmd 3481-3482).
  - (c) Reads `call_crash_risk_panel.parquet` (`:27,135`), which has **no producer** in the repo or its history.
- **Task:**
  - (1) Move the fit to `scripts/gold/posture/build_archetype_weights_yearly_asof.py` (sibling of the quarterly asof script),
    writing `data/gold/intermediate/posture/panel_expanding_archetypes.parquet` and `models/posture_archetype_expanding_yearly/`.
  - (2) Add `scripts/gold/call_beta/build_call_crash_risk_panel.py` (NCSKEW/DUVOL pre/post per call from
    `bronze.market_prices` and `bronze.market_factors_daily`; `build_ncskew_63.py` is the template)
    → `data/gold/intermediate/call_beta/call_crash_risk_panel.parquet`.
  - (3) Move the regressions to `scripts/analytics/crash_archetypes/call_crash_regressions.py`
    → `data/results/crash_archetypes/{regressions,regression_samples}.csv`. The `w_call` rank construction (`:143`) is a
    covariate and goes into the gold call dataset.
- **qmd:** 3481-3482 and 3783 change. **Deps:** (2) before A-H2.

### G-H4 HIGH: a gold predictions script reads a model that analytics writes
- **Where:** `scripts/gold/consolidate/predictions/incremental_signal_coefficients.py:34-62` reads
  `models/incremental_signal/<outcome>/model.pkl`, written by `scripts/analytics/shock/incremental_signal.py:313-327`, and
  emits a coefficient table with p-values and CIs (`:51-52`).
- **Rule broken:** gold reads analytics output, and gold holds a regression table.
- **Task:** Delete the gold script. `analytics/shock/incremental_signal.py` writes
  `data/results/shock/incremental_signal_coefficients.csv` directly. Either stop writing `models/incremental_signal/`, or
  write it under `data/results/shock/models/` if the fitted objects must be kept. Move
  `data/gold/predictions/yearly/incremental_signal_coefficients.parquet` and `models/incremental_signal/` to `data/deprecated/`.
  Remove the entry from the Makefile gold chain.
- **qmd:** reads `incremental_signal.json` at 227, 3553, 4549. These move to `data/results/shock/` with A-M6e.

### G-H5 HIGH: external_patents is ingestion in gold, and it writes into data/raw and scripts/
- **Where:** `scripts/gold/external_patents/*`
  - BigQuery pulls and loads: `run_oecd_ai_patents.py:35-53`, `build_sec_preshock_patents.py:106-127`,
    `upload_aliases_to_bigquery.py:26-78`.
  - Writes into `data/raw`: `build_sp500_patent_aliases.py:17-20, 355-359`. That is gold writing into raw, the wrong direction.
  - Generates SQL into `scripts/analytics/*.sql` (`build_oecd_patents_panel.py:215`, `build_sec_preshock_patents.py:102`),
    which gold then reads back (`build_sec_preshock_patents.py:25,30`). Those .sql files do not currently exist.
  - Outputs `data/processed/sp500_firm_year_ai_patents_oecd2025.{parquet,csv}` (qmd 4154) and
    `data/processed/sp500_firm_preshock_patent_capacity.parquet`.
- **Task:**
  - Move the scripts to `scripts/raw_ingestion/patents/`. The alias seed goes to `data/raw/reference/`, BigQuery results to
    `data/raw/patents/*.parquet`, and SQL templates to `scripts/raw_ingestion/patents/sql/`.
  - Add `scripts/bronze/patents.py` → `bronze.patents_firm_year` and `bronze.patents_preshock` (register in `layers.TABLES`).
  - Add `scripts/gold/consolidate/patents/build_covariates_yearly.py` → `data/gold/covariates/yearly/patents.parquet`.
- **qmd:** 4154 → data/gold covariates.

### G-H6 HIGH: gold consolidate scripts read `data/processed/clusters` (the antipattern)
These are thin copies from processed into data/gold. Each subtask repoints SRC after its upstream builder moves (G-H8x).

- **G-H6a** Yearly families built from `firm_year_master_v2`:
  - `consolidate/disclosure_volume/build_covariates_yearly.py:19,33`
  - `financial_ratios/build_covariates_yearly.py:32,51`
  - `market/build_covariates_yearly.py:19,26`
  - `posture/build_covariates_yearly.py:21,29`
  - `firm_year/build_targets.py:33,50`
  - **Task:** Build each family directly from its source intermediate: financials, market factors, strategy dimensions,
    `document_panel`/`ai_intensity`. This removes `firm_year_master_v2` as a dependency. Deps: G-H8a, G-H8b, G-H8c.
- **G-H6b** Quarterly families and call targets built from `call_beta_main_panel_10k10q_asof`:
  - `consolidate/call/build_targets.py:24`
  - `disclosure_volume/build_covariates_quarterly.py:21`
  - `financial_ratios/build_covariates_quarterly.py:19`
  - `market/build_covariates_quarterly.py:19`
  - **Task:** Read `data/gold/intermediate/call_beta/call_panel.parquet`. Deps: G-H8d.
- **G-H6c** `consolidate/activities/build_covariates_yearly.py:20` and `consolidate/predictions/washing_score.py:39`
  (`firm_year_activities`). **Task:** Read `data/gold/intermediate/posture/firm_year_activities.parquet`. Deps: G-H1.
- **G-H6d** `consolidate/predictions/posture_archetype_expanding_quarterly.py:28` and `..._yearly.py:26`. **Task:** Read the
  new intermediate paths. Deps: G-H3(1), G-H8b.
- **G-H6e** `consolidate/channel_gap/build_dataset.py:20`. Deps: G-H2.

### G-H8 HIGH: gold builders write analysis panels into `data/processed/clusters`
Consumers depend on these files; see the map in section 0. For each subtask: change OUT_DIR to
`GOLD_INTERMEDIATE/<topic>`, then repoint the listed consumers and the Makefile targets
(`analytics-financials`, `analytics-text`, `analytics-panels`, `analytics-call-beta`).

- **G-H8a financials:** `build_firm_financials.py:84,685`, `build_market_factors.py:62,152,201`, `build_roic_wacc.py:65,98,153`,
  `build_us_10q_financials_panel.py:357-359` (writes into `data/processed/` root) → `data/gold/intermediate/financials/`.
- **G-H8b posture:** `build_document_panel.py:24,31`, `build_strategy_dimensions.py:103,476-478`,
  `build_archetype_weights_quarterly_asof.py:27,66,139`, `activity_profiles.py:55,334-338` (after G-H1)
  → `data/gold/intermediate/posture/`. Outputs `firm_strategy_dimensions` and `firm_year_strategy_dimensions` are
  archetype labels, i.e. predictions; consider writing them as `data/gold/predictions/{yearly,pooled}/posture_archetype_static*`.
- **G-H8c washing:** `build_firm_panels.py:59,90,172-173` and `washing_score.py:92,298,313` → `data/gold/intermediate/washing/`.
  Treat `firm_year_washing_score` as a prediction (`models/washing_grounding_shrinkage` exists).
- **G-H8d call_beta:** `build_call_beta_panel.py:82-86,304`, `build_ncskew_63.py:15,79`,
  `crash_archetypes/build_call_fundamentals_panel.py:52-57,180` → `data/gold/intermediate/call_beta/`.
- **Consumer and qmd path changes:** every `_p(...)` / `find_path("data/processed/clusters/...")` in thesis.qmd for these
  files: 150, 160, 170, 192, 345, 351, 700, 796, 927, 2373, 3973, plus the `_p` helper at 119. The allowed alternative is to
  keep the qmd reads temporarily but point `_p` at the new directories.

### G-M1 MEDIUM: strategy_dimensions persists statistics, diagnostics and regressions
- **Where:** `build_strategy_dimensions.py`
  - `:313-316` correlation matrix
  - `:207-211, 318-321` PCA diagnostic
  - `:376-433` OLS `promotional_posture ~ log_activities` (with and without a frames control) and archetype activity-volume
    regressions with p-values and CIs
  - `:464-470` year-over-year persistence
  - `:480-492` all of the above in `strategy_dimensions_manifest.json`
- **Consumer:** thesis.qmd 211-216, 2182, 4342.
- **Task:**
  - Keep the Archetypal Analysis fit and bootstrap-floor k selection (model selection). Store `k` and `stability_by_k` in
    `models/posture_archetype_static/model.pkl`.
  - Move correlation, PCA, regressions, persistence and cluster sizes to
    `scripts/analytics/posture/strategy_dimensions_diagnostics.py` → `data/results/posture/strategy_dimensions_diagnostics.json`.
  - The `promotional_excess` residual is a model output. Move it to
    `scripts/gold/consolidate/predictions/promotional_excess.py` with a model at `models/promotional_excess/model.pkl`.
- **qmd:** 211-216, 2182 and 4342 read the results JSON. **Deps:** G-H1.

### G-M2 MEDIUM: stale posture data consumed through bootstrap stability
- **Where:** `analytics/posture/bootstrap_archetype_stability.py:55` imports `build_strategy_dimensions` internals.
  Acceptable (analytics importing gold code), but it breaks if G-M1 moves functions.
- **Task:** Keep the imported functions (`build_posture`, `shrink_to_prior`, `bootstrap_stability`) in gold and re-export
  them. Coordinate with G-M1.

### G-M3 MEDIUM: activity_profiles writes a descriptive report and bypasses bronze/silver
- **Where:** `activity_profiles.py:346-466` computes distributions, top behaviours, archetype crosstabs, exemplars, named
  providers → `activity_profiles.json`. `:145-160 load()` reads `data/interim/ai_activities/*.parquet` directly with its own
  "any run found activity" dedup, bypassing `bronze.ai_activities` / `silver.ai_activities`.
- **Consumer:** qmd 222-226, 4367, 4416.
- **Task:**
  - Move the report to `scripts/analytics/posture/activity_profiles_summary.py` → `data/results/posture/activity_profiles.json`.
  - The gold part (G-H1a) reads `silver.ai_activities`.
  - If the any-run-true dedup rule is intended, move it into `scripts/bronze/llm_outputs.py:26-40` so every consumer gets it.

### G-M4 MEDIUM: build_geo_provenance is purely descriptive
- **Where:** `scripts/gold/posture/build_geo_provenance.py:108-150` (crosstab shares and counts by year and archetype, reach
  stats) → `data/processed/clusters/geo_provenance_summary.json`. No consumer.
- **Task:** Move to `scripts/analytics/posture/geo_provenance.py` → `data/results/posture/geo_provenance_summary.json`. Read
  the archetype from data/gold predictions. Update Makefile ~175.

### G-M5 MEDIUM: washing_score contains a validation battery
- **Where:** `scripts/gold/washing/washing_score.py:86` (scipy), `:235-261 validations()` (Spearman consistency,
  persistence, economic coherence, with p-values), `:273-281` prints, `:315-322` `firm_year_washing_score_manifest.json`
  with `validations`.
- **Consumer:** `analytics/washing/validate_washing_score.py:35-37` imports `build` and `validations` from gold.
- **Task:** Move `validations()` into `validate_washing_score.py`, which already writes `washing_score_validation.json`,
  read at qmd 220, 2695-2777 and 4642. The analytics script reads the persisted gold washing score instead of calling
  `build()`. The gold manifest keeps only n/tail_q/inputs.
- **qmd:** `washing_score_validation.json` moves to `data/results/washing/` (A-M6f).

### G-M6 MEDIUM: gold reads data/raw directly, skipping bronze
- **Where:**
  - Prices and FF3: `build_call_beta_panel.py:84-85,197,206`, `build_ncskew_63.py:16-17,37,47`,
    `build_call_fundamentals_panel.py:55-56,111,161`, `build_market_factors.py:60,78,157`, `build_roic_wacc.py:107`.
  - Raw XBRL: `build_firm_financials.py:~280-290` (facts glob), `build_us_10q_financials_panel.py:216,244` (`raw_xbrl_frames_alt`).
- **Task:** Use `L.scan("bronze.market_prices")` and `"bronze.market_factors_daily"`. Add `scripts/bronze/xbrl.py`
  → `bronze.xbrl_facts` (from `data/raw/xbrl_facts/us_by_filing`) and `bronze.xbrl_frames_10q`, and read those.

### G-M7 MEDIUM: analytics recomputes gold data in-process instead of reading persisted gold
- **Where:**
  - `ai_intensity.document_table/aggregate` imported by `analytics/shock/incremental_signal.py:59-61`,
    `shock_analysis.py:95-96`, `shock_did_simple.py:65-66`, `washing/firm_year_aggregation_robustness.py:37-39`,
    `posture/bootstrap_archetype_stability.py:58`.
  - `washing_score.load_10k_years` imported by `incremental_signal.py:62`.
  - `channel_gap_analysis.load_frames` imported by `channel_gap_words_robustness.py:27`.
- **Why it matters:** analytics outputs cannot be reproduced from data/gold alone.
- **Task:** Analytics reads `data/gold/intermediate/posture/document_panel.parquet`, which `build_document_panel.py` already
  persists, plus gold datasets. Keep only pure helper imports. One PR per analytics topic, after G-H8b.

### G-L8 LOW: print-only diagnostics in gold
- `build_call_beta_panel.py:269-280 validate()` diffs against the previous panel. `build_firm_financials.py:689-693` prints
  coverage. `build_market_factors.py:203-205` prints medians. These are run logs; no action. Optionally move
  `validate()` to `scripts/verif/call_beta_panel_diff.py`.

### G-L9 LOW: orphan gold outputs
- `build_firm_panels.py:170-171` writes `cohort_2021_crosscheck` and `segment_financials`, and nothing reads them.
  `data/processed/clusters/gbdt_prefilter_importance.json` has no producer.
- **Task:** Drop the two writes; archive the JSON to `data/deprecated/`.

---

## 4. analytics

### A-H1 HIGH: incremental_signal.py writes a model that gold reads
Same fix as G-H4. Listed here so the analytics owner sees it.

### A-H2 HIGH: analytics reads a stale or unbuilt panel
- **Where:** `call_beta/call_beta_config_decoupling_asof.py:122` and `crash_archetypes/call_archetype_full_battery.py:35,81,176`
  read `call_crash_risk_panel.parquet`, which has no producer. `shock/shock_analysis.py:120-122` reads `firm_segments.parquet`,
  output of `scripts/deprecated/build_segments.py`.
- **Task:** Repoint the crash readers to G-H3(2). In `shock_analysis.py`, delete the "segmento" heterogeneity block, or
  switch it to archetype predictions from data/gold. The `sec_did_continuous.json` and `shock_analysis.json` contents change;
  check any thesis text that cites segment heterogeneity.

### A-H3 HIGH: one analytics script reads another analytics script's output
- **Where:** `shock/evolution_figures.py:59` reads `shock_analysis.json`, written by `shock/shock_analysis.py:247,260`.
- **Task:** Move the SEC event-study figure (`evolution_figures.py:56-71`) into `shock_analysis.py`, drawing from the
  in-memory report. Alternatively, recompute the estimate inside `evolution_figures.py`. Delete the JSON read either way.

### A-M4 MEDIUM: analytics builds datasets
- **Where:**
  - `call_beta/call_car_regressions.py:60-217` builds a CAR/SUE call panel from raw prices (`:37,87`), raw FF3 (`:38,121`),
    the interim 10-Q manifest (`:36,70,155`) and `data/processed/us_10q_financials_panel.parquet` (`:35,69,152`), then
    persists `call_car_panel.parquet` (`:217`), which nothing reads.
  - `call_beta/call_beta_regressions.py:32,77-94` attaches leverage covariates from raw XBRL facts (`attach_leverage`).
- **Task:**
  - Create `scripts/gold/call_beta/build_call_car_panel.py` (bronze prices/factors, `silver.filing_manifest_10q`,
    gold 10-Q financials) → `data/gold/intermediate/call_beta/call_car_panel.parquet`.
  - Move `attach_leverage` into `build_firm_financials.py` (debt_to_equity and liabilities_to_assets per accession), or into
    the quarterly `financial_ratios` covariate family.
  - Analytics only regresses. Deps: G-M6 (bronze.xbrl_facts), G-H8a.

### A-M5 MEDIUM: earnings_calls_analysis writes a data table and reads the legacy DB
- **Where:** `appendix/earnings_calls_analysis.py:49-50, 93, 214-219` reads `duckdb/thesis.duckdb` and writes
  `earnings_calls_summary.parquet` (per-firm table) and `.json` into `data/processed/clusters`.
- **Task:** Read `silver.filing_manifest` and `silver.ai_frames`. Write both files to `data/results/appendix/`.

### A-M6 MEDIUM: analytics outputs go to data/processed or docs/analytics instead of data/results/<topic>
One small PR per topic. Change the output constant, then update the listed thesis.qmd reads.

- **A-M6a call_beta:**
  - `call_beta_regressions.py:33,171-172`, `call_beta_robustness.py:40,369-371`, `call_beta_generalized_targets.py:40-41,108-109`
    and `call_car_regressions.py:39,224,241` → `data/results/call_beta/`.
  - `sec_comment_letter_cases.py:47,170` writes `data/processed/sec_comment_letter_cases.json` → `data/results/call_beta/`.
  - qmd: `call_beta_regressions.csv` (4 reads), `call_beta_regression_samples.csv`, `call_beta_robustness_{windows,market_model,firm_fe,delta_beta,placebo}.csv`,
    `call_beta_generalized_targets.csv`, `data/processed/sec_comment_letter_cases.json` (`_pp`, ~3102).
- **A-M6b channel_gap:** `channel_gap_words_robustness.py:29,93` → `data/results/channel_gap/`. qmd 208.
- **A-M6c crash_archetypes:** `call_archetype_full_battery.py:165,204,234` → `data/results/crash_archetypes/`. The self-read
  at `:218` of its own `OUT_TARGETS` is fine within one script. qmd: `call_archetype_full_battery_targets.csv`.
- **A-M6d posture:**
  - `bootstrap_archetype_stability.py:61-62,163-164` writes to both `data/processed/clusters` **and** `docs/analytics/`.
  - `check_archetype_document_channels.py:36,147` writes `docs/analytics/archetype_document_channels.json`.
  - `plot_archetypal_simplex.py:317-319` writes `docs/analytics/fig_archetypal_ternary_simplex.png`.
  - All → `data/results/posture/`. qmd: 1812 (`docs/analytics/bootstrap_jaccard_200_results.json`) and `_p("bootstrap_jaccard_200_results.json")`.
- **A-M6e shock:** `evolution_figures.py:27,46,54,71`, `incremental_signal.py:65,405`, `shock_analysis.py:64,247,281` and
  `shock_did_simple.py:46,186,215` → `data/results/shock/`. qmd 227, 3553, 4549 (`incremental_signal.json`).
- **A-M6f washing:** `activity_grounding.py:34,85,92`, `build_strategy_economic_profiles.py:29,82`,
  `firm_year_aggregation_robustness.py:42,91` and `validate_washing_score.py:99` (imports gold's OUT_DIR) → `data/results/washing/`.
  qmd: 220, 234, 2695-2777, 4642, `_p("activity_grounding.json")` ×2, `_p("washing_score_validation.json")` ×3.
- **A-M6g appendix:** `report_crosscheck_stats.py:45,267` and `earnings_calls_analysis.py` (A-M5) → `data/results/appendix/`.

### A-L7 LOW: analytics reads raw FF3 and an analytics module
- `call_beta_robustness.py:82` reads raw FF3; switch to `bronze.market_factors_daily`.
- `firm_year_aggregation_robustness.py:38-40` imports the analytics module `incremental_signal`. Code reuse only; acceptable.

---

## 5. common

### C-M1 MEDIUM: `common/build_duckdb.py` is a data-transformation stage living in common
- **What it does:** builds `duckdb/thesis.duckdb` views with real logic that duplicates bronze and silver:
  - paragraph splitting with a gaps-and-islands LAG (`:380-470`, duplicating `bronze/text_split.py`)
  - `gold_ai_frames` / `gold_ai_activities` / `gold_ai_entity_mentions` broadcast plus prefilter filter from interim globs
    (`:1150-1290`, duplicating `silver/ai_outputs.py`)
  - universe filtering (`:211`)
- **Consumers:**
  - enrichment: `ai_classify`, `ai_activities_from_frames`, `ai_embed`, `ai_prefilter`, `ai_prefilter_sentences`,
    `ai_entity_mentions`, `reconcile_*`
  - verif: `prefilter_form_validation`, `prefilter_logit_refit`, `prefilter_model_variants_eval`, `prefilter_rescue_eval`
  - analytics: `appendix/earnings_calls_analysis`
  - thesis.qmd: 85, `gold_ai_frames`/`paragraphs` in 140-640
- **Task:** Migrate the consumers to `layers.scan` (E-L9, V-L6, A-M5). thesis.qmd reads bronze/silver parquet via DuckDB
  `read_parquet`, or reads the funnel from E-M1a. Then move `build_duckdb.py` to `scripts/deprecated/`. Optionally keep a
  thin `scripts/common/duckdb_views.py` that only creates `CREATE VIEW x AS SELECT * FROM read_parquet(<layers path>)`,
  with no logic.

### C-L2 LOW: stage-specific libraries live in common
- `common/section_extraction.py` is used only by `raw_processing/*/02_extract_sections.py`.
- `common/filing_xbrl_facts.py` is a parser for `raw_processing/us/04_extract_inline_xbrl_facts.py`.
- **Task:** Move both to `scripts/raw_processing/lib/`.

---

## 6. raw_ingestion / raw_processing / bronze / silver

- **R-L1 LOW:** `raw_processing/us/earnings_calls/03_fill_gaps_equibles.py` (MCP HTTP fetch, gzip JSON to data/raw,
  `:97, 314-341`) and `04_fill_gaps_stockanalysis.py` (headless Chrome scrape, `:302-329`) are downloads. Move both to
  `scripts/raw_ingestion/us/earnings_calls/`.
- **R-L2 LOW:** `raw_ingestion/us/sector_map.py` is a resolver library, not a fetcher. `enrichment/golden_set.py:189` and
  `tui_tickers.py:39` import it. Move it to `scripts/common/`.
- **R-L3 LOW (no action):** `raw_ingestion/us/sec_letters/01_fetch_filings.py:196` prints `value_counts` and
  `tui_tickers.py:150` groups for display. Run logs only.
- **B-L1 LOW:** `scripts/bronze/_paths.py:10` puts `scripts/enrichment` on the path, and `bronze/prefilter.py:60-67`
  imports `ai_prefilter_anchors` to materialize anchors and entity terms. Bronze depends on enrichment code. Move the anchor
  and entity-term config loaders (pure YAML readers) to `scripts/common/prefilter_config.py`.
- **silver:** no violations found.
- No aggregation was found in raw_processing beyond the latest-run dedup in `section_segmenter.load_extraction_trace:937-940`,
  which is fine.

---

## 7. Orchestration and the thesis document

- **M-L1 LOW:** Makefile `analytics-*` targets (~146-275) run gold builders and analytics scripts interleaved in one
  `bash -c` chain. `analytics-text` runs `build_strategy_dimensions` twice to settle the G-H1 cycle. `analytics-gold`
  interleaves washing gold and analytics. After G-H1, G-H8 and A-M6, split into `gold-*` targets (cached on layer manifests)
  and `analytics-*` targets that depend only on `gold-*` outputs.
- **Q-L1 LOW (informational):**
  - thesis.qmd writes `docs/analytics/appendix_g_*.csv` and `appendix_g_slide_manifest.txt` (4885-4972). Analysis inside the
    document is acceptable for now; if the files are needed, move them to `scripts/analytics/appendix/appendix_g_exports.py`
    → `data/results/appendix/`.
  - Stale `docs/analytics/` artifacts with no producer script: `fig_geo_provenance_test.png`,
    `sec_scrutiny_event_study_{2panel,main_4period}.png`, `sec_targeted_event_study_2panel.png`,
    `fig_appendix_archetypal_sectors.png`. Archive them.

---

## Suggested order (dependency chain)

1. **P1**
2. **V-H1** → **E-M1a** → **E-M1b / E-M2 / E-M3 / E-M4 / E-M5**, then **V-M2..M5**
3. **G-M6** (bronze.xbrl_facts, bronze prices) → **G-H8a** (financials)
4. **G-H1** (break the cycle) → **G-H8b** → **G-M1, G-M3, G-M4**
5. **G-H8c** + **G-M5** (washing) → **G-H6a / G-H6c**
6. **G-H8d** + **G-H3** (crash panel builder, split) → **G-H6b / G-H6d** → **A-H2**
7. **G-H2** (channel gap split) → **G-H6e**
8. **G-H4 / A-H1** (incremental signal)
9. **G-H5** (patents to raw_ingestion, bronze, gold covariate)
10. **A-H3**, **A-M4**, **A-M5**, **A-M6a..g** (topic by topic, updating thesis.qmd reads each time), **G-M7**
11. **C-M1** (retire build_duckdb once E-L9 / V-L6 / A-M5 / qmd are migrated) → **M-L1** → **P2**

## Counts

| stage | HIGH | MEDIUM | LOW |
|---|---|---|---|
| verif | 1 | 4 | 1 |
| enrichment | 0 | 6 (E-M1a, E-M1b, E-M2..M5) | 5 (E-L6..L10; E-L10 no action) |
| gold | 14 (G-H1..H5, G-H6a..e, G-H8a..d) | 7 (G-M1..M7) | 2 |
| analytics | 3 (A-H1 duplicates G-H4) | 9 (A-M4, A-M5, A-M6a..g) | 1 |
| common | 0 | 1 | 1 |
| raw_ingestion / raw_processing / bronze / silver | 0 | 0 | 4 |
| orchestration / qmd | 0 | 0 | 2 |
| **total** | **18** (17 unique) | **27** | **16** |
