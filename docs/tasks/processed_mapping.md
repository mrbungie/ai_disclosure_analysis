# data/processed/ mapping (135 files: 130 in clusters/, 5 at top level)

Read-only trace, 2026-09-15. Paths are relative to the repo root. Script paths drop the `scripts/` prefix.
Abbreviations: **C** = `gold/consolidate/*` reader, **G** = other `gold/*` reader, **A** = `analytics/*` reader, **Q** = `thesis_document/thesis.qmd` (render.py only mentions files in docstrings and never reads them).
"(indirect)" means the script imports a function from another script that does the read.

Target-path rule (user clarification): writer in `scripts/gold/` goes to `data/gold/...` (fitted models go to `models/`). Writer in `scripts/analytics/` goes to `data/results/<topic>/`. Files from deprecated writers or with no writer go to `data/deprecated/processed/...`.

---

## 1. PANEL (intermediate datasets written by scripts/gold/<topic>/)

| file | writer | C readers | G readers | A readers | Q | gold equivalent / missing | proposed target |
|---|---|---|---|---|---|---|---|
| clusters/call_beta_main_panel_10k10q_asof.parquet | gold/call_beta/build_call_beta_panel.py | call/build_targets, disclosure_volume/build_covariates_quarterly, financial_ratios/build_covariates_quarterly, market/build_covariates_quarterly | call_beta/build_ncskew_63, crash_archetypes/build_call_fundamentals_panel, crash_archetypes/build_call_crash_and_archetypes | call_beta/call_beta_regressions, call_beta_robustness, call_car_regressions, call_beta_generalized_targets, call_beta_config_decoupling_asof; crash_archetypes/call_archetype_full_battery | yes (`_p`, l.4017) | **datasets/quarterly/call.parquet**: every column is there. Renames: beta_post_63/126/252 became beta_post_*d and return60 became return_post_60d. **Missing: `accession_number`** (the as-of 10-K/10-Q accession; `attach_leverage` in call_beta_regressions joins raw XBRL facts on it). | spine for data/gold (call) (internal); the columns already live in covariates/targets/datasets quarterly |
| clusters/call_fundamentals_panel.parquet | gold/crash_archetypes/build_call_fundamentals_panel.py | none | crash_archetypes/build_call_crash_and_archetypes | call_beta/call_beta_generalized_targets, crash_archetypes/call_archetype_full_battery | no | **none**. The file has 12 pre/post-call columns: log_market_cap, rd_intensity, gross_margin, ps_ratio, next_revenue_yoy, roic_minus_wacc (each with _pre and _post). Needs `_pre` in covariates/quarterly (a new family, or added to financial_ratios/market) and `_post` in targets/quarterly/call with window names. | data/gold/covariates/quarterly/fundamentals_pre.parquet + data/gold/targets/quarterly/call.parquet (post cols) |
| clusters/call_crash_risk_panel.parquet | **NO WRITER** (never tracked in git; first mentioned in d9d4405) | none | crash_archetypes/build_call_crash_and_archetypes | call_beta/call_beta_config_decoupling_asof, crash_archetypes/call_archetype_full_battery | no | **none**. Columns: ncskew_pre/post, duvol_pre/post, keyed on ticker+fecha (no call_accession_number). A builder must be written first. | data/gold/covariates/quarterly/crash_risk (pre) + targets/quarterly/call (ncskew_post_*, duvol_post_*) |
| clusters/ncskew_63.parquet | gold/call_beta/build_ncskew_63.py | none | none | call_beta/call_beta_config_decoupling_asof | no | **none** (ncskew_pre_63, ncskew_post_63) | covariates/quarterly/crash_risk (ncskew_pre_63d) + targets/quarterly/call (ncskew_post_63d) |
| clusters/panel_expanding_archetype_weights_quarterly.parquet | gold/posture/build_archetype_weights_quarterly_asof.py | predictions/posture_archetype_expanding_quarterly | none | call_beta/call_beta_config_decoupling_asof | no | **predictions/quarterly/posture_archetype_expanding.parquet**: same columns plus id. **Stale:** 4,977 vs 4,964 rows, max abs difference in w_voc 0.56. | model output belongs in data/gold/predictions/quarterly/ (writer should also dump models/posture_archetype_expanding/cutoff=*/model.pkl) |
| clusters/panel_expanding_archetypes.parquet | gold/crash_archetypes/build_call_crash_and_archetypes.py (l.129) | predictions/posture_archetype_expanding_yearly | none | crash_archetypes/call_archetype_full_battery | yes (l.3783) | **predictions/yearly/posture_archetype_expanding.parquet** (same arch_exp; 99.1% agreement, stale vintage) | data/gold/predictions/yearly/posture_archetype_expanding.parquet |
| clusters/document_panel.parquet | gold/posture/build_document_panel.py | none | posture/build_archetype_weights_quarterly_asof, crash_archetypes/build_call_crash_and_archetypes | call_beta/sec_comment_letter_cases | yes (`_p` ×4) | **none**. The grain is one row per document (channel, form, accession, per-document frame counts). Needs a document-grain covariate family. **AMBIGUOUS:** it is a pure aggregation of silver.ai_frames, so it could be silver instead. | data/gold/covariates/document/disclosure_counts.parquet (new grain), or silver.document_ai_counts |
| clusters/firm_year_master_v2.parquet | gold/washing/build_firm_panels.py | posture/, disclosure_volume/, financial_ratios/, market/ build_covariates_yearly; firm_year/build_targets | washing/washing_score, crash_archetypes/build_call_crash_and_archetypes, posture/activity_profiles | call_beta/sec_comment_letter_cases, shock/incremental_signal, shock/evolution_figures, appendix/report_crosscheck_stats, washing/build_strategy_economic_profiles, washing/firm_year_aggregation_robustness (indirect via incremental_signal.load), washing/validate_washing_score (indirect via washing_score.build) | yes (`_p`, l.345/796/2714/3738/4148/4607) | **datasets/yearly/firm_year.parquet** has all 75 other columns (targets renamed per build_targets RENAME). **Missing: `archetype`, `cluster`**, which live in predictions/yearly/posture_archetype_static.parquet. | spine/intermediate; content already in data/gold |
| clusters/firm_year_activities.parquet | gold/posture/activity_profiles.py | activities/build_covariates_yearly, predictions/washing_score | washing/washing_score | shock/incremental_signal; washing/validate_washing_score (indirect); washing/firm_year_aggregation_robustness (indirect) | no | **covariates/yearly/activities.parquet** (identical plus id; verified equal) | data/gold/covariates/yearly/activities.parquet |
| clusters/firm_year_washing_score.parquet | gold/washing/washing_score.py | none | none | call_beta/sec_comment_letter_cases, shock/incremental_signal, shock/shock_did_simple, shock/shock_analysis | yes (`_p`/find_path ×7) | **predictions/yearly/washing_score.parquet** (same 14 columns plus id). **Not identical:** 1,873 vs 1,869 rows, max abs difference in w 0.32. consolidate/predictions/washing_score.py reads `data/gold/covariates/yearly/firm_year.parquet`, **which does not exist**. | data/gold/predictions/yearly/washing_score.parquet |
| clusters/firm_year_strategy_dimensions.parquet | gold/posture/build_strategy_dimensions.py | none | washing/build_firm_panels | washing/build_strategy_economic_profiles (archetype) | yes (l.3652, 4268) | **covariates/yearly/posture.parquet** (posture dims verified equal) + **predictions/yearly/posture_archetype_static.parquet** (cluster, archetype; 99.3% agreement). consolidate/predictions/posture_archetype_static.py also reads the missing covariates/yearly/firm_year.parquet. | split: covariates/yearly/posture + predictions/yearly/posture_archetype_static |
| clusters/firm_strategy_dimensions.parquet | gold/posture/build_strategy_dimensions.py | none | posture/activity_profiles, posture/build_geo_provenance, washing/build_firm_panels | posture/check_archetype_document_channels, posture/plot_archetypal_simplex | yes (`_p` ×13, `_p_k2`) | **none at firm grain**. The file is pooled per firm: pooled posture dims, cluster, archetype, archetype_stability, promotional_excess, n_activities. The gold static prediction is firm-year only. | data/gold/predictions/firm/posture_archetype_static.parquet (new firm grain) |
| clusters/firm_year_full_crosscheck.parquet | gold/washing/build_firm_panels.py | none | none | appendix/report_crosscheck_stats | no | **Partial: datasets/yearly/firm_year.** **Missing:** revenue, rd_expense, capex, sga_expense (levels) and ret_m1_p5. | covariates/yearly/financial_levels (new) + targets/yearly/firm_year.ret_m1_p5d |
| clusters/firm_year_roic_wacc.parquet | gold/financials/build_roic_wacc.py | none | none (mentioned in build_firm_panels docstring only) | appendix/report_crosscheck_stats, washing/build_strategy_economic_profiles | no | **none**. Missing: effective_tax_rate, nopat, invested_capital, roic, rf_annualized, erp, cost_of_equity, cost_of_debt, wacc, roic_minus_wacc, wacc_market, roic_minus_wacc_market. | data/gold/covariates/yearly/value_creation.parquet (new family) |
| clusters/firm_year_financials_ratios.parquet | gold/financials/build_firm_financials.py | none | financials/build_market_factors, financials/build_roic_wacc, washing/build_firm_panels, call_beta/build_call_beta_panel, crash_archetypes/build_call_fundamentals_panel | call_beta/call_car_regressions (`attach_roa`: ticker, filing_date, roa); call_beta/call_beta_regressions (ROA_PATH/attach_roa defined but **never called**, a dead read) | no | **Partial: covariates/yearly/financial_ratios** (ratios only, keyed ticker_year). Missing the filing_date/accession keys and all levels (revenue, cost_of_revenue, opinc, net_income, da, interest, pretax, tax, eps_diluted, assets, equity, current_*, long_term_debt, cash, shares_out, ebitda). call_car needs an as-of **quarterly** roa event stream, not in gold. | source of covariates/yearly/financial_ratios (+ new financial_levels) |
| clusters/firm_year_financials.parquet | gold/financials/build_firm_financials.py | none | washing/build_firm_panels | none | no | Partial (growth leads in targets/yearly/firm_year; levels missing) | covariates/yearly/financial_levels |
| clusters/firm_year_market_factors.parquet | gold/financials/build_market_factors.py | none | financials/build_roic_wacc, washing/build_firm_panels | none | yes (`_p`, l.4487) | covariates/yearly/market + targets/yearly/firm_year; missing beta_n_obs | data/gold/covariates/yearly/market.parquet |
| clusters/firm_year_filing_returns.parquet | gold/financials/build_market_factors.py | none | washing/build_firm_panels | none | no | missing ret_m1_p5 | targets/yearly/firm_year (ret_m1_p5d) |
| clusters/firm_activities.parquet | gold/posture/activity_profiles.py | none | posture/build_strategy_dimensions, call_beta/build_call_beta_panel | none | yes (`_p` ×11) | none (one row per activity: silver.ai_activities + ticker/channel + family taxonomy). **AMBIGUOUS:** a deterministic enrichment of silver. | silver.ai_activities_enriched, or data/gold covariate source input |
| clusters/channel_activity_cells.parquet | gold/posture/activity_profiles.py | none | none | washing/activity_grounding | no | **none** (ticker × fy × channel grain; gold activities is firm-year pooled over channels) | data/gold/datasets/yearly/channel_activity.parquet (new) |
| clusters/channel_gap_cells.parquet | gold/channel_gap/channel_gap_analysis.py | channel_gap/build_dataset | none | none | yes (l.3758) | **datasets/yearly/channel_gap.parquet** (t renamed to year, plus id) | data/gold/datasets/yearly/channel_gap.parquet |
| clusters/channel_gap_firm.parquet | gold/channel_gap/channel_gap_analysis.py | none | none | none | no | none (firm-pooled; derivable) | no consumer: drop or data/gold/datasets/firm/channel_gap |
| clusters/cohort_2021_crosscheck.parquet | gold/washing/build_firm_panels.py | none | none | none | no | filter of full_crosscheck | no consumer: drop |
| clusters/segment_financials.parquet | gold/washing/build_firm_panels.py | none | none | none | no | firm medians of master_v2 | no consumer: drop |
| clusters/firm_activity_profiles.parquet | gold/posture/activity_profiles.py | none | none | none (Makefile dependency only) | no | none (firm-pooled) | no consumer: data/gold/covariates/firm/activity_profiles or drop |
| clusters/firm_washing_score.parquet | gold/washing/washing_score.py | none | channel_gap/channel_gap_analysis (optional `if exists`) | none | no | none (firm-pooled) | data/gold/predictions/firm/washing_score.parquet |
| us_10q_financials_panel.parquet | gold/financials/build_us_10q_financials_panel.py | none | none | call_beta/call_car_regressions (eps_diluted for SUE; net_income, assets for quarterly ROA) | no | **none**. Long XBRL metric table (ticker, year, quarter, metric, value, source, source_ref). | conformed source table: silver.xbrl_10q_financials. The derived SUE and quarterly ROA go to covariates/quarterly/financial_ratios. |

## 2. RESULT (analytics outputs, plus reports emitted by gold scripts)

| file | writer | readers | proposed target |
|---|---|---|---|
| clusters/activity_grounding.json | analytics/washing/activity_grounding.py | Q (l.2867, 2892) | data/results/washing/activity_grounding.json |
| clusters/fig_brecha_actividades.png | analytics/washing/activity_grounding.py | none | data/results/washing/ |
| clusters/firm_year_aggregation_robustness.json | analytics/washing/firm_year_aggregation_robustness.py | none | data/results/washing/ |
| clusters/strategy_economic_profiles.json | analytics/washing/build_strategy_economic_profiles.py | none (render.py docstring only) | data/results/washing/ |
| clusters/washing_score_validation.json | analytics/washing/validate_washing_score.py | Q (`_p` l.2694/2771/4605, l.4642) | data/results/washing/ |
| clusters/bootstrap_jaccard_200_results.json | analytics/posture/bootstrap_archetype_stability.py (also writes docs/analytics/ copy) | Q (l.1810) | data/results/posture/ |
| clusters/call_archetype_full_battery_targets.csv | analytics/crash_archetypes/call_archetype_full_battery.py | Q (l.3399, 4773) | data/results/crash_archetypes/ |
| clusters/call_archetype_full_battery_robustness.csv | same | none | data/results/crash_archetypes/ |
| clusters/call_archetype_full_battery_bh.csv | same | none | data/results/crash_archetypes/ |
| clusters/call_beta_generalized_targets.csv | analytics/call_beta/call_beta_generalized_targets.py | Q (l.3296) | data/results/call_beta/ |
| clusters/call_beta_generalized_targets_samples.csv | same | none | data/results/call_beta/ |
| clusters/call_beta_regressions.csv | analytics/call_beta/call_beta_regressions.py | Q (l.3218, 3297, 3361, 4690) | data/results/call_beta/ |
| clusters/call_beta_regression_samples.csv | same | Q (l.3262, 4694) | data/results/call_beta/ |
| clusters/call_beta_robustness_{windows,delta_beta,firm_fe,market_model,placebo}.csv | analytics/call_beta/call_beta_robustness.py | Q (windows l.3363/4689; delta_beta l.3365/4692; firm_fe l.3364/4691; market_model l.3362/4693; placebo l.3366) | data/results/call_beta/ |
| clusters/call_beta_robustness_{ai_correlation,ai_subsets,ai_vif,beta_threshold,call_frequency,influence,leave_one_sector_out,leave_one_year_out}.csv, call_beta_robustness_formal_tests.json | same (f-string `call_beta_robustness_{name}.csv`) | none | data/results/call_beta/ |
| clusters/call_car_regressions.csv, call_car_regression_samples.csv | analytics/call_beta/call_car_regressions.py (default outcome, prefix `call_car`) | none | data/results/call_beta/ |
| clusters/call_beta_sue_regressions.csv, call_beta_sue_regression_samples.csv | same (`--outcome beta_post_126 --include-sue`) | none | data/results/call_beta/ |
| clusters/call_car_panel.parquet | analytics/call_beta/call_car_regressions.py | none | **FLAG: a dataset written by analytics.** Call panel + beta_pre_car, idio_vol_pre, car_m1_p5, roa, assets/debt/equity/liabilities, leverage. Belongs in data/gold/datasets/quarterly/call_car.parquet via a gold builder, or drop it (a debug dump). |
| sec_comment_letter_cases.json | analytics/call_beta/sec_comment_letter_cases.py | Q (l.3105) | data/results/call_beta/ |
| clusters/incremental_signal.json | analytics/shock/incremental_signal.py | Q (l.3553, 4549) | data/results/shock/ |
| clusters/shock_analysis.json | analytics/shock/shock_analysis.py | **A: shock/evolution_figures** (l.59) | data/results/shock/ (results chained analytics→analytics; acceptable, flagged) |
| clusters/sec_did_continuous.json | analytics/shock/shock_analysis.py | none | data/results/shock/ |
| clusters/shock_did_simple.json | analytics/shock/shock_did_simple.py | none | data/results/shock/ |
| clusters/shock_did_{promo,quant,gov,spec,risk}_per_1k.png | analytics/shock/shock_did_simple.py (f-string over OUTCOMES) | none | data/results/shock/ |
| clusters/fig_evolucion_intensidad.png, fig_evolucion_composicion.png, fig_sec_event_study.png | analytics/shock/evolution_figures.py | none | data/results/shock/ |
| clusters/channel_gap_words_robustness.json | analytics/channel_gap/channel_gap_words_robustness.py | Q (l.3033) | data/results/channel_gap/ |
| clusters/earnings_calls_summary.json | analytics/appendix/earnings_calls_analysis.py | none | data/results/appendix/ |
| clusters/earnings_calls_summary.parquet | same | none | data/results/appendix/ (FLAG: parquet data from analytics, no reader) |
| clusters/crosscheck_stats.json | analytics/appendix/report_crosscheck_stats.py (only when `--json <path>` is passed) | none | data/results/appendix/ |
| clusters/call_crash_archetype_regressions.csv | **gold**/crash_archetypes/build_call_crash_and_archetypes.py (estimation inside a gold script) | Q (l.3481) | FLAG: split the regression into analytics/crash_archetypes/, then data/results/crash_archetypes/ |
| clusters/call_crash_archetype_regression_samples.csv | same | Q (l.3482) | same |
| clusters/channel_gap_analysis.json | **gold**/channel_gap/channel_gap_analysis.py (DiD/event-study estimates) | none (other scripts only import functions or mention it in docstrings) | FLAG: split into analytics/channel_gap/, then data/results/channel_gap/ |

## 3. OTHER (JSON manifests and summaries written by gold scripts)

| file | writer | readers | proposed target |
|---|---|---|---|
| clusters/activity_profiles.json | gold/posture/activity_profiles.py | Q (l.4367, 4416) | descriptive report, so data/results/posture/ once moved to analytics (AMBIGUOUS) |
| clusters/strategy_dimensions_manifest.json | gold/posture/build_strategy_dimensions.py | Q (l.2182, 4342) | model diagnostics (correlations, PCA): models/posture_archetype_static/manifest.json, or data/results/posture/ (AMBIGUOUS) |
| clusters/firm_year_washing_score_manifest.json | gold/washing/washing_score.py | none | models/washing_grounding_shrinkage/manifest.json |
| clusters/geo_provenance_summary.json | gold/posture/build_geo_provenance.py | none | data/results/posture/ once moved to analytics (AMBIGUOUS) |

## 4. EXTERNAL

| file | writer | readers | proposed target |
|---|---|---|---|
| sp500_firm_year_ai_patents_oecd2025.parquet | gold/external_patents/run_oecd_ai_patents.py (BigQuery OECD query) | Q (`_pp`, l.4154) | raw pull: data/bronze/oecd_ai_patents.parquet; conformed per firm-year: data/gold/covariates/yearly/patents.parquet (id=ticker_year) |
| sp500_firm_year_ai_patents_oecd2025.csv | same (CSV duplicate) | none (Q never reads the CSV) | drop, since it duplicates the parquet |
| sp500_firm_preshock_patent_capacity.parquet | gold/external_patents/build_sec_preshock_patents.py | deprecated/run_sec_external_gap_experiment.py only | no live consumer: data/deprecated/processed/, or derive from covariates/yearly/patents |

## 5. DEPRECATED (target: data/deprecated/processed/clusters/<file>)

**Written only by scripts/deprecated/:**
- behavior_block_eval.json (behavior_block_eval.py)
- cluster_diagnostics.json (cluster_diagnostics.py)
- economic_profiles.json (economic_profiles.py)
- firm_behavior_clusters.parquet, firm_clusters_manifest.json, firm_voice_scores.parquet, firm_year_archetype_behaviors.parquet, voice_x_behavior.parquet (build_firm_clusters.py)
- firm_segments.parquet, firm_segments_manifest.json, firm_year_segments.parquet (build_segments.py). **firm_segments.parquet is still read** in `if exists` branches of gold/channel_gap/channel_gap_analysis.py and analytics/shock/shock_analysis.py. Remove those dead reads before moving the file, or the outputs change silently.
- firm_voice_behavior_factors.parquet, voice_behavior_factors.json (voice_behavior_factors.py)
- firm_voice_behavior_grid.parquet, firm_year_voice_behavior_grid.parquet, voice_behavior_grid_manifest.json (build_voice_behavior_grid.py)
- firm_washing_hierarchical.parquet (washing_hierarchical.py)
- segments_no_intensity_robustness.json (segments_no_intensity_robustness.py)
- channel_gap_words_ci.json (channel_gap_words_ci.py)

**No existing writer (orphans):**
- call_beta_main_panel.parquet (older non-as-of call panel)
- call_beta_main_10k10q_table.csv, call_beta_main_results.csv
- call_multi_target_archetypes_results.csv (docstring of call_archetype_full_battery says it is orphaned)
- *_extensive variants from the removed two-margin runs (commit 382d3dd): channel_gap_cells_extensive.parquet, firm_year_extensive.parquet, crosscheck_stats_extensive.json, shock_analysis_extensive.json, shock_did_simple_extensive.json, shock_did_{gov,promo,quant,spec}_per_1k_extensive.png
- shock_did_gov_share.png, shock_did_promotional_rate.png, shock_did_quantified_rate.png, shock_did_specificity_index.png (outcomes from an older OUTCOMES list)
- channel_gap_firm_quarter.parquet
- sec_event_study.json
- firm_washing_score_all.parquet, firm_washing_score_manifest.json
- firm_year_washing_betabinom_panel.parquet, washing_betabinom_by_archetype.parquet, washing_betabinom_by_year.parquet, washing_betabinom_manifest.json
- gbdt_prefilter_importance.json, edgar_coverage_by_year.csv
- test_c1_fig.png, test_c2_fig.png, test_multi_target_fig.png
- **call_crash_risk_panel.parquet** has no writer but live readers (see section 1). **Do not move it** until a builder exists.

---

## Antipattern edges (reader is not gold/consolidate and not thesis.qmd)

### Readers in analytics/
| reader | processed panel | gold replacement |
|---|---|---|
| call_beta/call_beta_regressions | call_beta_main_panel_10k10q_asof | datasets/quarterly/call (**missing accession_number** for attach_leverage; leverage is computed inline from data/raw/xbrl_facts, so it should become a quarterly covariate) |
| call_beta/call_beta_regressions | firm_year_financials_ratios (dead `attach_roa`, never called) | delete the dead code |
| call_beta/call_beta_robustness | call_beta_main_panel_10k10q_asof | datasets/quarterly/call (same accession_number gap; it also recomputes windowed betas from raw prices) |
| call_beta/call_car_regressions | call_beta_main_panel_10k10q_asof | datasets/quarterly/call (accession_number gap) |
| call_beta/call_car_regressions | firm_year_financials_ratios (roa by filing_date) | no gold equivalent: needs an as-of ROA covariate in covariates/quarterly/financial_ratios (gold has roa but not the annual+quarterly event stream) |
| call_beta/call_car_regressions | us_10q_financials_panel (eps_diluted SUE, net_income/assets) | no gold equivalent: needs silver.xbrl_10q_financials plus quarterly covariates (sue, roa_q) |
| call_beta/call_beta_generalized_targets | call_beta_main_panel_10k10q_asof | datasets/quarterly/call |
| call_beta/call_beta_generalized_targets | call_fundamentals_panel | no gold equivalent: needs a new covariate/target family (pre/post fundamentals) |
| call_beta/call_beta_config_decoupling_asof | call_beta_main_panel_10k10q_asof | datasets/quarterly/call |
| call_beta/call_beta_config_decoupling_asof | panel_expanding_archetype_weights_quarterly | predictions/quarterly/posture_archetype_expanding (stale, rebuild) |
| call_beta/call_beta_config_decoupling_asof | call_crash_risk_panel | no gold equivalent: needs a new covariate/target family, and no writer exists |
| call_beta/call_beta_config_decoupling_asof | ncskew_63 | no gold equivalent: needs a new covariate/target family |
| call_beta/sec_comment_letter_cases | document_panel | no gold equivalent: needs a new document-grain family |
| call_beta/sec_comment_letter_cases | firm_year_washing_score | predictions/yearly/washing_score (values differ, rebuild) |
| call_beta/sec_comment_letter_cases | firm_year_master_v2 (market_cap) | covariates/yearly/market or datasets/yearly/firm_year |
| crash_archetypes/call_archetype_full_battery | call_beta_main_panel_10k10q_asof | datasets/quarterly/call |
| crash_archetypes/call_archetype_full_battery | call_crash_risk_panel | no gold equivalent: needs a new covariate/target family (no writer) |
| crash_archetypes/call_archetype_full_battery | call_fundamentals_panel | no gold equivalent: needs a new covariate/target family |
| crash_archetypes/call_archetype_full_battery | panel_expanding_archetypes | predictions/yearly/posture_archetype_expanding |
| shock/incremental_signal | firm_year_master_v2 | datasets/yearly/firm_year (+ predictions/yearly/posture_archetype_static for archetype dummies) |
| shock/incremental_signal | firm_year_activities | covariates/yearly/activities |
| shock/incremental_signal | firm_year_washing_score | predictions/yearly/washing_score |
| shock/shock_analysis | firm_year_washing_score | predictions/yearly/washing_score |
| shock/shock_analysis | firm_segments (optional, deprecated) | remove the read |
| shock/shock_did_simple | firm_year_washing_score | predictions/yearly/washing_score |
| shock/evolution_figures | firm_year_master_v2 | datasets/yearly/firm_year |
| appendix/report_crosscheck_stats | firm_year_master_v2 | datasets/yearly/firm_year |
| appendix/report_crosscheck_stats | firm_year_full_crosscheck | partial: missing revenue/rd_expense/capex/sga_expense levels and ret_m1_p5, so needs a financial_levels covariate and a ret target |
| appendix/report_crosscheck_stats | firm_year_roic_wacc | no gold equivalent: needs a new covariate family (value_creation) |
| washing/build_strategy_economic_profiles | firm_year_master_v2 | datasets/yearly/firm_year |
| washing/build_strategy_economic_profiles | firm_year_roic_wacc | no gold equivalent: needs a new covariate family |
| washing/build_strategy_economic_profiles | firm_year_strategy_dimensions (archetype) | predictions/yearly/posture_archetype_static |
| washing/activity_grounding | channel_activity_cells | no gold equivalent: needs a new dataset (ticker×fy×channel) |
| washing/validate_washing_score (indirect via gold washing_score.build) | firm_year_activities, firm_year_master_v2 | covariates/yearly/activities + datasets/yearly/firm_year |
| washing/firm_year_aggregation_robustness (indirect via incremental_signal.load) | firm_year_master_v2, firm_year_activities, firm_year_washing_score | same as incremental_signal |
| posture/check_archetype_document_channels | firm_strategy_dimensions | no gold equivalent at firm grain: needs predictions/firm/posture_archetype_static |
| posture/plot_archetypal_simplex | firm_strategy_dimensions | same |

### Readers in gold/ outside consolidate (gold topic builders chaining through processed/)
| reader | panel(s) | gold replacement |
|---|---|---|
| call_beta/build_ncskew_63 | call_beta_main_panel_10k10q_asof | spine (internal) |
| call_beta/build_call_beta_panel | firm_activities, firm_year_financials_ratios | none (spine inputs) |
| crash_archetypes/build_call_fundamentals_panel | call_beta_main_panel_10k10q_asof, firm_year_financials_ratios | spine; financial_levels is missing |
| crash_archetypes/build_call_crash_and_archetypes | call_beta_main_panel_10k10q_asof, call_fundamentals_panel, call_crash_risk_panel, firm_year_master_v2, document_panel | datasets/quarterly/call, datasets/yearly/firm_year; the rest have no gold equivalent |
| posture/build_archetype_weights_quarterly_asof | document_panel | no gold equivalent (document grain) |
| posture/activity_profiles | firm_year_master_v2 (n_words), firm_strategy_dimensions (archetype) | covariates/yearly/disclosure_volume; firm-grain archetype missing. **Circular:** build_strategy_dimensions reads activity_profiles' firm_activities and vice versa (Makefile runs strategy_dimensions twice). |
| posture/build_strategy_dimensions | firm_activities | none |
| posture/build_geo_provenance | firm_strategy_dimensions | firm-grain prediction missing |
| washing/build_firm_panels | firm_year_strategy_dimensions, firm_year_financials, firm_year_financials_ratios, firm_year_market_factors, firm_year_filing_returns, firm_strategy_dimensions | these are the spine inputs |
| washing/washing_score | firm_year_activities, firm_year_master_v2 | covariates/yearly/activities, datasets/yearly/firm_year |
| financials/build_market_factors, financials/build_roic_wacc | firm_year_financials_ratios (+ market_factors) | financial sources |
| channel_gap/channel_gap_analysis | firm_washing_score (optional), firm_segments (optional, deprecated) | remove the optional cross-checks, or use predictions |

## thesis.qmd reads that need path updates

**Results, going to data/results/:**
- `clusters/`: activity_grounding.json, bootstrap_jaccard_200_results.json, call_archetype_full_battery_targets.csv, call_beta_generalized_targets.csv, call_beta_regressions.csv, call_beta_regression_samples.csv, call_beta_robustness_{delta_beta,firm_fe,market_model,placebo,windows}.csv, channel_gap_words_robustness.json, incremental_signal.json, washing_score_validation.json
- top level: sec_comment_letter_cases.json
- currently written by gold scripts (need a split first): call_crash_archetype_regressions.csv, call_crash_archetype_regression_samples.csv, activity_profiles.json, strategy_dimensions_manifest.json

**Panels (qmd may stay on processed temporarily):** call_beta_main_panel_10k10q_asof, channel_gap_cells, document_panel, firm_activities, firm_strategy_dimensions, firm_year_market_factors, firm_year_master_v2, firm_year_strategy_dimensions, firm_year_washing_score, panel_expanding_archetypes, sp500_firm_year_ai_patents_oecd2025.parquet.

## Analytics files that write something another script reads (gold data misplaced in analytics)
1. **analytics/shock/incremental_signal.py** writes `models/incremental_signal/<outcome>/model.pkl`. That is read by gold/consolidate/predictions/incremental_signal_coefficients.py, so model fitting lives in analytics.
2. **analytics/shock/shock_analysis.py** writes shock_analysis.json, which analytics/shock/evolution_figures.py reads (analytics→analytics chain).
3. **analytics/call_beta/call_car_regressions.py** writes call_car_panel.parquet (a dataset; no reader today). It also exports `attach_leverage`/`load_price_series` code used by call_beta_regressions/robustness. These compute covariates (leverage, windowed betas, SUE, ROA as-of) inside analytics.
4. **analytics/appendix/earnings_calls_analysis.py** writes earnings_calls_summary.parquet (data, no reader).
5. The reverse direction also occurs: analytics/washing/validate_washing_score.py imports `OUT_DIR`/`build` from gold/washing/washing_score.py, and analytics/posture/bootstrap_archetype_stability.py imports the fit pipeline from gold/posture/build_strategy_dimensions.py.

## Other inconsistencies found
- Makefile l.273 runs `scripts/gold/consolidate/predictions/incremental_signal.py`, which does not exist. The file is `incremental_signal_coefficients.py`.
- gold/consolidate/predictions/washing_score.py and posture_archetype_static.py read `data/gold/covariates/yearly/firm_year.parquet`, which does not exist (only datasets/yearly/firm_year.parquet does). The current gold prediction files come from an older vintage, which explains the washing and quarterly-archetype mismatches.
- The quarterly gold id is `call_accession_number`, not the `ticker_FYQn` scheme in docs/gold_pipeline.md.
- gold/financials/build_us_10q_financials_panel.py writes to a cwd-relative `Path("data/processed/...")`, not REPO_ROOT.
- analytics/call_beta/call_beta_regressions.py keeps a dead ROA_PATH/attach_roa read of firm_year_financials_ratios.
