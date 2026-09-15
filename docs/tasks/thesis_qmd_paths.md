# thesis.qmd source mapping: DuckDB/data-processed → data/results + data/gold

Evidence-based, chunk-by-chunk mapping for the migration in
`docs/tasks/thesis_qmd_migration.md`. Every "new source" below was verified to
exist on disk (`data/results/...`, `data/gold/...`, `models/...`) with its
columns/keys read directly, not inferred from the plan. Where the plan's line
numbers or claims were stale, this document corrects them against the current
`thesis_document/thesis.qmd` (4,985 lines, 82 python chunks).

Counts: **56 READY**, **17 PARTIAL**, **2 MISSING**, **7 STAY INLINE** (pure
formatting/hardcoded, no data dependency).

## Setup-chunk helpers (lines 15-138), load-bearing for every chunk below

- `_p(name)` (119-121): resolves `../data/processed/clusters/<name>` (now
  `data/deprecated/processed_20260915/clusters/<name>`). Used by most chunks.
  No `_pp`/`find_path` is defined in the shared setup — chunks that use those
  names (e.g. 3097, 3960, 3204) **redefine them locally**, each with an
  identical `../<rel>` / `<rel>` fallback body. All of these collapse into the
  planned `results(topic, name)` / `gold(kind, grain, name)` helpers.
- DuckDB connection (82-113): `_shared_duckdb_conn = duckdb.connect(str(_db), read_only=True)`,
  wrapped in `_DuckDBProxy` whose `close()` is a no-op (line 98: `pass`) — so
  every later chunk's `con.close()` (e.g. line 541) does nothing and the
  connection stays live for the whole render. `duckdb.connect` is
  monkeypatched (112-113) so any chunk calling `duckdb.connect(...)` also gets
  the same shared connection. Per the file's own comment (lines 80-81) and
  confirmed by inspection, `firm_universe`/`filing_manifest` views are broken
  since the Chile/Italy removal — DuckDB-dependent chunks do not currently
  render.
- `_cached_read_parquet` (123-136): in-memory cache keyed by resolved path,
  monkeypatches `pd.read_parquet` — keep this pattern when swapping to
  `results()`/`gold()`.
- `PANEL_DOCS` (140-143) is a DuckDB SQL predicate over
  `filing_manifest`/`filing_manifest_10q`, **duplicated verbatim** again at
  529-532.
- `map_sector()` is duplicated **three times** (lines ~427, 1024, 1436) with
  identical logic — collapses to reading `agg_sector`/`sic2` off
  `data/results/corpus/universe_sectors.parquet`.
- The `_us_frames` DuckDB temp table (`CREATE TEMP TABLE _us_frames AS ...
  FROM gold_ai_frames`) is built **twice**, verbatim, at lines 586-590 and
  634-638.

## Summary table

| Chunk (start line) | Label | Old source (kind) | New source | Status |
|---|---|---|---|---|
| 15 | setup (hd_* vars) | DuckDB + `data/processed/clusters/*` (parquet+7 JSON) | `data/results/corpus/headline_counts.json` + `washing/ai_diffusion_by_year.parquet` + `posture/strategy_dimensions_diagnostics.json` + `washing/washing_score_validation.json` + `posture/activity_profiles.json` + `shock/incremental_signal.json` + `washing/activity_grounding.json` + `channel_gap/channel_gap_words_robustness.json` | PARTIAL |
| 338 | uni_/cov_/bal_ vars | DuckDB + `firm_year_master_v2.parquet`, `firm_activities.parquet` | `data/results/corpus/headline_counts.json` (uni_*/cov_* only) | PARTIAL |
| 375 | fig-sic-distribution | DuckDB `firm_universe` | `data/results/corpus/universe_sectors.parquet` | READY |
| 522 | (totals) | DuckDB `paragraphs` | `data/results/prefilter/funnel.json` | READY (field-choice caveat) |
| 544 | tbl-documents-by-form-year | DuckDB `paragraphs` | `data/results/corpus/documents_by_form_year.parquet` | READY |
| 568 | tbl-funnel-summary | DuckDB `read_parquet(interim/...)`, `gold_ai_frames` | `data/results/prefilter/funnel.json` | READY (n_activities discrepancy) |
| 627 | tbl-frame-schema | DuckDB `_us_frames` | `data/results/corpus/frame_schema_distribution.parquet` | PARTIAL |
| 691 | tbl-activity-schema | `data/processed/clusters/firm_activities.parquet` | `data/results/corpus/activity_schema_distribution.parquet` + `data/gold/covariates/document/activities.parquet` (function_family) | PARTIAL |
| 786 | fig-ai-diffusion | `firm_year_master_v2.parquet` | `data/results/washing/ai_diffusion_by_year.parquet` | READY |
| 915 | fig-strategy-dimensions-dist | `firm_strategy_dimensions.parquet` | `data/gold/predictions/firm/posture_archetype_static.parquet` | READY |
| 982 | arch_n_* vars | `firm_strategy_dimensions.parquet` | `data/gold/predictions/firm/posture_archetype_static.parquet` | READY |
| 1003 | fig-archetypal-ternary-simplex | DuckDB `firm_universe` + `firm_strategy_dimensions.parquet` + inline `AA(k=3)` fit | `data/results/posture/representative_firms.parquet` + `models/posture_archetype_static/model.pkl` + `data/gold/covariates/firm_year/posture.parquet` + `data/results/corpus/universe_sectors.parquet` | PARTIAL |
| 1226 | tbl-archetypes-glance | `firm_strategy_dimensions.parquet` | `data/gold/predictions/firm/posture_archetype_static.parquet` | READY |
| 1279 | fig-archetype-heatmap | `firm_strategy_dimensions.parquet` + inline `AA(k=3)` refit | `data/gold/predictions/firm/posture_archetype_static.parquet` (+ refit, or `models/posture_archetype_static/model.pkl`) | PARTIAL |
| 1339 | tbl-representative-firms | DuckDB `firm_universe` + `firm_strategy_dimensions.parquet` | `data/results/posture/representative_firms.parquet` (+ light silver `firm_universe.active_status`) | READY |
| 1407 | fig-sector-archetype-heatmap | DuckDB `firm_universe`/`filing_manifest`/`gold_ai_frames` + `firm_strategy_dimensions.parquet` | `data/results/posture/archetype_sector.parquet` | PARTIAL |
| 1511 | (10-K-only AA refit) | DuckDB `gold_ai_frames` JOIN `filing_manifest` | `silver.ai_frames`/`silver.filing_manifest` via `scripts/gold/posture/posture_features.py::load_frames()` extended with a `form` column | PARTIAL |
| 1596 | (expanding AA refit + composition) | DuckDB `gold_ai_frames`, `document_panel.parquet`, `firm_year_master_v2.parquet` | `data/gold/predictions/firm_year/posture_archetype_expanding.parquet` + `data/results/posture/archetype_composition_annual.parquet` + `archetype_transitions.parquet` | PARTIAL |
| 1719 | fig-archetype-composition-annual | (same as 1596) | `data/results/posture/archetype_composition_annual.parquet` + `archetype_transitions.parquet` | READY |
| 1798 | fig-archetype-stability | `docs/analytics/bootstrap_jaccard_200_results.json` | `data/results/posture/bootstrap_jaccard_200_results.json` | READY |
| 1879 | tbl-4-purpose-defs | none (hardcoded) | n/a | STAY INLINE |
| 1899 | tbl-4-action-defs | none (hardcoded) | n/a | STAY INLINE |
| 1915 | (sector purpose/action setup) | DuckDB `firm_universe` + `firm_activities.parquet` | `data/results/posture/sector_purpose_action.parquet` | READY |
| 2012 | fig-sector-purpose-action | (same as 1915) | `data/results/posture/sector_purpose_action.parquet` | READY |
| 2076 | fig-archetype-activity-heatmap | `firm_strategy_dimensions.parquet` + `firm_activities.parquet` | `data/results/posture/archetype_activity.parquet` | READY |
| 2172 | tbl-4-1b-volume-adjusted | `strategy_dimensions_manifest.json` | `data/results/posture/strategy_dimensions_diagnostics.json` | READY |
| 2210 | fig-archetype-activity-mix | `firm_strategy_dimensions.parquet` + `firm_activities.parquet` | `data/results/posture/archetype_domain.parquet` | READY |
| 2287 | (dead `_mix()` helper) | n/a | n/a (delete) | READY (delete) |
| 2301 | tbl-archetype-activity-examples | `act` df + live DuckDB `paragraphs` lookup | `data/results/posture/activity_examples.parquet` | READY |
| 2363 | fig-ladder-substance | `data/processed/clusters/firm_activities.parquet` | none found | MISSING |
| 2449 | fig-volume-vs-grounding | `firm_activities.parquet` + DuckDB `filing_manifest`/`filing_manifest_10q` | `data/results/washing/volume_vs_grounding.parquet` | READY |
| 2568 | (tech provenance stats) | `firm_activities.parquet` | `data/results/washing/tech_provenance.parquet` | PARTIAL |
| 2616 | fig-tech-provenance | (vars from 2568) | (same) | PARTIAL |
| 2658 | tbl-credibility-designs | none (hardcoded) | n/a | STAY INLINE |
| 2684 | (washing intro stats) | `firm_year_washing_score.parquet` + `washing_score_validation.json` | `data/gold/predictions/firm_year/washing_score.parquet` + `data/results/washing/washing_score_validation.json` | READY |
| 2704 | fig-ds-alignment-scatter | `firm_year_washing_score.parquet` + `firm_year_master_v2.parquet` | `data/gold/predictions/firm_year/washing_score.parquet` + `data/gold/predictions/firm_year/posture_archetype_static.parquet` | READY |
| 2755 | fig-washing-distribution | `firm_year_washing_score.parquet` + `washing_score_validation.json` | (same gold/results pair as 2684) | READY |
| 2840 | (derived washing stats) | (in-memory, from 2755) | n/a (contingent) | READY |
| 2860 | (activity_grounding stats) | `activity_grounding.json` | `data/results/washing/activity_grounding.json` | READY |
| 2880 | fig-venue-call-filing-differences | `activity_grounding.json` | `data/results/washing/activity_grounding.json` | READY |
| 2980 | (diff labels 1) | (vars from 2880) | n/a (contingent) | READY |
| 2997 | (diff labels 2) | (vars from 2880) | n/a (contingent) | READY |
| 3026 | (channel-gap words setup) | `channel_gap_words_robustness.json` | `data/results/channel_gap/channel_gap_words_robustness.json` | READY |
| 3044 | fig-promotion-and-evidence | (vars from 3026) | (same) | READY |
| 3097 | (SEC comment letter setup) | `data/processed/sec_comment_letter_cases.json` | `data/results/call_beta/sec_comment_letter_cases.json` | READY |
| 3133 | fig-sec-welltower | (vars from 3097) | (same) | READY |
| 3204 | fig-call-beta-baseline | `call_beta_regressions.csv` | `data/results/call_beta/call_beta_regressions.csv` | READY |
| 3259 | (call-beta sample stats) | `call_beta_regression_samples.csv` | `data/results/call_beta/call_beta_regression_samples.csv` | READY |
| 3286 | fig-economic-coefficients-c1 | `call_beta_generalized_targets.csv` + `call_beta_regressions.csv` | `data/results/call_beta/call_beta_generalized_targets.csv` + `call_beta_regressions.csv` | READY |
| 3349 | (robustness setup) | 6 `call_beta_robustness_*.csv` | `data/results/call_beta/call_beta_robustness_{market_model,windows,firm_fe,delta_beta,placebo}.csv` | READY (4 dead vars) |
| 3389 | fig-economic-coefficients-c2 | `call_archetype_full_battery_targets.csv` | `data/results/crash_archetypes/call_archetype_full_battery_targets.csv` | READY |
| 3467 | fig-call-crash-risk | `call_crash_archetype_regressions.csv` (+samples) | `data/results/crash_archetypes/call_crash_archetype_regressions.csv` (+ `_regression_samples.csv`) | READY |
| 3542 | fig-nlp-horse-race | `incremental_signal.json` | `data/results/shock/incremental_signal.json` | READY |
| 3636 | tbl-firm-comparisons | `firm_year_washing_score.parquet` + `firm_year_strategy_dimensions.parquet` + `firm_activities.parquet` | `data/gold/predictions/firm_year/washing_score.parquet` + `.../posture_archetype_expanding.parquet` + `data/gold/covariates/firm_year/activities_by_channel.parquet` | PARTIAL |
| 3727 | (2026 screen setup) | `firm_year_washing_score.parquet` + `firm_year_master_v2.parquet` + `channel_gap_cells.parquet` | `data/gold/predictions/firm_year/washing_score.parquet` + `data/gold/spines/firm_year/firm_year.parquet` + `data/gold/covariates/firm_year/channel_gap_cells_extensive.parquet` | PARTIAL |
| 3780 | (expanding-archetype quadrant) | `panel_expanding_archetypes.parquet` | `data/gold/predictions/firm_year/posture_archetype_expanding.parquet` | PARTIAL (producer hygiene) |
| 3801 | tbl-archetype-quadrant | (vars from 3780) | (same) | READY |
| 3891 | tbl-a0-sic-sector-mapping | DuckDB `firm_universe` | `data/results/corpus/universe_sectors.parquet` | READY |
| 3960 | (document_panel setup) | `_pp("document_panel.parquet")` | `data/gold/covariates/document/document_panel.parquet` | PARTIAL |
| 3999 | tbl-a0-derived-inventory | `firm_activities.parquet`, `firm_year_washing_score.parquet`, `call_beta_main_panel_10k10q_asof.parquet` | `data/gold/covariates/document/activities.parquet`, `data/gold/predictions/firm_year/washing_score.parquet`, `data/gold/datasets/call/call.parquet` | PARTIAL |
| 4035 | tbl-a0-judge-selection | none (hardcoded) | n/a | STAY INLINE |
| 4071 | tbl-a1-funnel-full | DuckDB `paragraphs` + interim globs + `_us_frames` | `data/results/prefilter/funnel.json` (+ `funnel_by_form.csv`) | READY |
| 4133 | tbl-b1-patents-controls | `sp500_firm_year_ai_patents_oecd2025.parquet` + `firm_year_master_v2.parquet` + inline `smf.ols` | `data/results/appendix/patents_controls.csv` | READY |
| 4211 | (k=2 AA out-of-sample fit) | `firm_strategy_dimensions.parquet` + inline `AA(k=2)` fit | none found | MISSING |
| 4257 | (OOS temporal validation) | `firm_year_strategy_dimensions.parquet` + inline k-means/ARI | `data/results/posture/oos_validation.json` | READY |
| 4331 | tbl-a3-posture-correlation | `strategy_dimensions_manifest.json` | `data/results/posture/strategy_dimensions_diagnostics.json` | READY |
| 4358 | tbl-a2-six-dimensions | `activity_profiles.json` | `data/results/posture/activity_profiles.json` | READY |
| 4407 | tbl-vendor-ecosystem | `activity_profiles.json` | `data/results/posture/activity_profiles.json` | READY |
| 4480 | (beta diagnostics setup) | `firm_year_market_factors.parquet` | `data/gold/covariates/firm_year/market_raw.parquet` | READY |
| 4506 | (universe delisted count) | DuckDB `firm_universe` | `data/silver/firm_universe.parquet` (light silver read) | READY |
| 4540 | tbl-c1-m0m3 | `incremental_signal.json` | `data/results/shock/incremental_signal.json` | READY |
| 4572 | tbl-e-10k-refit | (vars from chunk 1511) | n/a (contingent on 1511) | STAY INLINE (contingent) |
| 4597 | (washing validation setup) | `washing_score_validation.json`, `firm_year_master_v2.parquet`, `firm_strategy_dimensions.parquet` (len only) | `data/results/washing/washing_score_validation.json` + `data/gold/spines/firm_year/firm_year.parquet` | PARTIAL |
| 4633 | tbl-c4-washing-validation | `washing_score_validation.json` | `data/results/washing/washing_score_validation.json` | READY |
| 4676 | tbl-call-beta-paper | 6 `call_beta_*.csv` | `data/results/call_beta/*.csv` | READY |
| 4760 | tbl-e-archetype-regressions | `call_archetype_full_battery_targets.csv` | `data/results/crash_archetypes/call_archetype_full_battery_targets.csv` | READY |
| 4827 | tbl-g1-sec-letters | `sec_cases` (from 3097) | `data/results/call_beta/sec_comment_letter_cases.json` | READY |
| 4845 | tbl-g2-sec-cases | `sec_cases["cases"]` (from 3097) | (same) | READY |
| 4878 | (Appendix G export helper) | none — writes `docs/analytics/` | n/a | STAY INLINE (needs rework) |
| 4901 | tbl-h5-cohort-grounding | `firm_year_washing_score.parquet` | `data/results/washing/appendix_g_cohort_grounding.csv` (already produced) | READY |
| 4937 | tbl-h8-rollup-sensitivity | `firm_activities.parquet` | `data/results/washing/appendix_g_rollup_sensitivity.csv` (already produced) | READY |
| 4964 | (slide-manifest writer) | none — writes `docs/analytics/` | n/a | STAY INLINE (needs rework) |

---

## Per-chunk detail

### Setup and corpus (lines 15-1003)

**L15 setup — headline `hd_*` vars.** Reads (all via `_c.execute`/`_p`/`_cl/...`):
`con.execute` on `paragraphs`+`firm_universe` (146-149); `_p("firm_activities.parquet")`
(150); `_cl/"firm_year_master_v2.parquet"` (160); `_cl/"document_panel.parquet"`
(170, 192); `con.execute` 10-K/`gold_ai_frames` join (197-206); JSON reads of
`channel_gap_words_robustness.json` (208), `strategy_dimensions_manifest.json`
(211), `washing_score_validation.json` (220), `activity_profiles.json` (222),
`incremental_signal.json` (227), `activity_grounding.json` (234) — all under
`data/processed/clusters/`.
Vars used downstream (prose, later chapters): `hd_n_docs`, `hd_paragraphs_m`,
`hd_n_activities`, `hd_n_activity_firms`, `hd_share`, `hd_frames`,
`hd_frames_ratio(_26)`, `hd_cutoff_str`, `hd_int_ratio_est`, `hd_corr`,
`hd_stab(3)`, `hd_pca_ev`, `hd_vs_ratio`, `hd_gl_ratio`, `hd_vol_r2`,
`hd_w_rho`, `hd_quant_pooled`, `hd_quant_filings`, `hd_obj_unspec_firms`,
`hd_src_unspec`, `hd_inc_*`, `hd_gap_*`, `hd_promo_ratio`, `hd_quant_ratio`,
`hd_tenk_share`.
- READY (path-only swap, keys verified identical): `hd_n_docs/hd_n_paragraphs/
  hd_n_universe/hd_paragraphs_m` → `data/results/corpus/headline_counts.json`;
  `hd_share/hd_frames/hd_frames_ratio(_26)` → `data/results/washing/
  ai_diffusion_by_year.parquet` (`frames_per_1k`→`frames`, pre-aggregated by
  year, no groupby needed); `hd_corr/hd_stab(3)/hd_pca_ev/hd_vs_ratio/
  hd_gl_ratio/hd_vol_r2` → `data/results/posture/
  strategy_dimensions_diagnostics.json` (verified: `correlation_matrix`,
  `pca_diagnostic.explained_variance_ratio`, `stability_by_k`,
  `cluster_sizes`, `activity_volume_regression.{volume_adjusted,r2_adjusted}`
  all present, identical structure); `hd_w_rho` → `data/results/washing/
  washing_score_validation.json["persistence"]["spearman"]`;
  `hd_quant_pooled/hd_quant_filings/hd_obj_unspec_firms/hd_src_unspec` →
  `data/results/posture/activity_profiles.json` (`top_behaviours`,
  `top_behaviours_filings_only`, `object_families`, `distributions.source`,
  all confirmed present); `hd_inc_*` → `data/results/shock/
  incremental_signal.json["outcomes"]["beta"]` (identical nested structure
  down to `wald_washing.F`/`.p`); `hd_gap_quant_pp/hd_gap_named_pp` →
  `data/results/washing/activity_grounding.json["channel_gap"]["pooled"]`
  (verified `pooled` sub-key exists, same as old).
- PARTIAL/MISSING: `hd_n_activities`/`hd_n_activity_firms` are READY via
  `data/gold/covariates/document/activities.parquet` (columns `ticker,
  text_hash, activity_id` match exactly) **but the count disagrees across
  three sources** — see "n_activities discrepancy" below; pick one and
  reconcile. `hd_cutoff*, hd_int_factor, hd_share_factor, hd_int_est_2026,
  hd_share_est_2026, hd_int_ratio_est, hd_act_factor` (the seasonal
  year-to-date adjustment, lines 167-195) have no persisted producer — the
  row-level inputs exist (`data/gold/covariates/document/document_panel.parquet`),
  but no analytics script computes the factors. **MISSING**: new
  `scripts/analytics/corpus/seasonal_adjustment.py` (or extend
  `headline_counts.py`) → `results/corpus/seasonal_adjustment.json`.
  `hd_tenk_share` (10-K-only AI share by year, 197-207) has no producer.
  **MISSING**: new `scripts/analytics/corpus/tenk_ai_share.py` reading
  `silver.filing_manifest`(10-K)+`silver.ai_frames` → `results/corpus/
  tenk_ai_share_by_year.parquet`.
- Producers: `scripts/analytics/corpus/headline_counts.py` (commit `ccc45fb`).
- STATUS: **PARTIAL**.

**L338 — `uni_*`/`cov_*`/`bal_*` vars.** `con.execute` on `firm_universe`
(341); `filing_manifest`/`filing_manifest_10q` (354-360); `_p("firm_year_master_v2.parquet")`
(345); `_p("firm_activities.parquet")` (351).
- READY: `uni_frame_total, uni_frame_delisted, cov_last_date, cov_10k_2026,
  cov_10k_2025, cov_10q_2026, cov_calls_2026, cov_months_2026` — all present,
  verified identical values, in `data/results/corpus/headline_counts.json`.
- MISSING producer: `uni_n_firms, uni_n_activity_firms, bal_n, bal_share` —
  need a small extension to `headline_counts.py` or a new
  `scripts/analytics/corpus/panel_coverage.py` reading `data/gold/datasets/
  firm_year/firm_year.parquet` + `data/gold/covariates/document/activities.parquet`.
  (`bal_share` and `cov_months_2026` appear to have no downstream use beyond
  their own definition — check before migrating, candidates for deletion per
  CLAUDE.md rule 7.)
- STATUS: **PARTIAL**.

**L375 `fig-sic-distribution`.** `con.execute("SELECT ticker, company_name,
sic, industry_group, active_status FROM firm_universe...")` (384); pure
pandas/matplotlib after that (`sic_div()`, `map_sector()`, `SIC2_LABELS`).
New source: `data/results/corpus/universe_sectors.parquet` (`ticker,
company_name, sic, sic2, division, agg_sector`, 499 rows). Producer:
`scripts/analytics/corpus/universe_sectors.py`. Panel B's top-12-plus-other
bucketing (402-424) stays as light inline `value_counts()` reshaping.
STATUS: **READY**.

**L522 — `totals`.** `con.execute` on `paragraphs` (534-540); re-declares
`PANEL_DOCS` (duplicate). `totals=(documents, paragraphs, scorable,
unique_texts)` used at line 575 by the funnel chunk. New source:
`data/results/prefilter/funnel.json["funnel"]`. **Caveat**: the funnel splits
`raw_paragraph_instances` (9,708,471) from `total_paragraph_instances`
(9,020,981) — these differ by ~700K rows (likely earnings-call paragraph
handling); confirm which one matches the old single `count(*) FROM paragraphs
WHERE scope` before repointing (this is a reconciliation, not a rename).
`n_scorable`/`n_unique` map cleanly to `scorable_paragraph_instances`
(8,878,761) / `unique_scorable_texts` (5,110,850).
STATUS: **READY** (with the raw-vs-total field caveat).

**L544 `tbl-documents-by-form-year`.** `con.execute` groups `paragraphs` by
`form` only (551-557) — despite the label, no year breakdown exists today.
New source: `data/results/corpus/documents_by_form_year.parquet` (`form,
year, documents`, 31 rows: 6 forms × ~6 years). Producer:
`scripts/analytics/corpus/documents_by_form_year.py`. This is an upgrade: the
new file supports the year breakdown the label implies; recommend pivoting
`form × year` rather than reproducing the old form-only total.
STATUS: **READY**.

**L568 `tbl-funnel-summary`.** `con.execute(read_parquet('data/interim/
prefilter_predictions_unique/...'))` (578-584); `CREATE TEMP TABLE _us_frames
AS ... FROM gold_ai_frames` (586-590, duplicate of the one at 634); aggregate
query (591-598). Vars: `n_docs, n_raw, n_scorable, n_unique, n_prefiltered,
n_framed, n_frames_total, share_firm_subject, n_firm_texts, n_activities`
(also `= hd_n_activities`). New source: `data/results/prefilter/funnel.json
["funnel"]` — **field names are identical to the qmd's own variable names**,
no renames at all: `n_docs=59818, n_raw=9708471, n_scorable=8878761,
n_unique=5110850, n_prefiltered=36909, n_framed=31332, n_frames_total=70353,
share_firm_subject=78.868, n_firm_texts=28027, n_activities=47666`.
STATUS: **READY** (structurally), but flags the n_activities discrepancy (see
below) against `hd_n_activities`/`activity_profiles.json` (47,767).

**L627 `tbl-frame-schema`.** Rebuilds `_us_frames` (634-638, dup of 586);
`_us_frames_dedup` (644-647, dedup key `(text_hash, frame_id, subject)`);
6 `con.execute` aggregate queries (648-671). New source: `data/results/corpus/
frame_schema_distribution.parquet` (long format: `field, value, n, pct`, 50
rows: `subject, ai_type, temporal, concepts, specificity, rhetoric,
sentence_anchors`). Producer: `scripts/analytics/corpus/
frame_schema_distribution.py`, which dedups only on `(text_hash, frame_id)`
— **narrower than the qmd's `(..., subject)` dedup**, a genuine
granularity mismatch worth a numeric spot-check.
- MISSING: the "Functional Concepts" row (line 679: "Adoption/deploy 46.1%,
  Governance/control 18.2%, Risk 22.4%") is currently a **hardcoded string**,
  not computed by SQL. The new parquet reports 18 raw `concepts` tags
  instead of this 3-bucket rollup — needs a new mapping, either added to
  `frame_schema_distribution.py` or as light inline `groupby`/`map`.
STATUS: **PARTIAL**.

**L691 `tbl-activity-schema`.** Reads `data/processed/clusters/
firm_activities.parquet` directly (700-703, bypasses `_p()`). New source:
`data/results/corpus/activity_schema_distribution.parquet` (long format,
37 rows: `action, stage, source, target, evidence_type, domain,
named_entities`). Producer: `scripts/analytics/corpus/
activity_schema_distribution.py`, reading `silver.ai_activities`.
- Gap: the old `function_family` ("Business Domain": Operate/Sell/Build/
  Control/Serve/Enable, a gold-layer regex rollup) has **no equivalent** in
  `silver.ai_activities` (only a coarser 3-value `domain`). It **is** present
  in `data/gold/covariates/document/activities.parquet` (confirmed column) —
  so the Business Domain row should read that gold covariate instead of the
  corpus-level file; every other row is READY from
  `activity_schema_distribution.parquet`.
- "Operational Object" row (728) stays a hardcoded example string.
STATUS: **PARTIAL**.

**L786 `fig-ai-diffusion`.** `data/processed/clusters/firm_year_master_v2.parquet`
(796-800), `groupby("year").agg(any_ai=mean, frames=frames_per_1k mean)`. New
source: `data/results/washing/ai_diffusion_by_year.parquet` (`year, any_ai,
frames`, 6 rows, pre-aggregated — values verified: 2021 any_ai=0.3414,
frames=0.0168; 2025 any_ai=0.9347, frames=0.1236; 2026 any_ai=0.9528,
frames=0.1975). No groupby needed once repointed — becomes pure plotting.
STATUS: **READY**.

**L915 `fig-strategy-dimensions-dist`.** `_p("firm_strategy_dimensions.parquet")`
(927), 8 `FEATURES` columns, pure box/whisker plot. New source:
`data/gold/predictions/firm/posture_archetype_static.parquet` — **verified
exact match**: 498 rows, columns `ticker, n_frames, promotional_posture,
hedging_posture, risk_orientation, governance_orientation, temporal_posture,
ai_positioning, specificity, disclosure_intensity, cluster, archetype,
archetype_stability` — all 8 `FEATURES` present under identical names, same
grain (one row per firm), same population size as the old cross-sectional
file. (An earlier pass flagged this MISSING by comparing against the
*expanding* firm_year predictions filtered to 2026 YTD, which has a different,
partial-year population — the *static* firm-grain prediction file is the
correct, already-existing match.)
STATUS: **READY**.

**L982 — `arch_n_universe`/`arch_n_noai`/`arch_n_fit`.** Same
`firm_strategy_dimensions.parquet` re-read. New source: same
`data/gold/predictions/firm/posture_archetype_static.parquet` — verified
archetype counts `{Defensive Disclosers: 255, Governance-Led Disclosers: 99,
Vocal Substantives: 96, No AI: 48}`, total 498, matching the old
`arch_n_universe=498, arch_n_noai=48, arch_n_fit=450` exactly (also
cross-checked: 255+99+96=450 matches `representative_firms.parquet`'s row
count).
STATUS: **READY**.

**L1003 `fig-archetypal-ternary-simplex`.** `con.execute` on `firm_universe`
(1021, duplicate `map_sector()` logic); `_p("firm_strategy_dimensions.parquet")`
(1043); **inline model fit**: `AA(n_archetypes=3, random_state=42,
max_iter=500).fit_transform(X_std)` (1049-1050) — a live fit, exactly what
Phase 1 of the migration plan bars.
- READY pieces: sector mapping → `data/results/corpus/universe_sectors.parquet`
  (`agg_sector`/`sic2`, replacing the duplicated DuckDB query); archetype
  labels + exemplar ranking → `data/results/posture/representative_firms.parquet`;
  frozen fit → `models/posture_archetype_static/model.pkl` (confirmed via
  `joblib.load`: dict with `model` (fitted `archetypes.AA`), `mean`, `std`,
  `column_order`, `feature_names`) — `.transform()` against this frozen
  model is inference, not fitting, and is migration-plan-compliant; input
  features come from `data/gold/covariates/firm_year/posture.parquet` or the
  firm-grain `posture_archetype_static.parquet`.
- MISSING: no file currently persists the **continuous per-firm ternary
  weights** (`w_Vocal/w_Gov/w_Def` or barycentric x/y) needed to redraw the
  scatter — only the categorical `archetype` label + scalar
  `distance_to_centroid` are persisted in `representative_firms.parquet`.
  Fix: either persist a small `results/posture/archetype_weights.parquet` via
  a new one-shot analytics script that `.transform()`s the frozen model once,
  or do that single transform call inline (acceptable as inference against an
  already-fit model, not a fit).
STATUS: **PARTIAL**.

### Posture archetypes (lines 1226-2076)

**L1226 `tbl-archetypes-glance`.** `_p("firm_strategy_dimensions.parquet")`
(1235). New source: `data/gold/predictions/firm/posture_archetype_static.parquet`
— identical columns (`archetype`, 8 posture dims, `archetype_stability`).
Producer: `scripts/gold/posture/build_strategy_dimensions.py` (fit) + gold
consolidation. STATUS: **READY**.

**L1279 `fig-archetype-heatmap`.** Same source parquet, but **refits
`AA(n_archetypes=3, random_state=42, max_iter=500)` inline** (1305-1315) —
the matrix `A` itself (archetype coordinates) is a fit artifact, not
persisted anywhere. Two options: (a) keep the deterministic refit (matches
the established pattern in sibling scripts, e.g.
`strategy_dimensions_diagnostics.py`, several of which also refit rather than
unpickle), or (b) load `models/posture_archetype_static/model.pkl` and pull
its fitted coordinates directly. Either is workable — flagged as a design
choice, not a hard blocker.
STATUS: **PARTIAL**.

**L1339 `tbl-representative-firms`.** `con.execute` on `firm_universe`
(1346, `fu`/`fu_delisted`); `_p("firm_strategy_dimensions.parquet")` (1350);
in-chunk z-scored centroid distance (1356-1369). New source:
`data/results/posture/representative_firms.parquet` (450 rows: `ticker,
company_name, archetype, rank, distance_to_centroid`, already ranked)
**directly replaces** the whole centroid-distance computation — filter
`rank<=4` per archetype. `fu_delisted`'s `active_status` count still needs a
light `silver.firm_universe` read (not a heavy computation, and not
`data/processed`) since `universe_sectors.parquet` doesn't carry
`active_status`. Producer: `scripts/analytics/posture/representative_firms.py`.
STATUS: **READY**.

**L1407 `fig-sector-archetype-heatmap`.** `con.execute` on `firm_universe`
(1417), `firm_strategy_dimensions.parquet` (1418, 1433), `con.execute` on
`filing_manifest`/DEF 14A (1420), `con.execute` on `gold_ai_frames` JOIN
`filing_manifest` WHERE DEF 14A (1423-1427). New source for the heatmap
itself: `data/results/posture/archetype_sector.parquet` (`sector, archetype,
n_firms, pct_of_archetype, baseline_share, relative_representation`, 44
rows) — direct replacement for the `ct`/`baseline`/`rel_rep` crosstab, same
formula, same SIC→sector mapping. Producer: `scripts/analytics/posture/
archetype_sector.py`.
- MISSING: the DEF 14A institutional-channel coverage counts
  (`_ch_n_disclosing, _ch_n_coverage_pct, _ch_n_mention, _ch_n_omit`, used in
  prose 1503-1509) have **no equivalent** — verified directly by reading
  `data/results/posture/archetype_document_channels.json`: its keys are only
  `correlation_matrix, rates_by_form, rates_10k_by_section` (correlations and
  per-form frame rates), none of which are these specific coverage counts.
  Fix: extend `scripts/analytics/posture/check_archetype_document_channels.py`
  to also emit these counts, or add a small standalone `filing_manifest`/
  `gold_ai_frames`-equivalent silver query.
STATUS: **PARTIAL**.

**L1511 (10-K-only AA refit, unlabeled).** `con.execute` on `gold_ai_frames`
JOIN `filing_manifest` WHERE `form_type='10-K'` (1548-1559); refits
`AA(k=3)` on the 10-K-only subset. Dependency on `W_named`/`A`/`SIMPLEX_FEATS`/
`def_col` from chunk 1003. No gold covariate persists frame-level posture
features (`concepts, temporal, specificity, rhetoric`) filterable by form —
only `data/gold/covariates/document/activities.parquet` (activity-level, not
frame-level) exists at document grain. However,
`scripts/gold/posture/posture_features.py::load_frames()` already reads
`silver.ai_frames` joined to `silver.filing_manifest` (which has `form`) —
it just doesn't currently `select("form")`. A **light silver read** (per the
task's own allowance for "bronze/silver via scripts/common/layers.py where a
light read is acceptable") extending this function with a `form` column and
filter is the fix, not a new heavy pipeline.
STATUS: **PARTIAL** (small script extension, not a new pipeline).

**L1596 (expanding-window annual AA refit + composition, unlabeled) and
L1719 `fig-archetype-composition-annual`.** `con.execute` on `gold_ai_frames`
JOIN `filing_manifest` (1613-1625, no form filter); `_p("document_panel.parquet")`
(1655); front-matter `_m` (= `firm_year_master_v2.parquet`, line 160) used at
1686 for `_cross`/`_CROSS_FEATS_EXP`; walk-forward `AA(k=3)` refit per cutoff
year (1661-1692). **This is the single cleanest case**: `data/gold/
predictions/firm_year/posture_archetype_expanding.parquet` (2,893 rows:
`id, ticker, year, arch_exp`) is exactly the walk-forward panel this chunk
hand-computes, produced by `models/posture_archetype_expanding_yearly/
cutoff={2021..2026}/model.pkl`. `data/results/posture/
archetype_composition_annual.parquet` (24 rows: `year, archetype, n_firms,
share_pct`) and `archetype_transitions.parquet` (80 rows: `from_year, to_year,
origin, dest, n, pct_of_origin, persist_pct`) precompute `_share_all`/`_ct_yr`
and `_trans_pct`/`_persist` respectively — confirmed by reading
`scripts/analytics/posture/archetype_composition_annual.py` in full: reads
exactly the expanding-predictions file, renames `arch_exp`→`archetype`, same
`persist_pct` formula.
- PARTIAL: `_cross`/`_CROSS_FEATS_EXP` (the posture-feature columns used to
  fit each cutoff year, drawn from `_m`) need `data/gold/covariates/
  firm_year/posture.parquet` (has the 8 posture features, `id, ticker, year,
  n_frames, promotional_posture, ..., disclosure_intensity`) **joined** to
  `data/gold/predictions/firm_year/posture_archetype_static.parquet` (`id,
  ticker, year, cluster, archetype`) for the `archetype` label — this join
  isn't persisted as one file, but is a straightforward key join, not a
  heavy computation.
- `_row_pct2324`/`_row_n2324` (line 1715) appear dead — `_row_n2324` is never
  referenced again; verify before deleting rather than migrating.
STATUS: **PARTIAL** for L1596 (join needed); **READY** for L1719 (reads the
two precomputed results files directly).

**L1798 `fig-archetype-stability`.** `_p("bootstrap_jaccard_200_results.json")`
with a stale fallback to `docs/analytics/...` (1810-1814). New source:
`data/results/posture/bootstrap_jaccard_200_results.json` — verified
identical schema (`frame_level`/`firm_level` × k=2..5 ×
`mean_by_cluster/se_by_cluster/min/overall_mean`; sample k=3 frame-level
`overall_mean=0.753, min=0.655`). Producer:
`scripts/analytics/posture/bootstrap_archetype_stability.py`. Drop the
`docs/analytics/` fallback entirely.
STATUS: **READY**.

**L1879/L1899 `tbl-4-purpose-defs`/`tbl-4-action-defs`.** Fully hardcoded
definition lists, no data dependency. STATUS: **STAY INLINE**.

**L1915 (sector purpose/action setup) and L2012 `fig-sector-purpose-action`.**
`con.execute` on `firm_universe` (1922); `_p("firm_activities.parquet")`
(1979); in-chunk `get_domain4`/`get_action4` classification + crosstabs. New
source: `data/results/posture/sector_purpose_action.parquet` (long format:
`kind ∈ {purpose_pct, action_deviation_pp}, sector, category, value`, 130
rows) — confirmed identical `get_domain`/`get_action` branching logic in
`scripts/analytics/posture/sector_purpose_action.py`, reading `data/gold/
covariates/document/activities.parquet` (47,767 rows, per-activity grain,
confirmed as the right grain — the row-wise `.apply(get_domain4, axis=1)`
call implies per-activity rows, matching this file, not a per-firm
aggregate). Renames: `role`/`action_type` → `category` (filtered by `kind`).
Note: `act_sec4` (the per-activity dataframe) is *also* reused later at line
2222 (`fig-archetype-activity-mix`) — that chunk needs its own fresh read of
`activities.parquet` with `get_domain`/`get_action` applied, since
`sector_purpose_action.parquet` only has sector-level aggregates.
STATUS: **READY**.

**L2076 `fig-archetype-activity-heatmap`.** `firm_strategy_dimensions.parquet`
(2088) + `firm_activities.parquet` (2089) merged; in-chunk boolean-flag rate
computation (2096-2117). New source: `data/results/posture/
archetype_activity.parquet` (18 rows: `metric, archetype, pooled_rate,
archetype_rate, deviation_pp`) — confirmed via full read of
`scripts/analytics/posture/archetype_activity.py`: identical 5 rate metrics
with byte-identical boolean definitions, plus a 6th `n_activities_median`
row per archetype (note: for that row, `deviation_pp` is semantically a
*ratio*, not a percentage-point difference — same quirk as the qmd's own
code, documented in the script's docstring).
STATUS: **READY**.

**Cross-cutting for posture:** the three "cross-tab-shaped" outputs
(`archetype_composition_annual.parquet`+`archetype_transitions.parquet`,
`sector_purpose_action.parquet`, `archetype_activity.parquet`) are
**distinct files from distinct scripts**, sharing only the common upstream
inputs (`data/gold/covariates/document/activities.parquet` and
`data/gold/predictions/firm/posture_archetype_static.parquet`) — each script
independently re-derives its own domain/action classification by design (per
each script's own docstring).

### Washing / activities (lines 2172-2880)

**L2172 `tbl-4-1b-volume-adjusted`.** `strategy_dimensions_manifest.json
["activity_volume_regression"]` (2182). New source: `data/results/posture/
strategy_dimensions_diagnostics.json["activity_volume_regression"]` —
**verified schema match**: keys `reference, raw, volume_adjusted,
log_frames_coef, log_frames_p, r2_adjusted, n`, with `volume_adjusted`
containing `Governance-Led Disclosers`/`Vocal Substantives` exactly as the
old manifest did. No structural difference beyond the filename.
STATUS: **READY**.

**L2210 `fig-archetype-activity-mix`.** `firm_strategy_dimensions.parquet`
(2222) + `firm_activities.parquet` (2223). New source: `data/results/posture/
archetype_domain.parquet` (`archetype, domain, n_activities,
pct_of_archetype`). Producer: `scripts/analytics/posture/archetype_domain.py`
(docstring explicitly cites this chunk range). Rename: `role`→`domain`; old
per-ticker archetype join → `predictions/firm/posture_archetype_static.parquet`
(same 498-row grain).
STATUS: **READY**.

**L2287 (dead `_mix()` helper).** `_mix(a, role)` has zero call sites
anywhere in the file (grep-confirmed) — dead code, safe to delete rather than
migrate. STATUS: **READY** (delete).

**L2301 `tbl-archetype-activity-examples`.** Uses `act` from L2210 plus a
**live DuckDB lookup**: `get_db().execute("SELECT paragraph_text FROM
paragraphs WHERE accession_number=? AND paragraph_index=? LIMIT 1", ...)`
(2315-2316) — the clearest remaining DuckDB call in this section. New
source: `data/results/posture/activity_examples.parquet` (`ticker,
accession_number, item_key, paragraph_index, text_hash, archetype, domain,
action, object, evidence_type, quote, diagnostic`) — the paragraph text is
already resolved at build time by `scripts/analytics/posture/
activity_examples.py`, eliminating the runtime SQL lookup entirely.
STATUS: **READY**.

**L2363 `fig-ladder-substance`.** Hardcoded path (bypasses `_p()`):
`Path("../data/processed/clusters/firm_activities.parquet")` (2373-2375);
computes 5 nested substance-ladder counts + 5 independent rates
(`c1..c5, p_func, p_stage, p_named, p_metric, p_vendor`), used in prose
right after (line 2441). **No producer exists** — confirmed via `find
data/results/washing -type f` (no ladder/nested-concreteness file) and
`grep -rl "ladder" scripts/analytics/washing scripts/gold/washing` (empty).
By analogy with sibling scripts `volume_vs_grounding.py`/`tech_provenance.py`
(both read `data/gold/covariates/document/activities.parquet`): needs a new
`scripts/analytics/washing/ladder_substance.py` computing the same 5 nested
counts + 5 rates → `data/results/washing/ladder_substance.parquet` (or
`.json`).
STATUS: **MISSING**.

**L2449 `fig-volume-vs-grounding`.** `_p("firm_activities.parquet")` (2459)
+ live `con.execute` UNION ALL on `filing_manifest`/`filing_manifest_10q`
(2460-2467). New source: `data/results/washing/volume_vs_grounding.parquet`
(6 rows, years 2021-2026: `year, n_activities, deployed_share_pct,
generic_evidence_pct, named_evidence_pct, identified_function_pct`).
Producer: `scripts/analytics/washing/volume_vs_grounding.py`, replicating the
SQL via `L.scan("silver.filing_manifest"/"silver.filing_manifest_10q")`
(polars, no DuckDB) — note the new script joins on
`["country_code","accession_number"]` vs the old SQL's `accession_number`
only; minor behavioral-parity risk worth a spot-check. Renames: `vol`→
`n_activities`, `dep_share`→`deployed_share_pct`, `gen_ev`→
`generic_evidence_pct`, `named_prod`→`named_evidence_pct`, `ident_func`→
`identified_function_pct`. The 2026 same-window projection arithmetic
(`annual_factor`, `vol_2026_proj_*`) is **not** persisted — stays inline,
reading the header's `hd_act_factor` (itself flagged MISSING above, in L15).
STATUS: **READY** (projection math stays inline, contingent on `hd_act_factor`).

**L2568 (tech provenance stats) and L2616 `fig-tech-provenance`.**
`_p("firm_activities.parquet")` (2576). New source: `data/results/washing/
tech_provenance.parquet` (1 row: `n_activities_total, own_brand_pct,
ext_vendor_pct, partner_pct, ext_provider_firm_pct, openai_microsoft_pct,
alphabet_google_pct, nvidia_pct, aws_pct`). Producer: `scripts/analytics/
washing/tech_provenance.py`.
- MISSING field: `third_party_src` (`source=="third_party"` mean) — one of
  the four bars in the downstream figure — has **no column** in
  `tech_provenance.parquet` (verified: the 9 columns above are the complete
  set). Fix: one-line addition to `tech_provenance.py`:
  `row["third_party_pct"] = float((df["source"] == "third_party").mean() * 100)`.
STATUS: **PARTIAL** (both chunks, same missing field).

**L2658 `tbl-credibility-designs`.** Fully hardcoded prose table. STATUS:
**STAY INLINE**.

**L2684 (washing intro stats), L2704 `fig-ds-alignment-scatter`, L2755
`fig-washing-distribution`, L2840 (derived stats).** `_p("firm_year_washing_score.parquet")`
+ `_p("washing_score_validation.json")` + (2704 only) `_p("firm_year_master_v2.parquet")`.
New sources: `data/gold/predictions/firm_year/washing_score.parquet` (1,885
rows: `id, ticker, year, frames_per_1k, any_ai, n_promo, n_frames,
n_activities, grounding_index, substance, pct_disclosure, pct_substance, w,
washing, callada`) and `data/results/washing/washing_score_validation.json`
(`n_panel, n_firms, mechanical_consistency_vs_grounding, persistence,
economic_coherence, split_half` — **identical schema** to the old file).
For 2704's archetype join, use `data/gold/predictions/firm_year/
posture_archetype_static.parquet` (`id, ticker, year, cluster, archetype`) in
place of `firm_year_master_v2.parquet`. Producers: `scripts/gold/washing/
washing_score.py` (score/panel fit only, confirmed via commit `27adefc` — no
`data/processed` writes) and `scripts/analytics/washing/
validate_washing_score.py` (validation battery, confirmed **already moved
out of gold into analytics**, matching the migration plan's Phase 4 claim).
STATUS: **READY** (all four chunks).

**L2860 (activity_grounding stats) and L2880
`fig-venue-call-filing-differences`.** `_p("activity_grounding.json")`. New
source: `data/results/washing/activity_grounding.json`, produced by
`scripts/analytics/washing/activity_grounding.py` (reads `data/gold/
covariates/firm_year/activities_by_channel.parquet`). Verified
`channel_gap.n_cells`/`n_firms` **identical** between old and new (796
cells / 289 firms), and all 9-11 family keys present in `channel_gap.pooled`
with identical `{call, filing, gap_pp, t, p}` shape.
STATUS: **READY** (both chunks).

### n_activities discrepancy (flagged for resolution)

Three different counts of the same underlying quantity disagree — confirmed
by direct row counts, not inference:

| Source | Count |
|---|---|
| `data/deprecated/processed_20260915/clusters/firm_activities.parquet` (old, frozen) | **47,784** |
| `data/gold/covariates/document/activities.parquet` (new gold, live) | **47,767** |
| `data/results/prefilter/funnel.json` → `n_activities`/`activities_total` | **47,666** |

- Old (47,784) vs. new-gold (47,767): a 17-row difference, already documented
  in `scripts/analytics/washing/firm_comparisons.py`'s own docstring as
  expected drift from `silver.ai_activities` regenerating since the old
  snapshot was frozen.
- New-gold (47,767) vs. funnel (47,666): a **101-row gap that is not
  explained anywhere** in any script docstring found. `data/results/posture/
  activity_profiles.json` independently reports `n_activities=47767`,
  agreeing with the gold covariate, not the funnel — suggesting
  `funnel.py`'s in-panel scope predicate is stricter than
  `build_firm_activities.py`'s. **Action needed**: reconcile the two
  scripts' filtering logic and settle on one number for `hd_n_activities`,
  `tbl-funnel-summary`, and `tbl-activity-schema` before finalizing prose.

### Market consequences (lines 2980-3801)

Local helper note: `find_path(rel)` is redefined identically three times
(3214, 3357, 3477); `_pp(rel)` twice (3102, and again near 3960 out of this
section) — same body each time (`../<rel>` else `<rel>`), all collapse into
one shared `results()`/`gold()` helper.

**L2980/L2997 (diff labels).** Pure formatting of `diffs1/p1`/`diffs2/p2`
from the untouched upstream `fig-venue-call-filing-differences` chunk
(L2880, READY). No independent data read. STATUS: **READY** (contingent).

**L3026 (channel-gap words setup) / L3044 `fig-promotion-and-evidence`.**
`_p("channel_gap_words_robustness.json")` (3033). New source: `data/results/
channel_gap/channel_gap_words_robustness.json` — verified keys `n_cells,
n_firms, promo_per_1k_words{call_mean,filing_mean,ratio,gap},
promo_per_1k_paragraphs{...}, quant_per_1k_words{...},
quant_per_1k_paragraphs{...}, mean_words_per_paragraph{call,filing}` —
**exact structural match**, no renames. Producer: `scripts/analytics/
channel_gap/channel_gap_words_robustness.py`.
STATUS: **READY** (both chunks).

**L3097 (SEC comment letter setup) / L3133 `fig-sec-welltower`.**
`_pp("data/processed/sec_comment_letter_cases.json")` (3105). New source:
`data/results/call_beta/sec_comment_letter_cases.json` — diffed directly
against the deprecated copy: **identical** top-level keys (`search_window,
letters, relevance_counts, cases`), identical `letters[0]` keys, identical
`cases.{ANET,WELL}.{k,year,treated,controls_mean,controls,targeted}` and
inner metric keys (`claim_density, promo_density, quant_density,
spec_density, realized_density, specificity_index, w`). Producer:
`scripts/analytics/call_beta/sec_comment_letter_cases.py` (git `316037c`:
"sec_comment_letter_cases reads gold document_panel"). `sec_n_adjacent`
(3108) is defined but never used again (dead).
STATUS: **READY** (both chunks).

**L3204 `fig-call-beta-baseline`, L3259 (sample stats), L3286
`fig-economic-coefficients-c1`, L3349 (robustness setup), L3389
`fig-economic-coefficients-c2`, L3467 `fig-call-crash-risk`, L3542
`fig-nlp-horse-race`.** All read `find_path("data/processed/clusters/*.csv")`
or `.json`. New sources, all confirmed present with matching columns:
- `data/results/call_beta/call_beta_regressions.csv` (`model, variable,
  label, beta_std, ci95_low, ci95_high, p`), `call_beta_regression_samples.csv`
  (`model, n_calls, n_firms, n_fe_cells, r2`), `call_beta_generalized_targets.csv`
  (`target, variable, label, beta_std, ci95_low, ci95_high, p`;
  targets = `gross_margin, log_market_cap, next_revenue_yoy, ps_ratio,
  rd_intensity, roic_minus_wacc`), `call_beta_robustness_{market_model,
  windows, firm_fe, delta_beta, placebo}.csv` (all present, block/variable
  columns matching the qmd's filters). Producers: `scripts/analytics/
  call_beta/call_beta_regressions.py`, `call_beta_generalized_targets.py`,
  `call_beta_robustness.py`.
- `data/results/crash_archetypes/call_archetype_full_battery_targets.csv`
  (`target, label, variable, beta_std, se, p, n_calls, n_firms, r2,
  partial_r2_ai_arch` — no `ci95_low/high`, computed inline from `se` at
  3401-3402, which is fine as-is) and `call_crash_archetype_regressions.csv`
  (+`_regression_samples.csv`). Producers: `scripts/analytics/
  crash_archetypes/call_archetype_full_battery.py`, `call_crash_regressions.py`.
- `data/results/shock/incremental_signal.json` — `outcomes.{beta,
  volatilidad_idiosincratica, price_to_sales, rd_sobre_ventas,
  crecimiento_ingresos_t1}`, each with `r2_M0..r2_M2AB, delta_r2_washing`
  etc. — exact match. Producer: `scripts/analytics/shock/incremental_signal.py`.
- In L3349, 4 of the 6 defined robustness vars (`rob_acct, rob_ff3,
  rob_win252, rob_fe, rob_delta`) are never referenced again anywhere in the
  file (dead code, likely leftover from a prior draft) — candidates for
  deletion under CLAUDE.md rule 7; `rob_placebo_d` **is** used downstream.
STATUS: **READY** (all 7 chunks; L3349 flagged with 4 dead vars, not
blocking).

**L3636 `tbl-firm-comparisons`.** `find_path("firm_year_washing_score.parquet")`
(3649), `firm_year_strategy_dimensions.parquet` (3652, `archetype` column
only), `firm_activities.parquet` (3653).
- READY: washing score → `data/gold/predictions/firm_year/washing_score.parquet`
  (byte-for-byte column match plus an added, unused `id`).
- PARTIAL: archetype label → `data/gold/predictions/firm_year/
  posture_archetype_expanding.parquet` (`id, ticker, year, arch_exp`,
  content-equivalent to the old `panel_expanding_archetypes.parquet`, rename
  `archetype`→`arch_exp`) — **but its producer script is stale**: the file on
  disk was built by `scripts/deprecated/posture_archetype_expanding_firm_year.py`
  (confirmed present under `scripts/deprecated/`), whose docstring still
  points at a non-existent `scripts/gold/consolidate/predictions/
  posture_archetype_expanding_firm_year.py` — **only the `_firm_quarter.py`
  sibling exists live** (confirmed: `find scripts -iname
  '*posture_archetype_expanding*'` returns the deprecated firm_year script
  and the live firm_quarter one, nothing else). The parquet is current but
  has no live, non-deprecated builder.
- PARTIAL: activities piece — the old row-grain `firm_activities.parquet`
  needs per-instance rows filterable by `ticker`/`channel`
  (`len(a[a['channel']=='filing'])`). `data/gold/covariates/firm_year/
  activities.parquet` (2,893 rows, aggregated, no channel split) does not
  support this. Two candidates: `data/gold/covariates/document/activities.parquet`
  (47,767 rows, document grain, has `channel`) as a closer drop-in, or
  (recommended) `data/gold/covariates/firm_year/activities_by_channel.parquet`
  (3,237 rows: `id, ticker, fy, channel, n_activities, <family>_counts`) —
  purpose-built at the right grain, avoiding manual filtering.
STATUS: **PARTIAL**.

**L3727 (2026 screen setup).** `find_path("firm_year_washing_score.parquet")`
(washing), `firm_year_master_v2.parquet` (universe/`any_ai` for 2026),
`channel_gap_cells.parquet`.
- READY: washing score (as above); `channel_gap_cells.parquet` → **exact
  column match** at `data/gold/covariates/firm_year/
  channel_gap_cells_extensive.parquet` (29 identical columns, including `t`
  as string year and `gap_promo_per_1k`). Producer: `scripts/gold/
  channel_gap/build_channel_gap_cells.py`.
- RESOLVED (was flagged uncertain by the research pass, checked directly):
  the full-2026-universe ticker count (`_m26["ticker"].nunique()`) should
  come from `data/gold/spines/firm_year/firm_year.parquet` filtered to
  `year==2026` — verified **466** rows/tickers there, vs. only **444** in
  `washing_score.parquet` for 2026 (disclosers-only, not the full universe).
  Use the spine, not `washing_score.parquet`, for the universe count.
STATUS: **PARTIAL** (spine resolves the universe-count piece; verify the
remaining `_m26` column needs — e.g. `any_ai` — against
`data/gold/covariates/firm_year/disclosure_volume.parquet`, which carries
`any_ai` by ticker/year).

**L3780 (expanding-archetype quadrant) / L3801 `tbl-archetype-quadrant`.**
`find_path("panel_expanding_archetypes.parquet")` (3783). New source:
`data/gold/predictions/firm_year/posture_archetype_expanding.parquet` —
verified exact column match (`ticker, year, arch_exp`, `year` already int,
includes 2026). Same stale-producer flag as L3636's archetype piece.
STATUS: **PARTIAL** for L3780 (producer hygiene only — data itself is
correct); **READY** for L3801 (pure table format, contingent on L3780).

### Appendices and tail (lines 3891-4985)

**L3891 `tbl-a0-sic-sector-mapping`.** `con.execute` on `firm_universe`
(3927); local `map_sector()` (duplicate #4). New source: `data/results/corpus/
universe_sectors.parquet` (499 rows, matches "N=499" caption). Producer:
`scripts/analytics/corpus/universe_sectors.py`. Rename: `sector`→`agg_sector`.
STATUS: **READY**.

**L3960 (document_panel setup, feeds `tbl-a0-derived-inventory`).**
`_pp("document_panel.parquet")`; defines `_pp, INV_YEARS, _docs, _cell,
_by_year`. New source: `data/gold/covariates/document/document_panel.parquet`
(`channel, form, ticker, cik, fecha, period_end, call_fy, accession_number,
n_paragraphs, n_words, n_frames, ..., quarter, fye_month, fy`) — no `year`
column, derive via `fecha.dt.year` as before.
STATUS: **PARTIAL** (source exists; local helper needs rewriting to the gold
path).

**L3999 `tbl-a0-derived-inventory`.** `_pp("firm_activities.parquet")`,
`_pp("firm_year_washing_score.parquet")`, `_pp("call_beta_main_panel_10k10q_asof.parquet")`.
New sources: `data/gold/covariates/document/activities.parquet` (matches);
`data/gold/predictions/firm_year/washing_score.parquet` (matches);
`data/gold/datasets/call/call.parquet` (confirmed columns include `ticker,
fecha, beta_post_63d, beta_post_126d, beta_post_252d` — **rename**
`beta_post_63`→`beta_post_63d`). No dedicated producer script exists for this
specific inventory table — it's pure aggregation over already-gold files;
recommend either keeping it as a light in-qmd read/aggregate, or a new
`scripts/analytics/appendix/derived_inventory.py` by analogy with
`patents_controls.py`.
STATUS: **PARTIAL**.

**L4035 `tbl-a0-judge-selection`.** Fully hardcoded `candidates` list, no
data dependency. STATUS: **STAY INLINE**.

**L4071 `tbl-a1-funnel-full`.** `con.execute` on `paragraphs` +
`read_parquet('data/interim/prefilter_predictions_unique/...')` (4080-4089),
`con.execute` on `_us_frames` (4090), `_p("firm_activities.parquet")` (4078).
New source: `data/results/prefilter/funnel.json` — keys
`scorable_paragraph_instances` (8,878,761), `unique_scorable_texts`
(5,110,850), `prefiltered_unique_texts` (36,909), `frames_positive_texts`
(31,332), `activities_total`/`n_activities` (47,666). Producer:
`scripts/analytics/prefilter/funnel.py` (docstring explicitly cites
`tbl-a1-funnel-full`, replacing the removed `PANEL_DOCS` predicate); also
emits `data/results/prefilter/funnel_by_form.csv`.
STATUS: **READY**.

**L4133 `tbl-b1-patents-controls`.** `_pp("sp500_firm_year_ai_patents_oecd2025.parquet")`
(4154), `_p("firm_year_master_v2.parquet")` (4148), `_p("firm_strategy_dimensions.parquet")`
(4153), `_p("firm_activities.parquet")` (4158), inline `smf.ols("log_ai_pat ~
log_n_act + log_mktcap + C(sic2)")` and a grounded-activities variant
(4173-4174). New source: **`data/results/appendix/patents_controls.csv`
already exists** (contradicting the migration plan's claim that this needs a
new script) — columns `regressor, coefficient, se, p, n, n_sic, r2`, exactly
two rows for "Disclosed AI activities (log A)" and "Grounded activities
(log G)" (n=322, n_sic=47). Producer: `scripts/analytics/appendix/
patents_controls.py`, reading `gold/covariates/firm_year/patents.parquet`,
`gold/covariates/firm_year/market.parquet`, `gold/predictions/firm_year/
posture_archetype_static.parquet`, `silver.firm_universe`,
`silver.ai_activities`. The chunk becomes a CSV read + reformat, no re-fit.
STATUS: **READY**.

**L4211 (k=2 out-of-sample archetype fit, unlabeled).**
`_p_k2("firm_strategy_dimensions.parquet")` (4232); live refit
`AA(n_archetypes=2, random_state=42, max_iter=500).fit_transform(...)`
(4239-4241) — genuine inline model-fitting. Vars used downstream:
`k2_promo[0/1], k2_risk[0/1], k2_gov[0/1], k2_spec[0/1]` (prose 4249). No
existing `data/results/posture/*` file carries persisted k=2 AA archetype
coordinates (checked the full `data/results/posture/` listing —
`strategy_dimensions_diagnostics.json`'s `stability_by_k` has k=2 *stability*
numbers, not the coordinate vectors this chunk needs). By analogy with
`scripts/analytics/posture/oos_validation.py` (which explicitly reproduces
another qmd chunk's inline fit "verbatim" into a results file): needs a new
`scripts/analytics/posture/archetype_k2_diagnostic.py`, same seed, writing
`data/results/posture/archetype_k2_diagnostic.json` with per-archetype-vertex
promo/risk/gov/specificity values.
STATUS: **MISSING**.

**L4257 (out-of-sample temporal validation, unlabeled).**
`_p("firm_year_strategy_dimensions.parquet")` (4268); self-contained k-means/
ARI implementation; `_oos()` fit (4316-4318). New source: `data/results/
posture/oos_validation.json` — **keys match exactly**: `oos_ari_2124,
oos_jac_2124, oos_n_2025, oos_ari_2123, oos_jac_2123, oos_ari_2125,
oos_n_2026, oos_jac_min, oos_jac_max, oos_vs_2123` (values: `oos_ari_2124=
0.4795, oos_ari_2123=0.3276, oos_ari_2125=0.7047, oos_n_2025=384,
oos_n_2026=417`). Producer: `scripts/analytics/posture/oos_validation.py`
(docstring explicitly states it reproduces this exact chunk, reading gold
`covariates/firm_year/posture.parquet` + `predictions/firm_year/
posture_archetype_static.parquet`). No renames — variable names already
match JSON keys 1:1.
STATUS: **READY**.

**L4331 `tbl-a3-posture-correlation`.** `_p("strategy_dimensions_manifest.json")
["correlation_matrix"]`. New source: `data/results/posture/
strategy_dimensions_diagnostics.json["correlation_matrix"]` — confirmed
present alongside `pca_diagnostic`, `k`, `stability_by_k`, `cluster_sizes`
(this is the same file used by L15/L2172/L4211's sibling). STATUS: **READY**.

**L4358 `tbl-a2-six-dimensions` / L4407 `tbl-vendor-ecosystem`.**
`data/processed/clusters/activity_profiles.json`
(`distributions`/`function_families` for the six-dimensions table;
`named_providers` for vendor ecosystem). New source (same file for both):
`data/results/posture/activity_profiles.json` — both keys confirmed present.
Producer: `scripts/analytics/posture/activity_profiles.py`.
STATUS: **READY** (both chunks).

**L4480 (beta estimation diagnostics setup).** `_p("firm_year_market_factors.parquet")`
(4487). Vars: `mf_n_cells, mf_n_tickers, mf_year_min, mf_year_max,
mf_beta_n, mf_beta_pct, mf_win_mean, mf_win_median, mf_win_ge240_pct,
mf_n_short, mf_beta_mean, mf_beta_median, mf_beta_sd, mf_beta_q1, mf_beta_q3,
mf_beta_range`. New source: `data/gold/covariates/firm_year/market_raw.parquet`
— confirmed columns include `ticker, year, beta, idio_vol_252d, ..., beta_n_obs`
(note: the non-`_raw` `market.parquet` sibling lacks `beta`/`beta_n_obs` —
must use `market_raw.parquet`). Producer: `scripts/gold/financials/
build_market_factors.py`. No renames needed.
STATUS: **READY**.

**L4506 (universe delisted-count chunk).** `con.execute("SELECT
active_status, count(*) FROM firm_universe WHERE country_code='us' GROUP BY
1")`. New source: `data/silver/firm_universe.parquet` (a light silver read,
not `data/processed`) — confirmed `active_status` values `listed=460,
delisted=39`, total=499. STATUS: **READY**.

**L4540 `tbl-c1-m0m3`.** `data/processed/clusters/incremental_signal.json`
`["outcomes"]["beta"]`. New source: `data/results/shock/incremental_signal.json`
— confirmed `outcomes.beta` contains `r2_M0, adj_r2_M0, ..., r2_M2AB,
adj_r2_M2AB` (all fields the chunk needs, including the deliberate use of
`r2_M2AB` over `r2_M3`, matching the chunk's own comment). Producer:
`scripts/analytics/shock/incremental_signal.py`. STATUS: **READY**.

**L4572 `tbl-e-10k-refit`.** Pure formatting of vars
(`_r_vocal_10k, _r_def_10k, _r_gov_10k, _z_gov10k_on_gov, _z_gov10k_on_risk,
_z_def_full_on_risk, _z_def10k_on_risk, _common10k`) defined at L1511
(upstream, PARTIAL). This chunk itself has no DuckDB/data-processed/model
call of its own. STATUS: **STAY INLINE** (contingent on L1511's resolution).

**L4597 (washing-score-validation setup).** `_p("washing_score_validation.json")`
(4605), `_p("firm_year_master_v2.parquet")` (4607, `len()` only),
`_p("firm_strategy_dimensions.parquet")` (4608, `len()` only). Vars:
`w_n_panel, w_n_firms, w_n_firmyears_total, w_n_firms_total`.
- READY: `w_n_panel`/`w_n_firms` → `data/results/washing/
  washing_score_validation.json` (`n_panel, n_firms` — confirmed via
  `scripts/analytics/washing/validate_washing_score.py` line 102:
  `report = {"n_panel": len(panel), "n_firms": panel["ticker"].nunique()}`).
- PARTIAL: `w_n_firmyears_total`/`w_n_firms_total` (the *total* panel, not
  just disclosing firm-years) need a fresh `len()`/`nunique()` read of
  `data/gold/spines/firm_year/firm_year.parquet` — no precomputed field
  exists, but this is a trivial one-line read, not a new pipeline.
STATUS: **PARTIAL**.

**L4633 `tbl-c4-washing-validation`.** `data/processed/clusters/
washing_score_validation.json` → `mechanical_consistency_vs_grounding,
persistence, split_half, economic_coherence`. New source: `data/results/
washing/washing_score_validation.json` — all four sub-keys confirmed present
with matching inner structure (`mc.n/.spearman/.p`, same for `pers`/`sh`;
`econ[var]` for `rd_intensity, log_market_cap, beta, idio_vol_252d,
ps_ratio`). Producer: `scripts/analytics/washing/validate_washing_score.py`.
STATUS: **READY**.

**L4676 `tbl-call-beta-paper`.** `find_path(...)` on 6
`data/processed/clusters/call_beta_*.csv` files (also defines a local
`find_path` reused by L4760). New sources: all 6 confirmed present under
`data/results/call_beta/` (`call_beta_robustness_windows.csv`,
`call_beta_regressions.csv`, `call_beta_robustness_firm_fe.csv`,
`call_beta_robustness_delta_beta.csv`, `call_beta_robustness_market_model.csv`,
`call_beta_regression_samples.csv`). Only the path prefix changes.
STATUS: **READY**.

**L4760 `tbl-e-archetype-regressions`.**
`find_path("call_archetype_full_battery_targets.csv")`. New source:
`data/results/crash_archetypes/call_archetype_full_battery_targets.csv` —
confirmed columns and `variable` values (`dum_gov, dum_voc, hist_disclosure,
hist_substance, surprise_disclosure, surprise_substance`) match exactly.
STATUS: **READY**.

**L4827 `tbl-g1-sec-letters` / L4845 `tbl-g2-sec-cases`.** Both use
`sec_cases` loaded upstream at L3097 (READY). New source (same file):
`data/results/call_beta/sec_comment_letter_cases.json` — confirmed all
fields used by both tables (`ticker, company, date, accession, filing,
subject, relevance` for letters; `k, year, treated, controls_mean, controls`
for cases) are present. STATUS: **READY** (both chunks).

**L4878 (Appendix G export-helper setup).** Defines `_gdir()`/`_export()`,
writing to `docs/analytics/` — no DuckDB/data-processed/model read of its
own, but the mechanism is now **redundant**: `data/results/washing/
appendix_g_cohort_grounding.csv` and `appendix_g_rollup_sensitivity.csv`
already exist as independent gold-pipeline outputs (see L4901/L4937 below).
STATUS: **STAY INLINE**, but flagged for reconciliation/removal alongside
L4901/L4937/L4964.

**L4901 `tbl-h5-cohort-grounding`.** `_p("firm_year_washing_score.parquet")`
(4912); writes via `_export("appendix_g_cohort_grounding.csv", ...)`. New
source: `data/results/washing/appendix_g_cohort_grounding.csv` **already
exists**, produced by `scripts/analytics/washing/
appendix_g_cohort_and_rollup.py`, whose `cohort_grounding()` function is a
direct port of this chunk's logic (docstring cites the line range
explicitly), reading `gold/predictions/firm_year/washing_score.parquet`.
Chunk should become a plain CSV read instead of recomputing inline.
STATUS: **READY**.

**L4937 `tbl-h8-rollup-sensitivity`.** `_p("firm_activities.parquet")`
(4947). New source: `data/results/washing/appendix_g_rollup_sensitivity.csv`
**already exists**, same producer (`rollup_sensitivity()` function, reading
`gold/covariates/document/activities.parquet`). Rename (cosmetic): CSV
columns are `activity_weighted_pct`/`firm_weighted_pct` vs. the qmd's local
string labels "Activity-weighted (pooled)"/"Firm-weighted (equal per firm)".
STATUS: **READY**.

**L4964 (slide-manifest writer).** Writes `docs/analytics/
appendix_g_slide_manifest.txt` from `_manifest_rows` accumulated by L4901/
L4937's `_export()` calls. If those two chunks become plain CSV reads (no
`_export()` call), this manifest step either needs the `_manifest_rows.append(...)`
calls preserved manually or should be dropped/reworked as part of retiring
the `docs/analytics/` side channel.
STATUS: **STAY INLINE** (needs rework alongside L4878).

---

## MISSING — actionable tasks

1. **`fig-ladder-substance` (L2363)**: no producer for the 5 nested
   substance-ladder counts + 5 independent rates. Add
   `scripts/analytics/washing/ladder_substance.py` reading
   `data/gold/covariates/document/activities.parquet` → `data/results/
   washing/ladder_substance.parquet` (or `.json`), by analogy with
   `volume_vs_grounding.py`/`tech_provenance.py`.
2. **k=2 out-of-sample archetype fit (L4211)**: no persisted k=2 AA
   coordinates. Add `scripts/analytics/posture/archetype_k2_diagnostic.py`
   (same seed/method as the qmd's inline fit, by analogy with
   `oos_validation.py`'s "verbatim reproduction" pattern) → `data/results/
   posture/archetype_k2_diagnostic.json` with per-vertex promo/risk/gov/
   specificity values.
3. **Seasonal year-to-date adjustment (L15, `hd_cutoff*`/`hd_int_*`/
   `hd_share_est_2026`/`hd_act_factor`)**: no producer. Add
   `scripts/analytics/corpus/seasonal_adjustment.py` (or extend
   `headline_counts.py`) → `results/corpus/seasonal_adjustment.json`, reading
   `data/gold/covariates/document/document_panel.parquet` +
   `data/gold/covariates/document/activities.parquet`.
4. **10-K-only AI share by year (L15, `hd_tenk_share`)**: no producer. Add
   `scripts/analytics/corpus/tenk_ai_share.py` joining
   `silver.filing_manifest`(10-K)+`silver.ai_frames` by year → `results/
   corpus/tenk_ai_share_by_year.parquet`.
5. **Panel coverage counts (L338, `uni_n_firms`/`uni_n_activity_firms`/
   `bal_n`/`bal_share`)**: no producer. Extend `headline_counts.py` or add
   `scripts/analytics/corpus/panel_coverage.py` reading `data/gold/datasets/
   firm_year/firm_year.parquet` + `data/gold/covariates/document/activities.parquet`.
6. **DEF 14A institutional-channel coverage counts (L1407, `_ch_n_disclosing`/
   `_ch_n_coverage_pct`/`_ch_n_mention`/`_ch_n_omit`)**: confirmed absent from
   `data/results/posture/archetype_document_channels.json` (which has only
   `correlation_matrix`/`rates_by_form`/`rates_10k_by_section`). Extend
   `scripts/analytics/posture/check_archetype_document_channels.py` to emit
   these counts.
7. **`tech_provenance.parquet` missing `third_party_pct` field (L2568/L2616)**:
   add one line to `scripts/analytics/washing/tech_provenance.py`:
   `row["third_party_pct"] = float((df["source"] == "third_party").mean() * 100)`.
8. **"Functional Concepts" 3-bucket rollup (L627, `tbl-frame-schema`)**: not
   computed by `frame_schema_distribution.py`, currently hardcoded prose in
   the qmd. Either add the rollup to the script or do it as a light inline
   `map()` from the `concepts` rows already in
   `data/results/corpus/frame_schema_distribution.parquet`.
9. **Producer hygiene for `posture_archetype_expanding.parquet` at firm_year
   grain (L3636/L3727/L3780)**: the file on disk is current and correct, but
   its only builder, `scripts/deprecated/posture_archetype_expanding_firm_year.py`,
   is itself deprecated and its docstring points at a non-existent path. A
   live `scripts/gold/consolidate/predictions/
   posture_archetype_expanding_firm_year.py`, analogous to the existing
   `..._firm_quarter.py`, should replace it.
10. **`tbl-a0-derived-inventory` (L3999)**: no dedicated producer; all inputs
    are already in gold. Either keep as a light in-qmd aggregation or add
    `scripts/analytics/appendix/derived_inventory.py` (by analogy with
    `patents_controls.py`).
11. **n_activities discrepancy** (47,666 funnel vs. 47,767 gold covariate vs.
    47,784 deprecated processed file) — see dedicated section above. Needs a
    reconciliation decision, not a new script.
12. **Appendix G export mechanism (L4878/L4964)**: now redundant with
    `appendix_g_cohort_and_rollup.py`'s direct-to-`data/results/washing/`
    outputs. Reconcile or drop the `docs/analytics/` slide-manifest side
    channel as part of the migration.

## PARTIAL — actionable tasks

- **L15 setup**: repoint the 8 READY JSON/parquet reads; leave `hd_n_activities`/
  `hd_n_activity_firms` pending item 11's reconciliation; items 3-4 above
  cover the two genuinely missing pieces.
- **L338**: repoint `uni_frame_*`/`cov_*` now; item 5 above for the rest.
- **L627 `tbl-frame-schema`**: repoint 6 of 7 rows now; item 8 above for
  "Functional Concepts."
- **L691 `tbl-activity-schema`**: repoint 6 of 8 rows to
  `activity_schema_distribution.parquet`; source the "Business Domain" row
  from `data/gold/covariates/document/activities.parquet`'s `function_family`
  column instead.
- **L1003 / L1279 (archetypal fits)**: decide refit-vs-frozen-model policy
  once, and persist per-firm ternary weights for L1003's scatter (a one-shot
  `.transform()` against `models/posture_archetype_static/model.pkl`,
  written to a new small results file).
- **L1407**: repoint the sector heatmap now; item 6 above for the DEF 14A
  coverage counts.
- **L1511**: extend `scripts/gold/posture/posture_features.py::load_frames()`
  to select+filter on `form`.
- **L1596**: join `data/gold/covariates/firm_year/posture.parquet` to
  `data/gold/predictions/firm_year/posture_archetype_static.parquet` in place
  of `_m`.
- **L2568/L2616**: item 7 above.
- **L3636/L3727/L3780**: swap activities source to
  `data/gold/covariates/firm_year/activities_by_channel.parquet`; item 9
  above for the archetype producer; L3727's universe count is resolved via
  the firm_year spine (see detail above) — just needs verification of the
  remaining `any_ai`/2026 fields against `disclosure_volume.parquet`.
- **L3960/L3999**: rewrite local `_pp` to the gold path; rename
  `beta_post_63`→`beta_post_63d`; item 10 above for a dedicated producer
  (optional).
- **L4597**: repoint `w_n_panel`/`w_n_firms` now; add a one-line spine read
  for `w_n_firmyears_total`/`w_n_firms_total`.

## STAY INLINE (no data dependency, pure formatting/hardcoded)

L1879 `tbl-4-purpose-defs`, L1899 `tbl-4-action-defs`, L2658
`tbl-credibility-designs`, L4035 `tbl-a0-judge-selection`, L4572
`tbl-e-10k-refit` (contingent on L1511), L4878 (Appendix G export helper,
needs rework), L4964 (slide-manifest writer, needs rework).
