# Stage boundaries, gold spines, results — task log

Detail per task: `stage_violations_audit.md` (IDs below) and
`processed_mapping.md` (producer/reader of every file in `data/processed/`).
Line numbers there predate the polars migration; re-check before editing.

## Rules these tasks enforce

- Stages: raw → interim (extraction + append-only additive outputs) → bronze →
  silver → gold → results. A stage never reads a later stage.
- **enrichment**: one row per atomic unit plus the models needed to produce it
  (`models/ai_classification/`). No funnels, metrics, reports.
- **gold**: data only, spine pattern (`docs/gold_pipeline.md`): spines →
  covariates/targets → datasets → models → predictions. Builders are thin
  joins over existing spines/covariates/targets. No stats, regressions,
  figures or JSON reports. Models are gold (`models/`).
- **analytics**: the only writer of `data/results/<topic>/` — tables, figures,
  funnels, robustness, evaluation metrics too slow to compute inline in
  `thesis.qmd`. Writes nothing another script reads.
- Nothing reads `data/processed/`. Only `thesis.qmd` may, until it is repointed.
- Gold folders are named by entity grain, not frequency:
  `data/gold/<kind>/{firm_year,firm_quarter,call,document}/`. A `firm` grain is
  allowed but avoided: firm-level values come from the latest cutoff of an
  `*_expanding` firm_year/firm_quarter output, queried at read time. Cross-grain
  joins are fine; joined columns carry their source entity in the name, and each
  dataset manifest lists spine entity/grain and joined families.
- Aggregations of silver (per document, per firm-year) are gold covariates.

## Order

1. **Gold layout** — rename `yearly/`→`firm_year/`, `quarterly/`→`call/` (its id is
   `call_accession_number`); add `layers` catalog entries for gold and results.
   Replaces audit P1 (no `gold/intermediate`: panels are not a layer).
2. **Break the posture cycle** — G-H1 (strategy_dimensions ↔ activity_profiles ↔
   build_firm_panels), then G-M1, G-M3.
3. **Gold builders write gold, not processed** — G-H8a–d, G-H6a–e: every panel in
   `processed/clusters` becomes a covariate/target/prediction family or a thin
   dataset join. `document_panel`, `firm_activities` → gold covariates.
   `firm_strategy_dimensions`, `firm_washing_score`, `channel_gap_firm` → latest
   cutoff of the expanding firm_year outputs (add missing expanding outputs).
   `us_10q_financials_panel`, `firm_year_roic_wacc`, `ncskew_63`,
   `call_fundamentals_panel` → covariate/target families.
4. **Crash-risk dataset** — G-H3: `call_crash_risk_panel.parquet` has no builder.
   Add a thin join producing `data/gold/datasets/call/<name>` from existing
   spine/covariates/targets (name states content and whether expanding); move the
   NCSKEW/DUVOL regressions to analytics.
5. **Split analysis out of gold** — G-H2 (channel gap DiD), G-M4, G-M5, G-L8.
6. **Model ownership** — G-H4/A-H1: `incremental_signal` model fit moves to gold;
   the coefficient/p-value table moves to analytics.
7. **External patents** — G-H5: BigQuery fetch → raw_ingestion; firm matching →
   bronze/silver (`sp500_firm_year_ai_patents_oecd2025` → silver).
8. **Analytics reads gold only** — A-H2, A-H3, A-M4, A-M5, G-M7, A-L7 (edge list in
   `processed_mapping.md`). `call_car_regressions` computes covariates inline →
   gold. Fix its broken import (test_call_car_regressions fails).
9. **Results location** — A-M6: all analytics outputs → `data/results/<topic>/`;
   reports written into `docs/analytics/` by bootstrap_archetype_stability,
   check_archetype_document_channels, plot_archetypal_simplex move too.
10. **Prefilter reporting out of enrichment/verif** — E-M1a/b (funnel →
    `scripts/analytics/prefilter/funnel.py`), E-M2, E-M3, E-M4, E-M5, V-H1 (form
    sample/label → enrichment), V-M2–V-M5 (reports out of additive dirs; never
    delete the existing files there), E-L8.
11. **thesis.qmd** (plan: `thesis_qmd_migration.md`) — 39 inline DuckDB queries (funnels, corpus counts, frame
    stats) → analytics scripts writing `data/results/`; repoint every
    `data/processed` path. Its DuckDB views `firm_universe`/`filing_manifest` are
    already broken (Chile/Italy removal), so those cells fail until then.
    Coordinate: other sessions edit thesis.qmd.
12. **Retire DuckDB completely** — after 11: move `duckdb/thesis.duckdb` to
    `data/deprecated/`, migrate `apps/validator/build_sample.py` and
    `summarize.py`, drop `duckdb` from `pyproject.toml`, update `render.py`
    docstring.
13. **Cleanup** — orphan/deprecated files in `processed/clusters` →
    `data/deprecated/processed/` (audit P2); `scripts/deprecated/channel_gap_words_*`
    are broken by the migration (leave, deprecated); G-L9, C-L2.
14. **Regenerate gold and results**, diff against
    `data/deprecated/pre_polars_20260915/`, accept and report numeric changes.

## Progress: integration group (items 1-3 of its wave brief)

- **firm_year spine** (item 1, done): `scripts/gold/consolidate/firm_year/build_spine.py`
  writes `data/gold/spines/firm_year/firm_year.parquet` from `ai_intensity.py`
  directly; `financial_ratios`/`market` firm_year covariates and
  `firm_year/build_targets.py` repointed onto it. Value-identical to
  `firm_year_master_v2.parquet`'s row set.
- **Retire `firm_year_master_v2`** (item 2, done except the item-6 files):
  `build_firm_panels.py` deprecated (`git mv` to `scripts/deprecated/`, no
  remaining reader of its four outputs). `washing_score.py`,
  `report_crosscheck_stats.py`, `build_strategy_economic_profiles.py`,
  `build_call_beta_panel.py`, `disclosure_volume/build_covariates_call.py`,
  `call_crash_regressions.py`, `call_beta_generalized_targets.py`,
  `call_archetype_full_battery.py`, `call_beta_config_decoupling_asof.py`,
  `sec_comment_letter_cases.py` all repointed to gold; dual-writes into
  `data/processed` removed from `build_firm_financials.py`,
  `build_market_factors.py`, `build_roic_wacc.py`,
  `build_call_beta_panel.py`. `washing_score.py`'s own OUTPUT is
  deliberately left on `data/processed` for now: it is the only remaining
  input to `scripts/analytics/washing/{firm_comparisons,
  appendix_g_cohort_and_rollup,activity_grounding}.py`, which another
  session is migrating in the same wave (item 6, new files) — repointing
  the output from under that in-flight work would race it.
  Along the way, fixed a latent (ticker, fecha) merge-fanout bug (predates
  this wave) in every call-grain script that joined crash-risk/fundamentals
  data without `call_accession_number`: a handful of tickers have two
  quarters' calls on the same calendar date. Fixed in
  `call_crash_regressions.py`, `call_beta_generalized_targets.py`,
  `call_archetype_full_battery.py`, `call_beta_config_decoupling_asof.py`
  (n drops ~1-3% per regression, coefficients unchanged to 3 decimals).
- **Expanding archetypes firm_year** (item 3, done): `build_call_crash_and_archetypes.py`
  writes `data/gold/predictions/firm_year/posture_archetype_expanding.parquet`
  directly; the thin `consolidate/predictions/posture_archetype_expanding_firm_year.py`
  wrapper is deprecated. Value-identical `arch_exp` labels for all 2,893
  ticker-years.
- **Not done**: items 6 (other files in `scripts/analytics/{posture,washing}/`,
  owned by another session), 7 (Makefile dependency-order rebuild), 8
  (`data/processed` -> `data/deprecated/` move + full `make analytics` run).

## Known data notes

- `data/gold` was stale vs `processed` before the migration (washing_score 1,873 vs
  1,869 rows; quarterly expanding archetype 4,977 vs 4,964).
- `consolidate/predictions/washing_score.py` and the static archetype read
  `data/gold/covariates/yearly/firm_year.parquet`, which no script writes.
- `golden_set.py sample` order changed (seeded BLAKE2b instead of DuckDB `hash()`):
  a new sample differs from `golden_set_sample.parquet`; the pool is identical.
- Crash-risk archetype result is population-sensitive (attribution run 2026-09-15): the
  current deterministic silver population (42,359 prefiltered texts) gives NCSKEW `dum_gov`
  β −0.062 (p 0.23) and DUVOL β −0.089 (p 0.069), vs β −0.106 (p 0.021) / −0.130 (p 0.003)
  from the old nondeterministic DuckDB population. Dropping only the latest prefilter
  deployment file (227 texts) gives −0.094 (p 0.050) / −0.114 (p 0.015). No code bug; AA
  fits are deterministic; call_crash_risk_panel unchanged. Call-beta results are stable
  (max |Δβ| 0.001). Thesis Phase 7: report the crash-risk archetype finding with this
  sensitivity, not as a firm secondary finding.

## Status (2026-09-15, end of migration session)

Done: steps 1-12 and 14. Gold layout by grain; posture cycle broken; gold
builders write gold only; crash-risk dataset builder; analysis split out of
gold; incremental-signal model ownership; patents through raw_ingestion/bronze/
silver/gold; analytics read gold only; analytics outputs in data/results/<topic>/;
prefilter reporting moved to analytics; thesis.qmd reads only gold/results
(renders to docx/pdf with no duckdb/ and no data/processed/); DuckDB removed
(`duckdb/thesis.duckdb` archived at data/deprecated/duckdb/, dependency dropped);
data/processed archived at data/deprecated/processed_20260915/. `make layers
analytics` passes end to end.

Open:
- Prose edits in docs/tasks/thesis_prose_edits.md (crash-risk finding first).
- Step 13 leftovers: `scripts/deprecated/*` that import duckdb or old APIs no longer
  run (kept for history); `call_car_panel.parquet` still written under
  data/results/call_beta (it is a dataset: move to gold); `build_call_beta_panel.py`
  imports analytics/posture/activity_profiles (gold must not import analytics);
  `validate_washing_score.py` recomputes the index via gold's `build()` instead of
  reading the gold prediction.
