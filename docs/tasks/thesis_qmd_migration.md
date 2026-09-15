# thesis.qmd migration plan

Goal: `thesis_document/thesis.qmd` stops querying DuckDB, stops reading
`data/processed/`, and stops computing anything heavy inline. Every number,
table and figure comes from:

- `data/results/<topic>/` — written only by `scripts/analytics/` (tables,
  funnels, regressions, robustness, bootstrap, anything slow);
- `data/gold/` and `models/` — datasets, covariates and predictions, read for
  light work only (filter, pivot, format, plot).

Inline code may select, reshape, format and plot. It may not fit models, run
regressions, query paragraph-level corpora or recompute aggregates that an
analytics script can persist.

## Current state (inventory, 2026-09-15)

- 4,985 lines, 82 python chunks, 246 inline `{python}` expressions in prose; 39 tables (30 from chunks, 9 hand-written pandoc tables) and 25 figures in the compiled docx.
- One shared DuckDB connection opened in the setup chunk (lines 80-113) and
  monkeypatched over `duckdb.connect`; 39 queries across ~20 chunks against
  `firm_universe`, `filing_manifest(_10q)`, `paragraphs`, `gold_ai_frames`,
  `gold_ai_activities`. Its `firm_universe`/`filing_manifest` views are broken
  since the Chile/Italy removal, so those cells currently fail.
- ~45 reads of `data/processed/clusters/*` (most frequent:
  `firm_strategy_dimensions` ×13, `firm_activities` ×11,
  `firm_year_washing_score` ×5, `firm_year_master_v2` ×5), plus
  `docs/analytics/bootstrap_jaccard_200_results.json` and two
  `appendix_g_*.csv`.
- Heavy inline computation: Archetypal Analysis fits at lines ~1049, ~1308,
  ~4240 (k=3, k=2, out-of-sample validation feeding the prose around 4323);
  OLS in `tbl-b1-patents-controls` (~4173); nearest-to-archetype search in
  `tbl-representative-firms`; frame/activity cross-tabs in the chunks at
  1511-1717.

## Prerequisites (docs/tasks/README.md)

Steps 1-10 of the task list must land first. The ones this plan depends on
directly:

- gold layout by entity grain and the `layers` catalog entries for gold and
  results (step 1);
- posture cycle broken, and gold builders writing gold instead of processed
  (steps 2-3): `firm_strategy_dimensions` becomes the latest cutoff of the
  expanding firm_year archetype predictions, `firm_activities` becomes a gold
  covariate;
- crash-risk dataset and regressions split (step 4);
- all analytics outputs in `data/results/<topic>/` (step 9);
- prefilter funnel in `scripts/analytics/prefilter/funnel.py` (step 10).

## Phase 0 — Baseline and coordination

1. Other sessions edit thesis.qmd. Announce the migration, work on a branch
   (`thesis-qmd-migration`), and rebase often on main; keep each phase in its
   own commit so prose edits from elsewhere merge chunk by chunk.
2. Freeze a numeric baseline from the last successful compile
   (`thesis_document/compiled/GermanOviedo_FinalThesis_20260914_171656.docx`):
   extract every table cell, every inline number from the prose, and every
   figure's source data where it is persisted (done: see the README in that directory). Store under
   `data/results/thesis_baseline/` (a CSV of `label, location, value`). This is
   the reference for the diff in Phase 7, because the current qmd no longer
   renders.

## Phase 1 — Setup chunk

- Remove `import duckdb`, the shared connection, `_DuckDBProxy`, `get_db()` and
  the `duckdb.connect` monkeypatch.
- Replace `_p(name)` / `find_path` / `_pp` with two helpers that resolve through
  `scripts/common/layers.py`: `results(topic, name)` and `gold(kind, grain,
  name)`, each raising a clear error naming the producing `make` target when
  the file is missing. Keep the parquet read cache.
- Delete any global quantity computed from SQL here; headline corpus numbers
  move to Phase 2.

## Phase 2 — Corpus and pipeline description

New script family `scripts/analytics/corpus/` → `data/results/corpus/`:

| chunk / label | today | new source |
|---|---|---|
| setup headline quantities (15-237), 338-367 | SQL on manifests + paragraphs | `corpus/headline_counts.json` |
| `fig-sic-distribution`, `tbl-a0-sic-sector-mapping`, 4506-4512 | SQL on `firm_universe` | `corpus/universe_sectors.parquet` |
| 522-542, `tbl-documents-by-form-year` | SQL on paragraphs × manifests | `corpus/documents_by_form_year.parquet` |
| `tbl-funnel-summary`, `tbl-a1-funnel-full` | SQL + interim prediction globs | `prefilter/funnel.parquet` (task E-M1a) |
| `tbl-frame-schema` | SQL on `gold_ai_frames` | `corpus/frame_schema_distribution.parquet` from `silver.ai_frames` |
| `tbl-activity-schema` | `firm_activities.parquet` | `corpus/activity_schema_distribution.parquet` from `silver.ai_activities` |

## Phase 3 — Posture archetypes

- `fig-archetypal-ternary-simplex`, `tbl-archetypes-glance`,
  `tbl-representative-firms`, 1915-2006, and the k=2 fit at ~4240: stop fitting
  inline. Read `models/posture_archetype_*` bundles and the firm_year
  predictions (latest expanding cutoff for firm-level views). Representative
  firms (distance to each archetype) → `results/posture/representative_firms.parquet`.
- Out-of-sample validation numbers in the prose (~4323: ARI, Jaccard recovery
  for the 2021-2024 and 2021-2023 training windows) →
  `scripts/analytics/posture/archetype_oos_validation.py` →
  `results/posture/oos_validation.json`.
- `fig-sector-archetype-heatmap`, 1511-1588, 1596-1717,
  `fig-archetype-activity-heatmap` (frames × activities × archetypes by domain
  and sector) → `results/posture/archetype_{sector,domain,activity}_*.parquet`.
- `tbl-archetype-activity-examples` (paragraph text) →
  `results/posture/activity_examples.parquet` with the lineage keys
  `(accession_number, item_key, paragraph_index, text_hash)` next to the text.
- `fig-archetype-stability` → `results/posture/bootstrap_jaccard_200.json`
  (moved from `docs/analytics/`).
- `strategy_dimensions_manifest.json` reads → the model bundle metadata in
  `models/posture_archetype_expanding*/`.

## Phase 4 — Activities, grounding and washing

- `fig-ai-diffusion` (`firm_year_master_v2`) → `results/washing/ai_diffusion_by_year.parquet`.
- `fig-ladder-substance`, `tbl-a2-six-dimensions`, `tbl-vendor-ecosystem`,
  `activity_grounding.json`, `activity_profiles.json` → `results/washing/`
  (the JSON reports stop being written by gold; task G-M3).
- `fig-volume-vs-grounding` (SQL on manifests) → `results/washing/volume_vs_grounding.parquet`.
- `tbl-firm-comparisons` → gold predictions `washing_score` + latest
  expanding archetype + activities covariate; light join inline, or
  `results/washing/firm_comparisons.parquet` if it needs more than a join.
- `tbl-c4-washing-validation` → `results/washing/washing_score_validation.json`
  (validation battery moved out of gold; task G-M5).
- 2568-2608, `tbl-h5-cohort-grounding`, `tbl-h8-rollup-sensitivity` →
  `results/washing/` (the `appendix_g_*.csv` files).

## Phase 5 — Market consequences

Path repoints once task step 9 moved the files:

- `fig-call-beta-baseline`, 3259-3280, `fig-economic-coefficients-c1`,
  3349-3374, `tbl-call-beta-paper` → `results/call_beta/*.csv`.
- `fig-economic-coefficients-c2`, `tbl-e-archetype-regressions` →
  `results/crash_archetypes/call_archetype_full_battery_targets.csv`.
- `fig-call-crash-risk` → `results/crash_archetypes/call_crash_archetype_regressions*.csv`
  (after task G-H3 moves the regressions out of gold).
- 3780-3797 (`panel_expanding_archetypes`) → gold predictions
  `posture_archetype_expanding` (firm_year).
- `fig-nlp-horse-race`, `tbl-c1-m0m3` → `results/shock/incremental_signal.json`.
- 3097-3127 → `results/call_beta/sec_comment_letter_cases.json`.
- 3727-3763 (channel gap) → `results/channel_gap/` for estimates and
  `channel_gap_words_robustness.json`; the cells dataset from gold (task G-H2).

## Phase 6 — Appendices

- `tbl-b1-patents-controls`: the OLS moves to
  `scripts/analytics/appendix/patents_controls.py` →
  `results/appendix/patents_controls.csv`; patents panel read from silver via
  that script (task G-H5).
- 3960-3997, 4211-4247: repoint to results/gold as above.

## Phase 7 — Render, diff, prose

1. `render.py`: its `refresh_analytics()` runs `make layers analytics` (not
   `make analytics` alone); update the docstring that mentions `duckdb-text`
   and `gold_ai_frames`.
2. Render. Every chunk must execute with no DuckDB and no `data/processed` on
   disk: temporarily rename `data/processed/` and `duckdb/` during the test
   render to prove it.
3. Diff every table cell and inline number against the Phase 0 baseline. Known
   sources of change: deterministic prefilter population (+427 flagged texts,
   silver.ai_frames 92,866 vs ~92,300 rows), regenerated gold (washing score,
   expanding archetypes), US-only corpus, ECL price fix. Record each change
   with its cause in `data/results/thesis_baseline/diff.csv`.
4. Update prose where numbers moved, following CLAUDE.md: finding →
   magnitude → interpretation → caveat, no drafting-history narrative, one
   name per construct. Search globally for stale terms (`gold_ai_frames`,
   `DuckDB`, `processed`, superseded construct names).

## Phase 8 — Retire DuckDB

After a clean render: move `duckdb/thesis.duckdb` to `data/deprecated/duckdb/`,
migrate `ui-validator/build_sample.py` and `ui-validator/summarize.py` to
polars + layers, remove `duckdb` from `pyproject.toml`, and move
`data/processed/` to `data/deprecated/processed/` (task step 13).

## Done when

- `grep -n "duckdb\|data/processed\|AA(\|smf\.ols" thesis_document/thesis.qmd` returns nothing;
- the render succeeds with `data/processed/` and `duckdb/` absent;
- every baseline difference is explained in `diff.csv` and reflected in the prose.
