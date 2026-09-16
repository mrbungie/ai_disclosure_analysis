.PHONY: test install-deps tickers-tui data-views explorer validator validator-sample hf-sync hf-sync-models hf-sync-data hf-sync-interim hf-sync-raw hf-sync-all build-universe fetch-10k extract-sections collect-data fetch-10q extract-sections-10q collect-data-10q section-audit collect-market bronze silver layers prefilter gold gold-document gold-firm gold-firm-year gold-firm-quarter gold-call gold-datasets gold-check analytics analytics-corpus analytics-posture analytics-washing analytics-shock analytics-channel-gap analytics-call-beta analytics-crash-archetypes analytics-appendix analytics-stability analytics-prefilter-eval b2-check refresh-stale help

# Default target
all: test

# Run tests using the unittest module in the virtual environment
test:
	@echo "Running unit tests..."
	.venv/bin/python -m unittest discover -s scripts/raw_processing/us/tests -p "test_*.py" -v
	.venv/bin/python -m unittest discover -s scripts/common/tests -p "test_*.py" -v
	.venv/bin/python -m unittest discover -s scripts/bronze/tests -p "test_*.py" -v
	.venv/bin/python -m unittest discover -s scripts/enrichment/tests -p "test_*.py" -v
	.venv/bin/python -m unittest discover -s scripts/analytics/tests -p "test_*.py" -v

# Install developer/test dependencies (using uv as per project rules)
install-deps:
	@echo "Installing test dependencies..."
	uv pip install pytest

# ---- scripts/raw_ingestion + scripts/raw_processing: US/SEC EDGAR — firm universe + 10-K/10-Q pipelines ----
# Scripts are organized BY STAGE first (raw_ingestion/raw_processing/bronze/
# gold/analytics), country second (scripts/<stage>/<country>/...) so a
# future exchange/source can be added as its own sibling directory without
# touching this one. Fetch (raw_ingestion/us/) and extraction
# (raw_processing/us/section_segmenter.py, sensitive to local filing-format
# idiosyncrasies -- SEC's "Item N" convention) are separate stages now, but
# both still country-specific. Only the run/checkpoint plumbing
# (scripts/common/section_extraction.py) and truly generic infra
# (pipeline_logger, layers) are country-agnostic and live in
# scripts/common/.
# WHAT gets downloaded is config-driven: configs/universe.csv (universe,
# has a country column already) + configs/config.yaml: corpus.filings.
# filing_date (window). Fetch is via edgartools (scripts/raw_ingestion/us/
# edgar_fetch.py) — idempotent, skips any filing whose local gzip mirror
# already exists.

build-universe:
	@echo "Building firm universe from configured tickers (scripts/raw_ingestion/us/00)..."
	.venv/bin/python scripts/raw_ingestion/us/00_build_firm_universe.py

fetch-10k:
	@echo "Building the 10-K manifest and fetching primary documents (scripts/raw_ingestion/us/10k/01)..."
	.venv/bin/python scripts/raw_ingestion/us/10k/01_fetch_filings.py

extract-sections:
	@echo "Extracting Business/Risk/MD&A sections (scripts/raw_processing/us/10k/02)..."
	.venv/bin/python scripts/raw_processing/us/10k/02_extract_sections.py $(ARGS)

collect-data: build-universe fetch-10k extract-sections

tickers-tui:
	@.venv/bin/python scripts/raw_ingestion/us/tui_tickers.py

section-audit:
	@echo "Auditing extraction coverage against the full 10-K text (scripts/verif/section_audit)..."
	.venv/bin/python scripts/verif/section_audit/section_audit.py

# ---- scripts/raw_ingestion|raw_processing/us/10q: 10-Q shock series — a SEPARATE instrument, never pooled with 10-K ----
# Own manifest, own extraction run (Item 2/MD&A + Item 1A/Risk Factor
# updates — see scripts/raw_processing/us/10q/02_extract_sections.py), own
# bronze/silver tables — never pooled or unioned with the 10-K outputs above.

fetch-10q:
	@echo "Building the 10-Q manifest and fetching primary documents (scripts/raw_ingestion/us/10q/01)..."
	.venv/bin/python scripts/raw_ingestion/us/10q/01_fetch_filings.py

extract-sections-10q:
	@echo "Extracting Item 2 (MD&A) + Item 1A (Risk Factor updates) from 10-Qs (scripts/raw_processing/us/10q/02)..."
	.venv/bin/python scripts/raw_processing/us/10q/02_extract_sections.py $(ARGS)

collect-data-10q: fetch-10q extract-sections-10q

# ---- scripts/raw_ingestion/market: prices + Fama-French factors (independent of scripts/*/us) ----

collect-market:
	@echo "Snapshotting prices (per-ticker parquet) + Fama-French factors (scripts/raw_ingestion/market/01)..."
	.venv/bin/python scripts/raw_ingestion/market/01_collect_market_data.py $(ARGS)

# ---- parquet layers: data/bronze/ (one cleaned table per source) -> data/silver/ (analysis universe, gold-ready) ----
# Catalog and lineage keys: scripts/common/layers.py. Additive outputs
# (embeddings, prefilter scores/predictions, LLM frames/activities, golden
# set) are only read, never rewritten. Each table writes <table>._manifest.json.
#
# scripts/raw_ingestion/patents/ (BigQuery fetch of Google Patents / OECD AI-patent
# matches into data/raw/patents/) has no Makefile target: it makes network/BigQuery
# calls and is run manually, once, on a machine with BigQuery credentials.
# scripts/bronze/patents.py and scripts/silver/patents.py below only clean/conform
# what's already on disk under data/raw/patents/.

bronze:
	@echo "Building data/bronze/ (manifests, EDGAR submissions, call-transcript checks, paragraphs/sentences, unique texts, market, prefilter, LLM outputs)..."
	.venv/bin/python scripts/bronze/manifests.py
	.venv/bin/python scripts/bronze/sec_submissions.py
	.venv/bin/python scripts/bronze/paragraphs.py
	.venv/bin/python scripts/bronze/unique_paragraphs.py
	.venv/bin/python scripts/bronze/market.py
	.venv/bin/python scripts/bronze/prefilter.py
	.venv/bin/python scripts/bronze/llm_outputs.py
	.venv/bin/python scripts/bronze/call_transcripts.py
	.venv/bin/python scripts/bronze/patents.py
	.venv/bin/python scripts/bronze/xbrl_facts.py

silver:
	@echo "Building data/silver/ (analysis universe + LLM outputs and vendor mentions per paragraph instance)..."
	.venv/bin/python scripts/silver/universe.py
	.venv/bin/python scripts/silver/ai_outputs.py
	.venv/bin/python scripts/silver/ai_vendors.py
	.venv/bin/python scripts/silver/patents.py

layers: bronze silver

prefilter:
	@echo "Scoring paragraphs with lexical matching + BGE-M3 embeddings..."
	.venv/bin/python scripts/enrichment/ai_prefilter.py $(ARGS)

# ---- scripts/gold (data/gold, models/) -> scripts/analytics (data/results/<topic>/) ----
# Deterministic: no LLM or API call. Gold reads silver/bronze (the additive LLM
# outputs are never touched) and writes data/gold/<kind>/<grain>/<family>.parquet
# (docs/gold_pipeline.md): per grain, the spine first, then covariate and
# target families joined onto it; later grains read earlier gold tables.
# Analytics reads gold and writes data/results/<topic>/, which only
# thesis.qmd reads.
#
# Every target runs through scripts/common/run_cached.sh: a sha256 of (path,
# size, mtime) over the declared inputs (its scripts + upstream layer
# manifests / gold files) is compared with the hash of the last run, and an
# unchanged block is skipped. render.py calls `make analytics` before every
# render; the cache is what makes that cheap. State lives in .make_cache/
# (gitignored, delete it to force a full rebuild).
RUN_CACHED := scripts/common/run_cached.sh
LAYER_MANIFESTS := 'data/bronze/*._manifest.json' 'data/silver/*._manifest.json'
# Gold inputs of a block: the spine and families of the grains it reads, never
# datasets/<grain>/<grain> (rewritten by gold-datasets after every family
# block, which would make an unchanged block miss its cache on every run).
GOLD_DOCUMENT := 'data/gold/spines/document/*.parquet' 'data/gold/covariates/document/*.parquet' 'data/gold/targets/document/*.parquet'
GOLD_ACTIVITY := 'data/gold/spines/activity/*.parquet' 'data/gold/covariates/activity/*.parquet' 'data/gold/targets/activity/*.parquet'
GOLD_FIRM := 'data/gold/spines/firm/*.parquet' 'data/gold/covariates/firm/*.parquet' 'data/gold/targets/firm/*.parquet'
GOLD_FIRM_YEAR := 'data/gold/spines/firm_year/*.parquet' 'data/gold/covariates/firm_year/*.parquet' 'data/gold/targets/firm_year/*.parquet'
# Models gold fits. Analytics blocks watch these, not models/incremental_signal
# (written by analytics-shock itself, which would re-run every block).
GOLD_MODELS := 'models/posture_archetype_static/*.pkl' 'models/posture_archetype_weights/**/*.pkl' \
	'models/washing_grounding_shrinkage/*.pkl'
GOLD_LIBS := scripts/common/layers.py scripts/common/pit.py scripts/gold/posture/ai_intensity.py \
	scripts/gold/posture/posture_features.py scripts/gold/posture/warm_start_aa.py scripts/gold/financials/*.py

gold-document:
	@echo "Gold: document spine and disclosure volume, activity spine and families, document activities, AI vendor ecosystems..."
	@$(RUN_CACHED) gold-document \
		scripts/gold/document/build_document.py \
		scripts/gold/activity/build_activity.py \
		scripts/gold/document/build_activities.py \
		scripts/gold/document/build_ai_vendors.py \
		$(GOLD_LIBS) $(LAYER_MANIFESTS) \
		-- bash -c '.venv/bin/python scripts/gold/document/build_document.py $(ARGS) && \
			.venv/bin/python scripts/gold/activity/build_activity.py $(ARGS) && \
			.venv/bin/python scripts/gold/document/build_activities.py $(ARGS) && \
			.venv/bin/python scripts/gold/document/build_ai_vendors.py $(ARGS)'

gold-firm: gold-document
	@echo "Gold: firm spine and static posture archetype (full-sample fit, bootstrap k selection)..."
	@$(RUN_CACHED) gold-firm \
		scripts/gold/firm/build_posture_archetype_static.py \
		$(GOLD_DOCUMENT) \
		$(GOLD_LIBS) $(LAYER_MANIFESTS) \
		-- bash -c '.venv/bin/python scripts/gold/firm/build_posture_archetype_static.py $(ARGS)'

gold-firm-year: gold-document gold-firm
	@echo "Gold: firm-year spine, disclosure, posture, financials and market, patents, washing score..."
	@$(RUN_CACHED) gold-firm-year \
		scripts/gold/firm_year/build_spine.py \
		scripts/gold/firm_year/build_disclosure.py \
		scripts/gold/firm_year/build_posture.py \
		scripts/gold/firm_year/build_market_financials.py \
		scripts/gold/firm_year/build_patents.py \
		scripts/gold/firm_year/build_washing_score.py \
		$(GOLD_DOCUMENT) $(GOLD_FIRM) \
		'data/raw/xbrl_facts/us_by_filing/*.parquet' \
		$(GOLD_LIBS) $(LAYER_MANIFESTS) \
		-- bash -c '.venv/bin/python scripts/gold/firm_year/build_spine.py $(ARGS) && \
			.venv/bin/python scripts/gold/firm_year/build_disclosure.py $(ARGS) && \
			.venv/bin/python scripts/gold/firm_year/build_posture.py $(ARGS) && \
			.venv/bin/python scripts/gold/firm_year/build_market_financials.py $(ARGS) && \
			.venv/bin/python scripts/gold/firm_year/build_patents.py $(ARGS) && \
			.venv/bin/python scripts/gold/firm_year/build_washing_score.py $(ARGS)'

gold-firm-quarter: gold-document gold-firm-year
	@echo "Gold: firm-quarter spine and measures by channel family and window, posture archetype, washing score, financials, market and forward targets..."
	@$(RUN_CACHED) gold-firm-quarter \
		scripts/gold/firm_quarter/build_measures.py \
		scripts/gold/firm_quarter/build_posture_archetype.py \
		scripts/gold/firm_quarter/build_washing_score.py \
		scripts/gold/firm_quarter/build_financials.py \
		scripts/gold/firm_quarter/build_market.py \
		$(GOLD_DOCUMENT) 'data/raw/xbrl_frames_alt/*.parquet' \
		$(GOLD_LIBS) $(LAYER_MANIFESTS) \
		-- bash -c '.venv/bin/python scripts/gold/firm_quarter/build_measures.py $(ARGS) && \
			.venv/bin/python scripts/gold/firm_quarter/build_posture_archetype.py $(ARGS) && \
			.venv/bin/python scripts/gold/firm_quarter/build_washing_score.py $(ARGS) && \
			.venv/bin/python scripts/gold/firm_quarter/build_financials.py $(ARGS) && \
			.venv/bin/python scripts/gold/firm_quarter/build_market.py $(ARGS)'

gold-call: gold-document gold-firm-year gold-firm-quarter
	@echo "Gold: call spine, call disclosure, call market and financials (covariates and targets)..."
	@$(RUN_CACHED) gold-call \
		scripts/gold/call/build_spine.py \
		scripts/gold/call/build_disclosure.py \
		scripts/gold/call/build_market_financials.py \
		scripts/gold/activity/build_activity.py scripts/gold/firm_year/build_washing_score.py \
		$(GOLD_DOCUMENT) $(GOLD_ACTIVITY) $(GOLD_FIRM_YEAR) \
		$(GOLD_LIBS) $(LAYER_MANIFESTS) \
		-- bash -c '.venv/bin/python scripts/gold/call/build_spine.py $(ARGS) && \
			.venv/bin/python scripts/gold/call/build_disclosure.py $(ARGS) && \
			.venv/bin/python scripts/gold/call/build_market_financials.py $(ARGS)'

# One dataset per grain (datasets/<grain>/<grain>): the spine joined with every
# covariate and target family of the grain, after all of them exist; firm
# before activity (firm__ archetype) and firm_quarter before call (fq__ as of
# the call date).
gold-datasets: gold-document gold-firm gold-firm-year gold-firm-quarter gold-call
	@echo "Gold: one dataset per grain (spine + covariates + targets)..."
	@$(RUN_CACHED) gold-datasets \
		scripts/gold/datasets/join_families.py \
		scripts/gold/document/build_dataset.py scripts/gold/firm/build_dataset.py \
		scripts/gold/activity/build_dataset.py scripts/gold/firm_year/build_dataset.py \
		scripts/gold/firm_quarter/build_dataset.py scripts/gold/call/build_dataset.py \
		'data/gold/spines/*/*.parquet' 'data/gold/covariates/*/*.parquet' 'data/gold/targets/*/*.parquet' \
		scripts/common/layers.py scripts/common/pit.py \
		-- bash -c '.venv/bin/python scripts/gold/document/build_dataset.py $(ARGS) && \
			.venv/bin/python scripts/gold/firm/build_dataset.py $(ARGS) && \
			.venv/bin/python scripts/gold/activity/build_dataset.py $(ARGS) && \
			.venv/bin/python scripts/gold/firm_year/build_dataset.py $(ARGS) && \
			.venv/bin/python scripts/gold/firm_quarter/build_dataset.py $(ARGS) && \
			.venv/bin/python scripts/gold/call/build_dataset.py $(ARGS)'

gold: gold-document gold-firm gold-firm-year gold-firm-quarter gold-call gold-datasets
	@$(MAKE) --no-print-directory data-views

# DuckDB file with one view per parquet under data/ (scripts/common/build_data_views.py).
# Recreated from scratch after every gold build: a view keeps the column types
# of the parquet it was created on, so an old file shows stale types.
data-views:
	@echo "DuckDB views over data/ -> data_views.duckdb..."
	@rm -f data_views.duckdb
	@uv run --with duckdb python scripts/common/build_data_views.py --out data_views.duckdb

# Gradio dataset explorer (DuckDB in-memory over data/)
explorer:
	@echo "Starting Gradio dataset explorer (internal DuckDB)..."
	@uv run python -m apps.explorer.app

# Atomic validation tool for human verification of AI extractions
validator:
	@echo "Starting atomic validator at http://localhost:8765..."
	@cd apps/validator && python -m http.server 8765

validator-sample:
	@echo "Building flattened atomic sample for validator..."
	@uv run --frozen --no-sync python apps/validator/build_sample.py $(ARGS)

# Hugging Face synchronization (Dataset + Models)
hf-sync:
	@echo "Syncing data and models to Hugging Face concurrently..."
	@uv run python scripts/common/hf_sync.py --both $(ARGS)

hf-sync-models:
	@echo "Syncing models to Hugging Face..."
	@uv run python scripts/common/hf_sync.py --models $(ARGS)

hf-sync-data:
	@echo "Syncing data layers to Hugging Face..."
	@uv run python scripts/common/hf_sync.py --data $(ARGS)

hf-sync-interim:
	@echo "Syncing interim data layer to Hugging Face..."
	@uv run python scripts/common/hf_sync.py --interim $(ARGS)

hf-sync-raw:
	@echo "Packaging and syncing raw data archives (.tar.gz) to Hugging Face..."
	@uv run python scripts/common/hf_sync.py --raw $(ARGS)

hf-sync-all:
	@echo "Syncing models, structured data, and raw archives to Hugging Face..."
	@uv run python scripts/common/hf_sync.py --all $(ARGS)

# Every gold table: <kind>/<grain>/<family>.parquet (datasets: datasets/<grain>/<grain>), one row per spine key with
# the spine's id and date (nulls, never missing rows); nothing else under data/gold.
gold-check:
	@.venv/bin/python scripts/gold/check_spine_alignment.py

analytics-corpus:
	@echo "Analytics (corpus) -> data/results/..."
	@$(RUN_CACHED) analytics-corpus \
		scripts/analytics/corpus/headline_counts.py \
		scripts/analytics/corpus/universe_sectors.py \
		scripts/analytics/corpus/documents_by_form_year.py \
		scripts/analytics/corpus/frame_schema_distribution.py \
		scripts/analytics/corpus/activity_schema_distribution.py \
		scripts/analytics/prefilter/funnel.py \
		$(LAYER_MANIFESTS) \
		-- bash -c '.venv/bin/python scripts/analytics/corpus/headline_counts.py $(ARGS) && \
			.venv/bin/python scripts/analytics/corpus/universe_sectors.py $(ARGS) && \
			.venv/bin/python scripts/analytics/corpus/documents_by_form_year.py $(ARGS) && \
			.venv/bin/python scripts/analytics/corpus/frame_schema_distribution.py $(ARGS) && \
			.venv/bin/python scripts/analytics/corpus/activity_schema_distribution.py $(ARGS) && \
			.venv/bin/python scripts/analytics/prefilter/funnel.py $(ARGS)'

analytics-posture: gold
	@echo "Analytics (posture) -> data/results/..."
	@$(RUN_CACHED) analytics-posture \
		scripts/analytics/posture/activity_profiles.py \
		scripts/analytics/posture/vendor_ecosystems.py \
		scripts/analytics/posture/check_archetype_document_channels.py \
		scripts/analytics/posture/archetype_10k_refit.py \
		scripts/analytics/posture/plot_archetypal_simplex.py \
		scripts/analytics/posture/strategy_dimensions_diagnostics.py \
		scripts/analytics/posture/representative_firms.py \
		scripts/analytics/posture/archetype_vertices.py \
		scripts/analytics/posture/oos_validation.py \
		scripts/analytics/posture/archetype_k2_fit.py \
		scripts/analytics/posture/archetype_sector.py \
		scripts/analytics/posture/archetype_domain.py \
		scripts/analytics/posture/archetype_activity.py \
		scripts/analytics/posture/activity_examples.py \
		scripts/analytics/posture/sector_purpose_action.py \
		scripts/analytics/posture/archetype_composition_annual.py \
		scripts/analytics/posture/entity_mentions_summary.py \
		scripts/analytics/posture/interrupted_series.py \
		scripts/gold/posture/ai_intensity.py scripts/gold/posture/posture_features.py \
		'data/gold/**/*.parquet' $(GOLD_MODELS) \
		$(LAYER_MANIFESTS) \
		-- bash -c '.venv/bin/python scripts/analytics/posture/activity_profiles.py $(ARGS) && \
			.venv/bin/python scripts/analytics/posture/vendor_ecosystems.py $(ARGS) && \
			.venv/bin/python scripts/analytics/posture/check_archetype_document_channels.py $(ARGS) && \
			.venv/bin/python scripts/analytics/posture/archetype_10k_refit.py $(ARGS) && \
			.venv/bin/python scripts/analytics/posture/plot_archetypal_simplex.py $(ARGS) && \
			.venv/bin/python scripts/analytics/posture/strategy_dimensions_diagnostics.py $(ARGS) && \
			.venv/bin/python scripts/analytics/posture/representative_firms.py $(ARGS) && \
			.venv/bin/python scripts/analytics/posture/archetype_vertices.py $(ARGS) && \
			.venv/bin/python scripts/analytics/posture/oos_validation.py $(ARGS) && \
			.venv/bin/python scripts/analytics/posture/archetype_k2_fit.py $(ARGS) && \
			.venv/bin/python scripts/analytics/posture/archetype_sector.py $(ARGS) && \
			.venv/bin/python scripts/analytics/posture/archetype_domain.py $(ARGS) && \
			.venv/bin/python scripts/analytics/posture/archetype_activity.py $(ARGS) && \
			.venv/bin/python scripts/analytics/posture/activity_examples.py $(ARGS) && \
			.venv/bin/python scripts/analytics/posture/sector_purpose_action.py $(ARGS) && \
			.venv/bin/python scripts/analytics/posture/archetype_composition_annual.py $(ARGS) && \
			.venv/bin/python scripts/analytics/posture/entity_mentions_summary.py $(ARGS) && \
			.venv/bin/python scripts/analytics/posture/interrupted_series.py $(ARGS)'

analytics-washing: gold
	@echo "Analytics (washing) -> data/results/..."
	@$(RUN_CACHED) analytics-washing \
		scripts/analytics/washing/validate_washing_score.py \
		scripts/analytics/washing/activity_grounding.py \
		scripts/analytics/washing/build_strategy_economic_profiles.py \
		scripts/analytics/washing/ai_diffusion.py \
		scripts/analytics/washing/volume_vs_grounding.py \
		scripts/analytics/washing/tech_provenance.py \
		scripts/analytics/washing/ladder_substance.py \
		scripts/analytics/washing/firm_comparisons.py \
		scripts/analytics/washing/appendix_g_cohort_and_rollup.py \
		scripts/gold/posture/ai_intensity.py scripts/gold/posture/posture_features.py \
		'data/gold/**/*.parquet' $(GOLD_MODELS) \
		$(LAYER_MANIFESTS) \
		-- bash -c '.venv/bin/python scripts/analytics/washing/validate_washing_score.py $(ARGS) && \
			.venv/bin/python scripts/analytics/washing/activity_grounding.py $(ARGS) && \
			.venv/bin/python scripts/analytics/washing/build_strategy_economic_profiles.py $(ARGS) && \
			.venv/bin/python scripts/analytics/washing/ai_diffusion.py $(ARGS) && \
			.venv/bin/python scripts/analytics/washing/volume_vs_grounding.py $(ARGS) && \
			.venv/bin/python scripts/analytics/washing/tech_provenance.py $(ARGS) && \
			.venv/bin/python scripts/analytics/washing/ladder_substance.py $(ARGS) && \
			.venv/bin/python scripts/analytics/washing/firm_comparisons.py $(ARGS) && \
			.venv/bin/python scripts/analytics/washing/appendix_g_cohort_and_rollup.py $(ARGS)'

analytics-shock: gold
	@echo "Analytics (shock) -> data/results/..."
	@$(RUN_CACHED) analytics-shock \
		scripts/analytics/shock/incremental_signal.py \
		scripts/analytics/shock/evolution_figures.py \
		scripts/gold/posture/ai_intensity.py scripts/gold/posture/posture_features.py \
		'data/gold/**/*.parquet' $(GOLD_MODELS) \
		$(LAYER_MANIFESTS) \
		-- bash -c '.venv/bin/python scripts/analytics/shock/incremental_signal.py $(ARGS) && \
			.venv/bin/python scripts/analytics/shock/evolution_figures.py $(ARGS)'

analytics-channel-gap: gold
	@echo "Analytics (channel-gap) -> data/results/..."
	@$(RUN_CACHED) analytics-channel-gap \
		scripts/analytics/channel_gap/channel_gap_words_robustness.py \
		'data/gold/**/*.parquet' $(GOLD_MODELS) \
		$(LAYER_MANIFESTS) \
		-- bash -c '.venv/bin/python scripts/analytics/channel_gap/channel_gap_words_robustness.py $(ARGS)'

analytics-call-beta: gold
	@echo "Analytics (call-beta) -> data/results/..."
	@$(RUN_CACHED) analytics-call-beta \
		scripts/analytics/call_regression.py \
		scripts/analytics/call_beta/call_beta_regressions.py \
		scripts/analytics/call_beta/call_beta_robustness.py \
		scripts/analytics/call_beta/call_beta_generalized_targets.py \
		scripts/analytics/call_beta/sec_comment_letter_cases.py \
		'data/gold/**/*.parquet' $(GOLD_MODELS) \
		$(LAYER_MANIFESTS) \
		-- bash -c '.venv/bin/python scripts/analytics/call_beta/call_beta_regressions.py $(ARGS) && \
			.venv/bin/python scripts/analytics/call_beta/call_beta_robustness.py $(ARGS) && \
			.venv/bin/python scripts/analytics/call_beta/call_beta_generalized_targets.py $(ARGS) && \
			.venv/bin/python scripts/analytics/call_beta/sec_comment_letter_cases.py $(ARGS)'

analytics-crash-archetypes: gold
	@echo "Analytics (crash-archetypes) -> data/results/..."
	@$(RUN_CACHED) analytics-crash-archetypes \
		scripts/analytics/call_regression.py \
		scripts/analytics/crash_archetypes/call_crash_regressions.py \
		scripts/analytics/crash_archetypes/call_archetype_full_battery.py \
		'data/gold/**/*.parquet' $(GOLD_MODELS) \
		$(LAYER_MANIFESTS) \
		-- bash -c '.venv/bin/python scripts/analytics/crash_archetypes/call_crash_regressions.py $(ARGS) && \
			.venv/bin/python scripts/analytics/crash_archetypes/call_archetype_full_battery.py $(ARGS)'

analytics-appendix: gold
	@echo "Analytics (appendix) -> data/results/..."
	@$(RUN_CACHED) analytics-appendix \
		scripts/analytics/appendix/patents_controls.py \
		scripts/analytics/appendix/report_crosscheck_stats.py \
		scripts/analytics/appendix/earnings_calls_analysis.py \
		'data/gold/**/*.parquet' $(GOLD_MODELS) \
		$(LAYER_MANIFESTS) \
		-- bash -c '.venv/bin/python scripts/analytics/appendix/patents_controls.py $(ARGS) && \
			.venv/bin/python scripts/analytics/appendix/report_crosscheck_stats.py $(ARGS) && \
			.venv/bin/python scripts/analytics/appendix/earnings_calls_analysis.py $(ARGS)'

# 200-replicate dual bootstrap (frame-level + firm-level) of the k=3 posture
# archetype: minutes on a cache miss, so it is its own cached target.
analytics-stability: gold-document
	@echo "Bootstrap de estabilidad del archetype (200 réplicas x 2 regímenes x k=2..5)..."
	@$(RUN_CACHED) analytics-stability \
		scripts/analytics/posture/bootstrap_archetype_stability.py \
		scripts/gold/posture/ai_intensity.py scripts/gold/posture/posture_features.py \
		$(LAYER_MANIFESTS) \
		-- bash -c '.venv/bin/python scripts/analytics/posture/bootstrap_archetype_stability.py $(ARGS)'

# Prefilter evaluation reports (CV, holdout, form validation, run diff): they
# read the golden set and model bundles, not part of the thesis render path.
analytics-prefilter-eval:
	@echo "Prefilter evaluation reports -> data/results/prefilter/..."
	@$(RUN_CACHED) analytics-prefilter-eval \
		scripts/analytics/prefilter/logit_cv_metrics.py \
		scripts/analytics/prefilter/holdout_form_metrics.py \
		scripts/analytics/prefilter/form_validation_metrics.py \
		scripts/analytics/prefilter/run_diff.py \
		$(LAYER_MANIFESTS) \
		'models/ai_classification/**/*' \
		-- bash -c '.venv/bin/python scripts/analytics/prefilter/logit_cv_metrics.py $(ARGS) && \
			.venv/bin/python scripts/analytics/prefilter/holdout_form_metrics.py $(ARGS) && \
			.venv/bin/python scripts/analytics/prefilter/form_validation_metrics.py $(ARGS) && \
			.venv/bin/python scripts/analytics/prefilter/run_diff.py $(ARGS)'

analytics: analytics-corpus analytics-posture analytics-washing analytics-shock analytics-channel-gap analytics-call-beta analytics-crash-archetypes analytics-appendix analytics-stability

# ---- 10_fusion: merging the 10-K text pipeline with market data — not built yet ----

# ---- refresh-stale: rebuild everything the thesis .qmd cites, after any
# upstream change (new embeddings, new XBRL fixes, new AI classify/activities
# runs) — this project has multiple parallel agent sessions sharing the same
# B2 bucket, so "stale" here specifically means "some other session pushed
# raw/interim data this local checkout hasn't pulled yet," not just "code
# changed." b2-check surfaces that gap FIRST (report only, never auto-pulls —
# a blind pull can eat the B2 download cap, see sync_data_b2.sh's own
# comments) so a stale local run doesn't silently bake outdated numbers into
# data/gold or data/results. layers and analytics both regenerate from whatever is
# on local disk after that check.

b2-check:
	@echo "Verificando si B2 tiene datos que este checkout local no bajó todavía (no descarga nada, solo reporta)..."
	@bash scripts/common/sync_data_b2.sh pull --dry-run 2>&1 | tail -6
	@echo "Si hay archivos listados arriba: 'bash scripts/common/sync_data_b2.sh pull --path <subruta>' antes de refresh-stale."

refresh-stale: b2-check layers analytics
	@echo "bronze/silver/gold/results reconstruidos desde el disco local actual."
	@echo "Si b2-check reportó pendientes, esto NO los incluyó — bajarlos y volver a correr."

help:
	@echo "Available Makefile commands:"
	@echo "  make test                     Run the test suite"
	@echo "  make install-deps             Install pytest via uv"
	@echo ""
	@echo "  scripts/raw_ingestion|raw_processing/us/10k (what to download = configs/universe.csv + config.yaml corpus.filings):"
	@echo "  make tickers-tui              TUI: sectors over SIC groups, per-company overrides, add tickers"
	@echo "  make collect-data             build-universe -> fetch-10k -> extract-sections (resumable)"
	@echo "  make fetch-10k                Just the manifest+fetch step (idempotent, safe to rerun)"
	@echo "  make extract-sections ARGS='--comment my-run'   Rerun just the extraction step"
	@echo "  make section-audit            Verify extraction coverage against the full 10-K text"
	@echo ""
	@echo "  scripts/raw_ingestion|raw_processing/us/10q (separate instrument, own config.yaml corpus.filings_10q):"
	@echo "  make fetch-10q                Manifest+fetch for 10-Q (raw filings)"
	@echo "  make extract-sections-10q ARGS='--comment my-run'   Extract Item 2 (MD&A) + Item 1A from 10-Qs"
	@echo "  make collect-data-10q         fetch-10q -> extract-sections-10q (resumable)"
	@echo ""
	@echo "  scripts/gold + scripts/analytics (determinístico, sin LLM):"
	@echo "  make gold                     spines -> covariates/targets -> datasets by grain (data/gold/<kind>/<grain>/<family>, models/)"
	@echo "  make gold-check               Check every gold table: layout, one row per spine key, spine id/date"
	@echo "  make analytics                all thesis results -> data/results/<topic>/ (runs gold first)"
	@echo "  make analytics-<topic>        corpus | posture | washing | shock | channel-gap | call-beta | crash-archetypes | appendix | stability"
	@echo "  make analytics-prefilter-eval prefilter CV/holdout/form-validation reports"
	@echo ""
	@echo "  scripts/raw_ingestion/market:"
	@echo "  make collect-market           Snapshot prices (per-ticker parquet) + Fama-French factors"
	@echo ""
	@echo "  make bronze                   Build data/bronze/ (one cleaned parquet table per source, polars)"
	@echo "  make silver                   Build data/silver/ (analysis universe + LLM outputs per paragraph)"
	@echo ""
	@echo "  make b2-check                 Reporta si B2 tiene data que este checkout local no bajó (no descarga nada)"
	@echo "  make refresh-stale            b2-check -> layers -> analytics (todo lo que cita thesis.qmd, desde el disco local actual)"
	@echo ""
	@echo "  10_fusion: not built yet"
