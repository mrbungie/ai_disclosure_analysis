.PHONY: test install-deps tickers-tui build-universe fetch-10k extract-sections collect-data fetch-10q extract-sections-10q collect-data-10q section-audit collect-market duckdb duckdb-text prefilter analytics analytics-text analytics-financials analytics-panels analytics-activities analytics-call-beta b2-check refresh-stale help

# Default target
all: test

# Run tests using the unittest module in the virtual environment
test:
	@echo "Running unit tests..."
	.venv/bin/python -m unittest discover -s scripts/us/tests -p "test_*.py" -v
	.venv/bin/python -m unittest discover -s scripts/common/tests -p "test_*.py" -v

# Install developer/test dependencies (using uv as per project rules)
install-deps:
	@echo "Installing test dependencies..."
	uv pip install pytest

# ---- scripts/us: US/SEC EDGAR — firm universe + 10-K/10-Q pipelines ----
# Scripts are organized BY COUNTRY (scripts/<country>/...) so a future
# exchange/source can be added as its own sibling directory without
# touching this one — fetch AND extraction both live inside scripts/us/
# because extraction (section_segmenter.py) is sensitive to local filing-
# format idiosyncrasies (SEC's "Item N" convention), not just fetch
# mechanics. Only the run/checkpoint plumbing (scripts/common/
# section_extraction.py) and truly generic infra (pipeline_logger,
# build_duckdb) are country-agnostic and live in scripts/common/.
# WHAT gets downloaded is config-driven: configs/universe.csv (universe,
# has a country column already) + configs/config.yaml: corpus.filings.
# filing_date (window). Fetch is via edgartools (scripts/us/edgar_fetch.py)
# — idempotent, skips any filing whose local gzip mirror already exists.

build-universe:
	@echo "Building firm universe from configured tickers (scripts/us/00)..."
	.venv/bin/python scripts/us/00_build_firm_universe.py

fetch-10k:
	@echo "Building the 10-K manifest and fetching primary documents (scripts/us/10k/01)..."
	.venv/bin/python scripts/us/10k/01_fetch_filings.py

extract-sections:
	@echo "Extracting Business/Risk/MD&A sections (scripts/us/10k/02)..."
	.venv/bin/python scripts/us/10k/02_extract_sections.py $(ARGS)

collect-data: build-universe fetch-10k extract-sections

tickers-tui:
	@.venv/bin/python scripts/us/tui_tickers.py

section-audit:
	@echo "Auditing extraction coverage against the full 10-K text (scripts/verif/section_audit)..."
	.venv/bin/python scripts/verif/section_audit/section_audit.py

# ---- scripts/us/10q: 10-Q shock series — a SEPARATE instrument, never pooled with 10-K ----
# Own manifest, own extraction run (Item 2/MD&A + Item 1A/Risk Factor
# updates — see scripts/us/10q/02_extract_sections.py), own duckdb views
# — never pooled or unioned with the 10-K outputs above.

fetch-10q:
	@echo "Building the 10-Q manifest and fetching primary documents (scripts/us/10q/01)..."
	.venv/bin/python scripts/us/10q/01_fetch_filings.py

extract-sections-10q:
	@echo "Extracting Item 2 (MD&A) + Item 1A (Risk Factor updates) from 10-Qs (scripts/us/10q/02)..."
	.venv/bin/python scripts/us/10q/02_extract_sections.py $(ARGS)

collect-data-10q: fetch-10q extract-sections-10q

# ---- scripts/03_market_data: prices + Fama-French factors (independent of scripts/us) ----

collect-market:
	@echo "Snapshotting prices (per-ticker parquet) + Fama-French factors (scripts/03_market_data/01)..."
	.venv/bin/python scripts/03_market_data/01_collect_market_data.py $(ARGS)

# ---- SQL access to all of the above ----
# `duckdb` (fast, views only) is separate from `duckdb-text` (also
# materializes paragraphs/sentences — regex-heavy, ~52s) — an explicit,
# optional step, not chained into collect-data/collect-data-10q, so
# rebuilding the fast views never pays that cost unless asked for.

duckdb:
	@echo "(Re)building duckdb/thesis.duckdb views over every parquet output..."
	.venv/bin/python scripts/common/build_duckdb.py

duckdb-text:
	@echo "(Re)building duckdb views AND the paragraphs/sentences tables (~52s)..."
	.venv/bin/python scripts/common/build_duckdb.py --with-text-tables

prefilter:
	@echo "Scoring paragraphs with DuckDB lexical matching + BGE-M3 embeddings..."
	.venv/bin/python scripts/common/ai_prefilter.py $(ARGS)

# ---- scripts/analytics: el panel empresa-año que leen docs/analytics/ ----
# Todo determinístico: ni una llamada a LLM, ni un peso de API. Lee
# gold_ai_frames (que SÍ costó llamadas y nunca se toca), el XBRL crudo, los
# precios y los factores, y produce data/processed/clusters/.
#
# El orden importa y por eso `analytics` los encadena: los clusters de texto
# no dependen de nada financiero, los ratios sí dependen del manifest de
# 10-K, el mercado depende de los ratios (necesita shares_out/EPS/equity),
# ROIC/WACC depende de ambos, y el merge final depende de todo lo anterior.

analytics-text:
	@echo "Arquetipos de voz, clusters de comportamiento y panel empresa-año (build_firm_clusters)..."
	.venv/bin/python scripts/analytics/build_firm_clusters.py $(ARGS)
	.venv/bin/python scripts/analytics/build_document_panel.py $(ARGS)
	.venv/bin/python scripts/analytics/build_segments.py $(ARGS)
	.venv/bin/python scripts/analytics/build_voice_behavior_grid.py $(ARGS)

analytics-financials:
	@echo "Contables desde XBRL, mercado/beta/CAR y ROIC-WACC (build_firm_financials -> build_market_factors -> build_roic_wacc)..."
	.venv/bin/python scripts/analytics/build_firm_financials.py $(ARGS)
	.venv/bin/python scripts/analytics/build_market_factors.py $(ARGS)
	.venv/bin/python scripts/analytics/build_roic_wacc.py $(ARGS)

analytics-panels:
	@echo "Merge texto x finanzas + score de AI-washing (build_firm_panels, washing_score)..."
	.venv/bin/python scripts/analytics/build_firm_panels.py $(ARGS)
	.venv/bin/python scripts/analytics/washing_score.py $(ARGS)
	.venv/bin/python scripts/analytics/shock_analysis.py $(ARGS)
	.venv/bin/python scripts/analytics/shock_did_simple.py $(ARGS)

analytics-activities:
	@echo "Perfiles de actividad por empresa desde gold_ai_frames (activity_profiles)..."
	.venv/bin/python scripts/analytics/activity_profiles.py $(ARGS)

# El spine A (call) que leen call_car_regressions.py / call_beta_regressions.py /
# call_beta_robustness.py. Depende de firm_activities.parquet (analytics-activities)
# y firm_year_financials_ratios.parquet (analytics-financials), así que corre
# después de ambos. Se auto-valida contra la corrida anterior (ver
# build_call_beta_panel.py::validate) antes de sobrescribirla.
analytics-call-beta:
	@echo "Panel de calls para las regresiones de beta post-call (build_call_beta_panel)..."
	.venv/bin/python scripts/analytics/build_call_beta_panel.py $(ARGS)
	@echo "Estudio de eventos SEC (cartas de comentario sobre IA) sobre el panel (sec_stacked_event_study)..."
	.venv/bin/python scripts/analytics/sec_stacked_event_study.py $(ARGS)

analytics: analytics-text analytics-financials analytics-panels analytics-activities analytics-call-beta

# ---- 10_fusion: merging the 10-K text pipeline with market data — not built yet ----

# ---- refresh-stale: rebuild everything the thesis .qmd cites, after any
# upstream change (new embeddings, new XBRL fixes, new AI classify/activities
# runs) — this project has multiple parallel agent sessions sharing the same
# B2 bucket, so "stale" here specifically means "some other session pushed
# raw/interim data this local checkout hasn't pulled yet," not just "code
# changed." b2-check surfaces that gap FIRST (report only, never auto-pulls —
# a blind pull can eat the B2 download cap, see sync_data_b2.sh's own
# comments) so a stale local run doesn't silently bake outdated numbers into
# data/processed/. duckdb-text and analytics both regenerate from whatever is
# on local disk after that check.

b2-check:
	@echo "Verificando si B2 tiene datos que este checkout local no bajó todavía (no descarga nada, solo reporta)..."
	@bash scripts/common/sync_data_b2.sh pull --dry-run 2>&1 | tail -6
	@echo "Si hay archivos listados arriba: 'bash scripts/common/sync_data_b2.sh pull --path <subruta>' antes de refresh-stale."

refresh-stale: b2-check duckdb-text analytics
	@echo "duckdb + data/processed/clusters reconstruidos desde el disco local actual."
	@echo "Si b2-check reportó pendientes, esto NO los incluyó — bajarlos y volver a correr."

help:
	@echo "Available Makefile commands:"
	@echo "  make test                     Run the test suite"
	@echo "  make install-deps             Install pytest via uv"
	@echo ""
	@echo "  scripts/us/10k (what to download = configs/universe.csv + config.yaml corpus.filings):"
	@echo "  make tickers-tui              TUI: sectors over SIC groups, per-company overrides, add tickers"
	@echo "  make collect-data             build-universe -> fetch-10k -> extract-sections (resumable)"
	@echo "  make fetch-10k                Just the manifest+fetch step (idempotent, safe to rerun)"
	@echo "  make extract-sections ARGS='--comment my-run'   Rerun just the extraction step"
	@echo "  make section-audit            Verify extraction coverage against the full 10-K text"
	@echo ""
	@echo "  scripts/us/10q (separate instrument, own config.yaml corpus.filings_10q):"
	@echo "  make fetch-10q                Manifest+fetch for 10-Q (raw filings)"
	@echo "  make extract-sections-10q ARGS='--comment my-run'   Extract Item 2 (MD&A) + Item 1A from 10-Qs"
	@echo "  make collect-data-10q         fetch-10q -> extract-sections-10q (resumable)"
	@echo ""
	@echo "  scripts/analytics (determinístico, sin LLM — reconstruye data/processed/clusters/):"
	@echo "  make analytics                analytics-text -> analytics-financials -> analytics-panels"
	@echo "  make analytics-text           Clusters de voz/comportamiento + panel empresa-año"
	@echo "  make analytics-financials     XBRL -> ratios, precios -> beta/CAR, ROIC-WACC"
	@echo "  make analytics-panels         Merge texto x finanzas + score de AI-washing"
	@echo ""
	@echo "  scripts/03_market_data:"
	@echo "  make collect-market           Snapshot prices (per-ticker parquet) + Fama-French factors"
	@echo ""
	@echo "  make duckdb                   (Re)build duckdb/thesis.duckdb — SQL views over every parquet output (fast)"
	@echo "  make duckdb-text              Also (re)build paragraphs/sentences tables (~52s, regex-heavy)"
	@echo ""
	@echo "  make analytics-activities     Perfiles de actividad por empresa (activity_profiles.py)"
	@echo "  make analytics-call-beta      Panel de calls para regresiones de beta post-call (build_call_beta_panel.py)"
	@echo "  make b2-check                 Reporta si B2 tiene data que este checkout local no bajó (no descarga nada)"
	@echo "  make refresh-stale            b2-check -> duckdb-text -> analytics (todo lo que cita thesis.qmd, desde el disco local actual)"
	@echo ""
	@echo "  10_fusion: not built yet"
