.PHONY: test install-deps tickers-tui build-universe build-manifest select-batch download-filings extract-sections collect-data collect-market section-audit format help

# Default target
all: test

# Run tests using the unittest module in the virtual environment
test:
	@echo "Running unit tests..."
	.venv/bin/python -m unittest discover -s 01_10k/tests -p "test_*.py" -v

# Install developer/test dependencies (using uv as per project rules)
install-deps:
	@echo "Installing test dependencies..."
	uv pip install pytest

# ---- 01_10k: firm universe + 10-K text pipeline (network: SEC EDGAR; all resumable) ----
# WHAT gets downloaded is config-driven: pipeline.tickers, start_year/end_year,
# form_types in configs/config.json. 02 marks pending filings in that window
# as 'selected'; 03 downloads only 'selected'.

build-universe:
	@echo "Building firm universe from configured tickers (01_10k/scripts/00)..."
	.venv/bin/python 01_10k/scripts/00_build_firm_universe.py

build-manifest:
	@echo "Building filing manifest from SEC EDGAR (01_10k/scripts/01)..."
	.venv/bin/python 01_10k/scripts/01_build_filing_manifest.py

select-batch:
	@echo "Selecting pending filings inside the configured year window (01_10k/scripts/02)..."
	.venv/bin/python 01_10k/scripts/02_select_download_batch.py

download-filings:
	@echo "Downloading selected filings (01_10k/scripts/03)..."
	.venv/bin/python 01_10k/scripts/03_download_selected_filings.py

extract-sections:
	@echo "Extracting Business/Risk/MD&A sections (01_10k/scripts/04)..."
	.venv/bin/python 01_10k/scripts/04_extract_sections.py $(ARGS)

collect-data: build-universe build-manifest select-batch download-filings extract-sections

tickers-tui:
	@.venv/bin/python 01_10k/scripts/tui_tickers.py

section-audit:
	@echo "Auditing extraction coverage against the full 10-K text (01_10k/verif/section_audit)..."
	.venv/bin/python 01_10k/verif/section_audit/section_audit.py

# ---- 02_market_data: prices + Fama-French factors (independent of 01_10k) ----

collect-market:
	@echo "Snapshotting prices (per-ticker parquet) + Fama-French factors (02_market_data/scripts/01)..."
	.venv/bin/python 02_market_data/scripts/01_collect_market_data.py $(ARGS)

# ---- 10_fusion: merging the 10-K text pipeline with market data — not built yet ----

help:
	@echo "Available Makefile commands:"
	@echo "  make test                     Run the test suite"
	@echo "  make install-deps             Install pytest via uv"
	@echo ""
	@echo "  01_10k (what to download = configs/universe.csv + pipeline.{start_year,end_year}):"
	@echo "  make tickers-tui              TUI: sectors over SIC groups, per-company overrides, add tickers"
	@echo "  make collect-data             Run 00->04 in order (resumable)"
	@echo "  make extract-sections ARGS='--comment my-run'   Rerun just the extraction step"
	@echo "  make section-audit            Verify extraction coverage against the full 10-K text"
	@echo ""
	@echo "  02_market_data:"
	@echo "  make collect-market           Snapshot prices (per-ticker parquet) + Fama-French factors"
	@echo ""
	@echo "  10_fusion: not built yet"
