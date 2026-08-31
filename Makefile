.PHONY: test install-deps tickers-tui build-universe build-manifest select-batch download-filings extract-sections collect-data collect-market eval-sample eval-label eval-harness leaderboard apply-harness agreement-make agreement-score format help

# Default target
all: test

# Run tests using the unittest module in the virtual environment
test:
	@echo "Running unit tests..."
	.venv/bin/python -m unittest discover -s tests -p "test_*.py" -v

# Install developer/test dependencies (using uv as per project rules)
install-deps:
	@echo "Installing test dependencies..."
	uv pip install pytest

# ---- Data collection (00-04, network: SEC EDGAR; all resumable) ----
# WHAT gets downloaded is config-driven: pipeline.tickers, start_year/end_year,
# form_types in configs/config.json. 02 marks pending filings in that window
# as 'selected'; 03 downloads only 'selected'.

build-universe:
	@echo "Building firm universe from configured tickers (script 00)..."
	.venv/bin/python scripts/00_build_firm_universe.py

build-manifest:
	@echo "Building filing manifest from SEC EDGAR (script 01)..."
	.venv/bin/python scripts/01_build_filing_manifest.py

select-batch:
	@echo "Selecting pending filings inside the configured year window (script 02)..."
	.venv/bin/python scripts/02_select_download_batch.py

download-filings:
	@echo "Downloading selected filings (script 03)..."
	.venv/bin/python scripts/03_download_selected_filings.py

extract-sections:
	@echo "Extracting Business/Risk/MD&A sections (script 04)..."
	.venv/bin/python scripts/04_extract_sections.py

collect-data: build-universe build-manifest select-batch download-filings extract-sections

collect-market:
	@echo "Snapshotting prices (per-ticker parquet) + Fama-French factors (script 13)..."
	.venv/bin/python scripts/13_collect_market_data.py $(ARGS)

tickers-tui:
	@.venv/bin/python scripts/tui_tickers.py

# ---- The two harnesses (candidate programs under harnesses/<task>/) ----
# detection:      classify(text) -> bool          (pre-classification of AI text)
# classification: classify(text) -> 6 dim bools   (tags on detection's positives)
# One shared labeled eval set rewards both; iteration happens via the
# /meta-harness-opt skill (the proposer); freezes via --split test.

eval-sample:
	@echo "Sampling eval-set paragraphs, stratified by the ACTIVE detection candidate..."
	.venv/bin/python scripts/build_eval_set.py --sample $(ARGS)

eval-label:
	@echo "Obtaining reward labels for the eval set ($$, uses LLM_JUDGE_* env)..."
	.venv/bin/python scripts/build_eval_set.py --label $(ARGS)

eval-harness:
	@echo "Evaluate(H, X): scoring a candidate on the search split..."
	.venv/bin/python scripts/eval_harness.py $(ARGS)

leaderboard:
	@.venv/bin/python scripts/eval_harness.py --leaderboard

apply-harness:
	@echo "Applying the frozen (ACTIVE) candidates to the whole corpus..."
	.venv/bin/python scripts/apply_harness.py $(ARGS)

# ---- Reward-label audit (human anchor) ----

agreement-make:
	@echo "Exporting the reward-label audit workbook to hand-label..."
	.venv/bin/python scripts/agreement_check.py --cycle eval --make $(ARGS)

agreement-score:
	@echo "Scoring the filled-in audit workbook (Cohen's kappa)..."
	.venv/bin/python scripts/agreement_check.py --cycle eval --score

help:
	@echo "Available Makefile commands:"
	@echo "  make test                     Run the test suite"
	@echo "  make install-deps             Install pytest via uv"
	@echo ""
	@echo "  Data collection (00-04; what to download = configs/universe.csv + pipeline.{start_year,end_year}):"
	@echo "  make tickers-tui              TUI: sectors over SIC groups, per-company overrides, add tickers"
	@echo "  make collect-data             Run 00->04 in order (resumable)"
	@echo "  make collect-market           Snapshot prices (per-ticker parquet) + Fama-French factors (13)"
	@echo ""
	@echo "  Harness optimization (two tasks: detection, classification):"
	@echo "  make eval-sample              Draw eval-set paragraphs (free). ARGS='--n 800'"
	@echo "  make eval-label               Reward-label them ($$, LLM) + auto-export audit workbook"
	@echo "  make eval-harness ARGS='--task detection --candidate 001_x'   Score a candidate (search split)"
	@echo "  make leaderboard              Standings per task"
	@echo "  make apply-harness            Run the frozen ACTIVE candidates over the corpus"
	@echo "  /meta-harness-opt <task>      (in Claude Code) one proposer iteration; freeze = --split test"
	@echo ""
	@echo "  make agreement-make / agreement-score    Human audit of the reward labels (kappa)"
