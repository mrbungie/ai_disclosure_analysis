.PHONY: test install-deps tickers-tui build-universe build-manifest select-batch download-filings extract-sections collect-data collect-market seed-screen eval-sample-detection eval-label-detection eval-sample-classification eval-label-classification eval-harness leaderboard apply-harness apply-detection agreement-make agreement-score format help

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

# ---- The two-stage distillation cascade (docs/distillation_map.html) ----
# detection (stage 1):      classify(text) -> bool          (pre-classification of AI text)
# classification (stage 2): classify(text) -> 6 dim bools   (tags on chunks around detection's admits)
# Two separate eval sets, two separate reference judges. Stage 2 cannot
# sample until stage 1 is frozen and applied (apply-detection produces the
# candidate frame it samples from). Iteration happens via the
# /meta-harness-opt skill (the proposer); freezes via --split test.

seed-screen:
	@echo "Running the fixed lexical seed screen over the whole corpus (defines sampling strata)..."
	.venv/bin/python scripts/seed_screen.py

eval-sample-detection:
	@echo "Sampling the detection (stage 1) eval set from the seed-screen strata..."
	.venv/bin/python scripts/build_eval_set.py --stage detection --sample $(ARGS)

eval-label-detection:
	@echo "Reward-labeling the detection eval set (scope only, $$, uses LLM_JUDGE_* env)..."
	.venv/bin/python -u scripts/build_eval_set.py --stage detection --label $(ARGS)

apply-detection:
	@echo "Applying the frozen detection ACTIVE to produce the candidate frame (stage 2's population)..."
	.venv/bin/python scripts/apply_harness.py --detection-only $(ARGS)

eval-sample-classification:
	@echo "Sampling the classification (stage 2) eval set: chunks over the candidate frame..."
	.venv/bin/python scripts/build_eval_set.py --stage classification --sample $(ARGS)

eval-label-classification:
	@echo "Reward-labeling the classification eval set (six dimensions, $$, uses LLM_JUDGE_* env)..."
	.venv/bin/python -u scripts/build_eval_set.py --stage classification --label $(ARGS)

eval-harness:
	@echo "Evaluate(H, X): scoring a candidate on the search split..."
	.venv/bin/python scripts/eval_harness.py $(ARGS)

leaderboard:
	@.venv/bin/python scripts/eval_harness.py --leaderboard $(ARGS)

apply-harness:
	@echo "Applying both frozen (ACTIVE) candidates to the whole corpus..."
	.venv/bin/python scripts/apply_harness.py $(ARGS)

# ---- Reward-label audit (human anchor) ----

agreement-make:
	@echo "Exporting a reward-label audit workbook to hand-label. ARGS='--cycle eval_detection' or eval_classification"
	.venv/bin/python scripts/agreement_check.py --make $(ARGS)

agreement-score:
	@echo "Scoring a filled-in audit workbook (Cohen's kappa). ARGS='--cycle eval_detection' or eval_classification"
	.venv/bin/python scripts/agreement_check.py --score $(ARGS)

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
	@echo "  Stage 1 - detection (docs/distillation_map.html):"
	@echo "  make seed-screen                     Fixed lexical screen over the corpus (run once, or after keyword changes)"
	@echo "  make eval-sample-detection            Draw the detection eval set (free). ARGS='--n-search 560 --n-holdout 240'"
	@echo "  make eval-label-detection             Reward-label it ($$, LLM, scope only)"
	@echo "  /meta-harness-opt detection            (in Claude Code) one proposer iteration"
	@echo "  make eval-harness ARGS='--task detection --candidate 001_x'          Score on search (free, repeatable)"
	@echo "  make eval-harness ARGS='--task detection --candidate <best> --split test'   THE one look; promotes to ACTIVE"
	@echo "  make apply-detection                   Write the candidate frame (stage 2's population)"
	@echo ""
	@echo "  Stage 2 - classification (needs the candidate frame above):"
	@echo "  make eval-sample-classification        Draw chunks around admitted paragraphs (free). ARGS='--n 800'"
	@echo "  make eval-label-classification         Reward-label them ($$, LLM, six dimensions)"
	@echo "  /meta-harness-opt classification        (in Claude Code) one proposer iteration"
	@echo "  make eval-harness ARGS='--task classification --candidate <best> --split test'   THE one look; promotes to ACTIVE"
	@echo ""
	@echo "  make leaderboard              Standings per task"
	@echo "  make apply-harness            Run both frozen ACTIVE candidates over the corpus (paragraph-level output)"
	@echo ""
	@echo "  make agreement-make / agreement-score ARGS='--cycle eval_detection|eval_classification'"
	@echo "                                Human audit of the reward labels (kappa), per stage"
