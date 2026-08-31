.PHONY: test install-deps build-universe build-manifest select-batch download-filings extract-sections collect-data run-pipeline fit-prefilter-sample fit-prefilter-label fit-prefilter-harness fit-prefilter-selfcheck extract-atoms fit-tags-sample fit-tags-label fit-tags-harness tag-chunks agreement-make agreement-score format help

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

# Run pipeline validation scripts on pending filings
run-pipeline:
	@echo "Running pipeline prefiltering and chunking..."
	.venv/bin/python scripts/05_prefilter_ai_mentions.py
	.venv/bin/python scripts/06_chunk_candidates.py

# ---- Cycle 1: detection keywords (07-08) ----

fit-prefilter-sample:
	@echo "Sampling paragraphs for the prefilter fit cycle (script 07)..."
	.venv/bin/python scripts/07_sample_and_label_prefilter.py --sample $(ARGS)

fit-prefilter-label:
	@echo "LLM-labeling the prefilter fit sample (script 07)..."
	.venv/bin/python scripts/07_sample_and_label_prefilter.py --label $(ARGS)

fit-prefilter-harness:
	@echo "Fitting ai_keywords on dev, single holdout look (script 08)..."
	.venv/bin/python scripts/08_fit_prefilter_harness.py $(ARGS)

fit-prefilter-selfcheck:
	@echo "Self-checking the keyword search machinery (script 08, dev only)..."
	.venv/bin/python scripts/08_fit_prefilter_harness.py --self-check $(ARGS)

# ---- Cycle 2: classification formulas (09-12) ----

extract-atoms:
	@echo "Extracting keyword atoms per candidate chunk (script 09)..."
	.venv/bin/python scripts/09_extract_keyword_atoms.py

fit-tags-sample:
	@echo "Sampling chunks for the tag fit cycle (script 10)..."
	.venv/bin/python scripts/10_sample_and_label_tags.py --sample $(ARGS)

fit-tags-label:
	@echo "LLM-labeling the tag fit sample on 6 dimensions (script 10)..."
	.venv/bin/python scripts/10_sample_and_label_tags.py --label $(ARGS)

fit-tags-harness:
	@echo "Fitting boolean tag formulas on dev, single holdout look (script 11)..."
	.venv/bin/python scripts/11_fit_tag_harness.py $(ARGS)

tag-chunks:
	@echo "Tagging the corpus with the frozen formulas (script 12)..."
	.venv/bin/python scripts/12_tag_chunks.py

# ---- Judge validation (human anchor) ----

agreement-make:
	@echo "Exporting the agreement workbook to hand-label (CYCLE=prefilter|tags)..."
	.venv/bin/python scripts/agreement_check.py --cycle $(or $(CYCLE),prefilter) --make $(ARGS)

agreement-score:
	@echo "Scoring the filled-in agreement workbook (CYCLE=prefilter|tags)..."
	.venv/bin/python scripts/agreement_check.py --cycle $(or $(CYCLE),prefilter) --score

help:
	@echo "Available Makefile commands:"
	@echo "  make test                     Run the test suite using unittest"
	@echo "  make install-deps             Install pytest in the virtual environment using uv"
	@echo ""
	@echo "  Data collection (00-04; what to download = pipeline.{tickers,start_year,end_year,form_types} in config):"
	@echo "  make collect-data             Run 00->04 in order (or each: build-universe, build-manifest,"
	@echo "                                select-batch, download-filings, extract-sections)"
	@echo ""
	@echo "  make run-pipeline             Run the prefilter and chunking scripts (05-06)"
	@echo ""
	@echo "  Cycle 1 — detection keywords:"
	@echo "  make fit-prefilter-sample     Sample paragraphs (07). Pass ARGS='--n 800'."
	@echo "  make fit-prefilter-label      LLM-label the sample (07). Pass ARGS='--concurrency 3'."
	@echo "  make fit-prefilter-harness    Fit ai_keywords, one holdout look (08). Pass ARGS='--target-f1 0.85'."
	@echo "  make fit-prefilter-selfcheck  Validate the search machinery synthetically (08, spends nothing)."
	@echo ""
	@echo "  Cycle 2 — classification formulas:"
	@echo "  make extract-atoms            Extract keyword atoms per chunk (09)"
	@echo "  make fit-tags-sample          Sample chunks (10). Pass ARGS='--n 400'."
	@echo "  make fit-tags-label           LLM-label the sample on 6 dimensions (10)"
	@echo "  make fit-tags-harness         Fit boolean formulas, one holdout look (11)"
	@echo "  make tag-chunks               Apply frozen formulas to the corpus (12)"
	@echo ""
	@echo "  Judge validation:"
	@echo "  make agreement-make CYCLE=prefilter|tags   Export the hand-labeling workbook"
	@echo "  make agreement-score CYCLE=prefilter|tags  Score it (Cohen's kappa)"
