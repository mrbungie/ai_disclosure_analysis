.PHONY: test install-deps run-pipeline fit-prefilter-sample fit-prefilter-label fit-prefilter-harness format help

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

# Run pipeline validation scripts on pending filings
run-pipeline:
	@echo "Running pipeline prefiltering and chunking..."
	.venv/bin/python scripts/05_prefilter_ai_mentions.py
	.venv/bin/python scripts/06_chunk_candidates.py

fit-prefilter-sample:
	@echo "Sampling paragraphs for the prefilter fit cycle (script 07)..."
	.venv/bin/python scripts/07_sample_and_label_prefilter.py --sample $(ARGS)

fit-prefilter-label:
	@echo "LLM-labeling the prefilter fit sample (script 07)..."
	.venv/bin/python scripts/07_sample_and_label_prefilter.py --label $(ARGS)

fit-prefilter-harness:
	@echo "Running the CV keyword-fit harness (script 08)..."
	.venv/bin/python scripts/08_fit_prefilter_harness.py $(ARGS)

help:
	@echo "Available Makefile commands:"
	@echo "  make test                    Run the test suite using unittest"
	@echo "  make install-deps            Install pytest in the virtual environment using uv"
	@echo "  make run-pipeline            Run the prefilter and chunking scripts (05-06)"
	@echo "  make fit-prefilter-sample    Sample paragraphs for the prefilter fit cycle (07). Pass ARGS='--n 800'."
	@echo "  make fit-prefilter-label     LLM-label the sample (07). Pass ARGS='--concurrency 3 --delay 0.3'."
	@echo "  make fit-prefilter-harness   Fit ai_keywords via CV on dev, evaluate once on holdout (08). Pass ARGS='--target-f1 0.85'."
