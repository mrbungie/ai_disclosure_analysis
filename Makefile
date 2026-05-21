.PHONY: test install-deps run-pipeline format help

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

run-rules:
	@echo "Running rule-based text scoring..."
	.venv/bin/python scripts/07_score_with_rules.py

run-llm:
	@echo "Running LLM classification..."
	.venv/bin/python scripts/08_run_llm_classifier.py $(ARGS)

run-scoring:
	@echo "Running combined scoring..."
	.venv/bin/python scripts/09_score_with_llm_booleans.py

run-features:
	@echo "Building firm-year panel dataset..."
	.venv/bin/python scripts/10_build_features.py

run-full-pipeline: run-pipeline run-rules run-llm run-scoring run-features

help:
	@echo "Available Makefile commands:"
	@echo "  make test               Run the test suite using unittest"
	@echo "  make install-deps       Install pytest in the virtual environment using uv"
	@echo "  make run-pipeline       Run the prefilter and chunking scripts"
	@echo "  make run-rules          Run the rule-based text scoring script (07)"
	@echo "  make run-llm            Run the LLM classification script (08). Pass ARGS='--limit N' to limit."
	@echo "  make run-scoring        Run the combined scoring script (09)"
	@echo "  make run-features       Run the feature builder script (10)"
	@echo "  make run-full-pipeline  Run the entire pipeline from prefiltering to features"

