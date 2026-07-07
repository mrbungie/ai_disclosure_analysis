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

help:
	@echo "Available Makefile commands:"
	@echo "  make test               Run the test suite using unittest"
	@echo "  make install-deps       Install pytest in the virtual environment using uv"
	@echo "  make run-pipeline       Run the prefilter and chunking scripts (05-06)"
