.PHONY: test install-deps run-pipeline format help run-factor-analysis run-validate

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

run-bow:
	@echo "Running Bag-of-Words feature extraction..."
	.venv/bin/python scripts/07_extract_bow_features.py

run-llm:
	@echo "Running LLM classification..."
	.venv/bin/python scripts/08_run_llm_classifier.py $(ARGS)

run-scoring:
	@echo "Running combined scoring..."
	.venv/bin/python scripts/09_score_with_llm_booleans.py $(ARGS)

run-features:
	@echo "Building firm-year panel dataset..."
	.venv/bin/python scripts/10_build_features.py

run-clustering:
	@echo "Running programmatic behavioral clustering..."
	.venv/bin/python scripts/11_cluster_archetypes.py --method programmatic

run-full-pipeline: run-pipeline run-bow run-llm run-scoring run-features run-clustering

run-factor-analysis:
	@echo "Running Factor Analysis on BoW features (script 13)..."
	.venv/bin/python scripts/13_factor_analysis.py

run-event-study:
	@echo "Building event study panel (script 12)..."
	.venv/bin/python scripts/12_event_study_panel.py

run-validate-sample:
	@echo "Sampling and LLM-labeling validation chunks (val_01)..."
	.venv/bin/python scripts/val_01_sample_and_label.py $(ARGS)

run-validate-compare:
	@echo "Comparing pipeline vs LLM labels (val_02)..."
	.venv/bin/python scripts/val_02_validate.py

run-validate: run-validate-sample run-validate-compare

help:
	@echo "Available Makefile commands:"
	@echo "  make test               Run the test suite using unittest"
	@echo "  make install-deps       Install pytest in the virtual environment using uv"
	@echo "  make run-pipeline       Run the prefilter and chunking scripts"
	@echo "  make run-bow            Run the Bag-of-Words feature extraction script (07)"
	@echo "  make run-llm            Run the LLM classification script (08). Pass ARGS='--limit N' to limit."
	@echo "  make run-scoring        Run the combined scoring script (09)"
	@echo "  make run-features       Run the feature builder script (10)"
	@echo "  make run-clustering     Run the programmatic clustering script (11)"
	@echo "  make run-full-pipeline  Run the entire pipeline from prefiltering to clustering"
	@echo "  make run-factor-analysis  Run Factor Analysis on BoW features (script 13)"
	@echo "  make run-event-study      Build the event study panel (script 12)"
	@echo "  make run-validate-sample  Sample + LLM-label validation chunks (val_01). Pass ARGS='--n 200'."
	@echo "  make run-validate-compare Compare pipeline vs LLM labels (val_02)"
	@echo "  make run-validate         Run full validation: sample → label → compare"


