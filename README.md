# SEC AI Disclosure Archetypes Pipeline

A scalable, reproducible, and token-optimized Python data pipeline to download SEC 10-K filings, parse narrative sections, prefilter for AI mentions, and prepare candidate chunks for downstream LLM classification and analysis.

## Setup
1. Ensure `uv` is installed.
2. Create and activate the virtual environment:
   ```bash
   uv venv
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   uv pip install -r requirements.txt
   ```
