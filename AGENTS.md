# Guidelines for AI Agents

Welcome! If you are an AI assistant working on this project, please follow these instructions:

## 1. Environment Management
- **Always use `uv venv`** to manage virtual environments.
- Do not use `venv`, `virtualenv`, `conda`, or raw `pip` unless specifically requested.
- Use `uv pip install` or `uv add` to install packages.
- Always activate the virtual environment (`.venv/bin/activate` on macOS/Linux) when executing scripts.

## 2. Context & Design
- Before making changes, **read `README.md`** to understand the current scope: data acquisition only. Everything lives under `scripts/` (`scripts/01_10k/` — 10-K panel; `scripts/02_10q/` — 10-Q shock series, a SEPARATE instrument, never pooled with 10-K; `scripts/03_market_data/` — prices + factors; `scripts/common/` — shared infra). The downstream analysis stage (AI-text detection/classification) was deliberately archived out of this branch (see the `archive` git branch) to keep this repo's context to just data acquisition.
- Fetching uses `edgartools` (`scripts/common/edgar_fetch.py`), not hand-rolled `requests` calls — it pools HTTP connections and caches on disk. Don't add module-level `requests.get()` fetch loops; extend `edgar_fetch.py` instead. SEC rate limits (10 req/s) and User-Agent headers still apply — edgartools' client is configured via `edgar_fetch.configure()`.
- The pipeline follows a file-first, incremental processing paradigm using Parquet. Avoid introducing heavy database layers unless directed. `duckdb/thesis.duckdb` (via `scripts/common/build_duckdb.py`) is VIEWS over the parquet files, not a copy of the data — never point ingestion code at it.
