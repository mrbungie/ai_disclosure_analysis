# Guidelines for AI Agents

Welcome! If you are an AI assistant working on this project, please follow these instructions:

## 1. Environment Management
- **Always use `uv venv`** to manage virtual environments.
- Do not use `venv`, `virtualenv`, `conda`, or raw `pip` unless specifically requested.
- Use `uv pip install` or `uv add` to install packages.
- Always activate the virtual environment (`.venv/bin/activate` on macOS/Linux) when executing scripts.

## 2. Context & Design
- Before making changes, **read `README.md`** to understand the current scope: data acquisition only (`01_10k/` — EDGAR firm universe, filing manifest, download, section extraction; `02_market_data/` — prices + factors). The downstream analysis stage (AI-text detection/classification) was deliberately archived out of this branch (see the `archive` git branch) to keep this repo's context to just data acquisition.
- The pipeline follows a file-first, incremental processing paradigm using Parquet. Avoid introducing heavy database layers unless directed.
- Always keep SEC download rate limits (max 10 requests/second) and User-Agent headers in mind.
