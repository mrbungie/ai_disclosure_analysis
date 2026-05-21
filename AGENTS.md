# Guidelines for AI Agents

Welcome! If you are an AI assistant working on this project, please follow these instructions:

## 1. Environment Management
- **Always use `uv venv`** to manage virtual environments.
- Do not use `venv`, `virtualenv`, `conda`, or raw `pip` unless specifically requested.
- Use `uv pip install` or `uv add` to install packages.
- Always activate the virtual environment (`.venv/bin/activate` on macOS/Linux) when executing scripts.

## 2. Context & Design
- Before making changes, **always read the files in the `docs/` subfolder** (specifically `docs/description.md`) to understand the architecture, data models, and research design.
- The pipeline follows a file-first, incremental processing paradigm using Parquet and DuckDB. Avoid introducing heavy database layers unless directed.
- Always keep SEC download rate limits (max 10 requests/second) and User-Agent headers in mind.
