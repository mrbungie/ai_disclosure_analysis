# Guidelines for AI Agents

Welcome! If you are an AI assistant working on this project, please follow these instructions:

## 1. Environment Management
- **Always use `uv venv`** to manage virtual environments.
- Do not use `venv`, `virtualenv`, `conda`, or raw `pip` unless specifically requested.
- Use `uv pip install` or `uv add` to install packages.
- Always activate the virtual environment (`.venv/bin/activate` on macOS/Linux) when executing scripts.

## 2. Context & Design
- Before making changes, **read `README.md`** to understand the pipeline scope (scripts 00–06: EDGAR extraction, section parsing, AI-keyword prefiltering and chunking).
- The pipeline follows a file-first, incremental processing paradigm using Parquet. Avoid introducing heavy database layers unless directed.
- Always keep SEC download rate limits (max 10 requests/second) and User-Agent headers in mind.
