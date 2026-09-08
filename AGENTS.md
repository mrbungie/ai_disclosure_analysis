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

## 3. PDF handling is country-agnostic

Everything about parsing a filing PDF lives in `scripts/common/pdf/`, not
under `scripts/cl/`. Block segmentation, reading order, column detection,
table extraction, multi-part page numbering and the block taxonomy are
facts about PDFs, not about a regulator. The one genuinely per-country
piece — which strings a regulator's template repeats as page furniture —
is a `PdfProfile` in `scripts/common/pdf/profiles.py`. A new country adds
a profile there and nothing else.

- Call `scripts.common.pdf.pipeline.PdfExtractor` (reusable, loads model
  weights once) rather than the one-shot `extract_paragraphs` when
  processing more than a handful of documents.
- Backends are registered in `scripts/common/pdf/backends/` and selected
  by config (`pdf.backend`) or `--backend`. **Concurrency is the
  backend's call, not the caller's**: read `backend.max_workers` instead
  of assuming `os.cpu_count()`. A VLM backend reports 1 because it holds
  model weights on one GPU.
- The GPU backends are an optional dependency group (`pdf-vlm`), never
  base dependencies — a clean checkout must still run the whole Chilean
  pipeline on `pymupdf`.
- `scripts/cl/cmf_pdf_paragraphs.py` is a back-compat shim. Don't add to
  it.

See `docs/analytics/pdf-backend-poc.md` for the measured comparison
behind the default backend choice.
