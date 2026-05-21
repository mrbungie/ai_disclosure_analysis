# SEC AI Disclosure Archetypes Pipeline

A scalable, reproducible, and token-optimized Python data pipeline to download SEC 10-K filings, parse narrative sections, prefilter for AI mentions, and prepare candidate chunks for downstream LLM classification and analysis.

---

## 🚀 Setup & Installation

This project utilizes `uv` for python virtual environment and dependency management.

1. **Ensure `uv` is installed** (refer to [astral.sh/uv](https://astral.sh/uv) for instructions).
2. **Create and activate the virtual environment**:
   ```bash
   uv venv
   source .venv/bin/activate
   ```
3. **Install dependencies**:
   ```bash
   uv pip install -r requirements.txt
   ```

---

## 🛠️ Makefile Automation Commands

A [Makefile](file:///Users/goviedb/Development/mib_tesis_ai_disclosure/Makefile) is provided to simplify developer workflows and execution:

* **Run Unit Tests**:
  ```bash
  make test
  ```
  Runs the standard Python `unittest` suite testing regex extraction patterns, Honeywell fallbacks, and parsing bounds.
* **Install Test Dependencies**:
  ```bash
  make install-deps
  ```
  Installs `pytest` within the active virtual environment using `uv`.
* **Run Prefilter & Chunking Pipeline**:
  ```bash
  make run-pipeline
  ```
  Runs the prefiltering and candidate chunking scripts sequentially.
* **Run Bag-of-Words Feature Extraction (Script 07)**:
  ```bash
  make run-bow
  ```
  Applies multi-phase deterministic pattern matching, word counts, and ratio/formula calculations to candidate chunks (outputting to `ai_disclosure_bow_features.parquet`).
* **Run LLM Extraction (Script 08)**:
  ```bash
  make run-llm ARGS="--max-jobs 20 --concurrency 1 --delay 2.0"
  ```
  Runs semantic LLM classification using an async worker pool (outputting to `ai_disclosure_mentions.parquet`). Pass custom flags in `ARGS`.
* **Run Combined Scoring (Script 09)**:
  ```bash
  make run-scoring
  ```
  Combines deterministic rule flags and LLM classifications to compute composite risk, governance, specificity, and promotional scores (saved in `ai_scored_chunks.parquet`).
* **Build Firm-Year Panel Dataset (Script 10)**:
  ```bash
  make run-features
  ```
  Aggregates the scored chunks into a unified firm-year panel dataset (`firm_year_features.parquet`).
* **Run Full Pipeline**:
  ```bash
  make run-full-pipeline
  ```
  Executes all pipeline steps sequentially from prefiltering through feature building.

---

## 📋 Centralized Logging & Diagnostics

The pipeline features a unified, centralized structured logging framework. All major pipeline events are logged via [pipeline_logger.py](file:///Users/goviedb/Development/mib_tesis_ai_disclosure/scripts/pipeline_logger.py) to a single file-based JSON Lines log file:
* **Log Location**: `data/interim/manifests/pipeline_log.jsonl`
* **Format**: Standard JSONL including metadata (`timestamp`, `pipeline_step`, `level`, `message`, `ticker`, `cik`, `accession_number`, `duration_seconds`, `details`).

---

## 💻 Streamlit Analytics Dashboard

An interactive dashboard is implemented in [app.py](file:///Users/goviedb/Development/mib_tesis_ai_disclosure/app.py) to orchestrate and visualize the pipeline.

### Launching the Dashboard
```bash
streamlit run app.py
```

### Dashboard Tabs
* **⚙️ Configuration**: Set SEC email/headers and define search scope and forms.
* **🏢 Universe Management**: Add, search, and manage CIK/Ticker company scopes.
* **📂 Discovery & Manifests**: Interactively track downloaded/parsed status using the filing availability matrix and trigger batches.
* **🧩 Candidate Chunks**: Explore extracted paragraphs containing AI mentions.
* **📋 Pipeline Logs**:
  * **Real-time Diagnostic Metrics**: View metrics showing total logs, successes, warnings, and errors.
  * **Log Filtering**: Filter logs dynamically by severity level, pipeline step, and search terms.
  * **Interactive SQL Console**: Execute raw SQL queries directly against the logs data frame (registered as `log_df`) using **DuckDB**.
* **📘 Project Architecture**: Interactive markdown view of the system's design documentation.

---

## 🧪 Unit Testing

Unit tests are written in [test_extraction_regex.py](file:///Users/goviedb/Development/mib_tesis_ai_disclosure/tests/test_extraction_regex.py) to ensure section extraction robustness across SEC filings:
1. **Header Parsing Rules**: Verifies extraction of standard headers (e.g. `Item 1. Business`), Markdown pipe-wrapped tables, and plain-text fallbacks (e.g. Honeywell typography).
2. **Boundary Checks**: Ensures section boundary parsing isolates content correctly without running into subsequent items.
3. **Length Constraint Validation**: Filters out any extracted texts that are $\le 1000$ characters to avoid parsing empty headings or tables of contents.
