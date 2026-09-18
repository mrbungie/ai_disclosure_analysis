# AI Washing or Credible Disclosure? Patterns and Clusters in Corporate AI Disclosures

Empirical research pipeline, dataset catalog, and analytical codebase investigating corporate artificial intelligence disclosure across the S&P 500 panel (2021–2026).

The project processes over 1.5 million paragraphs from SEC filings (10-K, 10-Q, 8-K, DEF 14A proxy statements, SEC comment letters) and earnings call transcripts. It couples high-throughput machine learning prefilters, structured LLM information extraction, and econometric modeling to measure the divergence between narrative AI claims and documented operational substance.

**Author:** Germán Oviedo  
**Repository:** [github.com/mrbungie/ai_disclosure_analysis](https://github.com/mrbungie/ai_disclosure_analysis)

---

## 🏛️ Repository Architecture

The codebase follows a strict **pipeline-stage-first** layout. Every stage is decoupled, reproducible, and communicates via an immutable, file-first Parquet data lake using **Polars**.

```
ai_disclosure_analysis/
├── apps/                         # Interactive web applications
│   ├── explorer/                 # Gradio + DuckDB visual data explorer (mini-Tableau & SQL console)
│   └── validator/                # Atomic human validation web interface for audit benchmarks
│
├── configs/                      # Configuration files, firm universe, and sector definitions
│   └── us/                       # SEC/US-specific configurations (universe.csv, sector_mapping.yaml)
│
├── data/                         # File-first Parquet data lake (managed, gitignored)
│   ├── raw/                      # Raw gzipped HTML filings, transcripts, and financial feeds
│   ├── interim/                  # Additive, append-only model runs (embeddings, prefilter, LLM frames)
│   ├── bronze/                   # Source-level cleaned Parquet tables with JSON schema manifests
│   ├── silver/                   # Universe-restricted (S&P 500) paragraph-level joined data
│   ├── gold/                     # Aggregated, business-ready analytical panels (firm-year, firm-quarter, etc.)
│   └── results/                  # Econometric models, regression tables, DiD estimates, and figure data
│
├── models/                       # Trained machine learning models
│   └── ai_classification/        # Logit prefilter models, tokenizer weights, and evaluation runs
│
├── notebooks/                    # Exploratory analysis and scratch notebooks
│
├── scripts/                      # Core pipeline organized by stage first, topic/source second
│   ├── raw_ingestion/            # Fetchers and downloaders (EDGAR via edgartools, market data, transcripts)
│   │   ├── us/                   # US filings: 10k/, 10q/, 8k/, proxy/, sec_letters/, earnings_calls/
│   │   ├── market/               # Per-ticker stock prices, volumes, and Fama-French 3-factor data
│   │   ├── patents/              # Patent office metadata and innovation indicators
│   │   └── cl/, it/              # Experimental international pipelines (Chile, Italy)
│   │
│   ├── raw_processing/           # Deterministic extraction & segmentation (no LLMs)
│   │   └── us/                   # Item 1/1A/7 TOC-aware section segmenter, call speaker classifier
│   │
│   ├── bronze/                   # Builds data/bronze/ using Polars (manifests, text splitting, market)
│   ├── enrichment/               # Additive ML/LLM passes (prefilter, semantic frames, activities, embeddings)
│   ├── silver/                   # Builds data/silver/ applying the S&P 500 universe filter
│   │
│   ├── gold/                     # Aggregates analytical spines across analytical topics
│   │   ├── document/             # Document-level AI volumes, specificity scores, and posture weights
│   │   ├── firm/                 # Static firm attributes, sector classifications, and posture centroids
│   │   ├── firm_year/            # Annual panel: washing scores, substance ratios, capital expenditure
│   │   ├── firm_quarter/         # Quarterly panel: disclosure dynamics and earnings season timing
│   │   ├── call/                 # Earnings call transcripts, prepared remarks vs Q&A metrics
│   │   ├── activity/             # Catalog of atomic operational actions, objects, and vendor links
│   │   ├── financials/           # FactSet standardized fundamentals, EPS consensus, and ratios
│   │   └── posture/              # Tri-axial posture coordinates and cluster assignments
│   │
│   ├── analytics/                # Econometric estimation, event studies, robustness checks, and figures
│   │   ├── posture/              # Posture archetype clustering, Dirichlet models, and bootstrap stability
│   │   ├── washing/              # Washing index validation, persistence, and substantive grounding
│   │   ├── channel_gap/          # Call-versus-filing evidentiary divergence analytics
│   │   ├── shock/                # Event-study analysis around exogenous shocks (ChatGPT, DeepSeek)
│   │   ├── call_beta/            # Post-call market response, beta shift regressions, and price drift
│   │   ├── crash_archetypes/     # Crash risk models and Benjamini-Hochberg FDR adjustments
│   │   ├── coverage/             # Panel completeness and attrition audits
│   │   ├── corpus/               # Headline descriptive metrics and summary statistics
│   │   └── appendix/             # Robustness grids, alternative windows, and diagnostic tables
│   │
│   ├── sources/                  # External data provider connectors (e.g., FactSet API drivers)
│   ├── common/                   # Shared cross-stage infra: layers.py, pipeline_logger.py, B2 sync
│   ├── verif/                    # QA, extraction coverage audits, and model validation scripts
│   └── deprecated/               # Historical iterations kept for auditability
│
├── thesis_document/              # Thesis manuscript and publication build system
│   ├── thesis.qmd                # Main Quarto source document linking directly to data/results/
│   ├── render.py                 # Custom rendering engine enforcing typography and layout standards
│   ├── references.bib            # BibTeX bibliography
│   ├── template/                 # Document template (reference.docx)
│   ├── assets/                   # Typography (Calibri/Carlito), icons, and figures
│   ├── docs/                     # Research notes, defense preparation, and methodology summaries
│   ├── GermanOviedo_FinalThesis.docx # Rendered final manuscript (Word format)
│   └── GermanOviedo_FinalThesis.pdf  # Rendered final manuscript (PDF format)
│
├── data_views.duckdb             # Relational view layer querying the Parquet lake
├── Makefile                      # Comprehensive build, orchestration, and execution recipes
├── pyproject.toml / uv.lock      # Hermetic dependency management via uv
└── AGENTS.md / CLAUDE.md         # Machine and agent pair-programming guidelines
```

---

## 🔬 Methodological Overview

The investigation addresses how listed companies disclose their engagement with artificial intelligence, identifying whether disclosure represents substantive implementation or promotional rhetoric (*AI washing*).

### 1. The Empirical Universe
- **Scope:** S&P 500 constituents anchored as of 2021-01-01 plus historical delisted satellites across 17 aggregated sectors.
- **Reporting Period:** Fiscal years 2021 through 2026 (partial).
- **Instruments:**
  - Annual reports (Form 10-K, Items 1, 1A, and 7).
  - Quarterly reports (Form 10-Q, Items 2 and 1A) treated as an independent shock series.
  - Periodic reports (Form 8-K), proxy statements (DEF 14A), and SEC comment letters.
  - Quarterly earnings call transcripts (segmented into management presentation and analyst Q&A).
  - Financial fundamentals, historical EPS consensus, and market pricing via **FactSet** and daily market feeds.

### 2. Multi-Stage Distillation Pipeline
1. **Structural Segmentation:** Regex- and TOC-aware segmenters cleanly isolate operative narrative items while discarding tables of contents, exhibits, and XBRL metadata.
2. **High-Throughput Prefilter:** A calibrated logistic model trained on domain-specific vocabulary and embeddings filters the 1.5M+ paragraph corpus down to candidate AI disclosures with verified 98%+ recall.
3. **Semantic Frame Extraction:** LLM judges evaluate candidate paragraphs to extract atomic semantic frames across four dimensions:
   - *Temporal Orientation:* Realized, Planned, Expected, or Hypothetical.
   - *Promotional Rhetoric:* Factual vs. promotional hype.
   - *Documentary Grounding:* Concrete verification via product names, business workflows, identified partners, dates, or quantitative metrics.
4. **Operational Activity Mapping:** Classifies verified implementations into operational actions, technological objects, deployment stages, and provider provenance (*proprietary* vs. *third-party*).
5. **Decoupling Index & Posture Archetypes:** Computes the firm-level washing score ($w$) tracking narrative intensity against substantive operational grounding, classifying firms into empirical posture archetypes: *Vocal Substantives*, *Governance-Led Disclosers*, and *Defensive Disclosers*.
6. **Market & Risk Econometrics:** Panel regressions and event studies evaluate market consequences: beta shifts, extreme negative return asymmetries (crash risk), and post-earnings announcement drift.

---

## 💾 Parquet Data Layers & Lineage Guarantees

All analytical scripts read tables exclusively through `scripts/common/layers.py` via `polars.scan_parquet()`.

```python
import layers as L

# Scan Silver AI semantic frames with pushdown predicates
frames = L.scan("silver.ai_frames").filter(L.pl.col("ticker") == "NVDA").collect()

# Read Gold annual firm panel
gold_firm_year = L.read_gold("firm_year", "firm_year")
```

### Layer Specifications

| Layer | Path | Description | Access Pattern |
|---|---|---|---|
| **Raw** | `data/raw/` | Gzipped original HTML filings, transcripts, and financial market tables. | Read-only |
| **Interim** | `data/interim/` | Append-only model predictions, embeddings, and judge JSONL streams. | Additive only |
| **Bronze** | `data/bronze/` | Cleaned, typed, source-level Parquet tables with cryptographic manifests. | Read through `layers.py` |
| **Silver** | `data/silver/` | S&P 500 filtered data, paragraph instances joined with LLM outputs. | Read through `layers.py` |
| **Gold** | `data/gold/` | Aggregated business panels (`firm_year`, `firm_quarter`, `activity`). | Read through `layers.py` |
| **Results** | `data/results/` | Final regression estimates, bootstrap replicates, and figure matrices. | Read by thesis & apps |

### Natural Lineage Keys
Lineage is strictly maintained across the entire transformation graph without surrogate keys:
$$\text{Filing Instance } (country, form, accession, item, paragraph\_idx) \longrightarrow text\_hash \longrightarrow frame\_id \longrightarrow activity\_id$$

> [!IMPORTANT]
> **Additive Output Invariance:** All GPU/LLM outputs (`data/interim/`) are immutable and append-only. Reprocessing scripts append new run parts rather than overwriting existing artifacts.

---

## 🖥️ Interactive Web Applications

The project includes two browser-based web applications under `apps/`:

### 1. Corporate AI Disclosure Explorer (`apps/explorer`)
Interactive analytics dashboard powered by an in-memory **DuckDB** instance and **Gradio**:
- **Visual Gold Explorer:** Tableau-style drag-and-drop shelf for exploring the curated Gold panels.
- **Company Profile Inspector:** Firm-level search showing posture archetypes, trajectory charts, and classified AI paragraphs with metadata tags.
- **Interactive SQL Console:** Ad-hoc querying directly over the Parquet layer.
- **Cloud-Ready:** Runs locally or deploys seamlessly to Hugging Face Spaces.

```bash
make explorer        # Launches on http://localhost:7860
```

### 2. Atomic Human Validator (`apps/validator`)
Keyboard-driven micro-annotation tool for auditing model extractions:
- Validates prefilter accuracy, semantic frame fidelity, and operational activity extraction.
- Single-keystroke input (`Y` = Yes, `N` = No, `U` = Uncertain) with instant persistence to `localStorage`.
- Comprehensive metric summarizer computing confusion matrices and stratum-reweighted precision/recall.

```bash
make validator       # Launches on http://localhost:8765
```

---

## 🚀 Getting Started & Replication

### 1. Environment Setup

The project uses [uv](https://github.com/astral-sh/uv) for fast, deterministic Python environment management:

```bash
# Clone the repository
git clone https://github.com/mrbungie/ai_disclosure_analysis.git
cd ai_disclosure_analysis

# Create and activate virtual environment
uv venv
source .venv/bin/activate

# Synchronize all dependencies
uv sync

# Configure environment variables (SEC User-Agent, Hugging Face token)
cp .env.example .env
```

---

### 2. Exact Replication Commands

Depending on whether you want to reproduce the econometric findings directly or rerun the full raw data acquisition and model pipeline, follow one of the tracks below:

#### Track A: Fast Reproduction of Empirical Findings & Thesis Manuscript (Recommended)
This track pulls the pre-computed, immutable Parquet layers (`gold/`, `silver/`, `bronze/`, `results/`) and pre-trained classification models from Hugging Face Hub, allowing you to reproduce all empirical regressions, event studies, figures, tables, and the final thesis manuscript in minutes without requiring expensive GPU/LLM inference or SEC bulk downloads:

```bash
# 1. Download structured datasets and trained models from Hugging Face
make hf-download

# 2. Verify unit tests and spine alignment
make test
make gold-check

# 3. Run all econometric estimations, event studies, and figure generation
make analytics

# 4. Compile the final thesis manuscript (Word docx and PDF)
python thesis_document/render.py

# 5. (Optional) Launch interactive visual explorer and SQL playground
make explorer
```

#### Track B: End-to-End Pipeline Execution from Raw Data
This track executes the full extraction, transformation, layer building, and analysis workflow from raw filings to final econometric results:

```bash
# 1. Fetch raw SEC filings and parse operational sections (Item 1, 1A, 7)
make collect-data          # 10-K panel: build universe, fetch HTML, extract items
make collect-data-10q      # 10-Q series: MD&A and Risk Factor updates
make collect-market        # Market daily prices, returns, and Fama-French factors

# 2. Build immutable Parquet lake (Bronze & Silver layers)
make bronze                # Manifests, raw paragraphs, text splitting, market
make silver                # S&P 500 universe filtering and paragraph-instance joins

# 3. Compile multi-grain analytical Gold panels
make gold                  # Assembles firm-year, firm-quarter, documents, and activities
make gold-check            # Asserts strict spine key alignment across all gold tables

# 4. Execute the complete econometric and analytics battery
make analytics             # Runs posture archetypes, washing index, event studies, DiD

# 5. Render the publication-ready thesis manuscript
python thesis_document/render.py
```

---

### 3. Core Make Commands Reference

The root `Makefile` orchestrates all stages of the research pipeline:

```bash
# Display help and available recipes
make help

# Run unit and integration tests
make test

# Ingestion & Extraction
make collect-data          # Fetch 10-Ks and extract Item 1/1A/7 sections
make collect-data-10q      # Fetch 10-Qs and extract MD&A / Risk updates
make collect-market        # Ingest market prices and Fama-French factors

# Build Parquet Data Layers
make bronze                # Rebuild data/bronze/ tables
make silver                # Rebuild data/silver/ universe tables
make layers                # Build both bronze and silver
make gold                  # Compile all gold panels (firm, year, quarter, activities)

# Run Econometrics & Analytics
make analytics             # Execute full econometric battery and output data/results/

# Launch Applications
make explorer              # Run visual data explorer
make validator             # Launch human validation tool

# Hugging Face & Cloud Sync
make hf-download           # Download dataset layers and models from Hugging Face
make hf-download-data      # Download only dataset layers
make hf-download-models    # Download only models
make hf-sync               # Synchronize local data layers and models to Hugging Face
make hf-sync-all           # Upload models, structured data, and raw archives
make b2-check              # Check consistency with Backblaze B2 backup bucket
```

---

## 🤗 Hugging Face Hub Integration

The project's structured dataset layers and trained machine learning models are hosted on the Hugging Face Hub:

| Resource | Hub Repository | Description |
|---|---|---|
| **📊 Dataset** | [`mrbungie/ai-disclosure-analysis`](https://huggingface.co/datasets/mrbungie/ai-disclosure-analysis) | Full Parquet lake (`gold/`, `silver/`, `bronze/`, `results/`, `archive/`, and raw compressed archives). |
| **🧠 Models** | [`mrbungie/ai-disclosure-analysis`](https://huggingface.co/mrbungie/ai-disclosure-analysis) | Trained prefilter classifiers, archetype cluster weights, and empirical shrinkage models. |

### Downloading Data & Models

If starting from a fresh environment or cloning without local data partitions, you can pull the required assets using any of the following methods:

#### Method 1: Using Make Recipes (Recommended)
```bash
# Download both models into models/ and dataset layers into data/
make hf-download

# Or download only specific targets:
make hf-download-data       # Pulls gold, silver, results, archive, bronze
make hf-download-models     # Pulls trained models and weights
```

#### Method 2: Using the Project CLI (`hf_sync.py`)
```bash
# Pull both models and data layers:
uv run python scripts/common/hf_sync.py --download

# Pull specific data layers (e.g. only gold panels and results):
uv run python scripts/common/hf_sync.py --download-data --layers gold results

# Pull models only:
uv run python scripts/common/hf_sync.py --download-models
```

#### Method 3: Using Python `huggingface_hub` or the Hugging Face CLI
```python
from huggingface_hub import snapshot_download

# Download dataset layers directly into data/
snapshot_download(
    repo_id="mrbungie/ai-disclosure-analysis",
    repo_type="dataset",
    local_dir="data",
    ignore_patterns=[".git*", "README.md"],
)

# Download models directly into models/
snapshot_download(
    repo_id="mrbungie/ai-disclosure-analysis",
    repo_type="model",
    local_dir="models",
    ignore_patterns=[".git*", "README.md"],
)
```

Or from the command line:
```bash
huggingface-cli download --repo-type dataset mrbungie/ai-disclosure-analysis --local-dir data
huggingface-cli download --repo-type model mrbungie/ai-disclosure-analysis --local-dir models
```

### Uploading & Synchronizing Changes

To push updated empirical results, newly generated gold panels, or retrained models back to Hugging Face:

```bash
# Concurrent sync of models and structured data (gold, silver, results, bronze):
make hf-sync

# Full synchronization including raw .tar.gz archives:
make hf-sync-all
```

*(Requires `HF_TOKEN` defined in your `.env` file or environment).*

---

## 📄 Thesis Document Generation

The complete thesis manuscript is maintained under `thesis_document/` in Quarto (`thesis.qmd`). It dynamically binds to `data/gold/` and `data/results/`, ensuring that all numbers, tables, and figures in the text match the underlying data.

To compile the manuscript into Word (`.docx`) and PDF (`.pdf`):

```bash
# Render the manuscript using the custom styling engine
python thesis_document/render.py
```

The resulting outputs are generated in `thesis_document/`:
- `GermanOviedo_FinalThesis.docx`
- `GermanOviedo_FinalThesis.pdf`

