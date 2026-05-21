# Engineering Design Document

## AI Disclosure Archetypes Pipeline

### *AI Washing or Credible Disclosure? Patterns and Clusters in Corporate AI Disclosures*

**Author:** Germán Oviedo
**Language:** Python
**Storage Philosophy:** File-first, incremental, reproducible
**Primary Formats:** Parquet, CSV, JSONL, DuckDB
**Scope:** SEC 10-K / 10-Q disclosure analysis (2021–2025)

---

# 1. System Objective

Build a scalable and reproducible NLP pipeline that:

1. Collects SEC corporate filings
2. Extracts AI-related disclosure content
3. Classifies disclosure characteristics
4. Constructs firm-level disclosure feature vectors
5. Identifies disclosure archetypes/clusters
6. Enables longitudinal and quasi-causal analysis

The architecture must:

- Minimize LLM token consumption
- Support incremental processing
- Be resumable and fault tolerant
- Allow partial downloads and staged experimentation
- Support reproducible academic research

---

# 2. Research-to-System Mapping

| Research Need | System Capability |
|---|---|
| Analyze AI disclosure behavior | AI disclosure extraction + classification |
| Compare firms | Firm-year feature aggregation |
| Detect disclosure styles | Clustering pipeline |
| Observe evolution over time | Panel dataset |
| Analyze SEC 2024 impact | Event-study / DiD-ready structure |
| Avoid excessive token costs | Rule-based prefiltering + staged LLM use |

---

# 3. Architectural Principles

## 3.1 File-First Architecture

The system operates primarily on Parquet, CSV, and JSONL. Avoid heavy database dependencies.

DuckDB is used only for:

- Fast analytical querying
- Local joins
- Aggregation
- Development convenience

No production-style OLTP database required.

## 3.2 Lazy and Incremental Processing

The pipeline must never assume the entire SEC universe is processed. Every stage must be resumable:

```
Universe → Manifest → Download Batch → Parsed Sections → AI Candidates → LLM Classification
```

## 3.3 Deterministic Preprocessing

Cheap deterministic filters should remove the majority of irrelevant text before LLM usage. LLMs should only process AI-related candidate chunks, ambiguous cases, and high-information windows.

## 3.4 Reproducibility

All outputs should be versionable and reproducible. Manifests, prompts, rule configurations, model versions, embeddings, and cluster assignments must be persisted.

---

# 4. High-Level Pipeline

```
Firm Universe
    ↓
Filing Manifest
    ↓
Selective Download
    ↓
Section Extraction
    ↓
AI Mention Prefiltering
    ↓
Candidate Chunking
    ↓
    ├── 07_score_with_rules.py (hard text rules: phrases/words/patterns, no LLM)
    └── 08_run_llm_classifier.py (direct prompt on chunk with pydantic_ai, outputs booleans)
    ↓
09_score_with_llm_booleans.py (aggregates LLM booleans + hard rules for further feature engineering)
    ↓
10_build_features.py (generates the final panel parquet for analysis)
    ↓
11_cluster_archetypes.py (clustering / archetypes)
    ↓
12_event_study_panel.py (event study / panel analysis)
```

---

# 5. Repository Structure

```
ai_disclosure_project/
│
├── data/
│   ├── raw/
│   │   ├── sec_submissions/
│   │   └── filings_html/
│   │
│   ├── interim/
│   │   ├── manifests/
│   │   ├── batches/
│   │   ├── sections/
│   │   ├── candidate_chunks/
│   │   ├── llm_batches/
│   │   └── llm_outputs/
│   │
│   └── processed/
│       ├── features/
│       ├── clusters/
│       ├── panels/
│       └── validation/
│
├── scripts/
│   ├── 00_build_firm_universe.py
│   ├── 01_build_filing_manifest.py
│   ├── 02_select_download_batch.py
│   ├── 03_download_selected_filings.py
│   ├── 04_extract_sections.py
│   ├── 05_prefilter_ai_mentions.py
│   ├── 06_chunk_candidates.py
│   ├── 07_score_with_rules.py
│   ├── 08_run_llm_classifier.py
│   ├── 09_score_with_llm_booleans.py
│   ├── 10_build_features.py
│   ├── 11_cluster_archetypes.py
│   └── 12_event_study_panel.py
│
├── prompts/
├── configs/
├── notebooks/
├── reports/
├── duckdb/
└── README.md
```

---

# 6. Core Data Model

## 6.1 Firm Universe

**File:** `firm_universe.parquet`

| Column | Type |
|---|---|
| ticker | string |
| cik | string |
| company_name | string |
| sic | string |
| industry_group | string |
| exchange | string |

## 6.2 Filing Manifest

The core orchestration table.

**File:** `filing_manifest.parquet`

| Column | Type |
|---|---|
| document_id | string |
| cik | string |
| ticker | string |
| form_type | string |
| filing_date | date |
| period_end_date | date |
| accession_number | string |
| sec_url | string |
| local_path | string |
| download_status | string |
| parse_status | string |
| prefilter_status | string |
| llm_status | string |
| priority_score | float |
| batch_id | string |
| created_at | timestamp |
| updated_at | timestamp |

## 6.3 Extracted Sections

**File:** `filing_sections.parquet`

| Column | Type |
|---|---|
| accession_number | string |
| ticker | string |
| filing_date | date |
| section_name | string |
| section_text | string |
| char_len | int |
| word_count | int |

## 6.4 AI Candidate Chunks

Only text that survives prefiltering.

**File:** `ai_candidate_chunks.parquet`

| Column | Type |
|---|---|
| chunk_id | string |
| accession_number | string |
| ticker | string |
| filing_date | date |
| section_name | string |
| chunk_text | string |
| ai_keyword_count | int |
| keyword_family | string |
| rule_score | float |
| text_hash | string |

## 6.5 Hard Rule Outputs

Deterministic binary flags extracted from text patterns (no LLM usage).

**File:** `ai_disclosure_rules.parquet`

| Column | Type | Description |
|---|---|---|
| chunk_id | string | Unique identifier for candidate chunk |
| has_board_oversight | bool | true if matching board oversight keywords |
| has_audit_committee | bool | true if matching audit committee keywords |
| has_vendor_nvidia | bool | true if Nvidia is mentioned |
| has_vendor_openai | bool | true if OpenAI is mentioned |
| has_vendor_microsoft | bool | true if Microsoft is mentioned |
| has_vendor_google | bool | true if Google is mentioned |
| has_vendor_deepseek | bool | true if DeepSeek is mentioned |
| has_metric_percentage | bool | true if percentage signs/figures are present |
| has_metric_dollar | bool | true if dollar signs/amounts are present |

## 6.6 LLM Classification Outputs

Structured binary flags extracted via LLMs using `pydantic_ai` directly on candidate chunks. Every output is represented by boolean columns, making the classification directly queryable with SQL/DuckDB.

**File:** `ai_disclosure_mentions.parquet`

| Column | Type | Description |
|---|---|---|
| chunk_id | string | Unique identifier for candidate chunk |
| is_ai_related | bool | true if text relates to artificial intelligence / machine learning |
| is_substantive | bool | true if the mention has concrete details vs boilerplate |
| is_promotional | bool | true if it focuses on generic/marketing claims |
| is_risk_related | bool | true if discussing risk factors or potential issues |
| is_governance_related | bool | true if mentioning policies, board, or oversight committees |
| mentions_copilot | bool | true if specifically mentioning Copilot, assistant tools |
| mentions_cloud | bool | true if mentioning cloud computing providers or infra |
| mentions_vendor | bool | true if mentioning third-party vendors or external models |
| mentions_training | bool | true if mentioning model training or dataset curation |
| is_financial_impact | bool | true if discussing revenues, costs, or financial returns of AI |
| rationale_short | string | Short rationale explaining the classification decision |

## 6.7 Combined Scored Chunks

Enriched features generated by combining hard rules and LLM booleans.

**File:** `ai_scored_chunks.parquet`

| Column | Type | Description |
|---|---|---|
| chunk_id | string | Unique identifier for candidate chunk |
| final_specificity | int | Specificity score (0-3) computed from combination of rules and LLM flags |
| final_governance_score | int | Governance score computed from board oversight rules and LLM flags |
| final_risk_score | int | Risk score computed from risk-related LLM flags and rules |
| final_promotional_score | int | Promotional score |
| has_ai_disclosure | bool | true if chunk has a valid substantive or risk AI disclosure |

## 6.8 Firm-Year Features

Aggregated unit for clustering.

**File:** `firm_year_features.parquet`

| Column | Type |
|---|---|
| ticker | string |
| year | int |
| industry_group | string |
| ai_mentions_count | int |
| avg_specificity | float |
| avg_operational_grounding | float |
| avg_promotional_score | float |
| avg_risk_score | float |
| avg_governance_score | float |
| share_promotional | float |
| share_substantive | float |
| share_governance | float |
| share_risk | float |
| post_sec_2024 | bool |
| post_deepseek | bool |

---

# 7. Pipeline Components

## 7.1 `00_build_firm_universe.py` — Universe Builder

**Purpose:** Define the research scope.

**Output:** `data/interim/firm_universe.parquet`

Responsibilities:

- Define target industries
- Resolve tickers and CIKs
- Create research scope

Start with **1–3 industries**, not all listed firms. Good candidates:

- Software / SaaS
- Semiconductors
- Financial services
- Healthcare technology

## 7.2 `01_build_filing_manifest.py` — Filing Manifest Builder

**Output:** `filing_manifest.parquet`

Responsibilities:

- Query SEC indexes for each firm's 10-K and 10-Q filings from 2021–2025
- Enumerate all candidate filings
- Create manifest table
- Avoid downloads initially

## 7.3 `02_select_download_batch.py` — Batch Selector

Responsibilities:

- Select subsets of filings for partial experimentation
- Prioritize filings

Example filters:

- only 10-K
- only large firms
- only software industry
- only 2023–2025

## 7.4 `03_download_selected_filings.py` — Filing Downloader

Responsibilities:

- Download raw filing HTML/text only once
- Persist locally
- Skip existing files
- Retry failures

Storage:

```
data/raw/filings_html/{cik}/{accession_number}.html
```

Rules:

- Skip if file exists
- Sleep between requests
- Store by accession number

## 7.5 `04_extract_sections.py` — Section Extractor

Extract relevant 10-K/10-Q sections. **Do not process the whole filing with LLMs.**

| Filing Area | Priority |
|---|---|
| Item 1. Business | High |
| Item 1A. Risk Factors | High |
| Item 7. MD&A | High |
| Item 7A. Market Risk | Medium |
| Item 9A. Controls and Procedures | Medium |

## 7.6 `05_prefilter_ai_mentions.py` — AI Prefiltering

**Goal:** Reduce token volume drastically with cheap keyword + regex filtering.

Start broad:

```python
AI_TERMS = [
    "artificial intelligence",
    "generative ai",
    "gen ai",
    "machine learning",
    "large language model",
    "llm",
    "deep learning",
    "natural language processing",
    "predictive analytics",
    "algorithmic",
    "automation",
    "computer vision",
    "neural network",
]
```

Avoid false positives:

```python
FALSE_POSITIVES = [
    "Adobe Illustrator",
    "appreciation",
    "said",
    "paid",
]
```

Example keyword families:

| Family | Terms |
|---|---|
| GenAI | generative ai, llm |
| ML | machine learning |
| Infra | gpu, model training |
| Governance | ai policy, oversight |
| Risk | hallucination, model risk |

Keep only paragraphs or windows around matches.

## 7.7 `06_chunk_candidates.py` — Chunking

Chunk around AI mentions, not arbitrary filing chunks.

Strategy:

```
previous paragraph
+ matched paragraph
+ next paragraph
```

Parameters:

- max chunk size: 1,500–2,500 tokens
- deduplicate near-identical chunks

Advantages:

- preserves local context
- minimizes tokens
- avoids irrelevant filing content

## 7.8 `07_score_with_rules.py` — Rule-Based Tagger (Hard Rules)

**Purpose**: Apply deterministic pattern-matching heuristics on the raw chunk text *before* and *independently* of any LLM calls. Chunks do NOT go to the LLM during this step.

**Phrase & Pattern Existence (Super Important)**:
This script relies heavily on high-performance regex and exact phrase matching to search for specific topics. Rather than querying the LLM for simple keyword detection, we extract features deterministically:
* **Specific Phrases**: e.g., *"board oversight"*, *"audit committee"*, *"data security policy"*, *"risk assessment"*.
* **Specific Vendors/Models**: e.g., *Microsoft, OpenAI, ChatGPT, Nvidia, Google, Claude, Gemini, DeepSeek*.
* **Numeric Metrics**: Presence of dollar amounts, percentages, or budget/financial figures.
* **Deployment Verbs**: *deployed, integrated, launched, rolling out, implemented*.

These rules produce queryable boolean columns in `ai_disclosure_rules.parquet`.

## 7.9 `08_run_llm_classifier.py` — LLM Extractor

**Purpose**: Execute structured semantic classification using LLM models via `pydantic_ai` directly on candidate chunks.

**Tooling & Schema**: Powered by `pydantic_ai` to enforce structured JSON outputs mapping to a rich set of boolean fields (see schema in [Section 6.6](#66-llm-classification-outputs)). The LLM prompt focuses solely on extracting boolean flags based on the text. This ensures all LLM classifications are stored as strongly-typed boolean columns in Parquet (`ai_disclosure_mentions.parquet`), making them directly queryable with SQL/DuckDB.

**Fault Tolerance, Job Queueing & Resumability**:
The script itself manages its own backlog dynamically:
1. On execution, it queries the database/Parquet file to identify candidate chunks that do not yet have LLM classification outputs. These are treated as outstanding jobs.
2. If an API call fails, JSON parsing fails, or the script is interrupted, the pipeline logs the error/telemetry and proceeds.
3. The classifier can be run iteratively (e.g., in a loop or scheduled cron) until the number of pending rows converges to exactly 0.

## 7.10 `09_score_with_llm_booleans.py` — Combined Scoring

**Purpose**: Aggregate and engineer downstream features by combining the outputs of the hard rules (`07_score_with_rules.py`) and the LLM booleans (`08_run_llm_classifier.py`).

**Responsibilities**:
- Read `ai_disclosure_rules.parquet` and `ai_disclosure_mentions.parquet` and join them on `chunk_id`.
- Implement logical combinations (e.g. if `is_ai_related` from LLM is True AND `has_phrase_board_oversight` from hard rules is True, then label as high-level governance).
- Compute composite specificity, risk, governance, and promotional scores.
- Output the combined labeled chunks to `ai_scored_chunks.parquet`.

## 7.11 `10_build_features.py` — Feature Builder

Transforms chunk-level outputs (`ai_scored_chunks.parquet`) into firm-quarter, firm-year, and industry-level feature vectors (`firm_year_features.parquet`).

## 7.12 `11_cluster_archetypes.py` — Clustering Pipeline

| Task | Methods |
|---|---|
| Dimensionality reduction | PCA, UMAP |
| Clustering | KMeans, HDBSCAN, GMM |

Expected archetypes:
- Operational adopters
- Promotional / AI marketers
- Risk-aware governers
- Silent / minimal disclosers
- Infrastructure-heavy AI firms
- Speculative AI narrators

## 7.13 `12_event_study_panel.py` — Panel / Event Study Pipeline

Supports:
- Longitudinal analysis
- Difference-in-differences (DiD)
- Pre/post SEC 2024 scrutiny
- Pre/post DeepSeek emergence
- Firm fixed effects
- Industry-year controls

Note: DiD only if treatment/control definition is defensible.

---

# 8. Token Optimization Strategy

## 8.1 Core Principle

**Never send full filings to LLMs. Never send full sections to LLMs. Only send AI-relevant windows.**

## 8.2 Filtering Cascade

```
Full Filing          ~100k–300k tokens
    ↓
Relevant Sections    ~30k–80k tokens
    ↓
AI Keyword Windows   ~500–5,000 tokens
    ↓
Deduplicated Chunks
    ↓
Rule Scoring
    ↓
LLM Classification   ~1–5% of original filing
```

## 8.3 Deduplication

Use simhash, MinHash, or SHA-256 hashing to remove:

- Repeated legal boilerplate
- Repeated quarterly language
- Duplicate paragraphs across filings

Cache everything using `hash(chunk_text)` as `text_hash` — never classify the same text twice.

## 8.4 Multi-Stage Inference

**Stage 1 — Prefiltered Candidates**: Filter paragraphs using the broad `05_prefilter_ai_mentions.py` regex checks.
**Stage 2 — Structured LLM (`pydantic_ai`)**: Query a cheap LLM to extract boolean semantic flags for each chunk. If parsing fails, the job remains pending in the database to be retried on subsequent pipeline iterations.
**Stage 3 — Heuristics & Pattern Matching (`08_score_with_rules.py`)**: Run deterministic regex rules and phrase-existence checks (specific vendors, board phrases, numeric indicators) directly over raw text and LLM JSON outputs to enrich the feature space.
**Stage 4 — Eventually Consistent Resolution**: Iterate classification runs until remaining jobs for LLM labeling drop to 0.

---

# 9. DuckDB Usage

DuckDB is optional but recommended for local analytics. No server required; fast Parquet querying; notebook-friendly.

Example query:

```sql
SELECT ticker,
       AVG(avg_specificity)
FROM firm_year_features
GROUP BY ticker;
```

---

# 10. Python Stack

| Purpose | Library |
|---|---|
| SEC access | sec-edgar-downloader |
| Parsing | beautifulsoup4, lxml |
| NLP | spacy |
| Dataframes | pandas, polars |
| Storage | pyarrow |
| Local analytics | duckdb |
| Embeddings | sentence-transformers |
| Clustering | scikit-learn |
| Dimensionality reduction | umap-learn |
| Validation | pydantic |

---

# 11. Initial MVP Scope

Build only the first milestone first:

**Scripts:**

```
00_build_firm_universe.py
01_build_filing_manifest.py
02_select_download_batch.py
03_download_selected_filings.py
04_extract_sections.py
05_prefilter_ai_mentions.py
06_chunk_candidates.py
```

**First useful dataset:** `ai_candidate_chunks.parquet`

**Scope:**

- 20–50 firms (start with 20)
- One industry
- 2022–2025
- Only 10-K

**Deliverables:**

- Filing manifest
- Downloaded filings
- Extracted AI chunks
- First disclosure labels
- First cluster visualization

Once you have `ai_candidate_chunks.parquet`, the thesis becomes much easier because you can inspect whether the signal is real **before** spending tokens.

---

# 12. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Excessive token costs | Aggressive prefiltering |
| Noisy AI mentions | Rule-based filtering |
| Boilerplate duplication | Deduplication hashing |
| Weak cluster separation | Feature engineering refinement |
| SEC parsing inconsistencies | Robust section fallback logic |
| Hallucinated LLM labels | Structured outputs + validation |

---

# 13. Expected Research Outputs

**Technical outputs:**

- Reusable SEC NLP pipeline
- Labeled disclosure dataset
- Disclosure archetype taxonomy

**Academic outputs:**

- Empirical analysis of AI disclosure behavior
- Evolution of disclosure strategies over time
- Evidence around AI washing patterns
- Governance/risk communication analysis

**Practical outputs:**

Framework applicable to investor relations, boards, compliance teams, and future "X-washing" analysis.

---

# 14. Observability, Logging, and Testing Infrastructure

## 14.1 Centralized Structured Logging System
The data pipeline implements a centralized logger to record execution telemetry at each pipeline stage.

* **Module**: [pipeline_logger.py](file:///Users/goviedb/Development/mib_tesis_ai_disclosure/scripts/pipeline_logger.py)
* **Log File**: `data/interim/manifests/pipeline_log.jsonl` (JSON Lines format)
* **Attributes**: `timestamp`, `pipeline_step`, `level`, `message`, `ticker`, `cik`, `accession_number`, `duration_seconds`, `details`.
* **DuckDB Integration**: The JSONL format is directly queryable via DuckDB's native JSON reader, e.g.:
  ```sql
  SELECT pipeline_step, level, COUNT(*) 
  FROM read_json_auto('data/interim/manifests/pipeline_log.jsonl') 
  GROUP BY 1, 2;
  ```

## 14.2 Unit Testing and Automation
A comprehensive test suite verifies regex and section boundaries, preventing regression as extraction rules evolve.

* **Test Suite**: [test_extraction_regex.py](file:///Users/goviedb/Development/mib_tesis_ai_disclosure/tests/test_extraction_regex.py) (uses Python `unittest` library).
* **Automation**: [Makefile](file:///Users/goviedb/Development/mib_tesis_ai_disclosure/Makefile) handles automation tasks (`make test`, `make install-deps`, `make run-pipeline`).
* **Covered Scenarios**:
  * SEC standard Item headers matching.
  * Pipe-wrapped Markdown table formats.
  * Honeywell plaintext styling fallbacks.
  * Bounds extraction and character length filtering (> 1000 chars filter).

## 14.3 Streamlit Diagnostics Workspace
The Streamlit app in [app.py](file:///Users/goviedb/Development/mib_tesis_ai_disclosure/app.py) exposes a central logging tab that parses logs and exposes an **Interactive SQL Console** using DuckDB, enabling real-time ad-hoc querying against runtime execution events.

