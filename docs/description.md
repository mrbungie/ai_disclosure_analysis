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
Rule-Based Scoring
    ↓
Cheap LLM Classification
    ↓
Feature Aggregation
    ↓
Clustering / Panel Construction
    ↓
Event Study / DiD Analysis
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
│   ├── 08_run_cheap_llm_classifier.py
│   ├── 09_validate_llm_outputs.py
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

## 6.5 LLM Classification Outputs

LLM/rule-labeled chunks.

**File:** `ai_disclosure_mentions.parquet`

| Column | Type |
|---|---|
| chunk_id | string |
| is_ai_related | bool |
| is_substantive | bool |
| is_promotional | bool |
| is_risk_related | bool |
| is_governance_related | bool |
| specificity_score | int |
| operational_grounding | int |
| promotional_score | int |
| governance_score | int |
| risk_score | int |
| confidence | float |
| rationale_short | string |

## 6.6 Firm-Year Features

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

## 7.8 `07_score_with_rules.py` — Rule-Based Scoring

**Purpose:** Avoid unnecessary LLM calls. Only send ambiguous/high-value chunks to the LLM.

**Specificity signals (positive):**

- Named products or platforms
- Dollar amounts
- Customer segment references
- Named business process
- Named model/vendor
- Deployment verbs: *implemented, deployed, integrated, launched*

**Promotional signals:**

- Words like *transform, revolutionize, leading, cutting-edge, unlock, next-generation*
- No concrete use case nearby

**Governance/risk signals:**

- Words like *board, oversight, policy, controls, privacy, cybersecurity, model risk*

## 7.9 `08_run_cheap_llm_classifier.py` — LLM Classification

Use a cheap model for structured classification. Use cheap models first; escalate only uncertain cases.

Prompt should return only JSON:

```json
{
  "is_ai_related": true,
  "disclosure_type": "operational | promotional | risk | governance | financial | unclear",
  "specificity_score": 1,
  "operational_grounding": 1,
  "promotional_score": 1,
  "risk_score": 1,
  "governance_score": 1,
  "confidence": 0.82,
  "rationale_short": "Mentions AI-enabled customer support but gives no implementation details."
}
```

## 7.10 `09_validate_llm_outputs.py` — Validation Layer

Responsibilities:

- Schema validation
- Malformed JSON detection
- Retry invalid outputs
- Manual sample auditing

Recommended: manual labeling benchmark of **100–200 chunks**, comparing LLM labels vs. your own to calculate agreement.

## 7.11 `10_build_features.py` — Feature Builder

Transforms chunk-level outputs into firm-quarter, firm-year, and industry-level feature vectors.

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

**Stage 1 — Rules only:** Is this probably AI-related?

**Stage 2 — Cheap LLM:** Classify dimensions.

**Stage 3 — Better model:** Only for uncertain/low-confidence cases.

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

