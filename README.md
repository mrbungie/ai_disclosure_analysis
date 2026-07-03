# AI Washing or Credible Disclosure? Patterns and Clusters in Corporate AI Disclosures

Research pipeline analyzing how listed firms communicate about artificial intelligence in SEC 10-K filings (2021–2026), classifying disclosure archetypes, and measuring the gap between AI hype and operationally substantive AI communication ("AI washing").

**Author:** Germán Oviedo

---

## Research Questions

**Main:** How do listed firms communicate their policies, positions, and strategic posture toward AI in investor communications, and what distinct firm clusters emerge from these communication patterns?

**Specific:** Which clusters appear more operationally credible, governance-oriented, promotional, vague, or exposed to AI-washing risk — and did these patterns change after regulatory or market shocks (ChatGPT release, SEC AI-washing guidance, DeepSeek emergence)?

---

## Sample

- **115 US-listed firms** (of 120 in the initial universe; 5 excluded for incomplete filing histories) spanning **54 SIC industry groups**, covering both AI-intensive sectors (software, semiconductors) and traditional industrials, energy, healthcare, and finance.
- **Annual 10-K filings, fiscal years 2021–2026** (2026 is a partial year, filings submitted through May 2026), yielding **658 firm-year observations**.
- **4,204 AI-related candidate chunks** extracted, of which 43% fall in Risk Factors (Item 1A), 42% in Business (Item 1), and 16% in MD&A (Item 7).
- 70% of firm-years have at least one AI mention. Adoption rose from 49% of firms disclosing AI in 2021 to 97% in 2026, with the largest single-year jump (+28pp) between 2023 and 2024 — coinciding with the SEC's AI-washing guidance and the broader GenAI hype wave.
- Scope limitations: 10-K only (no 8-K/10-Q/proxy/earnings-call text), US-listed firms only, text-based (no market/financial outcome variables in the pipeline yet), and a sample size that limits power for fine-grained subgroup cells.

---

## Methodology / Data Pipeline

The pipeline is a file-first, incremental system (Parquet/JSONL, no production database) designed to minimize LLM token spend by aggressively prefiltering before any model call, and to be resumable at every stage.

```
SEC EDGAR
  → 00 Build firm universe                  (firm_universe.parquet)
  → 01 Build filing manifest                (filing_manifest.parquet)
  → 02 Select download batch
  → 03 Download filings                     (filings_html/)
  → 04 Extract sections (Business/Risk/MD&A) (filing_sections.parquet)
  → 05 Prefilter AI mentions (keyword/regex) (manifest updated)
  → 06 Chunk candidates                      (ai_candidate_chunks.parquet)
  → 07 Extract BoW / presence features       (ai_disclosure_bow_features.parquet)
  → 08 LLM structured classification         (ai_disclosure_mentions.parquet)  [optional]
  → 09 Combine BoW + LLM (LEFT JOIN, proxies)(ai_scored_chunks.parquet)
  → 10 Build firm-year panel                 (firm_year_features.parquet)
  → 11 Cluster into archetypes                (firm_year_clusters.parquet)
  → 12 Build event-study / DiD panel          (event_study_panel.parquet)
  → 13 Factor analysis (validate D1–D9)       (factor_loadings.csv, factor scores)
  → val_01 / val_02 Sample, hand/LLM-label, and validate pipeline scores
```

### 1. Collection and chunking (scripts 00–06)
Firm universe and filing manifest are built from SEC EDGAR; only 10-Ks are downloaded and cached locally by accession number. Business, Risk Factors, and MD&A sections are extracted with header-parsing rules (with fallbacks for non-standard filing typography). A broad AI-keyword regex prefilter (generative AI, LLM, machine learning, neural network, etc., with false-positive exclusions) flags candidate paragraphs, which are then chunked with surrounding context (previous + matched + next paragraph, deduplicated by hash) to keep only ~1–5% of the token volume of the original filings.

### 2. Feature extraction (scripts 07–09)
Two independent, then combined, sources of chunk-level features:

- **Bag-of-Words / presence features (07)** — deterministic, no LLM cost. Phased extraction: (1) boolean presence flags (vendors, deployment verbs, board oversight, risk categories, metrics), (2) word counts, (3) ratios (vague vs. specific word density), (4) composite scores (vagueness, specificity, risk/governance indicator counts).
- **LLM structured classification (08)** — `pydantic_ai` calls per chunk producing typed booleans (`is_substantive`, `is_promotional`, `is_risk_related`, `is_governance_related`, `mentions_training/vendor/cloud`, etc.), with resumable job queueing and backoff retry.
- **Combined scoring (09)** — LEFT JOINs candidate chunks and BoW features with LLM outputs so no chunk is ever dropped for lacking an LLM label; missing LLM values fall back to regex-based proxies. A `--bow-only` flag forces a fully deterministic, LLM-free scoring pass to guarantee homogeneity across years (avoiding artifacts from incomplete/uneven LLM coverage).

### 3. Firm-year panel (script 10)
Chunk-level scores are aggregated to firm-year observations (`firm_year_features.parquet`): mention counts, average specificity/governance/risk/promotional scores, section-location shares (`sec_pct_business/mda/risk_factors`), and per-topic `share_*` features used as the raw material for the dimensional constructs below.

### 4. Dimensional constructs (D1–D9)
Nine composite dimensions are built from the firm-year `share_*` features (min-max scaled, D1 log-transformed before scaling):

| Dim | Construct | Captures |
|---|---|---|
| D1 | AI Intensity | Volume of AI discussion |
| D2 | Operational Embeddedness | Concrete AI use in products/processes |
| D3 | Technical Specificity | Technical depth vs. vague claims |
| D4 | Quantification | Numbers/KPIs/$ attached to AI claims |
| D5 | Governance Maturity | Formal AI governance structures |
| D6 | Risk Disclosure Depth | Breadth/specificity of AI risk disclosure |
| D7 | Promotional Tone | Buzzword/hype density |
| D8 | Competitive/Defensive Exposure | AI framed as external threat |
| D9 | Section Location | Substantive (Business/MD&A) vs. boilerplate (Risk Factors) placement |

Each dimension has a BoW version (deterministic) and, where available, an LLM equivalent; the pipeline runs both and compares (see Validation below).

### 5. Clustering into archetypes (script 11)
D2–D9 feed the clustering feature vector (D1 is excluded — volume dominates distance metrics and would split clusters on verbosity rather than communication strategy; it is used post-hoc to characterize clusters). Two modes are implemented:

- **Programmatic** (default): rule-based assignment on dimension thresholds, producing five named archetypes — *Non-AI Disclosers*, *Governance & Compliance-Focused*, *Boilerplate Risk-Warners*, *Full-Stack AI Pioneers*, *Operational Application Adopters*.
- **Unsupervised**: Ward hierarchical clustering (k=5) on standardized dimensions, as a data-driven robustness check.

GMM (BIC-selected k) and HDBSCAN are also fit for schema compatibility/robustness comparison, and a 2D UMAP projection is stored for visualization. Output: `firm_year_clusters.parquet`.

### 6. Event-study / panel construction (script 12)
Firm-year features are merged with cluster assignments and enriched with event indicators/windows for three shocks — **ChatGPT release** (2022), **SEC AI-washing guidance** (2024), **DeepSeek emergence** (2025) — plus pre/post flags, event-relative year, firm fixed-effect IDs, and treatment-group flags defined from **2023 pre-period baselines** (high promotional tone, high technical specificity, high defensive framing, tech-sector membership) to avoid post-treatment contamination. Output: `event_study_panel.parquet`, ready for DiD where a treatment/control split is defensible.

### 7. Exploratory factor analysis (script 13)
An empirical check on the hand-built D1–D9 constructs: BoW features are filtered by prevalence (1–99%), tested for factorability (Bartlett's sphericity, KMO), and the number of latent factors is selected via Horn's parallel analysis (100 random-data iterations, actual vs. random eigenvalues). A varimax-rotated maximum-likelihood factor model is then fit, producing loadings, scree plots, a loadings heatmap, and chunk/firm-year factor scores — used to assess whether the researcher-defined dimensions correspond to the data's actual latent structure.

### 8. AI Washing Index (analysis layer, notebooks)
```
Hype Score      = z(share_promotional) + z(ratio_vague_words) + z(ratio_forward_to_realized)
Substance Score = z(avg_specificity) + z(share_ai_quantified_claim) + z(share_named_deployment)
                + z(share_dated_milestone) + z(share_substantive)   [LLM version only]

AI Washing Index      = Hype − Substance
AWI_norm (bounded)    = Hype / (|Hype| + |Substance| + ε)  ∈ (−1, 1)
```
z-scores are computed **within industry_group × year cells**, not across the full panel, so the 2022–2024 AI hype wave itself is not mistaken for washing.

### 9. Validation (val_01 / val_02)
Because BoW proxies and LLM labels are both fallible, a third, independent check is layered on top: a stratified sample of ~150–170 chunks (by year × section-group × substantiveness) is labeled by an LLM judge (Claude, via the Agent SDK) acting as approximate ground truth, then compared against the pipeline's own BoW/LLM-combined scores. Current results (`reports/validation_summary.txt`, n=167):

| Dimension | F1 | Precision | Recall | Accuracy |
|---|---|---|---|---|
| is_ai_related | 0.860 | 0.754 | 1.000 | 0.754 |
| is_substantive | 0.617 | 0.556 | 0.694 | 0.814 |
| is_promotional | 0.566 | 0.627 | 0.516 | 0.707 |
| is_risk_related | 0.683 | 0.641 | 0.732 | 0.772 |
| is_governance_related | 0.500 | 1.000 | 0.333 | 0.988 |

Specificity score vs. LLM-judged specificity: Spearman ρ = 0.399 (p < 0.001). These figures quantify where the deterministic/LLM pipeline proxies are reliable (AI relevance, risk) and where they are noisier (governance recall, promotional tone) — a required input for treating downstream washing claims as findings rather than assertions.

---

## Analysis Notebooks

| Notebook | Purpose |
|---|---|
| `clustering_archetypes.ipynb` | Build dimension scores, cluster firms, name archetypes |
| `ai_washing_by_cluster.ipynb` | AI Washing Index trend by cluster over time |
| `temporal_analysis.ipynb` | Cluster evolution, event plots, alluvial firm migration |
| `positioning_map.ipynb` | Specificity vs. promotional-tone scatter, section-location analysis |
| `ai_washing_analysis.ipynb` | Original AI-washing analysis (legacy, superseded by the above) |

Generated figures live in `reports/`; the fuller research/thesis plan (chapter structure, expected findings, validation and robustness plan) is in `docs/thesis_plan.md`, and the engineering design document is in `docs/description.md`.

---

## Setup & Installation

Uses `uv` for Python environment and dependency management.

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Makefile Automation

```bash
make test                                          # unit tests
make install-deps                                  # install pytest
make run-pipeline                                   # prefilter + chunk candidates
make run-bow                                        # script 07: BoW feature extraction
make run-llm ARGS="--max-jobs 20 --concurrency 1 --delay 2.0"   # script 08: LLM classification
make run-scoring ARGS="--bow-only"                  # script 09: combined/deterministic scoring
make run-features                                   # script 10: firm-year panel
make run-full-pipeline                              # 05 → 10 end to end
```

## Logging & Diagnostics

Centralized structured logging (`scripts/pipeline_logger.py`) writes JSONL events to `data/interim/manifests/pipeline_log.jsonl` (timestamp, pipeline_step, level, message, ticker, cik, accession_number, duration_seconds, details), directly queryable via DuckDB.

## Streamlit Dashboard

```bash
streamlit run app.py
```

Tabs: Configuration, Universe Management, Discovery & Manifests, Candidate Chunks, Pipeline Logs (with an interactive DuckDB SQL console), and Project Architecture.

## Unit Testing

`tests/` covers section-extraction regex robustness (standard/pipe-wrapped/plaintext-fallback headers, boundary checks, minimum-length filtering) and BoW feature extraction correctness.
