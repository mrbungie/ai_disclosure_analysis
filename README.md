# AI Washing or Credible Disclosure? Patterns and Clusters in Corporate AI Disclosures

Research pipeline analyzing how listed firms communicate about artificial intelligence in SEC 10-K filings (2021–2026), classifying disclosure archetypes, and measuring the gap between AI hype and operationally substantive AI communication ("AI washing").

**Author:** Germán Oviedo

---

## Research Questions

**Main:** How do listed firms communicate their policies, positions, and strategic posture toward AI in investor communications, and what distinct firm clusters emerge from these communication patterns?

**Specific:** Which clusters appear more operationally credible, governance-oriented, promotional, vague, or exposed to AI-washing risk — and did these patterns change after regulatory or market shocks (ChatGPT release, SEC AI-washing guidance, DeepSeek emergence)?

---

## Sample

- **115 US-listed firms** (of 120 in the initial universe; 5 excluded for incomplete filing histories) spanning **55 SIC industry groups**, covering both AI-intensive sectors (software, semiconductors) and traditional industrials, energy, healthcare, and finance.
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
  → 12 Build event-study / DiD panel          (event_study_panel.parquet, did_power_check.txt)
  → 13 Factor analysis (validate D1–D9)       (factor_loadings.csv, factor scores)
  → 14 D5 Governance sensitivity check        (governance_sensitivity.txt)
  → val_01 / val_02 Sample, LLM-judge-label, and validate pipeline scores
```

### 1. Collection and chunking (scripts 00–06)
Firm universe and filing manifest are built from SEC EDGAR; only 10-Ks are downloaded and cached locally by accession number. Business, Risk Factors, and MD&A sections are extracted with header-parsing rules (with fallbacks for non-standard filing typography). A broad AI-keyword regex prefilter (generative AI, LLM, machine learning, neural network, etc., with false-positive exclusions) flags candidate paragraphs, which are then chunked with surrounding context (previous + matched + next paragraph, deduplicated by hash) to keep only ~1–5% of the token volume of the original filings.

### 2. Feature extraction (scripts 07–09)
Two independent, then combined, sources of chunk-level features:

- **Bag-of-Words / presence features (07)** — deterministic, no LLM cost. Phased extraction: (1) boolean presence flags (vendors, deployment verbs, board oversight, risk categories, metrics), (2) word counts, (3) ratios (vague vs. specific word density), (4) composite scores (vagueness, specificity, risk/governance indicator counts).
- **LLM structured classification (08)** — `pydantic_ai` calls per chunk producing typed booleans (`is_substantive`, `is_promotional`, `is_risk_related`, `is_governance_related`, `mentions_training/vendor/cloud`, etc.), with resumable job queueing and backoff retry.
- **Combined scoring (09)** — branches on `--variant {rule_based,llm_full}` (default: `configs/config.json` `variants.active`). `rule_based` (today's `--bow-only` behavior) LEFT JOINs candidate chunks and BoW features, ignoring LLM outputs entirely so scoring is a fully deterministic, LLM-free pass that guarantees homogeneity across years. `llm_full` uses `ai_disclosure_mentions.parquet` with **no fallback to the BoW proxy**: chunks without a real LLM label are marked `llm_label_missing = TRUE` instead of being silently completed, and an aggregate coverage percentage is printed, logged, and written to `coverage_summary__llm_full.txt`. Scripts 09–14 and `val_01`/`val_02` all read/write under `data/processed/variant_{rule_based,llm_full}/`, isolated per variant — see `docs/plans/variant_selection_infrastructure.md`.

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

**Threshold provenance.** The programmatic cutoffs (`THRESHOLDS` in the script) were set by hand against the empirical dimension distribution, which is the most objectable design choice in the clustering step. The script therefore also writes `reports/cluster_threshold_justification__{variant}.txt`, reporting (1) each cutoff's empirical percentile rank within its dimension (e.g. "D5 ≥ 0.10 is the Nth percentile of D5 across 658 firm-years") and (2) a sensitivity check that shifts every threshold ±10% and reports the share of firm-years that change archetype — evidence that the archetypes aren't a knife-edge artifact of the exact cutoffs chosen.

### 6. Event-study / panel construction (script 12)
Firm-year features are merged with cluster assignments and enriched with event indicators/windows for three shocks — **ChatGPT release** (2022), **SEC AI-washing guidance** (2024), **DeepSeek emergence** (2025) — plus pre/post flags, event-relative year, firm fixed-effect IDs, and treatment-group flags defined from **2023 pre-period baselines** (high promotional tone, high technical specificity, high defensive framing, tech-sector membership) to avoid post-treatment contamination. Output: `event_study_panel.parquet`, ready for DiD where a treatment/control split is defensible.

**DiD power check.** With 115 firms across 55 SIC groups, industry × year × treatment cells shrink fast, and a single-flag treatment count can look adequate while the interacted cell actually used in a DiD spec (e.g. `high_promotional_pre2024 × tech_sector`) is underpowered. The script reports distinct-firm counts for every treatment cell used in the planned DiD specs (including interactions) to `reports/did_power_check__{variant}.txt`, flagging any cell below 15 firms — run this and check for warnings before reporting DiD results.

### 7. Exploratory factor analysis (script 13)
An empirical check on the hand-built D1–D9 constructs: BoW features are filtered by prevalence (1–99%), tested for factorability (Bartlett's sphericity, KMO), and the number of latent factors is selected via Horn's parallel analysis (100 random-data iterations, actual vs. random eigenvalues). A varimax-rotated maximum-likelihood factor model is then fit, producing loadings, scree plots, a loadings heatmap, and chunk/firm-year factor scores — used to assess whether the researcher-defined dimensions correspond to the data's actual latent structure.

### 8. AI Washing Index (analysis layer, notebooks)
```
Hype Score      = mean( z(share_promotional), z(avg_ratio_vague_words), z(avg_ratio_forward_to_realized) )
Substance Score = mean( z(avg_specificity), z(share_substantive), z(share_ai_quantified_claim),
                        z(share_named_deployment), z(share_dated_milestone) )

AI Washing Index      = Hype − Substance
AWI_norm (bounded)    = AI Washing Index / (|Hype| + |Substance| + ε)  ∈ (−1, 1)
```
Each score is the **mean** (not sum) of its z-scored components — deliberate, since Hype has 3 components and Substance has 5; summing instead of averaging would bias Substance's magnitude upward purely from having more terms. z-scores are computed **within industry_group × year cells** (`notebooks/ai_washing_by_cluster.ipynb`), not across the full panel, so the 2022–2024 AI hype wave itself is not mistaken for washing.

### 9. Validation (val_01 / val_02)
Because BoW proxies and LLM labels are both fallible, a third, independent check is layered on top: a stratified sample (by year × section-group × substantiveness) is labeled by an independent LLM judge — a `pydantic_ai` Agent against an OpenAI-compatible endpoint, configured via its own `LLM_JUDGE_*` env vars so the judge model is not the same one used for the pipeline's own classification. The judge currently runs on NVIDIA NIM (`LLM_JUDGE_MODEL=qwen/qwen3-next-80b-a3b-instruct`), deliberately a different model family from the pipeline classifier's `NVIDIA_MODEL=meta/llama-3.3-70b-instruct` so the judge isn't grading a close relative of itself.

`val_01` also guarantees positive-class coverage per dimension: `ensure_dimension_coverage()` checks `is_substantive`/`is_promotional`/`is_risk_related`/`is_governance_related` and force-tops-up any with fewer than 40 pipeline-flagged positives in the sample (`is_ai_related` is excluded — it's ~100% prevalent by construction). Governance is only ~1.4% prevalent pipeline-wide, so a plain stratified draw alone left it with a handful of positives and an unreliable recall estimate; this closes that gap generically rather than special-casing one dimension.

`val_01`/`val_02` only validate the **rule_based** variant (`--variant rule_based`, also the config default). `--variant llm_full` is rejected explicitly: judging `llm_full`'s output with another LLM would be circular, since both the classifier and the judge share the same failure modes. See `docs/plans/variant_selection_infrastructure.md` for the rule_based / llm_full variant selection design.

Current results (`reports/validation_summary__rule_based.txt`, n=236, deduped on `chunk_id` — see caveat below):

| Dimension | F1 | Precision | Recall | Accuracy |
|---|---|---|---|---|
| is_ai_related | 0.902 | 0.822 | 1.000 | 0.822 |
| is_substantive | 0.712 | 0.877 | 0.599 | 0.657 |
| is_promotional | 0.400 | 0.270 | 0.774 | 0.695 |
| is_risk_related | 0.749 | 0.761 | 0.737 | 0.801 |
| is_governance_related | 0.641 | 0.576 | 0.723 | 0.839 |

Specificity score vs. LLM-judged specificity: Spearman ρ = 0.301 (p < 0.001). These figures quantify where the deterministic pipeline proxies are reliable (AI relevance, risk, substantiveness) and where they are noisier (promotional tone, governance) — a required input for treating downstream washing claims as findings rather than assertions.

**`chunk_id` collision caveat.** `chunk_id = sha256(chunk_text)[:16]` (script 06) has no ticker/section/filing component, so verbatim-repeated boilerplate within one filing (the same paragraph appearing in both Item 1A and Item 7, sometimes up to 9×) shares an id across rows. `val_02` dedupes both the labeled sample and the scored chunks on `chunk_id` before merging so one labeled chunk isn't fanned out into several identical evaluation rows. This doesn't change P/R/F1 direction (label and pipeline score are identical across the duplicate rows) but a full fix would require giving `chunk_id` row identity (e.g. hashing ticker+filing_date+section+text) and regenerating the shared upstream `data/interim/candidate_chunks/*.parquet` artifacts — not done here, since 00–06 are explicitly out of scope for variant-selection changes (see `docs/plans/variant_selection_infrastructure.md`).

**Promotional/governance proxy fix (script 09, `build_proxy_sql`).** The original `is_promotional_proxy` fired on named-product mentions and deployment verbs ("launched", "copilot", "subscription") — i.e. "mentions a shipped AI product," not promotional *tone* — giving F1=0.277 (P=0.182, R=0.581) against the LLM judge on a 236-chunk validation sample. It's now `count_vague_words > count_negative_words AND NOT has_risk_factor AND bow_sentiment_score >= 0` (hype vocabulary outweighing negative-tone words, outside risk-factor language, non-negative overall sentiment), which measures F1=0.400 (P=0.270, R=0.774) on the same sample. `is_governance_related_proxy` gained a `has_compliance AND has_word_ai` clause (previously only board-oversight/audit-committee/ethics-policy), moving recall from 0.333 → 0.723 and F1 from 0.500 → 0.641. Both formulas were selected by grid-searching candidate boolean combinations of existing BoW features against this same 236-chunk validation set — an in-sample tuning caveat: report these numbers as "fit to this sample," not as an independent generalization estimate, and prefer a fresh/held-out validation draw before treating them as final.

**Governance recall follow-up (script 14).** `scripts/14_governance_sensitivity.py` reruns the programmatic archetype classifier with D5 built from the BoW proxy alone vs. the LLM-only signal alone (both already present in `firm_year_features__{variant}.parquet`), reporting the share of firm-years that change archetype and any shift in "Governance & Compliance-Focused" cluster size (`reports/governance_sensitivity__{variant}.txt`). On `rule_based`, `share_llm_is_governance` is ~0 by design (that variant's script 09 branch ignores the LLM merge entirely), so this check is more informative run on `llm_full`. Run before finalizing archetypes or the DiD; if churn is material, report both variants explicitly.

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
make run-scoring ARGS="--variant rule_based"        # script 09: combined/deterministic scoring (or --variant llm_full)
make run-features ARGS="--variant rule_based"       # script 10: firm-year panel
make run-clustering ARGS="--variant rule_based"     # script 11: programmatic archetypes
make run-event-study ARGS="--variant rule_based"    # script 12: DiD panel + power check
make run-factor-analysis ARGS="--variant rule_based" # script 13: exploratory factor analysis
make run-governance-sensitivity ARGS="--variant llm_full" # script 14: D5 BoW-only vs LLM-only sensitivity (llm_full most informative)
make run-validate ARGS="--n 150 --variant rule_based" # val_01 + val_02: LLM-judge validation (rule_based only, by design)
make run-full-pipeline                              # 05 → 11 end to end
```

## Logging & Diagnostics

Centralized structured logging (`scripts/pipeline_logger.py`) writes JSONL events to `data/interim/manifests/pipeline_log.jsonl` (timestamp, pipeline_step, level, message, ticker, cik, accession_number, duration_seconds, details), directly queryable via DuckDB.

## Streamlit Dashboard

```bash
streamlit run app.py
```

Tabs: Configuration, Universe Management, Discovery & Manifests, Candidate Chunks, Post-Pipeline (scripts 07–10, variant-aware — reads/writes under the sidebar's active `rule_based`/`llm_full` selection), Pipeline Logs (with an interactive DuckDB SQL console, filterable by variant), and Project Architecture.

## Unit Testing

`tests/` covers section-extraction regex robustness (standard/pipe-wrapped/plaintext-fallback headers, boundary checks, minimum-length filtering) and BoW feature extraction correctness.
