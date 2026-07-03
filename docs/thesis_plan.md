# Thesis Plan — AI Communication Strategies in Investor Filings

**Working title:** *How Listed Firms Position Themselves Around AI in Investor Communications: Evidence from SEC Filings 2021–2025*

**Author:** Germán Oviedo  
**Supervisor feedback:** Broaden from "AI washing" as the central question to a wider study of AI communication strategies, with AI washing / credible disclosure as one specific angle inside the broader analysis.

---

## Research Questions

**Main research question:**
> How do listed firms communicate their policies, positions, and strategic posture toward AI in investor communications, and what distinct firm clusters emerge from these communication patterns?

**Specific research question:**
> Which clusters appear more operationally credible, governance-oriented, promotional, vague, or exposed to AI-washing risk — and did these patterns change after regulatory or market shocks?

---

## Empirical Chapter Structure

### Chapter 1 — AI Communication Intensity Over Time

*How much do firms talk about AI, and did this change after ChatGPT and SEC scrutiny?*

**Analyses:**
- Total AI mentions per year across the full sample
- Share of firms disclosing AI (at least one chunk) per year
- Mean AI mention count by industry group over time
- Breakdown by filing section: Business / Risk Factors / MD&A

**Key plots:**
1. Line chart: mean `ai_mentions_count` per year, overall and by industry
2. Bar chart: % of firms mentioning AI per year (adoption diffusion)
3. Stacked bar: share of AI chunks per section type per year

**Expected finding:** AI communication increased sharply after 2022 (ChatGPT), but the increase was uneven across industries and communication styles.

---

### Chapter 2 — AI Communication Dimensions

*What axes of variation structure how firms talk about AI?*

Build composite **dimension scores** from existing BoW features. These are the inputs to the clustering. Each score is the mean of its constituent `share_*` features, **winsorized at the 1st/99th percentile then z-scored** (not min-max — min-max is dominated by outliers and makes the scale uninterpretable across years).

| # | Dimension | What it captures | Key constituent features |
|---|---|---|---|
| D1 | **AI Intensity** | Volume of AI discussion | `ai_mentions_count`, `share_word_ai`, `share_word_artificial_intelligence` |
| D2 | **Operational Embeddedness** | Concrete AI use in products/processes | `share_deployment_verb`, `share_specific_product`, `share_customer_facing`, `share_product_integration`, `share_internal_productivity` |
| D3 | **Technical Specificity** | Technical depth vs. vague claims | `share_model_training`, `share_compute_infra`, `share_gen_ai_mention`, `share_classical_ml_mention`, `share_word_gpu`, `share_word_fine_tuning`, `share_word_foundation_model` |
| D4 | **Quantification** | Numbers/KPIs/dollars attached to AI claims | `share_metric_dollar`, `share_metric_percentage`, `share_capex_mention`, `share_rd_mention`, `share_revenue_impact` |
| D5 | **Governance Maturity** | Formal AI governance structures | `share_board_oversight`, `share_ethics_policy`, `share_compliance`, `share_word_responsible_ai`, `share_word_gdpr`, `share_word_eu_ai_act` |
| D6 | **Risk Disclosure Depth** | Breadth and specificity of risk disclosure | `share_risk_factor`, `share_regulatory_risk`, `share_cyber_privacy_risk`, `share_ethics_bias_risk`, `share_ip_copyright_risk` |
| D7 | **Promotional Tone** | Buzzword density, hype language | `share_promotional`, `share_word_transformative`, `share_word_revolutionary`, `share_word_seamless`, `share_word_empower`, `share_word_cutting_edge` |
| D8 | **Competitive/Defensive Exposure** | AI framed as external threat, not internal capability | `share_competitor_mention`, `share_word_disruption`, `share_word_threat`, `share_word_adversely` |
| D9 | **Section Location** | Where AI appears: substantive (Business/MD&A) vs boilerplate (Risk Factors) | Computed from `ai_candidate_chunks.parquet` section distribution |

**BoW vs LLM versions:**
Each dimension has a BoW version (deterministic regex/presence features, no API cost) and, where available, an LLM version (structured boolean outputs from `08_run_llm_classifier.py`). The thesis runs both and compares. Global flags in the analysis notebooks switch between versions.

| Dimension | BoW proxy | LLM equivalent |
|---|---|---|
| D2 Operational | `share_deployment_verb`, `share_specific_product`, `share_customer_facing` | `is_substantive`, `mentions_cloud`, `mentions_vendor` |
| D3 Technical | `share_model_training`, `share_compute_infra`, `share_word_gpu` | `mentions_training` |
| D5 Governance | `share_board_oversight`, `share_ethics_policy`, `share_compliance` | `is_governance_related` |
| D6 Risk | `share_risk_factor`, `share_cyber_privacy_risk`, `share_regulatory_risk` | `is_risk_related` |
| D7 Promotional | `share_promotional`, `share_word_transformative` | `is_promotional` |

**Missing features to add in `10_build_features.py`:**
- `share_llm_is_substantive` — share of chunks classified as substantive by LLM
- `share_llm_is_governance` — share of chunks classified as governance-related by LLM
- `share_llm_is_risk` — share of chunks classified as risk-related by LLM
- `share_llm_is_promotional` — share of chunks classified as promotional by LLM
- `share_llm_mentions_training` — share of chunks mentioning model training per LLM
- `share_llm_mentions_vendor` — share of chunks mentioning third-party vendors per LLM
- `share_llm_mentions_cloud` — share of chunks mentioning cloud infra per LLM
- `share_llm_is_ai_related` — confirmation share (filters false positives)
- `share_llm_is_financial_impact` — share of chunks discussing financial AI impact
- `sec_pct_business` — % of AI chunks in Business section (from chunks, not LLM)
- `sec_pct_mda` — % of AI chunks in MD&A section
- `sec_pct_risk_factors` — % of AI chunks in Risk Factors section

**Key plot:**
- Scatter: D3 Technical Specificity (x) vs D7 Promotional Tone (y), coloured by cluster, sized by D1 Intensity
  → Separates credible adopters (high specificity, low hype) from promotional narrators (low specificity, high hype)

---

### Chapter 3 — AI Communication Archetypes (Clustering)

*What distinct firm-level communication strategies emerge from the data?*

**Method:** Ward hierarchical clustering on **D2–D9** (D1 AI Intensity excluded from the clustering feature vector — volume dominates distance metrics and would split clusters on verbosity rather than strategy; D1 used post-hoc to characterize cluster profiles). GMM and HDBSCAN as robustness checks. k selected by silhouette score + dendrogram visual inspection. Target 4–6 clusters.

**Target archetypes (post-hoc labelling after clustering):**

| Archetype | Expected dimension profile |
|---|---|
| **AI-Core Strategists** | Very high D1 + D3 + D4; AI is central to business model |
| **Concrete AI Operators** | High D2 + D3 + D4; moderate D6; high D9-substantive |
| **Governance-Mature Disclosers** | High D5 + D6; moderate D2; post-2024 skew |
| **Promotional AI Claimants** | High D1 + D7; low D2 + D3 + D4 |
| **Boilerplate Risk Disclosers** | High D9-risk-section + D6; low D2 + D3; mostly Risk Factors |
| **AI-Exposed / Defensive** | High D8; low D2 + D3; AI as competitive threat |
| **Minimal / Silent** | Low on all dimensions |

**Key plots:**
1. Cluster dimension heatmap (z-scores)
2. Radar charts per cluster
3. UMAP scatter coloured by cluster
4. Cluster share by industry (heatmap)
5. Stacked area chart: cluster share by year
6. Alluvial / Sankey: firm migration between clusters across years

---

### Chapter 4 — Temporal Evolution of AI Communication

*How did firm communication strategies change from 2021 to 2025?*

**Analyses:**
- Cluster share over time (stacked area)
- Dimension score trajectories per cluster
- Pre/post ChatGPT (2022) shift: D1, D7
- Pre/post SEC guidance (2024) shift: D5, D6, D7
- Pre/post DeepSeek (2025) shift: D3, D4, D8

**Key plots:**
1. Stacked area: archetype share by year with event markers
2. Line chart: mean D5 Governance and D6 Risk per year by cluster (SEC effect)
3. Bar: dimension shift (post − pre) per cluster for each event
4. Firm-level alluvial: which clusters gained/lost members post-2024

**Expected finding:** After SEC 2024 guidance, firms likely shifted toward more governance/risk language (D5, D6 increase) without necessarily increasing operational specificity (D2, D3). This would support the hypothesis that some firms responded with more boilerplate rather than more credible disclosure.

---

### Chapter 5 — AI Washing / Credible Disclosure Analysis

*Which firms and clusters show the largest gap between AI claims and supporting evidence?*

**AI Disclosure Credibility Gap (index):**
```
AI Washing Index = Hype Score − Substance Score

Hype Score     = z_iy(share_promotional)
               + z_iy(ratio_vague_words)
               + z_iy(ratio_forward_to_realized)   ← promise-heavy language

Substance Score = z_iy(avg_specificity)
                + z_iy(share_ai_quantified_claim)  ← AI term + metric co-occurrence
                + z_iy(share_named_deployment)      ← named model + deploy verb
                + z_iy(share_dated_milestone)        ← time-anchored AI commitments
                + z_iy(share_substantive)           ← LLM version only

where z_iy(x) = z-score within (industry_group × year) cell,
      not across the full panel (avoids encoding the 2022-2024 AI hype wave as washing)

Normalized form (interpretable bounded version):
  AWI_norm = Hype / (|Hype| + |Substance| + ε)  ∈ (−1, 1)
```

**Key changes vs prior formula:**
- Removed `share_gen_ai_mention` from Hype — GenAI topic ≠ hype; penalizes legitimate adopters
- Replaced `share_capex_mention` / `share_rd_mention` (presence of the words) with sentence-level co-occurrence features that require AI + metric in the same sentence
- Z-scoring within industry × year rather than the full panel prevents temporal confounding
- Added `ratio_forward_to_realized` as a direct promise-vs-evidence signal

*BoW version uses the above. LLM version substitutes `is_promotional` and `is_substantive` from `ai_disclosure_mentions.parquet`.*

**Key plots:**
1. Scatter: Hype (x) vs Substance (y), coloured by cluster, key firms labelled
2. AI Washing Index trend per cluster over time
3. Pre/post SEC shift in washing index by cluster
4. Industry-level mean washing index ranking
5. Firm-level credibility gap ranking (top/bottom 15)

**Expected finding:** Promotional AI Claimants and Boilerplate Risk Disclosers will have persistently high washing index. AI-Core Strategists and Concrete AI Operators will have low or negative index (substance exceeds hype). The SEC 2024 effect may reduce the index for some clusters but not others.

---

### Chapter 6 — Section-Location Analysis

*Is AI discussed as strategy, operations, risk, or legal boilerplate?*

**Analyses:**
- Distribution of AI chunks by section per cluster
- Shift in section distribution pre/post SEC 2024
- Section profile vs. AI washing index

**Key plots:**
1. Stacked bar: % of AI chunks per section by cluster
2. Line: share of AI chunks in Risk Factors vs Business/MD&A over time
3. Scatter: `sec_pct_risk_factors` (x) vs AI Washing Index (y) — tests whether boilerplate filers also have higher washing index

---

### Chapter 7 — Shock / Event Analysis

*Did ChatGPT, SEC scrutiny, and DeepSeek change how firms communicate about AI?*

| Event | Date | Hypothesis |
|---|---|---|
| ChatGPT release | Nov 2022 | D1 Intensity and D7 Promotional spike |
| SEC AI-washing guidance | Mar 2024 | D5 Governance and D6 Risk increase; D7 Promotional decreases |
| DeepSeek emergence | Jan 2025 | D8 Defensive increases; D3 Technical shifts toward cost/efficiency language; D4 Quantification increases for infra-heavy firms |

**Method:** Simple pre/post mean comparison with visual inspection. DiD if treatment/control groups are defensible (e.g., firms with high D7 in 2023 as treatment group for SEC effect).

---

## Sample Description

### Universe and Selection
The sample covers **115 large US listed firms** drawn from a broad cross-section of the S&P 500 and related indices, spanning 54 SIC-level industry groups. Firms were selected to represent both technology-intensive sectors (where AI adoption is most advanced) and traditional industrials, energy, healthcare, and finance (where the hype vs. substance tension is most visible). 5 firms in the initial universe of 120 were excluded due to incomplete filing histories.

### Time Coverage
Annual 10-K filings from **2021 to 2026** (fiscal year basis), yielding **658 firm-year observations** — an average of 5.7 filings per firm. The 2026 observations (91 firms) represent filings submitted by May 2026 and are included as a partial year.

### Data Funnel

| Stage | Count |
|---|---|
| Firms in universe | 120 |
| Unique firms with any filing | 115 |
| Firm-year observations | 658 |
| Firm-years with ≥1 AI mention | 459 (70%) |
| AI candidate chunks extracted | 4,204 |

### AI Mention Adoption by Year

| Year | Firms | AI disclosers | % disclosing | Mean chunks/filer |
|---|---|---|---|---|
| 2021 | 111 | 54 | 49% | 1.6 |
| 2022 | 113 | 53 | 47% | 1.7 |
| 2023 | 114 | 62 | 54% | 2.6 |
| 2024 | 114 | 94 | 82% | 5.3 |
| 2025 | 115 | 108 | 94% | 7.3 |
| 2026 | 91 | 88 | 97% | 8.0 |

The 2022→2023 step (+7pp) coincides with the ChatGPT release. The 2023→2024 jump (+28pp) is the largest single-year shift and aligns with both the SEC AI-washing guidance (March 2024) and the broader GenAI hype wave. By 2025–2026, AI disclosure has become near-universal.

### Section Distribution
Of 4,204 extracted AI chunks: 43% appear in **Risk Factors (Item 1A)**, 42% in **Business (Item 1)**, and 16% in **MD&A (Item 7)**. This split is the basis for D9 Section Location — a key dimension separating credible operational disclosure from boilerplate compliance language.

### Industry Composition (top groups by firm count)
Services-Prepackaged Software (14), Semiconductors (9), Pharmaceutical Preparations (8), Petroleum Refining (5), National Commercial Banks (5), Motor Vehicles (4), Biological Products (4). The remainder spans 47 additional SIC groups with 1–3 firms each.

### Scope Limitations
- **10-K only**: excludes 8-K (event-driven), 10-Q (quarterly), proxy statements, and earnings call transcripts. Captures annual strategic framing but misses real-time updates.
- **US-listed firms only**: no cross-country comparison.
- **Text-based**: no market or financial outcome variables in the current pipeline — cross-referencing stock returns or CapEx is possible but not yet implemented.
- **Sample size**: 115 firms limits statistical power for fine-grained subgroup analysis (e.g., industry × year × archetype cells can be small).

---

## Data Pipeline Summary

```
SEC EDGAR
    ↓
00_build_firm_universe.py       → firm_universe.parquet
    ↓
01_build_filing_manifest.py     → filing_manifest.parquet
    ↓
03_download_selected_filings.py → filings_html/
    ↓
04_extract_sections.py          → filing_sections.parquet
    ↓
05_prefilter_ai_mentions.py     → manifest updated
    ↓
06_chunk_candidates.py          → ai_candidate_chunks.parquet
    ↓
07_extract_bow_features.py      → ai_disclosure_bow_features.parquet   [BoW only]
    ↓
08_run_llm_classifier.py        → ai_disclosure_mentions.parquet       [LLM only, optional]
    ↓
09_score_with_llm_booleans.py   → ai_scored_chunks.parquet             [--bow-only flag available]
    ↓
10_build_features.py            → firm_year_features.parquet
    ↓
clustering_archetypes.ipynb     → firm_year_clusters.parquet
    ↓
ai_washing_by_cluster.ipynb     → plots + reports/
```

---

## Analysis Notebooks

| Notebook | Purpose | BoW/LLM flag |
|---|---|---|
| `clustering_archetypes.ipynb` | Build dimension scores, cluster firms, name archetypes | `USE_LLM = False` global |
| `ai_washing_by_cluster.ipynb` | Plot AI washing index through time by cluster | `USE_LLM = False` global |
| `temporal_analysis.ipynb` | Chapter 4 — cluster evolution, event plots, alluvial chart | `USE_LLM = False` global |
| `positioning_map.ipynb` | Chapter 2/3 — specificity vs. promotional scatter, section analysis | `USE_LLM = False` global |
| `ai_washing_analysis.ipynb` | Original AI washing analysis (legacy, superseded by above) | — |

---

## Features to Add

### In `10_build_features.py`

**LLM-derived firm-year aggregates** (require `ai_disclosure_mentions.parquet`):
```python
share_llm_is_ai_related        # confirmation rate
share_llm_is_substantive       # substantive mention share
share_llm_is_promotional       # promotional mention share
share_llm_is_risk_related      # risk mention share
share_llm_is_governance        # governance mention share
share_llm_is_financial_impact  # financial impact mention share
share_llm_mentions_training    # model training mention share
share_llm_mentions_vendor      # vendor dependency share
share_llm_mentions_cloud       # cloud infra share
share_llm_mentions_copilot     # copilot/assistant tool share
```

**Section-location aggregates** (require `ai_candidate_chunks.parquet`):
```python
sec_pct_business       # % of AI chunks in Business section
sec_pct_mda            # % of AI chunks in MD&A section
sec_pct_risk_factors   # % of AI chunks in Risk Factors section
sec_pct_other          # % of AI chunks in other sections
```

**BoW features added in `07_extract_bow_features.py`:**
```python
has_ai_quantified_claim     # AI term + dollar/% co-occurring in the same sentence (stronger than separate presence flags)
has_named_deployment        # named model/product + deployment verb in same sentence
has_dated_milestone         # time-anchored AI commitment ("by Q4 2025", "in fiscal 2026")
has_forward_looking         # will/expect/plan/intend/target language
has_realized_language       # delivered/launched/achieved/deployed language
ratio_forward_to_realized   # share of AI language that is promise vs evidence; key hype signal
```

**BoW features still to add** (future):
```python
text_reuse_ratio_vs_prior_year   # cosine similarity of this firm's AI section vs prior year
                                 # boilerplate carry-forward is itself a credibility gap signal
has_placebo_deepseek_mention     # DeepSeek + risk/impact language (Ch. 7 validation)
```

---

## Strongest Final Plots (Thesis Figures)

| # | Figure | Chapter |
|---|---|---|
| 1 | Data pipeline funnel diagram | Methodology |
| 2 | AI mention volume over time by industry | Ch. 1 |
| 3 | Share of firms mentioning AI per year | Ch. 1 |
| 4 | Specificity × Promotional tone scatter, coloured by cluster | Ch. 2/3 |
| 5 | Cluster dimension heatmap (z-scores) | Ch. 3 |
| 6 | UMAP scatter, coloured by archetype | Ch. 3 |
| 7 | Cluster share by industry | Ch. 3 |
| 8 | Stacked area: archetype share over time | Ch. 4 |
| 9 | Alluvial chart: firm migration across archetypes | Ch. 4 |
| 10 | Dimension shift pre/post SEC 2024 by cluster | Ch. 4/7 |
| 11 | Hype vs Substance scatter by firm | Ch. 5 |
| 12 | AI Washing Index trend by archetype | Ch. 5 |
| 13 | Section distribution per archetype | Ch. 6 |
| 14 | Pre/post DeepSeek: D3 Technical and D8 Defensive | Ch. 7 |

---

## Validation Plan

**Required for a defensible thesis — without this, washing claims are assertions, not findings.**

### BoW vs LLM agreement (Cohen's κ)
For each dimension with both a BoW and LLM proxy, compute chunk-level agreement:
- D2 Operational: `has_deployment_verb` vs `is_substantive`
- D5 Governance: `has_board_oversight` vs `is_governance_related`
- D6 Risk: `has_risk_factor` vs `is_risk_related`
- D7 Promotional: `ratio_vague_words` vs `is_promotional`

Report κ per dimension. κ < 0.4 means BoW proxy is unreliable for that dimension.

### Hand-labelling ground truth
Label ~150 chunks (25 per archetype post-clustering) on three axes: substantive / promotional / risk.
Two raters minimum. Compute inter-rater agreement, then precision/recall for both BoW and LLM.
This justifies the dimension definitions and is standard in content-analysis theses.

### Cluster stability robustness checks
- Vary k from 3 to 8; report silhouette score and cluster coherence
- Vary random seed (n=10); report % of firm-year observations with stable archetype assignment
- Run with BoW-only vs LLM-augmented features; report how many firms change archetype

### SEC 2024 placebo test
Run the pre/post analysis with a placebo cutoff (e.g., 2022 or 2023).
If D5/D6 show a significant jump at the placebo date, the SEC 2024 finding is not credible.

---

## Key Narrative

> The thesis maps how listed firms position themselves around AI in investor communications from 2021 to 2025. Using a large sample of 10-K filings, it identifies six distinct AI communication archetypes, ranging from operationally credible adopters to promotional narrators and boilerplate compliance filers. It then evaluates how these archetypes evolved after the ChatGPT boom, SEC AI-washing scrutiny in 2024, and the DeepSeek market shock in early 2025. The AI washing analysis emerges as a second layer: after identifying the archetypes, the thesis measures which clusters show the largest gap between AI claims and operational evidence, and whether regulatory and market events narrowed or widened that gap.
