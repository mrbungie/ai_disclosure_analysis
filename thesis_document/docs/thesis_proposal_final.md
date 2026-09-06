# AI Washing or Credible Disclosure? Patterns and Clusters in Corporate AI Disclosures

**Thesis proposal — final version**
Germán · MBA, MIB Trieste · September 2026

---

## 1. Business problem and research questions

Since the release of ChatGPT in late 2022, U.S.-listed firms have sharply increased the volume of AI-related language in their annual reports. Regulators and investors worry that a large share of this language is vague, promotional, or disconnected from operational reality. In March 2024 the SEC explicitly warned against "AI washing" and brought its first enforcement actions on the matter.

The managerial problem has three faces. For the disclosing firm, AI communication is a strategic choice with reputational and regulatory consequences, made without a benchmark of what peers do or what regulators treat as credible. For the capital allocator — asset managers, analysts, due-diligence teams — AI narrative is an input to valuation and risk assessment with no established way to separate substance from noise. For management scholarship, AI washing is the latest instance of talk–action decoupling, and it is unclear whether the frameworks built for greenwashing transfer to it.

**Main research question**
How do public firms differ in what they disclose about AI adoption, capabilities, risks and governance, and which distinct disclosure archetypes emerge from those differences?

**Extended research questions**
- RQ2 — Is it *what* firms say about AI, or simply *how much* they say, that distinguishes them?
- RQ3 — Are disclosure archetypes associated with coherent economic characteristics (R&D intensity, growth, margins, valuation, risk)?
- RQ4 — Does the semantic content of AI disclosure carry information beyond conventional fundamentals and simple AI-mention intensity?
- RQ5 — How have disclosure patterns evolved over FY2021–2025, and did firms with more promotional pre-2024 disclosures change behaviour after the SEC's 2024 AI-washing scrutiny?
- RQ6 — Did an external competitive shock (DeepSeek, January 2025) change how firms framed AI capabilities, costs and competitive positioning?

## 2. Literature and theoretical framing

The dissertation frames AI washing with established theory before measuring it, so that the empirical patterns are interpreted rather than merely described. Four lenses are used:

- **Voluntary disclosure and signalling theory** (Verrecchia 1983; Spence 1973): credible disclosure is costly to imitate; specificity, realization and verifiable metrics are the cost that separates signal from cheap talk.
- **Cheap talk and impression management** (Crawford and Sobel 1982; Merkl-Davies and Brennan 2007): when claims are unverifiable, managers have incentives to inflate; rhetoric and hypothetical framing are the observable footprint.
- **Legitimacy and institutional theory** (Suchman 1995; DiMaggio and Powell 1983): firms adopt AI language to conform with peers and stakeholder expectations, which predicts mimetic, low-specificity disclosure that spreads by sector.
- **Greenwashing as the reference case** (Delmas and Burbano 2011; Lyon and Montgomery 2015): the decoupling of talk from action, and the regulatory response to it, provides the template for "X-washing" and for Chapter 6.
- **Investor-side reading**: the literature on how analysts and fund managers process narrative disclosure (Loughran and McDonald 2011 on tone; Li 2008 on readability; Huang et al. 2014 on managerial hype) frames the capital-allocator implications.

Empirically, the thesis builds on three strands. First, the finance literature on textual measures of firm exposure: Eisfeldt, Schubert and Zhang (2023) measure generative-AI exposure and link it to firm value; Basnet et al. (2025) separate substantive from speculative AI narratives in filings and study market reaction. Second, the accounting literature on disclosure quality, boilerplate and cheap talk, which frames the question of whether narrative disclosure is informative or merely mimetic. Third, the Loughran–McDonald tradition of dictionary-based financial text analysis, which the prefilter follows in spirit (transparent lexical gate, auditable signals), extended with embedding-based semantic scores and an LLM extraction layer whose outputs are validated against human coding rather than taken on trust.

The gap this thesis addresses: existing work measures AI *exposure* or a binary substantive/speculative split. It does not characterise the multi-dimensional structure of AI disclosure, does not test whether that structure adds information beyond mention volume, and does not study how the structure responded to regulatory scrutiny.

## 3. Data and scope

- **Corpus:** S&P 500 constituents, fiscal years 2021–2025, across five document types. Current retrieval counts of AI-positive paragraphs:

| Source | Frequency | Nature of disclosure | AI paragraphs |
|---|---|---|---|
| Form 10-K | Annual | Mandated, audited narrative (business, risk factors, MD&A) | 11,806 |
| Earnings call transcripts | Quarterly | Voluntary, management-framed, analyst-questioned | 9,750 |
| DEF 14A (proxy statement) | Annual | Board oversight, governance, executive incentives | 7,407 |
| Form 10-Q | Quarterly | Mandated, interim updates | 2,729 |
| Form 8-K | Event-driven | Material events, press releases | 452 |

- **Firm characteristics:** XBRL financial statement data from EDGAR (revenue, R&D, capex, margins, assets) and market data (prices, returns, market cap) for valuation and risk measures.
- **Sector:** GICS classification, used as a characterisation variable, not as a sample filter.

The multi-source design matters for two reasons. First, it separates *mandated* (10-K, 10-Q) from *voluntary* (calls, 8-K) and *governance* (DEF 14A) disclosure, so the same firm's AI narrative can be compared across venues with different liability and audience. Second, calls and 10-Qs give quarterly frequency, which is what makes the two 2024–2025 shocks (SEC scrutiny, DeepSeek) separately identifiable.

## 4. Empirical methodology

The pipeline has three layers: measurement, aggregation, and analysis.

### 4.1 Measurement (paragraph level)

The corpus contains ~3.28M paragraphs (1.65M unique texts after BLAKE2b deduplication). Measurement proceeds in three stages, each producing a persisted, idempotent artefact keyed by text hash so that upstream changes do not force downstream recomputation.

1. **Embedding.** Every unique text in the corpus (1.65M) is embedded once with BAAI/bge-m3 (1024 dimensions), with no prior filtering. Semantic margin scores against positive and negative anchors are computed on these paragraph vectors for the whole corpus.

2. **AI prefilter.** A classifier answers "does this paragraph mention AI?" — deliberately *not* "is this substantive?", since vagueness is the object of study and must not be filtered out. It combines a lexical gate (strong and weak terms with word boundaries, including acronyms; recall 0.98, ~67k unique texts pass) with the paragraph-level semantic margin (max positive-anchor similarity minus negative-anchor similarity). Within the lexical-gate population, an additional sentence-level signal is computed: only the sentences containing an AI term (30,792 of 226,139) are embedded separately, because in long structural texts such as board skills matrices the paragraph vector is dominated by non-AI content even when a cell explicitly mentions AI. The deployed model is a decision-tree ensemble over 39 signals: 12 paragraph-level, 15 text-form features (term density, table/bullet structure, first person, biography/vote/regulation context, acronym vs full term) and 12 sentence-level. Document type is excluded as a feature to keep cross-venue comparisons non-circular. Trained on ~12,300 LLM-judge labels across all five sources with GroupKFold by filing; per-channel recall 0.94–0.98, out-of-domain holdout F1 0.81 and AP 0.89. Named entities (e.g. OpenAI) are force-included. Roughly 0.6% of unique texts pass (~30–33k), which is the population for the next stage.

3. **Frame extraction.** Each prefiltered paragraph is sent to an LLM (Qwen 3.7 Flash via OpenRouter, structured output through pydantic-ai) segmented into numbered sentences. The model returns zero or more *frames*, each a coherent proposition about AI, with evidence as sentence indices rather than copied text (no fabricated quotes). Frame schema:

   | Field | Values | Role in the thesis |
   |---|---|---|
   | `subject` | firm / suppliers or partners / customers / competitors or industry | Separates the firm's own claims from industry commentary |
   | `ai_type` | generative / predictive ML / unspecified | Distinguishes the post-ChatGPT wave from legacy ML |
   | `temporal` | realized / planned / expected / hypothetical | Core washing axis: doing vs promising vs speculating |
   | `domain` | internal / customer-facing / unspecified | Where AI is applied |
   | `concepts[]` | 29 values across use, investment, outcomes, risk, governance | Topic breadth and category presence |
   | `specificity` | 5 booleans: process, product, vendor, metric, date | Depth of the claim (0–5 when summed) |
   | `rhetoric` | promotional, strategic importance | Tone |
   | `evidence` | sentence ids | Auditability |

   The extraction rule is the minimum number of frames that preserves the relations in the paragraph, with explicit attribution rules for negations and subjects.

### 4.2 Validation

Validation is staged and is a first-class deliverable of the thesis.

- **Prefilter:** the LLM judge agrees with an independent model (Gemini) at κ = 0.87 on 6,038 rows, and the deployed classifier is evaluated against per-channel holdouts as reported above. A small human-coded check on the prefilter boundary (borderline and out-of-domain cases) will be added.
- **Frame extraction:** *not yet validated* and identified as the critical-path item. Plan: a stratified human sample (target ≥ 400 paragraphs across sources, years and prefilter score bands), double-coded, reporting agreement at frame level (count and boundaries) and field level (κ per field, with `temporal`, `specificity` and `rhetoric` as the fields that carry the thesis's claims). Systematic disagreements feed a schema revision before any firm-level result is finalised.

### 4.3 Aggregation (firm-period level)

Frames are aggregated to firm-year (all sources) and firm-quarter (calls, 10-Q, 8-K) features, computed within source and pooled. Only frames with `subject = firm` enter the core features; other subjects are kept as separate context measures.

- **Volume** — number and share of AI paragraphs and frames (the "how much" measure).
- **Realization** — share of frames that are realized vs planned/expected/hypothetical; the direct measure of promising versus doing.
- **Specificity** — mean specificity count and share of frames with a metric or date.
- **Rhetoric** — promotional share; strategic-importance share.
- **Breadth and presence** — number of concepts covered; presence of risk and governance concepts; presence of investment and outcome concepts.
- **Composition** — generative vs predictive share; internal vs customer-facing share.
- **Cross-venue consistency** — gap between the same firm's realization, specificity and rhetoric in the 10-K vs on calls vs in the proxy.

### 4.4 Analysis

- **Archetypes.** Clustering on the semantic features (realization, specificity, rhetoric, breadth, composition; excluding raw volume), with dimensionality reduction for interpretation and stability checks across algorithms and years. Archetypes are then characterised by sector, size, R&D intensity, growth, margins, valuation and risk (CAPM beta, total and idiosyncratic volatility).
- **Quantity vs content.** Test whether archetype membership and semantic features are separable from mention volume; if clusters simply reproduce a volume ranking, the semantic layer adds nothing.
- **Incremental signal.** Nested regressions of firm outcomes (risk, valuation, R&D, growth) on (i) fundamentals, (ii) fundamentals + AI volume, (iii) fundamentals + volume + semantic features/archetypes, reporting ΔR² and partial R². Interpreted as association, not causation.
- **Scrutiny.** A continuous-treatment difference-in-differences on the quarterly series (calls, 10-Q): the SEC's 18 March 2024 statements define the post-period, and pre-period promotional share and hypothetical/expected share define treatment intensity. Outcomes are decomposed (promotional share, realization, specificity, risk and governance presence) to distinguish "less promotion", "more substance" and "same content, different wording". Annual sources (10-K, DEF 14A) provide a slower-moving check. Event-study plots of quarterly pre-trends are reported; the design describes a regime shift, not a clean natural experiment.
- **Competitive shock.** The DeepSeek release (27 January 2025) falls within the post-SEC period but roughly three quarters later, so at quarterly frequency it can be modelled as a second break. The focus is on framing (competition category, capex/cost language, tone) rather than volume.
- **Venue comparison.** Whether archetypes and scrutiny effects are consistent across mandated, voluntary and governance venues — in particular whether call promotionality fell while 10-K content stayed flat, or vice versa.

## 5. Scope decisions imposed by the format

The dissertation is worth 4 of 60 credits, must fit 30–60 pages (≤150,000 characters, figures ≤20 pages) and is graded on managerial relevance, clarity of objectives and method, use of academic theory, coherence, and richness of conclusions. Those constraints drive the following decisions:

- **The measurement pipeline is infrastructure, not the thesis.** Its full description, validation tables and error taxonomy go to the appendix (excluded from the page count). The body carries a 5–6 page summary sufficient for a business reader to trust the measures.
- **One central empirical claim.** The thesis stands on whether disclosure *content* carries information beyond *volume* and fundamentals. Everything else supports or qualifies that claim.
- **The managerial chapter is a chapter, not an epilogue, and "managerial" is read broadly.** It addresses three decision-makers: the issuer deciding how to communicate, the asset manager deciding how to read that communication, and the general-management field deciding whether AI washing is a new phenomenon or a known one in new clothes.
- **Cut:** out-of-sample prediction, standalone heterogeneity analysis (folded into the scrutiny chapter), and per-field pipeline diagnostics in the body.
- **Kept but bounded:** DeepSeek as one section (2–3 pages) inside the scrutiny chapter, not a chapter.

## 6. Analysis plan

| # | Analysis | Output in the body | Question answered | Where |
|---|---|---|---|---|
| 1 | AI prefilter | Coverage of AI paragraphs by source, year, sector (1 figure) | Where and how much do firms talk about AI? | Ch. 2, App. A |
| 2 | Frame extraction | Frame schema and 2–3 worked examples | What are they actually saying? | Ch. 2, App. A |
| 3 | Validation | Summary table: prefilter holdouts; human-coded frame sample, κ per field | Can the measures be trusted? | Ch. 2 (summary), App. B (full) |
| 4 | Firm-period features | Feature definitions table; descriptive statistics | How is each firm's strategy characterised? | Ch. 2 |
| 5 | Archetypes | Clustering, stability, exemplar firms per cluster (2 figures) | Are there distinguishable disclosure strategies? | Ch. 3 |
| 6 | Archetype profiles | Sector, size, R&D, growth, margins, valuation, risk by cluster | Which firms follow each strategy? | Ch. 3 |
| 7 | Quantity vs content | Volume vs semantic features; do clusters survive controlling for volume? | Does content matter beyond volume? | Ch. 4 |
| 8 | Incremental signal | Nested regressions, ΔR², partial R² (1 table, 1 figure) | Does the text add information beyond fundamentals and volume? | Ch. 4 |
| 9 | Evolution 2021–2025 | Feature trajectories, archetype transition matrix | How has AI disclosure changed since ChatGPT? | Ch. 5 |
| 10 | SEC scrutiny | Continuous-treatment DiD on quarterly data, decomposed outcomes, pre-trends, by archetype and venue | Did behaviour change under scrutiny, and how? | Ch. 5 |
| 11 | DeepSeek | Quarterly break in framing and tone around Jan 2025 | Does a competitive shock change the discourse? | Ch. 5 |
| 12 | Venue comparison | Same firm across 10-K, calls, proxy | Do firms tell the same AI story to regulators, analysts and shareholders? | Ch. 5 |
| 13 | Management implications | Issuer benchmark and self-assessment; investor screening heuristics; conceptual mapping to greenwashing / X-washing | What should issuers, investors and management scholars do with this? | Ch. 6 |

## 7. Structure and page budget

| Part | Pages | Content |
|---|---|---|
| Executive summary | 2 | Problem, method in one paragraph, three main findings, what issuers and investors should do |
| Introduction | 3 | Why AI disclosure quality matters now; research questions; methodology in outline; expected outcomes; roadmap of chapters |
| **Ch. 1 — AI washing: context and theory** | 7 | 1.1 The post-ChatGPT disclosure wave · 1.2 The SEC's 2024 position and enforcement · 1.3 Theoretical lenses (§2 above) · 1.4 Prior empirical work and the gap · 1.5 Hypotheses derived from theory |
| **Ch. 2 — Data and measurement** | 8 | 2.1 Corpus and sources · 2.2 From text to frames (summary) · 2.3 Validation (summary) · 2.4 Firm-period features · 2.5 Financial and market data · 2.6 Analytical strategy |
| **Ch. 3 — Disclosure archetypes** | 9 | 3.1 Descriptive landscape · 3.2 Clustering and archetypes · 3.3 Who belongs where · 3.4 Naming and interpreting the archetypes through the theoretical lenses |
| **Ch. 4 — Do the words carry information?** | 8 | 4.1 Volume vs content · 4.2 Economic substance by archetype · 4.3 Incremental signal beyond fundamentals · 4.4 Interpretation: signal, cheap talk, or mimicry |
| **Ch. 5 — Disclosure under scrutiny** | 9 | 5.1 Evolution 2021–2025 · 5.2 The SEC event: design and results · 5.3 What changed and where (decomposition, archetypes, venues) · 5.4 DeepSeek · 5.5 Robustness |
| **Ch. 6 — Implications for management** | 8 | 6.1 For the disclosing firm: disclosure-profile benchmark, self-assessment for IR and boards, regulatory exposure indicators · 6.2 For capital allocators: reading AI narrative as a signal — which features carried information, screening heuristics for equity research and due diligence, what to discount · 6.3 For management theory and practice: AI washing as talk–action decoupling, what transfers from greenwashing and what does not, a general X-washing lens · 6.4 Limits of each set of implications |
| Conclusions | 2 | Main outcomes; indications for issuers, investors, regulators and management research; open questions |
| AI statement | ½ | Declared use of generative AI in pipeline (LLM judge) and in writing support |
| Bibliography | — | Harvard-style, alphabetical, per School rules |
| **Appendix A** — Pipeline | — | Embeddings, prefilter design and signals, frame schema and rules, operational details |
| **Appendix B** — Validation | — | Holdout tables, human coding protocol, per-field agreement, error taxonomy |
| **Appendix C** — Robustness | — | Alternative clusterings, alternative treatment definitions, placebo dates |
| **Total body** | **~55** | Within the 60-page / 150,000-character limit; ≤20 pages of figures and tables |

## 8. Expected contribution

- For **issuers (IR teams, boards, CFOs)**: a benchmark of AI disclosure profiles with named archetypes, and a checklist to locate their own communication against peers and against the SEC's stated concerns.
- For **capital allocators (asset managers, analysts, due-diligence teams)**: evidence on which features of AI narrative carried information beyond fundamentals and mention volume, and which did not — a basis for screening and discounting rules.
- For **regulators**: evidence on what actually changed after the 2024 scrutiny — less promotion, more substance, or rewording — and in which venue.
- For **management studies**: a frame-level, multi-venue measurement of AI disclosure interpreted through signalling, cheap-talk and legitimacy theory, and a test of how far the greenwashing framework transfers to a technology narrative.

## 9. Limitations and design choices

- **Event proximity.** The SEC action (March 2024) and DeepSeek (January 2025) are ten months apart; separable at quarterly frequency, but the post-SEC window before DeepSeek is only three quarters. Annual sources cannot distinguish the two and serve as a robustness layer.
- **No causal claim on economic outcomes.** Disclosure and fundamentals are jointly determined; incremental R² is interpreted as informational content, not causal effect.
- **Treatment intensity, not treatment groups.** All firms are exposed to the SEC action at once; identification relies on pre-period disclosure intensity and documented pre-trends.
- **Source heterogeneity.** Calls are transcribed speech with analyst prompting; filings are lawyer-reviewed text. Features are computed within source and compared across sources, never pooled naively.
- **Frame extraction not yet validated.** The critical-path item; results are provisional until the human-coded sample is completed and the schema revised if agreement on `temporal`, `specificity` or `rhetoric` is inadequate.
- **Sample.** S&P 500 constituents are large, well-covered firms; findings may not extend to small caps, where AI washing incentives could be stronger.

## 10. Compliance notes

- Written in English; numbered chapters and paragraphs per School convention.
- All quoted text in quotation marks with in-text citation; single-source share kept well under 1%. Filing excerpts used as illustrations are short, quoted and cited to the issuer and filing date.
- Generative AI use declared in the cover letter and AI statement: an LLM is used as a classification instrument in the measurement pipeline (described and validated in Appendices A–B); any AI assistance in drafting or editing is declared separately.
- Program Director as the interface for topic approval and feedback; faculty consulted for advice only.
