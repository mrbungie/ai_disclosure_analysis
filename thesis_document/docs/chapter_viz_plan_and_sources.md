# Chapter-by-chapter analysis plan: figures, tables, expected findings

Companion to `thesis_proposal_final.md`. Budget: ≤20 pages of figures and tables in the body. Target: 15 figures, 11 tables. Everything else goes to appendices.

Conventions: **F** = figure, **T** = table. "Expected" is the hypothesis derived from theory (Ch. 1.5); "If not" is the alternative reading so the chapter works either way.

---

## Chapter 1 — AI washing: context and theory (7 pages)

Mostly prose. Two visuals.

| ID | Type | Construction | Expected | If not |
|---|---|---|---|---|
| F1.1 | Line, dual axis | Monthly count of S&P 500 AI paragraphs (all sources, pooled) Jan 2021–Dec 2025; vertical markers at ChatGPT (Nov 2022), Gensler "AI washing" speech (Dec 2023), SEC actions (18 Mar 2024), DeepSeek (27 Jan 2025). Secondary axis: Google Trends "artificial intelligence" or NVDA share price as attention proxy. | Step change after Nov 2022, second acceleration mid-2023, flattening or dip after Mar 2024. Motivates the whole thesis in one picture. | Monotonic increase with no visible break → the SEC effect is compositional (what is said), not volumetric, which is exactly the argument for Ch. 5. |
| T1.1 | Table | Hypotheses H1–H6 with the theoretical lens each derives from and the chapter that tests it. | — | — |

**H1** (signalling): specificity and realization are costly, so they separate archetypes. **H2** (cheap talk): promotional/hypothetical frames concentrate where verification is hardest. **H3** (legitimacy): low-specificity AI talk spreads by sector, not by firm economics. **H4** (informativeness): semantic content adds explanatory power beyond volume. **H5** (regulatory response): scrutiny reduces promotional/hypothetical share more than it raises realized/specific share. **H6** (venue): voluntary venues react more than mandated ones.

---

## Chapter 2 — Data and measurement (8 pages)

| ID | Type | Construction | Expected | If not |
|---|---|---|---|---|
| T2.1 | Table | Corpus by source: filings, paragraphs, unique texts, lexical-gate passers, prefilter positives, frames extracted, firms covered, FY range. | — | — |
| F2.1 | Stacked bar | AI paragraphs per firm-year by source, FY2021–2025, median and IQR. | 10-K and calls dominate; DEF 14A appears from FY2023. | — |
| F2.2 | Heatmap | Share of firms with ≥1 AI paragraph, GICS sector × fiscal year. | Near-saturation in IT/Comm Services by FY2022; broad diffusion to Industrials, Financials, Health Care by FY2024 (H3). | Diffusion stays sector-bound → mimicry weaker than expected. |
| F2.3 | Pipeline schematic | Corpus → embeddings → lexical gate → prefilter → frames → features. One page, no code. | — | — |
| T2.2 | Table | Frame schema: field, values, feature it feeds, hypothesis it serves. | — | — |
| T2.3 | Table | Validation summary: prefilter per-channel recall/precision/F1/AP; frame extraction human-sample κ per field with n. Full detail in App. B. | κ ≥ 0.7 on `temporal`, `specificity`, `rhetoric`. | If κ < 0.6 on a field, that field is collapsed (e.g. planned/expected/hypothetical → "not realized") before any firm-level use, and this is stated. |
| T2.4 | Table | Firm-period feature definitions: name, formula, level (firm-year / firm-quarter), source scope, hypothesis. ~12 rows. | — | — |
| T2.5 | Table | Financial and market variables. See block below. | — | — |
| T2.6 | Table | Descriptive statistics: all features and fundamentals, N, mean, SD, p10/p50/p90, by source where relevant. | — | — |

**Financial and market variables (T2.5)** — all from EDGAR XBRL Financial Statement Data Sets and daily prices; fiscal-year aligned to the 10-K:

| Group | Variable | Definition |
|---|---|---|
| Size | log market cap; log total assets | at fiscal year-end |
| Investment | R&D / sales; capex / sales; SG&A / sales | R&D missing → 0 with indicator |
| Profitability | gross margin; EBIT margin; ROA | |
| Growth | revenue growth (YoY); 3-year revenue CAGR | |
| Valuation | price / sales; EV / EBITDA; book-to-market; Tobin's Q (approx.) | |
| Balance sheet | cash / assets; net debt / EBITDA | |
| Risk | CAPM beta (252 trading days vs S&P 500); total volatility; idiosyncratic volatility (residual SD from market model) | computed over the fiscal year |
| Classification | GICS sector, industry group | characterisation only |

---

## Chapter 3 — Disclosure archetypes (9 pages)

| ID | Type | Construction | Expected | If not |
|---|---|---|---|---|
| F3.1 | Small multiples (6 panels) | Distribution of each core feature (realization, specificity, promotional share, breadth, generative share, governance presence) across firm-years, pooled and by year overlaid. | Bimodality in realization and specificity — a "doing" mass and a "talking" mass — which is what makes clustering meaningful. | Unimodal → archetypes are gradations, not types; present as a continuum with cut-points. |
| F3.2 | Scatter, 2D projection | Firm-years in UMAP or PCA space of semantic features (volume excluded), coloured by cluster, sized by volume. | 3–5 clusters with clear separation on realization × specificity, and promotional share as the third axis. | — |
| T3.1 | Table | Cluster diagnostics: k selection (silhouette, gap), stability across algorithms (k-means, GMM, hierarchical) and across bootstrap, Adjusted Rand Index between methods. | Stable k in 3–5. | — |
| T3.2 | Table | Archetype profile: mean of each semantic feature by cluster, with cluster name and one-line interpretation. Provisional names: *Operators* (realized, specific, low promotion), *Strategists* (planned, moderate specificity, strategic-importance rhetoric), *Aspirants* (expected/hypothetical, low specificity, promotional), *Risk-framers* (risk and governance dominant, low realization), *Minimalists* (low volume, unspecified). | Operators and Aspirants exist as distinct poles. | — |
| F3.3 | Radar / parallel coordinates | Same as T3.2, visual. | — | — |
| T3.3 | Table | Exemplar firms per cluster (3 each) with one quoted frame (≤15 words, cited to issuer and filing) — the only verbatim filing text in the body. | — | — |
| F3.4 | Stacked bar | Cluster composition by GICS sector. | Operators concentrated in IT, Comm Services, selected Financials; Aspirants over-represented in Consumer, Industrials, Energy (H3). | Even distribution → sector is not the driver; move to firm economics (Ch. 4). |
| T3.4 | Table | **Fundamentals by archetype**: mean and median of every T2.5 variable by cluster, with pairwise tests (Operators vs Aspirants at minimum), sector-demeaned versions in a second panel. | Operators: higher R&D/sales, higher P/S, higher beta, larger. Aspirants: lower R&D, lower growth, valuation near sector median, **not** lower beta. Risk-framers: Financials and Health Care, lower volatility. | Aspirants with high R&D → "talking" is not "not doing"; the archetype is a communication style, not an economic state. Still a finding. |
| F3.5 | Box plots, 2×3 grid | R&D/sales, revenue growth, P/S, beta, idiosyncratic vol, gross margin by cluster, sector-demeaned. | Visual version of T3.4; the sector-demeaned panel is the one that matters. | — |

Prose in 3.4 maps each archetype to a theoretical lens: Operators = separating equilibrium (signalling); Aspirants = pooling / cheap talk; Risk-framers = legitimacy via compliance language; Minimalists = non-disclosure (Dye 1985, Verrecchia 1983: silence when the news is not good enough).

---

## Chapter 4 — Do the words carry information? (8 pages)

This is the fundamentals chapter and the central claim. Every visual must isolate *content* from *volume* and from *what we already knew*.

| ID | Type | Construction | Expected | If not |
|---|---|---|---|---|
| F4.1 | Scatter with marginal densities | AI volume (log paragraphs per firm-year) vs realization score, coloured by cluster. Add a LOESS line. | Weak correlation (ρ < 0.3). Aspirants spread across the whole volume range — you can talk a lot and say nothing, or talk little and say a lot. | Strong correlation → content is mostly a volume story; the thesis narrows to "how much" and the DiD becomes the main contribution. |
| T4.1 | Table | Correlation matrix: volume, each semantic feature, each fundamental. Partial correlations of semantic features with fundamentals after removing volume. | Specificity and realization survive partialling out volume; promotional share is nearly orthogonal to fundamentals. | — |
| F4.2 | Heatmap | Cluster membership after re-clustering with volume included vs excluded (confusion matrix, ARI). | High ARI (> 0.6): archetypes are not a volume artefact. | Low ARI → present clusters conditional on volume tercile. |
| T4.2 | Table | **Nested regressions, cross-sectional with year and sector FE.** Dependent variables (one column block each): CAPM beta, idiosyncratic vol, P/S, R&D/sales, revenue growth (t+1). Rows: (M0) fundamentals only; (M1) M0 + log AI volume; (M2) M1 + semantic features; (M3) M1 + archetype dummies. Report adjusted R², ΔR² vs previous model, partial R² of the added block, and F-test. Standard errors clustered by firm. | ΔR² from M1→M2 is small but significant for beta, idiosyncratic vol and P/S (0.5–2 pp); near zero for margins. Sign pattern: specificity ↑ P/S, realization ↓ idiosyncratic vol, promotional share ↑ idiosyncratic vol, hypothetical share ↑ beta. | ΔR² ≈ 0 everywhere → disclosure content is not informative beyond volume and fundamentals; the investor implication becomes "discount narrative entirely", still a clean result. |
| F4.3 | Bar with error bars | ΔR² (M1→M2) per dependent variable, with bootstrap CI; second series for M1→M3. | Risk and valuation variables show the increment; operating variables do not. | — |
| F4.4 | Coefficient plot | Standardised coefficients of the five core semantic features in M2, per dependent variable, 95% CI. | Realization and specificity carry the load; promotional share matters only for volatility. | — |
| T4.3 | Table | Robustness: (a) within-source features (10-K only, calls only); (b) firm FE panel instead of cross-section; (c) excluding IT and Comm Services; (d) winsorised fundamentals. Same ΔR² statistic. | Result holds in calls-only and outside tech; weakens in firm-FE panel (content is a slow-moving firm trait). | — |
| F4.5 | Binned scatter | For the strongest result (likely idiosyncratic vol on realization): 20 quantile bins of realization vs residualised outcome (after M1), with linear fit. | Monotonic negative relation. | — |

4.4 interprets the pattern through the lenses: an increment on risk/valuation but not on margins is consistent with the market pricing *credibility* rather than *performance* (signalling), and with promotional language adding *uncertainty* rather than *value* (cheap talk).

---

## Chapter 5 — Disclosure under scrutiny (9 pages)

| ID | Type | Construction | Expected | If not |
|---|---|---|---|---|
| F5.1 | Line, 5 panels | Each core semantic feature over fiscal years FY2021–2025, mean with 95% CI, by source (10-K, calls, DEF 14A). | Realization and specificity rise steadily; promotional share peaks FY2023; governance presence jumps in DEF 14A from FY2024. | — |
| F5.2 | Alluvial / Sankey | Archetype transitions FY2021→2023→2025 (firms with all years). | Aspirants → Strategists → Operators is the modal upgrade path; few downgrades; Minimalists shrink. | Sticky membership → strategies are firm traits, reinforcing Ch. 4 robustness (b). |
| T5.1 | Table | Transition matrix FY2023 → FY2025, with row percentages. | — | — |
| F5.3 | Event-study plot, quarterly | Calls + 10-Q, firm-quarter panel 2021Q1–2025Q4. Coefficients of (pre-period promotional share × quarter dummies) on promotional share, relative to 2024Q1. Vertical lines at SEC (2024Q1/Q2 boundary) and DeepSeek (2025Q1). | Flat pre-trends; negative coefficients from 2024Q2 for high-intensity firms (H5). | Pre-trends not flat → report honestly, move to a synthetic-control or matched comparison as robustness, and reframe as descriptive regime change. |
| T5.2 | Table | **Continuous-treatment DiD**: `y_it = α_i + λ_t + β·(Intensity_i × Post_t) + ε`, outcomes: promotional share, hypothetical+expected share, realization, specificity, risk presence, governance presence. Intensity = pre-period promotional share (main) and hypothetical share (alternative). Firm and quarter FE; SE clustered by firm; separate columns per source. | β < 0 on promotional and hypothetical; β ≈ 0 or small positive on realization and specificity; β > 0 on risk and governance presence. Reading: firms **removed** the rhetoric and **added** compliance framing, more than they added substance. | β > 0 on specificity/realization → scrutiny worked in the intended direction; equally publishable. |
| F5.4 | Coefficient plot | T5.2 β per outcome, by source (10-K vs calls vs DEF 14A), 95% CI. | Larger effects in calls than in 10-K (H6); governance effect only in DEF 14A. | — |
| F5.5 | Heatmap | DiD β by pre-period archetype × outcome. | Aspirants show the largest drop in promotional share; Operators unchanged. | — |
| F5.6 | Line, quarterly | Share of frames with `subject = competitors_or_industry` and with cost/capex/efficiency concepts, 2024Q1–2025Q4, by sector group. | Spike in 2025Q1–Q2 for IT and Comm Services (DeepSeek framing: cost, competition, model commoditisation), flat elsewhere. | — |
| T5.3 | Table | DeepSeek break: same DiD form, Post = 2025Q1 onward, intensity = generative-AI share in 2024, outcomes = competitor-subject share, cost concepts, promotional share. Reported with the caveat that only three post-SEC pre-DeepSeek quarters exist. | Competitive framing ↑; promotional share on own generative capabilities ↓. | — |
| F5.7 | Scatter | Same firm, same quarter: promotional share in calls (x) vs in 10-Q (y), FY2023 vs FY2025 panels. | Above-diagonal mass shrinks after scrutiny: the venue gap closes. | Gap persists → mandated text was already disciplined; scrutiny touched only the voluntary channel. |
| T5.4 | Table | Robustness summary: placebo event dates (2023Q1, 2023Q3), alternative intensity definitions, dropping IT sector, annual 10-K-only version, Callaway–Sant'Anna-style estimator with intensity terciles. | — | — |

---

## Chapter 6 — Implications for management (8 pages)

| ID | Type | Construction | Expected | If not |
|---|---|---|---|---|
| F6.1 | 2×2 or 2×3 positioning map | Realization (x) × specificity (y), archetype regions shaded, with a sample of named S&P 500 firms plotted (FY2025). The "benchmark" a board can locate itself on. | — | — |
| T6.1 | Table | Issuer self-assessment: 8–10 questions mapped to frame fields (e.g. "Of your AI statements, what share names a process, product, metric or date?"), with the S&P 500 median and Operator-cluster median as reference points. | — | — |
| T6.2 | Table | Regulatory-exposure indicators: features that (a) the SEC's own language targets, (b) fell most after scrutiny in Ch. 5, (c) carried no information in Ch. 4 — the intersection is what an IR team should cut first. | — | — |
| T6.3 | Table | Investor screening heuristics, one row per Ch. 4 finding: feature, direction, what it was associated with, how to use it (screen / discount / ignore), confidence level. Written so it reads correctly whether Ch. 4 found signal or not. | — | — |
| F6.2 | Comparison diagram | Greenwashing vs AI washing on: verifiability of claims, regulatory regime, time to observable outcome, cost of credible signal, typical venue. Drawn from Delmas & Burbano's drivers framework. | AI washing has *shorter* time-to-outcome (products ship or don't) and *lower* signal cost than green claims → the "X-washing" cycle should be faster. | — |
| T6.4 | Table | Generalised X-washing lens: the six frame fields restated as questions applicable to any technology narrative (quantum, blockchain, ESG). | — | — |

---

## Appendices (excluded from page count)

- **A. Pipeline**: embeddings, lexical gate term list, 39 prefilter signals with importance, decision-tree structure, frame schema with full concept list, extraction rules, throughput and cost.
- **B. Validation**: per-channel holdout tables; human coding protocol and instructions; per-field confusion matrices; error taxonomy with examples; inter-coder agreement; LLM-vs-Gemini κ.
- **C. Robustness**: alternative k, alternative clusterings, full regression tables with all coefficients, placebo and alternative-intensity DiD, annual-only replication.
- **D. Variable construction**: XBRL tag mapping, fiscal-year alignment, market-model estimation.

---

# Sources

Grouped by the role they play. ✔ = verified in this session (title, outlet, year); ◻ = standard reference cited from memory — confirm volume/pages before submission.

## AI disclosure and AI washing (direct empirical predecessors)

- ✔ Basnet, A., Elias, M., Salganik-Shoshan, G., Walker, T., Zhao, Y. (2025). Analyzing the market's reaction to AI narratives in corporate filings. *International Review of Financial Analysis*, 105, 104378. — Actionable vs speculative AI narratives in 10-K Item 1; only actionable narratives earn a valuation premium. Closest predecessor; the thesis replaces their manual binary with a multi-field frame schema.
- ✔ Song, X., Hou, W., Ouyang, Z., Hao, F. (2026). AI washing: Strategic disclosure and backlash. *Finance Research Letters*, 95, 109684. — BERT-classified AI claims vs AI patents, 2018–2023; unsubstantiated claims attract attention then underperform. Defines AI washing as narrative–outcome decoupling.
- ✔ Anantharaman, D., Parker, C., Rozario, A. M. (2026). AI Disclosure in Annual Reports: Evolution, Determinants, and Capital Market Participants' Perceptions. *Journal of Information Systems*. — Large-sample 10-K evidence 2017–2024; AI risk, ethics and opportunity disclosures. Use for descriptive benchmarks in Ch. 5.1.
- ✔ Babina, T., Fedyk, A., He, A., Hodson, J. (2024). Artificial intelligence, firm growth, and product innovation. *Journal of Financial Economics*, 151, 103745. — Resume-based measure of AI investment; AI-investing firms grow through product innovation. The "substance" benchmark the thesis cannot observe directly; motivates R&D and growth as proxies.
- ◻ Eisfeldt, A. L., Schubert, G., Zhang, M. B. (2023). Generative AI and Firm Values. NBER Working Paper 31222 / SSRN 4436627. — Occupational exposure to generative AI and returns after ChatGPT.
- ◻ Campbell, J. L. et al. (2025). Artificially Intelligent or Artificially Inflated? Determinants and Consequences of AI Disclosure. Working paper, Utah Winter Accounting Conference. — AI disclosure across 10-K, calls and earnings announcements, paired with AI employee investment. Directly relevant to the venue comparison.
- ✔ Strauss, I., O'Reilly, T., Rosenblat, M., Moore, S. (2025). Governing AI Through SEC Disclosure. SSRC Working Paper 04, October 2025. — 8-K AI disclosures by item; argues for a 10-K AI mandate. Context for Ch. 1.2 and for the thin 8-K layer.
- ◻ Blankespoor, E., deHaan, E., Li, Q. (2024). Generative AI in Financial Reporting. SSRN 4986017. — Firms are themselves using GAI to write disclosures; a confound worth one paragraph in limitations.
- ◻ Quantifying a firm's AI engagement: constructing objective, data-driven AI stock indices using 10-K filings (2025). *Technological Forecasting and Social Change*. — Term-frequency AI score for NASDAQ firms 2010–2022; the "volume" baseline the thesis argues is insufficient.

## Regulatory context

- ✔ U.S. Securities and Exchange Commission (2024). SEC Charges Two Investment Advisers with Making False and Misleading Statements About Their Use of Artificial Intelligence. Press Release 2024-36, 18 March 2024. sec.gov
- ◻ Gensler, G. (2023). "AI, Finance, Movies, and the Law." Remarks before the Yale Law School, 5 December 2023. — First public use of "AI washing" by the SEC Chair.
- ◻ SEC Division of Examinations (2024). 2024 Examination Priorities. — AI listed as an emerging risk area.
- ◻ Arnold & Porter (2024). SEC Targets "AI Washing". Client advisory, March 2024. (Already in the proposal; keep as practitioner source, not academic.)
- ◻ SEC (2023). Proposed Rule: Conflicts of Interest Associated with the Use of Predictive Data Analytics by Broker-Dealers and Investment Advisers. Release No. 34-97990.

## Disclosure theory (Ch. 1.3, lenses 1–2)

- ◻ Verrecchia, R. E. (1983). Discretionary disclosure. *Journal of Accounting and Economics*, 5, 179–194.
- ◻ Dye, R. A. (1985). Disclosure of nonproprietary information. *Journal of Accounting Research*, 23(1), 123–145.
- ◻ Spence, M. (1973). Job market signaling. *Quarterly Journal of Economics*, 87(3), 355–374.
- ◻ Crawford, V. P., Sobel, J. (1982). Strategic information transmission. *Econometrica*, 50(6), 1431–1451.
- ◻ Milgrom, P., Roberts, J. (1986). Relying on the information of interested parties. *RAND Journal of Economics*, 17(1), 18–32.
- ◻ Healy, P. M., Palepu, K. G. (2001). Information asymmetry, corporate disclosure, and the capital markets: a review of the empirical disclosure literature. *Journal of Accounting and Economics*, 31, 405–440.
- ◻ Beyer, A., Cohen, D. A., Lys, T. Z., Walther, B. R. (2010). The financial reporting environment: review of the recent literature. *Journal of Accounting and Economics*, 50, 296–343.
- ◻ Merkl-Davies, D. M., Brennan, N. M. (2007). Discretionary disclosure strategies in corporate narratives: incremental information or impression management? *Journal of Accounting Literature*, 26, 116–196.
- ◻ Bloomfield, R. (2008). Discussion of "Annual report readability, current earnings, and earnings persistence". *Journal of Accounting and Economics*, 45, 248–252. — Obfuscation hypothesis.

## Legitimacy, institutional theory, decoupling (lens 3)

- ◻ Suchman, M. C. (1995). Managing legitimacy: strategic and institutional approaches. *Academy of Management Review*, 20(3), 571–610.
- ◻ DiMaggio, P. J., Powell, W. W. (1983). The iron cage revisited: institutional isomorphism and collective rationality in organizational fields. *American Sociological Review*, 48(2), 147–160.
- ◻ Meyer, J. W., Rowan, B. (1977). Institutionalized organizations: formal structure as myth and ceremony. *American Journal of Sociology*, 83(2), 340–363. — Origin of talk–action decoupling.
- ◻ Bromley, P., Powell, W. W. (2012). From smoke and mirrors to walking the talk: decoupling in the contemporary world. *Academy of Management Annals*, 6(1), 483–530.

## Greenwashing and technology-label hype (lens 4, Ch. 6.3)

- ◻ Delmas, M. A., Burbano, V. C. (2011). The drivers of greenwashing. *California Management Review*, 54(1), 64–87.
- ◻ Lyon, T. P., Montgomery, A. W. (2015). The means and end of greenwash. *Organization & Environment*, 28(2), 223–249.
- ◻ Marquis, C., Toffel, M. W., Zhou, Y. (2016). Scrutiny, norms, and selective disclosure: a global study of greenwashing. *Organization Science*, 27(2), 483–504. — Scrutiny reduces selective disclosure; the direct analogue for H5.
- ◻ Cooper, M. J., Dimitrov, O., Rau, P. R. (2001). A rose.com by any other name. *Journal of Finance*, 56(6), 2371–2388. — Dot-com name changes.
- ◻ Cheng, S. F., De Franco, G., Jiang, H., Lin, P. (2019). Riding the blockchain mania: public firms' speculative 8-K disclosures. *Management Science*, 65(12), 5901–5913. — Speculative vs substantive blockchain 8-Ks; template for the realized/hypothetical split.
- ◻ Akyildirim, E., Corbet, S., Sensoy, A., Yarovaya, L. (2020). The impact of blockchain related name changes on corporate performance. *Journal of Corporate Finance*, 65, 101759.

## Financial text analysis and investor processing (Ch. 2, Ch. 6.2)

- ◻ Loughran, T., McDonald, B. (2011). When is a liability not a liability? Textual analysis, dictionaries, and 10-Ks. *Journal of Finance*, 66(1), 35–65.
- ◻ Loughran, T., McDonald, B. (2016). Textual analysis in accounting and finance: a survey. *Journal of Accounting Research*, 54(4), 1187–1230.
- ◻ Li, F. (2008). Annual report readability, current earnings, and earnings persistence. *Journal of Accounting and Economics*, 45, 221–247.
- ◻ Li, F. (2010). The information content of forward-looking statements in corporate filings — a naïve Bayesian machine learning approach. *Journal of Accounting Research*, 48(5), 1049–1102. — Forward-looking statement classification; precedent for the `temporal` field.
- ◻ Huang, X., Teoh, S. H., Zhang, Y. (2014). Tone management. *The Accounting Review*, 89(3), 1083–1113.
- ◻ Hoberg, G., Phillips, G. (2016). Text-based network industries and endogenous product differentiation. *Journal of Political Economy*, 124(5), 1423–1465.
- ◻ Gentzkow, M., Kelly, B., Taddy, M. (2019). Text as data. *Journal of Economic Literature*, 57(3), 535–574.
- ◻ Dyer, T., Lang, M., Stice-Lawrence, L. (2017). The evolution of 10-K textual disclosure: evidence from Latent Dirichlet Allocation. *Journal of Accounting and Economics*, 64, 221–245.
- ◻ Brown, S. V., Tucker, J. W. (2011). Large-sample evidence on firms' year-over-year MD&A modifications. *Journal of Accounting Research*, 49(2), 309–346. — Boilerplate persistence; relevant to sticky archetypes.
- ◻ Kim, A., Muhn, M., Nikolaev, V. (2023). Bloated disclosures: can ChatGPT help investors process information? Working paper, Chicago Booth. — LLMs as disclosure-processing instruments.
- ◻ Lopez-Lira, A., Tang, Y. (2023). Can ChatGPT forecast stock price movements? Return predictability and large language models. SSRN 4412788.
- ◻ Bochkay, K., Brown, S. V., Leone, A. J., Tucker, J. W. (2023). Textual analysis in accounting: what's next? *Contemporary Accounting Research*, 40(2), 765–805.
- ◻ Chen, X., Hu, T., Roberts, K. (2024). Large language models in finance: a survey. Working paper. (Confirm venue.)

## Earnings calls as a venue (Ch. 5.3, H6)

- ◻ Matsumoto, D., Pronk, M., Roelofsen, E. (2011). What makes conference calls useful? The information content of managers' presentations and analysts' discussion sessions. *The Accounting Review*, 86(4), 1383–1414.
- ◻ Larcker, D. F., Zakolyukina, A. A. (2012). Detecting deceptive discussions in conference calls. *Journal of Accounting Research*, 50(2), 495–540.
- ◻ Price, S. M., Doran, J. S., Peterson, D. R., Bliss, B. A. (2012). Earnings conference calls and stock returns: the incremental informativeness of textual tone. *Journal of Banking & Finance*, 36, 992–1011.

## Governance disclosure and proxy statements (DEF 14A layer)

- ◻ Larcker, D. F., Tayan, B. (2020). *Corporate Governance Matters* (3rd ed.). Pearson. — Board oversight and proxy disclosure.
- ◻ Confirm one recent paper on board AI oversight / skills matrices (search "board AI expertise disclosure proxy statement 2025").

## Methods: clustering, DiD, validation

- ◻ Callaway, B., Sant'Anna, P. H. C. (2021). Difference-in-differences with multiple time periods. *Journal of Econometrics*, 225(2), 200–230.
- ◻ Roth, J., Sant'Anna, P. H. C., Bilinski, A., Poe, J. (2023). What's trending in difference-in-differences? A synthesis of the recent econometrics literature. *Journal of Econometrics*, 235(2), 2218–2244.
- ◻ Callaway, B., Goodman-Bacon, A., Sant'Anna, P. H. C. (2024). Difference-in-differences with a continuous treatment. Working paper / NBER. — The exact design used in T5.2.
- ◻ Bertrand, M., Duflo, E., Mullainathan, S. (2004). How much should we trust differences-in-differences estimates? *Quarterly Journal of Economics*, 119(1), 249–275.
- ◻ Rousseeuw, P. J. (1987). Silhouettes: a graphical aid to the interpretation and validation of cluster analysis. *Journal of Computational and Applied Mathematics*, 20, 53–65.
- ◻ McInnes, L., Healy, J., Melville, J. (2018). UMAP: uniform manifold approximation and projection for dimension reduction. arXiv:1802.03426.
- ◻ Cohen, J. (1960). A coefficient of agreement for nominal scales. *Educational and Psychological Measurement*, 20(1), 37–46.
- ◻ Krippendorff, K. (2018). *Content Analysis: An Introduction to Its Methodology* (4th ed.). Sage. — Coding protocol and agreement thresholds for App. B.
- ◻ Chen, J., Xiao, S., Zhang, P., Luo, K., Lian, D., Liu, Z. (2024). BGE M3-Embedding. arXiv:2402.03216. — Embedding model.
- ◻ Gilardi, F., Alizadeh, M., Kubli, M. (2023). ChatGPT outperforms crowd workers for text-annotation tasks. *PNAS*, 120(30). — Justifies LLM-as-judge with human validation.
- ◻ Törnberg, P. (2024). Best practices for text annotation with large language models. *Sociologica*. — Same.

## Data sources (for the bibliography's "sources of data" requirement)

- SEC EDGAR full-text and XBRL Financial Statement Data Sets. sec.gov
- Earnings call transcripts: name the provider actually used.
- Daily prices and index data: name the provider actually used.
- GICS classification: S&P Dow Jones Indices / MSCI.
