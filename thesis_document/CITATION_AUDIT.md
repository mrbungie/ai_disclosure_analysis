# Citation audit — `thesis.qmd`

Audit date: 2026-09-08.  Scope: every external citation currently used in
`thesis.qmd`, checked against the local primary copy in `sources_pdf/` with
`pdftotext`/`pdfinfo`.  The locators now inserted in `thesis.qmd` identify the
**specific supporting passage**.  Where a journal's printed pagination is
visible, it is used; otherwise the locator is the PDF/document page.  `@tbl-*`
and `@fig-*` are internal Quarto cross-references, not bibliography citations.

## Corrections made

- Removed the citations to Eisfeldt and Babina from the thesis's own 2021--25
  corpus statistics.  Those numbers are results of this corpus, not results in
  either paper.  Anantharaman et al. is now cited only for its independently
  documented post-ChatGPT increase in 10-K disclosure.
- Corrected Gensler's date and bibliographic metadata: the local source is the
  Yale Law School speech of **13 February 2024**, not a December 2023 speech.
- Replaced the unsupported Strauss-to-Welltower attribution with the actual SEC
  comment letter held locally at
  `data/raw/sec_letters/WELL_0000000000-25-004473.txt.gz` (28 April 2025,
  p. 1).  The revised prose states exactly what the letter says and no longer
  treats it as a finding that a claim was false or that a universal
  cross-venue rule exists.
- Removed `@verrecchia1983` and `@bromley2012`: both remain in the bibliography
  but their PDF copies are absent from `sources_pdf/`, so they cannot meet this
  audit's evidence standard.
- Removed `@babina2024` and `@ziems2024` from the text.  The files bearing
  those names are not their bibliographic works: `babina2024.pdf` is *Option-
  Implied Spreads and Option Risk Premia*; `ziems2024.pdf` is *Stance Detection
  with Explanations*.  Neither supports the former claims.
- Removed `@campbell2001` and `@ang2006` from the thesis's own residual-
  volatility definition.  The local papers discuss idiosyncratic volatility,
  but do not establish that this exact 252-day one-factor implementation is
  theirs.
- Narrowed several assertions that exceeded the cited documents: keyword
  studies are no longer described as all simple counts; patent data are
  recognized as a partial external observable; and signalling theory is used
  as motivation rather than proof that specific AI claims are costly or true.

## Evidence register

| Citation key | Specific passage checked | Grounded use after revision |
|---|---|---|
| `anantharaman2026` | pp. 1--2 | 10-K sample, keyword/LLM retrieval, and sharp post-ChatGPT rise. |
| `basnet2025` | p. 1, abstract | 10-K narratives classified as actionable, speculative, or irrelevant. |
| `bozanic2017` | pp. 1--5 | What SEC comment letters can request and their documented disclosure effects. |
| `callaway2024` | pp. 1--2 | DiD with a continuous treatment and its parallel-trends assumptions. |
| `crawford1982` | pp. 1431--32 | Sender--receiver model of strategic information transmission. |
| `crilly2012` | pp. 1429--31 | Intentional "faking it" versus emergent "muddling through." |
| `delmas2011` | pp. 64--69 | Greenwashing as communication relative to environmental performance. |
| `dimaggio1983` | pp. 147--48 | Coercive, mimetic, and normative isomorphism. |
| `dye1985` | pp. 123--24 | Disclosure incentives and costs. |
| `eisfeldt2023` | p. 9 | Firm exposure is built from O*NET tasks and Revelio employment shares—not annual reports. |
| `fayyad1996` | pp. 40--41 | The KDD sequence and Figure 1's stages. |
| `frankel1999` | pp. 133--34 | Conference calls as a voluntary disclosure medium. |
| `gensler2023` | PDF pp. 4--5 | The speech's dedicated "AI Washing" section. |
| `gentzkow2019` | pp. 1--2 | Text as an input to economic research. |
| `gilardi2023` | PDF pp. 1--2 | Evaluated ChatGPT annotation against crowd and trained annotators. |
| `hennig2007` | pp. 1--2 | Bootstrap distribution of the Jaccard coefficient for cluster stability. |
| `huang2014` | pp. 1083--86 | Tone management and its market consequences. |
| `li2008` | PDF pp. 1--3 | Annual-report/10-K readability measurement. |
| `loughran2011` | pp. 35--36 | Financial word lists and 10-K textual analysis. |
| `lyon2015` | pp. 1--3 | Greenwash as misleading communication about environmental performance. |
| `marquis2016` | pp. 483--85 | Selective disclosure as a symbolic legitimacy strategy. |
| `matsumoto2011` | pp. 1383--84 | Earnings calls as voluntary disclosure with less constrained manager communication. |
| `merkldavies2007` | pp. 116--18 | Discretionary narrative disclosure and impression management. |
| `meyer1977` | pp. 340--41 | Ceremonial structures, legitimacy, and decoupling from ongoing activity. |
| `rogers2011` | PDF pp. 1--2 | Relation between optimistic disclosure tone and shareholder litigation. |
| `sec2024pressrelease` | PDF pp. 1--2 | Delphia/Global Predictions actions and their legal basis. |
| `shearer2000` | pp. 14--19 | The six CRISP-DM phases. |
| `song2026` | pp. 1--2 | BERT/patent distinction between planned and implemented AI. |
| `spence1973` | pp. 363--64 | Signalling costs in the separating-equilibrium construction. |
| `suchman1995` | pp. 571--74 | Strategic and institutional approaches to legitimacy. |
| `welltower2025letter` | SEC letter, p. 1 | The staff's request after referencing the 10-K, call, and press release. |
| `westphal1994` | pp. 367--69 | Separation of symbolic adoption from substantive LTIP use. |

## Remaining guardrails

1. The page locators inserted in the revised prose identify the evidence for
   the claims most likely to be independently checked.  Citations in theory and
   methods tables remain concise; the register above gives their precise PDF
   evidence page.
2. The thesis's corpus sizes, percentages, estimates, tables, and figures are
   internal results.  They should be traceable to the reproducible pipeline and
   generated tables, not attributed to an outside paper.
3. The Welltower letter is a motivating regulatory instance, not an external
   validation label for the AI-washing index and not proof that any statement was
   false.  The revised text preserves that distinction.
