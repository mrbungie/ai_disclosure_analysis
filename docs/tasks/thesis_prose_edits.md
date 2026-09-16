# Proposed prose edits after the gold/results migration

Numbers below come from the regenerated pipeline (deterministic prefilter
population, silver restricted to the analysis universe, regenerated gold, call
merge-fanout fix). Inline `{python}` values already update themselves; these are
hardcoded numbers, captions or claims that no longer match. None has been applied.
Full per-value diffs: `data/results/thesis_baseline/diff_*.csv`.

## 1. Crash risk: Governance-Led stated as a significant finding

- Where: Market Relevance chapter (fig-call-crash-risk discussion, ~3 sentences),
  Appendix E (two places), and every place that elevates it to a secondary finding
  (Executive Summary, §8.1/§8.2, Chapter 7 checklist — see commit b064afb).
- Current text: "β = −0.124, p = 0.046 … clears the conventional 5% threshold uncorrected."
- Data now: NCSKEW `dum_gov` β ≈ −0.06, p ≈ 0.22; DUVOL β ≈ −0.09, p ≈ 0.06.
- Sensitivity: the result depends on ~227 texts from the latest prefilter
  deployment (docs/tasks/README.md, known data notes); no code cause.
- Proposed: "Governance-Led posture is not associated with lower NCSKEW
  (β ≈ −0.06, p ≈ 0.22); DUVOL has the same sign with marginal significance
  (β ≈ −0.09, p ≈ 0.06), so the association is suggestive at most and does not
  survive changes in the prefilter population." Demote it from secondary finding
  in the Executive Summary and checklist. The Conclusions chapter already states it
  as non-significant.

## 2. Archetype battery: "smallest p-value" detail

- Market Relevance: Vocal Substantives / gross margin described as β = −0.041,
  p = 0.046, the smallest p-value of the 14 tests.
- Data now: p = 0.072, second-smallest (Governance-Led / log market cap, p = 0.071).
- Conclusion unchanged (nothing survives Benjamini–Hochberg). Proposed: drop the
  "smallest" claim or update both values.

## 3. fig-call-crash-risk sample note

- Stated: N = 8,642, 428 firms, R² = 0.113. Data now: N = 8,220, 445 firms, R² = 0.066.

## 4. Null-coefficient p-value floors

- "p ≥ 0.14" → 0.117; "p ≥ 0.13" → 0.071. Conclusions hold; the stated bounds do not.

## 5. NLP horse race

- "+4.2 pp" → about +3.7 to +4.0 pp. "ΔR² ≤ 0.0009" is exceeded by price_to_sales
  (ΔR² = 0.0016); "near-zero" still holds. Proposed: state the maximum ΔR² observed.

## 6. Appendix G roll-up sensitivity

- "roughly 15–18 pp below" pools two channels: filings ≈ 16.6 pp, calls ≈ 11.9 pp.
  Proposed: report each channel.

## 7. Archetype document channels (Chapter 3)

- "Aligns with Form 10-K Item 1A Risk Factors (r = +0.763, R² = 58.2%)" and the
  other two channel correlations: data now r = 0.771 / 0.681 / 0.507
  (data/results/posture/archetype_document_channels.json).
- "DEF 14A coverage reaches 100%": now exactly 450/450; the "active omission"
  reading of a low w_Gov weight stands.

## 8. tbl-activity-schema "Business Domain" row

- Lists 6 domains (Operate, Sell, Build, Control, Serve, Enable); the current
  activity taxonomy exposes 18 `function_family` values. Decide whether the table
  describes the 6-domain grouping (then document where it is derived) or the
  18-family field.

## Crash risk: new point-in-time specification (2026-09-15)

`call_crash_regressions.py` now estimates NCSKEW/DUVOL on W, HistW, cumulative
disclosure intensity (posture channels, expanding) and the three posture
archetype weights (trailing 12 months), all attached as-of the call date;
targets `ncskew_base/weights`, `duvol_base/weights`, contrasts `w_gov-w_def` etc.
The figure chunk already reads `ncskew_weights`. Prose, figure note (N, R²,
β, p), the H6c summary row, Appendix E sector/switcher checks and the 18-test
BH family still quote the old dummy specification (β −0.124, p 0.046,
N 8,642) and must be rewritten from
data/results/crash_archetypes/call_crash_archetype_regressions.csv
(N 8,222; weights vs no posture: w_def NCSKEW +0.166 p 0.001, DUVOL +0.188 p <0.001; w_gov and w_voc n.s.; intensity
−0.036 p 0.011 / −0.047 p 0.003; W and HistW n.s.).

## Archetype battery: same point-in-time specification (2026-09-15)

`call_archetype_full_battery.py` now uses W, HistW, cumulative intensity and the
three archetype weights (no hist/surprise substance or disclosure); the
18-test BH family is the Gov−Def and Voc−Def contrasts over 9 outcomes, computed
in code (no hand-entered p-values); nothing survives (smallest p 0.014, DUVOL
Gov−Def). Robustness: sector exclusion Gov−Def NCSKEW −0.254 (p 0.108), DUVOL
−0.281 (p 0.078); switchers −0.164 (p 0.090), −0.219 (p 0.028). The figure
(fig-economic-coefficients-c2) and tbl-e-archetype-regressions chunks read the
new variables; the surrounding prose (Channel 2 "operational premiums for
disclosed substance", table note, Appendix E checks) still describes the
substance specification and must be rewritten.

## Gold rebuild: variables and tables that no longer exist (2026-09-15)

Gold now has only spines, covariates and targets by grain (docs/gold_pipeline.md).
Code chunks were repointed; the prose below still describes removed constructs.

- **Annual archetype composition (fig-archetype-composition-annual, notes and the
  paragraph after it).** The yearly walk-forward archetype
  (`predictions/firm_year/posture_archetype_expanding`) is removed; the figure now
  uses the static posture archetype applied to each firm-year's posture
  (`covariates/firm_year/posture_archetype_static`; "No AI" = fewer than 3 filing
  frames that year). The note "assigned by Archetypal Analysis fit on every
  firm's own cumulative disclosed text through that year" no longer holds. Shares
  change in 2021-2023: Vocal Substantives 13.3% (2021) -> 16.3% (2026), not a
  tenfold rise; No AI 79.7% -> 10.5%; Defensive 4.0% -> 51.3%; Governance-Led
  3.0% -> 21.9%. Year-over-year persistence 86.8 / 80.9 / 56.5 / 59.8 / 63.3%.
  "Rises tenfold in relative terms" and "absorbed overwhelmingly by Defensive
  Disclosers, not by Vocal Substantives" must be rechecked against these values.
- **tbl-archetype-quadrant note.** "Archetype uses the expanding-window label ...
  classified from each firm's cumulative disclosure through the prior year; the
  small 'No AI' row ... reflects that one-year lag" -> the label is the static
  archetype of the 2026 firm-year; "No AI" rows are 2026 disclosers with fewer
  than 3 filing frames in 2026.
- **Appendix (vertex separation of "the annual archetype composition's expanding
  fits").** Refers to the removed yearly expanding fits.
- **Call-level design paragraphs (Market Relevance ANCOVA paragraph; Appendix E
  equation text; "Expanding-window archetype regressions across firm
  fundamentals" heading, caption and paragraph).** They list HistD, HistS,
  SurpriseD, SurpriseS, prior-year W_{t-1} and expanding-window archetype
  indicators with Defensive as reference; the call regressions use W, HistW,
  cumulative disclosure intensity and the three archetype weights (no omitted
  category). Several chunks of that section (tbl-call-beta-paper and the two
  chunks before it) still read `hist_disclosure`/`surprise_*` rows that the call
  results no longer contain and fail; this predates the gold rebuild.
- **Robustness battery prose.** The "disclosure-vs-substance" Wald tests are gone;
  the battery reports joint F of the AI block, joint F of the three weights and the
  partial R^2 of the AI block (data/results/call_beta/call_beta_robustness_*.csv,
  call_beta_robustness_formal_tests.json).
- **Market-factor appendix counts.** The panel is now firm-years with a price window
  around the 10-K (`beta_n_obs` not null): 2,671 cells / 452 tickers instead of
  2,685 / 455 (14 10-Ks of 3 tickers whose price series has no window around the
  filing were counted before); beta coverage 99.9% (was 99.4%), windows >= 240
  days 99.7% (99.2%), short-window firm-years 2 (16). Inline values update; any
  hardcoded mention must follow.
- **Revenue growth (t+1) as a call-level control.** `next_revenue_yoy_pre` (growth of
  the fiscal year after the last pre-call 10-K) is not known at the call and is
  stored as a target; the generalized-target and battery regressions still use it
  as the pre level of the revenue-growth outcome. Any prose calling the pre levels
  "predetermined" should exclude this one (or the specification should change).
- **Market Relevance, baseline post-call beta paragraph (after
  fig-call-beta-baseline).** The chunks now read the call regressors `w`,
  `hist_w`, `intensity_expanding`, `w_voc`, `w_gov`, `w_def`; the sentence still
  names historical disclosure intensity, single-call surprises and
  historical/single-call substance. Inline slots were repointed as
  historical disclosure intensity -> cumulative disclosure intensity
  (`intensity_expanding`: +0.030, p = 0.011); single-call surprise -> call
  decoupling W (+0.002, p = 0.801); historical substance -> HistW (+0.055,
  p < 0.001); single-call substance -> Vocal Substantives weight (+0.042,
  p = 0.261; no single-call substance regressor exists). The rest of the block:
  Governance-Led weight -0.023 (p = 0.675), Defensive weight -0.049 (p = 0.079).
  HistW, not intensity, is the strongest beta correlate; the sentence must be
  rewritten around W / HistW / intensity. Figure note: N = 8,637 calls, 446
  firms, 316 cells unchanged; R^2 = 0.626 (note says 0.628).
- **fig-economic-coefficients-c1 paragraph.** Series are now cumulative
  intensity, HistW and W. Hardcoded values (market cap +0.086, p < 0.001;
  revenue growth +0.147, p = 0.014; price-to-sales +0.013, p = 0.055) are for the
  removed HistDisclosure. Cumulative intensity: log market cap +0.059
  (p < 0.001), price-to-sales +0.001 (p = 0.729); HistW: market cap +0.015
  (p = 0.018), price-to-sales +0.005 (p = 0.188); W null on both. Revenue growth:
  see the revenue-growth control entry below.
- **Placebo sentence (Channel 1) and H6a summary row / structural-limits
  paragraph.** `rob_placebo_d` is now pre-call beta on cumulative intensity:
  beta = 0.113 (p < 0.001), "nearly double" -> about 3.8x the post-call +0.030.
  "substance null" has no counterpart: HistW placebo is +0.120 (p < 0.001), W
  +0.029 (p = 0.002), Governance-Led weight -0.316 (p = 0.005). The H6a row's
  hardcoded market cap (+0.086) and revenue growth (+0.147) must follow the
  values above.
- **tbl-call-beta-paper (Appendix E) and its note.** Rows are now W, HistW,
  cumulative intensity and the three weights. Cumulative intensity: baseline
  0.030 (p = 0.011), + accounting 0.018 (p = 0.071), FF3 0.035 (p = 0.003),
  [+21,+252] 0.048 (p < 0.001), firm FE -0.134 (p = 0.001), delta beta 0.007
  (p = 0.502). HistW: 0.055, 0.061, 0.079, 0.071 (all p < 0.001), firm FE -0.009
  (p = 0.751), delta beta 0.061 (p < 0.001). N: 8,637 / 5,949 / 8,257 / 7,398 /
  8,270 / 8,257; R^2 0.626 / 0.622. The note's "HistDisclosure remains positive
  across every specification" no longer holds for intensity (negative within
  firm); HistW is positive in every column except firm FE (null).
- **Revenue growth (t+1) control (supersedes the entry above).** The pre-call
  control of `next_revenue_yoy_post` is now `revenue_yoy_pre`, the revenue growth
  the last pre-call 10-K reports (its fiscal year over the prior one), known at
  the call. Revenue-growth row, generalized targets and archetype battery:
  N 6,334 calls / 441 firms -> 6,335 / 440; R^2 0.400 -> 0.399; partial R^2 of
  the AI block 0.0064 -> 0.0061. Cumulative intensity +0.093 (p = 0.010) ->
  +0.091 (p = 0.009); HistW -0.000 (p = 0.999) -> -0.001 (p = 0.942); W -0.012
  (p = 0.134) -> -0.012 (p = 0.126); Vocal Substantives weight -0.062
  (p = 0.302) -> -0.059 (p = 0.320); Governance-Led -0.051 (p = 0.470) -> -0.081
  (p = 0.274); Defensive -0.028 (p = 0.679) -> -0.024 (p = 0.714). BH family:
  still two survivors (Defensive weight on DUVOL and NCSKEW). Prose that calls
  every pre level "predetermined" now holds; the hardcoded "revenue growth
  +0.147, p = 0.014" (Channel 1 paragraph and H6a row) becomes cumulative
  intensity +0.091 (p = 0.009).

## Point-in-time MEDIUM fixes: firm-year W and posture within the year, switchers as of the call (2026-09-15)

The firm-year washing score W (grounding priors, percentile ranks, 5% tails) and
the firm-year posture covariate (shrinkage prior, disclosure_intensity rank) are
now fitted within each calendar year's cross-section; the posture-switcher sample
of the archetype battery is defined at each call. All 246 inline values evaluate;
these changed (old -> new):

- **Persistence of the decoupling posture (callout after fig-washing-distribution)
  and "What credibility means", item 1.** Year-over-year Spearman rho +0.636 ->
  +0.595; the rounded rho in item 1 0.64 -> 0.60.
- **Welltower case (fig-sec-welltower paragraph).** W at k = -1 0.225 -> 0.140,
  at k = 0 0.011 -> -0.140; "closed its decoupling gap" still holds (both years
  now on the within-year scale).
- **Operational portfolio screens.** Extreme Decoupling (FY2026 firm-years above
  the pooled 95th percentile of W) 34 -> 20; disclosers with negative W 137 ->
  219. W now has mean 0 in every year, so the 2026 cross-section no longer sits
  above the pooled distribution; "95th percentile of W in the full 2021-2026
  panel" still describes the computation.
- **Out-of-sample temporal validation (Appendix C).** 2021-2024 -> 2025: ARI
  0.480 -> 0.321, Jaccard range 34.6-77.2% -> 50.9-68.9%. 2021-2023 -> 2025: ARI
  0.328 -> 0.026, Vocal Substantives 45.2% -> 10.9%. 2021-2025 -> 2026: ARI
  0.705 -> 0.552. "Even the stricter pre-GenAI 2021-2023 training window keeps
  meaningful post-2023 recovery" no longer holds (ARI 0.026) and must be
  rewritten; the closing "not an artifact of the initial commercialization wave"
  is weaker.
- **tbl-c4-washing-validation note (Appendix E).** "W tracks R&D intensity
  negatively without tracking firm size, valuation, or risk" no longer holds:
  R&D intensity rho -0.044 (p 0.193) -> +0.065 (p 0.058); log market cap -0.047
  (p 0.048) -> -0.061 (p 0.010); beta -0.071 (p 0.003) -> +0.037 (p 0.117);
  idiosyncratic volatility +0.022 (p 0.349) -> +0.048 (p 0.046); P/S +0.018
  (p 0.449) -> +0.054 (p 0.023). Mechanical consistency vs Grounding -0.563 ->
  -0.463; split-half rho 0.600 -> 0.607 (table values, read from the JSON).
- **Incremental-value decomposition (fig-nlp-horse-race, washing layer).**
  Delta R^2 of W: P/S 0.0016 -> 0.0001, R&D/sales 0.0003 -> 0.0014, revenue
  growth 0.0001 -> 0.0008, idiosyncratic volatility 0.0015 -> 0.0017, beta 0.0000
  unchanged (inline hd_inc_* values unchanged at displayed precision).
- **Appendix E "sector exclusion and posture switchers".** Switchers are now the
  calls at which the firm has already shown two different dominant archetypes
  (argmax of the as-of weights) at that or an earlier call, plus its later calls:
  5,046 calls / 274 firms -> 2,591 / 246. Weights vs no posture, NCSKEW:
  Defensive +0.118 (p 0.077) -> +0.259 (p 0.049), Governance-Led +0.005
  (p 0.958) -> +0.080 (p 0.609), Vocal +0.089 (p 0.230) -> +0.144 (p 0.318);
  DUVOL: Defensive +0.173 (p 0.013) -> +0.339 (p 0.008), Governance-Led -0.008
  (p 0.941) -> +0.106 (p 0.495), Vocal +0.099 (p 0.173) -> +0.209 (p 0.122). The
  paragraph's "firm-years that switch into or out of the Governance-Led archetype
  over the panel" and its hardcoded beta = -0.060 (p = 0.287) are stale and must
  describe the as-of definition.
- Appendix G cohort grounding table and the firm comparison table (JPM, GS,
  MSFT, AAPL, WELL) read the new grounding index and W from gold/results; e.g.
  AAPL 2024 W 0.176 -> 0.140, WELL 2025 0.225 -> 0.140.

## Earnings-call manifest checks, delisted firms, CAR/SUE removal, point-in-time incremental signal (2026-09-16)

Silver keeps one results call of the firm per fiscal quarter (other companies'
calls, non-results events, duplicate transcripts and calls not yet enriched
excluded; the call's firm, date and fiscal period read from the transcript)
and drops every document and price of a delisted firm after its delisting
date. The incremental-signal decomposition moved to the firm-quarter grain.
All chunks and 246 inline expressions evaluate; the values that moved:

- **Corpus counts (Data chapter, Appendix A).** Documents 59,818 -> 59,392;
  AI frames 70,353 -> 69,186; disclosed activities 47,767 -> 46,531; firms
  with activities 477 -> 471; balanced-panel firms 466 -> 465; 2026 coverage
  10-K 424 -> 423, 10-Q 464 -> 463, calls 464 -> 451; 2025 10-K 474 -> 473;
  firm-years 2,893 -> 2,887; washing-score panel 1,885 -> 1,882 firm-years,
  476 -> 475 firms.
- **Posture chapter.** Firms in the archetype fit 450 -> 449 (No AI 48 -> 49);
  governance-promotional correlation 0.12 -> 0.13; Defensive share 2026 51.3
  -> 51.2%; transitions No AI -> Vocal max 4.8 -> 4.9%, No AI -> Defensive
  2021-2022 2.0 -> 2.1% and 2025-2026 47.8 -> 47.2%, No AI -> Governance-Led
  2025-2026 20.0 -> 20.2%; persistence 2023-2024 56.5 -> 56.1%, 2024-2025 59.8
  -> 59.7%, 2025-2026 63.3 -> 63.4%; Vocal/Defensive ratio 2.26 -> 2.15,
  Governance ratio 1.59 -> 1.61; volume R^2 0.70 -> 0.72; 10-K refit r Vocal
  0.926 -> 0.927, Governance-Led 0.273 -> 0.274, common tickers 422 -> 421;
  k = 2 vertices +2.05 -> +2.04 and -0.02 -> -0.01; PCA variance 42.9/18.1 ->
  43.0/18.2 (61.1 -> 61.2 total); 2026 out-of-sample firms 417 -> 416, ARI
  2021-2025 0.552 -> 0.551; intensity estimate 2026 0.196 -> 0.197.
- **Activity inventory.** Ladder shares 87.7/65.7/44.3 -> 87.8/65.8/44.2;
  growth ratio 5.2 -> 5.1, projection 7.6 -> 7.5; deployed share 2026 73 ->
  72%, named product 46 -> 45%; own-brand 87.2 -> 87.4%, external vendor 5.4
  -> 5.3%, firms with an external provider 66.9 -> 67.3%; vendor shares OpenAI
  / Microsoft 27.3 -> 27.0, Google 17.2 -> 16.8, NVIDIA 12.4 -> 12.1, AWS 9.6
  -> 9.3; sector matrix n 224 -> 222, ratio 6.8 -> 7.0, cells 24.8 -> 24.6 and
  35.2 -> 33.9; Industrials deploy deviation -8.3 -> -8.4.
- **Channel gap.** Cells 796 -> 786, firms 289 -> 283; internal-deployment p
  0.25 -> 0.23, customer-facing gap +7.4 -> +7.2 pp; named function on calls
  +6.7 -> +6.5 pp, quantified outcome +9.2 -> +9.3 pp, named product +8.8 ->
  +9.0 pp, governance -10.1 -> -10.2 pp; promotional per call 0.118 -> 0.119,
  quantified 0.074 -> 0.076, ratio 19.3 -> 19.5.
- **Call-level beta regression (beta_post_63).** N 8,637 calls / 446 firms ->
  8,583 / 445; W +0.002 (p 0.801) -> (p 0.832); HistW +0.055 -> +0.053
  (p < 0.001); cumulative intensity p 0.011 -> 0.009; w_voc +0.042 (p 0.261)
  -> +0.041 (p 0.269); placebo coefficient 0.113 -> 0.127. No sign or
  significance change.
- **Crash risk (NCSKEW, DUVOL).** Sample 8,222 -> 8,156 calls; W and HistW
  stay insignificant, cumulative intensity and the Defensive weight keep their
  sign and significance (NCSKEW intensity -0.037 -> -0.038, w_def +0.166 ->
  +0.163; DUVOL intensity -0.048 -> -0.049, w_def +0.188 -> +0.186).
- **Decoupling validation and screens.** Persistence pairs 1,373 -> 1,371,
  rho +0.595 -> +0.596; screen counts: validate 37.4 -> 37.5%, deprioritize
  188 -> 187 (40.3 -> 40.2%), indexed firms 466 -> 465, disclosing 444 -> 443,
  dual-channel 452 -> 440; comment-letter panel n 1,502 -> 1,501.
- **Incremental-value decomposition (fig-nlp-horse-race, Chapter 6 and
  Appendix C).** The decomposition is now point in time on firm-quarters:
  predictors are the firm-quarter covariates known at `as_of_date` (mention
  counts, posture rates and archetype weights, disclosed activities, the
  decoupling index W, all over the last four quarters of 10-K/10-Q/8-K/DEF
  14A text) and outcomes are measured after it (next-quarter beta and
  idiosyncratic volatility, price-to-sales at the next quarter end, and the
  R&D intensity and revenue growth of the fiscal quarter ending after the
  as-of date). Delta R^2 over fundamentals and sector x quarter fixed effects:

  | outcome | n | mentions | posture | activity | washing |
  |---|---|---|---|---|---|
  | next-quarter beta | 4,406 | +0.0565 | +0.0182 | +0.0139 | +0.0008 |
  | next-quarter idiosyncratic volatility | 4,406 | +0.0347 | +0.0164 | +0.0156 | +0.0014 |
  | price-to-sales at the next quarter end | 3,670 | +0.0007 | +0.0127 | +0.0234 | +0.0004 |
  | R&D intensity of the next fiscal quarter | 1,896 | +0.0000 | +0.0284 | +0.0282 | +0.0085 |
  | revenue growth of the next fiscal quarter | 3,828 | +0.0260 | +0.0035 | +0.0174 | +0.0000 |

  The prose keeps its finding: valuation is explained by how AI is described
  (posture +1.3 pp and disclosed activity +2.3 pp of R^2, mention counts
  +0.1 pp), market risk by how much (mentions +5.7 pp on beta, +3.5 pp on
  idiosyncratic volatility), and W adds almost nothing anywhere (<= 0.0009
  outside R&D intensity, where it adds 0.0085 on the smaller disclosing
  subsample). The hard-coded "+4.2 percentage points" for price-to-sales
  becomes +3.7 pp (posture + activity) and the models are relabelled: M2 is
  the posture block (seven posture rates and the archetype weights), M3 adds
  the disclosed activities, M4 the decoupling index. Appendix C's description
  of Model 0 ("sector-by-year fixed effects") is now sector-by-quarter, and
  the robustness list (10-K only text, excluding IT/communications, composition
  shares, permutation test) is replaced by firm fixed effects on the same
  panel.
- Hard-coded prose that quotes CAR, SUE or `car_m1_p5` must go, and the
  appendix crosscheck reports 9 FDR pairs, not 11 (the two CAR pairs are
  removed).
