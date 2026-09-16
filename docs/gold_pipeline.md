# The gold data pipeline

Gold turns the silver/bronze layers (and the additive LLM outputs they carry)
into analysis-ready tables. It is deterministic, calls no LLM or API, and can
be rebuilt from zero with `make gold`. Analytics (`scripts/analytics/`) read
gold and write `data/results/<topic>/`; `thesis.qmd` reads results (and a few
gold tables directly).

```
spines      one row per unit of analysis of a grain (a firm-year, a call, ...):
            its id, keys and date. Nothing else joins rows onto a grain.
covariates  measurement families known as of the spine date (call: before or
            at the call; firm_quarter/firm_year: published before as_of_date).
targets     measurement families observed after the spine date (post-call
            windows, next fiscal year, post-filing windows).
datasets    one table per grain: the spine left-joined with every covariate
            and target family of the grain (plus the cross-grain columns its
            analytics use).
```

There is no `predictions` kind. Analytics read a grain's dataset
(`layers.read_dataset`, same selection syntax as `layers.read_gold`) rather
than joining families; `read_gold` remains for reading a single family. A
fitted model's output that is a measurement of the entity (archetype weights,
the washing score) is a covariate family; the fitted artifacts live in
`models/<name>/`.

## Layout

```
data/gold/<kind>/<grain>/<family>.parquet  (+ <family>._manifest.json)
  kind   spines | covariates | targets | datasets
  grain  document | activity | call | firm_quarter | firm_year | firm
data/gold/datasets/<grain>/<grain>.parquet  (one dataset per grain)
```

### Table rules

- Columns: `id`, the spine keys, the spine date, then values
  (`layers.GOLD_SPINE_COLUMNS`). Key and date values equal the spine's.
- Exactly one row per spine key, no row outside the spine; an entity without
  the information has nulls. Gold does not impute (analytics fill, e.g.
  `scripts/analytics/fills.py`). Zero counts are real counts over existing
  documents.
- Written only through `layers.write_gold`, which enforces the column order and
  key uniqueness, drops spine attributes from a family, and records the manifest. A dataset has the same rules
  (spine columns and rows) and is named after its grain. `make gold-check`
  (`scripts/gold/check_spine_alignment.py`) validates every table against its
  spine and fails on any other file under `data/gold/`.

### Grains and spines

| grain | spine columns (id, keys, date) | extra spine attributes | row set |
|---|---|---|---|
| document | `id`=accession_number, `ticker`, `accession_number`, `fecha` | `channel`, `form`, `cik`, `period_end`, `call_fy`, `fye_month`, `fy` | every US filing (10-K, 10-Q, 8-K, DEF 14A) and earnings-call transcript of the analysis universe with scorable paragraphs |
| activity | `id`=`{ticker}_{text_hash}_{activity_id}`, `ticker`, `text_hash`, `activity_id`, `fecha` | `accession_number` (representative document of the judged text) | one disclosed AI activity per firm (unique text x activity index); `fecha` = earliest date of a document of that ticker containing the text |
| call | `id`=call_accession_number, `ticker`, `call_accession_number`, `fecha` | `fiscal_period`, `call_sequence`, `sic`, `sic2`, `fe` (`{sic2}_{call year}`), `delisted`, `delisting_date` | one results call of the firm per fiscal quarter reported (silver.filing_manifest after bronze.call_transcripts: other companies' calls, non-results events and duplicate transcripts excluded; `fecha` and `fiscal_period` from the transcript where it states them); consecutive calls of a ticker at least 30 days apart with the fiscal period advancing k >= 1 quarters in the date window of k quarters (`call_sequence` gap for k > 1, label_break for a fiscal-year change), checked by `make gold-check` |
| firm_quarter | `id`=`{ticker}_{quarter}`, `ticker`, `quarter`, `as_of_date` | `sic2`, `delisted`, `delisting_date` | universe ticker x closed calendar quarter in which the firm was listed (quarter start on or before `delisting_date`; the delisting quarter is the last row); `as_of_date` = quarter end + 1 day |
| firm_year | `id`=`{ticker}_{year}`, `ticker`, `year`, `as_of_date` | `delisted`, `delisting_date` | ticker x calendar filing year with at least one scorable filing, filed while listed (1 Jan of year on or before `delisting_date`); `as_of_date` = 1 Jan of year + 1 |
| firm | `id`=ticker, `ticker` | `delisted`, `delisting_date` | firms with at least one scorable filing |

### 8-K coverage

8-K ingestion applies no item-code filter. `scripts/raw_ingestion/us/8k/01_fetch_filings.py`
fetches every 8-K a universe firm filed in the corpus window, and
`scripts/raw_processing/us/8k/segmenter_8k.py` keeps the whole document as one
section (`item_key = "0"`) instead of selecting Items 2.02 / 7.01 / 8.01.
Topicality is decided downstream by the prefilter, so the 8-K channel's
document counts are the firm's full 8-K filing volume, not a topical subset.

### Delisted firms

`delisted` / `delisting_date` come from silver.firm_universe
(`scripts/silver/universe.py`): for a universe firm whose stock stopped
trading, the filing date of the EDGAR Form 25-NSE that removed it (the latest
one since 2020 followed within 30 days by a Form 15-12B/15-12G, otherwise the
first one after the firm's last 10-K/10-Q), or a documented completion date;
never the last price date (a delisted ticker is reused: data/raw/market/prices/INFO.parquet
holds another company's 2024-2026 prices). Silver drops the documents a
delisted firm filed after that date and its prices after it
(silver.market_prices, the price input of gold); every spine follows. A row
of a delisted firm is kept when the firm was listed at some point of its
window: a firm-quarter whose quarter starts on or before `delisting_date`, a
firm-year whose year starts on or before it (with only the documents filed up
to that date), a call dated on or before it. Gap Inc. (GPS), flagged
delisted in the raw firm universe, changed its ticker to GAP and has no
delisting filing: it is not delisted.

| ticker | delisting_date | source |
|---|---|---|
| TIF | 2021-01-07 | EDGAR Form 25-NSE 0000876661-21-000016 (followed by Form 15-12B/15-12G; no later 10-K/10-Q or an Item 2.01/3.01 8-K) |
| CXO | 2021-01-19 | EDGAR Form 25-NSE 0000876661-21-000081 (followed by Form 15-12B/15-12G; no later 10-K/10-Q or an Item 2.01/3.01 8-K) |
| VAR | 2021-04-15 | EDGAR Form 25-NSE 0000876661-21-000554 (followed by Form 15-12B/15-12G; no later 10-K/10-Q or an Item 2.01/3.01 8-K) |
| FLIR | 2021-05-14 | EDGAR Form 25-NSE 0001354457-21-000563 (followed by Form 15-12B/15-12G; no later 10-K/10-Q or an Item 2.01/3.01 8-K) |
| ALXN | 2021-07-21 | EDGAR Form 25-NSE 0001354457-21-000820 (followed by Form 15-12B/15-12G; no later 10-K/10-Q or an Item 2.01/3.01 8-K) |
| MXIM | 2021-08-26 | EDGAR Form 25-NSE 0001354457-21-000970 (followed by Form 15-12B/15-12G; no later 10-K/10-Q or an Item 2.01/3.01 8-K) |
| KSU | 2021-12-14 | EDGAR Form 25-NSE 0000876661-21-001750 (followed by Form 15-12B/15-12G; no later 10-K/10-Q or an Item 2.01/3.01 8-K) |
| XLNX | 2022-02-14 | EDGAR Form 25-NSE 0001354457-22-000131 (followed by Form 15-12B/15-12G; no later 10-K/10-Q or an Item 2.01/3.01 8-K) |
| INFO | 2022-02-28 | EDGAR Form 25-NSE 0000876661-22-000196 (followed by Form 15-12B/15-12G; no later 10-K/10-Q or an Item 2.01/3.01 8-K) |
| HFC | 2022-03-15 | EDGAR Form 25-NSE 0000876661-22-000283 (followed by Form 15-12B/15-12G; no later 10-K/10-Q or an Item 2.01/3.01 8-K) |
| PBCT | 2022-04-04 | EDGAR Form 25-NSE 0001354457-22-000218 (followed by Form 15-12B/15-12G; no later 10-K/10-Q or an Item 2.01/3.01 8-K) |
| CERN | 2022-06-08 | EDGAR Form 25-NSE 0001354457-22-000333 (followed by Form 15-12B/15-12G; no later 10-K/10-Q or an Item 2.01/3.01 8-K) |
| CTXS | 2022-09-30 | EDGAR Form 25-NSE 0001354457-22-000553 (followed by Form 15-12B/15-12G; no later 10-K/10-Q or an Item 2.01/3.01 8-K) |
| DRE | 2022-10-03 | EDGAR Form 25-NSE 0000876661-22-000803 (followed by Form 15-12B/15-12G; no later 10-K/10-Q or an Item 2.01/3.01 8-K) |
| NLSN | 2022-10-12 | EDGAR Form 25-NSE 0000876661-22-000825 (followed by Form 15-12B/15-12G; no later 10-K/10-Q or an Item 2.01/3.01 8-K) |
| TWTR | 2022-10-28 | EDGAR Form 25-NSE 0000876661-22-000890 (followed by Form 15-12B/15-12G; no later 10-K/10-Q or an Item 2.01/3.01 8-K) |
| ABMD | 2022-12-22 | EDGAR Form 25-NSE 0001354457-22-000772 (followed by Form 15-12B/15-12G; no later 10-K/10-Q or an Item 2.01/3.01 8-K) |
| FRC | 2023-05-01 | FDIC receivership and sale to JPMorgan Chase completed 2023-05-01 (First Republic Bank filed with the FDIC, not EDGAR) |
| SIVB | 2023-05-02 | EDGAR Form 25-NSE 0001354457-23-000327 (first after the last 10-K/10-Q) |
| ATVI | 2023-10-13 | EDGAR Form 25-NSE 0001354457-23-000768 (followed by Form 15-12B/15-12G; no later 10-K/10-Q or an Item 2.01/3.01 8-K) |
| DISH | 2023-12-29 | EDGAR CIK 0001001082 Form 25-NSE 0001354457-23-001016 (followed by Form 15-12B/15-12G; no later 10-K/10-Q or an Item 2.01/3.01 8-K) |
| PXD | 2024-05-03 | EDGAR Form 25-NSE 0000876661-24-000321 (followed by Form 15-12B/15-12G; no later 10-K/10-Q or an Item 2.01/3.01 8-K) |
| WRK | 2024-07-08 | EDGAR Form 25-NSE 0000876661-24-000561 (followed by Form 15-12B/15-12G; no later 10-K/10-Q or an Item 2.01/3.01 8-K) |
| MRO | 2024-11-22 | EDGAR Form 25-NSE 0000876661-24-001100 (followed by Form 15-12B/15-12G; no later 10-K/10-Q or an Item 2.01/3.01 8-K) |
| CTLT | 2024-12-18 | EDGAR Form 25-NSE 0000876661-24-001185 (followed by Form 15-12B/15-12G; no later 10-K/10-Q or an Item 2.01/3.01 8-K) |
| DFS | 2025-05-19 | EDGAR Form 25-NSE 0000876661-25-000350 (followed by Form 15-12B/15-12G; no later 10-K/10-Q or an Item 2.01/3.01 8-K) |
| JNPR | 2025-07-02 | EDGAR Form 25-NSE 0000876661-25-000489 (followed by Form 15-12B/15-12G; no later 10-K/10-Q or an Item 2.01/3.01 8-K) |
| ANSS | 2025-07-17 | EDGAR Form 25-NSE 0001354457-25-000689 (followed by Form 15-12B/15-12G; no later 10-K/10-Q or an Item 2.01/3.01 8-K) |
| HES | 2025-07-18 | EDGAR Form 25-NSE 0000876661-25-000523 (followed by Form 15-12B/15-12G; no later 10-K/10-Q or an Item 2.01/3.01 8-K) |
| PARA | 2025-08-07 | EDGAR Form 25-NSE 0001354457-25-000781 (followed by Form 15-12B/15-12G; no later 10-K/10-Q or an Item 2.01/3.01 8-K) |
| WBA | 2025-08-28 | EDGAR Form 25-NSE 0001354457-25-000854 (followed by Form 15-12B/15-12G; no later 10-K/10-Q or an Item 2.01/3.01 8-K) |
| IPG | 2025-11-28 | EDGAR Form 25-NSE 0000876661-25-000918 (followed by Form 15-12B/15-12G; no later 10-K/10-Q or an Item 2.01/3.01 8-K) |
| HBI | 2025-12-01 | EDGAR Form 25-NSE 0000876661-25-000928 (followed by Form 15-12B/15-12G; no later 10-K/10-Q or an Item 2.01/3.01 8-K) |
| K | 2025-12-11 | EDGAR Form 25-NSE 0000876661-25-000958 (followed by Form 15-12B/15-12G; no later 10-K/10-Q or an Item 2.01/3.01 8-K) |
| CMA | 2026-02-02 | EDGAR Form 25-NSE 0000876661-26-000079 (followed by Form 15-12B/15-12G; no later 10-K/10-Q or an Item 2.01/3.01 8-K) |
| HOLX | 2026-04-07 | EDGAR Form 25-NSE 0001354457-26-000329 (followed by Form 15-12B/15-12G; no later 10-K/10-Q or an Item 2.01/3.01 8-K) |
| SEE | 2026-04-09 | EDGAR Form 25-NSE 0000876661-26-000345 (followed by Form 15-12B/15-12G; no later 10-K/10-Q or an Item 2.01/3.01 8-K) |
| CTRA | 2026-05-07 | EDGAR Form 25-NSE 0000876661-26-000399 (followed by Form 15-12B/15-12G; no later 10-K/10-Q or an Item 2.01/3.01 8-K) |
| EA | 2026-08-04 | EDGAR Form 25-NSE 0001354457-26-000757 (followed by Form 15-12B/15-12G; no later 10-K/10-Q or an Item 2.01/3.01 8-K) |
| AVB | 2026-08-17 | EDGAR Form 25-NSE 0000876661-26-000689 (followed by Form 15-12B/15-12G; no later 10-K/10-Q or an Item 2.01/3.01 8-K) |
| LEG | 2026-08-27 | EDGAR Form 25-NSE 0000876661-26-000712 (followed by Form 15-12B/15-12G; no later 10-K/10-Q or an Item 2.01/3.01 8-K) |

### Families

| table | columns | point in time |
|---|---|---|
| covariates/document/disclosure_volume | n_paragraphs, n_words, n_frames, n_promo, n_quant, n_spec, n_risk, n_gov, n_hyp, n_realized, n_deployed, n_revenue_outcome, n_cost_outcome, n_ai_investment, n_ai_infrastructure | the document itself |
| covariates/document/ai_vendors | n_ai_paragraphs, n_vendor_paragraphs, n_vendor_self_paragraphs, and per ecosystem (us, cn, eu, other) vendor_stack_*, vendor_entsw_* presence flags and paragraph counts | the document itself |
| covariates/document/ai_vendor_families | one presence flag per provider family (vendor_openai, vendor_deepseek, ...), from silver.ai_vendor_mentions | the document itself |
| covariates/document/activities | activity-instance counts by family: customer_facing_deployment, internal_deployment, proprietary_ai, third_party_named_provider, infrastructure_investment, quantified_outcome, talent_or_training, governance_or_restriction, piloting_or_exploring, named_product_or_process, named_function, deployed_or_scaled, n_activities | the document itself |
| covariates/activity/extraction | the extracted fields: country_code, form, item_key, paragraph_index, frame_id, has_activity, action, object, function, target, stage, source, entities, metrics, evidence_type, sentence_ids, firm, judge_model, prompt_version, session_id, classified_at | |
| covariates/activity/taxonomy | channel (filing when the text appears in both), function_family, providers_or_models, own_brands, is_own_ai, provider_or_model, provider_families, provider_family, object_family, activity, activity_function | |
| covariates/call/disclosure | n_words, n_frames, disclosure, n_activities, grounding, substance, w, n_prior_calls, hist_disclosure, surprise_disclosure, hist_substance, surprise_substance, hist_w | the call's own text; priors and ranks from earlier quarters, history from earlier calls |
| covariates/call/market | price_pre, return60, beta_pre, ncskew_pre, duvol_pre, ncskew_pre_63, log_market_cap, ps_ratio_pre | trading days before the call |
| covariates/call/financials | accession_number, filing_date_pt, revenue, operating_income, total_assets, operating_margin, asset_turnover, roa, shares_out, debt_to_equity, liabilities_to_assets, rd_intensity_pre, gross_margin_pre, revenue_yoy_pre, roic_minus_wacc_pre, nopat_pre, equity_pre, long_term_debt_pre, cost_of_equity_pre, cost_of_debt_pre, effective_tax_rate_pre | last 10-K filed strictly before the call; `revenue_yoy_pre` = revenue growth that 10-K reports over its prior fiscal year |
| targets/call/market | beta_post_63, beta_post_126, beta_post_252, ncskew_post_63, ncskew_post_105d, duvol_post_105d, log_market_cap_post, ps_ratio_post | trading days from the call on |
| targets/call/financials | rd_intensity_post, gross_margin_post, next_revenue_yoy_post, roic_minus_wacc_post, nopat_post, equity_post, long_term_debt_post, cost_of_equity_post, cost_of_debt_post, effective_tax_rate_post | first 10-K filed strictly after the call; `next_revenue_yoy_post` = growth of the fiscal year after that 10-K |
| covariates/firm_quarter/disclosure_volume, activities, posture_rates | `<family>_<window>_<metric>`; channel families all, filings, periodic, posture, calls; windows quarter, ttm, expanding | documents published up to the quarter end |
| covariates/firm_quarter/posture_archetype | posture_ttm_w_voc, posture_ttm_w_gov, posture_ttm_w_def, posture_ttm_archetype | one Archetypal Analysis fit per quarter on that quarter's rows |
| covariates/firm_quarter/washing_score | `<family>_<window>_<metric>` for filings/calls x quarter/ttm/expanding: grounding_index, substance, pct_disclosure, pct_substance, w, washing, callada | population estimates within each quarter |
| covariates/firm_quarter/financials | for revenue, cogs, rd_expense, sga_expense, operating_income, net_income, eps_diluted, capex, assets, current_assets, current_liabilities, equity, debt: `<metric>` (latest fiscal quarter filed before as_of_date) and `<metric>_quarter` (calendar quarter of that fiscal period's end) | filing date of the 10-Q/10-K that disclosed the value |
| covariates/firm_quarter/market | price_pre (close before as_of_date), shares_out (cover page of the latest 10-K/10-Q), log_market_cap, revenue_ttm (latest four fiscal quarters), ps_ratio, beta_pre_252, idio_vol_pre_252 | prices before as_of_date, filings filed before it |
| targets/firm_quarter/market | beta_post_63, idio_vol_post_63, vol_post_63 (trading days [as_of_date, +63)), ps_ratio_post (ps_ratio at the next quarter's as_of_date) | the next quarter |
| targets/firm_quarter/financials | next_quarter, next_revenue_yoy (revenue of the fiscal quarter ending in the next calendar quarter over the same quarter a year earlier), next_rd_intensity (its R&D / revenue) | the fiscal quarter ending after as_of_date |
| covariates/firm_year/disclosure_volume | n_docs, n_paragraphs, n_words, the frame counts, `*_per_1k`, any_ai, `*_rate` (filings) | filings of the year |
| covariates/firm_year/activities | the document activity families and n_activities, summed over the year's filings, and their `*_per_1k` | filings of the year |
| covariates/firm_year/posture | promotional_posture, hedging_posture, risk_orientation, governance_orientation, temporal_posture, ai_positioning, specificity, disclosure_intensity | shrinkage prior and intensity rank within the calendar year's firm-years (information of that year only) |
| covariates/firm_year/posture_archetype_static | cluster, archetype | static full-sample model applied to the year's posture dimensions rebuilt with the pooled prior and rank (descriptive, not point in time) |
| covariates/firm_year/washing_score | n_activities, grounding_index, substance, pct_disclosure, pct_substance, w, washing, callada | grounding priors, ranks and tails within the calendar year's cross-section; activity inputs carried from the latest year with a 10-K |
| covariates/firm_year/financials | filing_date, accession_number, disclosed_period_end, revenue, cost_of_revenue, rd_expense, sga_expense, capex, operating_income, net_income, da, interest_expense, pretax_income, tax_expense, eps_diluted, total_assets, equity, current_assets, current_liabilities, long_term_debt, cash, shares_out, gross_margin, operating_margin, net_margin, roa, roe, current_ratio, debt_to_equity, asset_turnover, rd_intensity, capex_intensity, ebitda, sic2, revenue_yoy, has_10k, effective_tax_rate, nopat, invested_capital, roic, rf_annualized, erp, cost_of_equity, cost_of_debt, wacc, roic_minus_wacc, wacc_market, roic_minus_wacc_market | 10-K filed in the year (`revenue_yoy`: its fiscal year over the prior one, null when the annual periods are not 340-380 days apart) |
| covariates/firm_year/market | market_cap, pe_ratio, ps_ratio, pb_ratio, ev_revenue, ev_ebitda, beta, beta_n_obs, idio_vol_252d, vol_pre_60d, momentum_12_1 | trading days before that 10-K |
| covariates/firm_year/patents | total_patents, ai_patents_oecd, ai_patents_core, ai_patents_related, ai_patent_intensity | patent filings of the year |
| targets/firm_year/financials | next_period_end, next_gap_days, next_revenue_yoy, next_rd_expense_yoy, next_capex_yoy, next_sga_expense_yoy | next fiscal year |
| targets/firm_year/market | vol_post_60d, ret_m1_p5 | trading days from the 10-K filing on |
| covariates/firm/posture_archetype_static | n_frames, the seven posture rates, disclosure_intensity, cluster, archetype, archetype_stability | static full-sample fit (descriptive) |

### Datasets

`datasets/<grain>/<grain>`: the spine (id, keys, date and spine attributes,
one row per spine key) left-joined on `id` with every
`covariates/<grain>/*` and `targets/<grain>/*` family, nulls where a family
has no information. The builders only join existing gold tables
(`scripts/gold/datasets/join_families.py`); nothing is recomputed or imputed.

Column names. A family column keeps its name when no other joined table of
the grain has it. On a collision the values are compared over the spine rows
(nulls equal, dtype ignored): identical values keep one column under the plain
name; differing values are prefixed with their family in every family,
`<family>__<column>` (a spine attribute keeps its plain name). The dataset
manifest lists the collisions (`collisions`: column -> names in the dataset);
`layers.dataset_columns` maps a family's columns to their dataset names.

| dataset | rows x columns | collisions | cross-grain columns |
|---|---|---|---|
| datasets/document/document | 59,363 x 39 | none | none |
| datasets/activity/activity | 46,531 x 52 | none | `firm__*`: datasets/firm/firm on ticker (static posture archetype, descriptive; activity analytics group activities by it) |
| datasets/call/call | 10,271 x 1,332 | none | `fq__*`: datasets/firm_quarter/firm_quarter as of the call date (`pit.asof_join` of `fecha` on `as_of_date`, backward within ticker; the snapshot only contains information published before `as_of_date`); `fq__quarter`, `fq__as_of_date` name the attached snapshot, null for calls before the ticker's first closed quarter |
| datasets/firm_quarter/firm_quarter | 12,086 x 1,263 | `<channel>_<window>_n_words` in disclosure_volume and activities: identical, one column | none |
| datasets/firm_year/firm_year | 2,887 x 159 | `n_activities`: activities (summed over the year's filings) and washing_score (carried from the latest year with a 10-K) differ -> `activities__n_activities`, `washing_score__n_activities` | none (firm-year analytics use covariates/firm_year/posture_archetype_static, not the firm grain) |
| datasets/firm/firm | 498 x 16 | none | none |

## Builders

Entry points, in `make gold` order (`gold-document`, `gold-firm`,
`gold-firm-year`, `gold-firm-quarter`, `gold-call`, `gold-datasets`); each
writes a spine before the families that join onto it and reads earlier gold
tables instead of recomputing them. `gold-datasets` runs after every family
exists (firm before activity, firm_quarter before call).

| builder | writes |
|---|---|
| scripts/gold/document/build_document.py | spines/document/document, covariates/document/disclosure_volume |
| scripts/gold/activity/build_activity.py | spines/activity/activity, covariates/activity/{extraction, taxonomy} |
| scripts/gold/document/build_activities.py | covariates/document/activities |
| scripts/gold/document/build_ai_vendors.py | covariates/document/{ai_vendors, ai_vendor_families} |
| scripts/gold/firm/build_posture_archetype_static.py | spines/firm/firm, covariates/firm/posture_archetype_static, models/posture_archetype_static |
| scripts/gold/firm_year/build_spine.py | spines/firm_year/firm_year |
| scripts/gold/firm_year/build_disclosure.py | covariates/firm_year/{disclosure_volume, activities} |
| scripts/gold/firm_year/build_posture.py | covariates/firm_year/{posture, posture_archetype_static} |
| scripts/gold/firm_year/build_market_financials.py | covariates/firm_year/{financials, market}, targets/firm_year/{financials, market} |
| scripts/gold/firm_year/build_patents.py | covariates/firm_year/patents |
| scripts/gold/firm_year/build_washing_score.py | covariates/firm_year/washing_score, models/washing_grounding_shrinkage |
| scripts/gold/firm_quarter/build_measures.py | spines/firm_quarter/firm_quarter, covariates/firm_quarter/{disclosure_volume, activities, posture_rates} |
| scripts/gold/firm_quarter/build_posture_archetype.py | covariates/firm_quarter/posture_archetype, models/posture_archetype_weights |
| scripts/gold/firm_quarter/build_washing_score.py | covariates/firm_quarter/washing_score |
| scripts/gold/firm_quarter/build_financials.py | covariates/firm_quarter/financials |
| scripts/gold/firm_quarter/build_market.py | covariates/firm_quarter/market, targets/firm_quarter/{market, financials} |
| scripts/gold/call/build_spine.py | spines/call/call |
| scripts/gold/call/build_disclosure.py | covariates/call/disclosure |
| scripts/gold/call/build_market_financials.py | covariates/call/{market, financials}, targets/call/{market, financials} |
| scripts/gold/{document, firm, activity, firm_year, firm_quarter, call}/build_dataset.py | datasets/<grain>/<grain> |

Shared libraries (no gold writes): `scripts/gold/datasets/join_families.py`,
`scripts/gold/posture/{ai_intensity,
posture_features, warm_start_aa}.py`, `scripts/gold/activity/build_activity.py`
(taxonomy and `flags`), `scripts/gold/washing/washing_score.py` (shrinkage),
`scripts/gold/financials/{build_firm_financials, build_market_factors,
build_roic_wacc, market_windows}.py`.

## Old -> new mapping

Every table of the previous layout, moved to
`data/deprecated/gold_<timestamp>/`. Column names are unchanged unless listed.

| old table | new location |
|---|---|
| spines/firm_year/firm_year (id, ticker, year) | spines/firm_year/firm_year (+ as_of_date) |
| spines/firm_quarter/firm_quarter | unchanged |
| spines/call/call (35 columns) | spines/call/call: ticker, fecha, call_accession_number, sic, sic2, fe. n_words, n_frames, disclosure, n_activities, grounding, substance, w, n_prior_calls, hist_disclosure, surprise_disclosure, hist_substance, surprise_substance, hist_w -> covariates/call/disclosure. beta_pre, price_pre, return60, log_market_cap -> covariates/call/market. beta_post_63, beta_post_126, beta_post_252 -> targets/call/market. filing_date_pt, revenue, operating_income, total_assets, operating_margin, asset_turnover, roa, shares_out, accession_number -> covariates/call/financials |
| covariates/call/disclosure | covariates/call/disclosure |
| covariates/call/market (beta_pre, price_pre, log_market_cap, shares_out) | covariates/call/market (shares_out -> covariates/call/financials) |
| covariates/call/financial_ratios | covariates/call/financials; sic, sic2, fe -> spines/call/call |
| covariates/call/car_pre (beta_pre_car, idio_vol_pre, sue, debt_to_equity, liabilities_to_assets) | debt_to_equity, liabilities_to_assets -> covariates/call/financials; beta_pre_car, idio_vol_pre and sue removed (CAR and SUE are not part of the analysis) |
| covariates/call/crash_risk_pre, covariates/call/ncskew_pre_63 | covariates/call/market |
| covariates/call/fundamentals_pre | covariates/call/financials; log_market_cap_pre (identical to log_market_cap) dropped; ps_ratio_pre -> covariates/call/market; next_revenue_yoy_pre (growth of the fiscal year after the pre-call 10-K, not known at the call) dropped, the pre-call control is covariates/call/financials.revenue_yoy_pre |
| targets/call/call (beta_post_63d/126d/252d, return_post_60d) | beta_post_63/126/252 in targets/call/market; return_post_60d was the pre-call return `return60` -> covariates/call/market |
| targets/call/car_post, crash_risk_post, ncskew_post_63 | targets/call/market (car_m1_p5 removed) |
| targets/call/fundamentals_post | targets/call/financials; log_market_cap_post, ps_ratio_post -> targets/call/market |
| datasets/call/call | datasets/call/call: spine + every call family + `fq__` firm-quarter covariates as of the call |
| covariates/document/document_panel | spines/document/document (channel, form, ticker, cik, fecha, period_end, call_fy, accession_number, fye_month, fy) + covariates/document/disclosure_volume (n_paragraphs, n_words, frame counts); `quarter` dropped (calendar quarter of fecha) |
| covariates/document/activities (one row per ticker x text x activity) | spines/activity/activity (ticker, text_hash, activity_id, accession_number, + fecha) + covariates/activity/extraction + covariates/activity/taxonomy |
| covariates/firm_quarter/disclosure_volume, activities, posture_rates, washing_score | unchanged |
| covariates/firm_quarter/posture_archetype_weights | covariates/firm_quarter/posture_archetype |
| covariates/firm_quarter/us_10q_financials_raw (long: ticker, year, quarter of the fiscal period end, metric, value, source, source_ref) | covariates/firm_quarter/financials (wide, point in time by filing date) |
| covariates/firm_quarter/archetype_weights_expanding, predictions/firm_quarter/posture_archetype_expanding, predictions/firm_year/posture_archetype_expanding | removed (legacy expanding archetypes); consumers use covariates/firm_quarter/posture_archetype or covariates/firm_year/posture_archetype_static |
| covariates/firm_quarter/market | price_pre (close before as_of_date), shares_out (cover page of the latest 10-K/10-Q), log_market_cap, revenue_ttm (latest four fiscal quarters), ps_ratio, beta_pre_252, idio_vol_pre_252 | prices before as_of_date, filings filed before it |
| targets/firm_quarter/market | beta_post_63, idio_vol_post_63, vol_post_63 (trading days [as_of_date, +63)), ps_ratio_post (ps_ratio at the next quarter's as_of_date) | the next quarter |
| targets/firm_quarter/financials | next_quarter, next_revenue_yoy (revenue of the fiscal quarter ending in the next calendar quarter over the same quarter a year earlier), next_rd_intensity (its R&D / revenue) | the fiscal quarter ending after as_of_date |
| covariates/firm_year/disclosure_volume | unchanged |
| covariates/firm_year/activities | unchanged except n_words (-> covariates/firm_year/disclosure_volume) |
| covariates/firm_year/activities_by_channel (ticker, fy, channel) | removed: sum covariates/document/activities by the document spine's (ticker, fy, channel) |
| covariates/firm_year/posture | unchanged except n_frames (-> covariates/firm_year/disclosure_volume, identical values) |
| covariates/firm_year/financial_ratios_raw, financial_levels, financial_ratios, value_creation | covariates/firm_year/financials; next_period_end, next_gap_days, next_*_yoy -> targets/firm_year/financials; beta, market_cap -> covariates/firm_year/market; in_text_panel (constant true) dropped |
| covariates/firm_year/market_raw, market | covariates/firm_year/market; filing_date -> covariates/firm_year/financials; vol_post_60d -> targets/firm_year/market; car_m1_p5 removed |
| targets/firm_year/filing_returns_raw | targets/firm_year/market (ret_m1_p5) |
| targets/firm_year/firm_year | beta_252d = covariates/firm_year/market.beta; price_to_sales_t = market.ps_ratio; idio_vol_252d, vol_pre_60d = market; car_m1_p5d removed; vol_post_60d = targets/firm_year/market; rd_intensity_t = covariates/firm_year/financials.rd_intensity; revenue/rd_expense/capex/sga_expense_growth_lead1y = targets/firm_year/financials.next_*_yoy |
| datasets/firm_year/firm_year | datasets/firm_year/firm_year: spine + every firm_year family |
| covariates/firm_year/patents | unchanged |
| predictions/firm_year/posture_archetype_static | covariates/firm_year/posture_archetype_static |
| predictions/firm_year/washing_score (1,885 disclosing firm-years) | covariates/firm_year/washing_score on the full spine (null outside the scored sample); frames_per_1k, any_ai, n_promo, n_frames -> covariates/firm_year/disclosure_volume |
| predictions/firm/posture_archetype_static | spines/firm/firm + covariates/firm/posture_archetype_static |
| covariates/firm_year/channel_gap_cells_extensive, channel_gap_cells_paired, datasets/firm_year/channel_gap | removed with the channel-gap difference-in-differences (scripts/deprecated/) |

Removed builders (moved to `scripts/deprecated/`): `scripts/gold/consolidate/`
(one-table wrappers), `scripts/gold/channel_gap/build_channel_gap_cells.py`,
`scripts/gold/posture/build_archetype_weights_quarterly_asof.py`,
`scripts/gold/crash_archetypes/build_call_crash_and_archetypes.py`, and the
analytics difference-in-differences scripts
(`scripts/analytics/channel_gap/channel_gap_did.py`,
`scripts/analytics/shock/shock_did_simple.py`,
`scripts/analytics/shock/shock_analysis.py`). Their results
(`channel_gap/channel_gap_did.json`, `shock/shock_analysis.json`,
`sec_did_continuous.json`, `fig_sec_event_study.png`, `shock_did_simple.json`,
`shock_did_*_per_1k.png`) moved to `data/deprecated/results_did_20260915/`
(POINTER.json).

The incremental-signal analysis is point in time on the firm-quarter grain,
so the firm-year median-vs-aggregate robustness it was paired with has no
median left to compare: `scripts/analytics/washing/firm_year_aggregation_robustness.py`
moved to `scripts/deprecated/` and its result to
`data/deprecated/results_firm_year_aggregation_20260916/` (POINTER.json).

CAR and SUE are not computed: `scripts/analytics/call_beta/call_car_regressions.py`
and its test moved to `scripts/deprecated/`, their results
(`call_beta/call_car_regressions.csv`, `call_car_regression_samples.csv`,
`call_beta_sue_regressions.csv`, `call_beta_sue_regression_samples.csv`,
`call_beta_clean_regressions.csv`, `call_beta_clean_regression_samples.csv`)
to `data/deprecated/results_car_20260915/` (POINTER.json).
