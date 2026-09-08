# Accounting / XBRL data — sources

Both countries publish structured (XBRL-tagged) accounting data as a
first-class regulatory artifact, separate from the narrative filings
(10-K/10-Q text, Memoria/Análisis Razonado text) this project's document
pipeline already extracts. This is the OUTCOME side of the panel (the
"XBRL outcomes" instrument in the project's data-scope decision) — the
figures a disclosure-text regression would use as dependent/control
variables, not more text to run through the AI-disclosure detector.

## US — `scripts/us/03_fetch_accounting_data.py`

Source: SEC's own XBRL "company facts" API
(`data.sec.gov/api/xbrl/companyfacts/CIK##########.json`), wrapped by
`edgartools` (already a project dependency, already used by
`scripts/us/edgar_fetch.py` for the narrative filings) as
`edgar.Company(cik).get_facts().to_dataframe()`. No new HTTP client
needed — same `EDGAR_IDENTITY`/user-agent requirement as the existing
filing fetch (`configs/us/config.yaml:sec.user_agent`).

Verified directly (2026-09-03): `edgar.Company(320193).get_facts()` (Apple)
returns 25,135 rows spanning every `dei:`/`us-gaap:` concept the company
has ever tagged, across every period — one bulk pull per firm, no PDF
parsing, no per-statement scraping. Columns: `concept, label, value,
numeric_value, unit, period_type, period_start, period_end, fiscal_year,
fiscal_period`.

Stored as one parquet per ticker in `data/raw/xbrl_facts/us/{ticker}.parquet`
(`configs/us/config.yaml:storage.raw_xbrl_facts`), same
incremental-fetch/skip-if-exists contract as the market-data collector.

**Not what the 10-K panel is built from.** Company Facts collapses
restatements: the value it returns for a given `period_end` can be
whatever the company reported in a LATER filing, not what that specific
10-K actually disclosed at the time — the same as-of-filing-date problem
`scripts/us/04_extract_inline_xbrl_facts.py` exists to avoid for the 10-Q
shock series (see below). `scripts/analytics/build_firm_financials.py`
(the 10-K panel) reads `data/raw/xbrl_facts/us_by_filing/` — the same
inline-XBRL-per-filing source as the 10-Q panel — for exactly that reason.
This `us/` company-facts pull remains useful on its own as a quick
full-history sanity check per firm, just not as the panel's source of
truth.

**`long_term_debt` tag fallback widened (2026-09-08):** the original
2-tag chain (`LongTermDebtNoncurrent`, `LongTermDebt`) left 45 of 510
tickers with zero debt tag entirely — not debt-free firms, but filers
that disclose long-term borrowings under a different primary concept:
CSX, Cummins, Cardinal Health and 19 others fold finance-lease
obligations into `us-gaap:LongTermDebtAndCapitalLeaseObligations`;
homebuilders and distributors like D.R. Horton tag theirs
`us-gaap:NotesPayable`. Adding both as lower-priority fallbacks recovered
26 of the 45.

**Dimensional facts corrupting the consolidated total (2026-09-08),
found and fixed.** Unlike Company Facts, inline-XBRL keeps every context a
filer tags — including segment/geography breakdowns (`explicitMember`/
`typedMember` dimensions) that can share the SAME concept and period as
the consolidated total. Amazon's FY2021 `us-gaap:Revenues` existed ONLY as
a ~$55M dimensional context (a single business unit); the real ~$470B
total was tagged under a different concept
(`RevenueFromContractWithCustomerExcludingAssessedTax`,
non-dimensional) — but before this fix, the concept-priority fallback
still preferred `Revenues` and returned the $55M segment figure as
Amazon's total revenue. `load_facts()` now excludes any fact with
`has_dimensions = true`. This is the single highest-impact fix in this
file: it silently corrupted revenue (and every other metric with a
segment breakdown) for any multi-segment filer using the affected
concept, which is most of the S&P-sized end of this universe.

**As-of correctness (2026-09-08): "first disclosed" beats both "median"
and "most recent".** This panel feeds as-of joins — the value assigned to
a period must be what a reader of THAT filing could have known then,
never a figure a later filing corrected. XBRL repeats each annual figure
in 2-3 subsequent filings as prior-year comparatives, and those repeats
can disagree after a restatement. Two rules were tried and rejected before
landing on the right one:
- *Most recent filing wins* — the original rule — actively prefers the
  restated figure, the opposite of as-of-safe.
- *Median across filings* — robust to a single outlier IF the vote split
  favors the original, but not otherwise. Verified against DISH's 2024
  post-EchoStar-merger 10-K, which retags FY2021 revenue at ~10x what two
  prior, mutually consistent filings had reported for the exact same
  period: median happened to get this right (2 votes vs 1), but the rule
  itself doesn't know why it's right.
- *Value from the EARLIEST filing_date* — the rule actually used now. It
  is correct regardless of how later filings vote, because it is the only
  rule that never looks at a filing later than the one being resolved.

A second, subtler instance of the same bug lived in concept-fallback
priority itself: `pivot_metrics()` picked the highest-PRIORITY concept
(e.g. `us-gaap:Revenues` over `RevenueFromContractWithCustomerExcludingAssessedTax`)
regardless of when each concept was first tagged. Iron Mountain's original
Q1 2022 10-Q tagged only the contract-revenue concept ($497M); `Revenues`
for that same quarter first appears over a year later, in 2023, as a
restated comparative at ~2.5x the figure. Priority-first pulled in that
later restatement. Fixed by sorting DATE first, priority only as a
same-date tie-break (needed because some filers, e.g. Capital One, tag
both concepts in the same original filing at genuinely different
magnitudes — a bank's non-interest income from customer contracts is a
small slice of its total revenue).

**Validation:** summed quarterly revenue (10-Q panel, inline-only) against
independently-built annual revenue (10-K panel) for every ticker-year
where all 4 quarters came from inline-XBRL. Restricted to calendar-fiscal-
year firms (non-calendar firms, e.g. NVDA/ORCL/GIS, fail this check
trivially because "calendar year" quarter grouping doesn't match their
own fiscal year — a limitation of the check, not the panels): **98.7%
within 1% of the annual figure, 99.0% within 5%**, n=1,626 (remeasured
after adding the dimensional-singleton fallback below; the small drop
from 99.1%/99.4% is expected — a few newly-filled cells come from a
noisier, last-resort source, still net positive since coverage rose too).
The handful of
remaining outliers (BlackRock, Iron Mountain, Northern Trust) are
financial institutions/REITs whose own XBRL tagging inconsistently
switches concept scope across periods within the same fiscal year — a
data-quality property of those specific filings, not a pipeline defect.

Net coverage effect of all 10-K-panel fixes together (dimensional filter
+ dimensional-singleton fallback + dual-class shares + debt fallback +
as-of resolution + priority-tiebreak fix), 504 tickers, 2,868 aligned
10-K rows: `revenue` 98%, `net_income` 99.7%, `total_assets`/`equity`/
`shares_out` 100%, `capex` 87%, `sga_expense` 81%, `operating_income`
80%, `long_term_debt` 90%, `current_assets`/`current_liabilities` 85%,
`debt_to_equity` 84%, `cost_of_revenue` 62%, `rd_expense` 46%.

`shares_out` needed one more fix on top of the dimensional filter, not
just the filter itself: multi-class-stock filers (GOOGL, META, BRK.B, F,
CMCSA, and 36 others — 40 of 510 tickers) tag shares outstanding ONLY per
share class, as dimensional facts, with no non-dimensional total at all —
so excluding `has_dimensions` correctly protected revenue/etc. from
segment contamination but left these 40 tickers with zero shares_out.
`load_dual_class_shares()` sums the per-class dimensional facts within
each filing (never across filings), resolved to the earliest filing_date
per period like everything else. Verified against GOOGL: exactly 3
dimensional contexts per filing (its 3 share classes), summing to ~665M
pre-split (2021-2022) and ~12.6B post-20:1-split (mid-2022 on) — both
match GOOGL's actual known share counts. Took `shares_out` from
91%→100%.

**Dimensional-singleton fallback (2026-09-08), applied to every metric,
not just shares_out.** GM, General Dynamics, Sherwin-Williams and 3
others tag `rd_expense` with exactly ONE dimensional member every period
— not a segment breakdown needing summation, just the whole-company
figure filed under a dimensional context (GM's tagged R&D matches its
actual reported figures: $9.8B FY2022, $9.9B FY2023). `_load_dimensional_singletons()`
recovers any (ticker, concept, period) cell where every dimensional fact
agrees on a single value, ranked strictly BELOW every non-dimensional
concept in the fallback chain (see `is_dimensional` in `pivot_metrics`) —
a cell with 2+ DIFFERENT dimensional values (a genuine multi-segment
split, checked directly: AEP/AMT/APA/ATVI's `cost_of_revenue` has 4-17
distinct dimensional values per period, real segment/cost-type
breakdowns) is left NULL rather than guessed at. Lifted `long_term_debt`
88%→90%, `cost_of_revenue` 60%→62%, `sga_expense` 80%→81%, `rd_expense`
45%→46% — small because most of the remaining gap in each is a genuine
either-doesn't-disclose or discloses-only-as-a-real-multi-way-breakdown
case, not a single mislabeled total. The same fallback was ported to the
10-Q panel (below) for consistency.

## Chile — `scripts/cl/04_fetch_accounting_data.py`

Two candidate sources were checked directly against the live CMF site
(2026-09-03) before picking one:

1. **`sa_eeff_ifrs` bulk index — NOT the figures, a dead end for now.**
   `POST .../merc_valores/sa_eeff_ifrs/sa_eeff_ifrs2grid.php` with
   `xls=y` genuinely returns a real binary `.xls` (verified: valid CDFV2/
   OLE2 file, parses with `pandas.read_excel` + `xlrd`) covering EVERY
   Chilean SA in one request for a given period — but its one sheet
   (`IFRS_CL_CI`, "Cuadro de Identificación") is just the ROSTER of firms
   that reported that period (RUT, razón social, moneda, tipo de balance,
   fechas) — no balance-sheet/income-statement line items at all.
   Whatever endpoint serves the actual figures for this bulk system
   (if one exists — `cl-bs`/`cl-ei` are real taxonomy codes per
   `cmf_xbrl_taxonomias`, suggesting a per-statement export might exist)
   wasn't found in the time spent looking. Worth another pass later
   ONLY if the per-firm route below turns out to be too slow at the full
   101-firm scale — it currently isn't.

2. **Per-firm XBRL package — what's actually used.** The SAME EEFF tab
   (`pestania=3`) `01_fetch_filings.py` already scrapes for Análisis
   Razonado lists an "Estados financieros (XBRL)" document link right
   next to it. Verified directly for Empresas Copec FY2023: resolves to
   a real ZIP (`PK\x03\x04` magic bytes) containing
   `{rut}_{periodo}_C.xbrl` (the instance document), `.xsd` (schema) and
   `-definition.xml` — a complete, standard XBRL filing package, not a
   scraped re-derivation.

`scripts/cl/04_fetch_accounting_data.py` reuses `01_fetch_filings.py`'s
own EEFF-tab request shape (`cmf_direct_client.post_legacy`, same
`forma/mm/aa/tipo/tipo_norma` params) and `_EEFF_DOC_RE`-style matching,
just filtering for `"xbrl"` in the document name instead of
`"analisis"/"análisis"`, then downloads the ZIP as-is via
`cmf_direct_client.fetch_binary` (no PDF/text extraction — these are kept
as raw ZIPs, XBRL parsing is a downstream concern, not this fetch step's).

Stored gzip-free (already a ZIP) at
`data/raw/xbrl_cl/{rut}/{rut}_{period}_C.zip`
(`configs/cl/config.yaml:storage.raw_xbrl`), manifest at
`data/interim/manifests_cl/xbrl_manifest.parquet` — a SEPARATE manifest
from `filing_manifest.parquet` (different document type, different
lineage columns; same reasoning as filing_manifest_10q being its own
table rather than a form_type filter on one shared manifest).

Same annual/quarterly closes as `01_fetch_filings.py`
(03/06/09/12 + the FY close), same firm universe
(`configs/cl/universe.csv`) — including the 4 banks, since (per
`01_fetch_filings.py`'s own established finding) the EEFF tab lives on
`entidad.php` under `tipoentidad=RVEMI`, which is exactly the system the
4 banks aren't registered in. Expect the same 4-firm gap here as for the
narrative filings; no manual-download path has been built for XBRL yet
since it wasn't asked for.

## US 10-Q shock series — frames as an alternative source, coverage measured

`scripts/us/04_extract_inline_xbrl_facts.py` parses inline XBRL directly out
of each cached 10-Q filing's own HTML (point-in-time, firm's own fiscal
quarter). `scripts/us/05_fetch_xbrl_frames_alt.py` pulls the same 13 core
metrics from SEC's XBRL "frames" API instead — one call per (tag, calendar
quarter), returning every filer's value for that exact calendar period
across the whole SEC universe, then filtered down to the 517-ticker
universe. Stored separately, never merged, at
`data/raw/xbrl_frames_alt/us_10q_frames.parquet`
(`configs/us/config.yaml:storage.raw_xbrl_frames_alt`).

**Coverage bug #1 (2026-09-08): YTD/annual comparatives miscounted as
quarterly.** An inline-XBRL fact keeps every duration context tagged in a
filing under one concept, which mixes the single-quarter figure with YTD
and full-year comparatives under the SAME tag (a 10-Q routinely tags both
the 3-month and 6-month-YTD `Revenues` value). Counting any duration fact
landing in a calendar quarter as "covering" that quarter counted
YTD/annual figures as quarterly ones, inflating apparent coverage (a
bogus 91.8% for revenue). Restricting to duration facts spanning 75–100
days (one real fiscal quarter) before deduping fixed the overcounting —
but overcorrected into a second, larger bug.

**Coverage bug #2 (2026-09-08): the 75–100-day filter discards firms
that only ever tag YTD.** Many filers — verified against AAPL, where
`PaymentsToAcquirePropertyPlantAndEquipment` appears ONLY as 90/181/272/
363-day cumulative facts across its entire filing history, never as a
bare ~90-day figure — simply never tag a standalone discrete-quarter
duration fact for cash-flow-statement lines. The 75-100-day filter caught
each such firm's Q1 (which happens to equal its own YTD) and silently
dropped Q2-Q4 entirely, understating true coverage. `discrete_quarters()`
in `scripts/analytics/build_us_10q_financials_panel.py` recovers them by
differencing consecutive YTD facts that share the same `period_start`
(the fiscal-year anchor): `Q2 = YTD_Q2 - YTD_Q1`, `Q3 = YTD_Q3 - YTD_Q2`,
`Q4 = FY - YTD_Q3`. The differenced AAPL capex series was checked against
its known quarterly figures and matched (e.g. FY2020: Q2 $1.85B, Q3
$1.57B, Q4 $1.78B — all economically plausible for Apple's capex run-rate
in that period).

**Coverage bug #3 (2026-09-08): dimensional-singleton fallback, ported
from the 10-K panel.** Same fix as `_load_dimensional_singletons()` there
— a filer that tags a metric with exactly ONE dimensional member every
period (not a real segment split) is recovered; a cell with 2+ different
dimensional values is left NULL. Ranked strictly below any
non-dimensional concept, at every dedup step (`is_dimensional` sorts
before `filing_date` throughout `build_us_10q_financials_panel.py`), so
it only ever fills a genuine gap.

True coverage against the 517-ticker × 23-quarter (2021Q1–2026Q3, 11,891
possible firm-quarter cells) 10-Q panel, inline-XBRL only vs. combined
with the frames fallback, after all three fixes:

| metric | inline-only | combined (+ frames) | gain |
|---|---|---|---|
| rd_expense | 30.4% | 30.8% | +0.4pp |
| cogs | 55.5% | 55.7% | +0.2pp |
| operating_income | 71.1% | 72.1% | +1.0pp |
| sga_expense | 72.0% | 72.9% | +0.9pp |
| capex | 76.6% | 77.4% | +0.8pp |
| debt | 77.6% | 78.3% | +0.7pp |
| revenue | 87.6% | 87.9% | +0.3pp |
| eps_diluted | 87.7% | 88.7% | +1.0pp |
| net_income | 89.6% | 90.6% | +1.0pp |
| assets | 90.3% | 91.3% | +0.9pp |
| equity | 90.9% | 91.7% | +0.9pp |
| current_assets / current_liabilities | 76.1% | 77.0% | +0.9pp |

The YTD-differencing fix alone moved these numbers far more than any
tag-fallback widening or the frames fallback: capex 24.7%→76.3% (+52pp),
revenue 69.9%→87.4%, net_income 71.4%→89.5%, eps_diluted 69.7%→87.3%,
sga_expense 55.6%→71.3%, operating_income 56.0%→70.9%. (Instant metrics —
`assets`, `equity`, `current_assets`, `current_liabilities`, `debt` — are
balance-sheet snapshots, not YTD-accumulated, so differencing doesn't
apply to them; their numbers are unchanged from the tag-fallback fixes
above.)

Two metrics stay genuinely low even after differencing:

- **`rd_expense` (30.2%)** barely moved (23.6%→30.2%) — a filer who omits
  a discrete R&D figure overwhelmingly also omits R&D from the YTD
  cash-flow/income-statement breakout, so there's little left to
  difference. Matches the 10-K panel's 46% and its sector explanation
  (utilities, insurers, airlines, REITs mostly don't tag R&D at all).
- **`cogs` (54.0%)** — cost of revenue is disclosed quarterly by product
  companies but not by the services/finance-heavy share of this universe,
  the same sector pattern as the 10-K panel's 72%.

A small amount of noise comes with differencing: 17 of 10,392 quarterly
revenue values and 28 of 9,069 capex values come out negative (<0.3% of
either), from restated comparatives landing on either side of a
difference. Left as-is rather than filtered — downstream ratio
construction already winsorizes at 1/99% (`docs/analytics/10_pipeline.md`).

Frames alone (not as a fallback) are still worse than inline-XBRL for
every duration metric — a frame only returns a value when a filer's
fiscal period exactly matches the calendar-quarter boundary
(`CY2024Q2`/`CY2024Q2I`), silently dropping every non-December
fiscal-year-end filer for that quarter. Used as a gap-filler instead
(only where inline-XBRL has nothing for that firm-quarter), it adds a
real, if modest, 0.7–3.2pp per metric.

Consolidated panel (one row per ticker/year/quarter/metric, `source`
column tracing inline_xbrl vs. frames_fallback, `source_ref` the
accession number/filing) is built by
`scripts/analytics/build_us_10q_financials_panel.py` into
`data/processed/us_10q_financials_panel.parquet`.

## Actual run results (2026-09-03)

**US** (`scripts/us/03_fetch_accounting_data.py`, 516 tickers): 514
fetched, 2 with no facts returned — `FRC` (First Republic Bank) and
`SBNY` (Signature Bank), both failed/were seized in the 2023 banking
crisis and stopped filing afterward. Not a fetch bug.

**Chile** (`scripts/cl/04_fetch_accounting_data.py`, 100 firms ×
5 years × 4 quarterly closes = 2,000 possible documents):
- 1,478 `completed`
- 509 `not_found` — same shape as the narrative-filing gaps (a firm
  simply didn't file EEFF for that particular close), confirmed
  including the 4 banks (not registered under RVEMI at all, per the
  finding above)
- 13 transient network failures (`Read timed out` / connection reset) —
  resumable on a plain re-run, same idempotency contract as
  `01_fetch_filings.py` (only rows without a local file get retried)
