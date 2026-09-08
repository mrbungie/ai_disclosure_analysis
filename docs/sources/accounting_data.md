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
Switching it from Company Facts to inline-XBRL (2026-09-08) also raised
coverage: `shares_out` 91%→100%, `long_term_debt` 73%→84%,
`operating_income` 79%→83%, `sga_expense` 79%→81% (509 tickers, 2,874
aligned 10-K rows). `rd_expense` stays at 46% either way — genuinely
sector-driven (utilities, insurers, airlines, REITs mostly don't tag an
R&D line at all), not a coverage bug. This `us/` company-facts pull
remains useful on its own as a quick full-history sanity check per firm,
just not as the panel's source of truth.

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

**Coverage bug found and fixed (2026-09-08):** an inline-XBRL fact keeps
every duration context tagged in a filing under one concept, which mixes
the single-quarter figure with YTD and full-year comparatives under the
SAME tag (a 10-Q routinely tags both the 3-month and 6-month-YTD
`Revenues` value). An initial coverage measurement that counted any
duration fact landing in a calendar quarter as "covering" that quarter
was therefore counting YTD/annual figures as quarterly ones, inflating
apparent coverage (e.g. a bogus 91.8% for revenue). Restricting to
duration facts spanning 75–100 days (one real fiscal quarter) before
deduping gives the true figure — see
`scripts/analytics/build_us_10q_financials_panel.py`.

True coverage against the 517-ticker × 23-quarter (2021Q1–2026Q3, 11,891
possible firm-quarter cells) 10-Q panel, inline-XBRL only vs. combined
with the frames fallback (remeasured 2026-09-08 after widening the
`rd_expense` tag fallback to the same 4-tag chain the 10-K panel uses —
it had shipped with only `us-gaap:ResearchAndDevelopmentExpense`, missing
the `...ExcludingAcquiredInProcessCost` / software / IFRS variants):

| metric | inline-only | combined (+ frames) | gain |
|---|---|---|---|
| rd_expense | 23.6% | 25.1% | +1.5pp |
| capex | 24.7% | 27.2% | +2.5pp |
| cogs | 42.6% | 43.2% | +0.6pp |
| sga_expense | 55.6% | 57.9% | +2.3pp |
| operating_income | 56.0% | 58.9% | +2.9pp |
| debt | 58.7% | 59.4% | +0.7pp |
| revenue | 69.9% | 70.8% | +0.9pp |
| eps_diluted | 69.7% | 72.8% | +3.1pp |
| net_income | 71.4% | 74.3% | +2.9pp |
| current_assets / current_liabilities | 76.1% | 77.0% | +0.9pp |
| assets | 90.3% | 91.3% | +1.0pp |
| equity | 90.9% | 91.7% | +0.8pp |

(The widened `rd_expense` chain alone lifted inline-only coverage from
20.8% to 23.6% — a bigger gain than the entire frames fallback gives for
that metric. Absolute levels below this also drifted slightly from the
prior measurement because the corpus gained a quarter and a few tickers'
extractions since 2026-09-03/08; the fallback gain per metric — the
number that matters for the source-selection call — is stable.)

Frames alone (not as a fallback) are still worse than inline-XBRL for
every duration metric — a frame only returns a value when a filer's
fiscal period exactly matches the calendar-quarter boundary
(`CY2024Q2`/`CY2024Q2I`), silently dropping every non-December
fiscal-year-end filer for that quarter. Used as a gap-filler instead
(only where inline-XBRL has nothing for that firm-quarter), it adds a
real, if modest, 0.7–3.2pp per metric.

`rd_expense` and `capex` remain genuinely low-coverage (~20-28%) even
combined — most filers simply don't break those out as a distinct
line item every quarter, which no additional XBRL source fixes; a
tighter tag-fallback list or industry-conditional imputation would be
the next lever, not another data source.

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
