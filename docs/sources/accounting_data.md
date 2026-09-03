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
