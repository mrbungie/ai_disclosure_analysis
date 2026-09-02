# AI Washing or Credible Disclosure? Patterns and Clusters in Corporate AI Disclosures

Data acquisition for a research project analyzing how listed firms
communicate about artificial intelligence in SEC filings. This repo
currently covers only data acquisition — the analysis stage (identifying
AI-related text and what it says) is future work, not built yet.

**Author:** Germán Oviedo

---

## Structure

Fetch AND extraction scripts are organized **by country** under
`scripts/<country>/` — extraction is sensitive to local filing-format
idiosyncrasies (SEC's "Item N" heading convention doesn't generalize to
another country's filings), so both live together rather than being
split into generic pipeline-stage folders. Only genuinely country-agnostic
plumbing (checkpointing, logging, the DuckDB view builder) lives in
`scripts/common/`. `configs/` mirrors the same per-country layout.

```
scripts/
  us/                 SEC EDGAR / US-listed firms — the only country today
    00_build_firm_universe.py
    edgar_fetch.py            SEC EDGAR fetch logic (edgartools wrapper)
    section_segmenter.py      TOC-vs-heading segmentation — US/SEC-specific
    known_segmenter_issues.yaml  documented, investigated extraction gaps
    sector_map.py, tui_tickers.py
    docs/              universe construction notes
    tests/             unit tests

    10k/               10-K panel (core instrument)
      01_fetch_filings.py      manifest + fetch, via edgartools, idempotent
      02_extract_sections.py   Item 1/1A/7 -> data/interim/sections/

    10q/               10-Q shock series — a SEPARATE instrument, never
      01_fetch_filings.py      pooled with 10-K.
      02_extract_sections.py   Item 2 (MD&A) + Item 1A (Risk Factor
                                updates) -> data/interim/sections/

  03_market_data/      Prices + Fama-French factors — independent of
    01_collect_market_data.py   both filing tracks, joins on ticker/date later

  common/              Country-AGNOSTIC infra only: pipeline_logger.py,
                       section_extraction.py (run/checkpoint/traceability
                       plumbing — takes a country's segmenter functions as
                       parameters rather than importing one directly),
                       build_duckdb.py

  verif/section_audit/ Verification, not pipeline — checks scripts/us/10k's
                       extraction coverage against the full filing text

10_fusion/             Merging scripts/us + 03_market_data — not built yet

configs/
  us/                  config.yaml (SEC user agent, filing windows, sector
                       mapping, storage paths) + universe.csv +
                       universe_membership.csv — a future 2nd country gets
                       its own configs/<country>/ sibling
docs/                  thesis_proposal.md and other project-level docs
duckdb/thesis.duckdb   SQL views over every parquet output (see below)
```

## Sample

- **US-listed firms** in the frozen S&P 500 (2021-12-31) universe plus a
  delisted-satellite core (see `scripts/us/docs/universe_expansion_plan.md`),
  spanning aggregated sectors (tech, semis, defense, industrials, telecom,
  autos, retail, consumer, energy, utilities, health, financials, insurance,
  real estate, materials, media, travel/leisure).
- **Two separate instruments, never pooled**: the 10-K panel (fiscal years
  2021–2026, 2026 partial) is the core; the 10-Q shock series covers the
  same window as its own independently-configurable instrument
  (`configs/us/config.yaml: corpus.filings_10q`).
- Scope limitations: 10-K/10-Q only (no 8-K/proxy/earnings-call text),
  US-listed firms only.

## Fetching: edgartools, not hand-rolled requests

`scripts/us/edgar_fetch.py` is shared by both `10k/` and `10q/`.
Uses `edgartools` (`Company.get_filings()` + `Filing.html()`) rather than
plain `requests.get()` calls, for two concrete reasons, both verified in
this repo's history, not assumed:

1. **Connection reuse.** Module-level `requests.get()` opens a fresh
   connection per call — no keep-alive. edgartools' httpx-based client
   pools connections to the same host.
2. **edgartools' `.download()` isn't the right primitive for a targeted
   subset.** It downloads whole daily/quarterly bulk index feed files
   (every filer, not just ours) and filters locally — wasteful for ~500
   companies. `Filing.html()` fetches exactly one filing's primary
   document; that's what we use.

**Idempotent**: a filing whose local gzip mirror
(`data/raw/filings_html/*.html.gz`) already exists is skipped outright —
verified: a rerun over an already-fetched set drops from ~30s to <1s for
a handful of filings. edgartools' own on-disk cache
(`.edgartools_data/`, gitignored, repo-internal — see
`edgar_fetch.configure()`) is a second layer under that.

**Storage**: primary documents are gzip-compressed on save. iXBRL-era
10-K/10-Qs are ~90%+ repeated markup/XBRL-context boilerplate — a real
AAPL 10-K: 1.5MB raw → 111KB gzipped. SEC already serves gzip over the
wire (`Content-Encoding: gzip`, verified directly against
`www.sec.gov`); this re-compresses on save so that ratio holds on disk
too, not just in transit.

## How to run

### 0. Setup

```bash
uv venv && source .venv/bin/activate && uv sync
```

### 1. 10-K panel (core instrument)

**What gets downloaded**: `configs/us/universe.csv` (ticker, cik, company_name,
inclusion_rule, active_status) decides which companies; `configs/us/config.yaml:
corpus.filings.filing_date` decides the window. `corpus.universe.tickers` is
only a generated mirror (sorted tickers) that script 00 rewrites from
universe.csv every run — never edit it directly; edit universe.csv, or:

```bash
make tickers-tui      # view sectors -> SIC groups -> tickers; create sectors
                      # from SIC groups; add new tickers to the universe
```

Then:

```bash
make collect-data     # build-universe -> fetch-10k -> extract-sections
```

Or step by step: `make build-universe` → `make fetch-10k` (manifest +
gzip'd HTML, idempotent) → `make extract-sections` (Item 1/1A/7, writes
`data/interim/sections/filing_sections__run=<id>__part=<N>__<comment>.parquet`
— one row per (filing, target item), found or not, with `run_id`/`part_num`/
`part_file` for full traceability; glob-read by
`section_segmenter.py:load_extraction_trace` / `:load_filing_sections`).

Verify extraction coverage against the full filing text:

```bash
make section-audit
```

### 2. 10-Q shock series (separate instrument)

Own config block (`corpus.filings_10q`), own storage
(`data/raw/filings_html_10q/`, `filing_manifest_10q.parquet`) — never
pooled with the 10-K panel:

```bash
make fetch-10q
make extract-sections-10q   # Item 2 (MD&A) + Item 1A (Risk Factor updates)
```

Item 1A is legitimately empty most quarters — it only has content when a
filer reports a material change to risk factors since the 10-K; `found ==
False` there means "no update reported," not a missed extraction.

### 3. Market data (independent of both filing tracks)

```bash
make collect-market   # per-ticker price parquets + Fama-French 3-factor files
```

Every price row carries a `source` column (`yfinance` / `ken_french`); a
CRSP export dropped into the same per-ticker layout with `source='crsp'`
upgrades the data with no code changes — also the path to returns for
delisted firms.

### 4. SQL access: duckdb/thesis.duckdb

```bash
make duckdb
duckdb duckdb/thesis.duckdb
```

VIEWS (not materialized tables — re-evaluate their `read_parquet(glob)` on
every query, so they're always current, no reimport step after a new
extraction run or fetch):

| view | source |
|---|---|
| `firm_universe` | `data/interim/manifests/firm_universe.parquet` |
| `filing_manifest` | `data/interim/manifests/filing_manifest.parquet` |
| `extraction_trace` | every `filing_sections__run=*__part=*.parquet`, deduped to the latest run per (filing, item) — found or not |
| `filing_sections` | `extraction_trace` filtered to `found` |
| `filing_manifest_10q` | `data/interim/manifests/filing_manifest_10q.parquet` |
| `extraction_trace_10q` / `filing_sections_10q` | 10-Q equivalents of the two views above — never unioned with the 10-K ones |
| `market_prices` | every per-ticker price parquet |
| `market_factors_daily` / `market_factors_monthly` | Fama-French factor files |

### 5. 10_fusion (not built yet)

Merging scripts/us + 03_market_data is future work.

## Logging & Diagnostics

Centralized structured logging (`scripts/common/pipeline_logger.py`)
writes JSONL events to `data/interim/manifests/pipeline_log.jsonl`
(timestamp, pipeline_step, level, message, ticker, cik, accession_number,
duration_seconds, details) — queryable directly from DuckDB too
(`read_json_auto('data/interim/manifests/pipeline_log.jsonl')`).

## Unit Testing

```bash
make test
```

Runs everything under `scripts/us/tests/` — currently
`test_section_segmenter.py`, covering the item-heading regex (the 3
letter-suffix formats filers actually use) and the TOC-vs-real-heading
structural detection.
