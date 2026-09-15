# AI Washing or Credible Disclosure? Patterns and Clusters in Corporate AI Disclosures

Data acquisition for a research project analyzing how listed firms
communicate about artificial intelligence in SEC filings. This repo
currently covers only data acquisition — the analysis stage (identifying
AI-related text and what it says) is future work, not built yet.

**Author:** Germán Oviedo

---

## Structure

Scripts are organized **by pipeline stage first, country/topic second**
under `scripts/<stage>/`: fetch (`raw_ingestion/`) and extraction
(`raw_processing/`) are separate stages, but each still splits by
country/form underneath, since extraction is sensitive to local
filing-format idiosyncrasies (SEC's "Item N" heading convention doesn't
generalize to another country's filings). Only genuinely country-agnostic
plumbing (checkpointing, logging, the parquet layer catalog) lives in
`scripts/common/`. `configs/` mirrors the per-country layout.

```
scripts/
  raw_ingestion/
    us/               SEC EDGAR / US-listed firms — the only country today
      00_build_firm_universe.py
      edgar_fetch.py          SEC EDGAR fetch logic (edgartools wrapper)
      sector_map.py, tui_tickers.py
      docs/            universe construction notes

      10k/             10-K panel (core instrument)
        01_fetch_filings.py    manifest + fetch, via edgartools, idempotent

      10q/             10-Q shock series — a SEPARATE instrument, never
        01_fetch_filings.py    pooled with 10-K.

    market/            Prices + Fama-French factors — independent of
      01_collect_market_data.py  both filing tracks, joins on ticker/date later

  raw_processing/
    us/                 section_segmenter.py (TOC-vs-heading segmentation —
                        US/SEC-specific), known_segmenter_issues.yaml
                        (documented, investigated extraction gaps), tests/

      10k/             02_extract_sections.py   Item 1/1A/7 -> data/interim/sections/
      10q/             02_extract_sections.py   Item 2 (MD&A) + Item 1A
                                                (Risk Factor updates) -> data/interim/sections/

  common/              Country-AGNOSTIC infra only: pipeline_logger.py,
                       section_extraction.py (run/checkpoint/traceability
                       plumbing — takes a country's segmenter functions as
                       parameters rather than importing one directly),
                       layers.py (catalog of the bronze/silver parquet tables)

  bronze/              data/bronze/: one cleaned table per source (polars)
                       manifests.py, paragraphs.py (+ text_split.py),
                       unique_paragraphs.py, market.py, prefilter.py,
                       llm_outputs.py
  enrichment/          embeddings, prefilter, LLM frames/activities, golden
                       set -> append-only data/interim/ outputs
  silver/              data/silver/: analysis universe + LLM outputs per
                       paragraph instance, ready for gold
                       universe.py, ai_outputs.py

  verif/section_audit/ Verification, not pipeline — checks the 10-K
                       extraction coverage against the full filing text

10_fusion/             Merging scripts/raw_ingestion/us + scripts/raw_ingestion/market — not built yet

configs/
  us/                  config.yaml (SEC user agent, filing windows, sector
                       mapping, storage paths) + universe.csv +
                       universe_membership.csv — a future 2nd country gets
                       its own configs/<country>/ sibling
docs/                  thesis_proposal.md and other project-level docs
models/                trained models (ai_classification/ = prefilter)
```

## Sample

- **US-listed firms** in the frozen S&P 500 (2021-12-31) universe plus a
  delisted-satellite core (see `scripts/raw_ingestion/us/docs/universe_expansion_plan.md`),
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

`scripts/raw_ingestion/us/edgar_fetch.py` is shared by both `10k/` and `10q/`.
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

### 4. Parquet layers: data/bronze/ and data/silver/

```bash
make bronze silver     # or: make layers
```

```python
import sys; sys.path.insert(0, "scripts/common")
import layers as L
L.scan("silver.ai_frames")          # polars LazyFrame
```

| layer | tables | rule |
|---|---|---|
| `bronze` | `firm_universe`, `filing_manifest`, `filing_manifest_10q`, `extraction_trace`, `paragraphs`, `sentences`, `unique_paragraphs`, `market_prices`, `market_factors_*`, `prefilter_scores`, `prefilter_predictions`, `prefilter_anchors`, `prefilter_entity_terms`, `ai_frames`, `ai_activities`, `ai_entity_mentions` | one table per source, current run per key, US, no universe filter |
| `silver` | `firm_universe`, `filing_manifest`, `filing_manifest_10q`, `ai_frames`, `ai_activities`, `ai_entity_mentions` | S&P 500 at 2021-01-01; LLM outputs broadcast to paragraph instances and restricted to the deployed prefilter population |

Lineage keys: `(country_code, form, accession_number, item_key, paragraph_index)`
→ `text_hash` → `frame_id` → `activity_id`; `bronze.extraction_trace` points
each section at its interim part file. Every table writes a
`<table>._manifest.json` (rows, key uniqueness, input files, git sha).

**Additive outputs** (`layers.ADDITIVE_SOURCES`: embeddings, prefilter
scores/predictions, LLM frames/activities, entity mentions, golden set)
cost GPU hours or LLM calls. They are append-only, live in
`data/interim/` and B2, and the layer builders only read them.

### 5. 10_fusion (not built yet)

Merging scripts/raw_ingestion/us + scripts/raw_ingestion/market is future work.

## Logging & Diagnostics

Centralized structured logging (`scripts/common/pipeline_logger.py`)
writes JSONL events to `data/interim/manifests/pipeline_log.jsonl`
(timestamp, pipeline_step, level, message, ticker, cik, accession_number,
duration_seconds, details) — readable with
`polars.read_ndjson('data/interim/manifests/pipeline_log.jsonl')`.

## Unit Testing

```bash
make test
```

Runs everything under `scripts/raw_processing/us/tests/` — currently
`test_section_segmenter.py`, covering the item-heading regex (the 3
letter-suffix formats filers actually use) and the TOC-vs-real-heading
structural detection.
