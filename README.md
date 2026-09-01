# AI Washing or Credible Disclosure? Patterns and Clusters in Corporate AI Disclosures

Data acquisition for a research project analyzing how listed firms
communicate about artificial intelligence in SEC 10-K filings (2021–2026).
This repo currently covers only data acquisition: building the firm/filing
universe, downloading 10-Ks, extracting their narrative sections, and
snapshotting market data. The analysis stage (identifying AI-related text
and what it says) is future work, not built yet.

**Author:** Germán Oviedo

---

## Structure

```
01_10k/           10-K text pipeline: firm universe -> manifest -> download -> section extraction
  scripts/        00-04, run in order
  docs/           universe construction notes
  verif/          verification scripts (not pipeline) — section_audit checks extraction coverage
  tests/          unit tests

02_market_data/   Prices + Fama-French factors — independent of 01_10k, joins on ticker/date later
  scripts/

10_fusion/        Merging 01_10k text output with 02_market_data — not built yet

common/           Shared infra (pipeline_logger.py) used by both 01_10k and 02_market_data

configs/          config.json (SEC user agent, filing window, sector mapping, paths) + universe.csv
docs/             thesis_proposal.md and other project-level docs not tied to a specific stage
```

## Sample

- **US-listed firms** in the frozen S&P 500 (2021-12-31) universe plus a
  delisted-satellite core (see `01_10k/docs/universe_expansion_plan.md`),
  spanning aggregated sectors (tech, semis, defense, industrials, telecom,
  autos, retail, consumer, energy, utilities, health, financials, insurance,
  real estate, materials, media, travel/leisure).
- **Annual 10-K filings, fiscal years 2021–2026** (2026 is a partial year).
- Scope limitations: 10-K only (no 8-K/10-Q/proxy/earnings-call text),
  US-listed firms only.

## What 01_10k does

Firm universe and filing manifest are built from SEC EDGAR (universe
defined in `configs/universe.csv`; sectors via config `pipeline.sector_groups`
+ per-company overrides, resolved by `01_10k/scripts/sector_map.py`); only
10-Ks are downloaded and cached by accession number. Business (Item 1),
Risk Factors (Item 1A), and MD&A (Item 7) sections are then extracted —
see `01_10k/scripts/section_segmenter.py` for how a real heading is told
apart from a table-of-contents row (structurally, not by a length guess),
and `01_10k/verif/section_audit/README.md` for how that extraction's
coverage is verified against the full filing text.

## How to run

### 0. Setup

Uses `uv` for Python environment and dependency management.

```bash
uv venv && source .venv/bin/activate && uv sync
```

### 1. Data collection (network, resumable)

**What gets downloaded is decided by `configs/universe.csv`** (ticker, cik,
company_name, inclusion_rule, active_status — see
`01_10k/docs/universe_expansion_plan.md` Phase A) plus `configs/config.json`'s
filing window (`pipeline.start_year` / `end_year`) and `pipeline.form_types`
(10-K). `pipeline.tickers` in config.json is only a generated mirror (sorted
tickers) that script 00 rewrites from universe.csv every run — never edit it
directly; edit universe.csv, or use the TUI. Firms are fetched from EDGAR by
CIK, so a ticker with no live mapping (delisted/acquired) still gets
manifest and download coverage. `pipeline.sector_groups` maps the thesis'
aggregated sectors to the SIC industry groups that compose them. To browse
or edit without touching JSON:

```bash
make tickers-tui      # view sectors -> SIC groups -> tickers; create sectors
                      # from SIC groups; add new tickers to the universe
```

Then collect:

```bash
make collect-data     # runs 01_10k/scripts 00->04 in order
```

Or step by step: `make build-universe` (00) → `make build-manifest` (01) →
`make select-batch` (02: marks pending filings inside the configured window
as selected) → `make download-filings` (03: downloads only the selected
ones) → `make extract-sections` (04). All resumable — re-running skips what's
already done. `make extract-sections ARGS='--comment my-run'` labels the
output parts with a run comment (`data/interim/sections/filing_sections__run=<id>__part=<N>__<comment>.parquet`,
glob-read by `01_10k/scripts/section_segmenter.py:load_filing_sections`).

Verify coverage against the full 10-K text (not just what got extracted):

```bash
make section-audit
```

**Market data** (for later outcome linkage — not joined to the text
pipeline yet): `make collect-market` snapshots daily adjusted prices —
**one parquet per ticker** under `data/raw/market/prices/` — plus
Fama-French 3-factor files (daily + monthly, Ken French data library).
Every row carries a `source` column (`yfinance` / `ken_french`); a CRSP
export dropped into the same per-ticker layout with `source='crsp'`
upgrades the data with no code changes — that is also the path to returns
for delisted firms, whose EDGAR filings still flow through the 01_10k
text pipeline.

### 2. 10_fusion (not built yet)

Merging the 01_10k text output with 02_market_data's price/factor panel is
future work.

## Logging & Diagnostics

Centralized structured logging (`common/pipeline_logger.py`) writes JSONL
events to `data/interim/manifests/pipeline_log.jsonl` (timestamp,
pipeline_step, level, message, ticker, cik, accession_number,
duration_seconds, details), directly queryable via DuckDB or any
JSONL-aware tool.

## Unit Testing

```bash
make test
```

Runs everything under `01_10k/tests/` — currently `test_section_segmenter.py`,
covering the item-heading regex (including the 3 letter-suffix formats
filers actually use) and the TOC-vs-real-heading structural detection.
