# AI Washing or Credible Disclosure? Patterns and Clusters in Corporate AI Disclosures

Data collection and preprocessing pipeline for a research project analyzing how listed firms communicate about artificial intelligence in SEC 10-K filings (2021–2026): builds the firm/filing universe, extracts filing sections, and identifies which text discusses AI.

**Author:** Germán Oviedo

---

## Sample

- **115 US-listed firms** (of 120 in the initial universe; 5 excluded for incomplete filing histories) spanning **55 SIC industry groups**, covering both AI-intensive sectors (software, semiconductors) and traditional industrials, energy, healthcare, and finance.
- **Annual 10-K filings, fiscal years 2021–2026** (2026 is a partial year, filings submitted through May 2026), yielding **658 firm-year observations**.
- **AI-related text** is identified by the current ACTIVE detection candidate, so counts evolve with the harness state (see `.claude/skills/meta-harness-opt/journals/`).
- Scope limitations: 10-K only (no 8-K/10-Q/proxy/earnings-call text), US-listed firms only.

---

## Pipeline

File-first, incremental system (Parquet/JSONL, no production database), resumable at every stage.

```
SEC EDGAR
  → 00 Build firm universe                  (firm_universe.parquet)
  → 01 Build filing manifest                (filing_manifest.parquet)
  → 02 Select download batch
  → 03 Download filings                     (filings_html/)
  → 04 Extract sections (Business/Risk/MD&A) (filing_sections.parquet)

  The two harnesses (candidate programs, harnesses/<task>/<candidate>/harness.py):
    detection       classify(text) -> bool           "is this text AI-related?"
    classification  classify(text) -> 6 dim booleans "what does the AI text say?"

  → build_eval_set   sample paragraphs + reward-label them (one labeled set
                     serves both tasks; fixed search/test split; weights)
  → eval_harness     Evaluate(H, X): score a candidate on the search split,
                     log scores + per-instance traces to its directory
  → /meta-harness-opt <task>   the proposer: reads all prior candidates'
                     code/scores/traces, writes a NEW candidate, evaluates it
  → eval_harness --split test  ONE look per task per batch = the freeze
                     (promotes the candidate to harnesses/<task>/ACTIVE)
  → apply_harness    frozen detection pre-classifies every corpus paragraph;
                     frozen classification tags the positives
                     -> data/processed/classified_paragraphs.parquet
```

This is the meta-harness pattern (docs/meta_harness_methodology.md): each
task's harness is a self-contained program searched by an agentic proposer
with filesystem access to every prior candidate's source, scores, and
per-instance traces. The search split is spent freely; the test split is
looked at once per labeled batch and then locked. Reward labels come from an
LLM labeler and are human-audited (Cohen's kappa). Both harnesses start from
deliberately minimal `000_seed` candidates and grow only through journaled
proposer iterations (`.claude/skills/meta-harness-opt/journals/`).

### Collection (scripts 00–04)
Firm universe and filing manifest are built from SEC EDGAR (universe defined
in `configs/universe.csv`; sectors via config sector_groups + per-company
overrides, resolved by `scripts/sector_map.py`); only 10-Ks are downloaded
and cached by accession number. Business, Risk Factors, and MD&A sections
are extracted with header-parsing rules.

### The harnesses (harnesses/, scripts/build_eval_set.py + eval_harness.py)
Each candidate is ONE self-contained stdlib-only `harness.py` — keywords,
patterns, staging are internal to the candidate, so the whole program is
auditable and the proposer can rewrite any part. `eval_harness.py` scores a
candidate against the labeled eval set: detection on all rows
(`is_ai_related`), classification on the AI-labeled rows (six dimensions:
substantive, promotional, risk, governance, use-case-specific, quantified);
reward = (macro-)F1, reported raw (sample as drawn — optimistic by design)
and inverse-probability weighted (the honest population estimate, via each
row's `sampling_weight`). Scores and full per-instance traces land in the
candidate's directory — the filesystem history the proposer greps. The
search/test split is fixed when the eval set is sampled; `--split test` is
refused after its one use until a fresh batch is labeled.

### Reward labels (human anchor)
`build_eval_set --label` obtains all seven labels per sampled paragraph from
an LLM labeler (`LLM_JUDGE_*` env vars, full text) and auto-exports a
balanced audit workbook (`reports/agreement_eval.xlsx`; stored labels on a
separate sheet so they can't anchor you). `make agreement-score` computes
Cohen's kappa; the freeze output includes it or an explicit UNVALIDATED
warning. The chain the thesis reports: human ↔ reward labels (kappa) →
harness ↔ reward labels (test F1) → corpus (deterministic).

## How to run

### 0. Setup

Uses `uv` for Python environment and dependency management.

```bash
uv venv && source .venv/bin/activate && uv sync
cp .env.example .env    # then set LLM_JUDGE_API_KEY / LLM_JUDGE_BASE_URL / LLM_JUDGE_MODEL
```

The judge env vars are only needed for the `--label` steps; everything else is
offline regex/pandas work.

### 1. Data collection (network, resumable)

**What gets downloaded is decided by `configs/universe.csv`** (the firm
universe: ticker, cik, company_name, inclusion_rule, active_status — see
`docs/universe_expansion_plan.md` Phase A) plus `configs/config.json`'s
filing window (`pipeline.start_year` / `end_year`) and `pipeline.form_types`
(10-K). `pipeline.tickers` in config.json is only a generated mirror (sorted
tickers) that script 00 rewrites from universe.csv every run, kept for
legacy/TUI code paths — never edit it directly; edit universe.csv, or use the
TUI. Firms are fetched from EDGAR by CIK, so a ticker with no live mapping
(delisted/acquired) still gets manifest and download coverage. On top of
that, `pipeline.sector_groups`
maps the thesis' **aggregated sectors** (tech, semis, defensa, industriales,
telecom, autos, retail, consumo, energia, utilities, salud, financieras) to
the SIC industry groups that compose them — a sector's tickers are derived
from its SIC groups. To browse or edit without touching JSON:

```bash
make tickers-tui      # view sectors -> SIC groups -> tickers; create sectors
                      # from SIC groups; add new tickers to the universe
```

Then collect:

```bash
make collect-data     # runs 00->04 in order
```

Or step by step: `make build-universe` (00) → `make build-manifest` (01) →
`make select-batch` (02: marks pending filings inside the configured window
as selected) → `make download-filings` (03: downloads only the selected
ones) → `make extract-sections` (04). All resumable — re-running skips what's
already done.

**Market data** (outcome linkage): `make collect-market` (13) snapshots daily
adjusted prices — **one parquet per ticker** under `data/raw/market/prices/`,
so adding tickers later is just a re-run and a broken download can't corrupt
the rest — plus Fama-French 3-factor files (daily + monthly, Ken French data
library) for abnormal returns. Every row carries a `source` column
(`yfinance` / `ken_french`); a CRSP export dropped into the same per-ticker
layout with `source='crsp'` upgrades the data with no code changes — that is
also the path to returns for delisted firms (e.g. DFS, acquired 2025; SQ,
renamed XYZ), whose EDGAR filings still flow through the text pipeline.

### 2. Build the eval set (one labeled batch serves both tasks)

```bash
make eval-sample                # draw paragraphs (free; stratified by ACTIVE detection)
make eval-label                 # reward labels ($, needs .env) -> auto-exports reports/agreement_eval.xlsx
# hand-label the workbook's 'label_me' sheet (instructions inside)
```

### 3. Optimize the harnesses (the proposer loop)

```
/meta-harness-opt detection         # in Claude Code — one iteration: read traces,
/meta-harness-opt classification    # write a new candidate, score it on search
```

Search-split evaluations are free and repeatable
(`make eval-harness ARGS='--task detection --candidate 001_x'`,
`make leaderboard`).

### 4. Freeze and apply

```bash
make eval-harness ARGS='--task detection --candidate <best> --split test'   # THE one look; promotes to ACTIVE
make eval-harness ARGS='--task classification --candidate <best> --split test'
make apply-harness              # frozen candidates -> data/processed/classified_paragraphs.parquet
```

A spent test split means the next freeze needs a fresh `eval-sample` +
`eval-label` batch.

### Rules of the road

- Search-split evaluation is always safe to re-run; `--split test` spends
  that task's one look for the batch — the next freeze needs a fresh labeled
  eval set.
- Never edit an evaluated candidate's `harness.py`: new idea = new candidate
  directory. History is immutable; `harnesses/<task>/ACTIVE` is the freeze.
- The proposer (skill) is the only search operator — one optimization at a
  time (`OPT_LOCK`).
- `make test` runs the unit tests; `make help` lists every target.

## Logging & Diagnostics

Centralized structured logging (`scripts/pipeline_logger.py`) writes JSONL events to `data/interim/manifests/pipeline_log.jsonl` (timestamp, pipeline_step, level, message, ticker, cik, accession_number, duration_seconds, details), directly queryable via DuckDB or any JSONL-aware tool.

## Unit Testing

`tests/` covers section-extraction regex robustness (standard/pipe-wrapped/plaintext-fallback headers, boundary checks, minimum-length filtering).
