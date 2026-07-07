# AI Washing or Credible Disclosure? Patterns and Clusters in Corporate AI Disclosures

Data collection and preprocessing pipeline for a research project analyzing how listed firms communicate about artificial intelligence in SEC 10-K filings (2021–2026): builds the firm/filing universe, extracts filing sections, and identifies which text discusses AI.

**Author:** Germán Oviedo

---

## Sample

- **115 US-listed firms** (of 120 in the initial universe; 5 excluded for incomplete filing histories) spanning **55 SIC industry groups**, covering both AI-intensive sectors (software, semiconductors) and traditional industrials, energy, healthcare, and finance.
- **Annual 10-K filings, fiscal years 2021–2026** (2026 is a partial year, filings submitted through May 2026), yielding **658 firm-year observations**.
- **4,228 AI-related candidate chunks** extracted, of which 43% fall in Risk Factors (Item 1A), 42% in Business (Item 1), and 16% in MD&A (Item 7).
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
  → 05 Prefilter AI mentions (keyword/regex) (manifest updated)
  → 06 Chunk candidates                      (ai_candidate_chunks.parquet)

  → 07 Sample + LLM-label for the prefilter fit cycle (data/interim/prefilter_fit/)
  → 08 CV harness: fit ai_keywords against dev, evaluate once on holdout
       (updates configs/config.json)
```

### Collection and chunking (scripts 00–06)
Firm universe and filing manifest are built from SEC EDGAR; only 10-Ks are downloaded and cached locally by accession number. Business, Risk Factors, and MD&A sections are extracted with header-parsing rules (with fallbacks for non-standard filing typography). This is also where filings are classified as AI-related or not: a keyword/regex prefilter (`ai_keywords` in `configs/config.json` — generative AI, LLM, machine learning, neural network, named vendors/products, etc., with false-positive exclusions) flags candidate paragraphs at the filing level (script 05) and the paragraph level (script 06), which are then chunked with surrounding context (previous + matched + next paragraph, deduplicated by hash) to keep only ~1–5% of the token volume of the original filings.

### Semi-reproducible prefilter fit (scripts 07–08)
The keyword list isn't hand-tuned once and left alone — it's fit through a repeatable cycle, not ad hoc inspection:

1. **07 (`--sample`)** draws an industry-stratified sample across the *entire* paragraph universe, half from paragraphs the current prefilter already flags (`in_candidate_window=True`, for precision) and half from paragraphs it currently excludes (for recall / new-keyword candidates) — proportional allocation, no oversampling.
2. **07 (`--label`)** gets an independent LLM judge's `is_ai_related` label for each sampled paragraph (`LLM_JUDGE_*` env vars — a different model family from any classifier used elsewhere, so the judge isn't grading a close relative of itself).
3. **08** splits the labeled sample into a dev set and a holdout, stratified by label. On dev, a greedy k-fold-CV search mines candidate keywords from dev's false negatives (LLM says AI-related, current regex misses it — the same kind of gap "AIP"/"AIOps" were, just found automatically instead of by manual inspection) and adds whichever candidate most improves the CV mean F1 (penalized by its std across folds, so a keyword that only helps one fold doesn't get accepted), repeating until a target F1 is hit, no candidate helps anymore, or an iteration budget runs out. The frozen keyword list is then evaluated **once** on the untouched holdout (`reports/prefilter_fit_holdout_eval.txt`) — that report refuses to be regenerated against the same holdout; improving further requires sampling a fresh batch from 07.
4. `configs/config.json`'s `ai_keywords` is only overwritten after the dev search converges (never mid-run), and only re-running scripts 05–06 actually applies a changed list to the corpus.

**Keyword list provenance.** The list started from a broad set of AI/ML terms and named vendors/products, and was extended after this cycle found two gaps: Palantir's "AIP" product name and Snowflake's use of "ML" as a standalone acronym — neither matched because the regex requires the full word (`\bai\b` doesn't match inside "AIP"). `aip` and `aiops` were added after confirming (via direct corpus search) that they only ever appear for Palantir/Panw/Datadog/Broadcom in this sample, with zero false-positive risk — unlike a broader `AI[A-Z]+` acronym pattern, which mostly matches unrelated terms (AIDS, AICPA, aircraft). The most recent fit run (700 sampled paragraphs, 479 dev / 205 holdout) found the current list already meets the target F1 (dev CV F1=0.923, holdout F1=0.960) with no further candidates needed.

---

## Setup & Installation

Uses `uv` for Python environment and dependency management.

```bash
uv venv
source .venv/bin/activate
uv sync
```

## Makefile Automation

```bash
make test               # unit tests
make install-deps       # install pytest
make run-pipeline        # prefilter + chunk candidates (scripts 05-06)
```

Scripts 00–04 are run directly (`.venv/bin/python scripts/00_build_firm_universe.py`, etc.) since they involve one-time setup and network calls to SEC EDGAR rather than being re-run repeatedly.

## Logging & Diagnostics

Centralized structured logging (`scripts/pipeline_logger.py`) writes JSONL events to `data/interim/manifests/pipeline_log.jsonl` (timestamp, pipeline_step, level, message, ticker, cik, accession_number, duration_seconds, details), directly queryable via DuckDB or any JSONL-aware tool.

## Unit Testing

`tests/` covers section-extraction regex robustness (standard/pipe-wrapped/plaintext-fallback headers, boundary checks, minimum-length filtering).
