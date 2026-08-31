# AI Washing or Credible Disclosure? Patterns and Clusters in Corporate AI Disclosures

Data collection and preprocessing pipeline for a research project analyzing how listed firms communicate about artificial intelligence in SEC 10-K filings (2021–2026): builds the firm/filing universe, extracts filing sections, and identifies which text discusses AI.

**Author:** Germán Oviedo

---

## Sample

- **115 US-listed firms** (of 120 in the initial universe; 5 excluded for incomplete filing histories) spanning **55 SIC industry groups**, covering both AI-intensive sectors (software, semiconductors) and traditional industrials, energy, healthcare, and finance.
- **Annual 10-K filings, fiscal years 2021–2026** (2026 is a partial year, filings submitted through May 2026), yielding **658 firm-year observations**.
- **AI-related candidate chunks** are extracted by the current state of the detection harness, so their count evolves with it (3,564 under the initial seed keywords; see `docs/journals/harness1_detection.md` for the state history).
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

  Cycle 1 — detection keywords ("is this text AI-related?"):
  → 07 Sample + LLM-judge label                       (data/interim/prefilter_fit/)
  → 08 Fit ai_keywords on dev, single holdout look    (updates configs/config.json)

  Cycle 2 — classification formulas ("what does the AI text say?"):
  → 09 Extract keyword atoms per chunk                (keyword_atom_features.parquet)
  → 10 Sample + LLM-judge label on 6 dimensions       (data/interim/tag_fit/)
  → 11 Fit boolean formulas on dev, single holdout look (config: tagging.formulas)
  → 12 Tag the corpus with the frozen formulas        (data/processed/tagged_chunks.parquet)
```

Both cycles are instantiations of one shared harness-optimization loop
(`scripts/harness_fit.py`) — sample → LLM judge → dev-only search → freeze →
single holdout look → gated config write — the meta-harness framing described
in `docs/meta_harness_methodology.md` (diagram: `docs/meta_harness_map.html`).
`scripts/agreement_check.py` anchors the judge itself to a human rater
(Cohen's kappa on a hand-labeled Excel subsample per cycle).

Both harnesses start from a deliberately minimal seed and grow only through
journaled optimization iterations (`docs/journals/harness1_detection.md`,
`docs/journals/harness2_classification.md` — append-only, one entry per
iteration). Iterations are run by the `meta-harness-opt` skill
(`.claude/skills/meta-harness-opt/`), one harness at a time (lock file),
always in the fit scripts' `--dev-only` mode until a state is frozen against
a fresh holdout.

### Collection and chunking (scripts 00–06)
Firm universe and filing manifest are built from SEC EDGAR; only 10-Ks are downloaded and cached locally by accession number. Business, Risk Factors, and MD&A sections are extracted with header-parsing rules (with fallbacks for non-standard filing typography). This is also where filings are classified as AI-related or not: a keyword/regex prefilter (`ai_keywords` in `configs/config.json` — generative AI, LLM, machine learning, neural network, named vendors/products, etc., with false-positive exclusions) flags candidate paragraphs at the filing level (script 05) and the paragraph level (script 06), which are then chunked with surrounding context (previous + matched + next paragraph, deduplicated by hash) to keep only ~1–5% of the token volume of the original filings.

### Cycle 1 — prefilter keyword fit (scripts 07–08)
The keyword list isn't hand-tuned once and left alone — it's fit through a repeatable cycle, not ad hoc inspection:

1. **07 (`--sample`)** draws a sample across the *entire* paragraph universe, half from paragraphs the current prefilter already flags (`in_candidate_window=True`, for precision) and half from paragraphs it currently excludes (for recall / new-keyword candidates), industry-proportional within each stratum. The 50/50 stratum draw deliberately oversamples the (small) flagged stratum, so every row records its `sampling_weight` — 08 reports inverse-probability-weighted (population) metrics next to the raw ones, which would otherwise overstate recall/F1.
2. **07 (`--label`)** gets an independent LLM judge's `is_ai_related` label for each sampled paragraph, on the **full text** (`LLM_JUDGE_*` env vars — a different model family from any classifier used elsewhere, so the judge isn't grading a close relative of itself).
3. **08** splits the labeled sample into a dev set and a holdout, stratified by label. On dev, a greedy search mines candidate keywords from dev's false negatives (LLM says AI-related, current regex misses it) and adds whichever candidate most improves the **fold-stability score** (mean F1 across stratified folds minus a std penalty, so a keyword that only helps one lucky fold doesn't get accepted), repeating until a target F1 is hit, no candidate helps anymore, or an iteration budget runs out. The fold machinery is deliberately *not* called cross-validation — nothing is trained per fold and candidates are mined from all of dev, so the dev score is optimistic by construction; the holdout is the honest number. **Every round's full candidate evaluation** (not just the winner) is written to `reports/prefilter_fit_search_trace.json` after each iteration, so the search's actual trajectory is inspectable and the run survives an interruption without losing progress already made. The frozen keyword list is then evaluated **once** on the untouched holdout (`reports/prefilter_fit_holdout_eval.txt`) — that report refuses to be regenerated against the same holdout; improving further requires sampling a fresh batch from 07.
4. `configs/config.json`'s `ai_keywords` is only overwritten after the dev search converges (never mid-run) **and** the fitted list doesn't underperform the baseline on holdout; only re-running scripts 05–06 actually applies a changed list to the corpus.
5. The search machinery itself is validated by `08 --self-check`: drop known keywords, confirm the search recovers their F1 — dev only, nothing spent (`reports/prefilter_fit_selfcheck.txt`).

### Cycle 2 — chunk classification fit (scripts 09–12)
The same cycle, instantiated for "what does the AI text say?": **09** extracts ~345 boolean keyword/regex atoms per candidate chunk; **10** samples chunks (industry-proportional, weights recorded) and has the LLM judge label all six dimensions — substantive, promotional, risk-related, governance-related, use-case-specific, quantified; **11** enumerates boolean formulas over each dimension's atom pool (singles, negations, AND/OR/AND-NOT pairs of the top singles; the pre-strip production formulas ride along as baseline candidates), scores them with the same fold-stability penalty on dev, takes one holdout look, and freezes into `configs/config.json` only the winners that beat the always-True prevalence baseline on holdout (`reports/tag_fit_*`); **12** applies the frozen formulas to every chunk → `data/processed/tagged_chunks.parquet`.

### Judge validation (human anchor)
Fully wired into the cycle — no separate step to remember: finishing a labeling run (07/10 `--label`) **automatically exports** a balanced Excel workbook of to-be-validated rows (`reports/agreement_{prefilter,tags}.xlsx`; the judge's answers sit on a separate sheet so they can't anchor you), and the fit scripts (08/11) **automatically score it** into the holdout report — Cohen's kappa per label if you filled it in, an explicit `UNVALIDATED` warning if you haven't. `scripts/agreement_check.py --cycle {prefilter,tags} --make/--score` remains as the manual CLI for the same flow. The measurement-error chain the thesis reports is: human ↔ judge (kappa) → judge ↔ harness (holdout F1) → harness → corpus (deterministic).

**Keyword list provenance.** The list starts from the minimal seed `["ai", "artificial intelligence", "machine learning"]` (no false-positive exclusions) and grows ONLY through journaled optimization iterations — every addition, its evidence, and its dev delta are recorded in `docs/journals/harness1_detection.md`. No result from any pre-reset state is carried forward; the corpus artifacts on disk are regenerated from the current seed.

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
