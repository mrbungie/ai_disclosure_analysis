# AI Washing or Credible Disclosure? Patterns and Clusters in Corporate AI Disclosures

Data collection and preprocessing pipeline for a research project analyzing how listed firms communicate about artificial intelligence in SEC 10-K filings (2021–2026): builds the firm/filing universe, extracts filing sections, and identifies which text discusses AI and how.

**Author:** Germán Oviedo

---

## Sample

- **US-listed firms** in the frozen S&P 500 (2021-12-31) universe plus a delisted-satellite core (see `docs/universe_expansion_plan.md`), spanning aggregated sectors (tech, semis, defense, industrials, telecom, autos, retail, consumer, energy, utilities, health, financials, insurance, real estate, materials, media, travel/leisure).
- **Annual 10-K filings, fiscal years 2021–2026** (2026 is a partial year).
- **AI-related text** is identified by the ACTIVE detection candidate (optionally widened by the ACTIVE phase0 candidate); **what it says about AI** by the ACTIVE classification candidate — so both counts evolve with harness state (see `.claude/skills/detection-opt/journals/`, `.claude/skills/phase0-opt/journals/`, `.claude/skills/classification-opt/journals/`).
- Scope limitations: 10-K only (no 8-K/10-Q/proxy/earnings-call text), US-listed firms only.

---

## Methodology: a two-stage distillation cascade

Full spec: **`docs/distillation_map.html`**. Summary:

An expensive LLM reference classifier can't run over the whole corpus, so two
things are distilled into cheap programs instead, in sequence:

1. **Detection (stage 1).** A lexical **seed screen** (`scripts/seed_screen.py`)
   — fixed, deliberately over-inclusive keyword matching, never scored itself
   — flags every paragraph and every filing, producing three sampling strata:
   `hit`, `no_hit_filing_hits` (a miss inside a filing that hit elsewhere),
   `no_hit_filing_clean` (a miss inside a filing with no hits anywhere). All
   three keep positive inclusion probability. A search sample (weighted
   toward hard cases) and a probability holdout (stratified by year/sector,
   with guaranteed coverage of all three strata) are drawn from this
   population and reward-labeled by an LLM judge against a short **scope
   rule**: does this paragraph discuss AI at all? Candidates
   (`harnesses/detection/<name>/harness.py`, `classify(text) -> bool`) are
   scored by **minimizing predicted-positive volume subject to a weighted
   search recall floor** — recall is a gate, not something to maximize past
   it, since unconstrained recall has a trivial optimum (admit everything).
2. **Classification (stage 2).** Once detection is frozen and applied, its
   admitted paragraphs form the **candidate frame**
   (`data/processed/candidate_frame.parquet`). Chunks are built around them
   (±1 paragraph window, merged when overlapping —
   `harness_fit.build_chunks`) so the reference judge has enough context.
   Chunks are reward-labeled against the six-dimension **codebook**
   (substantive, promotional, risk-related, governance-related, use-case-
   specific, quantified). Candidates
   (`harnesses/classification/<name>/harness.py`, `classify(text) -> dict[6
   bools]`) are scored by **macro-averaged balanced accuracy** (mean of
   sensitivity and specificity per label) — not F1, since the six labels'
   base rates differ enough that raw agreement can be trivially high.
3. **Rollout.** A chunk's six tags are broadcast back to every paragraph
   inside it, so the final panel (`data/processed/classified_paragraphs.parquet`)
   stays **one row per paragraph** even though the classification decision
   was made at chunk granularity. Non-AI paragraphs get `NA` dimension tags
   (not `False` — the task is only defined on AI text).

Both stages (plus phase0, an optional third task that widens detection's
seed screen with embedding/ConceptSeed candidates instead of literal
keywords — `docs/distillation_map.html` §0) share one outer loop
(`scripts/harness_fit.py`): a search split (spent freely, weighted toward
hard cases) and a probability holdout (inclusion weights recorded, opened
once). Iteration is done by an agentic proposer — one skill per task
(`/detection-opt`, `/phase0-opt`, `/classification-opt`) — with filesystem
access to every prior candidate's source, scores, and per-instance traces —
one candidate per iteration, immutable history, one optimization at a time
per task (`OPT_LOCK`). Reward labels come from an LLM judge and are
human-audited (Cohen's kappa) before being trusted.

### Collection (scripts 00–04)
Firm universe and filing manifest are built from SEC EDGAR (universe defined
in `configs/universe.csv`; sectors via config `sector_groups` + per-company
overrides, resolved by `scripts/sector_map.py`); only 10-Ks are downloaded
and cached by accession number. Business, Risk Factors, and MD&A sections
are extracted with header-parsing rules.

## How to run

### 0. Setup

Uses `uv` for Python environment and dependency management.

```bash
uv venv && source .venv/bin/activate && uv sync
cp .env.example .env    # then set LLM_JUDGE_API_KEY / LLM_JUDGE_BASE_URL / LLM_JUDGE_MODEL
```

The judge env vars are only needed for `--label` steps; everything else is
offline regex/pandas work.

### 1. Data collection (network, resumable)

**What gets downloaded is decided by `configs/universe.csv`** (ticker, cik,
company_name, inclusion_rule, active_status — see
`docs/universe_expansion_plan.md` Phase A) plus `configs/config.json`'s
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
make collect-data     # runs 00->04 in order
```

Or step by step: `make build-universe` (00) → `make build-manifest` (01) →
`make select-batch` (02: marks pending filings inside the configured window
as selected) → `make download-filings` (03: downloads only the selected
ones) → `make extract-sections` (04). All resumable — re-running skips what's
already done.

**Market data** (outcome linkage): `make collect-market` (13) snapshots daily
adjusted prices — **one parquet per ticker** under `data/raw/market/prices/`
— plus Fama-French 3-factor files (daily + monthly, Ken French data library)
for abnormal returns. Every row carries a `source` column (`yfinance` /
`ken_french`); a CRSP export dropped into the same per-ticker layout with
`source='crsp'` upgrades the data with no code changes — that is also the
path to returns for delisted firms, whose EDGAR filings still flow through
the text pipeline.

### 2. Stage 1 — detection

The seed screen must exist before sampling (run once; re-run only if the
corpus grows or the keyword list in `configs/config.json: seed_screen`
changes):

```bash
make seed-screen
```

Then the sample/label/iterate/freeze cycle:

```bash
make eval-sample-detection             # free. ARGS='--n-search 560 --n-holdout 240'
make eval-label-detection              # $, LLM judge — scope only (is_ai_related)
# hand-label reports/agreement_eval_detection.xlsx's 'label_me' sheet (auto-exported)
make agreement-score ARGS='--cycle eval_detection'   # Cohen's kappa
```

```
/detection-opt                         # in Claude Code — one proposer iteration:
                                        # reads traces, writes ONE new candidate, scores it
```

```bash
make eval-harness ARGS='--task detection --candidate 001_x'   # free, repeatable
make leaderboard ARGS='--task detection'                       # standings + who the floor selects
make eval-harness ARGS='--task detection --candidate <best> --split test'   # THE one look; promotes to ACTIVE
make apply-detection                   # writes data/processed/candidate_frame.parquet
```

The freeze is gated: it refuses if the candidate's recorded search recall
is below `configs/config.json: seed_screen.recall_floor`.

### 2b. phase0 (optional — widens the seed screen semantically)

Reuses detection's eval set; no separate sampling step. `harnesses/phase0/`
candidates are `classify(text) -> bool` too, but scored as unique recall
gain over the current detection ACTIVE (not raw precision/recall):

```bash
uv run python scripts/phase0_discovery.py --sample     # free, discovery sample only
uv run python scripts/phase0_discovery.py --induce     # $, one LLM call -> a ConceptSeed suggestion
```

```
/phase0-opt                            # one proposer iteration: curates the suggestion into a
                                        # harnesses/phase0/<name>/{harness.py,concept_seed.json}, scores it
```

```bash
make eval-harness ARGS='--task phase0 --candidate 001_x'                     # free, repeatable
make leaderboard ARGS='--task phase0'                                        # standings vs the volume ceiling
make eval-harness ARGS='--task phase0 --candidate <best> --split test'       # THE one look; promotes to ACTIVE
make seed-screen                       # re-run so semantic_hit takes effect (needs phase0.enabled: true)
```

The freeze is gated: it refuses if the candidate's recorded added volume
is above `configs/config.json: phase0.max_added_volume_frac`. Embeddings
are cached per model (`configs/config.json: phase0.active_embedding_model`
picks which) under `data/interim/embeddings/` and are never deleted —
switching models costs disk, not lost work.

### 3. Stage 2 — classification (needs the candidate frame above)

```bash
make eval-sample-classification        # free. ARGS='--n 800'. builds chunks over the candidate frame
make eval-label-classification         # $, LLM judge — six dimensions
# hand-label reports/agreement_eval_classification.xlsx
make agreement-score ARGS='--cycle eval_classification'
```

```
/classification-opt                    # one proposer iteration
```

```bash
make eval-harness ARGS='--task classification --candidate 001_x'
make eval-harness ARGS='--task classification --candidate <best> --split test'   # THE one look; promotes to ACTIVE
```

### 4. Full corpus pass

```bash
make apply-harness      # both frozen candidates -> data/processed/classified_paragraphs.parquet
                        # (paragraph-level; chunk tags broadcast back to member paragraphs)
```

A spent test split (for either stage) means the next freeze on that stage
needs a fresh `eval-sample-*` + `eval-label-*` batch.

### Rules of the road

- Search-split evaluation is always safe to re-run; `--split test` spends
  that stage's one look for the batch — the next freeze needs a fresh
  labeled eval set.
- Never edit an evaluated candidate's `harness.py`: new idea = new candidate
  directory. History is immutable; `harnesses/<task>/ACTIVE` is the freeze.
- Classification can't sample until detection is frozen and
  `make apply-detection` has written the candidate frame — sampling refuses
  otherwise.
- The proposer (skill) is the only search operator — one optimization at a
  time (`OPT_LOCK`).
- `make test` runs the unit tests; `make help` lists every target.

## Logging & Diagnostics

Centralized structured logging (`scripts/pipeline_logger.py`) writes JSONL events to `data/interim/manifests/pipeline_log.jsonl` (timestamp, pipeline_step, level, message, ticker, cik, accession_number, duration_seconds, details), directly queryable via DuckDB or any JSONL-aware tool.

## Unit Testing

`tests/` covers section-extraction regex robustness (standard/pipe-wrapped/plaintext-fallback headers, boundary checks, minimum-length filtering).
