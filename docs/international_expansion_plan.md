# International expansion plan

Decided 2026-09-02. Extends the US-only corpus (SEC EDGAR, S&P 500 core —
see `scripts/us/docs/universe_expansion_plan.md`) into a genuinely
comparative panel across three distinct disclosure regimes: **US
(mandatory quarterly), Italy (mandatory annual + semi-annual, quarterly
voluntary since 2016), Chile (mandatory quarterly, CMF)**. Order of
work: **US (done) → Chile (next) → Italy**. Chile goes first because its
tooling is the most EDGAR-like already and its universe is small enough
to take whole, not because it's less work overall.

## Decision: incremental downloader per regulator, normalize after

**Not** a second `edgartools`-equivalent per country, and **not** one
unified fetch abstraction attempted up front. Each regulator publishes
different things in different shapes (SEC: HTML/iXBRL via a documented
REST-ish API; CONSOB/CMF-equivalent-Italy: XBRL via `filings.xbrl.org`,
a public aggregator, not a scrape target; CMF Chile: PDF + XBRL + an
existing MCP server that already speaks its data model). Country-specific
fetch logic, one per source, feeding a **shared normalized manifest
schema** downstream is exactly the design already built for the US pipe
— `scripts/us/` (fetch + extraction together, since extraction is
sensitive to that regulator's own formatting idiosyncrasies) and
`scripts/common/` (country-agnostic checkpointing, logging, the DuckDB
view builder) — a second country is a new sibling `scripts/<country>/` +
`configs/<country>/`, not a rewrite of either. `build_duckdb.py` already
UNIONs BY NAME across every `configs/<country>/config.yaml` found (see
its module docstring) — this was built ahead of need specifically so
adding Chile is additive, not a migration.

## Instrument mapping (US ↔ Chile ↔ Italy)

No claim of legal equivalence between these documents — they are
*functionally* comparable disclosure events for a firm-year/firm-period
panel, not the same legal instrument. The manifest keeps the original
regulatory name (`source_filing_type`) separate from a normalized
`filing_type` used for cross-country joins (see schema below), so this
mapping is a query-time convenience, never baked into raw storage.

| Normalized `filing_type` | US (SEC) | Chile (CMF) | Italy (CONSOB/ESMA) |
|---|---|---|---|
| `annual` | 10-K | Memoria anual + EEFF anual (IFRS/XBRL) + Análisis Razonado | Relazione finanziaria annuale (ESEF/XBRL) |
| `half_year` | — (no exact US equivalent; nearest is 10-Q for the covering quarter) | EEFF junio + Análisis Razonado | Relazione finanziaria semestrale (mandatory) |
| `quarterly` | 10-Q | EEFF trimestral + Análisis Razonado (Q1/Q3) | Resoconto intermedio di gestione (**voluntary since 2016** — sparse by construction, not a gap in coverage) |

Cross-country comparison is done at `firm-year` (`annual` row) as the
primary panel unit; intra-year instruments (`half_year`/`quarterly`) are
a supplementary, country-specific shock series exactly like the existing
10-Q track — never pooled into the annual panel (same rule as
`[[data-scope-two-instruments]]`: two genuinely different objects stay
two objects, joined by key, not merged into one).

## Universe per country

- **US**: S&P 500 frozen at 2021-12-31 + sector top-ups + satellite —
  see `scripts/us/docs/universe_expansion_plan.md`. Already built.
- **Chile**: **all CMF equity issuers**, not just IPSA — the universe is
  small enough (~200 issuers) that narrowing it loses data without
  saving meaningful download/compute time. Source: `mcp-cmf-chile`
  (JoaquinMulet/mcp-cmf-chile) — exposes listed companies, memorias
  anuales, EEFF IFRS/XBRL, análisis razonado, hechos esenciales, and a
  package-level operation that assembles a firm's full filing set.
- **Italy**: start with FTSE MIB (~40 constituents, publicly listed by
  Borsa Italiana), expand to FTSE Italia All-Share only if the pipeline
  proves out — smaller universe than Chile, so start narrower and widen
  is the safer order here, opposite of Chile's "just take it all" case.
  Source: `filings.xbrl.org` (`lsalmela/xbrl-filings-api`), the same
  aggregator ESMA's own toolkit uses instead of crawling national OAMs
  directly — filtered by `filter[country]=IT`, paginated. XHTML pulled
  alongside the XBRL: for AI-disclosure narrative (adoption, governance,
  risk, promotional claims), the XBRL numeric facts are close to
  irrelevant — the prose is in the XHTML, same reasoning as why this
  project extracts Item 1/1A/7 text rather than XBRL facts for the US.

## Adaptability constraint: additive, never re-downloaded

Per `CLAUDE.md`, no data-affecting step in this expansion may require
deleting or re-fetching anything already in `data/` — this shapes the
design choices below, not just a caveat at the end:

1. **Per-country storage never collides.** Each `configs/<country>/
   config.yaml:storage` points at `data/.../<country>/...` — enforced by
   convention today (`build_duckdb.py`'s `_country_configs()` docstring
   flags this explicitly), should be enforced by a startup check before
   Chile's config is added, so a future path typo can't silently
   overwrite another country's manifest.
2. **Fetch is idempotent by construction, so widening a universe or
   date window is a re-run, not a redo.** The US fetcher already skips
   any filing whose local mirror exists (`scripts/us/edgar_fetch.py`);
   Chile/Italy fetchers must keep that same contract — adding 50 more
   CMF issuers next year re-runs `fetch-cl` and only touches the new
   rows, same pattern as `make fetch-10k` today.
3. **Normalized schema on top of untouched raw storage** — a
   `filing_type` remap (e.g. deciding `half_year` should map differently
   for cross-country panels) is a change to a DuckDB view definition
   (code), never a rewrite of the parquet manifest or raw filings
   underneath it. This mirrors the `paragraphs`/`sentences` design
   already in place: the SQL that defines a normalized shape lives as
   code (`scripts/common/build_duckdb.py`), re-runnable any time, over
   raw data that's written once.
4. **`source_filing_type` is never discarded**, even after normalization
   — the day a fourth country's "closest thing to a 10-Q" turns out not
   to fit `quarterly`/`half_year` cleanly, the fix is a new enum value or
   a mapping-table edit, not a re-derivation from raw filings (which
   would still work, since raw filings are untouched, but shouldn't be
   *necessary*).
5. **LLM-derived artifacts stay under `data/archive/` on removal**, per
   `CLAUDE.md` — applies identically once Chile/Italy filings start
   feeding the same `eval_set_detection.parquet`-style labeled sets.

## Shared manifest schema (target)

Extends the existing US-only manifest (`document_id`, `cik`, `ticker`,
`country`, `source`, `form_type`, `filing_date`, `period_end_date`,
`accession_number`, `sec_url`, `local_path`, ... — see
`scripts/us/edgar_fetch.py`) with the normalization layer described
above, added as new columns rather than renamed ones (so the US
manifest's existing columns need no migration):

```
document_id         existing per-country natural key (accession_number for US, ...)
cik                  existing — country-specific issuer id (CIK / RUT / Italian fiscal code)
ticker
country_code         'us' | 'cl' | 'it' | ...
source               'SEC_EDGAR' | 'CMF' | 'filings.xbrl.org' | ...
form_type            existing — VERBATIM regulator label ("10-K", "Memoria Anual", "Relazione finanziaria annuale")
filing_type          NEW — normalized: 'annual' | 'half_year' | 'quarterly'
filing_date
period_end_date
local_path
format               NEW — 'html' | 'xhtml' | 'pdf' | 'xbrl_zip' (US is HTML-only today; Chile/Italy are not)
...download_status / parse_status / prefilter_status / llm_status / priority_score  (unchanged, per-country)
```

`country_code` already exists end-to-end in the DuckDB layer
(`paragraphs`/`sentences`/every unioned view carries it — see
`scripts/common/build_duckdb.py`); this table only adds `filing_type`
and `format` as genuinely new concepts the US-only schema didn't need.

## Execution order (Chile next)

**Phase A — Chile pipeline skeleton (no network beyond `mcp-cmf-chile`'s own calls)**
1. `scripts/cl/` (fetch + extraction together, same reasoning as
   `scripts/us/`: CMF's own filing-format idiosyncrasies belong next to
   its fetch code, not in `scripts/common/`) + `configs/cl/config.yaml`
   (mirrors `configs/us/config.yaml`'s shape: universe table +
   membership tags + filing-date window, per the existing
   `universe.csv`/`universe_membership.csv` pattern).
2. Universe: all CMF equity issuers, resolved to RUT (or CMF's own
   issuer id), stored the same way as `configs/us/universe.csv` — one
   versioned table, inclusion-reason tags in a side file.
3. Fetch via `mcp-cmf-chile`'s package-level operation (memoria anual +
   EEFF anual/trimestral + análisis razonado), gzip'd raw storage under
   `data/raw/filings_<fmt>_cl/`, own manifest
   (`data/interim/manifests/filing_manifest_cl.parquet`) — idempotent
   from day one, not retrofitted later.

**Phase B — Extraction**
4. Chile's "sections" are less item-numbered than SEC filings (no
   "Item 1A" convention) — extraction target is closer to "AI-relevant
   passages within Análisis Razonado + Memoria narrative sections"
   than a fixed item list; expect this to need its own segmentation
   logic, not a port of `scripts/us/section_segmenter.py`'s TOC/Item-N
   approach, which is SEC-specific by construction.

**Phase C — Wire into build_duckdb.py**
5. `configs/cl/config.yaml` appearing under `configs/` is enough for
   `_country_configs()` to pick it up automatically — confirm the
   UNION ALL BY NAME across `us`/`cl` produces clean `firm_universe`/
   `filing_manifest`/`paragraphs`/`sentences` views with no code change
   beyond whatever new columns Chile's schema genuinely needs (`format`,
   `filing_type`).

**Phase D — Italy**, once Chile's pipeline is validated end-to-end
(fetch → extract → DuckDB views → a real sample read), repeating the
same phases against `filings.xbrl.org` instead of `mcp-cmf-chile`.

## Open questions, stated up front

- Whether Chile's `período` reporting for análisis razonado maps cleanly
  1:1 to a single `period_end_date`, or needs a range (some CMF documents
  cover year-to-date, not the discrete quarter).
- Italy's voluntary-quarterly sparsity means `quarterly`-level Italian
  rows will be a small, self-selected subset (firms that choose to keep
  disclosing quarterly) — worth flagging in the eventual methodology
  section as a selection effect, not silently treating a missing
  Q1/Q3 as "no AI disclosure that quarter."
- Whether CMF's XBRL-tagged EEFF data can shortcut universe/CIK
  resolution the way SEC's `company_tickers.json` does for the US, or
  whether `mcp-cmf-chile` already exposes an equivalent.
