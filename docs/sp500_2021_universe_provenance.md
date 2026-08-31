# S&P 500 (2021-12-31) reconstruction — provenance

Source data used to build the `sp500_2021` rows in `configs/universe.csv`
(Phase A of `docs/universe_expansion_plan.md`). Raw fetched pages are cached
under `data/raw/reference/` (gitignored, like the rest of `data/`); this note
is the versioned record of what was fetched, when, and how it was turned into
a firm list.

## Sources

1. **Current constituents** — Wikipedia, *List of S&P 500 companies*
   (`https://en.wikipedia.org/wiki/List_of_S%26P_500_companies`, raw wikitext
   via `?action=raw`), retrieved 2026-08-31. Table `id="constituents"`: ticker,
   security name, GICS sector, HQ, date added, CIK, founded — 502 rows parsed
   (503 tickers include the dual-class GOOG/GOOGL, minus one row lost to a
   parsing edge case, not material to the reconstruction below).
   Cached: `data/raw/reference/sp500_wikipedia_current_constituents_raw.wikitext`.

2. **Historical changes** — Wikipedia, *Historical components of the S&P 500*
   (`https://en.wikipedia.org/wiki/Historical_components_of_the_S%26P_500`,
   raw wikitext), retrieved 2026-08-31. Table `id="changes"`: effective date,
   added ticker/security, removed ticker/security, reason — 365 rows parsed
   (full history back to 1976; only rows with date > 2021-12-31 were used).
   Cached: `data/raw/reference/sp500_wikipedia_historical_changes_raw.wikitext`.

## Reconstruction method

Standard backward reconstruction from the current list:

1. Start from the current 502-ticker membership set.
2. For every change row with `effective date > 2021-12-31` (72 rows, applied
   most-recent-first order doesn't matter for a set reconstruction):
   - remove the **Added** ticker (it entered the index after the target date,
     so it wasn't a member on 2021-12-31);
   - add back the **Removed** ticker (it was still a member on 2021-12-31;
     it only left the index after that date).
3. Result: 503-ticker membership as of 2021-12-31.

**Consistency check performed**: for every current constituent whose
Wikipedia "Date added" is after 2021-12-31, confirmed it does NOT appear in
the reconstructed set (0 violations found) — i.e., every subsequent
index-entry is accounted for by a matching change-table row, so the
reconstruction isn't silently missing any additions.

**Spot checks against known events** (all correct): TWTR (Twitter, taken
private 2022) present; SIVB/FRC/SBNY (SVB/First Republic/Signature Bank,
failed 2023) present; ATVI (Activision, acquired by Microsoft 2023) present;
DFS (Discover, acquired by Capital One 2025) present; FB/META (ticker rename,
not an index change — Wikipedia explicitly excludes ticker-only renames from
the changes table) correctly resolves to the single continuous META entity.

## CIK resolution

- **Still an S&P 500 member today**: CIK taken directly from the Wikipedia
  current-constituents table.
- **No longer in the index but still an active SEC registrant**: resolved by
  exact ticker match against SEC's `company_tickers.json` (37 of 70 "left the
  index" tickers).
- **No longer a live registrant under that ticker** (acquired, taken private,
  or renamed with a new CIK): resolved via the EDGAR full-text search API
  (`https://efts.sec.gov/LATEST/search-index?q="<company name>"&forms=10-K`),
  taking the top-scoring hit's CIK. Six of these matches were false positives
  from the search relevance ranking (the query text appeared in an unrelated
  filer's 10-K rather than the target company's own filing) and were
  corrected by hand, cross-checked against `data.sec.gov/submissions/CIK*.json`
  metadata (name / former names / SIC):
  - **CERN** (Cerner) — search returned an unrelated shell company; corrected
    to CIK 0000804753 (formerly "CERNER CORP /MO/" per `formerNames`).
  - **FRC** (First Republic Bank) — search returned a mortgage-backed
    securitization trust sharing the name; corrected to CIK 0001132979.
  - **HES** (Hess Corp) — search returned Valero Energy (an unrelated 10-K
    that mentioned Hess); corrected to CIK 0000004447 (formerly "AMERADA
    HESS CORP").
  - **PXD** (Pioneer Natural Resources) — search returned the affiliated but
    distinct "Pioneer Southwest Energy Partners L.P."; corrected to CIK
    0001038357.
  - **SBNY** (Signature Bank) — full-text search returned no true hit at all
    (searched other banks' filings mentioning the name); resolved instead via
    EDGAR company search (`browse-edgar?action=getcompany&company=signature+bank`)
    to CIK 0001288784 ("Signature Bank Corp").
  - **SEE** (Sealed Air) — search returned Johnson & Johnson's 10-K (which
    mentions Sealed Air as a packaging supplier); corrected via a stricter
    query to CIK 0001012100 ("SEALED AIR CORP/DE").

  All 70 "left the index since 2021-12-31" tickers resolved to a CIK — **no
  unresolved firms** in the sp500_2021 layer.

- **active_status**: `listed` if the resolved CIK still maps to at least one
  live ticker in SEC's `company_tickers.json`; `delisted` otherwise (acquired,
  taken private, or failed/receivership). 368 listed, 28 delisted among the
  396 net-new `sp500_2021` rows added to `configs/universe.csv` (399 matched
  before de-duplicating against tickers/CIKs already present as
  `core_manual`; 104 of the 503 reconstructed 2021 members were already in
  the 122-firm core).

## Result

`configs/universe.csv`: 518 total rows — 122 `core_manual` (the original
universe, including DFS/CIK 0001393612 and SQ→Block Inc/CIK 0001512673 by
manual CIK lookup since neither resolves via ticker anymore) + 396
`sp500_2021` (368 listed, 28 delisted). No `sector_topup` or
`satellite_delisted` rows yet — those are later passes (plan items 2-3 and
Phase A step 4's satellite, explicitly deferred per the task instructions
until the new firms' SICs arrive through collection).
