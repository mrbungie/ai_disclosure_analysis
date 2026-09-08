# US narrative filings — corpus coverage

Ticker-level coverage of the 517-firm US universe (`firm_universe`,
`country_code='us'`) against `filing_manifest` / `filing_manifest_10q`,
measured 2026-09-08. This is coverage of *which firms have at least one
filing of each type in the corpus* — a different question from the
embedding coverage in `docs/sources/accounting_data.md`'s sibling
document or from within-filing extraction completeness
(`docs/analytics/00_funnel_del_corpus.md`).

Six tickers (ASML, HMC, TM, TSM, UL, and TEAM which self-elects to also
file domestically) are foreign private issuers that report via 20-F
instead of 10-K/DEF 14A, and via 6-K instead of 8-K (6-K isn't fetched —
out of scope until asked for). They're excluded from the domestic-form
denominators below rather than counted as missing.

| form | coverage | denominator | missing |
|---|---|---|---|
| 10-K | **99.4%** (508/511) | domestic-filer universe | FRC, HONA, SBNY |
| 10-Q | **99.4%** (508/511) | domestic-filer universe | FDXF, FRC, SBNY |
| 8-K | **99.6%** (509/511) | domestic-filer universe | FRC, SBNY |
| DEF 14A | **99.0%** (506/511) | domestic-filer universe | FDXF, FRC, HONA, PSKY, SBNY |
| 20-F | **100%** (6/6) | known foreign-private-issuer tickers | — |
| Earnings call transcript | **96.1%** (497/517) | full universe | see below, 20 tickers |

Every gap above is explained, not just observed:

- **FRC** (First Republic Bank) and **SBNY** (Signature Bank) — seized in
  the 2023 banking crisis, stopped filing everything afterward. Not a
  fetch bug (already documented in `accounting_data.md`).
- **FDXF** (FedEx Freight) and **HONA** (Honeywell Aerospace) — 2026
  spinoffs, too recently independent to have filed their first 10-K/proxy
  yet (FDXF: 1 10-K + 6 8-Ks so far; HONA: 4 8-Ks, 1 10-Q, zero 10-K).
- **PSKY** (Paramount Skydance) — recently merged/renamed entity, one
  10-K so far, proxy not yet due.

**Bug found and fixed (2026-09-08): AFL (Aflac) and AME (Ametek) had ZERO
DEF 14A in `filing_manifest`** despite 6 complete years of 10-K/8-K/
earnings-call coverage each — implausible for two large, stable S&P 500
firms that file a proxy every year. Root cause: the raw HTML files were
already downloaded and sitting in `data/raw/filings_proxy/` (AFL: all 6
years; AME: 4 of 6), but never got written into
`filing_manifest_proxy.parquet` — a manifest-registration gap, not a
fetch failure. `scripts/us/proxy/01_fetch_filings.py` is idempotent by
design (skips re-downloading anything whose gzip already exists, keyed
off the manifest rather than the filesystem for its "already done" check
— see the script's own docstring), so re-running it against the full
517-firm universe cost only 2 real network calls (AME's missing 2023/2024
filings) and zero for AFL — the rest of the universe short-circuited via
the existing-manifest check. Followed by `scripts/us/proxy/
02_extract_sections.py` (also idempotent) and a full `build_duckdb.py
--with-text-tables` rebuild. Result: AFL and AME each went from 0 to 6
DEF 14A, now present in `unique_paragraphs` (5,131 and 1,648 paragraph
rows respectively). DEF 14A coverage moved from 504/511 (98.6%) to
506/511 (99.0%).

**Earnings call transcripts (96.1%, 20 tickers missing)** — the one
category with a real, unresolved gap, and it's a third-party data
availability limit, not a registration bug:

- **BRK.B, NVR** — these two genuinely don't hold public earnings calls;
  correctly absent.
- **CDAY, CTXS, DISCA, PBCT, WLTW** — acquired/taken private/merged
  during the corpus window (2021-2026); no further calls exist to fetch.
- **HMC, TM, TSM, UL** — foreign private issuers, thin US-transcript
  coverage from these vendors.
- **ED (Con Edison), EXPD, FBHS, FLT, FOX, SQ, VMRK, FDXF, HONA** — the
  remaining, genuinely unexplained gaps. Checked both existing gap-fill
  sources (`scripts/us/earnings_calls/03_fill_gaps_equibles.py`,
  `04_fill_gaps_stockanalysis.py`) — both have already run
  (74 and 308 backfilled ticker-quarters respectively) and neither ever
  found any of these, including ED, a large stable utility that
  plausibly does hold public calls. Re-running the gap-fillers needs
  `EQUIBLES_API_KEY` (a paid third-party service) — not run without
  checking first, unlike the free public-EDGAR fix above.
