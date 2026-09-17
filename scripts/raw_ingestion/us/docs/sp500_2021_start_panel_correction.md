# S&P 500 analytical panel — anchor correction (2021-01-01)

Supersedes the anchor date used in `sp500_2021_universe_provenance.md`
(2021-12-31) for the purpose of the **analytical panel** — the firm-year set
regressions and coverage figures should filter to. The raw fetch universe
(`configs/us/universe.csv` / `firm_universe`) is intentionally kept broader
than this panel and is never trimmed to match it (see "Why two lists"
below).

## What changed

1. **Anchor moved from 2021-12-31 to 2021-01-01** (start of the sample
   period, not its end) — reconstructed independently from the same cached
   Wikipedia sources (`data/raw/reference/sp500_wikipedia_*_raw.wikitext`),
   this time parsing the *current-constituents* table's own `Date added`
   column plus the *historical changes* table, reversed most-recent-first
   down to 2021-01-01. Result: **505 tickers** were S&P 500 members at
   2021-01-01 (vs. 503 at 2021-12-31 — the one-year gap includes real
   membership turnover, e.g. SBNY joined the index in Dec 2021, after the
   2021-01-01 anchor but before 2021-12-31).

2. **27 tickers previously tagged `sp500_2021` were contaminated** — they
   were *not* 2021-01-01 members; they are successor tickers from spinoffs
   or reconstitutions that happened between 2021-01-01 and 2026 (e.g. GEHC,
   GEV, HONA, SOLV — GE/Honeywell/3M spinoffs; SBNY — joined the index Dec
   2021). Confirmed two of these successor tickers had **silently replaced**
   a real 2021 member in `universe.csv` rather than sitting alongside it:
   PSKY (Paramount Skydance, 2025) had replaced PARA (Paramount Global);
   SW (Smurfit Westrock, 2024) had replaced WRK (WestRock). Both PARA and
   WRK were real S&P 500 members for most of 2021-2024 with zero
   representation in the corpus until this fix.
   Retagged to `sp500_post2021_addition` (not deleted — see below).

3. **34 real 2021-01-01 members were missing from `universe.csv` entirely**
   (PARA/WRK plus 32 others: ALXN, BBWI, CAG, CXO, DISCK, DXC, ECHO, EMN,
   FLIR, FLS, FOXA, FTI, GOOG, GPS, HBI, HFC, KMX, KSU, LEG, MXIM, NOV,
   NWSA, PRGO, SLG, TIF, UA, UNM, VAR, VFC, VNO, VNT, WU, XRAY, XRX). Of
   these, 3 (GOOG, FOXA, NWSA) are dual-class share tickers of companies
   already covered under a sister ticker (GOOGL, FOX, NWS — same CIK, same
   filings) and needed no new fetch. The other 31 were fully backfilled:
   10-K/10-Q/8-K/DEF 14A via `edgar_fetch`, earnings-call transcripts via
   the HF dataset + `04_fill_gaps_sa.py`, XBRL facts via
   `04_extract_inline_xbrl_facts.py`. Residual legitimate absences (CXO,
   TIF acquired Jan 2021; VAR acquired Apr 2021; ALXN acquired Jul 2021 —
   all closed too early in the sample to have a 2021+ 10-K/10-Q/proxy or,
   for ALXN/CXO/TIF/VAR, even an earnings call) were individually verified
   against `data.sec.gov/submissions/CIK*.json` filing history before being
   accepted as "no filing exists," not treated as a fetch failure.

## Why two lists (raw universe vs. analytical panel)

`configs/us/universe.csv` / `firm_universe` stays at 548 US tickers (517
prior + 31 backfilled) — it is deliberately allowed to hold more than the
analytical panel needs; nothing is trimmed out of it for this fix. The new
`sp500_2021_start_panel` group in `configs/us/universe_membership.csv`
(502 tickers — 505 reconstructed members minus the 3 dual-class collapses)
is the explicit flag scripts should filter to for the "was in the S&P 500
at the start of the sample" analytical panel, instead of assuming
`firm_universe` itself is that panel.

## Result

`configs/us/universe_membership.csv`: `sp500_2021_start_panel` group added,
502 tickers, reconstructed independently of the pre-existing `sp500_2021`
tag (which mixed a 2021-12-31 anchor with the contamination described
above and is left as-is for its own purposes, not deleted).
