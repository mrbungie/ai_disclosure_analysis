# Universe expansion plan

Decided 2026-08-31 (supersedes the three-tier "core + benchmark" sketch: the
S&P 500 layer merges INTO the core instead of sitting beside it). Companion:
[design_assessment_2026-08-31.md](design_assessment_2026-08-31.md) (why the
current 120-firm universe is academically thin: ad-hoc frame + survivorship).

## Target design

**One expanded core + one satellite. No separate benchmark tier.**

1. **Expanded core** = current 120 hand-picked firms ∪ **S&P 500 with
   membership frozen at 2021-12-31** ∪ sector top-ups. Expected size ~520–560
   firms (~100 of the current 120 are already S&P 500 members), ~3,100–3,400
   firm-years over FY2021–2026. Freezing membership at the sample start is
   what kills survivorship at the root: firms that were later acquired,
   delisted, or collapsed stay in by construction, and their 10-Ks (which
   EDGAR keeps) enter the panel for the years they filed.
2. **Sector completeness pass**: the current 12 aggregated sectors miss whole
   GICS territories (real estate/REITs, materials/chemicals, insurance,
   media/entertainment, hospitality). The S&P 500 brings all GICS sectors in;
   after ingestion, audit the aggregated-sector map (TUI's unassigned-SIC
   view), create the missing aggregated sectors, and — where a sector is
   still thin (< ~10 firms) — top up with its largest US filers by market
   cap, under a written rule.
3. **"Sizeably decent" stopping rule** (written, so the frame is a rule and
   not a vibe): every aggregated sector has ≥ ~10 firms, total ≥ ~500 firms,
   and no further ad-hoc additions after the rule is met. The universe
   definition becomes: *"S&P 500 as of 2021-12-31, plus [n] additional large
   US filers to guarantee sector coverage and AI-intensity coverage, listed
   explicitly with their inclusion rule."*
4. **Satellite of delisted / enforcement cases** (unchanged from prior
   decision): ~10–15 firms chosen by explicit criteria — SEC AI-washing
   enforcement targets that were public issuers (e.g. Presto Automation,
   Kubient) plus prominent AI-hype delistings 2022–2025. Processed by the
   same pipeline, analyzed as a **separate case chapter**, never pooled with
   the core panel. Its role: external validation that the "promotional"
   archetype matches what regulators actually prosecuted.

## Why merging S&P 500 into the core (vs. keeping it as a benchmark)

- The harnesses are deterministic: measuring 550 firms costs downloads +
  compute, not labeling. There is no marginal-measurement argument for a
  smaller core.
- Nothing is labeled yet — the fit cycles haven't run — so expanding NOW has
  zero sunk cost: the 07/10 samples get drawn from the expanded corpus and
  the harnesses are fitted to the population they will measure. (Expanding
  *after* fitting would have required a transfer-validation batch; expanding
  before makes that batch unnecessary.)
- One panel is simpler to defend than two overlapping ones; the two-instrument
  rule stays reserved for genuinely different objects (10-K vs 10-Q, core vs
  satellite).

## Execution phases

**Phase A — Universe definition (no network)**
1. Move the universe out of `pipeline.tickers` into a versioned table
   (`configs/universe.csv`: ticker, cik, company, inclusion_rule ∈
   {core_manual, sp500_2021, sector_topup, satellite_delisted}, active_status).
   Config keeps a pointer; the TUI and script 00 read the table. Rationale:
   a 550-row list with per-row provenance doesn't belong in a JSON config,
   and the inclusion_rule column IS the sampling-frame documentation.
2. Obtain S&P 500 constituents as of 2021-12-31 from a public, citable,
   reconstructible source (index changes are public record); store the raw
   list + source note under `data/raw/reference/`.
3. Resolve every member to a CIK. **Script 00 change**: current resolution
   uses the SEC's `company_tickers.json`, which only lists *current*
   registrants — delisted members and the satellite must resolve by CIK
   directly (EDGAR full-text company search / historical ticker-CIK maps).
   Firms that left the index but still file resolve normally; genuinely
   deregistered ones enter by CIK with `active_status=delisted`.

**Phase B — Collection at scale (network, resumable)**
4. Manifest + download + extract for the ~430 new firms: roughly 2,400–2,900
   additional 10-Ks. EDGAR rate limits make this hours-to-a-day of wall
   clock; everything is already incremental, so it can run in background
   batches. Disk: tens of GB of HTML.
5. `make collect-market` re-run: per-ticker price files for the new names via
   the existing fallback chain (Yahoo → Stooq → Tiingo for the dead ones);
   `sector_groups` audit in the TUI for the new SICs.

**Phase C — Measurement (the cycles, unchanged discipline)**
6. Re-run 05–06 (seed keywords) on the expanded corpus, then draw the 07
   sample from it — stratum weights now reflect the true (much less
   AI-dense) population, which is exactly what the reweighting machinery was
   built for. Fit cycle 1, freeze, re-run 05–06, extract atoms (09), fit
   cycle 2 on a chunk sample of the expanded corpus. Labeling budget is
   unchanged (samples are sized by n, not by universe size).
7. Satellite firms flow through the frozen harnesses at the end; their
   chapter never touches the fit samples.

**Phase D — Documentation**
8. Rewrite the README sample section and the methodology doc's sampling
   frame with the inclusion rule; limitations section keeps only what
   remains true (large-cap skew stays; survivorship largely resolved for the
   index layer, documented for the pre-2021 period).

## Costs and risks, stated up front

- **Calendar**: Phase B is the long pole (EDGAR volume). Everything else is
  code already written or small changes (00 by-CIK, universe table, TUI read
  path).
- **Section-parser coverage**: 4× more filers means more filing-format
  variety; expect a parser-robustness pass on 04's failure list before
  trusting the corpus (the failure modes are visible in the manifest's
  parse_status).
- **AI-mention sparsity**: many S&P 500 firms barely discuss AI in 2021–2022.
  That is signal, not noise (adoption breadth is part of the story), but it
  shifts chunk-volume expectations and the prefilter stratum balance —
  the weights handle it; the reports will show it.
- **Judge cost**: unchanged per batch; possibly one extra fit iteration per
  harness because the population got harder. Budget 2 batches per cycle as
  before.
