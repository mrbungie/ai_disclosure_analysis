# Section audit — is the 3-item corpus (Item 1 + 1A + 7) enough?

Verification script, not pipeline. Runs on demand, no LLM calls, cheap to
rerun. Code: `section_audit.py`. Run: `uv run python
verif_scripts/section_audit/section_audit.py`.

Answers two questions raised about `scripts/04_extract_sections.py`'s fixed
scope (Item 1 Business, Item 1A Risk Factors, Item 7 MD&A only):

1. Of the filings where that script found no text for one of its 3 items,
   is the gap a real extraction bug or a structurally different/absent
   filing?
2. Across the *full* 10-K text, what fraction of AI/ML keyword mentions
   falls outside Items 1/1A/7 — and specifically, how much lands in Item 1C
   (Cybersecurity) and Item 8 (Financial Statements)?

Latest run: `20260901T172410`, 2748 filings with at least one item
extracted (of 2889 total — 141 have none, see below), 0 errors.

## 0. The extractor itself changed while investigating this

The first pass of this audit ran against `04_extract_sections.py`'s
original PATTERNS approach: per-item regex requiring the section title on
the same line as "Item N", with a hardcoded `> 1000` character length
threshold to tell a real section from a table-of-contents fragment. That
threshold had no comment, no data behind it, and no structural
justification (confirmed via `git log` — it was never explained). It
turned out to work by accident (real sections run 50K-100K chars, TOC
fragments run 50-750 — any cutoff in between "works"), but it was an
unjustifiable proxy for a question ("is this a TOC row or a real heading")
that has a real, structural answer. `scripts/section_segmenter.py` now
answers it directly and is shared by both `04_extract_sections.py` and
this audit script — no more duplicated regex, no length guess.

**The structural rule, and what it took to get right** (three real bugs
found and fixed along the way, each caught by rerunning the full 2889-filing
extraction and comparing aggregate counts before/after — a silent
per-filing "close enough" check would have missed all three):

1. **A TOC row is a markdown link; a real heading isn't** (`is_link_row`).
   First cut. Undercounts: inapplicable items ("Item 1B. Unresolved Staff
   Comments ... N/A") often have no page link in the TOC, so this alone
   treats them as real headings.
2. Tried a density heuristic next ("a candidate packed within N lines of
   several others is TOC noise") to catch case 1's gap. **This broke
   things worse** — real headings almost always sit *immediately* after
   the TOC block ends, so a proximity window swept up genuine headings
   (AAPL's real Item 1 heading, 30 lines after the last TOC entry, got
   discarded — Item 1 coverage dropped from 2672 to 1264 filings, a 53%
   regression). Caught only because the full-corpus rerun was compared
   against the prior run's aggregate counts, not spot-checked on a few
   tickers.
3. **The actual fix**: a TOC is the *maximal prefix* of item-heading
   candidates, starting from the very first one, that increases strictly
   with no repeats and no big gaps (`_toc_prefix_end` in
   `section_segmenter.py`, gated by `MIN_TOC_ITEMS=5` and
   `MAX_TOC_GAP=150` so a handful of early real headings — e.g. Honeywell,
   whose TOC doesn't spell "Item 1" at all — don't get misread as a TOC
   themselves). The first candidate that repeats an already-seen item
   number is, by construction, the real heading for that item appearing
   again after the TOC listed it once.
4. Separately, the item-number regex only handled "Item 1A" (suffixed,
   no punctuation). Two more real formats surfaced from actual filings and
   were added: "Item 1 (B)" (parenthesized letter — ~30% of a 300-filing
   sample contained this format somewhere) and "Item 1.C." (period before
   the letter — found via a CLX filing where an entire Item 1C
   Cybersecurity section was silently absorbed into "Item 1" because the
   old regex read "1.C." as just "1"). Both are now parsed into the
   correct `1B`/`1C` item key instead of colliding with plain `1`.

Net effect vs. the very first (pre-fix) run: Item 1 coverage moved
2730 → 2476, Item 1A 2862 → 2740, Item 7 2870 → 2727. Lower on Item 1
specifically — not a regression, a correction: some of the original 2730
were CLX-style filings where "Item 1"'s extracted text was actually Item
1C content wrongly merged in. Real coverage of genuinely-real headings
is what the numbers above reflect.

**Residual gap, not chased further**: 141 filings (of 2889) now come up
with zero items at all — up from 0 under the old (looser but wrong)
matching. Spot-checked: this is the same class of failure as the COP case
below (iXBRL/Workiva-templated filings where "Item 1" isn't matchable
text after HTML normalization at all, in any format), disproportionately
National Commercial Banks, Semiconductors, and REITs. This needs a
different HTML→text pipeline for these filers, not a better regex — see
"Root cause" below.

## 1. Missing-section audit

Method: the shared general segmenter (any `Item N[A-C]` heading, not just
the 3 hardcoded ones) is now identical between the pipeline and this
audit, so a filing missing an item here is genuinely missing it in the
pipeline output too — there's no more "found by the general segmenter but
not by PATTERNS" bucket (that WAS the CEG-style bug; now fixed upstream).

| item | not found (of 2748 completed filings) |
|---|---|
| Item 1 | 272 |
| Item 1A | 8 |
| Item 7 | 21 |

Plus the 141 filings with zero items at all (see above) — call it ~413
filings missing Item 1 in total across the full 2889-filing corpus.

**Still clusters by sector.** Item 1 `not_found`, top industries: Crude
Petroleum & Natural Gas (32), Pharmaceutical Preparations (27), Petroleum
Refining (17), Semiconductors (14), Computer Programming/Data Processing
(12), Electric Services (12), Prepackaged Software (11).

**Root cause, confirmed on a concrete example (COP, `0001163165-26-000009`,
2026 10-K):** this is a Workiva-generated inline-XBRL filing where neither
`markdownify`'s HTML→Markdown pass nor a raw `BeautifulSoup.get_text()`
extraction contains the literal string "Item 1" or "BUSINESS" as
contiguous, matchable text anywhere in the 573KB of extracted text (verified
directly — zero regex matches for `\bitem\s*1\b` case-insensitive across the
whole document). Large-cap energy/financial/pharma filers disproportionately
use this filing-agent template; the heading text itself doesn't survive HTML
normalization in a form any regex can see. This is a genuine limitation of
a markdownify+regex extractor — it needs per-template handling or a
different HTML→text pipeline for these filers, not a looser pattern.

## 2. AI-mention leakage audit

Same `configs/config.json: seed_screen.ai_keywords` / `false_positives`
construct as the detection seed screen, applied to every segmented item
across the full filing (not just the 3-item corpus).

| | mentions | % of total |
|---|---|---|
| Total AI mentions, full filing | 33,422 | 100% |
| **In core (Item 1 + 1A + 7)** | **30,710** | **91.89%** |
| Outside core | 2,712 | 8.11% |

Top items outside the core, by mention count:

| item | mentions | % of total |
|---|---|---|
| Item 8 (Financial Statements) | 1,298 | 3.88% |
| Item 16 (Form 10-K Summary) | 411 | 1.23% |
| **Item 1C (Cybersecurity)** | **368** | **1.10%** |
| Item 15 (Exhibits) | 126 | 0.38% |
| everything else (11, 6, 4, 5, 7A, 10, 3, 2, 12, 1B, 9A, 9B, 4A, ...) | 509 | 1.52% |

777 filings have at least one AI mention outside the core 3 items.
Item 1C specifically: 368 mentions. Item 8 specifically: 1,298 mentions.

Core-mention share rose from an earlier (pre-fix) estimate of 88.2% to
91.9% — expected, since Item 1C content that used to be silently absorbed
into "Item 1"'s count (the CLX-style bug) is now correctly attributed to
Item 1C instead of double-counted as core.

(Items "30" and "40" in the raw output, 55 and 3 mentions respectively,
are segmenter noise — likely exhibit-index or cross-reference text that
happens to match `Item\s*NN` outside a real heading context. Negligible
volume; not worth chasing further.)

## Recommendation

**Keep Item 1 + 1A + 7 as the core textual corpus** — 91.9% of all AI
mentions live there, and extending to "the whole 10-K" would dilute
comparability with financial/legal boilerplate for an ~8% recall gain
spread thin across 20+ items.

Two items are worth follow-up, not corpus expansion yet:

- **Item 1C (Cybersecurity)**: newer (2023+ SEC rule), 1.1% of mentions
  but concentrated — plausibly AI-adjacent threat/governance language
  systematically excluded from the current corpus for filings in that
  period. Worth a coverage check split by filing year before deciding
  whether to add it.
- **Item 8 (Financial Statements)**: the single largest "leak," 3.9% of
  mentions. Per the original framing (AI-washing vs. credible disclosure),
  this is better used as a **corroboration signal** (capex, software
  costs, acquisitions, intangibles tied to AI mentions) than mixed into
  the primary embedding/text score — its language and structure are too
  different from Items 1/1A/7 to score with the same candidate.

Separately, worth deciding: is the ~413-filing iXBRL-invisibility gap on
Item 1 (concentrated in energy/pharma/banks/semis) worth a dedicated
HTML→text path, or accepted as a known non-random gap in this corpus? Not
decided here — this script surfaces it, doesn't resolve it.

## Outputs

Gitignored under `data/interim/audits/section_audit/` (no LLM involved,
cheap to regenerate — doesn't need the archive-on-delete policy in
`CLAUDE.md`):

- `missing_section_audit.parquet` — one row per (filing, missing target
  item), with classification and segment length.
- `ai_leakage_by_item.parquet` — one row per (filing, item-with-any-AI-hit),
  with hit count and segment length.
- `summary.json` — the aggregate numbers above.
