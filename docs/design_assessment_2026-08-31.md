# Design assessment — 2026-08-31

Pre-implementation review of the pipeline design, answering five questions posed
before the meta-harness restructuring was green-lit: fundamental problems, will
it find something, is it interesting enough, is it expressive enough for
behaviours, and is data missing. Companion documents:
[meta_harness_plan.md](meta_harness_plan.md) (what is being built),
[meta_harness_map.html](meta_harness_map.html) (system diagram).

## Fundamental problems with the design?

No fatal ones — as a **measurement** pipeline it is sound and unusually
well-disciplined (dev/holdout separation, single-look holdout, full search
traces). Two structural weaknesses, flagged now rather than at the defense:

1. **Four binary dimensions are too coarse for "behaviours".** The research
   question is about disclosure *behaviours and archetypes*. If each firm-year
   reduces to four shares (% substantive / promotional / risk / governance),
   clustering will yield 3–4 archetypes a skeptical committee can call
   restatements of the inputs ("the promotional cluster is firms high on
   promotional"). The thesis proposal itself names dimensions the four don't
   cover: **use-case specificity** and **quantification**. The old pipeline had
   nine. **Decision: cycle 2 fits six dimensions** (substantive, promotional,
   risk, governance, use-case specificity, quantification). The keyword-harness
   method carries this fine — the expressiveness ceiling is the schema, not the
   method.

2. **The causal extensions are the weak flank.** The SEC-2024 DiD defines
   treatment ("vague pre-2024 disclosers") from the same variable family as the
   outcome — an open invitation to a mean-reversion critique — and 115 firms ×
   annual filings gives ~3 pre / ~2 post periods, with power only for large
   effects. The DeepSeek shock (Jan 2025) is worse: one annual filing boundary.
   **Frame these as event-study / descriptive-shift evidence with the power
   check reported honestly, not as causal identification.** Framed that way
   they are defensible; framed as DiD proper they will get shredded.

## Will something be found?

Descriptively, almost certainly. AI-mention volume and risk-factor AI language
exploded 2023–2024 — the pipeline will replicate that (good: it validates the
measurement), and then add the layer that is actually novel here: the archetype
structure and **firms migrating between archetypes over time** (e.g.
promotional → substantive after SEC scrutiny). That migration story is the
"behaviour" finding the committee is asking for, and it is near-guaranteed to
show *something*; the open question is only whether the pattern is clean.

## Interesting enough?

As pure text description: passable but not memorable. The cheap upgrade is
**linking archetypes to one or two firm-level outcomes** — even free ones
(market cap, R&D intensity, sector returns from public sources).
"Promotional-archetype firms have X" is a finding; "there exist
promotional-archetype firms" is a taxonomy. One credible outcome linkage moves
the thesis a tier up; CRSP/Compustat-grade panels are not required.

## Expressive enough for behaviours?

Yes, **if** the dimension schema is widened to six (see above) and the
per-chunk → firm-year aggregation stays rich: dimension shares, but also
within-filing placement (Risk Factors vs Business vs MD&A — section labels
already exist, and *where* a firm talks about AI is itself a behaviour,
essentially free to include).

## Missing data?

Three candidates, descending value-per-effort:

- **(a) Firm outcome covariates** — highest value (see "interesting enough").
- **(b) 10-Qs** — the proposal promised them; they would quadruple time
  resolution around the 2024/2025 events. Significant scope cost; only worth it
  if the DiD-ish questions stay central.
- **(c) Earnings calls** — best signal for promotional language, but a scope
  blowout; mention as future work.

Also: be ready to defend how the 120-firm universe was selected — if
hand-picked, that is a sampling caveat to document.

## Meta-point

The meta-harness framing is a strength for the methodology chapter — it
quantifies measurement error at every link of the chain (human → judge kappa →
harness F1 → corpus measurement) — but it is garnish. The committee grades the
economics; don't let harness engineering crowd out the findings.

## Prior code-review findings folded into the plan

From the review of scripts 07–08 that preceded this assessment:

1. The 50/50 candidate/excluded stratum split **inflates recall and F1**
   relative to the population — false negatives live in the (undersampled)
   excluded stratum. Fix: record per-stratum sampling fractions and report
   inverse-probability-weighted metrics alongside raw ones.
2. The "k-fold CV" in 08 is **not cross-validation** — nothing is trained per
   fold, and candidates are mined from all of dev including every fold's test
   rows. It is a fold-stability check; the holdout is the only honest number.
   Rename and reframe it so a reviewer doesn't flag "CV" as leaky.
3. 08 **writes the fitted keywords to config unconditionally**, even when the
   fitted list underperforms the baseline on holdout. Gate the write.
4. The search machinery has never executed a real round (baseline already met
   target). Add a synthetic self-check (drop a known keyword, confirm
   recovery) that never consumes the real holdout.
5. The LLM judge is a single judge with no human agreement check, and
   paragraphs were truncated to 2,500 chars before judging. Remove the
   truncation; add a human agreement sample (~60 paragraphs, Cohen's kappa).
