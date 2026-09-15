# Posture archetypes: stability and point-in-time plan

Status: in progress (2026-09-15). Covers what was established about the
expanding posture archetypes used as predictors (crash risk, call beta), the
decisions taken, the point-in-time (PIT) design being implemented, and what is
still open.

## 1. What was observed

- After the parquet-layer migration, the crash-risk Governance-Led result moved:
  NCSKEW `dum_gov` β −0.106 (p 0.021) → −0.063 (p 0.23); DUVOL β −0.130
  (p 0.003) → −0.091 (p 0.058). Call-beta results did not move (max |Δβ| 0.001).
- The thesis prose (β −0.124, p 0.046, N 8,642, R² 0.113) already came from an
  earlier specification: the pre-migration code run on 2026-09-15 gave N 7,955,
  R² 0.071, β −0.106.

## 2. What caused it (verified)

1. **Only the archetype input changed.** Swapping each of the four regression
   inputs (call panel, crash panel, fundamentals, expanding archetypes) between
   old and new versions in all 16 combinations: only the archetype file moves
   β/p. Same code, same sample (7,955 calls).
2. **Frames population.** Silver takes the newest prefilter deployment
   deterministically (the old DuckDB view picked arbitrarily among deployments
   of the same model_version and was not even run-to-run deterministic).
   Frames grew ~0.6% per year.
3. **The 2023 yearly cutoff sits on a near-tie.** With k=3 and 161 training
   firms, two solutions differ by ~2% RSS: {Vocal, Defensive(+governance),
   hedging-dominated third vertex} (best, 710.5) vs {Vocal, Defensive,
   Governance} (724.4). The old data fell on the governance side, the new data
   on the hedging side. The naming rule (Defensive = max risk; Gov = max
   governance among the rest) labels the third vertex "Governance-Led" even
   when it has no governance (gov 0.03), so firms like META/CRM entered the
   group (15% of Governance-Led calls turned over).
4. **2021–2022 were already weak before the migration.** Old models: 2021 Gov
   vertex gov 0.24 (weak but real), 2022 gov 0.08 with hedging 0.22 (not
   governance). Only 2023 flipped.
5. **Why early cutoffs are fragile.** Few firms discuss AI governance before
   2024 (firms with gov > 0.2: 6 / 11 / 20 in 2021–2023 vs 47–134 later);
   a handful of extreme firms (z > 5: AMZN, KMX, UNH) define vertices; the static
   (full-sample) vertices fit early data 2–3× worse than a cutoff's own fit and
   converge with it from 2024.
6. **Not caused by:** BLAS threading (1/4/8 threads identical), the crash-risk
   panel reconstruction (same β with old or reconstructed panel), or code
   translation.

## 3. Bugs found in the archetype construction

| # | Problem | Status |
|---|---|---|
| B1 | Single uniform AA start (`n_init=1`): solution depends on the local optimum reached | Fixed: `posture_features.fit_aa` (FurthestSum, 25 restarts) for every AA fit; static archetypes unchanged (0/498 firms) |
| B2 | Train/projection mismatch: expanding fit trained on cumulative firm rates (shrinkage and scaling within the cutoff) but labels assigned by projecting gold firm-year features (different shrinkage/scale) | Replaced by the PIT builder (§5): same construction for training and scoring |
| B3 | Posture frames silently excluded earnings calls (38,728 frames, no `accession_number` in the manifest) and 10-Q (4,742, separate manifest): 59% of AI frames never entered posture, while `disclosure_intensity` did include calls and 10-Q | Decided: posture keeps 10-K/8-K/DEF 14A (`POSTURE_FORMS`); adding calls and 10-Q removes the governance vertex (§5 diagnostics). `load_frames` now carries `available_date` |
| B4 | Look-ahead: crash regressions merge the yearly archetype of the SAME calendar year as the call (`merge on ticker, year`); a March call gets a label built from filings up to December (≤ 11 months) | Replaced by as-of join on quarterly snapshots (§5); full audit running (§7) |
| B5 | Correlation-naming with "Undefined" vertices (commit "Stable Archetypal Analysis…") | To revert: rejected; profile naming kept |

## 4. Options evaluated (single runs unless noted)

| Variant | Gov vertex identified | Fit cost | Crash `dum_gov` NCSKEW / DUVOL | Verdict |
|---|---|---|---|---|
| k=3, uniform, 1 start (old) | depends on seed/data | — | −0.063 (0.23) / −0.091 (0.058) | unstable |
| k=4, uniform, 10 seeds | 86–94% label agreement | — | always negative, 8/10 and 9/10 p<.05 | unstable labels |
| k=3/k=4, FurthestSum 25 restarts, 3 seeds | 100% agreement | — | k3 −0.099 (0.028) / −0.126 (0.005); k4 −0.128 (0.012) / −0.141 (0.007) | stable, but k3 early third vertex is hedging |
| Cumulative train + cumulative projection, MIN_FRAMES 5/10/20 | weak 2021–23 | — | −0.043 (0.44) … −0.105 (0.056) | sensitive to threshold |
| Firm-year train + projection, MIN_FRAMES 3/5/10 | min 5: gov 0.42/0.53/0.64 from 2022 | — | 3: −0.096 (0.081); **5: −0.158 (0.006) / −0.185 (0.002)**; 10: −0.075 (0.16) | best identification, threshold-sensitive |
| Firm-year + warm start (λ=0) | same as above from 2022 | +0.0% vs best free fit every cutoff | −0.157 (0.006) / −0.183 (0.002) | chosen direction |
| Warm start + concept anchors (λc 0.5) + continuity (λt 0.1) | gov compressed 0.34–0.55 | +2% to +9% | −0.036 (0.55) / −0.060 (0.34) | over-regularized, rejected |
| 2021 init from extreme firm / anchor directions / static vertices | identical solutions | 0% | identical | 2021 solution is unique |
| Global PCA naming (correlation with PCs) | PC1 42.9% substance-vs-defensive, PC2 18.1% governance, PC3 11.7% hedging | — | — | used as diagnostic/justification only |
| Hungarian matching to static vertices | — | — | — | rejected: uses full-sample structure (look-ahead) |

Notes:
- Governance and Defensive vertices share risk because AI governance text is
  about risk: P(risk | gov frame) = 0.47 in every year; the static archetypes
  show the same overlap (Gov risk 0.43, Def 0.80). At firm-year level gov and
  risk correlate 0.09, and the vertices separate on governance (0.62 vs 0.03)
  and realized tone (0.83 vs 0.37).
- The early Governance-Led vertex (2021–2022) is mixed (governance plus high
  intensity/promotion); it is the data's structure (unique solution), not an
  initialization artifact.

## 5. Point-in-time design (being implemented)

Convention (`scripts/common/pit.py`):
- Every event row carries `available_date` (filing date, call date, trading date).
- Accumulating constructs are materialized as snapshots on one calendar:
  **quarter starts (1 Jan / 1 Apr / 1 Jul / 1 Oct)**. A snapshot dated
  `as_of_date` uses only rows with `available_date < as_of_date`, always the
  latest available.
- Events join snapshots with `pit.asof_join` (backward, by ticker): an event on
  date t gets the latest snapshot with `as_of_date <= t`.
- Predictive regressions use only as-of joins; calendar-year firm panels stay
  descriptive.

Posture archetypes (`scripts/gold/posture/build_posture_asof.py`):
- Posture at q = seven posture rates over frames of 10-K, 8-K and DEF 14A
  (`posture_features.POSTURE_FORMS`) published in the 12 months before q, plus
  intensity (frames per 1,000 words over the same documents).
- Training rows = the 1 January snapshots ≤ q (calendar years, no overlap)
  with ≥ MIN_FRAMES (5) frames: archetypes built on year c exist from
  1 January c+1; April/July/October snapshots reuse them and only refresh each
  firm's 12-month posture. Shrinkage prior, standardization and intensity
  scale fitted on training rows.
- AA k=3: first snapshot FurthestSum 25 restarts; later snapshots warm start
  from the previous snapshot's archetypes (`scripts/gold/posture/warm_start_aa.py`).
- Vertex names by profile (Defensive = max risk; Governance-Led = max governance
  among the rest; Vocal = remaining) — uses only the snapshot's own vertices.
- Every universe firm scored at q with the same construction; no frames in the
  window → "No AI".
- Output `data/gold/predictions/firm_quarter/posture_archetype_asof.parquet`
  (ticker, as_of_date, n_frames_window, w_voc, w_gov, w_def, archetype) and
  `models/posture_archetype_asof/as_of=YYYY-MM-DD/model.pkl`.

### First runs of the snapshot builder (diagnostic, scratchpad only)

| Run | Governance vertex (gov share) |
|---|---|
| All channels, stacked quarterly training, warm start | 0.01–0.04 at every snapshot: no governance vertex |
| All channels, stacked quarterly training, no warm start | 0.01–0.03 (2021-07 … 2024-01): warm start is not the cause |
| No calls, stacked quarterly training, warm start | 0.17–0.19 (2021–2022), 0.01–0.05 (2023-07 … 2025-07), 0.58–0.65 from 2026-01 |

Governance vertex at 1 January snapshots (2022 … 2026), warm start:

| Channels | Training rows | Gov vertex |
|---|---|---|
| 10-K, 8-K, DEF 14A | 1 Jan snapshots (calendar years) | 0.24 / 0.41 / 0.53 / 0.64 / 0.54 (= validated firm-year design) |
| 10-K, 8-K, DEF 14A | every quarter (stacked) | 0.21 / 0.37 / 0.53 / 0.66 / 0.57 |
| + 10-Q | 1 Jan snapshots | 0.18 / 0.13 / 0.01 / 0.04 / 0.55 |
| + 10-Q + calls | 1 Jan snapshots | 0.02 / 0.01 / 0.02 / 0.02 / 0.03 |

Cause: channel mix. Posture rates by channel (gov / risk / promotional): DEF 14A
0.375 / 0.26 / 0.28; 10-K 0.058 / 0.56 / 0.10; 10-Q 0.033 / 0.24 / 0.18;
earnings calls 0.024 / 0.056 / 0.36. Calls are about half of all frames, so a
frame-pooled rate measures a firm's channel mix rather than its posture.
Overlapping training windows are not the cause. Recommended (pending user
decision): posture from 10-K, 8-K, DEF 14A; calendar-year training rows.

## 6. Next steps

1. Finish the first run of `build_posture_asof.py`; inspect vertex profiles per
   snapshot (governance/risk/specificity) and archetype counts.
2. Crash-risk and archetype battery with `pit.asof_join` on the call date;
   compare with the same-year (look-ahead) merge.
3. Sensitivity: MIN_FRAMES 3/10, window 12 months vs expanding; one run each,
   bootstrap only after the specification is fixed (parallelized: 8 processes).
4. Revert the "Undefined" correlation naming (B5); replace
   `build_call_crash_and_archetypes.py` and `build_archetype_weights_quarterly_asof.py`
   with the snapshot builder; repoint crash/battery/config-decoupling analytics.
5. Static fit (Chapter 3) with all five channels (B3): rerun and diff archetypes;
   descriptive only, never used as a predictor.
6. Apply the PIT audit fixes (§7).
7. Thesis: rewrite the crash-risk finding and methodology (PIT snapshots,
   channels, early-period identification caveat); update docs/tasks/thesis_prose_edits.md.

## 7. PIT audit

A read-only audit of scripts/gold and scripts/analytics is running (report:
scratchpad `pit_audit.md`, to be copied here). Scope: same-year merges,
period-end vs filing-date alignment, full-sample ranks/standardization/priors
used as predictors (e.g. washing score percentile ranks), expanding/shift
constructs, market window overlaps, merge_asof settings, survivorship.

Out of scope: difference-in-differences analyses (channel gap DiD, shock DiD)
are no longer used in the thesis; candidates for scripts/deprecated.

## 8. Crash risk with point-in-time archetypes (first run, 2026-09-15)

Posture from 10-K/8-K/DEF 14A, calendar-year training, gold
`posture_archetype_asof` joined with `pit.asof_join` on the call date
(scratch: pit_diag/crash_asof.py; results CSVs not yet rewritten).

| Label | N | Gov calls | NCSKEW dum_gov | DUVOL dum_gov | DUVOL dum_voc |
|---|---|---|---|---|---|
| Same-year expanding (look-ahead) | 8,140 | 660 (2024–26 only) | −0.101 (0.025) | −0.128 (0.005) | −0.040 (0.62) |
| As-of quarterly snapshot (PIT) | 7,424 | 458 | −0.065 (0.19) | −0.092 (0.094) | −0.106 (0.035) |
| As-of 1 Jan snapshots only (PIT) | 7,424 | 342 | −0.002 (0.98) | −0.052 (0.47) | −0.103 (0.047) |

Calls before 2022 have no snapshot (first model at 2022-01-01). Same-year and
as-of labels agree on 69% of calls. The Governance-Led crash-risk association
does not survive point-in-time labels at conventional levels; the sign stays
negative with the quarterly snapshot.
