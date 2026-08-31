# Journal — Harness 2: classification (`tagging.atom_pools` / `tagging.formulas`)

Change log of the classification harness's state (level 0: active atom pools
and frozen formulas in `configs/config.json`) and of the cycle code that fits
it (level 1: scripts 09–12). One entry per meta-optimization iteration,
appended by the proposer (see `.claude/skills/meta-harness-opt/`). Never edit
or delete past entries — this journal is part of the method's audit trail
(docs/meta_harness_methodology.md).

Entry template:

```
## NNN — YYYY-MM-DD — <one-line summary>
- State before: <atom pools / formulas / cycle-code state>
- Evidence read: <traces, reports, journal entries consulted>
- Change: <pool additions/removals, new 09 atoms (code), formula freezes>
- Validation: <dev fold-stability results per dimension — NEVER holdout before freeze>
- Holdout: <untouched | spent this iteration: result>
- Commit: <hash>
```

---

## 000 — 2026-08-31 — Initial basic state

- **State before:** none — this harness was rebuilt on this branch
  (pre-strip predecessor: `val_06` boolean search harness, removed in
  `9e6138c`; its four production formulas survive as LEGACY baseline
  candidates in `scripts/tag_harness_defs.py`).
- **Change:** level-0 state initialized deliberately minimal in
  `configs/config.json`:
  - `tagging.formulas`: `{}` (nothing frozen — the corpus cannot be tagged
    until a fit run earns a formula past the holdout gate).
  - `tagging.atom_pools` (active search space per dimension, 1–2 obvious
    atoms each):
    - `is_substantive`: has_ai_own_use, has_deployment_verb
    - `is_promotional`: has_word_transform, has_word_leader
    - `is_risk_related`: has_risk_factor
    - `is_governance_related`: has_board_oversight, has_compliance
    - `is_use_case_specific`: has_ai_use_case_specific
    - `is_quantified`: has_metric_percentage, has_metric_dollar
  - The full ~345-atom extractor (script 09) and the per-dimension reference
    library (`tag_harness_defs.DIMENSION_FEATURE_POOLS`) exist as the
    proposer's MENU — atoms enter the active pools only through journaled
    iterations, justified by dev evidence (trace false negatives/positives).
- **Known consequences:** with pools this small, the first `--dev-only` fit
  will be weak on most dimensions by construction. That is the point: each
  pool addition's marginal value is measurable against this floor.
- **Validation:** none needed (initialization, no search run).
- **Holdout:** untouched (never yet created for this harness).
- **Commit:** (this commit)
