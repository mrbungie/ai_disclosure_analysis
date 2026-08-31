# Journal — Harness 1: detection (`ai_keywords`)

Change log of the detection harness's state (level 0) and of the cycle code
that fits it (level 1). One entry per meta-optimization iteration, appended by
the proposer (see `.claude/skills/meta-harness-opt/`). Never edit or delete
past entries — this journal is part of the method's audit trail
(docs/meta_harness_methodology.md).

Entry template:

```
## NNN — YYYY-MM-DD — <one-line summary>
- State before: <keywords / cycle-code state>
- Evidence read: <traces, reports, journal entries consulted>
- Change: <what changed, level 0 and/or level 1>
- Validation: <dev / self-check results — NEVER holdout before freeze>
- Holdout: <untouched | spent this iteration: result>
- Commit: <hash>
```

---

## 000 — 2026-08-31 — Reset to basic initial state

- **State before (archived):** 23 keywords fit through the pre-journal cycle
  and manual curation — `ai, artificial intelligence, generative ai, gen ai,
  machine learning, large language model, llm, deep learning, natural language
  processing, predictive analytics, algorithmic, automation, computer vision,
  neural network, openai, anthropic, claude, deepseek, chatgpt, copilot,
  gemini, aip, aiops` — plus false positives `adobe illustrator, appreciation,
  said, paid`. Preserved verbatim in git at commit `29ea77d`
  (`configs/config.json`) and evaluated at holdout F1=0.960 (raw, balanced
  sample; see `reports/prefilter_fit_holdout_eval.txt`).
- **Change:** level-0 state reset to the minimal seed
  `["ai", "artificial intelligence", "machine learning"]`, false positives
  `[]`. From here, every keyword the harness gains must be earned through a
  journaled optimization iteration — the growth trajectory becomes the
  documented result, instead of a tuned list with mixed provenance.
- **Known consequences:** corpus artifacts currently on disk
  (`filing_manifest` flags, `ai_candidate_chunks.parquet`) still reflect the
  archived 23-keyword list until 05–06 are re-run after the next freeze. The
  existing labeled batch (`data/interim/prefilter_fit/labeled.parquet`, 700
  rows) was drawn under the archived list's candidate windows and its holdout
  is spent — the first optimization iteration from this seed needs a fresh 07
  batch (which will also carry `sampling_weight`).
- **Validation:** a `--dev-only` smoke run on the archived batch scored the
  3-keyword seed at dev fold-stability F1=0.959 — nearly the full list's
  score, evidence that this batch (drawn under the archived list's windows)
  cannot discriminate seed from full list and MUST NOT be used to justify
  additions; it validates only the `--dev-only` plumbing.
- **Holdout:** untouched (the spent one belongs to the archived state).
- **Commit:** (this commit)
