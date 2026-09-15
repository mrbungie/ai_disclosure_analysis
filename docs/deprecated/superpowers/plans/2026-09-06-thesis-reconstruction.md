# Thesis reconstruction implementation plan

**Goal:** Restore the proposal's six empirical and managerial chapters with verified full-population paragraph-instance measures.

**Architecture:** Keep acquisition and the user's analytics edits untouched. Compute document, firm-year and firm-quarter measures from the Quarto corpus; place reusable statistical routines in `thesis_document/analysis.py`. Reuse saved archetypes only as explicitly conditional, historical supplementary evidence. Execute and render the complete document.

**Spec:** `thesis_document/docs/thesis_proposal_final.md`, `chapter_viz_plan_and_sources.md`, and the user's denominator/funnel corrections.

## Constraints

- Count all paragraph instances in each document-period, including repeated text and completed zero-AI observations.
- Keep 10-K, 10-Q and calls as distinct instruments; no pooled 10-K/10-Q estimation.
- Distinguish paragraph incidence from frame density and conditional shares.
- Preserve the supplied full-corpus funnel with its date and units.
- Exclude incomplete processing from inference and report coverage.
- Use actual results, including failed identification; never turn a failed pre-trend test into a null effect.
- English thesis; retain hypotheses, literature, six chapters, managerial applications and appendices.

## Tasks

- [ ] Implement verified aggregation, same-sample nested financial regressions, separate-venue event studies and paired venue comparisons in `thesis_document/analysis.py`.
- [ ] Restore Chapters 1–6, introduction, executive summary, conclusions and appendices in `thesis_document/thesis.qmd`; connect every empirical conclusion to executed tables.
- [ ] Execute the entire document, inspect results and revise interpretations; render a reviewable DOCX.
- [ ] Audit coverage of the proposal and report genuine limits (human validation and identification) separately from completed work.

## Verification

Use activated `.venv` for all scripts. Check aggregate denominators against the paragraph corpus, retain processed zeros, require unique financial join keys, compare fitted samples across nested models, verify funnel identities, execute every Python cell, and run Quarto DOCX rendering. Review the rendered document's text and tables against its computed output.
