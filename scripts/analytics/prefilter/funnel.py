"""
scripts/analytics/prefilter/funnel.py — the prefilter/frame/activity inference
funnel used throughout the thesis (`tbl-funnel-summary`, `tbl-a1-funnel-full`),
computed from bronze/silver instead of the DuckDB SQL those qmd chunks ran
inline.

One construct, one scope: the analysis universe is the S&P 500 at 2021-01-01
panel (`silver.firm_universe`). Panel documents are the union of
`silver.filing_manifest.document_id` and `silver.filing_manifest_10q.
accession_number` — the same set the removed `PANEL_DOCS` DuckDB predicate
and `scripts/analytics/corpus/headline_counts.py`'s `panel_docs` use
(`document_id` already equals `accession_number` for every form except
earnings calls, which have no SEC accession number). There is no
corpus-wide variant: the thesis's funnel is defined over the panel, so a
text_hash is "in panel" if at least one of its paragraph instances belongs
to a panel document, regardless of whether the same boilerplate text also
repeats in a non-panel filing.

Every unique-text stage counts DISTINCT text_hash, exactly as
`tbl-funnel-summary`/`tbl-a1-funnel-full` did against `gold_ai_frames` and
the deduped prefilter-predictions glob.

Output:
    data/results/prefilter/funnel.json        stage totals used by both tables
    data/results/prefilter/funnel_by_form.csv same stages, by (country_code, form)

Usage:
    uv run python scripts/analytics/prefilter/funnel.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "common"))
import layers as L  # noqa: E402

# Same form list tbl-funnel-summary/tbl-a1-funnel-full scope `paragraphs` to.
FORMS = ["10-K", "10-Q", "20-F", "DEF 14A", "8-K", "Earnings call"]


def panel_docs() -> list[str]:
    """document key of every filing in the S&P 500 at 2021-01-01 analysis
    panel, matching `corpus/headline_counts.py`'s `panel_docs`."""
    fm = L.scan("silver.filing_manifest").filter(pl.col("country_code") == "us")
    fmq = L.scan("silver.filing_manifest_10q").filter(pl.col("country_code") == "us")
    docs = pl.concat([
        fm.select(pl.col("document_id").alias("doc")),
        fmq.select(pl.col("accession_number").alias("doc")),
    ]).unique().collect()["doc"]
    return docs.to_list()


def panel_paragraphs(docs: list[str]) -> pl.LazyFrame:
    """bronze.paragraphs restricted to the panel and to `FORMS` — Stage 1-3
    population (`n_raw`, `n_scorable`, `n_unique`)."""
    return L.scan("bronze.paragraphs").filter(
        (pl.col("country_code") == "us")
        & pl.col("form").is_in(FORMS)
        & pl.col("accession_number").is_in(docs))


def panel_scorable_texts(docs: list[str]) -> pl.LazyFrame:
    """Every `text_hash` with >=1 scorable instance in a panel document --
    the exact membership test `tbl-funnel-summary`'s `n_prefiltered`
    subquery ran directly against `paragraphs` (`text_hash IN (SELECT
    text_hash FROM paragraphs WHERE ... is_scorable AND {PANEL_DOCS})`),
    not a filter on whichever single instance a downstream table happens to
    carry as its own `accession_number`."""
    return panel_paragraphs(docs).filter(pl.col("is_scorable")).select("text_hash").unique()


def scored_unique_texts(docs: list[str]) -> pl.LazyFrame:
    """`bronze.prefilter_scores` JOIN `bronze.unique_paragraphs` by
    `text_hash`, restricted to `is_scorable` texts with at least one
    instance in a panel document. Reports the lexical-gate signals only for
    texts bronze has already scored -- `n_unique`/stage 3 itself is counted
    straight from `paragraphs` in `paragraph_counts`, not through this join,
    so it isn't short-changed by any scoring backlog."""
    panel_hashes = panel_scorable_texts(docs)
    unique = L.scan("bronze.unique_paragraphs").select(
        "text_hash", "duplicate_count", "is_scorable",
        pl.col("country_code").alias("up_country_code"), pl.col("form").alias("up_form"))
    scores = L.scan("bronze.prefilter_scores").select(
        "text_hash", "strong_lexical_match", "weak_lexical_match")
    return (scores.join(unique, on="text_hash")
                  .filter(pl.col("is_scorable"))
                  .join(panel_hashes, on="text_hash", how="semi"))


def predictions(docs: list[str]) -> pl.LazyFrame:
    """`bronze.prefilter_predictions` (already deduped to the latest
    `model_version` per `text_hash`), restricted by `text_hash` membership
    in `panel_scorable_texts` -- matching the qmd's subquery exactly,
    because the surviving (latest-`model_version`) row for a `text_hash` can
    carry an arbitrary instance's `accession_number`, not necessarily one
    that is itself in-panel even when another instance of the same text is.
    The model-only decision is recomputed from the stored
    probability/threshold rather than re-derived from `is_ai_prefiltered`
    (which already folds in the named-entity rescue) -- Stage 4
    (`n_prefiltered`)."""
    panel_hashes = panel_scorable_texts(docs)
    return (L.scan("bronze.prefilter_predictions")
             .filter(pl.col("country_code") == "us")
             .join(panel_hashes, on="text_hash", how="semi")
             .with_columns((pl.col("predicted_proba") >= pl.col("threshold")).alias("model_positive")))


def frames(docs: list[str]) -> pl.LazyFrame:
    """`silver.ai_frames` restricted to panel documents and `has_frame`,
    deduped on `(text_hash, frame_id, subject)` -- same dedup
    `tbl-frame-schema`'s `_us_frames_dedup` used, needed because the same
    paragraph/frame can appear under more than one `accession_number`.
    Stage 5 (`n_framed`, `n_frames_total`, `n_firm_texts`)."""
    return (L.scan("silver.ai_frames")
             .filter((pl.col("country_code") == "us") & pl.col("accession_number").is_in(docs) & pl.col("has_frame"))
             .select("country_code", "form", "text_hash", "frame_id", "subject")
             .unique())


def activities(docs: list[str]) -> pl.LazyFrame:
    """`silver.ai_activities` restricted to panel documents and
    `has_activity`, deduped on `(text_hash, frame_id, activity_id)`.
    Stage 6 (`n_activities`) -- an activity-row count, distinct from the
    thesis's ticker-attributed gold activity spine count
    (`spines/activity/activity`); the two are cross-checked in
    this module's own verification, not reproduced here, since this script
    stays within bronze/silver."""
    return (L.scan("silver.ai_activities")
             .filter((pl.col("country_code") == "us") & pl.col("accession_number").is_in(docs) & pl.col("has_activity"))
             .select("country_code", "form", "text_hash", "frame_id", "activity_id")
             .unique())


def _stage_aggs() -> list[pl.Expr]:
    return [
        pl.len().alias("total_unique_paragraphs"),
        pl.col("duplicate_count").sum().alias("total_paragraph_instances"),
        (pl.col("strong_lexical_match") | pl.col("weak_lexical_match")).sum().alias("lexical_strong_or_weak"),
        pl.col("strong_lexical_match").sum().alias("lexical_strong_only"),
    ]


def paragraph_counts(docs: list[str], group: list[str] | None = None) -> pl.DataFrame:
    """Stages 1-3, straight from `paragraphs` -- `n_unique` (stage 3,
    deduplication) is `count(DISTINCT text_hash WHERE is_scorable)`, exactly
    the qmd's `totals` query, independent of whether `bronze.prefilter_scores`
    has scored that text yet."""
    p = panel_paragraphs(docs)
    aggs = [pl.len().alias("raw_paragraph_instances"),
            pl.col("is_scorable").sum().alias("scorable_paragraph_instances"),
            pl.col("text_hash").filter(pl.col("is_scorable")).n_unique().alias("unique_scorable_texts"),
            pl.col("accession_number").n_unique().alias("documents")]
    return (p.group_by(group).agg(*aggs) if group else p.select(*aggs)).collect()


def lexical_counts(docs: list[str], group: list[str] | None = None) -> pl.DataFrame:
    lf = scored_unique_texts(docs)
    if group:
        by = [pl.col(f"up_{g}").alias(g) for g in group]
        return lf.group_by(by).agg(*_stage_aggs()).collect()
    return lf.select(*_stage_aggs()).collect()


def prediction_counts(docs: list[str], group: list[str] | None = None) -> pl.DataFrame:
    lf = predictions(docs)
    aggs = [
        pl.col("model_positive").sum().alias("prefilter_model_only_positive"),
        (pl.col("named_entity_match") & ~pl.col("model_positive")).sum().alias("named_entity_rescued"),
        pl.col("is_ai_prefiltered").sum().alias("prefilter_model_positive"),
        pl.col("duplicate_count").filter(pl.col("is_ai_prefiltered")).sum()
          .alias("prefilter_model_positive_instances"),
    ]
    if group:
        by = [pl.col(g) for g in group]
        return lf.group_by(by).agg(
            *aggs, pl.col("text_hash").filter(pl.col("is_ai_prefiltered")).n_unique().alias("prefiltered_unique_texts"),
        ).collect()
    return lf.select(*aggs, pl.col("text_hash").filter(pl.col("is_ai_prefiltered")).n_unique().alias("prefiltered_unique_texts")).collect()


def frame_counts(docs: list[str], group: list[str] | None = None) -> pl.DataFrame:
    lf = frames(docs)
    aggs = [
        pl.col("text_hash").n_unique().alias("frames_positive_texts"),
        pl.len().alias("frames_total"),
        (pl.col("subject") == "firm").sum().alias("firm_subject_frames"),
        pl.col("text_hash").filter(pl.col("subject") == "firm").n_unique().alias("firm_texts"),
    ]
    return (lf.group_by(group).agg(*aggs) if group else lf.select(*aggs)).collect()


def activity_counts(docs: list[str], group: list[str] | None = None) -> pl.DataFrame:
    lf = activities(docs)
    aggs = [
        pl.col("text_hash").n_unique().alias("activities_positive_texts"),
        pl.len().alias("activities_total"),
    ]
    return (lf.group_by(group).agg(*aggs) if group else lf.select(*aggs)).collect()


def totals(docs: list[str]) -> dict:
    row = {}
    row.update(paragraph_counts(docs).row(0, named=True))
    row.update(lexical_counts(docs).row(0, named=True))
    row.update(prediction_counts(docs).row(0, named=True))
    row.update(frame_counts(docs).row(0, named=True))
    row.update(activity_counts(docs).row(0, named=True))
    row = {k: int(v or 0) for k, v in row.items()}

    n_raw = row["raw_paragraph_instances"]
    n_frames_total = row["frames_total"]
    # Names that match the thesis.qmd inline variables directly, so the
    # tables/prose can read this file without renaming anything:
    #   n_docs, n_raw, n_scorable, n_unique, n_prefiltered, n_framed,
    #   n_frames_total, share_firm_subject, n_firm_texts, n_activities.
    row["n_docs"] = row["documents"]
    row["n_raw"] = n_raw
    row["n_scorable"] = row["scorable_paragraph_instances"]
    row["n_unique"] = row["unique_scorable_texts"]
    row["n_prefiltered"] = row["prefiltered_unique_texts"]
    row["n_framed"] = row["frames_positive_texts"]
    row["n_frames_total"] = n_frames_total
    row["share_firm_subject"] = round(row["firm_subject_frames"] / n_frames_total * 100, 4) if n_frames_total else 0.0
    row["n_firm_texts"] = row["firm_texts"]
    # n_activities is the thesis's single construct for "disclosed activities":
    # the gold, ticker-attributed activity spine (data/gold/spines/activity/activity.parquet),
    # not this module's own bronze/silver activity-row count (`activities_total`,
    # narrower panel-document definition -- see `activities()` above).
    row["n_activities"] = pl.scan_parquet(L.gold_path("spines", "activity", "activity")).select(pl.len()).collect().item()
    return row


def by_form(docs: list[str]) -> pl.DataFrame:
    group = ["country_code", "form"]
    frame = (paragraph_counts(docs, group)
             .join(lexical_counts(docs, group), on=group, how="full", coalesce=True)
             .join(prediction_counts(docs, group), on=group, how="full", coalesce=True)
             .join(frame_counts(docs, group), on=group, how="full", coalesce=True)
             .join(activity_counts(docs, group), on=group, how="full", coalesce=True))
    numeric = [c for c in frame.columns if c not in group]
    return frame.with_columns([pl.col(c).fill_null(0).cast(pl.Int64) for c in numeric]).sort(group)


def main() -> None:
    docs = panel_docs()
    funnel = totals(docs)
    form_frame = by_form(docs)

    out_json = L.results_path("prefilter", "funnel.json")
    out_csv = L.results_path("prefilter", "funnel_by_form.csv")
    payload = {"funnel": funnel, "created_at": datetime.now(timezone.utc).isoformat()}
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    form_frame.write_csv(out_csv)

    print(f"-> {out_json}")
    for stage, count in funnel.items():
        print(f"  {stage}: {count:,}" if isinstance(count, int) else f"  {stage}: {count}")
    print(f"-> {out_csv} ({form_frame.height} forms)")


if __name__ == "__main__":
    main()
