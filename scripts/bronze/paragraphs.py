"""
scripts/bronze/paragraphs.py — bronze.extraction_trace, bronze.paragraphs,
bronze.sentences (partitioned by form).

extraction_trace: the current extraction of every (form, accession_number,
item_key), latest run wins, with `source_file`/`run_id` pointing at the
interim part it came from — the step from a paragraph back to raw.

paragraphs/sentences: split rules live in text_split.py. Filings are processed
one interim part file at a time, so memory stays bounded by the largest part.
Earnings calls are already one row per speaker turn and are mapped as-is:
form='Earnings call', accession_number=document_id, item_key=call section.

Usage:
    uv run python scripts/bronze/paragraphs.py [--forms 10-K 8-K ...]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import polars as pl
from _paths import COUNTRY, REPO_ROOT, SECTIONS, L, files, log
from text_split import KEY, is_scorable_expr, split_paragraphs, split_sentences, text_hash_expr

BUILDER = "scripts/bronze/paragraphs.py"

# form label -> interim section part-file stem
SECTION_FORMS = {
    "10-K": "filing_sections",
    "10-Q": "filing_sections_10q",
    "DEF 14A": "filing_sections_proxy",
    "20-F": "filing_sections_20f",
    "6-K": "filing_sections_6k",
    "8-K": "filing_sections_8k",
}
# Each revision of the prepared-remarks/Q&A segmentation wrote a new run next
# to the old ones; exactly one extractor version is read.
EARNINGS_CALLS_FORM = "Earnings call"
EARNINGS_CALLS_VERSION = "3"

TRACE_COLUMNS = ["accession_number", "ticker", "filing_date", "item_key", "section_name", "found",
                 "char_len", "word_count", "note", "run_id", "run_date", "run_comment", "part_num"]


def section_files(form: str) -> list[Path]:
    return files(SECTIONS, f"{SECTION_FORMS[form]}__run=*__part=*.parquet")


def call_files() -> list[Path]:
    return files(SECTIONS, f"earnings_call_paragraphs__v={EARNINGS_CALLS_VERSION}__run=*__part=*.parquet")


def current_trace(parts: list[Path]) -> pl.DataFrame:
    frames = []
    for f in parts:
        cols = [c for c in TRACE_COLUMNS if c in pl.read_parquet_schema(f)]
        frames.append(pl.scan_parquet(f).select(cols).with_row_index("source_row")
                      .with_columns(pl.lit(str(f.relative_to(REPO_ROOT))).alias("source_file")))
    return (pl.concat(frames, how="diagonal_relaxed")
            .sort(["run_date", "run_id", "part_num", "source_file", "source_row"], descending=True, nulls_last=True, maintain_order=True)
            .unique(["accession_number", "item_key"], keep="first", maintain_order=True)
            .with_columns(pl.lit(COUNTRY).alias("country_code"))
            .collect())


def build_form(form: str) -> None:
    parts = section_files(form)
    if not parts:
        log(f"skip {form}: no interim parts")
        return
    t0 = time.time()
    trace = current_trace(parts)
    trace_dir = L.reset_partition("bronze.extraction_trace", form=form)
    trace.drop("source_row").write_parquet(trace_dir / "part-00000.parquet", compression="zstd")

    par_dir = L.reset_partition("bronze.paragraphs", form=form)
    sen_dir = L.reset_partition("bronze.sentences", form=form)
    winners = trace.filter(pl.col("found")).select("source_file", "source_row")
    n_par = n_sen = 0
    for i, f in enumerate(parts):
        rows = winners.filter(pl.col("source_file") == str(f.relative_to(REPO_ROOT))).select("source_row")
        if rows.is_empty():
            continue
        sections = (pl.scan_parquet(f).select("accession_number", "item_key", "section_text")
                    .with_row_index("source_row")
                    .join(rows.lazy(), on="source_row", how="semi")
                    .with_columns(pl.lit(COUNTRY).alias("country_code"), pl.lit(form).alias("form")))
        paragraphs = split_paragraphs(sections).collect()
        sentences = split_sentences(paragraphs.lazy()).collect()
        paragraphs.drop("form").write_parquet(par_dir / f"part-{i:05d}.parquet", compression="zstd")
        sentences.drop("form").write_parquet(sen_dir / f"part-{i:05d}.parquet", compression="zstd")
        n_par += paragraphs.height
        n_sen += sentences.height
    log(f"{form}: {trace.height} sections, {n_par} paragraphs, {n_sen} sentences ({time.time() - t0:.0f}s)")


def build_calls() -> None:
    parts = call_files()
    if not parts:
        log("skip earnings calls: no interim parts")
        return
    calls = (pl.concat([pl.scan_parquet(f) for f in parts], how="diagonal_relaxed")
             .select(pl.lit(COUNTRY).alias("country_code"), pl.lit(EARNINGS_CALLS_FORM).alias("form"),
                     pl.col("document_id").alias("accession_number"), pl.col("section").alias("item_key"),
                     "content_type", "paragraph_index", "paragraph_text",
                     text_hash_expr("paragraph_text").alias("text_hash"),
                     is_scorable_expr("paragraph_text").alias("is_scorable"))
             .collect())
    par_dir = L.reset_partition("bronze.paragraphs", form=EARNINGS_CALLS_FORM)
    sen_dir = L.reset_partition("bronze.sentences", form=EARNINGS_CALLS_FORM)
    calls.drop("form").write_parquet(par_dir / "part-00000.parquet", compression="zstd")
    split_sentences(calls.lazy()).collect().drop("form").write_parquet(sen_dir / "part-00000.parquet",
                                                                         compression="zstd")
    log(f"{EARNINGS_CALLS_FORM}: {calls.height} paragraphs")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--forms", nargs="+", choices=[*SECTION_FORMS, EARNINGS_CALLS_FORM],
                        default=[*SECTION_FORMS, EARNINGS_CALLS_FORM])
    args = parser.parse_args()
    for form in args.forms:
        build_calls() if form == EARNINGS_CALLS_FORM else build_form(form)

    inputs = [f for form in SECTION_FORMS for f in section_files(form)] + call_files()
    L.finalize_partitioned("bronze.extraction_trace", keys=["form", "accession_number", "item_key"],
                           inputs=[f for form in SECTION_FORMS for f in section_files(form)], builder=BUILDER)
    L.finalize_partitioned("bronze.paragraphs", keys=[*KEY, "paragraph_index"], inputs=inputs, builder=BUILDER)
    L.finalize_partitioned("bronze.sentences", keys=[*KEY, "paragraph_index", "sentence_index"],
                           inputs=inputs, builder=BUILDER)


if __name__ == "__main__":
    main()
