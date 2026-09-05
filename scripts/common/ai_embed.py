"""Stage 1 of the prefilter: embed every paragraph once, keep the vectors.

Split from ai_prefilter.py because the two halves have wildly different costs
and lifetimes. Embedding 3.28M paragraphs is ~35 min of RTX 5090 time; scoring
those vectors against 20 anchors is a 3.28M x 1024 by 1024 x 20 matmul that
runs in seconds. Keeping only the scores meant that editing a single anchor —
the thing most likely to change while tuning a prefilter — forced a full
re-embed. Vectors are anchor-independent, so they are written once and every
later scoring pass reads them.

Storage: fixed_size_list<float16, 1024>, uncompressed. fp16 because the
encoder itself runs in fp16, so fp32 storage would double the size (13.5 GB
vs 6.3 GB) to record precision the vectors never had. Uncompressed because
normalized embeddings are near-random: zstd buys 0.2% here and costs read
speed on every downstream pass.

Resumable and additive: parts are verified on startup and only the missing
paragraphs are embedded, so adding a country later, or resuming after a
crash, costs only the new rows.

Scope is a WHERE clause, not a separate script. `paragraphs` is one table
across every form and country, so a run says which slice of it it owns:
--form/--country-code narrow the population (a country whose vectors
aren't wanted yet stays untouched instead of riding along), and
--source-relation unique_paragraphs embeds each distinct TEXT once rather
than once per instance. That last one matters at this corpus's duplication
rate: DEF 14A + 8-K are 5,303,459 paragraph instances but only 2,668,526
distinct texts, so instance-level embedding buys ~2.6M duplicate vectors
and doubles the GPU time for nothing the prefilter reads — it already
scores `unique_paragraphs` (docs/prefilter_evaluation.md §8.8).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from embedding_runtime import (
    DEFAULT_MEMORY_FRACTION, PARAGRAPH_KEY, TORCH_DTYPES, ThroughputReporter,
    _gib, _load_model, autotune_batching, encode_planned, resolve_device,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = "BAAI/bge-m3"
DEFAULT_DATABASE = REPO_ROOT / "duckdb" / "thesis.duckdb"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "interim" / "embeddings" / "paragraphs"
PART_GLOB = "paragraph_embeddings__run=*.parquet"
EMBEDDING_DIM = 1024


def parts(output_dir: Path) -> list[Path]:
    """Committed parts only. `.partial` files are staged writes from a killed
    run and must never be mistaken for coverage."""
    return sorted(output_dir.glob(PART_GLOB))


def verify_parts(output_dir: Path, model_name: str, dtype: str) -> dict:
    """Validate each part before counting it as work already done.

    A part is usable only if it reads back, carries the paragraph key and the
    embedding column, and came from this model at this precision. Failures are
    reported and skipped, never deleted.
    """
    usable: list[Path] = []
    unreadable: list[dict] = []
    mismatched: list[dict] = []
    con = duckdb.connect()
    try:
        for part in parts(output_dir):
            try:
                columns = {name for name, *_ in con.execute(
                    f"DESCRIBE SELECT * FROM read_parquet('{part}')").fetchall()}
                row = con.execute(
                    f"SELECT count(*), any_value(model), any_value(dtype) FROM read_parquet('{part}')"
                ).fetchone()
            except (duckdb.Error, OSError) as error:
                unreadable.append({"path": str(part), "error": str(error).splitlines()[0]})
                continue
            missing = [c for c in (*PARAGRAPH_KEY, "embedding") if c not in columns]
            if missing or row[1] != model_name or row[2] != dtype:
                mismatched.append({"path": str(part), "rows": row[0], "model": row[1],
                                   "dtype": row[2], "missing_columns": missing})
                continue
            usable.append(part)
    finally:
        con.close()
    return {"usable": usable, "unreadable": unreadable, "mismatched": mismatched}


def register_embedded_keys(con, usable: list[Path], view: str = "embedded_keys") -> int:
    columns = ", ".join(PARAGRAPH_KEY)
    if not usable:
        con.execute(f"CREATE OR REPLACE TEMP VIEW {view} AS SELECT {columns} FROM paragraphs WHERE false")
        return 0
    files = ", ".join(f"'{part}'" for part in usable)
    con.execute(f"CREATE OR REPLACE TEMP VIEW {view} AS "
                f"SELECT DISTINCT {columns} FROM read_parquet([{files}], union_by_name=True)")
    return con.execute(f"SELECT count(*) FROM {view}").fetchone()[0]


def population_filter(forms=(), countries=(), scorable_only: bool = False,
                      alias: str = "p") -> str:
    """SQL predicate narrowing WHICH paragraphs a run is responsible for.

    `paragraphs` is deliberately ONE table across every form and country
    (see build_duckdb.py:_paragraph_select_sql), so "embed the new DEF 14A
    and 8-K text" would otherwise also sweep up every other unembedded row
    sharing that table — including a country whose vectors are deliberately
    not wanted yet, which is not a hypothetical: CL paragraphs entered
    `paragraphs` with no embeddings of their own. Scoping a run is a WHERE
    clause here, the same way `form`/`country_code` is how every other
    query in the pipeline re-imposes that split.

    Empty means the whole table, so the unfiltered path is byte-identical
    to what it was before this existed.
    """
    def in_list(column: str, values) -> str:
        literals = ", ".join("'" + str(v).replace("'", "''") + "'" for v in values)
        return f"{alias}.{column} IN ({literals})"

    clauses = []
    if forms:
        clauses.append(in_list("form", forms))
    if countries:
        clauses.append(in_list("country_code", countries))
    if scorable_only:
        # `is_scorable` (build_duckdb.py) marks the rows extraction left as
        # paragraphs but that carry no content — lone bullets, zero-width
        # spaces, dashes. Off by default deliberately: nothing downstream
        # filters on the column yet, and the 10-K/10-Q vectors already on
        # disk were written without it, so turning it on by default would
        # make coverage mean something different per form.
        clauses.append(f"{alias}.is_scorable")
    return " AND ".join(clauses) if clauses else "TRUE"


def pending_sql(limit: int = 0, against: str = "embedded_keys", population: str = "TRUE",
                source_relation: str = "paragraphs") -> str:
    keys = " AND ".join(f"e.{c} IS NOT DISTINCT FROM p.{c}" for c in PARAGRAPH_KEY)
    return f"""
        SELECT {", ".join(f"p.{c}" for c in PARAGRAPH_KEY)}, p.paragraph_text
        FROM {source_relation} AS p
        WHERE ({population})
          AND NOT EXISTS (SELECT 1 FROM {against} AS e WHERE {keys})
        ORDER BY p.form, p.country_code, p.accession_number, p.item_key, p.paragraph_index
        {f'LIMIT {int(limit)}' if limit and limit > 0 else ''}
    """


def _text_hashes(texts: list[str]) -> pa.Array:
    """8-byte digest per paragraph, so a later extraction rerun that changes a
    paragraph's text is detectable even though its key stayed the same."""
    return pa.array([int.from_bytes(hashlib.blake2b(t.encode("utf-8"), digest_size=8).digest(),
                                    "big", signed=False) for t in texts], type=pa.uint64())


def _embedding_column(vectors: np.ndarray) -> pa.Array:
    flat = pa.array(vectors.astype(np.float16).ravel(), type=pa.float16())
    return pa.FixedSizeListArray.from_arrays(flat, EMBEDDING_DIM)


def _commit_part(table: pa.Table, output_dir: Path, run_id: str, index: int) -> Path:
    """Atomic: stage under .partial, then rename. A killed run leaves debris the
    coverage glob ignores, never a truncated part that looks complete."""
    final = output_dir / f"paragraph_embeddings__run={run_id}__part={index:05d}.parquet"
    staging = final.with_suffix(".parquet.partial")
    writer = pq.ParquetWriter(staging, table.schema, compression="none")
    try:
        writer.write_table(table)
    finally:
        writer.close()
    staging.replace(final)
    return final


def run_embed(
    database: Path,
    output_dir: Path,
    model_name: str,
    device: str | None,
    dtype: str = "fp16",
    memory_fraction: float = DEFAULT_MEMORY_FRACTION,
    progress_seconds: float = 60.0,
    limit: int = 0,
    fetch_rows: int = 20_000,
    part_rows: int = 50_000,
    probe_rows: int = 1_000,
    sample_seed: int = 42,
    resume: bool = True,
    verify_only: bool = False,
    forms: list[str] | None = None,
    countries: list[str] | None = None,
    scorable_only: bool = False,
    source_relation: str = "paragraphs",
) -> tuple[list[Path], Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    forms, countries = tuple(forms or ()), tuple(countries or ())
    population = population_filter(forms, countries, scorable_only)

    audit = verify_parts(output_dir, model_name, dtype)
    usable = audit["usable"] if resume else []
    for bad in audit["unreadable"]:
        print(f"  UNREADABLE part ignored: {bad['path']} ({bad['error']})", flush=True)
    for bad in audit["mismatched"]:
        print(f"  part from another config ignored: {Path(bad['path']).name} "
              f"(model={bad['model']}, dtype={bad['dtype']}, rows={bad['rows']})", flush=True)

    con = duckdb.connect(str(database), read_only=True)
    try:
        if con.sql(f"SELECT count(*) FROM {source_relation}").fetchone()[0] == 0:
            raise ValueError(f"{source_relation} is empty; build it with `make duckdb-text` first")
        row_count = con.execute(
            f"SELECT count(*) FROM {source_relation} AS p WHERE {population}").fetchone()[0]
        if row_count == 0:
            raise ValueError(f"no paragraphs match this run's population ({population}); "
                             f"check --form/--country-code against `SELECT DISTINCT country_code, "
                             f"form FROM paragraphs`")
        embedded = register_embedded_keys(con, usable)
        keys = " AND ".join(f"e.{c} IS NOT DISTINCT FROM p.{c}" for c in PARAGRAPH_KEY)
        # Coverage is reported over the WHOLE table, not just this run's
        # population: the point of the line is to show what does and does
        # not have vectors, and hiding the rows a filter excluded is how a
        # country silently stays unembedded for months. `in scope` marks
        # which of them this run will actually touch.
        coverage = con.execute(f"""
            SELECT p.country_code, p.form, count(*) AS paragraphs,
                   count(*) FILTER (WHERE EXISTS (SELECT 1 FROM embedded_keys AS e WHERE {keys})) AS embedded,
                   coalesce(bool_or({population}), false) AS in_scope
            FROM {source_relation} AS p GROUP BY 1, 2 ORDER BY 1, 2
        """).df().to_dict("records")

        print(f"{len(usable)} usable part(s), {embedded:,} paragraphs already embedded", flush=True)
        if forms or countries:
            print(f"Population this run: {population}", flush=True)
        for row in coverage:
            done, total = int(row["embedded"]), int(row["paragraphs"])
            scope = "" if row["in_scope"] else "  [out of scope this run]"
            print(f"  {row['country_code']}/{row['form']}: {done:,}/{total:,} embedded, "
                  f"{total - done:,} pending{scope}", flush=True)
        pending = con.execute(f"""
            SELECT count(*) FROM {source_relation} AS p
            WHERE ({population})
              AND NOT EXISTS (SELECT 1 FROM embedded_keys AS e WHERE {keys})
        """).fetchone()[0]
        target_rows = min(limit, pending) if limit and limit > 0 else pending
        print(f"Pending this run: {target_rows:,} paragraphs", flush=True)

        manifest = {
            "run_id": run_id, "model": model_name, "dtype": dtype,
            "embedding_dim": EMBEDDING_DIM, "storage": "fixed_size_list<float16>, uncompressed",
            "corpus_paragraphs": row_count, "already_embedded": embedded,
            "population_filter": population, "population_forms": list(forms),
            "population_countries": list(countries), "population_scorable_only": scorable_only,
            "source_relation": source_relation,
            "pending_at_start": pending, "coverage_before": coverage,
            "sample_seed": sample_seed, "reused_parts": [str(p) for p in usable],
            "unreadable_parts": audit["unreadable"], "mismatched_parts": audit["mismatched"],
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        if verify_only or target_rows == 0:
            manifest["verify_only"] = True
            if target_rows == 0 and not verify_only:
                print("Nothing pending — every paragraph already has a vector.", flush=True)
            return [], _write_manifest(output_dir, run_id, manifest)

        device = resolve_device(device)
        print(f"Device: {device} | Precision: {dtype}", flush=True)
        model = _load_model(model_name, device, dtype)
        sample = con.sql(f"SELECT paragraph_text FROM "
                         f"(SELECT paragraph_text FROM {source_relation} AS p WHERE {population}) "
                         f"USING SAMPLE {max(1, min(probe_rows, row_count))} ROWS "
                         f"(reservoir, {sample_seed})").fetchall()
        plan = autotune_batching(model, [r[0] or "" for r in sample], device, memory_fraction)
        print(f"Autotune: {_gib(plan['free_bytes'])} free of {_gib(plan['total_bytes'])} | "
              f"{plan['bytes_per_token']:,.0f} B/token | budget {plan['token_budget']:,} tokens/batch "
              f"(<= {plan['max_rows_per_batch']:,} rows, projected peak {_gib(plan['projected_peak_bytes'])}) | "
              f"stopped on {plan['stop_reason']} at {plan['saturated_tokens_per_second']/1000:,.1f}k tok/s",
              flush=True)

        result = con.execute(pending_sql(limit, population=population,
                                         source_relation=source_relation))
        reporter = ThroughputReporter(target_rows, "embedding", progress_seconds, device)
        written: list[Path] = []
        buffered: list[pa.Table] = []
        buffered_rows = 0
        embedded_rows = 0
        encode_seconds = 0.0

        def flush() -> None:
            nonlocal buffered, buffered_rows
            if buffered:
                written.append(_commit_part(pa.concat_tables(buffered), output_dir, run_id, len(written)))
                buffered, buffered_rows = [], 0

        while True:
            chunk = result.fetch_df_chunk(vectors_per_chunk=max(1, fetch_rows // 2048))
            if chunk.empty:
                break
            texts = chunk.pop("paragraph_text").fillna("").tolist()
            vectors, chunk_seconds = encode_planned(model, texts, plan, reporter)
            encode_seconds += chunk_seconds
            table = pa.Table.from_pandas(chunk, preserve_index=False).append_column(
                "embedding", _embedding_column(vectors)).append_column(
                "text_hash", _text_hashes(texts)).append_column(
                "model", pa.array([model_name] * len(texts)).dictionary_encode()).append_column(
                "dtype", pa.array([dtype] * len(texts)).dictionary_encode()).append_column(
                "run_id", pa.array([run_id] * len(texts)).dictionary_encode())
            buffered.append(table)
            buffered_rows += len(texts)
            embedded_rows += len(texts)
            if buffered_rows >= part_rows:
                flush()
        flush()
        reporter.close()
    finally:
        con.close()

    manifest.update({
        "device": str(model.device), "autotune": plan, "embedded_rows": embedded_rows,
        "parts_written": [str(p) for p in written], "encode_seconds": encode_seconds,
        "paragraphs_per_second": embedded_rows / encode_seconds if encode_seconds else None,
        "wall_seconds": time.perf_counter() - reporter.started,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    })
    return written, _write_manifest(output_dir, run_id, manifest)


def _write_manifest(output_dir: Path, run_id: str, metadata: dict) -> Path:
    path = output_dir / f"paragraph_embeddings_manifest__run={run_id}.json"
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True, default=str) + "\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", dest="model_name", default=DEFAULT_MODEL)
    parser.add_argument("--device", default=None, help="cuda, mps, cpu (default: best available)")
    parser.add_argument("--dtype", choices=tuple(TORCH_DTYPES), default="fp16",
                        help="Encoder precision; fp16 is ~2.9x faster than fp32 with min cosine 0.9998")
    parser.add_argument("--memory-fraction", type=float, default=DEFAULT_MEMORY_FRACTION)
    parser.add_argument("--progress-seconds", type=float, default=60.0)
    parser.add_argument("--limit", type=int, default=0, help="Embed only the first N pending paragraphs")
    parser.add_argument("--fetch-rows", type=int, default=20_000)
    parser.add_argument("--part-rows", type=int, default=50_000)
    parser.add_argument("--probe-rows", type=int, default=1_000)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--form", dest="forms", action="append", metavar="FORM",
                        help="Embed only this form (repeatable), e.g. --form 'DEF 14A' --form 8-K. "
                             "Default: every form in `paragraphs`.")
    parser.add_argument("--country-code", dest="countries", action="append", metavar="CC",
                        help="Embed only this country (repeatable), e.g. --country-code us. "
                             "Default: every country in `paragraphs`.")
    parser.add_argument("--scorable-only", action="store_true",
                        help="Skip paragraphs marked is_scorable=false (lone bullets, empty "
                             "cells, dashes). Off by default: nothing downstream filters on the "
                             "column yet and the existing 10-K/10-Q vectors were written without it.")
    parser.add_argument("--source-relation", default="paragraphs",
                        help="'unique_paragraphs' embeds each unique paragraph TEXT once instead "
                             "of once per corpus instance (see docs/prefilter_evaluation.md §8.8). "
                             "Score it with ai_prefilter.py --source-relation unique_paragraphs.")
    args = parser.parse_args()
    written, manifest = run_embed(**vars(args))
    print(f"Parts written: {len(written)}")
    print(f"Manifest -> {manifest}")


if __name__ == "__main__":
    main()
