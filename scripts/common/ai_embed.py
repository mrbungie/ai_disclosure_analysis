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
    """Registers already-embedded TEXTS, keyed by `text_hash` -- not by
    PARAGRAPH_KEY. (2026-09-05) The original per-instance design re-embedded
    every duplicate INSTANCE of the same text separately (a paragraph shared
    by 3.4M 8-K filings' cover pages got encoded 3.4M times), the exact
    "reimplement its own dedup instead of using unique_paragraphs" mistake
    already fixed once in ai_classify.py (docs/prefilter_evaluation.md §8.7)
    -- this is that same fix applied to the embedding step, which pays the
    real GPU cost `paragraphs`' own docstring warns about. Existing parts
    already carry `text_hash` per row (see `_text_hashes`), so this is a
    pure query-side change; no re-embedding or migration of old parts."""
    if not usable:
        con.execute(f"CREATE OR REPLACE TEMP VIEW {view} AS SELECT NULL::UBIGINT AS text_hash WHERE false")
        return 0
    files = ", ".join(f"'{part}'" for part in usable)
    con.execute(f"CREATE OR REPLACE TEMP VIEW {view} AS "
                f"SELECT DISTINCT text_hash FROM read_parquet([{files}], union_by_name=True)")
    return con.execute(f"SELECT count(*) FROM {view}").fetchone()[0]


def register_embedded_keys_combined(con, index_path: Path | None, extra_parts: list[Path],
                                     view: str = "embedded_keys") -> int:
    """Same `view` contract as `register_embedded_keys`, but unions TWO
    sources instead of picking one: the lightweight index (whatever
    coverage it already claims) plus any local part files NOT already
    reflected in it (`extra_parts` -- e.g. a part this box just wrote
    itself, on top of an index it only pulled). Never re-reads a part
    the index already accounts for. Safe with either input empty/None."""
    sources = []
    if index_path is not None:
        sources.append(f"SELECT text_hash FROM read_parquet('{index_path}')")
    if extra_parts:
        files = ", ".join(f"'{p}'" for p in extra_parts)
        sources.append(f"SELECT text_hash FROM read_parquet([{files}], union_by_name=True)")
    if not sources:
        con.execute(f"CREATE OR REPLACE TEMP VIEW {view} AS SELECT NULL::UBIGINT AS text_hash WHERE false")
        return 0
    con.execute(f"CREATE OR REPLACE TEMP VIEW {view} AS "
                f"SELECT DISTINCT text_hash FROM ({' UNION ALL '.join(sources)})")
    return con.execute(f"SELECT count(*) FROM {view}").fetchone()[0]


INDEX_FILENAME = "paragraph_embeddings_index.parquet"
INDEX_MANIFEST_FILENAME = "paragraph_embeddings_index.json"


def load_embedded_index(output_dir: Path, model_name: str, dtype: str) -> Path | None:
    """A lightweight substitute for pulling every embedding part just to
    learn WHICH texts are already done. The full parts are ~12GB
    (fixed_size_list<float16, 1024> per row); the set of already-embedded
    `text_hash` values is a few MB (one uint64 per unique text, ~4.8M rows
    at time of writing). A consumer that only needs to know what's
    PENDING -- not read any existing vector -- never needs the 12GB at
    all: `run_embed` uses this instead of `verify_parts`+
    `register_embedded_keys` whenever no local parts are present but this
    index is.

    Returns the index parquet path if a valid, config-matching index
    exists, else None (falls back to the full-parts path, or to "nothing
    embedded yet" if neither exists)."""
    index_path = output_dir / INDEX_FILENAME
    manifest_path = output_dir / INDEX_MANIFEST_FILENAME
    if not (index_path.exists() and manifest_path.exists()):
        return None
    try:
        sidecar = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if sidecar.get("model") != model_name or sidecar.get("dtype") != dtype:
        return None
    return index_path


def register_embedded_keys_from_index(con, index_path: Path, view: str = "embedded_keys") -> int:
    """Same contract as `register_embedded_keys` (creates the `view` TEMP
    VIEW every downstream query reads), sourced from the lightweight index
    instead of the full embedding parts."""
    con.execute(f"CREATE OR REPLACE TEMP VIEW {view} AS "
                f"SELECT DISTINCT text_hash FROM read_parquet('{index_path}')")
    return con.execute(f"SELECT count(*) FROM {view}").fetchone()[0]


def write_embedded_index(output_dir: Path, model_name: str, dtype: str, text_hashes,
                          part_files: list[str]) -> tuple[Path, int]:
    """Writes/overwrites the lightweight index from an ALREADY-KNOWN set of
    text_hash values (`run_embed` builds this in-memory as the union of
    whatever `embedded_keys` view it started the run with -- however that
    was sourced -- plus any hashes newly embedded THIS run; see the call
    site). Never re-scans the big parts to produce it: the set is already
    known to the caller more cheaply than a rescan would recompute it.

    `part_files` is bookkeeping only (which part files this index's
    coverage claims to reflect) -- not re-verified here, since the whole
    point of this path is not needing those files locally."""
    index_path = output_dir / INDEX_FILENAME
    staging = index_path.with_suffix(".parquet.partial")
    table = pa.Table.from_arrays(
        [pa.array(sorted(text_hashes), type=pa.uint64())], names=["text_hash"])
    writer = pq.ParquetWriter(staging, table.schema, compression="zstd")
    try:
        writer.write_table(table)
    finally:
        writer.close()
    staging.replace(index_path)
    manifest_path = output_dir / INDEX_MANIFEST_FILENAME
    manifest_path.write_text(json.dumps({
        "model": model_name, "dtype": dtype, "n_hashes": len(text_hashes),
        "part_files": sorted(part_files),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }, indent=2, sort_keys=True) + "\n")
    return index_path, len(text_hashes)


def pending_sql(limit: int = 0, against: str = "embedded_keys", source: str = "paragraphs",
                 countries: tuple[str, ...] | None = None) -> str:
    """`source` is `paragraphs` (one row per INSTANCE -- the original,
    cost-scales-with-duplication behavior) or `unique_paragraphs` (one row
    per unique `text_hash` -- the cheap, correct default; see
    register_embedded_keys' docstring). Either way the output schema is
    identical (same PARAGRAPH_KEY columns, naming ONE representative
    instance per text when source is unique_paragraphs) -- downstream
    readers never need to know which was used.

    `countries`, when given, restricts which country_code(s) get embedded
    this run -- e.g. Chile's paragraphs are wired into `paragraphs`/
    `unique_paragraphs` (docs/prefilter_evaluation.md §8.9) but its document
    parsing isn't trusted yet, so embedding/scoring it needs an explicit,
    separate go-ahead rather than riding along on a `paragraphs`-wide run."""
    country_filter = ""
    if countries:
        quoted = ", ".join(f"'{c}'" for c in countries)
        country_filter = f"AND p.country_code IN ({quoted})"
    return f"""
        SELECT {", ".join(f"p.{c}" for c in PARAGRAPH_KEY)}, p.paragraph_text
        FROM {source} AS p
        WHERE NOT EXISTS (SELECT 1 FROM {against} AS e WHERE e.text_hash = p.text_hash)
        {country_filter}
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
    source_relation: str = "unique_paragraphs",
    countries: tuple[str, ...] | None = None,
) -> tuple[list[Path], Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    audit = verify_parts(output_dir, model_name, dtype)
    usable = audit["usable"] if resume else []
    for bad in audit["unreadable"]:
        print(f"  UNREADABLE part ignored: {bad['path']} ({bad['error']})", flush=True)
    for bad in audit["mismatched"]:
        print(f"  part from another config ignored: {Path(bad['path']).name} "
              f"(model={bad['model']}, dtype={bad['dtype']}, rows={bad['rows']})", flush=True)

    # A box may have the lightweight index, some/all of the real parts, or
    # both -- e.g. a box that only pulled the index, then itself wrote one
    # new part, has 1 usable part locally that ISN'T reflected in the
    # index yet. Using "usable parts" and "the index" as mutually
    # exclusive alternatives (only fall back to the index when NO local
    # parts exist at all) silently drops the index's coverage the moment
    # even one local part shows up -- caught by testing exactly this
    # sequence (index-only run, then a second run after it had written a
    # part). The correct rule: always prefer the index for whatever it
    # already covers, and ADD ONLY the local parts NOT already reflected
    # in it (by filename) -- never re-derive coverage the index already
    # has just because some unrelated new file is now sitting locally.
    prior_index = load_embedded_index(output_dir, model_name, dtype) if resume else None
    indexed_files: set[str] = set()
    if prior_index is not None:
        try:
            indexed_files = set(json.loads((output_dir / INDEX_MANIFEST_FILENAME).read_text())
                                 .get("part_files", []))
        except (OSError, json.JSONDecodeError):
            prior_index = None
    extra_usable = [p for p in usable if p.name not in indexed_files]
    if prior_index is not None:
        print(f"Using lightweight index {prior_index.name} ({len(indexed_files)} part(s) reflected) "
              f"+ {len(extra_usable)} local part(s) not yet folded into it.", flush=True)

    con = duckdb.connect(str(database), read_only=True)
    try:
        country_where = ""
        if countries:
            quoted = ", ".join(f"'{c}'" for c in countries)
            country_where = f"WHERE p.country_code IN ({quoted})"
        row_count = con.sql(f"SELECT count(*) FROM {source_relation} AS p {country_where}").fetchone()[0]
        if row_count == 0:
            raise ValueError(f"{source_relation} is empty (for countries={countries}); "
                              f"build it with `make duckdb-text` first")
        embedded = register_embedded_keys_combined(con, prior_index, extra_usable)
        prior_part_files = sorted(indexed_files)
        coverage = con.execute(f"""
            SELECT p.country_code, p.form, count(*) AS paragraphs,
                   count(*) FILTER (WHERE EXISTS (SELECT 1 FROM embedded_keys AS e WHERE e.text_hash = p.text_hash)) AS embedded
            FROM {source_relation} AS p {country_where} GROUP BY 1, 2 ORDER BY 1, 2
        """).df().to_dict("records")

        print(f"{len(usable)} usable part(s), {embedded:,} paragraphs already embedded", flush=True)
        for row in coverage:
            done, total = int(row["embedded"]), int(row["paragraphs"])
            print(f"  {row['country_code']}/{row['form']}: {done:,}/{total:,} embedded, "
                  f"{total - done:,} pending", flush=True)
        pending = row_count - embedded
        target_rows = min(limit, pending) if limit and limit > 0 else pending
        print(f"Pending this run: {target_rows:,} paragraphs", flush=True)

        manifest = {
            "run_id": run_id, "model": model_name, "dtype": dtype,
            "embedding_dim": EMBEDDING_DIM, "storage": "fixed_size_list<float16>, uncompressed",
            "corpus_paragraphs": row_count, "already_embedded": embedded,
            "pending_at_start": pending, "coverage_before": coverage,
            "sample_seed": sample_seed, "reused_parts": [str(p) for p in usable],
            "unreadable_parts": audit["unreadable"], "mismatched_parts": audit["mismatched"],
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        if verify_only or target_rows == 0:
            manifest["verify_only"] = True
            if target_rows == 0 and not verify_only:
                print("Nothing pending — every paragraph already has a vector.", flush=True)
            if extra_usable:
                # Bootstrap/refresh case: this box has local parts not yet
                # folded into the index (either there's no index yet, or
                # some local parts postdate it) -- publish/resync the
                # index even though no new embedding happened this run,
                # so a box that only pulls the index later stays correct.
                # `embedded_keys` is already the combined (index +
                # extra_usable) view set up above; no rescan needed.
                index_path, n_hashes = write_embedded_index(
                    output_dir, model_name, dtype,
                    con.execute("SELECT text_hash FROM embedded_keys").fetchnumpy()["text_hash"].tolist(),
                    sorted(indexed_files | {p.name for p in usable}))
                manifest["index_path"], manifest["index_hashes"] = str(index_path), n_hashes
            return [], _write_manifest(output_dir, run_id, manifest)

        device = resolve_device(device)
        print(f"Device: {device} | Precision: {dtype}", flush=True)
        model = _load_model(model_name, device, dtype)
        sample = con.sql(f"SELECT paragraph_text FROM {source_relation} AS p {country_where} "
                         f"USING SAMPLE {max(1, min(probe_rows, row_count))} ROWS "
                         f"(reservoir, {sample_seed})").fetchall()
        plan = autotune_batching(model, [r[0] or "" for r in sample], device, memory_fraction)
        print(f"Autotune: {_gib(plan['free_bytes'])} free of {_gib(plan['total_bytes'])} | "
              f"{plan['bytes_per_token']:,.0f} B/token | budget {plan['token_budget']:,} tokens/batch "
              f"(<= {plan['max_rows_per_batch']:,} rows, projected peak {_gib(plan['projected_peak_bytes'])}) | "
              f"stopped on {plan['stop_reason']} at {plan['saturated_tokens_per_second']/1000:,.1f}k tok/s",
              flush=True)

        # Snapshot of what "already embedded" meant AT THE START of this run
        # (however `embedded_keys` was sourced -- full parts or the
        # lightweight index) -- kept in memory so the index can be
        # refreshed at the end as prior_hashes | newly_embedded_hashes,
        # with no rescan of any parquet, big or small.
        prior_hashes = con.execute("SELECT text_hash FROM embedded_keys").fetchnumpy()["text_hash"]

        result = con.execute(pending_sql(limit, source=source_relation, countries=countries))
        reporter = ThroughputReporter(target_rows, "embedding", progress_seconds, device)
        written: list[Path] = []
        buffered: list[pa.Table] = []
        buffered_rows = 0
        embedded_rows = 0
        encode_seconds = 0.0
        new_hash_arrays: list[pa.Array] = []

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
            hash_array = _text_hashes(texts)
            new_hash_arrays.append(hash_array)
            table = pa.Table.from_pandas(chunk, preserve_index=False).append_column(
                "embedding", _embedding_column(vectors)).append_column(
                "text_hash", hash_array).append_column(
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
    if written:
        # Refresh the lightweight index: prior_hashes (whatever coverage
        # this run STARTED with, full-parts- or index-sourced) union the
        # hashes just embedded -- no rescan needed either way.
        new_hashes = (pa.concat_arrays(new_hash_arrays) if new_hash_arrays
                     else pa.array([], type=pa.uint64()))
        merged_hashes = set(prior_hashes.tolist()) | set(new_hashes.to_pylist())
        contributing = sorted(set(prior_part_files) | {p.name for p in usable} | {p.name for p in written})
        index_path, n_hashes = write_embedded_index(output_dir, model_name, dtype, merged_hashes, contributing)
        manifest["index_path"], manifest["index_hashes"] = str(index_path), n_hashes
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
    parser.add_argument("--source-relation", choices=("paragraphs", "unique_paragraphs"),
                        default="unique_paragraphs",
                        help="unique_paragraphs (default): embed each unique text ONCE, keyed by "
                             "text_hash -- the correct default now that a downstream join back to "
                             "instances is available wherever needed. paragraphs: the original "
                             "per-instance behavior (re-embeds every duplicate), kept only for "
                             "exact reproduction of pre-2026-09-05 runs.")
    parser.add_argument("--countries", nargs="+", default=None,
                        help="Restrict this run to these country_code(s) (e.g. --countries us). "
                             "Default: no restriction, embeds every country wired into the source "
                             "relation -- pass this explicitly when a country's paragraphs exist "
                             "but its document parsing/scoring hasn't been signed off yet (see "
                             "Chile in docs/prefilter_evaluation.md §8.9).")
    args = parser.parse_args()
    if args.countries:
        args.countries = tuple(args.countries)
    written, manifest = run_embed(**vars(args))
    print(f"Parts written: {len(written)}")
    print(f"Manifest -> {manifest}")


if __name__ == "__main__":
    main()
