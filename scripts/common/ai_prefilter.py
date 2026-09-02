"""Terminal, score-only lexical and semantic AI prefilter for paragraphs."""

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

from ai_prefilter_anchors import POSITIVE_ANCHORS, anchor_rows, register_anchors
import ai_embed
from embedding_runtime import (  # maquinaria compartida con ai_embed.py
    DEFAULT_MEMORY_FRACTION, PARAGRAPH_KEY, TORCH_DTYPES, ThroughputReporter,
    _encode, _gib, _load_model, resolve_device,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = "BAAI/bge-m3"
DEFAULT_DATABASE = REPO_ROOT / "duckdb" / "thesis.duckdb"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "interim" / "prefilter_scores"

STRONG_AI_TERMS = (
    "artificial intelligence", "generative ai", "genai", "machine learning",
    "deep learning", "large language model", "large language models",
    "foundation model", "foundation models", "neural network", "neural networks",
    "chatgpt", "openai", "gpt-4", "copilot", "gemini", "claude",
)
WEAK_AI_TERMS = (
    "algorithm", "predictive", "prediction", "recommendation", "classifier",
    "automation", "automated decision", "computer vision",
    "natural language processing", "nlp", "model",
)


def _sql_list(items: tuple[str, ...]) -> str:
    return "[" + ", ".join("'" + item.replace("'", "''") + "'" for item in items) + "]"


def lexical_scores_sql(source_relation: str = "paragraphs") -> str:
    """DuckDB-only simple substring matching; no Python lexical matcher."""
    text = "lower(coalesce(p.paragraph_text, ''))"
    return f"""
        SELECT p.paragraph_index,
               list_filter({_sql_list(STRONG_AI_TERMS)}, term -> contains({text}, term)) AS strong_matched_terms,
               list_filter({_sql_list(WEAK_AI_TERMS)}, term -> contains({text}, term)) AS weak_matched_terms,
               list_count(list_filter({_sql_list(STRONG_AI_TERMS)}, term -> contains({text}, term))) > 0 AS strong_lexical_match,
               list_count(list_filter({_sql_list(WEAK_AI_TERMS)}, term -> contains({text}, term))) > 0 AS weak_lexical_match
        FROM {source_relation} AS p
        ORDER BY p.paragraph_index
    """


PART_GLOB = "prefilter_scores__run=*.parquet"


def lexical_input_sql(source_relation: str = "paragraphs", limit: int = 0, pending_against: str | None = None) -> str:
    """Paragraph lineage/text plus lexical features, all computed in DuckDB.

    `pending_against` names a relation of already-scored keys to ANTI JOIN
    away, which is what makes a run additive: a re-run scores the complement
    of what is already on disk, so adding a country (or re-running after a
    crash) never re-embeds a paragraph that already has a vector.
    """
    text = "lower(coalesce(p.paragraph_text, ''))"
    anti = ""
    if pending_against:
        keys = " AND ".join(f"s.{column} IS NOT DISTINCT FROM p.{column}" for column in PARAGRAPH_KEY)
        anti = f"WHERE NOT EXISTS (SELECT 1 FROM {pending_against} AS s WHERE {keys})"
    return f"""
        SELECT p.*, 
               list_filter({_sql_list(STRONG_AI_TERMS)}, term -> contains({text}, term)) AS strong_matched_terms,
               list_filter({_sql_list(WEAK_AI_TERMS)}, term -> contains({text}, term)) AS weak_matched_terms,
               list_count(list_filter({_sql_list(STRONG_AI_TERMS)}, term -> contains({text}, term))) > 0 AS strong_lexical_match,
               list_count(list_filter({_sql_list(WEAK_AI_TERMS)}, term -> contains({text}, term))) > 0 AS weak_lexical_match
        FROM {source_relation} AS p
        {anti}
        ORDER BY p.form, p.country_code, p.accession_number, p.item_key, p.paragraph_index
        {f'LIMIT {int(limit)}' if limit and limit > 0 else ''}
    """


def score_embeddings(paragraph_embeddings: np.ndarray, anchor_embeddings: np.ndarray, anchors: list[dict[str, str]]) -> dict[str, np.ndarray]:
    """Dot-product scores for normalized paragraph and anchor embeddings."""
    similarities = paragraph_embeddings @ anchor_embeddings.T
    result: dict[str, np.ndarray] = {}
    positive_scores = []
    categories = tuple(
        category for category in POSITIVE_ANCHORS
        if any(anchor["category"] == category for anchor in anchors)
    )
    for category in categories:
        indices = [i for i, anchor in enumerate(anchors) if anchor["category"] == category]
        score = similarities[:, indices].max(axis=1)
        result[f"score_{category}"] = score
        positive_scores.append(score)

    negative_indices = [i for i, anchor in enumerate(anchors) if anchor["polarity"] == "negative"]
    result["negative_similarity"] = similarities[:, negative_indices].max(axis=1)
    matrix = np.column_stack(positive_scores)
    best_indices = matrix.argmax(axis=1)
    result["max_semantic_score"] = matrix.max(axis=1)
    result["best_semantic_anchor"] = np.asarray(categories, dtype=object)[best_indices]
    return result



def anchors_fingerprint(anchors: list[dict[str, str]]) -> str:
    """Content hash of the anchor set — a run's scores are only comparable to,
    and only reusable alongside, scores from the SAME anchors.

    Content-addressed rather than a hand-bumped version constant: editing an
    anchor's text without remembering to bump a number would silently mix two
    incompatible score populations in one parquet directory.
    """
    payload = json.dumps(
        [[row.get("anchor_id"), row.get("category"), row.get("polarity"), row.get("anchor_text")] for row in anchors],
        sort_keys=True, ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def score_parts(output_dir: Path) -> list[Path]:
    """Committed part files. `.partial` files are deliberately excluded: parts
    are written to a temp name and renamed, so a killed run leaves debris that
    is never mistaken for coverage."""
    return sorted(output_dir.glob(PART_GLOB))


def verify_parts(output_dir: Path, model_name: str, fingerprint: str, dtype: str = "fp32") -> dict:
    """Check every part on disk before trusting it as done work.

    A part counts as coverage only if it reads back, carries the key columns,
    and was produced by this model, these anchors AND this precision. dtype is
    part of the identity because fp16 and fp32 scores differ by ~5e-4: small,
    but enough that silently unioning both would leave a score population no
    single number describes. Anything that fails is reported and ignored
    rather than deleted — per CLAUDE.md, embedding output is never removed as
    a side effect of a later run.
    """
    usable: list[Path] = []
    unreadable: list[dict] = []
    mismatched: list[dict] = []
    con = duckdb.connect()
    try:
        for part in score_parts(output_dir):
            try:
                columns = {name for name, *_ in con.execute(f"DESCRIBE SELECT * FROM read_parquet('{part}')").fetchall()}
                # dtype is absent from parts written before it became part of
                # the identity; those were all fp32.
                dtype_expr = "any_value(dtype)" if "dtype" in columns else "'fp32'"
                row = con.execute(
                    f"SELECT count(*), any_value(model), any_value(anchors_fingerprint), {dtype_expr} "
                    f"FROM read_parquet('{part}')"
                ).fetchone()
            except (duckdb.Error, OSError) as error:
                unreadable.append({"path": str(part), "error": str(error).splitlines()[0]})
                continue
            missing = [column for column in PARAGRAPH_KEY if column not in columns]
            if missing or row[1] != model_name or row[2] != fingerprint or row[3] != dtype:
                mismatched.append({
                    "path": str(part), "rows": row[0], "model": row[1],
                    "anchors_fingerprint": row[2], "dtype": row[3], "missing_key_columns": missing,
                })
                continue
            usable.append(part)
    finally:
        con.close()
    return {"usable": usable, "unreadable": unreadable, "mismatched": mismatched}


def register_scored_keys(con, output_dir: Path, usable: list[Path], view: str = "prefilter_scored_keys") -> int:
    """Expose already-scored keys as a temp view for the pending anti-join."""
    columns = ", ".join(PARAGRAPH_KEY)
    if not usable:
        con.execute(f"CREATE OR REPLACE TEMP VIEW {view} AS SELECT {columns} FROM paragraphs WHERE false")
        return 0
    files = ", ".join(f"'{part}'" for part in usable)
    con.execute(
        f"CREATE OR REPLACE TEMP VIEW {view} AS "
        f"SELECT DISTINCT {columns} FROM read_parquet([{files}], union_by_name=True)"
    )
    return con.execute(f"SELECT count(*) FROM {view}").fetchone()[0]


def coverage_report(con, view: str = "prefilter_scored_keys") -> list[dict]:
    """Scored vs total per country/form — the additive picture, straight from SQL."""
    keys = " AND ".join(f"s.{column} IS NOT DISTINCT FROM p.{column}" for column in PARAGRAPH_KEY)
    return con.execute(f"""
        SELECT p.country_code, p.form, count(*) AS paragraphs,
               count(*) FILTER (WHERE EXISTS (SELECT 1 FROM {view} AS s WHERE {keys})) AS scored
        FROM paragraphs AS p
        GROUP BY 1, 2 ORDER BY 1, 2
    """).df().to_dict("records")


def _commit_part(table, output_dir: Path, run_id: str, part_index: int) -> Path:
    """Write one part atomically: a run killed mid-write leaves a .partial that
    the coverage glob ignores, never a truncated file that looks like coverage."""
    final = output_dir / f"prefilter_scores__run={run_id}__part={part_index:05d}.parquet"
    staging = final.with_suffix(".parquet.partial")
    writer = pq.ParquetWriter(staging, table.schema)
    try:
        writer.write_table(table)
    finally:
        writer.close()
    staging.replace(final)
    return final


def _write_benchmark(output_dir: Path, run_id: str, metadata: dict) -> Path:
    path = output_dir / f"prefilter_scores_manifest__run={run_id}.json"
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True, default=str) + "\n")
    return path


def ensure_anchor_table(database: Path) -> bool:
    """Persist the versioned anchors in DuckDB for SQL inspection/retrieval.

    Best-effort: this needs the write lock, and DuckDB refuses it while any
    other process holds the file (a concurrent reader, an open CLI). The
    anchors table is an inspection convenience — the run embeds from
    `anchor_rows()` either way — so a lock conflict warns instead of killing
    a multi-hour scoring run.
    """
    try:
        con = duckdb.connect(str(database))
    except duckdb.IOException as error:
        print(f"  anchors table not refreshed ({str(error).splitlines()[0]})", flush=True)
        return False
    try:
        register_anchors(con)
    finally:
        con.close()
    return True


def run_prefilter(
    database: Path,
    embeddings_dir: Path,
    output_dir: Path,
    model_name: str,
    device: str | None,
    dtype: str = "fp16",
    progress_seconds: float = 10.0,
    limit: int = 0,
    fetch_rows: int = 50_000,
    part_rows: int = 250_000,
    resume: bool = True,
    verify_only: bool = False,
) -> tuple[list[Path], Path]:
    """Score stored paragraph vectors against the anchors — no re-embedding.

    Stage 2 of the prefilter. The vectors come from ai_embed.py, so changing an
    anchor costs this pass (seconds) instead of a full re-embed (~35 min of GPU).
    Only the ~20 anchor texts touch the encoder here.

    Additive on both sides: rows already scored under this model/anchors/dtype
    are skipped, and paragraphs whose vector has not been written yet are simply
    reported as pending rather than silently dropped — so this can be run
    against a half-finished embedding pass and re-run later to pick up the rest.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    ensure_anchor_table(database)
    anchors = anchor_rows()
    fingerprint = anchors_fingerprint(anchors)

    audit = verify_parts(output_dir, model_name, fingerprint, dtype)
    usable = audit["usable"] if resume else []
    for bad in audit["unreadable"]:
        print(f"  UNREADABLE score part ignored: {bad['path']} ({bad['error']})", flush=True)
    for bad in audit["mismatched"]:
        print(f"  score part from another config ignored: {Path(bad['path']).name} "
              f"(model={bad['model']}, anchors={bad['anchors_fingerprint']}, dtype={bad['dtype']})", flush=True)

    embedding_audit = ai_embed.verify_parts(embeddings_dir, model_name, dtype)
    for bad in embedding_audit["unreadable"]:
        print(f"  UNREADABLE embedding part ignored: {bad['path']} ({bad['error']})", flush=True)
    for bad in embedding_audit["mismatched"]:
        print(f"  embedding part from another config ignored: {Path(bad['path']).name} "
              f"(model={bad['model']}, dtype={bad['dtype']})", flush=True)
    if not embedding_audit["usable"]:
        raise ValueError(
            f"no usable embedding parts in {embeddings_dir} for model={model_name} dtype={dtype}; "
            f"run scripts/common/ai_embed.py first")

    con = duckdb.connect(str(database), read_only=True)
    try:
        row_count = con.sql("SELECT count(*) FROM paragraphs").fetchone()[0]
        scored_keys = register_scored_keys(con, output_dir, usable)
        files = ", ".join(f"'{part}'" for part in embedding_audit["usable"])
        con.execute(f"CREATE OR REPLACE TEMP VIEW paragraph_vectors AS "
                    f"SELECT * FROM read_parquet([{files}], union_by_name=True)")
        vector_count = con.execute("SELECT count(*) FROM paragraph_vectors").fetchone()[0]

        key_join = " AND ".join(f"v.{c} IS NOT DISTINCT FROM p.{c}" for c in PARAGRAPH_KEY)
        scored_join = " AND ".join(f"s.{c} IS NOT DISTINCT FROM p.{c}" for c in PARAGRAPH_KEY)
        coverage = con.execute(f"""
            SELECT p.country_code, p.form, count(*) AS paragraphs,
                   count(*) FILTER (WHERE EXISTS (SELECT 1 FROM paragraph_vectors v WHERE {key_join})) AS embedded,
                   count(*) FILTER (WHERE EXISTS (SELECT 1 FROM prefilter_scored_keys s WHERE {scored_join})) AS scored
            FROM paragraphs AS p GROUP BY 1, 2 ORDER BY 1, 2
        """).df().to_dict("records")
        print(f"Anchors {fingerprint} | {len(embedding_audit['usable'])} embedding part(s), "
              f"{vector_count:,} vectors | {scored_keys:,} paragraphs already scored", flush=True)
        for row in coverage:
            print(f"  {row['country_code']}/{row['form']}: {int(row['embedded']):,}/{int(row['paragraphs']):,} "
                  f"embedded, {int(row['scored']):,} scored", flush=True)

        pending = con.execute(f"""
            SELECT count(*) FROM paragraphs p
            WHERE EXISTS (SELECT 1 FROM paragraph_vectors v WHERE {key_join})
              AND NOT EXISTS (SELECT 1 FROM prefilter_scored_keys s WHERE {scored_join})
        """).fetchone()[0]
        target_rows = min(limit, pending) if limit and limit > 0 else pending
        print(f"Scorable this run: {target_rows:,} paragraphs "
              f"({row_count - vector_count:,} still waiting on embeddings)", flush=True)

        manifest = {
            "run_id": run_id, "model": model_name, "anchors_fingerprint": fingerprint, "dtype": dtype,
            "corpus_paragraphs": row_count, "vectors_available": vector_count,
            "already_scored": scored_keys, "pending_at_start": pending, "coverage_before": coverage,
            "embedding_parts": [str(part) for part in embedding_audit["usable"]],
            "reused_parts": [str(part) for part in usable],
            "unreadable_parts": audit["unreadable"], "mismatched_parts": audit["mismatched"],
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        if verify_only or target_rows == 0:
            manifest["verify_only"] = True
            if target_rows == 0 and not verify_only:
                print("Nothing to score — every available vector already has a score.", flush=True)
            return [], _write_benchmark(output_dir, run_id, manifest)

        # Only the anchors go through the encoder: 20 short texts.
        device = resolve_device(device)
        model = _load_model(model_name, device, dtype)
        anchor_embeddings = _encode(model, [row["anchor_text"] for row in anchors], len(anchors))

        text = "lower(coalesce(p.paragraph_text, ''))"
        query = f"""
            SELECT p.form, p.country_code, p.accession_number, p.item_key, p.content_type,
                   p.paragraph_index,
                   list_filter({_sql_list(STRONG_AI_TERMS)}, term -> contains({text}, term)) AS strong_matched_terms,
                   list_filter({_sql_list(WEAK_AI_TERMS)}, term -> contains({text}, term)) AS weak_matched_terms,
                   list_count(list_filter({_sql_list(STRONG_AI_TERMS)}, term -> contains({text}, term))) > 0 AS strong_lexical_match,
                   list_count(list_filter({_sql_list(WEAK_AI_TERMS)}, term -> contains({text}, term))) > 0 AS weak_lexical_match,
                   v.embedding, v.text_hash
            FROM paragraphs AS p
            JOIN paragraph_vectors AS v ON {key_join}
            WHERE NOT EXISTS (SELECT 1 FROM prefilter_scored_keys s WHERE {scored_join})
            ORDER BY p.form, p.country_code, p.accession_number, p.item_key, p.paragraph_index
            {f'LIMIT {int(limit)}' if limit and limit > 0 else ''}
        """
        reader = con.execute(query).to_arrow_reader(fetch_rows)
        reporter = ThroughputReporter(target_rows, "scoring", progress_seconds, device)
        written: list[Path] = []
        buffered: list[pa.Table] = []
        buffered_rows = 0
        scored_rows = 0
        for batch in reader:
            # FixedSizeList/List<float> -> (rows, dim) without a per-row Python loop.
            flat = batch.column("embedding").flatten().to_numpy(zero_copy_only=False)
            vectors = flat.reshape(len(batch), -1).astype(np.float32)
            start = time.perf_counter()
            scores = score_embeddings(vectors, anchor_embeddings, anchors)
            elapsed = time.perf_counter() - start
            table = pa.Table.from_batches([batch]).drop_columns(["embedding"])
            for column, values in scores.items():
                table = table.append_column(column, pa.array(values))
            table = table.append_column("model", pa.array([model_name] * len(batch)).dictionary_encode())
            table = table.append_column("anchors_fingerprint", pa.array([fingerprint] * len(batch)).dictionary_encode())
            table = table.append_column("dtype", pa.array([dtype] * len(batch)).dictionary_encode())
            table = table.append_column("run_id", pa.array([run_id] * len(batch)).dictionary_encode())
            buffered.append(table)
            buffered_rows += len(batch)
            scored_rows += len(batch)
            reporter.update(len(batch), elapsed)
            if buffered_rows >= part_rows:
                written.append(_commit_part(pa.concat_tables(buffered), output_dir, run_id, len(written)))
                buffered, buffered_rows = [], 0
        if buffered:
            written.append(_commit_part(pa.concat_tables(buffered), output_dir, run_id, len(written)))
        reporter.close()
    finally:
        con.close()

    manifest.update({
        "device": str(model.device), "scored_rows": scored_rows,
        "parts_written": [str(part) for part in written],
        "wall_seconds": time.perf_counter() - reporter.started,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    })
    return written, _write_benchmark(output_dir, run_id, manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--embeddings-dir", type=Path, default=ai_embed.DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", dest="model_name", default=DEFAULT_MODEL)
    parser.add_argument("--device", default=None, help="cuda, mps, cpu (default: best available)")
    parser.add_argument("--dtype", choices=tuple(TORCH_DTYPES), default="fp16",
                        help="Must match the precision the vectors were embedded at")
    parser.add_argument("--progress-seconds", type=float, default=10.0)
    parser.add_argument("--limit", type=int, default=0, help="Score only the first N pending paragraphs")
    parser.add_argument("--fetch-rows", type=int, default=50_000)
    parser.add_argument("--part-rows", type=int, default=250_000, help="Rows per parquet part file")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    parts, manifest_path = run_prefilter(**vars(args))
    print(f"Parts written: {len(parts)}")
    print(f"Manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
