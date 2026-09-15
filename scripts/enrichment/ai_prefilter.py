"""Terminal, score-only lexical and semantic AI prefilter for paragraphs."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from ai_prefilter_anchors import POSITIVE_ANCHORS, anchor_rows, strong_terms, weak_terms
from ai_prefilter_classify import lower_text
import ai_embed
from embedding_runtime import (  # maquinaria compartida con ai_embed.py
    DEFAULT_MEMORY_FRACTION, PARAGRAPH_KEY, TORCH_DTYPES, ThroughputReporter,
    _encode, _gib, _load_model, resolve_device,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "common"))
import layers as L  # noqa: E402

DEFAULT_MODEL = "BAAI/bge-m3"
# --source-relation -> tabla de capas.
SOURCE_TABLES = {"paragraphs": "bronze.paragraphs", "unique_paragraphs": "bronze.unique_paragraphs"}
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "interim" / "prefilter_scores"

# Lexical terms come from configs/ai_prefilter.yaml alongside the anchors, so
# tuning either signal is a config change, not a code change.
STRONG_AI_TERMS = strong_terms()
WEAK_AI_TERMS = weak_terms()


# Límite de palabra: un término sólo cuenta si no viene pegado a otra letra o
# dígito. Es lo único que separa "ai" de "said", "chair" y "remain" — y con el
# límite puesto, distinguir mayúsculas ya no aporta (medido: `AI` sensible da
# 11.963 párrafos, `ai` insensible 11.992, 29 de diferencia en 3 millones).
# Sin esto la lista se perdía la mitad del corpus: 8.347 párrafos capturados
# contra 16.608 reales, porque las empresas escriben la sigla y no el término
# largo.
BOUNDARY_BEFORE = "(^|[^a-z0-9])"
BOUNDARY_AFTER = "([^a-z0-9]|$)"


def _term_regex(terms: tuple[str, ...]) -> str:
    """Alternación con límite de palabra, para el booleano."""
    return f"{BOUNDARY_BEFORE}({'|'.join(terms)}){BOUNDARY_AFTER}"


def _matched_terms(terms: tuple[str, ...], lowered: pl.Expr) -> pl.Expr:
    """Qué términos matchearon (en el orden de la config), con el mismo límite
    de palabra que el booleano."""
    return pl.concat_list([
        pl.when(lowered.str.contains(f"{BOUNDARY_BEFORE}{term}{BOUNDARY_AFTER}")).then(pl.lit(term))
        for term in terms]).list.drop_nulls()


def lexical_columns(text: pl.Expr = pl.col("paragraph_text")) -> list[pl.Expr]:
    """Términos matcheados y booleanos léxicos fuerte/débil de `text`."""
    lowered = lower_text(text)
    return [
        _matched_terms(STRONG_AI_TERMS, lowered).alias("strong_matched_terms"),
        _matched_terms(WEAK_AI_TERMS, lowered).alias("weak_matched_terms"),
        lowered.str.contains(_term_regex(STRONG_AI_TERMS)).alias("strong_lexical_match"),
        lowered.str.contains(_term_regex(WEAK_AI_TERMS)).alias("weak_lexical_match"),
    ]


def lexical_scores(paragraphs: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
    """paragraph_index + señales léxicas de un frame con `paragraph_text`,
    ordenado por paragraph_index."""
    return (paragraphs.lazy().select("paragraph_index", *lexical_columns())
            .sort("paragraph_index").collect())


PART_GLOB = "prefilter_scores__run=*.parquet"
# Leading columns of a score part, before the score columns and config tags.
PART_KEY_SCHEMA = pa.schema([
    ("form", pa.string()), ("country_code", pa.string()), ("accession_number", pa.string()),
    ("item_key", pa.string()), ("content_type", pa.string()), ("paragraph_index", pa.int64()),
    ("strong_matched_terms", pa.list_(pa.field("element", pa.string()))),
    ("weak_matched_terms", pa.list_(pa.field("element", pa.string()))),
    ("strong_lexical_match", pa.bool_()), ("weak_lexical_match", pa.bool_()),
    ("text_hash", pa.uint64()),
])


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
    # Margen sobre los anchors negativos. Medido contra el golden set, rankear
    # por este margen en vez de por la similitud cruda sube el F1 del score
    # semántico de 0.493 a 0.608: los negativos absorben el boilerplate que
    # eleva el piso de todo el corpus.
    result["semantic_margin"] = result["max_semantic_score"] - result["negative_similarity"]
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
        {
            "anchors": [[row.get("anchor_id"), row.get("category"), row.get("polarity"),
                         row.get("anchor_text")] for row in anchors],
            # Lexical terms are in the hash too: they decide the *_matched_terms and
            # *_lexical_match columns, so a run with different terms is a different
            # score population even when the anchors are untouched.
            "strong_terms": list(STRONG_AI_TERMS),
            "weak_terms": list(WEAK_AI_TERMS),
        },
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
    for part in score_parts(output_dir):
        try:
            columns = set(pl.read_parquet_schema(part))

            def any_value(column: str, missing=None) -> pl.Expr:
                if column not in columns:
                    return pl.lit(missing, pl.String)
                return pl.col(column).cast(pl.String).drop_nulls().first()
            # dtype is absent from parts written before it became part of
            # the identity; those were all fp32.
            row = pl.scan_parquet(part).select(
                pl.len(), any_value("model"), any_value("anchors_fingerprint"),
                any_value("dtype", "fp32")).collect().row(0)
        except (pl.exceptions.PolarsError, OSError) as error:
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
    return {"usable": usable, "unreadable": unreadable, "mismatched": mismatched}


def scored_keys(usable: list[Path]) -> pl.DataFrame:
    """Already-scored paragraph keys (distinct), for the pending anti-join."""
    if not usable:
        return pl.DataFrame(schema={"country_code": pl.String, "form": pl.String,
                                    "accession_number": pl.String, "item_key": pl.String,
                                    "paragraph_index": pl.Int64})
    return (pl.concat([pl.scan_parquet(part).select(PARAGRAPH_KEY) for part in usable],
                      how="diagonal_relaxed")
            .with_columns(pl.col(c).cast(pl.String) for c in PARAGRAPH_KEY if c != "paragraph_index")
            .unique().collect())


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


def run_prefilter(
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
    source_relation: str = "paragraphs",
) -> tuple[list[Path], Path]:
    """Score stored paragraph vectors against the anchors — no re-embedding.

    Stage 2 of the prefilter. The vectors come from ai_embed.py, so changing an
    anchor costs this pass (seconds) instead of a full re-embed (~35 min of GPU).
    Only the ~20 anchor texts touch the encoder here.

    Additive on both sides: rows already scored under this model/anchors/dtype
    are skipped, and paragraphs whose vector has not been written yet are simply
    reported as pending rather than silently dropped — so this can be run
    against a half-finished embedding pass and re-run later to pick up the rest.

    `source_relation` (docs/prefilter_evaluation.md §8.7/§8.8): pass
    "unique_paragraphs" (bronze.unique_paragraphs) instead of the default "paragraphs"
    to score each unique paragraph TEXT exactly once instead of once per
    corpus instance — ~50% of the corpus is literal boilerplate repeated
    across filings, so scoring `paragraphs` directly duplicates work for no
    reason once a real dedup surface exists. Embeddings need no rechange:
    `unique_paragraphs`' representative key already has a vector from
    ai_embed.py (it embedded every instance, representative included), this
    just skips scoring every OTHER instance sharing that same text.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
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
            f"run scripts/enrichment/ai_embed.py first")

    keys = list(PARAGRAPH_KEY)
    source = L.scan(SOURCE_TABLES[source_relation])
    row_count = source.select(pl.len()).collect().item()
    done_keys = scored_keys(usable)
    vector_keys = (pl.concat([pl.scan_parquet(part).select(*keys, "text_hash")
                              for part in embedding_audit["usable"]], how="diagonal_relaxed")
                   .with_columns(pl.col(c).cast(pl.String) for c in keys if c != "paragraph_index")
                   .collect())
    vector_count = vector_keys.height
    source_keys = source.select(keys).collect()
    embedded_flag = vector_keys.select(keys).unique().with_columns(pl.lit(True).alias("embedded"))
    scored_flag = done_keys.with_columns(pl.lit(True).alias("scored"))
    flagged = (source_keys.join(embedded_flag, on=keys, how="left", nulls_equal=True)
               .join(scored_flag, on=keys, how="left", nulls_equal=True))
    coverage = (flagged.group_by("country_code", "form")
                .agg(pl.len().alias("paragraphs"), pl.col("embedded").sum().alias("embedded"),
                     pl.col("scored").sum().alias("scored"))
                .sort("country_code", "form").to_dicts())
    print(f"Anchors {fingerprint} | {len(embedding_audit['usable'])} embedding part(s), "
          f"{vector_count:,} vectors | {done_keys.height:,} paragraphs already scored", flush=True)
    for row in coverage:
        print(f"  {row['country_code']}/{row['form']}: {int(row['embedded']):,}/{int(row['paragraphs']):,} "
              f"embedded, {int(row['scored']):,} scored", flush=True)

    pending = flagged.filter(pl.col("embedded").fill_null(False) & pl.col("scored").is_null()).height
    target_rows = min(limit, pending) if limit and limit > 0 else pending
    print(f"Scorable this run: {target_rows:,} paragraphs "
          f"({row_count - vector_count:,} still waiting on embeddings)", flush=True)

    manifest = {
        "run_id": run_id, "model": model_name, "anchors_fingerprint": fingerprint, "dtype": dtype,
        "corpus_paragraphs": row_count, "vectors_available": vector_count,
        "already_scored": done_keys.height, "pending_at_start": pending, "coverage_before": coverage,
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
    del flagged, source_keys, embedded_flag

    # Only the anchors go through the encoder: 20 short texts.
    device = resolve_device(device)
    model = _load_model(model_name, device, dtype)
    anchor_embeddings = _encode(model, [row["anchor_text"] for row in anchors], len(anchors))

    # Pending rows (embedded, not yet scored) with their lexical signals, in key
    # order; `limit` keeps the first N. The vectors are then read part by part,
    # so no more than one embedding part plus its scores sits in memory.
    work = (source.join(vector_keys.lazy().select(keys).unique(), on=keys, how="semi", nulls_equal=True)
            .join(done_keys.lazy(), on=keys, how="anti", nulls_equal=True)
            .select("form", "country_code", "accession_number", "item_key", "content_type",
                    "paragraph_index", *lexical_columns())
            .sort("form", "country_code", "accession_number", "item_key", "paragraph_index"))
    if limit and limit > 0:
        work = work.head(int(limit))
    work = work.collect()
    del vector_keys, done_keys

    reporter = ThroughputReporter(target_rows, "scoring", progress_seconds, device)
    written: list[Path] = []
    buffered: list[pa.Table] = []
    buffered_rows = 0
    scored_rows = 0
    for part in embedding_audit["usable"]:
        joined = (pl.scan_parquet(part).select(*keys, "embedding", "text_hash")
                  .with_columns(pl.col(c).cast(pl.String) for c in keys if c != "paragraph_index")
                  .join(work.lazy(), on=keys, how="inner", nulls_equal=True)
                  .select("form", "country_code", "accession_number", "item_key", "content_type",
                          "paragraph_index", "strong_matched_terms", "weak_matched_terms",
                          "strong_lexical_match", "weak_lexical_match", "embedding", "text_hash")
                  .collect())
        for offset in range(0, joined.height, fetch_rows):
            batch = joined.slice(offset, fetch_rows)
            vectors = batch["embedding"].to_numpy().reshape(batch.height, -1).astype(np.float32)
            start = time.perf_counter()
            scores = score_embeddings(vectors, anchor_embeddings, anchors)
            elapsed = time.perf_counter() - start
            table = batch.drop("embedding").to_arrow(compat_level=pl.CompatLevel.oldest()).cast(PART_KEY_SCHEMA)
            for column, values in scores.items():
                table = table.append_column(column, pa.array(values))
            table = table.append_column("model", pa.array([model_name] * batch.height).dictionary_encode())
            table = table.append_column("anchors_fingerprint", pa.array([fingerprint] * batch.height).dictionary_encode())
            table = table.append_column("dtype", pa.array([dtype] * batch.height).dictionary_encode())
            table = table.append_column("run_id", pa.array([run_id] * batch.height).dictionary_encode())
            buffered.append(table)
            buffered_rows += batch.height
            scored_rows += batch.height
            reporter.update(batch.height, elapsed)
            if buffered_rows >= part_rows:
                written.append(_commit_part(pa.concat_tables(buffered), output_dir, run_id, len(written)))
                buffered, buffered_rows = [], 0
    if buffered:
        written.append(_commit_part(pa.concat_tables(buffered), output_dir, run_id, len(written)))
    reporter.close()

    manifest.update({
        "device": str(model.device), "scored_rows": scored_rows,
        "parts_written": [str(part) for part in written],
        "wall_seconds": time.perf_counter() - reporter.started,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    })
    return written, _write_benchmark(output_dir, run_id, manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--source-relation", choices=tuple(SOURCE_TABLES), default="paragraphs",
                        help="'unique_paragraphs' scores each unique paragraph text once "
                             "instead of once per corpus instance (see docs/prefilter_evaluation.md §8.8)")
    args = parser.parse_args()
    parts, manifest_path = run_prefilter(**vars(args))
    print(f"Parts written: {len(parts)}")
    print(f"Manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
