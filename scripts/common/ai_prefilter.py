"""Terminal, score-only lexical and semantic AI prefilter for paragraphs."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
from sentence_transformers import SentenceTransformer

from ai_prefilter_anchors import POSITIVE_ANCHORS, anchor_rows, register_anchors

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


def lexical_input_sql(source_relation: str = "paragraphs") -> str:
    """Return paragraph lineage/text plus lexical features, all computed in DuckDB."""
    text = "lower(coalesce(p.paragraph_text, ''))"
    return f"""
        SELECT p.*, 
               list_filter({_sql_list(STRONG_AI_TERMS)}, term -> contains({text}, term)) AS strong_matched_terms,
               list_filter({_sql_list(WEAK_AI_TERMS)}, term -> contains({text}, term)) AS weak_matched_terms,
               list_count(list_filter({_sql_list(STRONG_AI_TERMS)}, term -> contains({text}, term))) > 0 AS strong_lexical_match,
               list_count(list_filter({_sql_list(WEAK_AI_TERMS)}, term -> contains({text}, term))) > 0 AS weak_lexical_match
        FROM {source_relation} AS p
        ORDER BY p.form, p.country_code, p.accession_number, p.item_key, p.paragraph_index
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


def _load_model(model_name: str, device: str | None) -> SentenceTransformer:
    return SentenceTransformer(model_name, device=device)


def _encode(model: SentenceTransformer, texts: list[str], batch_size: int) -> np.ndarray:
    return model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float32, copy=False)


def _write_benchmark(output_dir: Path, run_id: str, metadata: dict) -> Path:
    path = output_dir / f"prefilter_embedding_benchmark__run={run_id}.json"
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return path


def ensure_anchor_table(database: Path) -> None:
    """Persist the versioned anchors in DuckDB for SQL inspection/retrieval."""
    con = duckdb.connect(str(database))
    try:
        register_anchors(con)
    finally:
        con.close()


def run_prefilter(database: Path, output_dir: Path, model_name: str, batch_size: int, device: str | None, benchmark_rows: int) -> tuple[Path, Path]:
    """Score every materialized paragraph, preserving all scores and no thresholds."""
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    ensure_anchor_table(database)
    con = duckdb.connect(str(database), read_only=True)
    try:
        row_count = con.sql("SELECT count(*) FROM paragraphs").fetchone()[0]
        if row_count == 0:
            raise ValueError("paragraphs is empty; build it with `make duckdb-text` first")
        model = _load_model(model_name, device)
        anchors = anchor_rows()
        anchor_embeddings = _encode(model, [row["anchor_text"] for row in anchors], batch_size)

        sample = con.sql(
            f"SELECT paragraph_text FROM paragraphs LIMIT {max(1, min(benchmark_rows, row_count))}"
        ).fetchall()
        sample_texts = [row[0] for row in sample]
        _encode(model, sample_texts[:min(batch_size, len(sample_texts))], batch_size)  # warm-up
        benchmark_start = time.perf_counter()
        _encode(model, sample_texts, batch_size)
        benchmark_seconds = time.perf_counter() - benchmark_start

        output_path = output_dir / f"prefilter_scores__run={run_id}.parquet"
        query = lexical_input_sql("paragraphs")
        result = con.execute(query)
        writer = None
        embedded_rows = 0
        embedding_seconds = 0.0
        while True:
            batch = result.fetch_df_chunk(vectors_per_chunk=batch_size)
            if batch.empty:
                break
            texts = batch.pop("paragraph_text").fillna("").tolist()
            start = time.perf_counter()
            embeddings = _encode(model, texts, batch_size)
            embedding_seconds += time.perf_counter() - start
            scores = score_embeddings(embeddings, anchor_embeddings, anchors)
            for column, values in scores.items():
                batch[column] = values
            embedded_rows += len(batch)
            table = __import__("pyarrow").Table.from_pandas(batch, preserve_index=False)
            if writer is None:
                writer = __import__("pyarrow.parquet", fromlist=["ParquetWriter"]).ParquetWriter(output_path, table.schema)
            writer.write_table(table)
        if writer is not None:
            writer.close()
    finally:
        con.close()

    benchmark = {
        "run_id": run_id,
        "model": model_name,
        "device": device or str(model.device),
        "batch_size": batch_size,
        "benchmark_rows": len(sample_texts),
        "benchmark_seconds": benchmark_seconds,
        "benchmark_paragraphs_per_second": len(sample_texts) / benchmark_seconds if benchmark_seconds else None,
        "scored_rows": embedded_rows,
        "scoring_embedding_seconds": embedding_seconds,
        "scoring_embedding_paragraphs_per_second": embedded_rows / embedding_seconds if embedding_seconds else None,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    benchmark_path = _write_benchmark(output_dir, run_id, benchmark)
    return output_path, benchmark_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None, help="SentenceTransformers device, e.g. mps, cuda, or cpu")
    parser.add_argument("--benchmark-rows", type=int, default=1_000)
    args = parser.parse_args()
    if args.batch_size < 1 or args.benchmark_rows < 1:
        parser.error("--batch-size and --benchmark-rows must be positive")
    output_path, benchmark_path = run_prefilter(**vars(args))
    print(f"Scores -> {output_path}")
    print(f"Embedding benchmark -> {benchmark_path}")


if __name__ == "__main__":
    main()
