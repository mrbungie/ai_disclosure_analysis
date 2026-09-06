"""Puntúa la ORACIÓN que menciona IA, no el párrafo entero.

El prefiltro semántico embebe el párrafo completo, y ahí está su límite
estructural: en una matriz de habilidades del directorio de 14.350 caracteres,
"artificial intelligence" es una celda entre cientos, y el vector denso del
párrafo no se parece en nada al de una divulgación de IA. Es el mismo problema
que `docs/prefilter_evaluation.md` §9 anota como "contexto en el embedding",
visto desde el otro lado: no es que falte contexto, es que sobra.

Medido: sobre los 66.972 textos que pasan la compuerta léxica hay 226.139
oraciones, y sólo **30.792 contienen un término de IA**. El 86% de lo que el
embedding del párrafo promedia es ruido para esta decisión.

Este script embebe SÓLO las oraciones con término (con el mismo modelo, los
mismos anchors y la misma métrica que `ai_prefilter.py`) y agrega por texto:
el máximo y el promedio del margen semántico entre sus oraciones de IA, el
máximo por categoría, y cuántas oraciones de IA tiene. Esas columnas entran al
clasificador como señales adicionales — no reemplazan a las del párrafo, porque
las dos cosas son distintas: una dice "de qué habla este párrafo" y la otra "de
qué habla la parte que menciona IA".

Salida: `data/interim/prefilter_sentence_scores/`, una fila por `text_hash`.

Uso:
    uv run python scripts/common/ai_prefilter_sentences.py
    uv run python scripts/common/ai_prefilter_sentences.py --limit 2000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "common"))
from ai_prefilter import score_embeddings, anchors_fingerprint  # noqa: E402
from ai_prefilter_anchors import anchor_rows, strong_terms, weak_terms  # noqa: E402
from embedding_runtime import _load_model, resolve_device  # noqa: E402

DB = REPO_ROOT / "duckdb" / "thesis.duckdb"
OUT_DIR = REPO_ROOT / "data" / "interim" / "prefilter_sentence_scores"
SCORES_GLOB = "data/interim/prefilter_scores_unique/*.parquet"
DEFAULT_MODEL = "BAAI/bge-m3"
CATEGORIES = ("ai_use", "ai_exploration", "ai_capability", "ai_outcome",
              "ai_risk", "ai_governance", "ai_strategy")


def term_regex(terms) -> str:
    """Mismo criterio de límite de palabra que el prefiltro: `\\b` sobre
    minúsculas, con los espacios y puntos escapados para RE2."""
    alternation = "|".join(t.replace(".", "\\.").replace(" ", "[ ]") for t in terms)
    return f"\\b({alternation})\\b"


def load_sentences(con, limit: int = 0) -> pd.DataFrame:
    """Oraciones con término de IA de los textos que pasan la compuerta léxica.

    La compuerta se aplica al PÁRRAFO (es la población candidata del prefiltro)
    y el filtro de término a la ORACIÓN, que es lo que se va a embeber."""
    pattern = term_regex(strong_terms() + weak_terms())
    return con.execute(f"""
        WITH candidates AS (
            SELECT up.country_code, up.form, up.accession_number, up.item_key,
                   up.paragraph_index, up.text_hash
            FROM unique_paragraphs up
            JOIN (SELECT * FROM read_parquet('{SCORES_GLOB}')
                  QUALIFY row_number() OVER (PARTITION BY text_hash ORDER BY run_id DESC) = 1) s
              USING (text_hash)
            WHERE up.is_scorable AND (s.strong_lexical_match OR s.weak_lexical_match)
        )
        SELECT c.text_hash, sn.sentence_index, sn.sentence_text
        FROM candidates c
        JOIN sentences sn USING (country_code, form, accession_number, item_key, paragraph_index)
        WHERE regexp_matches(lower(sn.sentence_text), '{pattern}')
          AND length(sn.sentence_text) > 15
        ORDER BY c.text_hash, sn.sentence_index
        {f'LIMIT {int(limit)}' if limit else ''}
    """).df()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--database", type=Path, default=DB)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="fp16", choices=("fp16", "fp32"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-seq-length", type=int, default=256)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    con = duckdb.connect(str(args.database), read_only=True)
    try:
        sentences = load_sentences(con, args.limit)
    finally:
        con.close()
    print(f"{len(sentences):,} oraciones con término de IA en "
          f"{sentences['text_hash'].nunique():,} textos candidatos")

    anchors = anchor_rows()
    device = resolve_device(args.device)
    model = _load_model(args.model, device, args.dtype)
    # bge-m3 acepta 8.192 tokens y reserva memoria para eso; una oración no los
    # necesita ni de lejos, y con el largo por defecto un batch de 256 se lleva
    # la GPU entera. Truncar a `--max-seq-length` es lo que hace viable correr
    # esto en minutos.
    model.max_seq_length = args.max_seq_length
    print(f"modelo {args.model} en {device} ({args.dtype}), "
          f"max_seq_length={model.max_seq_length}")

    anchor_vectors = model.encode([a["anchor_text"] for a in anchors],
                                  batch_size=32, normalize_embeddings=True,
                                  show_progress_bar=False)
    vectors = model.encode(sentences["sentence_text"].tolist(),
                           batch_size=args.batch_size, normalize_embeddings=True,
                           show_progress_bar=True)
    scores = score_embeddings(np.asarray(vectors), np.asarray(anchor_vectors), anchors)

    frame = sentences[["text_hash"]].copy()
    for name, values in scores.items():
        if name == "best_semantic_anchor":
            continue
        frame[name] = values
    # Agregación por texto: el MÁXIMO es lo que importa — basta con que UNA
    # oración hable de IA en serio para que el párrafo sea candidato. El
    # promedio se guarda igual porque distingue "una oración buena entre
    # veinte" de "todas hablan de lo mismo".
    grouped = frame.groupby("text_hash")
    aggregated = grouped.agg(
        sent_n_ai=("semantic_margin", "size"),
        sent_max_margin=("semantic_margin", "max"),
        sent_mean_margin=("semantic_margin", "mean"),
        sent_max_semantic=("max_semantic_score", "max"),
        sent_min_negative=("negative_similarity", "min"),
        **{f"sent_max_{category}": (f"score_{category}", "max") for category in CATEGORIES},
    ).reset_index()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    fingerprint = anchors_fingerprint(anchors)
    aggregated["model"] = args.model
    aggregated["anchors_fingerprint"] = fingerprint
    aggregated["run_id"] = run_id
    destination = args.output_dir / f"prefilter_sentence_scores__run={run_id}.parquet"
    pq.write_table(pa.Table.from_pandas(aggregated, preserve_index=False), destination,
                   compression="zstd")
    manifest = args.output_dir / f"prefilter_sentence_scores_manifest__run={run_id}.json"
    manifest.write_text(json.dumps({
        "run_id": run_id, "model": args.model, "dtype": args.dtype, "device": device,
        "anchors_fingerprint": fingerprint, "sentences": int(len(sentences)),
        "texts": int(len(aggregated)),
        "created_at": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n")
    print(f"-> {destination} ({len(aggregated):,} textos)")
    print(f"-> {manifest}")


if __name__ == "__main__":
    main()
