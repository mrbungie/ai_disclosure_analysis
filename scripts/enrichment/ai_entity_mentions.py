"""Tags AI-relevant paragraphs with which named AI entity they mention
(product, geography, modality) — see configs/ai_prefilter.yaml's
`lexical.entities` and docs/prefilter_evaluation.md §8.10/§8.12.

No LLM here: literal word-boundary regex matching, the SAME technique
ai_prefilter.py already uses for lexical scoring — this just also carries
the geo/modality metadata that scoring throws away (a paragraph either
matched `strong_lexical_match` or it didn't; this records WHICH entity,
and where it's from).

COUNTRY-AGNOSTIC BY DESIGN (explicit requirement, 2026-09-04): the
population is "whatever the deployed prefilter flagged
`is_ai_prefiltered=True`", read from bronze.prefilter_predictions
without filtering by `country_code` here.

Output is one row per (text_hash, matched entity term) — one row per
UNIQUE TEXT, not per paragraph instance, same "dedup as a phase"
discipline as the rest of this pipeline (docs/prefilter_evaluation.md
§8.8). silver.ai_entity_mentions is what broadcasts this back out to every
real paragraph instance via `paragraphs.text_hash`.

Cheap (regex over ~13k paragraphs, not the whole corpus, no GPU/LLM), so
this recomputes from scratch every run rather than tracking an additive
checkpoint the way ai_embed.py/ai_classify.py do for their much more
expensive steps.

Usage:
    uv run python scripts/enrichment/ai_entity_mentions.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from ai_prefilter_anchors import entity_terms
from ai_prefilter_classify import lower_text

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "common"))
import layers as L  # noqa: E402

OUT_DIR = REPO_ROOT / "data" / "interim" / "ai_entity_mentions"

BOUNDARY_BEFORE = "(^|[^a-z0-9])"
BOUNDARY_AFTER = "([^a-z0-9]|$)"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    terms = entity_terms()
    print(f"{len(terms)} términos de entidad registrados", flush=True)

    # Una predicción por text_hash (no por llave de instancia): identidad
    # estable del contenido, no de qué instancia una corrida particular eligió
    # como representante.
    ai_relevant = (L.scan("bronze.prefilter_predictions").filter(pl.col("is_ai_prefiltered"))
                   .select("text_hash")
                   .join(L.scan("bronze.unique_paragraphs").select("text_hash", "duplicate_count",
                                                                  "paragraph_text"), on="text_hash")
                   .with_columns(lower_text(pl.col("paragraph_text")).alias("lowered"))
                   .collect())
    n = ai_relevant.height
    print(f"{n:,} textos únicos IA-relevantes a revisar", flush=True)

    # Un término por vez sobre ~40k textos: una fila por (texto, término) que matchea.
    matches = []
    for term, meta in terms.items():
        hit = ai_relevant.filter(pl.col("lowered").str.contains(f"{BOUNDARY_BEFORE}{term}{BOUNDARY_AFTER}"))
        if hit.height:
            matches.append(hit.select("text_hash", "duplicate_count", pl.lit(term).alias("term"),
                                      pl.lit(meta["geo"]).alias("geo"),
                                      pl.lit(meta["modality"]).alias("modality")))
    mentions = (pl.concat(matches) if matches else pl.DataFrame(
        schema={"text_hash": pl.UInt64, "duplicate_count": pl.Int64, "term": pl.String,
                "geo": pl.String, "modality": pl.String}))
    df = mentions.sort("text_hash", "term").to_pandas()
    print(f"{len(df):,} menciones encontradas ({df['text_hash'].nunique():,} textos con al menos una)",
          flush=True)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    df["run_id"] = run_id
    out_path = OUT_DIR / f"ai_entity_mentions__run={run_id}.parquet"
    staging = out_path.with_suffix(".parquet.partial")
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), staging, compression="zstd")
    staging.replace(out_path)

    manifest = {
        "run_id": run_id, "entity_terms": len(terms), "output": str(out_path),
        "rows": int(len(df)), "created_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = OUT_DIR / f"ai_entity_mentions_manifest__run={run_id}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    print(f"\nParquet -> {out_path}")
    print(f"Manifiesto -> {manifest_path}")
    print("Distribución por geografía/modalidad: "
          "scripts/analytics/posture/entity_mentions_summary.py")


if __name__ == "__main__":
    main()
