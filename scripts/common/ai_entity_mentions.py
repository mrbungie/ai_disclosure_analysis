"""Tags AI-relevant paragraphs with which named AI entity they mention
(product, geography, modality) — see configs/ai_prefilter.yaml's
`lexical.entities` and docs/prefilter_evaluation.md §8.10/§8.12.

No LLM here: literal word-boundary regex matching, the SAME technique
ai_prefilter.py already uses for lexical scoring — this just also carries
the geo/modality metadata that scoring throws away (a paragraph either
matched `strong_lexical_match` or it didn't; this records WHICH entity,
and where it's from).

COUNTRY-AGNOSTIC BY DESIGN (explicit requirement, 2026-09-04): the
population is "whatever the latest prefilter run flagged
`is_ai_prefiltered=True`", read from `prefilter_predictions_unique`
without ever filtering by `country_code`. Chile (or any later country)
starts showing up here automatically the moment its own paragraphs enter
that population — no code change, no re-run of this script's logic,
just a re-run of the script itself once that data exists.

Output is one row per (text_hash, matched entity term) — one row per
UNIQUE TEXT, not per paragraph instance, same "dedup as a phase"
discipline as the rest of this pipeline (docs/prefilter_evaluation.md
§8.8). `gold_ai_entity_mentions` (build_duckdb.py) is what broadcasts this
back out to every real paragraph instance via `paragraphs.text_hash`.

Cheap (regex over ~13k paragraphs, not the whole corpus, no GPU/LLM), so
this recomputes from scratch every run rather than tracking an additive
checkpoint the way ai_embed.py/ai_classify.py do for their much more
expensive steps.

Usage:
    uv run python scripts/common/ai_entity_mentions.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from ai_prefilter_anchors import register_entity_terms

REPO_ROOT = Path(__file__).resolve().parents[2]
DB = REPO_ROOT / "duckdb" / "thesis.duckdb"
OUT_DIR = REPO_ROOT / "data" / "interim" / "ai_entity_mentions"
PREDICTIONS_GLOB = "data/interim/prefilter_predictions_unique/prefilter_predictions__run=*.parquet"

BOUNDARY_BEFORE = "(^|[^a-z0-9])"
BOUNDARY_AFTER = "([^a-z0-9]|$)"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB), read_only=False)
    try:
        terms = register_entity_terms(con)
        print(f"{len(terms)} términos de entidad registrados", flush=True)

        # QUALIFY por text_hash (no por llave de instancia) -- misma razón
        # que en ai_classify.py: identidad estable del contenido, no de qué
        # instancia una corrida particular eligió como representante.
        con.execute(f"""
            CREATE OR REPLACE TEMP VIEW latest_predictions AS
            SELECT * FROM read_parquet('{PREDICTIONS_GLOB}', union_by_name=True)
            QUALIFY row_number() OVER (PARTITION BY text_hash ORDER BY model_version DESC) = 1
        """)
        con.execute("""
            CREATE OR REPLACE TEMP VIEW ai_relevant AS
            SELECT p.text_hash, p.duplicate_count, p.paragraph_text
            FROM latest_predictions lp
            JOIN unique_paragraphs p USING (text_hash)
            WHERE lp.is_ai_prefiltered
        """)
        n = con.execute("SELECT COUNT(*) FROM ai_relevant").fetchone()[0]
        print(f"{n:,} textos únicos IA-relevantes a revisar", flush=True)

        df = con.execute(f"""
            SELECT r.text_hash, r.duplicate_count, e.term, e.geo, e.modality
            FROM ai_relevant r
            JOIN ai_prefilter_entity_terms e
              ON regexp_matches(lower(r.paragraph_text), '{BOUNDARY_BEFORE}' || e.term || '{BOUNDARY_AFTER}')
        """).df()
        print(f"{len(df):,} menciones encontradas ({df['text_hash'].nunique():,} textos con al menos una)",
              flush=True)
    finally:
        con.close()

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    df["run_id"] = run_id
    out_path = OUT_DIR / f"ai_entity_mentions__run={run_id}.parquet"
    staging = out_path.with_suffix(".parquet.partial")
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), staging, compression="zstd")
    staging.replace(out_path)

    manifest = {
        "run_id": run_id, "entity_terms": len(terms), "ai_relevant_texts": int(n),
        "mentions_found": int(len(df)), "texts_with_mention": int(df["text_hash"].nunique()),
        "by_geo": df.groupby("geo")["text_hash"].nunique().to_dict(),
        "by_modality": df.groupby("modality")["text_hash"].nunique().to_dict(),
        "output": str(out_path), "created_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = OUT_DIR / f"ai_entity_mentions_manifest__run={run_id}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    print(f"\nParquet -> {out_path}")
    print(f"Manifiesto -> {manifest_path}")
    print("\nPor geografía (textos únicos):")
    for geo, count in sorted(manifest["by_geo"].items(), key=lambda kv: -kv[1]):
        print(f"  {geo}: {count:,}")
    print("\nPor modalidad (textos únicos):")
    for modality, count in sorted(manifest["by_modality"].items(), key=lambda kv: -kv[1]):
        print(f"  {modality}: {count:,}")


if __name__ == "__main__":
    main()
