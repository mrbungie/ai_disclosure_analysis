"""Arma `data.json` para la UI de validación humana (`index.html`).

Dos tareas, una muestra cada una, todo desde `data/bronze`, `data/silver` y
los parquets del prefiltro (vía `scripts/common/layers.py`) — sin LLM, sin red:

  frames     N párrafos con frames extraídos: el texto numerado por oración
             (como lo vio el juez) y cada frame con sus etiquetas. La persona
             valida frame por frame: ¿existe?, ¿promocional?, ¿temporal?,
             ¿specificity? — las dimensiones que sostienen 09/11/12/13.
  activities N párrafos con TODAS sus actividades divulgadas
             (`ai_activities_from_frames.py`), cada una escrita como frase ("la
             empresa despliega copilot para desarrollo de software, empleados,
             escalado, proveedores OpenAI y Azure") sobre el párrafo con la
             evidencia resaltada. La persona dice, por actividad, si está bien
             o qué campo está mal, y si al párrafo le falta alguna actividad. Sostiene `09` y lo que `02`, `03`,
             `05`, `06` y `08` toman de ahí.
  prefilter  N párrafos con la decisión del prefiltro v2, estratificados por
             probabilidad (positivos seguros, zona gris, negativos con
             término de IA). La persona dice si menciona IA. Sirve para
             medir recall/precisión humanos donde el holdout de juez no llega.

La muestra es determinística (semilla) y estratificada por formulario para
que las calls y los proxies no queden sub-representados.

    uv run --frozen --no-sync python ui-validator/build_sample.py --frames 300 --prefilter 300
    cd ui-validator && python -m http.server 8765      # abrir http://localhost:8765
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "common"))
import layers as L  # noqa: E402

PRED_DIR = L.ADDITIVE_SOURCES["prefilter_predictions_unique"]["path"]
CALLS_MANIFEST = REPO_ROOT / "data" / "interim" / "manifests" / "filing_manifest_earnings_calls.parquet"
ACTIVITIES_PATH = REPO_ROOT / "data" / "processed" / "clusters" / "firm_activities.parquet"
OUT = Path(__file__).resolve().parent / "data.json"
FORMS = ["10-K", "10-Q", "DEF 14A", "8-K", "Earnings call"]
PARAGRAPH_KEY = ["country_code", "form", "accession_number", "item_key", "paragraph_index"]

# `specificity`/`rhetoric` en `silver.ai_frames` son listas de tags cortos
# (una fila por frame puede llevar varios), no columnas booleanas separadas.
# La UI (`index.html`, fuera de esta migración) espera las claves largas de
# abajo, así que se traducen acá para no tocarla.
FRAME_SPECIFICITY = {"business_process": "process", "product_or_system": "product",
                     "vendor_or_partner": "vendor", "quantified_metric": "metric",
                     "date_or_timeline": "timeline"}


def latest_predictions() -> Path:
    runs = sorted(PRED_DIR.glob("prefilter_predictions__run=*.parquet"))
    if not runs:
        raise SystemExit(f"no hay predicciones del prefiltro en {PRED_DIR}")
    return runs[-1]


def _seeded_order(values: pl.Series, salt: str) -> pl.Series:
    """Orden pseudoaleatorio reproducible: BLAKE2b de 8 bytes de `valor|salt`.

    Reemplaza el `hash(text_hash + seed)` de DuckDB (retirado del proyecto)
    usado para elegir, dentro de cada estrato, qué filas entran a la muestra
    top-k. Mismo patrón que `scripts/enrichment/golden_set.py::_seeded_order`.
    El algoritmo de desempate cambia (BLAKE2b en vez del hash interno de
    DuckDB), pero el POOL de filas elegibles antes del top-k es el mismo:
    depende sólo de los filtros, no del algoritmo de orden — igual que
    documentado para `golden_set.py sample` (docs/tasks/README.md).
    """
    return pl.Series([int.from_bytes(hashlib.blake2b(f"{v}|{salt}".encode(), digest_size=8).digest(),
                                     "big", signed=False) for v in values.to_list()], dtype=pl.UInt64)


def register_firm_lookup() -> dict[str, dict]:
    """accession_number -> {ticker, company}, para que la persona sepa de
    quién es el párrafo (si dice "AMD" y la empresa es AMD, no es un
    proveedor externo)."""
    fm = (L.scan("bronze.filing_manifest").filter(pl.col("country_code") == "us")
          .select("accession_number", "ticker"))
    fmq = (L.scan("bronze.filing_manifest_10q").filter(pl.col("country_code") == "us")
           .select("accession_number", "ticker"))
    calls = (pl.scan_parquet(CALLS_MANIFEST)
             .select(pl.col("document_id").alias("accession_number"), "ticker"))
    docs = pl.concat([fm, fmq, calls], how="vertical_relaxed").unique()
    names = (L.scan("bronze.firm_universe").filter(pl.col("country_code") == "us")
             .group_by("ticker").agg(pl.col("company_name").drop_nulls().first()))
    doc_firm = (docs.join(names, on="ticker", how="left")
                .unique(subset=["accession_number"], keep="first")
                .collect(engine="streaming"))
    return {r["accession_number"]: {"ticker": r["ticker"], "company": r["company_name"]}
            for r in doc_firm.iter_rows(named=True)}


def firm_of(lookup: dict, accession_number: str | None) -> dict:
    return lookup.get(accession_number, {"ticker": None, "company": None})


def _sentences_by_paragraph(paragraph_keys: pl.DataFrame) -> pl.DataFrame:
    return (L.scan("bronze.sentences")
            .join(paragraph_keys.lazy().select(*PARAGRAPH_KEY), on=PARAGRAPH_KEY, how="semi")
            .group_by(PARAGRAPH_KEY)
            .agg(pl.struct(idx=pl.col("sentence_index"), text=pl.col("sentence_text"))
                 .sort_by(pl.col("sentence_index")).alias("sentences"))
            .collect(engine="streaming"))


def sample_frames(n: int, seed: int, firm_lookup: dict) -> list[dict]:
    per_form = max(1, n // len(FORMS))
    # `silver.ai_frames` YA es el equivalente construido en polars de la
    # antigua vista DuckDB `gold_ai_frames`: dedup por última llamada del
    # juez y restringido a la población vigente del prefiltro
    # (`scripts/silver/ai_outputs.py`), así que no hay que rehacer ese join acá.
    frames_all = (L.scan("silver.ai_frames")
                  .filter((pl.col("country_code") == "us") & pl.col("has_frame"))
                  .select(*PARAGRAPH_KEY, "text_hash", "frame_id", "subject", "ai_type", "temporal",
                          "concepts", "specificity", "rhetoric", "sentence_ids")
                  .collect(engine="streaming"))
    paras = frames_all.select(*PARAGRAPH_KEY, "text_hash").unique()
    paras = paras.with_columns(pl.Series("_rnd", _seeded_order(paras["text_hash"], f"build_sample.frames|{seed}")))
    paras = (paras.with_columns(pl.col("_rnd").rank("ordinal").over("form").alias("_rn"))
             .filter(pl.col("_rn") <= per_form))
    paras = (paras.join(_sentences_by_paragraph(paras.select(*PARAGRAPH_KEY)), on=PARAGRAPH_KEY, how="left")
             .sort(PARAGRAPH_KEY))  # orden estable antes del shuffle con semilla, sin depender del join/collect
    frames_all = frames_all.join(paras.select(*PARAGRAPH_KEY), on=PARAGRAPH_KEY, how="semi")

    items = []
    for r in paras.iter_rows(named=True):
        fr = frames_all.filter((pl.col("accession_number") == r["accession_number"])
                                & (pl.col("item_key") == r["item_key"])
                                & (pl.col("paragraph_index") == r["paragraph_index"])).sort("frame_id")
        sentences = [{"idx": s["idx"], "text": s["text"]} for s in (r["sentences"] or [])]
        frame_items = []
        for f in fr.iter_rows(named=True):
            spec_tags, rhet_tags = set(f["specificity"] or []), set(f["rhetoric"] or [])
            frame_items.append({
                "frame_index": int(f["frame_id"]), "subject": f["subject"], "ai_type": f["ai_type"],
                "temporal": f["temporal"],
                "domain": "",  # `domain` no existe por frame en `silver.ai_frames` (sólo en activities)
                "concepts": [str(c) for c in (f["concepts"] or [])],
                "specificity": {k: (tag in spec_tags) for k, tag in FRAME_SPECIFICITY.items()},
                "promotional": "promotional" in rhet_tags, "strategic": "strategic" in rhet_tags,
                "evidence": [int(x) for x in (f["sentence_ids"] or [])],
            })
        items.append({
            "id": f"F:{r['form']}:{r['accession_number']}:{r['item_key']}:{int(r['paragraph_index'])}",
            "form": r["form"], "accession_number": r["accession_number"], "text_hash": str(r["text_hash"]),
            **firm_of(firm_lookup, r["accession_number"]),
            "sentences": sentences, "frames": frame_items,
        })
    random.Random(seed).shuffle(items)
    return items


def sample_activities(n: int, seed: int, firm_lookup: dict) -> list[dict]:
    """Un ítem por PÁRRAFO con todas las actividades que el modelo le sacó, para
    que la persona vea el conjunto y pueda decir si falta alguna. Estratificado
    por formulario y por número de actividades (1 / 2 / 3 o más), para que los
    párrafos largos con varias actividades no queden sub-representados."""
    if not ACTIVITIES_PATH.exists():
        print("sin firm_activities.parquet: corré activity_profiles.py; se omite la muestra de actividades")
        return []
    per = max(1, n // (len(FORMS) * 3))
    a = pl.scan_parquet(ACTIVITIES_PATH).select(*PARAGRAPH_KEY, "text_hash", "activity_id").unique()
    paras = (a.group_by(*PARAGRAPH_KEY, "text_hash").agg(pl.len().alias("n_act"))
             .with_columns(pl.when(pl.col("n_act") == 1).then(pl.lit("1"))
                           .when(pl.col("n_act") == 2).then(pl.lit("2"))
                           .otherwise(pl.lit("3+")).alias("estrato"))
             .collect(engine="streaming"))
    paras = paras.with_columns(pl.Series("_rnd", _seeded_order(paras["text_hash"],
                                                               f"build_sample.activities|{seed}")))
    paras = (paras.with_columns(pl.col("_rnd").rank("ordinal").over(["form", "estrato"]).alias("_rn"))
             .filter(pl.col("_rn") <= per))
    paras = (paras.join(_sentences_by_paragraph(paras.select(*PARAGRAPH_KEY)), on=PARAGRAPH_KEY, how="left")
             .sort(PARAGRAPH_KEY))  # orden estable antes del shuffle con semilla, sin depender del join/collect

    # `firm_activities.parquet` (scripts/gold/posture/activity_profiles.py):
    # `source` es el origen de la IA (propia/terceros) y `evidence_type` la
    # fuerza de la evidencia; `sentence_ids` son posiciones dentro del texto
    # numerado que el juez vio, no el `sentence_index` real (se traduce abajo).
    acts = (pl.scan_parquet(ACTIVITIES_PATH)
            .join(paras.lazy().select("text_hash"), on="text_hash", how="semi")
            .select("text_hash", "activity_id", "action", "object", "function", "target", "stage",
                    "source", "entities", "evidence_type", "sentence_ids")
            .unique().sort(["text_hash", "activity_id"])
            .collect(engine="streaming"))

    items = []
    for r in paras.iter_rows(named=True):
        sents = [{"idx": s["idx"], "text": s["text"]} for s in (r["sentences"] or [])]
        activities = []
        for x in acts.filter(pl.col("text_hash") == r["text_hash"]).iter_rows(named=True):
            ev_pos = [int(v) for v in (x["sentence_ids"] or [])]
            activities.append({
                "activity_index": int(x["activity_id"]), "action": x["action"], "object": x["object"],
                "function": x["function"], "target": x["target"], "stage": x["stage"],
                "ai_source": x["source"],
                "entities": [{"name": str(e["name"]), "role": str(e["role"])} for e in (x["entities"] or [])],
                "evidence_strength": x["evidence_type"],
                "evidence": [sents[i]["idx"] for i in ev_pos if 0 <= i < len(sents)],
            })
        items.append({"id": f"A:{r['text_hash']}", "form": r["form"], "accession_number": r["accession_number"],
                      **firm_of(firm_lookup, r["accession_number"]),
                      "text_hash": str(r["text_hash"]), "sentences": sents, "activities": activities})
    random.Random(seed).shuffle(items)
    return items


def sample_prefilter(n: int, seed: int, firm_lookup: dict) -> list[dict]:
    """Tres estratos por formulario: positivos (proba ≥ umbral), zona gris
    (0,05 ≤ proba < umbral) y negativos con término de IA (proba < 0,05 y
    match léxico). Los negativos sin término no se muestran: son 4,7M y el
    prefiltro los descarta con recall 0,98 medido."""
    pred = latest_predictions()
    per = max(1, n // (len(FORMS) * 3))
    ai_terms = r"(^|[^a-z])(ai|artificial intelligence|machine learning|generative|llm|chatgpt|copilot)([^a-z]|$)"
    pred_us = (pl.scan_parquet(pred).filter(pl.col("country_code") == "us")
               .select("text_hash", "form", "predicted_proba", "is_ai_prefiltered", "threshold",
                        "named_entity_match")
               .collect())

    # `bronze.unique_paragraphs` es ~5,5M filas de texto completo: hay que
    # revisarlas TODAS para saber cuáles negativos mencionan un término de IA
    # (no hay atajo — es la comprobación misma), así que se lee en lotes con
    # pyarrow en vez de un join en memoria completo, para mantenerse bajo el
    # límite de memoria del validador.
    matches = []
    reader = pq.ParquetFile(L.path("bronze.unique_paragraphs"))
    for batch in reader.iter_batches(batch_size=100_000, columns=["text_hash", "paragraph_text", "is_scorable"]):
        up = (pl.from_arrow(batch)
              .filter(pl.col("is_scorable") & pl.col("paragraph_text").str.len_chars().is_between(80, 2500)))
        if up.height == 0:
            continue
        joined = (up.join(pred_us, on="text_hash", how="inner")
                  .filter(pl.col("is_ai_prefiltered") | (pl.col("predicted_proba") >= 0.05)
                          | pl.col("paragraph_text").str.to_lowercase().str.contains(ai_terms)))
        if joined.height:
            matches.append(joined.select("text_hash", "form", "predicted_proba", "is_ai_prefiltered",
                                         "threshold", "named_entity_match", "paragraph_text"))
    empty_schema = {**pred_us.schema, "paragraph_text": pl.String}
    p = (pl.concat(matches) if matches else pl.DataFrame(schema=empty_schema)).with_columns(
        pl.when(pl.col("is_ai_prefiltered")).then(pl.lit("positivo"))
        .when(pl.col("predicted_proba") >= 0.05).then(pl.lit("zona_gris"))
        .otherwise(pl.lit("negativo")).alias("estrato"))
    p = p.with_columns(pl.Series("_rnd", _seeded_order(p["text_hash"], f"build_sample.prefilter|{seed}")))
    p = (p.with_columns(pl.col("_rnd").rank("ordinal").over(["form", "estrato"]).alias("_rn"))
         .filter(pl.col("_rn") <= per).sort("text_hash"))  # orden estable antes del shuffle con semilla

    acc = (L.scan("bronze.paragraphs").filter(pl.col("country_code") == "us")
           .join(p.lazy().select("text_hash"), on="text_hash", how="semi")
           .group_by("text_hash").agg(pl.col("accession_number").first())
           .collect(engine="streaming"))
    acc_by_hash = {r["text_hash"]: r["accession_number"] for r in acc.iter_rows(named=True)}

    items = []
    for r in p.iter_rows(named=True):
        items.append({
            "id": f"P:{r['text_hash']}", "form": r["form"], "text_hash": str(r["text_hash"]),
            "estrato": r["estrato"], **firm_of(firm_lookup, acc_by_hash.get(r["text_hash"])),
            "text": r["paragraph_text"], "proba": round(float(r["predicted_proba"]), 3),
            "prefilter_says_ai": bool(r["is_ai_prefiltered"]), "threshold": float(r["threshold"]),
            "named_entity_match": bool(r["named_entity_match"]),
        })
    random.Random(seed).shuffle(items)
    return items


def _run_isolated(fn, *args):
    """Corre `fn(*args)` en un proceso hijo nuevo (spawn) que se cierra al
    terminar. `bronze.sentences`/`bronze.paragraphs`/`bronze.unique_paragraphs`
    son grandes y cada muestra los recorre una vez; sin esto, la memoria
    liberada entre `sample_frames`/`sample_prefilter`/`sample_activities` no
    vuelve al sistema operativo dentro del mismo proceso y el pico acumulado
    supera el límite de memoria del validador."""
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=1, mp_context=ctx) as ex:
        return ex.submit(fn, *args).result()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--prefilter", type=int, default=300)
    parser.add_argument("--activities", type=int, default=120)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    firm_lookup = register_firm_lookup()
    frames = _run_isolated(sample_frames, args.frames, args.seed, firm_lookup)
    prefilter = _run_isolated(sample_prefilter, args.prefilter, args.seed, firm_lookup)
    activities = _run_isolated(sample_activities, args.activities, args.seed, firm_lookup)
    payload = {"seed": args.seed, "predictions_run": latest_predictions().name,
               "frames": frames, "prefilter": prefilter, "activities": activities}
    args.out.write_text(json.dumps(payload, ensure_ascii=False))
    print(f"frames: {len(frames)} párrafos ({sum(len(i['frames']) for i in frames)} frames) | "
          f"prefilter: {len(prefilter)} párrafos | activities: {len(activities)} párrafos ({sum(len(i['activities']) for i in activities)} actividades) -> {args.out}")


if __name__ == "__main__":
    main()
