"""¿El prefiltro funciona en DEF 14A y 8-K, o sólo en los formularios con los
que se lo evaluó?

El golden set (`docs/golden_set_sampling.md`) es **100% 10-K y 10-Q**: 6.991 y
2.909 párrafos, cero proxies, cero 8-K. Sobre esas etiquetas se eligió el
umbral (0,75), se midió el F1 ponderado y se decidió desplegar. Después el
mismo modelo se aplicó tal cual a DEF 14A y 8-K, que hoy aportan 6.533 y 262
frames — el 24% de la población analizada — y sostienen el hallazgo titular de
`01_...md` (#8: 16,1% de frames promocionales en el proxy contra 7,2% en el
10-K).

Ese hallazgo compara formularios, y **la comparación supone que el instrumento
mide lo mismo en los dos**. Eso nunca se testeó, y hay evidencia directa de que
el dominio cambió: el override de entidades nombradas explotó en los proxies
porque hay directores que se llaman Claude.

Este script muestrea y etiqueta, con el mismo diseño del golden set:

  sample    muestra estratificada por (formulario × tier de palabra clave) con
            `inclusion_weight` para poder reponderar a la población de cada
            formulario. El tier sale del diccionario CONGELADO de golden_set.py,
            que es anterior e independiente del prefiltro — la regla dura de
            §2 de docs/golden_set_sampling.md: el muestreo no puede leer la
            salida de lo que se quiere evaluar.
  label     etiqueta con el mismo juez, prompt y esquema que el golden set,
            reusando su maquinaria (partes atómicas, aditivo, errores en la
            fila). Directorio aparte: NO contamina el golden set de
            entrenamiento.

`scripts/verif/prefilter_form_validation.py` mide precisión/recall/F1
ponderados del modelo desplegado contra estas etiquetas.

Uso:
    uv run python scripts/enrichment/golden_set_forms.py sample
    uv run python scripts/enrichment/golden_set_forms.py label
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import polars as pl
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "enrichment"))
import golden_set as gs  # noqa: E402

L = gs.L
OUT_DIR = REPO_ROOT / "data" / "interim" / "golden_set_forms"
SAMPLE_PATH = OUT_DIR / "form_validation_sample.parquet"
SAMPLING_VERSION = "forms-v1"

# Cuotas por (formulario, tier). El tier fuerte se sobre-muestrea porque es
# donde vive casi todo el positivo: con prevalencia de ~0,3% una muestra
# proporcional daría un puñado de positivos y no permitiría medir precisión.
# `inclusion_weight` devuelve la muestra a la escala de la población.
QUOTAS = {("DEF 14A", "strong"): 400, ("DEF 14A", "weak"): 300, ("DEF 14A", "none"): 300,
          ("8-K", "strong"): 200, ("8-K", "weak"): 150, ("8-K", "none"): 150,
          # Earnings calls: es HABLA, no documento escrito — preguntas de
          # analistas, muletillas, y un registro estructuralmente más
          # promocional. El modelo nunca vio ese canal, así que su error ahí es
          # desconocido hasta que se mida, igual que pasó con proxy y 8-K.
          ("Earnings call", "strong"): 400, ("Earnings call", "weak"): 200,
          ("Earnings call", "none"): 200}
# Muestra de ENTRENAMIENTO, disjunta de la de validación (`--purpose train`).
# Existe porque el umbral del prefiltro se elige sobre el golden set, que sólo
# tiene 10-K y 10-Q: medido en `prefilter_feature_eval.py`, las señales de texto
# mejoran el ORDENAMIENTO fuera de dominio (AP 0,725 -> 0,775) pero el umbral
# elegido en 10-K/10-Q no viaja, y el F1 con ese corte no sube. La solución es
# que el conjunto donde se elige el corte cubra los formularios donde se aplica
# — el mismo argumento por el que existe stage3_random.
# En 8-K los estratos léxicos son minúsculos (231 strong y 213 weak en TODO el
# formulario) y la validación ya se llevó 350, así que acá entra lo que queda.
TRAIN_QUOTAS = {("DEF 14A", "strong"): 800, ("DEF 14A", "weak"): 600, ("DEF 14A", "none"): 600,
                ("8-K", "strong"): 200, ("8-K", "weak"): 200, ("8-K", "none"): 300}
TRAIN_SAMPLE_PATH = OUT_DIR / "form_train_sample.parquet"
# Directorio aparte, no un prefijo distinto dentro del mismo: `gs.label_rows`
# nombra sus partes con el patrón del golden set, así que una corrida de
# entrenamiento a medio camino escribiría archivos con el MISMO nombre que los
# de validación y cualquier lectura con glob los mezclaría — justo lo que la
# separación train/holdout existe para impedir. Separar por carpeta lo hace
# imposible en vez de depender de renombrar al final.
TRAIN_DIR = OUT_DIR / "train"
CALLS_DIR = OUT_DIR / "calls"
CALLS_SAMPLE_PATH = OUT_DIR / "calls_validation_sample.parquet"


def cmd_sample(args) -> None:
    manifest = L.scan("silver.filing_manifest").select("country_code", "accession_number", "filing_date")
    pool = (L.scan("bronze.unique_paragraphs")
            .filter((pl.col("country_code") == "us") & pl.col("is_scorable")
                    & pl.col("form").is_in(["DEF 14A", "8-K", "Earnings call"]))
            .join(manifest, on=["country_code", "accession_number"], how="left")
            .select("country_code", "form", "accession_number", "item_key", "paragraph_index",
                    "content_type", "paragraph_text",
                    gs._keyword_tier_expr("paragraph_text").alias("keyword_tier"),
                    pl.col("filing_date").cast(pl.Date, strict=False).dt.year().cast(pl.String)
                    .fill_null("NA").alias("filing_year"))
            .sort(list(gs.PARAGRAPH_KEY))
            .collect().to_pandas())
    print(f"población puntuable: {len(pool):,} textos únicos")
    sizes = pool.groupby(["form", "keyword_tier"]).size()
    print(sizes.to_string())

    if args.purpose == "train":
        # Excluir lo ya muestreado para validación: si un párrafo entrenara y
        # validara a la vez, la métrica de generalización sería una ilusión.
        validation = pd.read_parquet(SAMPLE_PATH)
        before = len(pool)
        pool = pool.merge(validation[list(gs.PARAGRAPH_KEY)], on=list(gs.PARAGRAPH_KEY),
                          how="left", indicator=True)
        pool = pool[pool["_merge"] == "left_only"].drop(columns=["_merge"])
        print(f"excluidos {before - len(pool):,} párrafos ya usados en validación")

    chunks = []
    quotas = TRAIN_QUOTAS if args.purpose == "train" else QUOTAS
    forms = getattr(args, "forms", None)
    for (form, tier), quota in quotas.items():
        if forms and form not in forms:
            continue
        stratum = pool[(pool["form"] == form) & (pool["keyword_tier"] == tier)]
        if stratum.empty:
            print(f"  aviso: estrato vacío {form}/{tier}")
            continue
        take = min(quota, len(stratum))
        picked = stratum.sample(n=take, random_state=args.seed).copy()
        picked["stage"] = f"form_{tier}"
        picked["stratum"] = f"{form}|{tier}"
        # Peso de inclusión = población del estrato / muestreados. Sin esto, la
        # precisión medida describe la muestra sobre-representada, no el corpus.
        picked["inclusion_weight"] = len(stratum) / take
        chunks.append(picked)
        print(f"  {form:8s} {tier:6s}: {take:4d} de {len(stratum):9,d} "
              f"(peso {len(stratum) / take:,.1f})")

    sample = pd.concat(chunks, ignore_index=True)
    sample["sector"] = "na"
    sample["sampling_version"] = SAMPLING_VERSION + ("-train" if args.purpose == "train" else "")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    destination = TRAIN_SAMPLE_PATH if args.purpose == "train" else args.sample_path
    sample.to_parquet(destination, index=False)
    print(f"\n-> {destination} ({len(sample):,} filas)")


def cmd_label(args) -> None:
    load_dotenv(REPO_ROOT / ".env")
    if not gs.os.environ.get("OPENROUTER_API_KEY"):
        sys.exit("Falta OPENROUTER_API_KEY en .env")
    training = args.purpose == "train"
    sample = pd.read_parquet(TRAIN_SAMPLE_PATH if training else args.sample_path)
    output_dir = TRAIN_DIR if training else args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    done = set()
    for part in sorted(output_dir.glob(gs.LABEL_GLOB)):
        table = pd.read_parquet(part)
        table = table[table["error"].isna()]
        done |= set(map(tuple, table[list(gs.PARAGRAPH_KEY)].to_numpy()))
    pending = [row for row in sample.to_dict("records")
               if tuple(row[k] for k in gs.PARAGRAPH_KEY) not in done]
    if args.limit:
        pending = pending[: args.limit]
    print(f"{len(sample):,} muestreados | {len(done):,} ya etiquetados | "
          f"{len(pending):,} pendientes")
    if not pending:
        return
    session_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    written, stats = asyncio.run(gs.label_rows(
        pending, output_dir, session_id, args.judge_model, args.concurrency,
        args.part_rows, args.progress_every, start_index=0))
    print(f"\n{stats['labeled']:,} etiquetados, {stats['failed']:,} con error "
          f"| partes: {len(written)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--sample-path", type=Path, default=SAMPLE_PATH)
    subparsers = parser.add_subparsers(dest="command", required=True)

    sampler = subparsers.add_parser("sample", help="Muestra estratificada por forma y tier")
    sampler.add_argument("--seed", type=int, default=42)
    sampler.add_argument("--forms", nargs="*", default=None,
                         help="Limitar el muestreo a estos formularios (default: todos "
                              "los que tengan cuota definida).")
    sampler.add_argument("--purpose", choices=("validation", "train"), default="validation",
                         help="'validation' (default) escribe la muestra intocable; 'train' "
                              "escribe una muestra DISJUNTA para meter al ajuste.")
    sampler.set_defaults(func=cmd_sample)

    labeler = subparsers.add_parser("label", help="Etiqueta con el juez del golden set")
    labeler.add_argument("--judge-model", default=gs.DEFAULT_JUDGE_MODEL)
    labeler.add_argument("--purpose", choices=("validation", "train"), default="validation")
    labeler.add_argument("--limit", type=int, default=0)
    labeler.add_argument("--concurrency", type=int, default=8)
    labeler.add_argument("--part-rows", type=int, default=250)
    labeler.add_argument("--progress-every", type=int, default=50)
    labeler.set_defaults(func=cmd_label)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
