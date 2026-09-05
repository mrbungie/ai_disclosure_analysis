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

Este script mide lo que faltaba, con el mismo diseño del golden set:

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
  evaluate  precisión/recall/F1 ponderados del modelo desplegado, por
            formulario, contra 10-K/10-Q como referencia.

Uso:
    uv run python scripts/verif/prefilter_form_validation.py sample
    uv run python scripts/verif/prefilter_form_validation.py label
    uv run python scripts/verif/prefilter_form_validation.py evaluate
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "common"))
import golden_set as gs  # noqa: E402

DB = REPO_ROOT / "duckdb" / "thesis.duckdb"
OUT_DIR = REPO_ROOT / "data" / "interim" / "golden_set_forms"
SAMPLE_PATH = OUT_DIR / "form_validation_sample.parquet"
PREDICTIONS = REPO_ROOT / "data" / "interim" / "prefilter_predictions_unique"
SAMPLING_VERSION = "forms-v1"

# Cuotas por (formulario, tier). El tier fuerte se sobre-muestrea porque es
# donde vive casi todo el positivo: con prevalencia de ~0,3% una muestra
# proporcional daría un puñado de positivos y no permitiría medir precisión.
# `inclusion_weight` devuelve la muestra a la escala de la población.
QUOTAS = {("DEF 14A", "strong"): 400, ("DEF 14A", "weak"): 300, ("DEF 14A", "none"): 300,
          ("8-K", "strong"): 200, ("8-K", "weak"): 150, ("8-K", "none"): 150}


def cmd_sample(args) -> None:
    con = duckdb.connect(str(args.database), read_only=True)
    try:
        pool = con.execute(f"""
            SELECT up.country_code, up.form, up.accession_number, up.item_key,
                   up.paragraph_index, up.content_type, up.paragraph_text,
                   {gs._keyword_case_sql('up.paragraph_text')} AS keyword_tier,
                   coalesce(CAST(extract(year from fm.filing_date) AS VARCHAR), 'NA') AS filing_year
            FROM unique_paragraphs up
            LEFT JOIN filing_manifest fm
                   ON fm.country_code = up.country_code
                  AND fm.accession_number = up.accession_number
            WHERE up.country_code = 'us' AND up.is_scorable
              AND up.form IN ('DEF 14A', '8-K')
        """).fetchdf()
    finally:
        con.close()
    print(f"población puntuable: {len(pool):,} textos únicos")
    sizes = pool.groupby(["form", "keyword_tier"]).size()
    print(sizes.to_string())

    chunks = []
    for (form, tier), quota in QUOTAS.items():
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
    sample["sampling_version"] = SAMPLING_VERSION
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample.to_parquet(args.sample_path, index=False)
    print(f"\n-> {args.sample_path} ({len(sample):,} filas)")


def cmd_label(args) -> None:
    load_dotenv(REPO_ROOT / ".env")
    if not gs.os.environ.get("OPENROUTER_API_KEY"):
        sys.exit("Falta OPENROUTER_API_KEY en .env")
    sample = pd.read_parquet(args.sample_path)
    done = set()
    for part in sorted(args.output_dir.glob(gs.LABEL_GLOB)):
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
        pending, args.output_dir, session_id, args.judge_model, args.concurrency,
        args.part_rows, args.progress_every, start_index=0))
    print(f"\n{stats['labeled']:,} etiquetados, {stats['failed']:,} con error "
          f"| partes: {len(written)}")


def weighted_metrics(frame: pd.DataFrame) -> dict:
    """Precisión, recall y F1 ponderados por `inclusion_weight` — la muestra es
    deliberadamente no representativa, así que sin pesos las tres cifras
    describen el estrato sobre-muestreado y no el formulario."""
    w = frame["inclusion_weight"].to_numpy(float)
    y = frame["is_ai_mention"].to_numpy(bool)
    yhat = frame["is_ai_prefiltered"].to_numpy(bool)
    tp = float(w[y & yhat].sum())
    fp = float(w[~y & yhat].sum())
    fn = float(w[y & ~yhat].sum())
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
          if precision and recall and not np.isnan(precision + recall) else float("nan"))
    return {"n": int(len(frame)), "positivos": int(y.sum()),
            "prevalencia_pond": float(w[y].sum() / w.sum()),
            "precision": precision, "recall": recall, "f1": f1}


def cmd_evaluate(args) -> None:
    parts = sorted(args.output_dir.glob(gs.LABEL_GLOB))
    if not parts:
        sys.exit(f"No hay etiquetas en {args.output_dir}; corré primero `label`.")
    labels = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    labels = labels[labels["error"].isna()].drop_duplicates(list(gs.PARAGRAPH_KEY))
    labels["is_ai_mention"] = labels["relevance"] != "none"
    print(f"{len(labels):,} etiquetas legibles | jueces: "
          f"{labels['judge_model'].value_counts().to_dict()}")

    con = duckdb.connect(":memory:")
    predictions = con.execute(f"""
        SELECT country_code, form, accession_number, item_key, paragraph_index,
               predicted_proba, is_ai_prefiltered, threshold, model_version
        FROM read_parquet('{args.predictions}/*.parquet', union_by_name=True)
        QUALIFY row_number() OVER (
            PARTITION BY country_code, form, accession_number, item_key, paragraph_index
            ORDER BY model_version DESC) = 1
    """).fetchdf()
    con.close()
    merged = labels.merge(predictions, on=list(gs.PARAGRAPH_KEY), how="inner")
    print(f"{len(merged):,} con predicción del modelo desplegado "
          f"({merged['model_version'].iloc[0]}, umbral {merged['threshold'].iloc[0]})\n")

    rows = []
    for form, group in merged.groupby("form"):
        rows.append({"form": form, **weighted_metrics(group)})
    rows.append({"form": "TODOS", **weighted_metrics(merged)})
    table = pd.DataFrame(rows)
    print(table.round(4).to_string(index=False))

    print("\npor estrato (sin ponderar — cobertura cruda de cada tier):")
    by_stratum = merged.groupby("stratum").apply(
        lambda g: pd.Series({
            "n": len(g), "positivos": int(g["is_ai_mention"].sum()),
            "marcados": int(g["is_ai_prefiltered"].sum()),
            "prec": float((g["is_ai_mention"] & g["is_ai_prefiltered"]).sum()
                          / max(g["is_ai_prefiltered"].sum(), 1)),
            "recall": float((g["is_ai_mention"] & g["is_ai_prefiltered"]).sum()
                            / max(g["is_ai_mention"].sum(), 1)),
        }), include_groups=False)
    print(by_stratum.round(3).to_string())

    misses = merged[merged["is_ai_mention"] & ~merged["is_ai_prefiltered"]]
    print(f"\nfalsos negativos: {len(misses)} (proba mediana "
          f"{misses['predicted_proba'].median() if len(misses) else float('nan'):.3f})")
    for _, row in misses.nlargest(min(5, len(misses)), "predicted_proba").iterrows():
        print(f"  [{row['form']}] p={row['predicted_proba']:.3f} "
              f"{str(row['evidence_quote'])[:110]}")

    destination = args.output_dir / "form_validation_metrics.json"
    destination.write_text(json.dumps(
        {"rows": rows, "by_stratum": json.loads(by_stratum.to_json(orient="index")),
         "model_version": str(merged["model_version"].iloc[0]),
         "threshold": float(merged["threshold"].iloc[0]),
         "evaluated_at": datetime.now(timezone.utc).isoformat()}, indent=2, default=float))
    print(f"\n-> {destination}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--database", type=Path, default=DB)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--sample-path", type=Path, default=SAMPLE_PATH)
    subparsers = parser.add_subparsers(dest="command", required=True)

    sampler = subparsers.add_parser("sample", help="Muestra estratificada por forma y tier")
    sampler.add_argument("--seed", type=int, default=42)
    sampler.set_defaults(func=cmd_sample)

    labeler = subparsers.add_parser("label", help="Etiqueta con el juez del golden set")
    labeler.add_argument("--judge-model", default=gs.DEFAULT_JUDGE_MODEL)
    labeler.add_argument("--limit", type=int, default=0)
    labeler.add_argument("--concurrency", type=int, default=8)
    labeler.add_argument("--part-rows", type=int, default=250)
    labeler.add_argument("--progress-every", type=int, default=50)
    labeler.set_defaults(func=cmd_label)

    evaluator = subparsers.add_parser("evaluate", help="Métricas ponderadas por formulario")
    evaluator.add_argument("--predictions", type=Path, default=PREDICTIONS)
    evaluator.set_defaults(func=cmd_evaluate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
