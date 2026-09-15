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

  evaluate  precisión/recall/F1 ponderados del modelo desplegado, por
            formulario, contra 10-K/10-Q como referencia.

La muestra y el etiquetado de las que depende `evaluate` viven en
`scripts/enrichment/golden_set_forms.py` (subcomandos `sample` y `label`).

Uso:
    uv run python scripts/verif/prefilter_form_validation.py evaluate
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "enrichment"))
import golden_set as gs  # noqa: E402

L = gs.L
OUT_DIR = REPO_ROOT / "data" / "interim" / "golden_set_forms"
PREDICTIONS = REPO_ROOT / "data" / "interim" / "prefilter_predictions_unique"


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

    # Última predicción por llave de instancia (todas las corridas del
    # directorio): el `model_version` más nuevo y, entre archivos del mismo
    # modelo, el archivo más nuevo.
    keys = list(gs.PARAGRAPH_KEY)
    predictions = (pl.concat([pl.scan_parquet(f).with_columns(pl.lit(f.name).alias("_file"))
                              for f in sorted(args.predictions.glob("*.parquet"))], how="diagonal_relaxed")
                   .select(*keys, "predicted_proba", "is_ai_prefiltered", "threshold",
                           pl.col("model_version").cast(pl.String), "_file")
                   .sort("model_version", "_file", descending=True, nulls_last=True)
                   .unique(keys, keep="first")
                   .drop("_file")
                   .collect().to_pandas())
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
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    subparsers = parser.add_subparsers(dest="command", required=True)

    evaluator = subparsers.add_parser("evaluate", help="Métricas ponderadas por formulario")
    evaluator.add_argument("--predictions", type=Path, default=PREDICTIONS)
    evaluator.set_defaults(func=cmd_evaluate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
