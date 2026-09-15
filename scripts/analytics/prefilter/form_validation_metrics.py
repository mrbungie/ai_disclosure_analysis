"""Precisión/recall/F1 ponderados del prefiltro desplegado en DEF 14A y 8-K,
por formulario, contra 10-K/10-Q como referencia.

Las etiquetas humanas-en-el-loop vienen de
`scripts/enrichment/golden_set_forms.py` (subcomandos `sample`/`label`),
guardadas en `data/interim/golden_set_forms/` — 1.500 párrafos de DEF 14A / 8-K
etiquetados con el mismo juez y prompt que el golden set de 10-K/10-Q. Este
script las cruza contra `bronze.prefilter_predictions` (el modelo desplegado,
ya deduplicado a la corrida vigente) y reporta las mismas métricas ponderadas
por `inclusion_weight` que decidieron el despliegue.

Uso:
    uv run python scripts/analytics/prefilter/form_validation_metrics.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "common"))
import layers as L  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "scripts" / "enrichment"))
import golden_set as gs  # noqa: E402

LABELS_DIR = REPO_ROOT / "data" / "interim" / "golden_set_forms"


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


def main() -> None:
    parts = sorted(LABELS_DIR.glob(gs.LABEL_GLOB))
    if not parts:
        sys.exit(f"No hay etiquetas en {LABELS_DIR}; corré primero "
                 f"`scripts/enrichment/golden_set_forms.py label`.")
    labels = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    labels = labels[labels["error"].isna()].drop_duplicates(list(gs.PARAGRAPH_KEY))
    labels["is_ai_mention"] = labels["relevance"] != "none"
    print(f"{len(labels):,} etiquetas legibles | jueces: "
          f"{labels['judge_model'].value_counts().to_dict()}")

    # Última predicción por llave de instancia: el `model_version` más nuevo,
    # por si `bronze.prefilter_predictions` retiene más de una corrida.
    keys = list(gs.PARAGRAPH_KEY)
    predictions = (L.scan("bronze.prefilter_predictions")
                   .select(*keys, "predicted_proba", "is_ai_prefiltered", "threshold",
                           pl.col("model_version").cast(pl.String))
                   .sort("model_version", descending=True, nulls_last=True)
                   .unique(keys, keep="first")
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

    destination = L.results_path("prefilter", "form_validation_metrics.json")
    destination.write_text(json.dumps(
        {"rows": rows, "by_stratum": json.loads(by_stratum.to_json(orient="index")),
         "model_version": str(merged["model_version"].iloc[0]),
         "threshold": float(merged["threshold"].iloc[0]),
         "evaluated_at": datetime.now(timezone.utc).isoformat()}, indent=2, default=float))
    print(f"\n-> {destination}")


if __name__ == "__main__":
    main()
