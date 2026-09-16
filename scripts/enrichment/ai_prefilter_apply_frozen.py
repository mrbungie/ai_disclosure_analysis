"""Aplica el modelo de prefiltro YA CONGELADO (freeze del 2026-09-06) al corpus
completo actual, sin reentrenar.

`ai_prefilter_deploy.py` siempre reentrena via `nested_cv()` sobre las labels
doradas + form_train VIGENTES. Si esas labels cambiaron desde el freeze (por
poco que sea), el modelo resultante puede tener otra profundidad -> viola
`docs/FREEZE.md`. Este script carga el .joblib ya desplegado
(`models/ai_classification/run=<FROZEN_RUN>/model.joblib`, guardado por
`ai_prefilter_deploy.py` como {"model", "columns", "threshold"}) y lo aplica
tal cual a `pc.scores_relation()` (que ya incluye cualquier score nuevo, p.
ej. párrafos recién embebidos), replicando exactamente el bloque "Aplicando
al corpus completo" de `ai_prefilter_deploy.py` — misma población (`deploy.corpus_features()`), mismo
named_entity rescue, mismo esquema de salida — pero sin tocar el modelo.

Uso:
    uv run python scripts/enrichment/ai_prefilter_apply_frozen.py
    uv run python scripts/enrichment/ai_prefilter_apply_frozen.py --frozen-run 20260906T160624Z
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "enrichment"))
import ai_prefilter_classify as pc  # noqa: E402
import ai_prefilter_deploy as deploy  # noqa: E402

OUT_DIR = pc.OUT_DIR
DEFAULT_FROZEN_RUN = "20260906T160624Z"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frozen-run", default=DEFAULT_FROZEN_RUN,
                        help="run_id del modelo congelado a cargar (no reentrenar).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Aplica y reporta pero no escribe nada.")
    args = parser.parse_args()

    frozen_model_path = deploy.MODEL_DIR / f"run={args.frozen_run}" / "model.joblib"
    if not frozen_model_path.exists():
        raise SystemExit(f"No existe el modelo congelado: {frozen_model_path}")
    frozen = joblib.load(frozen_model_path)
    model, columns, threshold = frozen["model"], frozen["columns"], frozen["threshold"]
    print(f"Modelo congelado cargado: {frozen_model_path} "
          f"({len(columns)} señales, umbral {threshold:.2f}, sin reentrenar)")

    print("Aplicando al corpus completo (sin reentrenar)...")
    corpus = deploy.corpus_features()
    print(f"{len(corpus):,} textos únicos")

    proba = model.predict_proba(corpus[columns].astype(float).to_numpy())[:, 1]
    named_entity = corpus["named_entity_match"].astype(bool).to_numpy()
    # Un término fuerte nombra la IA directamente, así que decide por sí solo:
    # el modelo puede añadir textos que el léxico no ve, nunca quitar los que sí.
    # Medido antes de esto: de 1.778 párrafos con "agentic", 197 no llegaban al
    # juez, y al leerlos la mayoría era divulgación real (workloads agénticos,
    # sistemas que actúan con autonomía), no boilerplate. La asimetría ya está
    # escrita en `weighted_f1`: un falso positivo cuesta una llamada al juez y se
    # descarta, un falso negativo no se recupera nunca.
    strong_lexical = corpus["strong_ge1"].astype(bool).to_numpy()
    model_positive = proba >= threshold
    is_positive = model_positive | named_entity | strong_lexical
    rescued = int((named_entity & ~model_positive).sum())
    by_lexicon = int((strong_lexical & ~model_positive & ~named_entity).sum())
    instances = int(corpus.loc[is_positive, "duplicate_count"].sum())
    print(f"Marcados: {int(is_positive.sum()):,} textos únicos ({100 * is_positive.mean():.2f}%), "
          f"{instances:,} instancias — {rescued:,} sólo por named_entity_match, "
          f"{by_lexicon:,} sólo por término fuerte")

    anchors_run = pc.current_scores_run()

    if args.dry_run:
        print("\n--dry-run: no se escribió nada.")
        return

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = corpus[[*pc.PARAGRAPH_KEY, "text_hash", "duplicate_count"]].copy()
    out["predicted_proba"] = proba.astype("float32")
    out["named_entity_match"] = named_entity
    out["is_ai_prefiltered"] = is_positive
    out["threshold"] = threshold
    out["model_version"] = args.frozen_run  # el modelo NO cambió: se referencia el congelado
    out["anchors_run"] = anchors_run
    destination = OUT_DIR / f"prefilter_predictions__run={run_id}.parquet"
    pq.write_table(pa.Table.from_pandas(out, preserve_index=False), destination, compression="zstd")

    manifest = OUT_DIR / f"prefilter_predictions_manifest__run={run_id}.json"
    manifest.write_text(json.dumps({
        "run_id": run_id, "anchors_run": anchors_run, "model_kind": "hist_gradient_boosting",
        "model_path": str(frozen_model_path), "frozen_run": args.frozen_run,
        "features": columns, "threshold": threshold,
        "retrained": False,
        "output": str(destination),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }, indent=2, default=float) + "\n")
    print(f"\n-> {destination}\n-> {manifest}")
    print("(modelo reutilizado, no se escribió un .joblib nuevo)")


if __name__ == "__main__":
    main()
