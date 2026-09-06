"""Ajusta y despliega el prefiltro: árboles sobre señales de párrafo, de texto
y de oración.

Reemplaza el camino de entrenamiento de `ai_prefilter_classify.py` (logística
sobre 11 señales), que quedó corto donde más importa: F1 ponderado 0,925 en
10-K/10-Q y 0,755 / 0,647 en DEF 14A / 8-K, formularios que hoy aportan el 24%
de los frames (`docs/prefilter_evaluation.md` §8.15).

Tres cambios, medidos siempre contra las 1.500 etiquetas de DEF 14A / 8-K que
NUNCA entran al ajuste (`prefilter_form_validation.py`, muestra de validación):

  1. Señales de TEXTO (`ai_prefilter_classify.text_feature_columns`): densidad
     del término, marcado de tabla y viñeta, largo, primera persona, contexto de
     biografía / votación / regulación, sigla suelta vs. término escrito
     completo. Salen de leer los falsos positivos reales, que comparten FORMA y
     no tema.
  2. Señales de ORACIÓN (`ai_prefilter_sentences.py`): el mismo scoring
     semántico aplicado sólo a las oraciones que mencionan IA. De las 226.139
     oraciones de los textos candidatos, sólo el 14% menciona IA: el resto es lo
     que el vector del párrafo estaba promediando.
  3. Modelo no lineal (gradient boosting): las señales nuevas se usan en
     INTERACCIÓN — "hay término" Y "densidad baja" Y "mucho marcado de tabla"
     -> negativo — y una logística sólo puede sumarlas por separado.

Lo que NO se hace, deliberadamente: meter el formulario como feature. Sería un
atajo — el modelo aprendería "los proxies mencionan más IA" en vez de leer el
párrafo — y como el hallazgo central del proyecto ES una comparación entre
formularios (`docs/analytics/01_...md` #8), la medición quedaría circular. Todas
las señales se calculan del texto y valen igual en cualquier documento.

Etiquetas de ajuste: golden set (10-K/10-Q, un solo juez) + la muestra de
ENTRENAMIENTO de DEF 14A / 8-K, disjunta de la de validación. Sin esa segunda
fuente el umbral se elige en un dominio y se aplica en otro, que es exactamente
lo que hacía fallar al modelo anterior.

Uso:
    uv run python scripts/common/ai_prefilter_deploy.py
    uv run python scripts/common/ai_prefilter_deploy.py --dry-run   # sin escribir corpus
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "common"))
import ai_prefilter_classify as pc  # noqa: E402
import golden_set as gs  # noqa: E402

OUT_DIR = pc.OUT_DIR
FORMS_DIR = REPO_ROOT / "data" / "interim" / "golden_set_forms"
DEPTH_GRID = (3, 4, 6)
THRESHOLD_GRID = np.arange(0.05, 0.96, 0.01)


def model_factory(depth: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_depth=depth, max_iter=300, learning_rate=0.06, min_samples_leaf=40,
        l2_regularization=1.0, random_state=42)


def feature_names() -> list[str]:
    return (list(pc.ALL_SIGNAL_COLUMNS) + list(pc.TEXT_FEATURE_NAMES)
            + list(pc.SENTENCE_FEATURE_NAMES))


def feature_sql(text_expr: str) -> str:
    parts = [f"p.{c}" for c in pc.SIGNAL_COLUMNS]
    parts += [f"{sql} AS {name}" for name, sql in pc.LEXICAL_STEP_COLUMNS]
    parts += [f"{sql} AS {name}" for name, sql in pc.text_feature_columns(text_expr)]
    parts.append(pc.sentence_feature_sql())
    return ", ".join(parts)


def sentence_join() -> str:
    return f"LEFT JOIN {pc.sentence_scores_relation()} sent ON sent.text_hash = up.text_hash"


def load_golden_labels(con, judge_model: str) -> pd.DataFrame:
    frame = con.execute(f"""
        SELECT l.relevance != 'none' AS y, l.inclusion_weight, l.accession_number,
               {pc._named_entity_sql("lower(coalesce(par.paragraph_text, ''))")} AS named_entity_match,
               {feature_sql('par.paragraph_text')}
        FROM read_parquet('data/interim/golden_set/golden_set_labels__session=*__part=*.parquet',
                          union_by_name=True) l
        JOIN paragraphs par
          ON par.country_code = l.country_code AND par.form = l.form
         AND par.accession_number = l.accession_number AND par.item_key = l.item_key
         AND par.paragraph_index = l.paragraph_index
        JOIN unique_paragraphs up ON up.text_hash = par.text_hash
        JOIN {pc.scores_relation()} p
          ON p.country_code = up.country_code AND p.form = up.form
         AND p.accession_number = up.accession_number AND p.item_key = up.item_key
         AND p.paragraph_index = up.paragraph_index
        {sentence_join()}
        WHERE l.error IS NULL AND up.is_scorable AND l.judge_model = '{judge_model}'
    """).df()
    frame["source"] = "golden_10k_10q"
    return frame


def load_form_labels(con, subdir: str, source: str) -> pd.DataFrame:
    """Etiquetas de DEF 14A / 8-K. `subdir='train'` es la muestra de ajuste;
    el directorio de validación NUNCA se lee acá."""
    pattern = f"data/interim/golden_set_forms/{subdir}/golden_set_labels__session=*.parquet"
    if not list(Path(REPO_ROOT).glob(pattern)):
        return pd.DataFrame()
    frame = con.execute(f"""
        SELECT l.relevance != 'none' AS y, l.inclusion_weight, l.accession_number, l.form,
               {pc._named_entity_sql("lower(coalesce(up.paragraph_text, ''))")} AS named_entity_match,
               {feature_sql('up.paragraph_text')}
        FROM read_parquet('{pattern}', union_by_name=True) l
        JOIN unique_paragraphs up
          ON up.country_code = l.country_code AND up.form = l.form
         AND up.accession_number = l.accession_number AND up.item_key = l.item_key
         AND up.paragraph_index = l.paragraph_index
        JOIN {pc.scores_relation()} p ON p.text_hash = up.text_hash
        {sentence_join()}
        WHERE l.error IS NULL AND up.is_scorable
    """).df()
    frame["source"] = source
    return frame


def weighted_f1(y: np.ndarray, predicted: np.ndarray, w: np.ndarray,
                beta: float = 1.0) -> dict:
    """Precisión, recall y F-beta ponderados.

    `beta` es una decisión de proyecto, no un detalle: en este pipeline un
    FALSO POSITIVO del prefiltro cuesta una llamada al juez LLM y se descarta
    (pasó literal con `claude` en proxies: 191 textos rescatados, 5 con frame),
    mientras que un FALSO NEGATIVO no se recupera nunca — el párrafo no vuelve a
    aparecer en ninguna etapa posterior. La asimetría es de costo marginal
    (centavos) contra sesgo permanente en la población de análisis, así que el
    umbral se elige con beta > 1."""
    tp = w[y & predicted].sum(); fp = w[~y & predicted].sum(); fn = w[y & ~predicted].sum()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    denominator = (beta ** 2) * precision + recall
    fbeta = (1 + beta ** 2) * precision * recall / denominator if denominator else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": float(precision), "recall": float(recall),
            "f1": float(f1), "fbeta": float(fbeta)}


def nested_cv(X, y, w, groups, beta: float = 1.0) -> dict:
    """Profundidad y umbral se eligen SÓLO dentro de cada fold de entrenamiento.
    Es la única forma de comparar un modelo flexible contra uno lineal sin
    regalarle el corte."""
    outer = GroupKFold(n_splits=5)
    oof = np.zeros(len(y))
    chosen = []
    for train_idx, test_idx in outer.split(X, y, groups):
        best = (-1.0, DEPTH_GRID[0], 0.5)
        inner = GroupKFold(n_splits=3)
        for depth in DEPTH_GRID:
            inner_proba = np.zeros(len(train_idx))
            for a, b in inner.split(X[train_idx], y[train_idx], groups[train_idx]):
                model = model_factory(depth)
                model.fit(X[train_idx][a], y[train_idx][a],
                          sample_weight=np.sqrt(w[train_idx][a]))
                inner_proba[b] = model.predict_proba(X[train_idx][b])[:, 1]
            for cut in THRESHOLD_GRID:
                score = weighted_f1(y[train_idx].astype(bool), inner_proba >= cut,
                                    w[train_idx], beta)["fbeta"]
                if score > best[0]:
                    best = (score, depth, float(cut))
        model = model_factory(best[1])
        model.fit(X[train_idx], y[train_idx], sample_weight=np.sqrt(w[train_idx]))
        oof[test_idx] = model.predict_proba(X[test_idx])[:, 1]
        chosen.append({"depth": best[1], "threshold": best[2]})
    threshold = float(np.median([c["threshold"] for c in chosen]))
    metrics = weighted_f1(y.astype(bool), oof >= threshold, w, beta)
    metrics["ap_pond"] = float(average_precision_score(y, oof, sample_weight=w))
    return {"metrics": metrics, "folds": chosen, "threshold": threshold,
            "depth": int(np.median([c["depth"] for c in chosen]))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--judge-model", default=pc.DEFAULT_JUDGE_MODEL)
    parser.add_argument("--beta", type=float, default=2.0,
                        help="Peso del recall frente a la precisión al elegir el umbral. "
                             "2.0 (default) = el recall vale el doble; 1.0 = F1 clásico. "
                             "Ver el docstring de weighted_f1 para por qué acá el default "
                             "no es 1.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Ajusta y evalúa, pero no escribe predicciones del corpus.")
    args = parser.parse_args()

    con = duckdb.connect(str(pc.DB), read_only=True)
    golden = load_golden_labels(con, args.judge_model)
    form_train = load_form_labels(con, "train", "form_train")
    labels = pd.concat([golden, form_train], ignore_index=True)
    print(f"ajuste: {len(golden):,} etiquetas 10-K/10-Q + {len(form_train):,} de DEF 14A/8-K "
          f"= {len(labels):,} ({int(labels['y'].sum()):,} positivas)")

    columns = feature_names()
    X = labels[columns].astype(float).to_numpy()
    y = labels["y"].astype(int).to_numpy()
    w = labels["inclusion_weight"].astype(float).to_numpy()
    groups = labels["accession_number"].to_numpy()

    print(f"CV anidado sobre {len(columns)} señales "
          f"({len(pc.ALL_SIGNAL_COLUMNS)} párrafo + {len(pc.TEXT_FEATURE_NAMES)} texto + "
          f"{len(pc.SENTENCE_FEATURE_NAMES)} oración)...")
    cv = nested_cv(X, y, w, groups, args.beta)
    print(f"  F1 pond. {cv['metrics']['f1']:.3f} | F{args.beta:g} {cv['metrics']['fbeta']:.3f} | "
          f"prec {cv['metrics']['precision']:.3f} | recall {cv['metrics']['recall']:.3f} | "
          f"AP {cv['metrics']['ap_pond']:.3f}")
    print(f"  profundidad {cv['depth']}, umbral {cv['threshold']:.2f} "
          f"(mediana de los folds: {[c['threshold'] for c in cv['folds']]})")

    model = model_factory(cv["depth"])
    model.fit(X, y, sample_weight=np.sqrt(w))
    threshold = cv["threshold"]

    holdout = load_form_labels(con, ".", "form_validation")
    holdout_metrics = {}
    if not holdout.empty:
        Xh = holdout[columns].astype(float).to_numpy()
        yh = holdout["y"].astype(bool).to_numpy()
        wh = holdout["inclusion_weight"].astype(float).to_numpy()
        proba_h = model.predict_proba(Xh)[:, 1]
        holdout_metrics["total"] = weighted_f1(yh, proba_h >= threshold, wh, args.beta)
        holdout_metrics["total"]["ap"] = float(average_precision_score(yh, proba_h, sample_weight=wh))
        for form, group in holdout.groupby("form"):
            mask = (holdout["form"] == form).to_numpy()
            holdout_metrics[form] = weighted_f1(yh[mask], (proba_h >= threshold)[mask],
                                                wh[mask], args.beta)
        print("\nHOLDOUT (1.500 etiquetas de DEF 14A / 8-K que nunca entraron al ajuste):")
        for name, values in holdout_metrics.items():
            print(f"  {name:10s} F1 {values['f1']:.3f} | prec {values['precision']:.3f} | "
                  f"recall {values['recall']:.3f}"
                  + (f" | AP {values['ap']:.3f}" if "ap" in values else ""))

    if not holdout.empty:
        print("\ncurva de decisión (umbral -> recall/precisión fuera de dominio):")
        print(f"  {'umbral':>7s} {'recall':>7s} {'prec':>7s} {'F2':>7s}")
        for cut in (0.5, 0.36, 0.25, 0.2, 0.15, 0.12, 0.1, 0.07, 0.05):
            point = weighted_f1(yh, proba_h >= cut, wh, 2.0)
            marker = "  <- desplegado" if abs(cut - threshold) < 0.005 else ""
            print(f"  {cut:7.2f} {point['recall']:7.3f} {point['precision']:7.3f} "
                  f"{point['fbeta']:7.3f}{marker}")

    if args.dry_run:
        con.close()
        print("\n--dry-run: no se escribió nada.")
        return

    print("\nAplicando al corpus completo...")
    corpus = con.execute(f"""
        SELECT p.{', p.'.join(pc.PARAGRAPH_KEY)}, p.text_hash, up.duplicate_count,
               {pc._named_entity_sql("lower(coalesce(up.paragraph_text, ''))")} AS named_entity_match,
               {feature_sql('up.paragraph_text')}
        FROM {pc.scores_relation()} p
        JOIN unique_paragraphs up ON up.text_hash = p.text_hash
        {sentence_join()}
        WHERE up.is_scorable
    """).df()
    print(f"{len(corpus):,} textos únicos")
    proba = model.predict_proba(corpus[columns].astype(float).to_numpy())[:, 1]
    named_entity = corpus["named_entity_match"].astype(bool).to_numpy()
    model_positive = proba >= threshold
    is_positive = model_positive | named_entity
    rescued = int((named_entity & ~model_positive).sum())
    instances = int(corpus.loc[is_positive, "duplicate_count"].sum())
    print(f"Marcados: {int(is_positive.sum()):,} textos únicos ({100 * is_positive.mean():.2f}%), "
          f"{instances:,} instancias — {rescued:,} sólo por named_entity_match")

    funnel = pc.funnel_counts(con)
    anchors_run = pc.current_scores_run(con)
    con.close()

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = corpus[[*pc.PARAGRAPH_KEY, "text_hash", "duplicate_count"]].copy()
    out["predicted_proba"] = proba.astype("float32")
    out["named_entity_match"] = named_entity
    out["is_ai_prefiltered"] = is_positive
    out["threshold"] = threshold
    out["model_version"] = run_id
    out["anchors_run"] = anchors_run
    destination = OUT_DIR / f"prefilter_predictions__run={run_id}.parquet"
    pq.write_table(pa.Table.from_pandas(out, preserve_index=False), destination, compression="zstd")

    model_path = OUT_DIR / f"prefilter_model__run={run_id}.joblib"
    joblib.dump({"model": model, "columns": columns, "threshold": threshold}, model_path)

    funnel["prefilter_model_only_positive"] = int(model_positive.sum())
    funnel["named_entity_rescued"] = rescued
    funnel["prefilter_model_positive"] = int(is_positive.sum())
    funnel["prefilter_model_positive_instances"] = instances
    manifest = OUT_DIR / f"prefilter_predictions_manifest__run={run_id}.json"
    manifest.write_text(json.dumps({
        "run_id": run_id, "anchors_run": anchors_run, "model_kind": "hist_gradient_boosting",
        "model_path": str(model_path), "features": columns, "threshold": threshold,
        "depth": cv["depth"], "judge_model": args.judge_model,
        "labels_golden": int(len(golden)), "labels_form_train": int(len(form_train)),
        "cv_metrics": cv["metrics"], "cv_folds": cv["folds"], "beta": args.beta,
        "holdout_form_metrics": holdout_metrics,
        "named_ai_entities": list(pc.NAMED_AI_ENTITIES),
        "named_entity_used_in_deployment": True,
        "funnel": funnel, "output": str(destination),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }, indent=2, default=float) + "\n")
    print(f"\n-> {destination}\n-> {model_path}\n-> {manifest}")
    for stage, count in funnel.items():
        print(f"  {stage}: {count:,}")


if __name__ == "__main__":
    main()
