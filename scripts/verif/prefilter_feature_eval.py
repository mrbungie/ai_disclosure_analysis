"""¿Las señales de texto arreglan el prefiltro fuera de 10-K/10-Q?

El modelo de 11 señales rinde F1 ponderado 0,925 en los formularios con los que
se ajustó y 0,755 / 0,647 en DEF 14A / 8-K (`docs/prefilter_evaluation.md`
§8.15). Este script mide si las señales de texto de
`ai_prefilter_classify.text_feature_columns()` cierran esa brecha, con una
separación de datos que no se puede discutir:

  ENTRENA  con el golden set (10-K + 10-Q, juez qwen) — nada de los formularios
           nuevos entra al ajuste ni a la elección del umbral.
  EVALÚA   con las 1.500 etiquetas de DEF 14A / 8-K de
           `prefilter_form_validation.py`, que el modelo nunca vio.

Es decir: generalización fuera de dominio, no ajuste dentro de él. Si las
señales nuevas sólo estuvieran memorizando, el número de la derecha no se
movería.

Compara tres conjuntos de features:
  base       las 11 señales desplegadas (7 anchors + negative + margin + 2 pasos léxicos)
  texto      base + las señales de texto (densidad, marcado, contexto, primera persona)
  solo_texto las de texto sin los anchors, para ver cuánto aporta cada bloque

Uso:
    uv run python scripts/verif/prefilter_feature_eval.py
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from sklearn.linear_model import LogisticRegression

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "enrichment"))
import ai_prefilter_classify as pc  # noqa: E402
import ai_prefilter_deploy as deploy  # noqa: E402
import golden_set as gs  # noqa: E402

FORMS_DIR = REPO_ROOT / "data" / "interim" / "golden_set_forms"
JUDGE = "qwen/qwen3.7-flash"


def load_training() -> pd.DataFrame:
    """Golden set (10-K/10-Q) con todas las señales, un solo juez."""
    return (pc.join_sentence_scores(
                pc.join_scores_via_paragraphs(pc.golden_labels(JUDGE)).filter(pl.col("is_scorable")))
            .sort([*pc.PARAGRAPH_KEY, "session_id"])
            .select((pl.col("relevance") != "none").alias("y"), "inclusion_weight", "accession_number",
                    *deploy.feature_columns("paragraph_text"))
            .collect().to_pandas())


def _labels_with_features(labels: pd.DataFrame, extra: list[str]) -> pd.DataFrame:
    """Etiquetas cuya llave es la del representante de unique_paragraphs, con
    todas las señales (sin filtrar `is_scorable`)."""
    return (pc.join_sentence_scores(pc.join_scores_via_unique_paragraphs(pl.from_pandas(labels).lazy()))
            .sort(pc.PARAGRAPH_KEY)
            .select("y", "inclusion_weight", *extra, *deploy.feature_columns("paragraph_text"))
            .collect().to_pandas())


def load_form_training(labels: pd.DataFrame) -> pd.DataFrame:
    """Etiquetas de DEF 14A / 8-K reservadas para ENTRENAR (muestra disjunta de
    la de validación, `prefilter_form_validation.py sample --purpose train`).

    Los pesos de inclusión de estas filas se calculan sobre la población de su
    propio formulario, y los del golden set sobre la de 10-K/10-Q. Como los dos
    diseños cubren particiones disjuntas del corpus, la unión sigue siendo una
    muestra ponderada válida del corpus completo."""
    return _labels_with_features(labels, ["accession_number"])


def load_holdout(labels: pd.DataFrame) -> pd.DataFrame:
    """Las 1.500 de DEF 14A / 8-K, con las mismas señales."""
    return _labels_with_features(labels, ["form", "stratum"])


def weighted_scores(y: np.ndarray, yhat: np.ndarray, w: np.ndarray) -> dict:
    tp = w[y & yhat].sum(); fp = w[~y & yhat].sum(); fn = w[y & ~yhat].sum()
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else float("nan")
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


def nested_cv(X, y, w, groups, factory, grid) -> tuple[object, float, dict]:
    """CV anidado propio, porque `pc.cv_threshold_and_metrics` está atado a la
    logística y acá se comparan familias de modelos distintas.

    El hiperparámetro y el umbral se eligen SOLO con el fold de entrenamiento;
    el fold de test nunca participa de ninguna decisión. Sin esto, comparar un
    modelo flexible contra uno lineal es una trampa: el flexible gana por
    elegirse su propio corte sobre los datos donde se lo mide."""
    from sklearn.model_selection import GroupKFold
    folds = GroupKFold(n_splits=5)
    probabilities = np.zeros(len(y))
    for train_idx, test_idx in folds.split(X, y, groups):
        best = (-1, None, 0.5)
        inner = GroupKFold(n_splits=3)
        for param in grid:
            inner_proba = np.zeros(len(train_idx))
            for a, b in inner.split(X[train_idx], y[train_idx], groups[train_idx]):
                model = factory(param)
                model.fit(X[train_idx][a], y[train_idx][a],
                          sample_weight=np.sqrt(w[train_idx][a]))
                inner_proba[b] = model.predict_proba(X[train_idx][b])[:, 1]
            for cut in np.arange(0.05, 0.96, 0.01):
                score = weighted_scores(y[train_idx].astype(bool),
                                        inner_proba >= cut, w[train_idx])["f1"]
                if score > best[0]:
                    best = (score, param, float(cut))
        model = factory(best[1])
        model.fit(X[train_idx], y[train_idx], sample_weight=np.sqrt(w[train_idx]))
        probabilities[test_idx] = model.predict_proba(X[test_idx])[:, 1]
        chosen_threshold = best[2]
        chosen_param = best[1]
    metrics = weighted_scores(y.astype(bool), probabilities >= chosen_threshold, w)
    return chosen_param, chosen_threshold, {"f1_pond": metrics["f1"],
                                            "prec_pond": metrics["precision"],
                                            "recall_pond": metrics["recall"]}


def make_factory(kind: str):
    """(constructor, grilla) de cada familia comparada."""
    if kind == "logit":
        return (lambda c: LogisticRegression(C=c, max_iter=2000),
                (0.03, 0.1, 0.3, 1.0, 3.0, 10.0))
    if kind == "hgb":
        from sklearn.ensemble import HistGradientBoostingClassifier
        # Árboles: las señales nuevas se usan en INTERACCIÓN ("hay término de IA"
        # Y "densidad baja" Y "mucho marcado de tabla" -> negativo), y una
        # logística sólo puede sumarlas por separado.
        return (lambda d: HistGradientBoostingClassifier(
                    max_depth=d, max_iter=300, learning_rate=0.06,
                    min_samples_leaf=40, l2_regularization=1.0, random_state=42),
                (3, 4, 6))
    raise ValueError(kind)


def evaluate(train: pd.DataFrame, holdout: pd.DataFrame, columns: list[str],
             kind: str = "logit") -> dict:
    X = train[columns].astype(float).to_numpy()
    y = train["y"].astype(bool).to_numpy()
    w = train["inclusion_weight"].astype(float).to_numpy()
    groups = train["accession_number"].to_numpy()
    factory, grid = make_factory(kind)
    deploy_c, threshold, cv = nested_cv(X, y.astype(int), w, groups, factory, grid)

    model = factory(deploy_c)
    model.fit(X, y.astype(int), sample_weight=np.sqrt(w))

    Xh = holdout[columns].astype(float).to_numpy()
    yh = holdout["y"].astype(bool).to_numpy()
    wh = holdout["inclusion_weight"].astype(float).to_numpy()
    proba = model.predict_proba(Xh)[:, 1]
    predicted = proba >= threshold
    # El F1 del umbral elegido mezcla dos cosas: qué tan bien SEPARA el modelo y
    # qué tan bien viajó el umbral de un dominio a otro. AP y el F1 al mejor
    # umbral del holdout aíslan lo primero — si el ranking mejora pero el F1 con
    # umbral fijo no, el problema es la regla de corte, no las features.
    from sklearn.metrics import average_precision_score, roc_auc_score
    sweep = [(weighted_scores(yh, proba >= cut, wh)["f1"], float(cut))
             for cut in np.unique(np.round(proba, 3))]
    best_f1, best_cut = max(sweep) if sweep else (float("nan"), float("nan"))
    out = {"C": deploy_c, "threshold": threshold,
           "holdout_ap": float(average_precision_score(yh, proba, sample_weight=wh)),
           "holdout_auc": float(roc_auc_score(yh, proba, sample_weight=wh)),
           "holdout_best_f1": best_f1, "holdout_best_threshold": best_cut,
           "cv_f1_pond": cv.get("f1_pond"), "cv_prec_pond": cv.get("prec_pond"),
           "cv_recall_pond": cv.get("recall_pond"),
           "holdout": weighted_scores(yh, predicted, wh)}
    for form, group in holdout.groupby("form"):
        idx = holdout["form"] == form
        out[f"holdout_{form}"] = weighted_scores(
            yh[idx.to_numpy()], predicted[idx.to_numpy()], wh[idx.to_numpy()])
    if hasattr(model, "coef_"):
        out["coef"] = dict(zip(columns, model.coef_[0].round(3)))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--forms-dir", type=Path, default=FORMS_DIR)
    parser.add_argument("--output", type=Path,
                        default=REPO_ROOT / "data" / "interim" / "audits" / "prefilter" / "feature_eval.json")
    parser.add_argument("--include-form-train", action="store_true",
                        help="Sumar al ajuste las etiquetas de DEF 14A / 8-K reservadas para "
                             "entrenar. El holdout de validación NO cambia, así que las cifras "
                             "de las dos corridas son comparables.")
    args = parser.parse_args()

    def read_labels(pattern: str, extra: list[str]) -> pd.DataFrame:
        files = sorted(glob.glob(str(args.forms_dir / pattern)))
        if not files:
            return pd.DataFrame()
        frame = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        frame = frame[frame["error"].isna()].drop_duplicates(list(gs.PARAGRAPH_KEY))
        frame["y"] = frame["relevance"] != "none"
        return frame[list(gs.PARAGRAPH_KEY) + ["y", "inclusion_weight"] + extra]

    labels = read_labels("golden_set_labels__*.parquet", ["stratum"])
    form_train = read_labels("train/golden_set_labels__*.parquet", [])

    train = load_training()
    holdout = load_holdout(labels)
    if args.include_form_train and not form_train.empty:
        extra = load_form_training(form_train)
        print(f"+ {len(extra):,} etiquetas de entrenamiento de DEF 14A / 8-K "
              f"({int(extra['y'].sum()):,} positivas), disjuntas del holdout")
        train = pd.concat([train, extra], ignore_index=True)
    print(f"entrenamiento: {len(train):,} etiquetas ({int(train['y'].sum()):,} positivas)")
    print(f"holdout: {len(holdout):,} etiquetas DEF 14A / 8-K "
          f"({int(holdout['y'].sum()):,} positivas) — nunca vistas por el modelo\n")

    base = pc.ALL_SIGNAL_COLUMNS
    text = pc.TEXT_FEATURE_NAMES
    sentence = pc.SENTENCE_FEATURE_NAMES
    variants = {"logit base (desplegado)": (list(base), "logit"),
                "árboles base": (list(base), "hgb"),
                "árboles base + texto": (list(base) + list(text), "hgb"),
                "árboles base + oración": (list(base) + list(sentence), "hgb"),
                "árboles base + texto + oración": (list(base) + list(text) + list(sentence), "hgb")}

    report = {}
    header = (f"{'variante':30s} {'thr':>5s} {'CV F1':>7s} | {'HOLD F1':>8s} {'prec':>6s} "
              f"{'rec':>6s} {'AP':>6s} {'AUC':>6s} {'F1*':>6s} {'thr*':>5s}")
    print(header)
    print("-" * len(header))
    for name, (columns, kind) in variants.items():
        result = evaluate(train, holdout, columns, kind)
        report[name] = result
        print(f"{name:30s} {result['threshold']:>5.2f} {result['cv_f1_pond']:>7.3f} | "
              f"{result['holdout']['f1']:>8.3f} {result['holdout']['precision']:>6.3f} "
              f"{result['holdout']['recall']:>6.3f} {result['holdout_ap']:>6.3f} "
              f"{result['holdout_auc']:>6.3f} {result['holdout_best_f1']:>6.3f} "
              f"{result['holdout_best_threshold']:>5.2f}")

    best = max(report, key=lambda k: report[k]["holdout"]["f1"])
    print(f"\nmejor fuera de dominio: {best}")
    if "coef" in report[best]:
        print("coeficientes (leer el signo, no la magnitud — no hay estandarización):")
        for feature, coefficient in sorted(report[best]["coef"].items(),
                                           key=lambda kv: -abs(kv[1]))[:12]:
            print(f"  {feature:24s} {coefficient:+.3f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=float))
    print(f"\n-> {args.output}")


if __name__ == "__main__":
    main()
