"""¿Cambia algo entrenar el prefiltro con un solo juez en vez de con la mezcla?

El golden set quedó etiquetado por dos modelos repartidos por estrato y el
modelo desplegado se ajustó con la unión de ambos. Re-etiquetar todo con un
solo juez cuesta plata, así que corresponde medir si la decisión cambia algo
antes de tratarla como una mejora — y si no cambia nada, decirlo con el número
al lado en vez de dejarlo como "buena práctica".

Compara tres conjuntos de entrenamiento sobre EXACTAMENTE el mismo pipeline
(nested GroupKFold por filing, C y threshold elegidos sólo con el fold de
entrenamiento, `sqrt(inclusion_weight)` en el fit):

  qwen      sólo etiquetas de qwen3.7-flash (el juez único después del relabel)
  gemini    sólo etiquetas de gemini-3.8-flash (las originales)
  all       la mezcla, que es lo que se desplegó

y reporta, para cada uno: F1 ponderado out-of-fold, precisión, recall, el C y
el threshold elegidos, y —lo que de verdad importa— **cuántos textos del corpus
marcaría cada modelo y cuánto se solapan esas poblaciones**. Dos modelos con
métricas parecidas pueden marcar corpus distintos; dos modelos con métricas
distintas pueden marcar casi lo mismo. Lo que decide si el gasto valió la pena
es el solapamiento, no el F1.

Uso:
    uv run python scripts/verif/prefilter_judge_comparison.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "common"))
import ai_prefilter_classify as pc  # noqa: E402

JUDGES = {"qwen": "qwen/qwen3.7-flash", "gemini": "gemini-3.8-flash", "all": None}


def fit_and_score(golden: pd.DataFrame) -> dict:
    y = golden["is_ai_mention"].astype(int).to_numpy()
    weights = golden["inclusion_weight"].astype(float).to_numpy()
    groups = golden["accession_number"].to_numpy()
    X = golden[pc.ALL_SIGNAL_COLUMNS].astype(float).to_numpy()
    deploy_c, threshold, metrics = pc.cv_threshold_and_metrics(X, y, weights, groups)
    model = LogisticRegression(C=deploy_c, max_iter=pc.MAX_ITER if hasattr(pc, "MAX_ITER") else 1000)
    model.fit(X, y, sample_weight=np.sqrt(weights))
    return {"n_labels": int(len(golden)), "positives": int(y.sum()),
            "C": float(deploy_c), "threshold": float(threshold),
            **{k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))},
            "model": model}


def corpus_marks(model: LogisticRegression, threshold: float, features: np.ndarray) -> np.ndarray:
    return model.predict_proba(features)[:, 1] >= threshold


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path,
                        default=REPO_ROOT / "data" / "interim" / "prefilter_predictions_unique"
                        / "judge_comparison.json")
    parser.add_argument("--corpus-limit", type=int, default=0,
                        help="0 = corpus completo. Un número lo trunca, para probar rápido.")
    args = parser.parse_args()

    fits = {}
    for name, judge in JUDGES.items():
        golden = pc.load_golden(judge)
        if golden.empty:
            print(f"{name}: sin etiquetas, se omite")
            continue
        fits[name] = fit_and_score(golden)
        row = fits[name]
        print(f"{name:7s} n={row['n_labels']:6,d} pos={row['positives']:5,d} "
              f"C={row['C']:<6g} thr={row['threshold']:.2f} "
              + " ".join(f"{k}={v:.3f}" for k, v in row.items()
                         if k.startswith(("f1", "prec", "recall"))))

    print("\nAplicando cada modelo al corpus puntuable...")
    con = duckdb.connect(str(pc.DB), read_only=True)
    try:
        limit = f"LIMIT {args.corpus_limit}" if args.corpus_limit else ""
        corpus = con.execute(f"""
            SELECT up.form, {', '.join(f'p.{c}' for c in pc.SIGNAL_COLUMNS)},
                   {', '.join(f'{sql} AS {name}' for name, sql in pc.LEXICAL_STEP_COLUMNS)}
            FROM {pc.scores_relation()} p
            JOIN unique_paragraphs up
              ON up.country_code = p.country_code AND up.form = p.form
             AND up.accession_number = p.accession_number AND up.item_key = p.item_key
             AND up.paragraph_index = p.paragraph_index
            WHERE up.is_scorable
            {limit}
        """).fetchdf()
    finally:
        con.close()
    features = corpus[pc.ALL_SIGNAL_COLUMNS].astype(float).to_numpy()
    print(f"{len(corpus):,} textos únicos puntuables")

    marks = {}
    for name, row in fits.items():
        marks[name] = corpus_marks(row["model"], row["threshold"], features)
        by_form = corpus.assign(m=marks[name]).groupby("form")["m"].sum().astype(int)
        print(f"  {name:7s} marca {int(marks[name].sum()):7,d} textos "
              f"({marks[name].mean()*100:.3f}% del corpus) | por forma: {by_form.to_dict()}")

    print("\nSolapamiento entre poblaciones marcadas (Jaccard):")
    names = list(marks)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            union = (marks[a] | marks[b]).sum()
            shared = (marks[a] & marks[b]).sum()
            print(f"  {a:7s} vs {b:7s}: {shared / union if union else float('nan'):.3f} "
                  f"({int(shared):,} en común, {int(marks[a].sum() - shared):,} sólo {a}, "
                  f"{int(marks[b].sum() - shared):,} sólo {b})")

    report = {name: {k: v for k, v in row.items() if k != "model"} for name, row in fits.items()}
    for name in marks:
        report[name]["corpus_marked"] = int(marks[name].sum())
    args.output.write_text(json.dumps(report, indent=2, default=float))
    print(f"\n-> {args.output}")


if __name__ == "__main__":
    main()
