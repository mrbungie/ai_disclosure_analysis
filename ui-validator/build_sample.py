"""Arma `data.json` para la UI de validación humana (`index.html`).

Dos tareas, una muestra cada una, todo desde `duckdb/thesis.duckdb` y los
parquets del prefiltro — sin LLM, sin red:

  frames     N párrafos con frames extraídos: el texto numerado por oración
             (como lo vio el juez) y cada frame con sus etiquetas. La persona
             valida frame por frame: ¿existe?, ¿promocional?, ¿temporal?,
             ¿specificity? — las dimensiones que sostienen 09/11/12/13.
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
import json
import random
from pathlib import Path

import duckdb

REPO_ROOT = Path(__file__).resolve().parents[1]
DB = REPO_ROOT / "duckdb" / "thesis.duckdb"
PRED_DIR = REPO_ROOT / "data" / "interim" / "prefilter_predictions_unique"
OUT = Path(__file__).resolve().parent / "data.json"
FORMS = ["10-K", "10-Q", "DEF 14A", "8-K", "Earnings call"]


def latest_predictions() -> Path:
    runs = sorted(PRED_DIR.glob("prefilter_predictions__run=*.parquet"))
    if not runs:
        raise SystemExit(f"no hay predicciones del prefiltro en {PRED_DIR}")
    return runs[-1]


def sample_frames(con: duckdb.DuckDBPyConnection, n: int, seed: int) -> list[dict]:
    per_form = max(1, n // len(FORMS))
    rows = con.execute(f"""
        WITH paras AS (
            SELECT country_code, form, accession_number, item_key, paragraph_index, text_hash,
                   row_number() OVER (PARTITION BY form ORDER BY hash(text_hash + {seed})) AS rn
            FROM (SELECT DISTINCT country_code, form, accession_number, item_key, paragraph_index, text_hash
                  FROM gold_ai_frames WHERE country_code = 'us' AND has_frame)
        ), chosen AS (SELECT * FROM paras WHERE rn <= {per_form})
        SELECT c.form, c.accession_number, c.item_key, c.paragraph_index, c.text_hash,
               list(struct_pack(idx := s.sentence_index, text := s.sentence_text) ORDER BY s.sentence_index) AS sentences
        FROM chosen c JOIN sentences s USING (country_code, form, accession_number, item_key, paragraph_index)
        GROUP BY ALL
    """).df()
    frames = con.execute("""
        SELECT form, accession_number, item_key, paragraph_index, text_hash, frame_index, subject, ai_type,
               temporal, domain, concepts, specificity_business_process, specificity_product_or_system,
               specificity_vendor_or_partner, specificity_quantified_metric, specificity_date_or_timeline,
               rhetoric_promotional, rhetoric_strategic_importance, evidence_sentence_ids
        FROM gold_ai_frames WHERE country_code = 'us' AND has_frame
    """).df()
    frames = frames.merge(rows[["form", "accession_number", "item_key", "paragraph_index"]],
                          on=["form", "accession_number", "item_key", "paragraph_index"])
    items = []
    for _, r in rows.iterrows():
        fr = frames[(frames.accession_number == r.accession_number) & (frames.item_key == r.item_key)
                    & (frames.paragraph_index == r.paragraph_index)].sort_values("frame_index")
        items.append({
            "id": f"F:{r.form}:{r.accession_number}:{r.item_key}:{int(r.paragraph_index)}",
            "form": r.form, "accession_number": r.accession_number, "text_hash": str(r.text_hash),
            "sentences": [{"idx": int(s["idx"]), "text": s["text"]} for s in list(r.sentences)],
            "frames": [{
                "frame_index": int(f.frame_index), "subject": f.subject, "ai_type": f.ai_type,
                "temporal": f.temporal, "domain": f.domain, "concepts": [str(c) for c in (list(f.concepts) if f.concepts is not None else [])],
                "specificity": {k.replace("specificity_", ""): bool(getattr(f, k)) for k in
                                ("specificity_business_process", "specificity_product_or_system",
                                 "specificity_vendor_or_partner", "specificity_quantified_metric",
                                 "specificity_date_or_timeline")},
                "promotional": bool(f.rhetoric_promotional), "strategic": bool(f.rhetoric_strategic_importance),
                "evidence": [int(x) for x in (list(f.evidence_sentence_ids) if f.evidence_sentence_ids is not None else [])],
            } for f in fr.itertuples()],
        })
    random.Random(seed).shuffle(items)
    return items


def sample_prefilter(con: duckdb.DuckDBPyConnection, n: int, seed: int) -> list[dict]:
    """Tres estratos por formulario: positivos (proba ≥ umbral), zona gris
    (0,05 ≤ proba < umbral) y negativos con término de IA (proba < 0,05 y
    match léxico). Los negativos sin término no se muestran: son 4,7M y el
    prefiltro los descarta con recall 0,98 medido."""
    pred = latest_predictions()
    per = max(1, n // (len(FORMS) * 3))
    rows = con.execute(f"""
        WITH p AS (
            SELECT p.text_hash, p.form, p.predicted_proba, p.is_ai_prefiltered, p.threshold, p.named_entity_match,
                   CASE WHEN p.is_ai_prefiltered THEN 'positivo'
                        WHEN p.predicted_proba >= 0.05 THEN 'zona_gris'
                        ELSE 'negativo' END AS estrato,
                   up.paragraph_text
            FROM read_parquet('{pred}') p
            JOIN unique_paragraphs up USING (text_hash)
            WHERE p.country_code = 'us' AND up.is_scorable AND length(up.paragraph_text) BETWEEN 80 AND 2500
              AND (p.is_ai_prefiltered OR p.predicted_proba >= 0.05
                   OR regexp_matches(lower(up.paragraph_text), '(^|[^a-z])(ai|artificial intelligence|machine learning|generative|llm|chatgpt|copilot)([^a-z]|$)'))
        )
        SELECT * FROM (
            SELECT *, row_number() OVER (PARTITION BY form, estrato ORDER BY hash(text_hash + {seed})) AS rn FROM p
        ) WHERE rn <= {per}
    """).df()
    items = [{
        "id": f"P:{r.text_hash}", "form": r.form, "text_hash": str(r.text_hash), "estrato": r.estrato,
        "text": r.paragraph_text, "proba": round(float(r.predicted_proba), 3),
        "prefilter_says_ai": bool(r.is_ai_prefiltered), "threshold": float(r.threshold),
        "named_entity_match": bool(r.named_entity_match),
    } for r in rows.itertuples()]
    random.Random(seed).shuffle(items)
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--prefilter", type=int, default=300)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--database", type=Path, default=DB)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    con = duckdb.connect(str(args.database), read_only=True)
    frames = sample_frames(con, args.frames, args.seed)
    prefilter = sample_prefilter(con, args.prefilter, args.seed)
    payload = {"seed": args.seed, "predictions_run": latest_predictions().name,
               "frames": frames, "prefilter": prefilter}
    args.out.write_text(json.dumps(payload, ensure_ascii=False))
    print(f"frames: {len(frames)} párrafos ({sum(len(i['frames']) for i in frames)} frames) | "
          f"prefilter: {len(prefilter)} párrafos -> {args.out}")


if __name__ == "__main__":
    main()
