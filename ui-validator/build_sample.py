"""Arma `data.json` para la UI de validación humana (`index.html`).

Dos tareas, una muestra cada una, todo desde `duckdb/thesis.duckdb` y los
parquets del prefiltro — sin LLM, sin red:

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


CALLS_MANIFEST = REPO_ROOT / "data" / "interim" / "manifests" / "filing_manifest_earnings_calls.parquet"


def register_firms(con: duckdb.DuckDBPyConnection) -> None:
    """Vista `doc_firm`: accession_number -> ticker y nombre de la empresa, para
    que la persona sepa de quién es el párrafo (si dice "AMD" y la empresa es
    AMD, no es un proveedor externo)."""
    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW doc_firm AS
        WITH docs AS (
            SELECT accession_number, ticker FROM filing_manifest WHERE country_code = 'us'
            UNION ALL SELECT accession_number, ticker FROM filing_manifest_10q WHERE country_code = 'us'
            UNION ALL SELECT document_id, ticker FROM read_parquet('{CALLS_MANIFEST}')
        ), names AS (
            SELECT ticker, any_value(company_name) AS company_name FROM firm_universe WHERE country_code = 'us' GROUP BY 1
        )
        SELECT DISTINCT d.accession_number, d.ticker, n.company_name FROM docs d LEFT JOIN names n USING (ticker)
    """)


def firm_of(con: duckdb.DuckDBPyConnection, accessions: list[str]) -> dict:
    if not accessions:
        return {}
    con.register("acc_list", __import__("pandas").DataFrame({"accession_number": list(set(accessions))}))
    rows = con.execute("SELECT f.accession_number, f.ticker, f.company_name FROM doc_firm f JOIN acc_list USING (accession_number)").fetchall()
    return {a: {"ticker": t, "company": c} for a, t, c in rows}


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
    firms = firm_of(con, rows["accession_number"].tolist())
    items = []
    for _, r in rows.iterrows():
        fr = frames[(frames.accession_number == r.accession_number) & (frames.item_key == r.item_key)
                    & (frames.paragraph_index == r.paragraph_index)].sort_values("frame_index")
        items.append({
            "id": f"F:{r.form}:{r.accession_number}:{r.item_key}:{int(r.paragraph_index)}",
            "form": r.form, "accession_number": r.accession_number, "text_hash": str(r.text_hash),
            **firms.get(r.accession_number, {"ticker": None, "company": None}),
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


def sample_activities(con: duckdb.DuckDBPyConnection, n: int, seed: int) -> list[dict]:
    """Un ítem por PÁRRAFO con todas las actividades que el modelo le sacó, para
    que la persona vea el conjunto y pueda decir si falta alguna. Estratificado
    por formulario y por número de actividades (1 / 2 / 3 o más), para que los
    párrafos largos con varias actividades no queden sub-representados."""
    acts = REPO_ROOT / "data" / "processed" / "clusters" / "firm_activities.parquet"
    if not acts.exists():
        print("sin firm_activities.parquet: corré activity_profiles.py; se omite la muestra de actividades")
        return []
    per = max(1, n // (len(FORMS) * 3))
    rows = con.execute(f"""
        WITH a AS (
            SELECT DISTINCT country_code, form, accession_number, item_key, paragraph_index, text_hash, activity_index
            FROM read_parquet('{acts}')
        ), paras AS (
            SELECT country_code, form, accession_number, item_key, paragraph_index, text_hash,
                   count(*) AS n_act, CASE WHEN count(*) = 1 THEN '1' WHEN count(*) = 2 THEN '2' ELSE '3+' END AS estrato
            FROM a GROUP BY ALL
        ), s AS (
            SELECT *, row_number() OVER (PARTITION BY form, estrato ORDER BY hash(text_hash + {seed})) AS rn FROM paras
        ), chosen AS (SELECT * FROM s WHERE rn <= {per})
        SELECT c.form, c.accession_number, c.item_key, c.paragraph_index, c.text_hash, c.n_act,
               list(struct_pack(idx := se.sentence_index, text := se.sentence_text) ORDER BY se.sentence_index) AS sentences
        FROM chosen c JOIN sentences se USING (country_code, form, accession_number, item_key, paragraph_index)
        GROUP BY ALL
    """).df()
    con.register("chosen_paras", rows[["text_hash"]])
    acts_df = con.execute(f"""
        SELECT DISTINCT a.text_hash, a.activity_index, a.action, a.object, a.function, a.target, a.stage, a.ai_source,
               a.named_entities, a.evidence_strength, a.evidence_sentence_ids
        FROM read_parquet('{acts}') a JOIN chosen_paras USING (text_hash) ORDER BY a.text_hash, a.activity_index
    """).df()
    firms = firm_of(con, rows["accession_number"].tolist())
    items = []
    for r in rows.itertuples():
        sents = [{"idx": int(s["idx"]), "text": s["text"]} for s in list(r.sentences)]
        activities = []
        for x in acts_df[acts_df.text_hash == r.text_hash].itertuples():
            ev_pos = [int(v) for v in (list(x.evidence_sentence_ids) if x.evidence_sentence_ids is not None else [])]
            activities.append({
                "activity_index": int(x.activity_index), "action": x.action, "object": x.object, "function": x.function,
                "target": x.target, "stage": x.stage,
                "ai_source": x.ai_source,
                "entities": [{"name": str(e["name"]), "role": str(e["role"])} for e in (list(x.named_entities) if x.named_entities is not None else [])],
                "evidence_strength": x.evidence_strength,
                "evidence": [sents[i]["idx"] for i in ev_pos if 0 <= i < len(sents)],   # posiciones del prompt -> sentence_index real
            })
        items.append({"id": f"A:{r.text_hash}", "form": r.form, "accession_number": r.accession_number,
                      **firms.get(r.accession_number, {"ticker": None, "company": None}),
                      "text_hash": str(r.text_hash), "sentences": sents, "activities": activities})
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
    con.register("pref_hashes", rows[["text_hash"]])
    acc = con.execute("""
        SELECT p.text_hash, any_value(p.accession_number) AS accession_number
        FROM paragraphs p JOIN pref_hashes h USING (text_hash) WHERE p.country_code = 'us' GROUP BY 1
    """).df()
    firms = firm_of(con, acc["accession_number"].tolist())
    firm_by_hash = {int(r.text_hash): firms.get(r.accession_number, {}) for r in acc.itertuples()}
    items = [{
        "id": f"P:{r.text_hash}", "form": r.form, "text_hash": str(r.text_hash), "estrato": r.estrato,
        **firm_by_hash.get(int(r.text_hash), {"ticker": None, "company": None}),
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
    parser.add_argument("--activities", type=int, default=120)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--database", type=Path, default=DB)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    con = duckdb.connect(str(args.database), read_only=True)
    register_firms(con)
    frames = sample_frames(con, args.frames, args.seed)
    prefilter = sample_prefilter(con, args.prefilter, args.seed)
    activities = sample_activities(con, args.activities, args.seed)
    payload = {"seed": args.seed, "predictions_run": latest_predictions().name,
               "frames": frames, "prefilter": prefilter, "activities": activities}
    args.out.write_text(json.dumps(payload, ensure_ascii=False))
    print(f"frames: {len(frames)} párrafos ({sum(len(i['frames']) for i in frames)} frames) | "
          f"prefilter: {len(prefilter)} párrafos | activities: {len(activities)} párrafos ({sum(len(i['activities']) for i in activities)} actividades) -> {args.out}")


if __name__ == "__main__":
    main()
