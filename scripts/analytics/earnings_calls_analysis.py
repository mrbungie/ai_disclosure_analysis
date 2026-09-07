"""El canal de earnings calls: cómo hablan de IA las empresas cuando NO están
escribiendo un documento presentado.

Por qué el canal importa para esta tesis. Una transcripción no es un documento
presentado ante la SEC: no está sujeta a la Sección 18, tiene el safe harbor del
PSLRA para declaraciones prospectivas, y nadie la revisa antes de decirla. Es el
mismo emisor, el mismo trimestre y el mismo tema, con otra exposición legal. Y
es el canal del único caso del corpus con evidencia externa de AI-washing:
Welltower dijo "industry-leading" en su call y no puso nada proporcional en el
10-K.

Y dentro del canal hay una estructura que ningún formulario tiene: la
transcripción viene partida en **`prepared`** (el guion que la empresa lee, y que
pasa por legales e IR antes de la call) y **`qa`** (las respuestas improvisadas a
los analistas). Mismo emisor, mismo día, mismo tema, distinto grado de
preparación. Es el contraste más limpio disponible para preguntar si el registro
promocional sobre IA es una decisión deliberada o algo que se escapa hablando.

Este script describe el canal por sí solo (que es lo pedido): volumen, registro,
evolución, el corte guion/improvisado, y quién habla. La comparación formal
ENTRE canales es otra cosa y va aparte — ver
`docs/pregunta_identificacion_sec.md`.

Advertencia de medición que hay que leer antes de las cifras: el prefiltro se
entrenó con 10-K/10-Q y se ajustó en DEF 14A / 8-K; **en calls no tiene
ninguna validación** (`docs/analytics/00_funnel_del_corpus.md`, tabla de
conjuntos etiquetados) y el registro hablado es distinto. Las tasas de abajo se
calculan sobre frames que el juez LLM confirmó (`has_frame`), que filtra falsos
positivos del prefiltro, pero la comparación entre canales arrastra error de
medición distinto por canal hasta que la muestra humana lo mida.

Uso:
    uv run python scripts/analytics/earnings_calls_analysis.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DB = REPO_ROOT / "duckdb" / "thesis.duckdb"
OUT_DIR = REPO_ROOT / "data" / "processed" / "clusters"
CALLS_MANIFEST = "data/interim/manifests/filing_manifest_earnings_calls.parquet"
BEHAVIOR = ["deployed", "pilot_or_testing", "exploring", "ai_investment",
            "ai_infrastructure", "ai_talent", "proprietary_ai", "third_party_ai",
            "expansion_or_scaling", "productivity_outcome", "revenue_outcome",
            "cost_outcome", "customer_outcome"]


def load_calls(con) -> pd.DataFrame:
    """Frames de earnings calls con su fecha y empresa.

    El manifiesto de calls no está en `filing_manifest` (que es de formularios
    SEC), así que se lee de su parquet; `filing_date` viene como VARCHAR ahí y
    como DATE en los otros, de ahí el CAST."""
    frames = con.execute("""
        SELECT accession_number, item_key, text_hash, frame_index, subject, ai_type, temporal,
               domain, concepts, rhetoric_promotional, rhetoric_strategic_importance,
               specificity_quantified_metric
        FROM gold_ai_frames
        WHERE country_code = 'us' AND has_frame AND form = 'Earnings call'
    """).fetchdf()
    # El manifiesto de calls identifica cada transcripción con `document_id`
    # (ej. "AAP_2022Q3"), que es lo que el extractor usó como
    # `accession_number` — las calls no tienen número de accession de la SEC
    # porque no son un filing.
    manifest = con.execute(f"""
        SELECT document_id AS accession_number, ticker,
               CAST(filing_date AS DATE) AS fecha
        FROM read_parquet('{REPO_ROOT / CALLS_MANIFEST}') WHERE ticker IS NOT NULL
    """).fetchdf()
    data = frames.merge(manifest, on="accession_number", how="inner")
    data = data.drop_duplicates(["ticker", "fecha", "item_key", "text_hash", "frame_index"])
    data["seccion"] = data["item_key"].map({"prepared": "guion", "qa": "improvisado"}).fillna("otro")
    data["anio"] = pd.to_datetime(data["fecha"]).dt.year
    data["trimestre"] = pd.PeriodIndex(pd.to_datetime(data["fecha"]), freq="Q").astype(str)
    concepts = data["concepts"].apply(lambda c: set(c) if c is not None else set())
    data["conducta"] = concepts.apply(lambda s: float(bool(s & set(BEHAVIOR))))
    data["riesgo"] = concepts.apply(
        lambda s: float(any(str(c).startswith("risk_") for c in s)))
    data["gobernanza"] = concepts.apply(
        lambda s: float(any(str(c).startswith("gov_") for c in s)))
    data["promocional"] = data["rhetoric_promotional"].astype(float)
    data["cuantificado"] = data["specificity_quantified_metric"].astype(float)
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--database", type=Path, default=DB)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    import sys
    sys.path.insert(0, str(REPO_ROOT / "scripts" / "common"))
    import ai_prefilter_classify as pc
    con = pc.connect_read_only(args.database)
    try:
        calls = load_calls(con)
    finally:
        con.close()
    if calls.empty:
        print("todavía no hay frames de earnings calls clasificados")
        return
    print(f"{len(calls):,} frames de earnings calls | {calls['ticker'].nunique():,} empresas "
          f"| {calls['accession_number'].nunique():,} transcripciones "
          f"| {calls['fecha'].min()} a {calls['fecha'].max()}\n")

    print("=" * 70)
    print("1. VOLUMEN Y REGISTRO POR AÑO")
    print("=" * 70)
    by_year = calls.groupby("anio").agg(
        frames=("promocional", "size"), empresas=("ticker", "nunique"),
        pct_promocional=("promocional", lambda s: round(s.mean() * 100, 1)),
        pct_conducta=("conducta", lambda s: round(s.mean() * 100, 1)),
        pct_cuantificado=("cuantificado", lambda s: round(s.mean() * 100, 1)),
        pct_riesgo=("riesgo", lambda s: round(s.mean() * 100, 1)),
        pct_gobernanza=("gobernanza", lambda s: round(s.mean() * 100, 1)))
    print(by_year.to_string())

    print("\n" + "=" * 70)
    print("2. TEMPORALIDAD Y SUJETO — ¿de qué habla la empresa en la call?")
    print("=" * 70)
    for column in ("temporal", "subject", "ai_type", "domain"):
        distribution = (calls[column].value_counts(normalize=True) * 100).round(1)
        print(f"{column:10s} " + " | ".join(f"{k}: {v}%" for k, v in distribution.head(4).items()))

    print("\n" + "=" * 70)
    print("3. GUION vs. IMPROVISADO — lo que sólo este canal permite mirar")
    print("=" * 70)
    con2 = pc.connect_read_only(args.database)
    try:
        universe = con2.execute("""
            SELECT item_key, count(*) AS parrafos FROM unique_paragraphs
            WHERE form = 'Earnings call' GROUP BY 1
        """).fetchdf()
    finally:
        con2.close()
    universe["seccion"] = universe["item_key"].map(
        {"prepared": "guion", "qa": "improvisado"}).fillna("otro")
    sizes = universe.groupby("seccion")["parrafos"].sum()
    by_section = calls.groupby("seccion").agg(
        frames=("promocional", "size"),
        pct_promocional=("promocional", lambda s: round(s.mean() * 100, 1)),
        pct_conducta=("conducta", lambda s: round(s.mean() * 100, 1)),
        pct_cuantificado=("cuantificado", lambda s: round(s.mean() * 100, 1)),
        pct_riesgo=("riesgo", lambda s: round(s.mean() * 100, 1)))
    by_section["parrafos_del_canal"] = sizes
    # Densidad: frames de IA por cada mil párrafos de esa sección. Es la
    # comparación que corresponde, porque el guion es diez veces más corto que
    # el Q&A y un conteo crudo diría lo contrario de lo que pasa.
    by_section["frames_por_mil_parrafos"] = (
        by_section["frames"] / by_section["parrafos_del_canal"] * 1000).round(1)
    print(by_section.to_string())

    print("\n" + "=" * 70)
    print("4. QUIÉN HABLA MÁS DE IA EN SUS CALLS")
    print("=" * 70)
    by_firm = calls.groupby("ticker").agg(
        frames=("promocional", "size"), transcripciones=("accession_number", "nunique"),
        pct_promocional=("promocional", lambda s: round(s.mean() * 100, 1)),
        pct_conducta=("conducta", lambda s: round(s.mean() * 100, 1)))
    by_firm["frames_por_call"] = (by_firm["frames"] / by_firm["transcripciones"]).round(1)
    print(by_firm.nlargest(15, "frames").to_string())

    print("\n" + "=" * 70)
    print("5. CONTRA EL RESTO DEL CORPUS (referencia, no comparación formal)")
    print("=" * 70)
    con = pc.connect_read_only(args.database)
    try:
        others = con.execute("""
            SELECT form, count(*) frames,
                   avg(CASE WHEN rhetoric_promotional THEN 1.0 ELSE 0 END) promocional,
                   avg(CASE WHEN specificity_quantified_metric THEN 1.0 ELSE 0 END) cuantificado
            FROM gold_ai_frames WHERE country_code = 'us' AND has_frame
            GROUP BY 1 ORDER BY 2 DESC
        """).fetchdf()
    finally:
        con.close()
    others["promocional"] = (others["promocional"] * 100).round(1)
    others["cuantificado"] = (others["cuantificado"] * 100).round(1)
    print(others.to_string(index=False))
    print("\nOJO: los formularios difieren en error de medición del prefiltro")
    print("(precisión ponderada 0,98 en 10-K/10-Q, 0,73 en proxy/8-K; sin validación en calls),")
    print("así que esta tabla ordena magnitudes, no sostiene una comparación formal.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    destination = args.output_dir / "earnings_calls_summary.parquet"
    by_firm.reset_index().to_parquet(destination, index=False)
    (args.output_dir / "earnings_calls_summary.json").write_text(json.dumps({
        "frames": int(len(calls)), "empresas": int(calls["ticker"].nunique()),
        "transcripciones": int(calls["accession_number"].nunique()),
        "por_anio": json.loads(by_year.to_json(orient="index"))}, indent=2, default=float))
    print(f"\n-> {destination}")


if __name__ == "__main__":
    main()
