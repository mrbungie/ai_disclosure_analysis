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
entrenó con 10-K/10-Q y se ajustó en DEF 14A / 8-K; el registro hablado es
distinto. Medido sobre 800 párrafos de calls etiquetados por el juez
(`scripts/verif/prefilter_form_validation.py`, `golden_set_forms/calls/`), el
modelo desplegado (v2, umbral 0,17) queda en **precisión 0,59 y recall 0,98**
ponderados al corpus: recall casi perfecto, pero cuatro de cada diez textos
marcados no son divulgación de IA. Las tasas de abajo se calculan sobre frames
que el juez LLM confirmó (`has_frame`), que filtra buena parte de eso, pero la
comparación entre canales sigue arrastrando error de medición distinto por canal.

Uso:
    uv run python scripts/analytics/earnings_calls_analysis.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "common"))
import layers as L  # noqa: E402

BEHAVIOR = ["deployed", "pilot", "exploring", "investment",
            "infrastructure", "talent", "proprietary_ai", "third_party_ai",
            "scaling", "productivity_outcome", "revenue_outcome",
            "cost_outcome", "customer_outcome"]


def load_calls() -> pd.DataFrame:
    """Frames de earnings calls con su fecha y empresa.

    El manifiesto se lee de `silver.filing_manifest`, filtrado a
    `form_type == "Earnings call transcript"`.

    `domain` no vive en el frame en v2 (ver
    docs/migration_v1_to_v2_analytics.md §1.3) -- se trae del join a
    silver.ai_activities por (text_hash, frame_id), cobertura parcial (solo
    frames con actividad asociada), 'unspecified' si no hay match."""
    frame_domains = (L.scan("silver.ai_activities").filter(pl.col("has_activity"))
                     .group_by("text_hash", "frame_id")
                     .agg(pl.when((pl.col("domain") == "customer_facing").any()).then(pl.lit("customer_facing"))
                          .when((pl.col("domain") == "internal").any()).then(pl.lit("internal"))
                          .otherwise(pl.lit("unspecified")).alias("domain")))
    frames = (L.scan("silver.ai_frames")
              .filter((pl.col("country_code") == "us") & pl.col("has_frame")
                      & (pl.col("form") == "Earnings call"))
              .join(frame_domains, on=["text_hash", "frame_id"], how="left")
              .select("accession_number", "item_key", "text_hash", "frame_id", "subject", "ai_type",
                      "temporal", pl.col("domain").fill_null("unspecified"),
                      "concepts", "rhetoric", "specificity")
              .collect().to_pandas())
    # `silver.filing_manifest` identifica cada transcripción con `document_id`
    # (ej. "AAP_2022Q3"), que es lo que el extractor usó como
    # `accession_number` en `silver.ai_frames` — las calls no tienen número de
    # accession de la SEC porque no son un filing (su columna
    # `accession_number` en el manifiesto viene nula para ellas). `filing_date`
    # ya llega como fecha en silver, a diferencia del manifiesto interim donde
    # era texto.
    manifest = (L.scan("silver.filing_manifest")
                .filter((pl.col("form_type") == "Earnings call transcript")
                        & pl.col("ticker").is_not_null())
                .select(pl.col("document_id").alias("accession_number"), "ticker",
                        pl.col("filing_date").alias("fecha"))
                .collect().to_pandas())
    data = frames.merge(manifest, on="accession_number", how="inner")
    data = data.drop_duplicates(["ticker", "fecha", "item_key", "text_hash", "frame_id"])
    data["seccion"] = data["item_key"].map({"prepared": "guion", "qa": "improvisado"}).fillna("otro")
    data["anio"] = pd.to_datetime(data["fecha"]).dt.year
    data["trimestre"] = pd.PeriodIndex(pd.to_datetime(data["fecha"]), freq="Q").astype(str)
    concepts = data["concepts"].apply(lambda c: set(c) if c is not None else set())
    data["conducta"] = concepts.apply(lambda s: float(bool(s & set(BEHAVIOR))))
    data["riesgo"] = concepts.apply(
        lambda s: float(any(str(c).startswith("risk_") for c in s)))
    data["gobernanza"] = concepts.apply(
        lambda s: float(any(str(c).startswith("gov_") for c in s)))
    data["promocional"] = data["rhetoric"].apply(
        lambda r: float("promotional" in list(r)) if r is not None else 0.0)
    data["cuantificado"] = data["specificity"].apply(
        lambda s: float("metric" in list(s)) if s is not None else 0.0)
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    calls = load_calls()
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
    universe = (L.scan("bronze.unique_paragraphs").filter(pl.col("form") == "Earnings call")
                .group_by("item_key").agg(pl.len().cast(pl.Int64).alias("parrafos"))
                .collect().to_pandas())
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
    others = (L.scan("silver.ai_frames").filter((pl.col("country_code") == "us") & pl.col("has_frame"))
              .group_by("form")
              .agg(pl.len().cast(pl.Int64).alias("frames"),
                   pl.col("rhetoric").list.contains("promotional").cast(pl.Int32).mean().alias("promocional"),
                   pl.col("specificity").list.contains("metric").cast(pl.Int32).mean().alias("cuantificado"))
              .sort(["frames", "form"], descending=[True, False])
              .collect().to_pandas())
    others["promocional"] = (others["promocional"] * 100).round(1)
    others["cuantificado"] = (others["cuantificado"] * 100).round(1)
    print(others.to_string(index=False))
    print("\nOJO: los formularios difieren en error de medición del prefiltro")
    print("(precisión ponderada 0,98 en 10-K/10-Q, 0,73 en proxy/8-K, 0,59 en calls),")
    print("así que esta tabla ordena magnitudes, no sostiene una comparación formal.")

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        destination = args.output_dir / "earnings_calls_summary.parquet"
        json_destination = args.output_dir / "earnings_calls_summary.json"
    else:
        destination = L.results_path("appendix", "earnings_calls_summary.parquet")
        json_destination = L.results_path("appendix", "earnings_calls_summary.json")
    by_firm.reset_index().to_parquet(destination, index=False)
    json_destination.write_text(json.dumps({
        "frames": int(len(calls)), "empresas": int(calls["ticker"].nunique()),
        "transcripciones": int(calls["accession_number"].nunique()),
        "por_anio": json.loads(by_year.to_json(orient="index"))}, indent=2, default=float))
    print(f"\n-> {destination}")


if __name__ == "__main__":
    main()
