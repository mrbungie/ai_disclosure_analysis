"""Margen extensivo: cada documento del corpus cuenta, tenga o no frames de IA.

Los análisis sobre `silver.ai_frames` condicionan a hablar de IA (una
empresa-año entra al panel si tiene ≥3 frames, una empresa al score si tiene
≥5, etc.). Eso selecciona sobre el propio fenómeno: la empresa que deja de
hablar de IA sale del panel en vez de contar como cero. Este módulo arma la
tabla de DOCUMENTOS —10-K, 10-Q, DEF 14A, 8-K y earnings calls de EE.UU.— con
su cantidad de párrafos y sus conteos de frames (cero si no tiene), para que
cualquier análisis pueda correrse sobre intensidades por 1.000 palabras con
los ceros adentro.

    from ai_intensity import document_table, aggregate
    docs = document_table()                          # una fila por documento
    fy   = aggregate(docs, ["ticker", "fy"])         # intensidades por empresa-ejercicio
    q    = aggregate(docs, ["ticker", "quarter"], forms=("10-K", "10-Q"))

Año fiscal alineado por lo que el documento cubre (ver
`channel_gap_analysis.py`): 10-K/10-Q por `period_end_date` con el mes de
cierre de cada empresa; calls por su `fiscal_period` (el trimestre fiscal que
nombra la transcripción, scripts/bronze/call_transcripts.py);
DEF 14A y 8-K por fecha de presentación. `quarter` es el trimestre calendario
de la fecha de presentación (lo que usan los shocks).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "common"))
import layers as L  # noqa: E402

# Tres fuentes de transcripciones, todas en `silver.filing_manifest`: la base
# de Hugging Face (01_fetch_transcripts.py, 2005-2025, no se actualiza) más
# dos rellenos de huecos (03/04, scripts/us/earnings_calls/) que sí cubren
# 2026. Si un mismo document_id apareciera en más de una, gana la primera de
# esta lista (prefijo de `source`).
CALL_SOURCE_PRIORITY = ["huggingface:", "equibles:", "stockanalysis.com:"]
CALL_FORM_TYPE = "Earnings call transcript"
FILING_FORMS = ("10-K", "10-Q", "DEF 14A", "8-K")


def _calls_manifest() -> pl.LazyFrame:
    """Una fila por document_id de earnings call del universo de análisis
    (`silver.filing_manifest` ya viene filtrado al S&P 500 a 2021-01-01)."""
    rank = pl.lit(len(CALL_SOURCE_PRIORITY))
    for i, prefix in reversed(list(enumerate(CALL_SOURCE_PRIORITY))):
        rank = pl.when(pl.col("source").str.starts_with(prefix)).then(pl.lit(i)).otherwise(rank)
    return (L.scan("silver.filing_manifest")
            .filter(pl.col("form_type") == CALL_FORM_TYPE)
            .select("document_id", "ticker", "filing_date", "fiscal_period", rank.alias("_rank"))
            .sort(["document_id", "_rank", "ticker"], nulls_last=True)
            .unique("document_id", keep="first", maintain_order=True)
            .drop("_rank"))


COUNT_COLUMNS = ["n_frames", "n_promo", "n_quant", "n_spec", "n_risk", "n_gov", "n_hyp", "n_realized",
                 "n_deployed", "n_revenue_outcome", "n_cost_outcome", "n_ai_investment", "n_ai_infrastructure"]


def _any_concept_like(prefix: str) -> pl.Expr:
    """¿Algún concepto calza con el patrón SQL LIKE '<prefix>_%'? (`_` es un
    carácter cualquiera, así que exige al menos uno después del prefijo)."""
    return pl.col("concepts").list.eval(pl.element().str.contains(f"(?s)^{prefix}.")).list.any()


def _frame_counts() -> pd.DataFrame:
    """Conteos de frames por documento (accession_number), sólo documentos con frames."""
    flag = lambda e: e.cast(pl.Int32).sum()  # noqa: E731
    return (L.scan("silver.ai_frames")
            .filter(pl.col("has_frame"))
            .group_by("accession_number")
            .agg(pl.len().cast(pl.Int64).alias("n_frames"),
                 flag(pl.col("rhetoric").list.contains("promotional")).alias("n_promo"),
                 flag(pl.col("specificity").list.contains("metric")).alias("n_quant"),
                 (pl.col("specificity").list.len() / 5.0).sum().alias("n_spec"),
                 flag(_any_concept_like("risk")).alias("n_risk"),
                 flag(_any_concept_like("gov")).alias("n_gov"),
                 flag(pl.col("temporal") == "hypothetical").alias("n_hyp"),
                 flag(pl.col("temporal") == "realized").alias("n_realized"),
                 flag(pl.col("concepts").list.contains("deployed")).alias("n_deployed"),
                 flag(pl.col("concepts").list.contains("revenue_outcome")).alias("n_revenue_outcome"),
                 flag(pl.col("concepts").list.contains("cost_outcome")).alias("n_cost_outcome"),
                 flag(pl.col("concepts").list.contains("investment")).alias("n_ai_investment"),
                 flag(pl.col("concepts").list.contains("infrastructure")).alias("n_ai_infrastructure"))
            .collect()
            .to_pandas())


def _paragraph_counts() -> pl.LazyFrame:
    """Párrafos puntuables y palabras por documento. Palabras = trozos al
    partir el texto (sin espacios en los extremos) por corridas de blancos
    ASCII."""
    return (L.scan("bronze.paragraphs")
            .filter(pl.col("is_scorable"))
            .select("accession_number", "paragraph_text")
            .group_by("accession_number")
            .agg(pl.len().cast(pl.Int64).alias("n_paragraphs"),
                 (pl.col("paragraph_text").str.strip_chars(" ").str.count_matches(r"[\t\n\f\r ]+") + 1)
                 .cast(pl.Int64).sum().alias("n_words")))


def document_table() -> pd.DataFrame:
    """Una fila por documento de EE.UU. con párrafos puntuables y conteos de
    frames (cero si el documento no habla de IA)."""
    manifest_cols = ["accession_number", "ticker", "cik", "filing_date", "period_end_date"]
    manifest = pl.concat([
        L.scan("silver.filing_manifest").filter(pl.col("form_type") != CALL_FORM_TYPE)
        .select(*manifest_cols, pl.col("form_type").alias("form")),
        L.scan("silver.filing_manifest_10q").select(*manifest_cols, pl.lit("10-Q").alias("form")),
    ])
    paras = _paragraph_counts()
    filings = (manifest.join(paras, on="accession_number", how="inner")
               .filter(pl.col("ticker").is_not_null() & pl.col("filing_date").is_not_null()
                       & pl.col("form").is_in(FILING_FORMS))
               .select(pl.lit("filing").alias("channel"), "form", "ticker", "cik",
                       pl.col("filing_date").alias("fecha"), pl.col("period_end_date").alias("period_end"),
                       pl.lit(None, dtype=pl.Int32).alias("call_fy"),
                       "accession_number", "n_paragraphs", "n_words"))
    calls = (_calls_manifest().join(paras, left_on="document_id", right_on="accession_number", how="inner")
             .filter(pl.col("ticker").is_not_null())
             .select(pl.lit("call").alias("channel"), pl.lit("Earnings call").alias("form"), "ticker",
                     pl.lit(None, dtype=pl.String).alias("cik"), pl.col("filing_date").alias("fecha"),
                     pl.lit(None, dtype=pl.Date).alias("period_end"),
                     pl.col("fiscal_period").str.slice(0, 4).cast(pl.Int32).alias("call_fy"),
                     pl.col("document_id").alias("accession_number"), "n_paragraphs", "n_words"))
    docs = (pl.concat([filings, calls]).with_columns(pl.col("n_words").cast(pl.Float64))
            .collect(engine="streaming").to_pandas())
    docs = docs.astype({"fecha": "datetime64[us]", "period_end": "datetime64[us]", "call_fy": "Int32"})
    docs = docs.merge(_frame_counts(), on="accession_number", how="left")
    docs[COUNT_COLUMNS] = docs[COUNT_COLUMNS].fillna(0.0)
    docs["fecha"] = pd.to_datetime(docs["fecha"])
    docs["quarter"] = docs["fecha"].dt.to_period("Q")
    # mes de cierre fiscal por empresa: moda del period_end de sus 10-K; diciembre si no hay
    fye = (docs[(docs.form == "10-K") & docs.period_end.notna()]
           .assign(m=lambda d: pd.to_datetime(d.period_end).dt.month)
           .groupby("ticker")["m"].agg(lambda s: int(s.mode().iloc[0])))
    docs["fye_month"] = docs["ticker"].map(fye).fillna(12).astype(int)
    pe = pd.to_datetime(docs["period_end"])
    fy_date = docs["fecha"].dt.year + (docs["fecha"].dt.month > docs["fye_month"]).astype(int)
    fy_pe = pe.dt.year + (pe.dt.month > docs["fye_month"]).astype(int)
    is_pe = docs["form"].isin(["10-K", "10-Q"]) & pe.notna()
    docs["fy"] = np.where(docs["channel"] == "call", docs["call_fy"], np.where(is_pe, fy_pe, fy_date)).astype(int)
    return docs


def aggregate(docs: pd.DataFrame, keys: list[str], forms: tuple[str, ...] | None = None,
              min_paragraphs: int = 1) -> pd.DataFrame:
    """Suma documentos por `keys` y devuelve intensidades por 1.000 PALABRAS
    (`*_per_1k`) más los conteos y `any_ai`. `forms` restringe los formularios.

    La unidad de intensidad es palabras, no párrafos: un párrafo de call
    (turno de conversación, ~129 palabras en promedio) y uno de filing
    (~67 palabras) no son la misma unidad, y contarlos como si lo fueran
    infla artificialmente cualquier comparación entre venues (ver Cap. 6).
    Esta es la única definición de `*_per_1k` en todo el proyecto: todo lo
    que consume esta función —segmentación, regresiones, event studies—
    hereda automáticamente la unidad de palabras."""
    d = docs if forms is None else docs[docs["form"].isin(forms)]
    g = d.groupby(keys).agg(n_docs=("accession_number", "nunique"), n_paragraphs=("n_paragraphs", "sum"),
                            n_words=("n_words", "sum"),
                            **{c: (c, "sum") for c in COUNT_COLUMNS}).reset_index()
    g = g[g["n_paragraphs"] >= min_paragraphs]
    for c in COUNT_COLUMNS:
        g[c.replace("n_", "", 1) + "_per_1k"] = 1000.0 * g[c] / g["n_words"]
    g["any_ai"] = (g["n_frames"] > 0).astype(float)
    # tasas condicionales, NaN cuando no hay frames (para comparar con el margen intensivo)
    with np.errstate(divide="ignore", invalid="ignore"):
        for c in ("n_promo", "n_quant", "n_spec", "n_risk", "n_gov", "n_hyp", "n_realized"):
            g[c.replace("n_", "", 1) + "_rate"] = np.where(g["n_frames"] > 0, g[c] / g["n_frames"], np.nan)
    return g


if __name__ == "__main__":
    docs = document_table()
    print(f"documentos: {len(docs):,} | con frames: {int((docs.n_frames > 0).sum()):,} | por form:")
    print(docs.groupby("form").agg(docs=("accession_number", "size"), con_ia=("n_frames", lambda s: int((s > 0).sum())),
                                   parrafos=("n_paragraphs", "sum"), frames=("n_frames", "sum")).to_string())
    fy = aggregate(docs, ["ticker", "fy"], forms=FILING_FORMS)
    print(f"\nempresa-ejercicio (filings): {len(fy):,} | con IA: {int(fy.any_ai.sum()):,} | promo/1k media {fy.promo_per_1k.mean():.2f}")


def firm_intensity(keys: list[str] = ("ticker",)) -> pd.DataFrame:
    """Intensidad de IA de los FILINGS por empresa (o empresa-año): todas las
    empresas con filings, incluidas las que no hablan de IA (frames_per_1k = 0).
    Es la lista de unidades sobre la que se construyen segmentos y grilla:
    una empresa sin frames entra con sus tasas en el prior y su intensidad en
    cero, no se excluye."""
    docs = document_table()
    docs = docs[docs["form"].isin(FILING_FORMS)].assign(year=lambda d: d["fecha"].dt.year)
    return aggregate(docs, list(keys))[list(keys) + ["n_docs", "n_paragraphs", "n_words", "n_frames", "frames_per_1k", "any_ai"]]
