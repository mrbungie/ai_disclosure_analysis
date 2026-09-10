"""Margen extensivo: cada documento del corpus cuenta, tenga o no frames de IA.

Los análisis sobre `gold_ai_frames` condicionan a hablar de IA (una
empresa-año entra al panel si tiene ≥3 frames, una empresa al score si tiene
≥5, etc.). Eso selecciona sobre el propio fenómeno: la empresa que deja de
hablar de IA sale del panel en vez de contar como cero. Este módulo arma la
tabla de DOCUMENTOS —10-K, 10-Q, DEF 14A, 8-K y earnings calls de EE.UU.— con
su cantidad de párrafos y sus conteos de frames (cero si no tiene), para que
cualquier análisis pueda correrse sobre intensidades por 1.000 palabras con
los ceros adentro.

    from ai_intensity import document_table, aggregate
    docs = document_table(con)                       # una fila por documento
    fy   = aggregate(docs, ["ticker", "fy"])         # intensidades por empresa-ejercicio
    q    = aggregate(docs, ["ticker", "quarter"], forms=("10-K", "10-Q"))

Año fiscal alineado por lo que el documento cubre (ver
`channel_gap_analysis.py`): 10-K/10-Q por `period_end_date` con el mes de
cierre de cada empresa; calls por el trimestre fiscal de su `document_id`;
DEF 14A y 8-K por fecha de presentación. `quarter` es el trimestre calendario
de la fecha de presentación (lo que usan los shocks).
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
# Tres fuentes de transcripciones, cada una con su propio manifest: la base
# de Hugging Face (01_fetch_transcripts.py, 2005-2025, no se actualiza) más
# dos rellenos de huecos (03/04, scripts/us/earnings_calls/) que sí cubren
# 2026. Leer sólo la base deja fuera cualquier call que exista únicamente en
# un relleno -- todo 2026 y parte de 2025.
CALLS_MANIFESTS = [
    REPO_ROOT / "data" / "interim" / "manifests" / "filing_manifest_earnings_calls.parquet",
    REPO_ROOT / "data" / "interim" / "manifests" / "filing_manifest_earnings_calls_equibles.parquet",
    REPO_ROOT / "data" / "interim" / "manifests" / "filing_manifest_earnings_calls_stockanalysis.parquet",
]
FILING_FORMS = ("10-K", "10-Q", "DEF 14A", "8-K")


def _calls_manifest_sql() -> str:
    """UNION ALL BY NAME de las tres fuentes de calls, una fila por
    document_id (la primera fuente que lo tenga gana, mismo criterio que
    build_duckdb.py aplica en la vista `filing_manifest`)."""
    parts = [f"SELECT document_id, ticker, filing_date FROM read_parquet('{p}')" for p in CALLS_MANIFESTS if p.exists()]
    union = " UNION ALL BY NAME ".join(parts)
    return f"""
        SELECT document_id, ticker, TRY_CAST(filing_date AS DATE) AS filing_date FROM (
            SELECT *, row_number() OVER (PARTITION BY document_id ORDER BY 1) AS rn
            FROM ({union})
        ) WHERE rn = 1
    """
COUNT_COLUMNS = ["n_frames", "n_promo", "n_quant", "n_spec", "n_risk", "n_gov", "n_hyp", "n_realized",
                 "n_deployed", "n_revenue_outcome", "n_cost_outcome", "n_ai_investment", "n_ai_infrastructure"]


def _frame_counts(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Conteos de frames por documento (accession_number), sólo documentos con frames."""
    return con.execute("""
        SELECT accession_number,
               count(*) AS n_frames,
               sum(rhetoric_promotional::INT) AS n_promo,
               sum(specificity_quantified_metric::INT) AS n_quant,
               sum((specificity_business_process::INT + specificity_product_or_system::INT
                    + specificity_vendor_or_partner::INT + specificity_quantified_metric::INT
                    + specificity_date_or_timeline::INT) / 5.0) AS n_spec,
               sum(list_bool_or(list_transform(concepts, c -> c LIKE 'risk_%'))::INT) AS n_risk,
               sum(list_bool_or(list_transform(concepts, c -> c LIKE 'gov_%'))::INT) AS n_gov,
               sum((temporal = 'hypothetical')::INT) AS n_hyp,
               sum((temporal = 'realized')::INT) AS n_realized,
               sum(list_contains(concepts, 'deployed')::INT) AS n_deployed,
               sum(list_contains(concepts, 'revenue_outcome')::INT) AS n_revenue_outcome,
               sum(list_contains(concepts, 'cost_outcome')::INT) AS n_cost_outcome,
               sum(list_contains(concepts, 'ai_investment')::INT) AS n_ai_investment,
               sum(list_contains(concepts, 'ai_infrastructure')::INT) AS n_ai_infrastructure
        FROM gold_ai_frames WHERE country_code = 'us' AND has_frame
        GROUP BY 1
    """).df()


def document_table(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Una fila por documento de EE.UU. con párrafos puntuables y conteos de
    frames (cero si el documento no habla de IA)."""
    docs = con.execute(rf"""
        WITH manifest AS (
            SELECT country_code, accession_number, ticker, cik, filing_date, period_end_date, form_type AS form
            FROM filing_manifest WHERE form_type != 'Earnings call transcript'
            UNION ALL
            SELECT country_code, accession_number, ticker, cik, filing_date, period_end_date, '10-Q'
            FROM filing_manifest_10q
        ), paras AS (
            SELECT country_code, accession_number, count(*) AS n_paragraphs,
                   sum(list_count(regexp_split_to_array(trim(paragraph_text), '\s+'))) AS n_words
            FROM paragraphs WHERE is_scorable GROUP BY 1, 2
        )
        SELECT 'filing' AS channel, m.form, m.ticker, m.cik, m.filing_date AS fecha,
               TRY_CAST(m.period_end_date AS DATE) AS period_end, NULL::INTEGER AS call_fy,
               m.accession_number, p.n_paragraphs, p.n_words
        FROM manifest m JOIN paras p USING (country_code, accession_number)
        WHERE m.country_code = 'us' AND m.ticker IS NOT NULL AND m.filing_date IS NOT NULL AND m.form IN {FILING_FORMS}
        UNION ALL
        SELECT 'call', 'Earnings call', m.ticker, NULL, m.filing_date, NULL::DATE,
               CAST(regexp_extract(m.document_id, '_([0-9]{{4}})Q', 1) AS INTEGER), m.document_id, p.n_paragraphs, p.n_words
        FROM ({_calls_manifest_sql()}) m
        JOIN paras p ON p.accession_number = m.document_id AND p.country_code = 'us'
        WHERE m.ticker IS NOT NULL
    """).df()
    docs = docs.merge(_frame_counts(con), on="accession_number", how="left")
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
    con = duckdb.connect(str(REPO_ROOT / "duckdb" / "thesis.duckdb"), read_only=True)
    docs = document_table(con)
    print(f"documentos: {len(docs):,} | con frames: {int((docs.n_frames > 0).sum()):,} | por form:")
    print(docs.groupby("form").agg(docs=("accession_number", "size"), con_ia=("n_frames", lambda s: int((s > 0).sum())),
                                   parrafos=("n_paragraphs", "sum"), frames=("n_frames", "sum")).to_string())
    fy = aggregate(docs, ["ticker", "fy"], forms=FILING_FORMS)
    print(f"\nempresa-ejercicio (filings): {len(fy):,} | con IA: {int(fy.any_ai.sum()):,} | promo/1k media {fy.promo_per_1k.mean():.2f}")


def firm_intensity(con: duckdb.DuckDBPyConnection, keys: list[str] = ("ticker",)) -> pd.DataFrame:
    """Intensidad de IA de los FILINGS por empresa (o empresa-año): todas las
    empresas con filings, incluidas las que no hablan de IA (frames_per_1k = 0).
    Es la lista de unidades sobre la que se construyen segmentos y grilla:
    una empresa sin frames entra con sus tasas en el prior y su intensidad en
    cero, no se excluye."""
    docs = document_table(con)
    docs = docs[docs["form"].isin(FILING_FORMS)].assign(year=lambda d: d["fecha"].dt.year)
    return aggregate(docs, list(keys))[list(keys) + ["n_docs", "n_paragraphs", "n_words", "n_frames", "frames_per_1k", "any_ai"]]
