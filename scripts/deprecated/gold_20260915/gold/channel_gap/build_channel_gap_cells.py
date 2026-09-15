"""Brecha entre canales, la TABLA DE CELDAS: la misma empresa, el mismo
período, earnings call contra filing SEC -- sólo la construcción de las
celdas comparables (frames/documentos por canal, sus tasas y la brecha
`call - filing`), sin ningún ajuste estadístico.

    gap[i, t] = y[i, call, t] − y[i, filing, t]

La estimación (FE de empresa, event study, robustez, casos notorios, cruce
con el washing score) vive en `scripts/analytics/channel_gap/channel_gap_did.py`,
que lee las salidas de este script -- gold sólo construye el dato (G-H2).

PERÍODO = AÑO FISCAL, alineado por lo que el documento CUBRE, no por la
fecha en que se presenta. El 10-K de febrero de 2025 habla del ejercicio
2024 y las calls de 2024 discuten los trimestres de 2024: parearlos por
año calendario de la fecha mezcla ejercicios. Asignación:
  10-K, 10-Q       año fiscal de `period_end_date`
  earnings call    año fiscal del `document_id` (`TICKER_YYYYQn` = trimestre
                   fiscal discutido)
  8-K, DEF 14A     año fiscal en que se presentan (no cubren un período;
                   el proxy mezcla compensación pasada y gobernanza actual)
El año fiscal de una fecha se calcula con el mes de cierre de cada empresa
(mes de `period_end_date` de sus 10-K). Los filings con frames de IA son
anuales (el 10-Q casi nunca tiene frames), así que la unidad natural es el
año: por trimestre calendario sólo hay celda donde un 10-K coincide con
una call. `--period quarter` conserva esa versión, por fecha calendario.

Una celda puede mezclar documentos anteriores y posteriores al 2023-12-05
(aviso de Gensler sobre AI-washing) -- p. ej. el 10-K de FY2023 presentado
en febrero y el proxy de abril -- por eso se guarda `share_post_docs` por
celda, y `mixed`/`weight` para que la estimación pueda excluir esas celdas
o ponderar por tamaño.

Salidas:
  data/gold/covariates/firm_year/channel_gap_cells_extensive.parquet
      una fila por empresa-ejercicio con >=1 documento por canal (MODO
      FINAL: intensidad por 1.000 palabras, con ceros)
  data/gold/covariates/firm_year/channel_gap_cells_paired.parquet
      una fila por empresa-ejercicio con >=MIN_FRAMES frames en AMBOS
      canales (celdas con suficiente señal para las tasas retóricas)

`firm_gap()` (brecha promedio por empresa, pooled sobre todos sus períodos
-- sin dependencia de un cutoff, no hay una versión walk-forward de este
comparativo) es una simple media por grupo sobre `channel_gap_cells_paired`
y no se persiste como parquet firm-grain aparte: `channel_gap_did.py` (su
único lector) la calcula al leer, siguiendo la regla de
docs/gold_pipeline.md de no duplicar un grain `firm` cuando es derivable en
el momento de la lectura.

Determinístico, sin LLM. Lee `silver.ai_frames`, `silver.filing_manifest`,
`silver.filing_manifest_10q` y `bronze.paragraphs` (`make bronze silver`).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "common"))
import layers as L  # noqa: E402

FILING_FORMS = ("10-K", "10-Q", "DEF 14A", "8-K")
MIN_FRAMES = 3
EVENT = {"fy": 2024, "quarter": pd.Period("2024Q2", freq="Q")}
EVENT_DATE = "2023-12-05"      # aviso de Gensler sobre AI-washing; el enforcement es del 2024-03-18
OUTCOMES = ["promotional_rate", "quantified_rate", "specificity_index", "realized_share",
            "hypothetical_share", "gov_share"]
EXT_OUTCOMES = ["frames_per_1k", "promo_per_1k", "quant_per_1k", "gov_per_1k", "any_ai"]


def _calls_manifest() -> pl.LazyFrame:
    """Una fila por transcripción (`document_id` = `TICKER_YYYYQn`). Las tres
    fuentes de transcripciones (ver ai_intensity.py para el detalle: la base
    de Hugging Face, 2005-2025, más dos rellenos de huecos que sí cubren 2026)
    ya vienen unidas y restringidas al universo de análisis en
    `silver.filing_manifest`."""
    return (L.scan("silver.filing_manifest")
            .filter((pl.col("form_type") == "Earnings call transcript") & pl.col("ticker").is_not_null())
            .select("document_id", "ticker", "filing_date"))


def _filings_manifest(form_col: bool = False) -> pl.LazyFrame:
    """Filings de EE.UU. (10-K, DEF 14A, 8-K, ... y 10-Q), sin transcripciones."""
    cols = ["country_code", "accession_number", "ticker", "filing_date", "period_end_date"]
    main = L.scan("silver.filing_manifest").filter(pl.col("form_type") != "Earnings call transcript")
    tenq = L.scan("silver.filing_manifest_10q")
    if form_col:
        return pl.concat([main.select(cols + [pl.col("form_type").alias("form")]),
                          tenq.select(cols + [pl.lit("10-Q").alias("form")])])
    return pl.concat([main.select(cols), tenq.select(cols)])


def _call_fy(col: str) -> pl.Expr:
    """Año fiscal de la call, desde el `document_id` sintético `TICKER_YYYYQn`."""
    return pl.col(col).str.extract(r"_([0-9]{4})Q", 1).cast(pl.Int32).alias("call_fy")


def load_frames() -> pd.DataFrame:
    """Frames de EE.UU. con empresa, fecha y canal. Los filings resuelven
    ticker/fecha por `silver.filing_manifest` (10-K, DEF 14A, 8-K) y
    `silver.filing_manifest_10q`; las calls por su manifiesto propio, cuyo
    `document_id` es el `accession_number` sintético `TICKER_YYYYQn`."""
    ai_frames = L.scan("silver.ai_frames").filter((pl.col("country_code") == "us") & pl.col("has_frame"))
    frame_cols = ["text_hash", "frame_id", "rhetoric", "specificity", "temporal", "concepts"]
    # orden determinístico antes del drop_duplicates de abajo: un mismo frame
    # repetido en dos documentos del mismo día conserva la instancia del form
    # que ordena primero (10-K < 10-Q < 8-K < DEF 14A)
    filings = (ai_frames.filter(pl.col("form").is_in(FILING_FORMS))
               .join(_filings_manifest(), on=["country_code", "accession_number"], how="inner")
               .filter(pl.col("ticker").is_not_null() & pl.col("filing_date").is_not_null())
               .sort(["form", "accession_number", "item_key", "paragraph_index", "frame_id"])
               .select(pl.lit("filing").alias("channel"), "form", "ticker", pl.col("filing_date").alias("fecha"),
                       pl.col("period_end_date").alias("period_end"), *frame_cols)
               .collect().to_pandas())
    calls = (ai_frames.filter(pl.col("form") == "Earnings call")
             .join(_calls_manifest(), left_on="accession_number", right_on="document_id", how="inner")
             .sort(["accession_number", "item_key", "paragraph_index", "frame_id"])
             .select(pl.lit("call").alias("channel"), "form", "ticker", pl.col("filing_date").alias("fecha"),
                     pl.lit(None, dtype=pl.Date).alias("period_end"), _call_fy("accession_number"), *frame_cols)
             .collect().to_pandas())
    frames = pd.concat([filings, calls], ignore_index=True)
    frames = frames.drop_duplicates(["channel", "ticker", "fecha", "text_hash", "frame_id"])
    frames["quarter"] = pd.to_datetime(frames["fecha"]).dt.to_period("Q")
    frames["year"] = frames["quarter"].dt.year
    # mes de cierre fiscal por empresa: el de sus 10-K (moda); diciembre si no hay
    fye = (frames[(frames.form == "10-K") & frames.period_end.notna()]
           .assign(m=lambda d: pd.to_datetime(d.period_end).dt.month)
           .groupby("ticker")["m"].agg(lambda s: int(s.mode().iloc[0])))
    frames["fye_month"] = frames["ticker"].map(fye).fillna(12).astype(int)
    def fiscal_year(dates: pd.Series, fye_month: pd.Series) -> pd.Series:
        d = pd.to_datetime(dates)
        return (d.dt.year + (d.dt.month > fye_month).astype(int)).astype("Int64")
    fy = pd.Series(pd.NA, index=frames.index, dtype="Int64")
    is_pe = frames.form.isin(["10-K", "10-Q"]) & frames.period_end.notna()
    fy[is_pe] = fiscal_year(frames.loc[is_pe, "period_end"], frames.loc[is_pe, "fye_month"])
    is_call = frames.channel == "call"
    fy[is_call] = frames.loc[is_call, "call_fy"].astype("Int64")
    rest = fy.isna()
    fy[rest] = fiscal_year(frames.loc[rest, "fecha"], frames.loc[rest, "fye_month"])
    frames["fy"] = fy.astype(int)
    frames["post_doc"] = (pd.to_datetime(frames["fecha"]) >= pd.Timestamp(EVENT_DATE)).astype(float)
    frames["is_promo"] = frames["rhetoric"].apply(
        lambda r: "promotional" in list(r) if r is not None else False)
    frames["is_quant"] = frames["specificity"].apply(
        lambda s: "metric" in list(s) if s is not None else False)
    # `specificity` overwritten in place: from the raw v2 list (process/
    # product/vendor/metric/timeline, up to 5 entries) to the same 0-1 index
    # v1 reported (mean of the 5 specificity flags == len(list)/5 here).
    frames["specificity"] = frames["specificity"].apply(lambda s: len(s) / 5.0 if s is not None else 0.0)
    frames["is_gov"] = frames["concepts"].apply(
        lambda c: any(str(x).startswith("gov_") for x in (list(c) if c is not None else [])))
    return frames


def document_counts() -> pd.DataFrame:
    """Una fila por documento (filing o call) con sus párrafos puntuables y
    palabras. Palabras = trozos del párrafo sin espacios de borde partido por
    espacios en blanco ASCII; un párrafo vacío cuenta como una palabra."""
    paras = (L.scan("bronze.paragraphs")
             .filter(pl.col("is_scorable"))
             .group_by(["country_code", "accession_number"])
             .agg(pl.len().cast(pl.Int64).alias("n_paragraphs"),
                  (pl.col("paragraph_text").str.strip_chars(" ")
                   .str.count_matches(r"[\t\n\f\r ]+") + 1).sum().cast(pl.Float64).alias("n_words")))
    filings = (_filings_manifest(form_col=True)
               .join(paras, on=["country_code", "accession_number"], how="inner")
               .filter((pl.col("country_code") == "us") & pl.col("ticker").is_not_null()
                       & pl.col("filing_date").is_not_null() & pl.col("form").is_in(FILING_FORMS))
               .select(pl.lit("filing").alias("channel"), "form", "ticker", pl.col("filing_date").alias("fecha"),
                       pl.col("period_end_date").alias("period_end"), pl.lit(None, dtype=pl.Int32).alias("call_fy"),
                       "accession_number", "n_paragraphs", "n_words"))
    calls = (_calls_manifest()
             .join(paras.filter(pl.col("country_code") == "us"), left_on="document_id",
                   right_on="accession_number", how="inner")
             .select(pl.lit("call").alias("channel"), pl.lit("Earnings call").alias("form"), "ticker",
                     pl.col("filing_date").alias("fecha"), pl.lit(None, dtype=pl.Date).alias("period_end"),
                     _call_fy("document_id"), pl.col("document_id").alias("accession_number"),
                     "n_paragraphs", "n_words"))
    return pl.concat([filings, calls]).collect(engine="streaming").to_pandas()


def load_documents(frames: pd.DataFrame) -> pd.DataFrame:
    """Todos los documentos con su cantidad de párrafos, tengan o no frames de
    IA: una call que no menciona IA es una observación con cero, no una celda
    perdida. Mismo año fiscal que en `load_frames`."""
    docs = document_counts()
    fye = frames.groupby("ticker")["fye_month"].first()
    docs["fye_month"] = docs["ticker"].map(fye).fillna(12).astype(int)
    d = pd.to_datetime(docs["fecha"]); pe = pd.to_datetime(docs["period_end"])
    fy_date = d.dt.year + (d.dt.month > docs["fye_month"]).astype(int)
    fy_pe = pe.dt.year + (pe.dt.month > docs["fye_month"]).astype(int)
    is_pe = docs["form"].isin(["10-K", "10-Q"]) & pe.notna()
    docs["fy"] = np.where(docs["channel"] == "call", docs["call_fy"], np.where(is_pe, fy_pe, fy_date)).astype(int)
    docs["post_doc"] = (d >= pd.Timestamp(EVENT_DATE)).astype(float)
    # frames por documento (0 si no tiene)
    per_doc = frames.groupby(["channel", "ticker", "fy"]).agg(
        n_frames=("text_hash", "size"), n_promo=("is_promo", "sum"),
        n_quant=("is_quant", "sum"), n_gov=("is_gov", "sum")).reset_index()
    cell = docs.groupby(["ticker", "fy", "channel"]).agg(
        n_docs=("accession_number", "nunique"), n_paragraphs=("n_paragraphs", "sum"),
        n_words=("n_words", "sum"),
        share_post_docs=("post_doc", "mean")).reset_index()
    cell = cell.merge(per_doc, on=["channel", "ticker", "fy"], how="left").fillna({"n_frames": 0, "n_promo": 0, "n_quant": 0, "n_gov": 0})
    for k in ("frames", "promo", "quant", "gov"):
        cell[f"{k}_per_1k"] = 1000.0 * cell[f"n_{k}"] / cell["n_words"]
    cell["any_ai"] = (cell["n_frames"] > 0).astype(float)
    return cell


def cells_extensive(cell: pd.DataFrame) -> pd.DataFrame:
    """Una fila por empresa × ejercicio con AL MENOS UN documento en cada canal.
    Sin umbral de frames: los ceros son datos."""
    wide = cell.pivot(index=["ticker", "fy"], columns="channel", values=EXT_OUTCOMES + ["n_docs", "n_paragraphs", "n_frames", "share_post_docs"])
    wide.columns = [f"{v}_{ch}" for v, ch in wide.columns]
    wide = wide.dropna(subset=["n_docs_call", "n_docs_filing"]).reset_index().rename(columns={"fy": "t"})
    for y in EXT_OUTCOMES:
        wide[f"gap_{y}"] = wide[f"{y}_call"] - wide[f"{y}_filing"]
    wide["post"] = (wide["t"] >= EVENT["fy"]).astype(int)
    wide["share_post_docs"] = (wide["share_post_docs_call"] * wide["n_docs_call"] + wide["share_post_docs_filing"] * wide["n_docs_filing"]) / (wide["n_docs_call"] + wide["n_docs_filing"])
    wide["mixed"] = (wide["share_post_docs"] > 0) & (wide["share_post_docs"] < 1)
    wide["weight"] = np.minimum(wide["n_docs_call"], wide["n_docs_filing"]).astype(float)
    return wide


def cells(frames: pd.DataFrame, period: str) -> pd.DataFrame:
    g = frames.groupby(["ticker", period, "channel"])
    cell = g.agg(
        n_frames=("text_hash", "size"),
        promotional_rate=("is_promo", "mean"),
        quantified_rate=("is_quant", "mean"),
        specificity_index=("specificity", "mean"),
        realized_share=("temporal", lambda s: float((s == "realized").mean())),
        hypothetical_share=("temporal", lambda s: float((s == "hypothetical").mean())),
        gov_share=("is_gov", "mean"),
        share_post_docs=("post_doc", "mean"),
    ).reset_index()
    cell = cell[cell["n_frames"] >= MIN_FRAMES]
    wide = cell.pivot(index=["ticker", period], columns="channel", values=OUTCOMES + ["n_frames", "share_post_docs"])
    wide.columns = [f"{v}_{ch}" for v, ch in wide.columns]
    wide = wide.dropna(subset=["n_frames_call", "n_frames_filing"]).reset_index().rename(columns={period: "t"})
    for y in OUTCOMES:
        wide[f"gap_{y}"] = wide[f"{y}_call"] - wide[f"{y}_filing"]
    wide["post"] = (wide["t"] >= EVENT[period]).astype(int)
    wide["share_post_docs"] = (wide["share_post_docs_call"] * wide["n_frames_call"] + wide["share_post_docs_filing"] * wide["n_frames_filing"]) / (wide["n_frames_call"] + wide["n_frames_filing"])
    wide["mixed"] = (wide["share_post_docs"] > 0) & (wide["share_post_docs"] < 1)
    wide["weight"] = np.minimum(wide["n_frames_call"], wide["n_frames_filing"])
    return wide


def firm_gap(paired: pd.DataFrame, period: str) -> pd.DataFrame:
    """Brecha promedio por empresa (pooled sobre todos sus períodos)."""
    firm = paired.groupby("ticker").agg(
        n_cells=("t", "size"), gap_promotional=("gap_promotional_rate", "mean"),
        gap_specificity=("gap_specificity_index", "mean"), gap_quantified=("gap_quantified_rate", "mean"),
        promotional_call=("promotional_rate_call", "mean"), promotional_filing=("promotional_rate_filing", "mean"),
    ).reset_index()
    min_cells = 2 if period == "fy" else 3
    firm = firm[firm["n_cells"] >= min_cells].sort_values("gap_promotional", ascending=False)
    firm["z_gap_promotional"] = (firm["gap_promotional"] - firm["gap_promotional"].mean()) / firm["gap_promotional"].std(ddof=1)
    return firm


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--period", choices=("fy", "quarter"), default="fy",
                        help="fy = año fiscal alineado por período cubierto (default); quarter = trimestre calendario de la fecha")
    args = parser.parse_args()
    period = args.period

    frames = load_frames()
    print(f"frames: {len(frames):,} | filings {int((frames.channel == 'filing').sum()):,} "
          f"| calls {int((frames.channel == 'call').sum()):,} | empresas {frames.ticker.nunique():,} "
          f"| calls cubren {frames.loc[frames.channel == 'call', 'quarter'].min()}–{frames.loc[frames.channel == 'call', 'quarter'].max()}")

    paired = cells(frames, period)
    print(f"celdas empresa×{period} con ambos canales (≥{MIN_FRAMES} frames cada uno): {len(paired):,} "
          f"| empresas {paired.ticker.nunique():,} | por período {paired.groupby('t').size().to_dict()} "
          f"| celdas mixtas pre/post: {int(paired['mixed'].sum())}")
    paired_out = L.gold_path("covariates", "firm_year" if period == "fy" else "call", "channel_gap_cells_paired")
    paired.assign(t=paired["t"].astype(str)).to_parquet(paired_out, index=False)

    firm = firm_gap(paired, period)
    print(f"empresas con brecha pooled (no persistido, ver channel_gap_did.py): {len(firm):,}")

    ext = None
    if period == "fy":
        ext = cells_extensive(load_documents(frames))
        print(f"celdas empresa×ejercicio con ≥1 transcripción y ≥1 filing, con ceros: {len(ext):,} "
              f"| empresas {ext.ticker.nunique():,} | por ejercicio {ext.groupby('t').size().to_dict()}")
        ext_out = L.gold_path("covariates", "firm_year", "channel_gap_cells_extensive")
        ext.assign(t=ext["t"].astype(str)).to_parquet(ext_out, index=False)

    print(f"\n-> {paired_out}" + (f"\n-> {ext_out}" if ext is not None else ""))


if __name__ == "__main__":
    main()
