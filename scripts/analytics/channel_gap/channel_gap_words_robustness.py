"""Robustez de denominador para la brecha call-vs-filing.

El comité de tesis observó que "por 1.000 párrafos" trata un turno de call y
un párrafo estatutario como unidades equivalentes, y que parte del 20x/40x
podría ser un artefacto de cómo cada venue se segmenta en párrafos, no solo
de cuánto se dice. Este script calcula la comparación extensive-margin
(empresa × año fiscal con >=1 transcripción y >=1 filing, con ceros) con
PALABRAS y con PÁRRAFOS como denominador.

PERÍODO = AÑO FISCAL, alineado por lo que el documento CUBRE:
  10-K, 10-Q       año fiscal de `period_end_date`
  earnings call    año fiscal del `document_id` (`TICKER_YYYYQn`)
  8-K, DEF 14A     año fiscal en que se presentan
con el mes de cierre de cada empresa (moda del mes de `period_end_date` de
sus 10-K con frames; diciembre si no hay).

`reconcile_2026`: en el ejercicio 2026, número de celdas con ambos canales
(`n_dual`) y cuántas tienen una brecha promocional por 1.000 palabras
(call − filing) con z > 1.5 (`n_gap_z_above_1_5`).

Determinístico, sin LLM. Lee `silver.ai_frames`, `silver.filing_manifest`,
`silver.filing_manifest_10q` y `bronze.paragraphs`.

Salida: `data/results/channel_gap/channel_gap_words_robustness.json`
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import layers as L  # noqa: E402

FILING_FORMS = ("10-K", "10-Q", "DEF 14A", "8-K")
CALL_FORM_TYPE = "Earnings call transcript"
RECONCILE_FY = 2026
GAP_Z = 1.5


def _calls_manifest() -> pl.LazyFrame:
    """Una fila por transcripción (`document_id` = `TICKER_YYYYQn`), ya
    restringida al universo de análisis en `silver.filing_manifest`."""
    return (L.scan("silver.filing_manifest")
            .filter((pl.col("form_type") == CALL_FORM_TYPE) & pl.col("ticker").is_not_null())
            .select("document_id", "ticker", "filing_date", "fiscal_period"))


def _filings_manifest(form_col: bool = False) -> pl.LazyFrame:
    """Filings de EE.UU. (10-K, DEF 14A, 8-K, ... y 10-Q), sin transcripciones."""
    cols = ["country_code", "accession_number", "ticker", "filing_date", "period_end_date"]
    main = L.scan("silver.filing_manifest").filter(pl.col("form_type") != CALL_FORM_TYPE)
    tenq = L.scan("silver.filing_manifest_10q")
    if form_col:
        return pl.concat([main.select(cols + [pl.col("form_type").alias("form")]),
                          tenq.select(cols + [pl.lit("10-Q").alias("form")])])
    return pl.concat([main.select(cols), tenq.select(cols)])


def _call_fy(col: str) -> pl.Expr:
    """Año fiscal de la call, desde su `fiscal_period` ('2024Q1') de silver.filing_manifest."""
    return pl.col(col).str.slice(0, 4).cast(pl.Int32).alias("call_fy")


def load_frames() -> pd.DataFrame:
    """Frames de EE.UU. con empresa, canal, año fiscal y flags promocional /
    cuantificado. Los filings resuelven ticker/fecha por
    `silver.filing_manifest` (10-K, DEF 14A, 8-K) y `silver.filing_manifest_10q`;
    las calls por su manifiesto propio (`document_id` = `accession_number`
    sintético `TICKER_YYYYQn`, año fiscal desde `fiscal_period`)."""
    ai_frames = L.scan("silver.ai_frames").filter((pl.col("country_code") == "us") & pl.col("has_frame"))
    frame_cols = ["text_hash", "frame_id", "rhetoric", "specificity"]
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
                     pl.lit(None, dtype=pl.Date).alias("period_end"), _call_fy("fiscal_period"), *frame_cols)
             .collect().to_pandas())
    frames = pd.concat([filings, calls], ignore_index=True)
    frames = frames.drop_duplicates(["channel", "ticker", "fecha", "text_hash", "frame_id"])
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
    frames["is_promo"] = frames["rhetoric"].apply(
        lambda r: "promotional" in list(r) if r is not None else False)
    frames["is_quant"] = frames["specificity"].apply(
        lambda s: "metric" in list(s) if s is not None else False)
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
                     _call_fy("fiscal_period"), pl.col("document_id").alias("accession_number"),
                     "n_paragraphs", "n_words"))
    return pl.concat([filings, calls]).collect(engine="streaming").to_pandas()


def load_documents_words(frames: pd.DataFrame) -> pd.DataFrame:
    """Una fila por empresa × año fiscal × canal con todos sus documentos,
    tengan o no frames de IA (una call sin IA es un cero, no una celda
    perdida), y las tasas por 1.000 palabras y por 1.000 párrafos."""
    docs = document_counts()
    fye = frames.groupby("ticker")["fye_month"].first()
    docs["fye_month"] = docs["ticker"].map(fye).fillna(12).astype(int)
    d = pd.to_datetime(docs["fecha"]); pe = pd.to_datetime(docs["period_end"])
    fy_date = d.dt.year + (d.dt.month > docs["fye_month"]).astype(int)
    fy_pe = pe.dt.year + (pe.dt.month > docs["fye_month"]).astype(int)
    is_pe = docs["form"].isin(["10-K", "10-Q"]) & pe.notna()
    docs["fy"] = np.where(docs["channel"] == "call", docs["call_fy"], np.where(is_pe, fy_pe, fy_date)).astype(int)

    per_doc = frames.groupby(["channel", "ticker", "fy"]).agg(
        n_frames=("text_hash", "size"), n_promo=("is_promo", "sum"),
        n_quant=("is_quant", "sum")).reset_index()
    cell = docs.groupby(["ticker", "fy", "channel"]).agg(
        n_docs=("accession_number", "nunique"), n_words=("n_words", "sum"),
        n_paragraphs=("n_paragraphs", "sum")).reset_index()
    cell = cell.merge(per_doc, on=["channel", "ticker", "fy"], how="left").fillna({"n_frames": 0, "n_promo": 0, "n_quant": 0})
    for k in ("promo", "quant", "frames"):
        cell[f"{k}_per_1k_words"] = 1000.0 * cell[f"n_{k}"] / cell["n_words"]
        cell[f"{k}_per_1k_paragraphs"] = 1000.0 * cell[f"n_{k}"] / cell["n_paragraphs"]
    cell["any_ai"] = (cell["n_frames"] > 0).astype(float)
    return cell


def main() -> None:
    frames = load_frames()
    cell = load_documents_words(frames)

    outcomes = ["promo_per_1k_words", "quant_per_1k_words", "promo_per_1k_paragraphs", "quant_per_1k_paragraphs"]
    wide = cell.pivot(index=["ticker", "fy"], columns="channel", values=outcomes + ["n_docs", "n_words", "n_paragraphs", "n_frames"])
    wide.columns = [f"{v}_{ch}" for v, ch in wide.columns]
    wide = wide.dropna(subset=["n_docs_call", "n_docs_filing"]).reset_index()
    n_cells = len(wide)
    n_firms = wide["ticker"].nunique()

    result = {"n_cells": int(n_cells), "n_firms": int(n_firms)}
    print(f"Extensive-margin cells (both channels present): {n_cells:,} across {n_firms:,} firms\n")
    for base in ("promo", "quant"):
        for denom in ("words", "paragraphs"):
            col = f"{base}_per_1k_{denom}"
            call_mean = wide[f"{col}_call"].mean()
            filing_mean = wide[f"{col}_filing"].mean()
            ratio = call_mean / filing_mean if filing_mean > 0 else float("inf")
            result[col] = {"call_mean": float(call_mean), "filing_mean": float(filing_mean),
                           "ratio": float(ratio), "gap": float(call_mean - filing_mean)}
            label = "promotional" if base == "promo" else "quantified"
            print(f"{label:12s} per 1,000 {denom:10s}: call {call_mean:6.2f} | filing {filing_mean:5.2f} | ratio {ratio:5.1f}x")
        print()

    # word/paragraph ratio by channel — is a "paragraph" simply a different-sized unit in each venue?
    wp_call = (wide["n_words_call"] / wide["n_paragraphs_call"]).mean()
    wp_filing = (wide["n_words_filing"] / wide["n_paragraphs_filing"]).mean()
    result["mean_words_per_paragraph"] = {"call": float(wp_call), "filing": float(wp_filing)}
    print(f"Mean words per scorable paragraph: call {wp_call:.1f} | filing {wp_filing:.1f} "
          f"(ratio {wp_filing / wp_call:.2f}x)")

    # ejercicio 2026: celdas con ambos canales y brechas promocionales (call - filing,
    # por 1.000 palabras) a más de GAP_Z desviaciones estándar de la media del año
    g26 = wide[wide["fy"] == RECONCILE_FY]
    gap = g26["promo_per_1k_words_call"] - g26["promo_per_1k_words_filing"]
    z_gap = (gap - gap.mean()) / gap.std()
    result["reconcile_2026"] = {"n_dual": int(len(g26)), "n_gap_z_above_1_5": int((z_gap > GAP_Z).sum())}
    print(f"FY{RECONCILE_FY}: {len(g26):,} celdas con ambos canales | brecha promocional z > {GAP_Z}: "
          f"{result['reconcile_2026']['n_gap_z_above_1_5']}")

    out_path = L.results_path("channel_gap", "channel_gap_words_robustness.json")
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"\n-> {out_path}")


if __name__ == "__main__":
    main()
