"""Robustez de denominador para la brecha call-vs-filing (Cap. 5, `channel_gap_analysis.py`).

El comité de tesis observó que "por 1.000 párrafos" trata un turno de call y
un párrafo estatutario como unidades equivalentes, y que parte del 20x/40x
podría ser un artefacto de cómo cada venue se segmenta en párrafos, no solo
de cuánto se dice. Este script recalcula exactamente la misma comparación
extensive-margin (empresa × año fiscal, ambos canales presentes, con ceros)
pero con PALABRAS como denominador en lugar de PÁRRAFOS, reutilizando
`load_frames` de `channel_gap_analysis.py` sin modificarlo.

Determinístico, sin LLM. Requiere `duckdb/thesis.duckdb`.

Salida: `data/processed/clusters/channel_gap_words_robustness.json`
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from channel_gap_analysis import load_frames, FILING_FORMS, DB, _calls_manifest_sql  # noqa: E402

OUT_DIR = REPO_ROOT / "data" / "processed" / "clusters"


def load_documents_words(con: duckdb.DuckDBPyConnection, frames: pd.DataFrame) -> pd.DataFrame:
    """Igual que `load_documents` en channel_gap_analysis.py, pero con
    n_words (conteo de palabras de los párrafos scorable) en vez de
    n_paragraphs."""
    docs = con.execute(f"""
        WITH manifest AS (
            SELECT country_code, accession_number, ticker, filing_date, period_end_date, form_type AS form
            FROM filing_manifest WHERE form_type != 'Earnings call transcript'
            UNION ALL
            SELECT country_code, accession_number, ticker, filing_date, period_end_date, '10-Q' FROM filing_manifest_10q
        ), words AS (
            SELECT country_code, accession_number,
                   sum(list_count(regexp_split_to_array(trim(paragraph_text), '\\s+'))) AS n_words,
                   count(*) AS n_paragraphs
            FROM paragraphs
            WHERE is_scorable GROUP BY 1, 2
        )
        SELECT 'filing' AS channel, m.form, m.ticker, m.filing_date AS fecha, TRY_CAST(m.period_end_date AS DATE) AS period_end,
               NULL::INTEGER AS call_fy, m.accession_number, w.n_words, w.n_paragraphs
        FROM manifest m JOIN words w USING (country_code, accession_number)
        WHERE m.country_code = 'us' AND m.ticker IS NOT NULL AND m.filing_date IS NOT NULL AND m.form IN {FILING_FORMS}
        UNION ALL
        SELECT 'call', 'Earnings call', m.ticker, m.filing_date, NULL::DATE,
               CAST(regexp_extract(m.document_id, '_([0-9]{{4}})Q', 1) AS INTEGER), m.document_id, w.n_words, w.n_paragraphs
        FROM ({_calls_manifest_sql()}) m
        JOIN words w ON w.accession_number = m.document_id AND w.country_code = 'us'
        WHERE m.ticker IS NOT NULL
    """).df()
    fye = frames.groupby("ticker")["fye_month"].first()
    docs["fye_month"] = docs["ticker"].map(fye).fillna(12).astype(int)
    d = pd.to_datetime(docs["fecha"]); pe = pd.to_datetime(docs["period_end"])
    fy_date = d.dt.year + (d.dt.month > docs["fye_month"]).astype(int)
    fy_pe = pe.dt.year + (pe.dt.month > docs["fye_month"]).astype(int)
    is_pe = docs["form"].isin(["10-K", "10-Q"]) & pe.notna()
    docs["fy"] = np.where(docs["channel"] == "call", docs["call_fy"], np.where(is_pe, fy_pe, fy_date)).astype(int)

    per_doc = frames.groupby(["channel", "ticker", "fy"]).agg(
        n_frames=("text_hash", "size"), n_promo=("rhetoric_promotional", "sum"),
        n_quant=("specificity_quantified_metric", "sum")).reset_index()
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
    con = duckdb.connect(str(DB), read_only=True)
    frames = load_frames(con)
    cell = load_documents_words(con, frames)

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

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "channel_gap_words_robustness.json"
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"\n-> {out_path}")


if __name__ == "__main__":
    main()
