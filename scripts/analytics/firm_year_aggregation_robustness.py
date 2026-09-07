"""Robustez de la agregación documento -> empresa-año usada en Chapter 4 (RQ4).

`incremental_signal.py` toma sus siete columnas de contenido semántico
(`*_per_1k`) de `firm_year_master_v2.parquet`, que las agrega por MEDIANA
sobre los documentos del ejercicio (ver Aggregation, Cap. 2). El comité de
tesis pidió comparar contra la alternativa obvia: agregado
numerador-sobre-denominador (equivalente algebraicamente a una media
ponderada por párrafos, ver nota abajo), construida aquí con
`ai_intensity.aggregate()` sobre TODOS los filing forms, sin tocar
`incremental_signal.py` ni sus salidas.

Nota algebraica: para una tasa por 1.000 párrafos, la media ponderada por
párrafos de las tasas documento-a-documento
    sum(rate_i * n_parrafos_i) / sum(n_parrafos_i)
es idénticamente igual al agregado numerador/denominador
    sum(n_X_i) * 1000 / sum(n_parrafos_i)
así que solo hay DOS reglas de agregación económicamente distintas a
comparar contra la mediana: esta (numerador/denominador = ponderada por
párrafos) y la mediana ya reportada. No hay una tercera regla que probar.

Determinístico, sin LLM. Requiere `duckdb/thesis.duckdb` y los parquets ya
construidos en `data/processed/clusters/`.

Salida: `data/processed/clusters/firm_year_aggregation_robustness.json`
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
from ai_intensity import document_table, aggregate, FILING_FORMS  # noqa: E402
import incremental_signal as isig  # noqa: E402

OUT_DIR = REPO_ROOT / "data" / "processed" / "clusters"
DB = REPO_ROOT / "duckdb" / "thesis.duckdb"


def build_altagg_panel(con) -> pd.DataFrame:
    """Empresa-año, todos los filing forms, tasas por numerador/denominador
    agregado (= media ponderada por párrafos) en vez de mediana de documentos.
    `year` es el año calendario de la fecha de presentación (aproximación a
    la alineación por ejercicio fiscal usada en `firm_year_master_v2`)."""
    docs = document_table(con)
    docs = docs[docs["form"].isin(FILING_FORMS)].assign(year=lambda d: d["fecha"].dt.year)
    alt = aggregate(docs, ["ticker", "year"])
    alt["capability_per_1k"] = alt["ai_investment_per_1k"] + alt["ai_infrastructure_per_1k"]
    return alt


def main() -> None:
    con = duckdb.connect(str(DB), read_only=True)
    try:
        m = isig.load(con)          # panel principal (mediana), sin tocar
        alt = build_altagg_panel(con)
    finally:
        con.close()

    content_cols = ["frames_per_1k"] + list(isig.SEMANTIC.values())
    alt_renamed = alt[["ticker", "year"] + content_cols].rename(columns={c: f"{c}__altagg" for c in content_cols})
    n_before = len(m)
    m2 = m.merge(alt_renamed, on=["ticker", "year"], how="left")
    n_matched = m2[[f"{c}__altagg" for c in content_cols]].notna().all(axis=1).sum()
    print(f"Firm-years in main panel: {n_before:,} | matched to alt-aggregation panel: {n_matched:,} "
          f"({n_matched / n_before:.1%}) — gap is mostly non-Dec fiscal-year-end firms, "
          f"since alt panel uses calendar filing year as an approximation to fiscal year.")

    print("\n" + "=" * 100)
    print("MAIN (median-of-documents, as reported in the thesis) vs. ALT-AGG (aggregate numerator/denominator)")
    print("=" * 100)
    hdr = f"{'outcome':26s} {'n (main)':>9s} {'ΔR² sem (main)':>15s} {'n (altagg)':>11s} {'ΔR² sem (altagg)':>17s} {'partial R² (altagg)':>20s} {'Wald p (altagg)':>16s}"
    print(hdr)
    report = {"n_main": int(n_before), "n_matched_altagg": int(n_matched), "outcomes": {}}
    for label, y in isig.OUTCOMES.items():
        res_main, _ = isig.nested(m, y)
        res_alt, d_alt = isig.nested(m2, y, "sector_year", "__altagg")
        wald_alt = isig.wald_semantic(d_alt, y, "sector_year", "__altagg")
        print(f"{label:26s} {res_main['n']:9d} {res_main['delta_r2_semantica']:+15.4f} "
              f"{res_alt['n']:11d} {res_alt['delta_r2_semantica']:+17.4f} "
              f"{res_alt['partial_r2_semantica']:20.4f} {wald_alt['p']:16.3f}")
        report["outcomes"][label] = {
            "main_n": res_main["n"], "main_delta_r2": res_main["delta_r2_semantica"],
            "altagg_n": res_alt["n"], "altagg_delta_r2": res_alt["delta_r2_semantica"],
            "altagg_partial_r2": res_alt["partial_r2_semantica"], "altagg_wald_p": wald_alt["p"],
            "altagg_r2_M0": res_alt["r2_M0"], "altagg_r2_M1": res_alt["r2_M1"], "altagg_r2_M2": res_alt["r2_M2"],
        }

    out_path = OUT_DIR / "firm_year_aggregation_robustness.json"
    out_path.write_text(json.dumps(report, indent=2, default=float) + "\n")
    print(f"\n-> {out_path}")


if __name__ == "__main__":
    main()
