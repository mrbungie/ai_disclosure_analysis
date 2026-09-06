"""Perfiles económicos de los segmentos vigentes (RQ3).

Sobre TODAS las empresas-año con filings (`firm_year_master_v2`, ejercicios
2021-2025) y la asignación de segmento por empresa-año
(`firm_year_segments`: desplegadores de producto, adoptantes con gobernanza,
listadores de riesgo, sin IA). Dos paneles:

  A  mediana cruda por segmento: caracterización.
  B  residualizado por sector × año:  X[i,t] = a[sector×t] + γ_k·Segmento[i,k] + e,
     SE cluster por empresa, referencia = sin IA. Responde si los segmentos
     textuales corresponden a tipos económicamente coherentes de empresa
     DENTRO de sectores comparables.

Variables: market cap (log), R&D/ventas, margen bruto, margen operativo,
crecimiento de ingresos t+1, beta, volatilidad realizada pre-filing, P/S,
ROIC − WACC. Winsorización 1/99. Salida: `economic_profiles.json`.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "data" / "processed" / "clusters"
YEARS = (2021, 2022, 2023, 2024, 2025)
VARS = {"log_market_cap": "log market cap", "rd_intensity": "R&D / ventas", "gross_margin": "margen bruto",
        "operating_margin": "margen operativo", "next_revenue_yoy": "crecimiento ingresos t+1", "beta": "beta",
        "vol_pre_60d": "volatilidad pre-filing", "ps_ratio": "P/S", "roic_minus_wacc": "ROIC − WACC"}
ORDER = ["sin_ia", "listadores_de_riesgo", "adoptantes_con_gobernanza", "desplegadores_de_producto"]
REF = "sin_ia"


def winsor(s: pd.Series, lo=0.01, hi=0.99) -> pd.Series:
    return s.clip(s.quantile(lo), s.quantile(hi))


def main() -> None:
    m = pd.read_parquet(OUT_DIR / "firm_year_master_v2.parquet")
    m = m[m["year"].isin(YEARS)].copy()
    rw = pd.read_parquet(OUT_DIR / "firm_year_roic_wacc.parquet")[["ticker", "year", "roic_minus_wacc"]]
    seg = pd.read_parquet(OUT_DIR / "firm_year_segments.parquet")[["ticker", "year", "segmento"]]
    m = m.merge(rw, on=["ticker", "year"], how="left").merge(seg, on=["ticker", "year"], how="left")
    m["segmento"] = m["segmento"].fillna("sin_ia")
    m["log_market_cap"] = np.log(m["market_cap"])
    for v in VARS:
        m[v] = winsor(m[v])
    m["sector_year"] = m["sic2"].astype(str) + "_" + m["year"].astype(str)
    print(f"panel: {len(m):,} empresas-año, {m.ticker.nunique()} empresas | por segmento: {m.segmento.value_counts().to_dict()}\n")

    print("PANEL A — mediana cruda por segmento")
    a = m.groupby("segmento")[list(VARS)].median().reindex(ORDER)
    a["n"] = m.groupby("segmento").size().reindex(ORDER)
    print(a.round(3).T.to_string())

    print("\nPANEL B — residualizado por sector × año: γ de cada segmento contra 'sin IA' (SE cluster por empresa)")
    rows = {}
    for v, label in VARS.items():
        d = m.dropna(subset=[v, "sector_year", "segmento"])
        d = d[d.groupby("sector_year")[v].transform("size") >= 2]
        X = pd.DataFrame({f"seg_{s}": (d["segmento"] == s).astype(float) for s in ORDER if s != REF})
        g = d["sector_year"].to_numpy()
        Xd = X - X.groupby(g).transform("mean"); yd = d[v] - d[v].groupby(g).transform("mean")
        res = sm.OLS(yd.to_numpy(dtype=float), Xd).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d["ticker"])[0]})
        rows[v] = {s: {"gamma": float(res.params[f"seg_{s}"]), "se": float(res.bse[f"seg_{s}"]), "p": float(res.pvalues[f"seg_{s}"])}
                   for s in ORDER if s != REF}
        rows[v]["n"] = int(len(d)); rows[v]["sd"] = float(d[v].std())
        print(f"  {label:26s} n={len(d):5d} | " + " | ".join(
            f"{s.split('_')[0]}: {rows[v][s]['gamma']:+.3f} (p={rows[v][s]['p']:.3f})" for s in ORDER if s != REF))
    payload = {"years": YEARS, "panel_a": json.loads(a.to_json(orient="index")), "panel_b": rows,
               "n_by_segment": {k: int(v) for k, v in m.segmento.value_counts().items()}}
    (OUT_DIR / "economic_profiles.json").write_text(json.dumps(payload, indent=2, default=float) + "\n")
    print(f"\n-> {OUT_DIR / 'economic_profiles.json'}")


if __name__ == "__main__":
    main()
