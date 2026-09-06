"""Event study empresa-trimestre del escrutinio de la SEC (marzo 2024), con
efectos fijos y errores estándar clusterizados.

Reemplaza el "DiD de tendencia" de `docs/analytics/01_ai_disclosure_analytics.md`,
que no era un DiD: era una regresión segmentada sobre MEDIAS TRIMESTRALES POR
GRUPO. Sus cuatro problemas, y qué hace este script con cada uno:

  1. **Reversión a la media por construcción.** Los grupos se definían por
     valores extremos de las MISMAS métricas que después se medían, así que el
     grupo "específico" tenía que bajar y el "vago" subir aunque no pasara
     nada. Acá los grupos se siguen formando con 10-K pre-2024 (el instrumento
     es distinto del que se mide, que es el 10-Q), pero además se reporta el
     placebo: si los coeficientes pre-evento no son cero, es reversión, no
     efecto.

  2. **Composición cambiante.** La media de cada grupo se calculaba sobre las
     empresas que divulgaron ESE trimestre, y entre 2023 y 2025 entraron cientos
     de empresas nuevas al corpus. Con efectos fijos de empresa, la comparación
     es dentro de la misma empresa a lo largo del tiempo: quién entra o sale ya
     no mueve el coeficiente.

  3. **Sin efectos fijos ni SE clusterizados.** Acá hay efectos fijos de empresa
     y de trimestre, y la matriz de covarianza es cluster-robusta por empresa —
     los trimestres de una misma empresa están correlacionados y tratarlos como
     independientes infla los t.

  4. **Sin tendencias paralelas.** El event study estima un coeficiente por
     trimestre relativo al evento; los previos SON el test.

También corrige algo más sutil: el diseño anterior comparaba dos grupos
formados por una mediana, lo que descarta la mitad de la información. Acá el
tratamiento es continuo (el índice de vaguedad pre-2024) además de binario.

Uso:
    uv run python scripts/analytics/sec_event_study.py
    uv run python scripts/analytics/sec_event_study.py --outcome specificity_index
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

REPO_ROOT = Path(__file__).resolve().parents[2]
DB = REPO_ROOT / "duckdb" / "thesis.duckdb"
OUT_DIR = REPO_ROOT / "data" / "processed" / "clusters"
# 18 de marzo de 2024: la SEC anuncia las primeras acciones por "AI washing".
# El trimestre 2024Q1 mezcla filings de antes y después, así que el evento se
# fija al comienzo de 2024Q2 y 2024Q1 queda como el trimestre de referencia.
EVENT_QUARTER = pd.Period("2024Q2", freq="Q")
REFERENCE_OFFSET = -1
MIN_FRAMES_QUARTER = 3
MIN_FRAMES_PRE = 5
SPECIFICITY_FLAGS = ["specificity_business_process", "specificity_product_or_system",
                     "specificity_vendor_or_partner", "specificity_quantified_metric",
                     "specificity_date_or_timeline"]


def load_10q_frames(con) -> pd.DataFrame:
    return con.execute("""
        SELECT fm.ticker, fm.filing_date, f.text_hash, f.frame_index, f.temporal,
               f.concepts, f.rhetoric_promotional,
               f.specificity_business_process, f.specificity_product_or_system,
               f.specificity_vendor_or_partner, f.specificity_quantified_metric,
               f.specificity_date_or_timeline
        FROM gold_ai_frames f
        JOIN filing_manifest_10q fm USING (country_code, accession_number)
        WHERE f.country_code = 'us' AND f.has_frame AND fm.ticker IS NOT NULL
        ORDER BY fm.ticker, fm.filing_date, f.text_hash, f.frame_index
    """).fetchdf().drop_duplicates(["ticker", "filing_date", "text_hash", "frame_index"])


def load_10k_pre2024(con) -> pd.DataFrame:
    return con.execute("""
        SELECT fm.ticker, f.text_hash, f.frame_index, f.rhetoric_promotional,
               f.specificity_business_process, f.specificity_product_or_system,
               f.specificity_vendor_or_partner, f.specificity_quantified_metric,
               f.specificity_date_or_timeline
        FROM gold_ai_frames f
        JOIN filing_manifest fm USING (country_code, accession_number)
        WHERE f.country_code = 'us' AND f.has_frame AND fm.form_type = '10-K'
          AND fm.filing_date < DATE '2024-01-01' AND fm.ticker IS NOT NULL
        ORDER BY fm.ticker, f.text_hash, f.frame_index
    """).fetchdf().drop_duplicates(["ticker", "text_hash", "frame_index"])


def metrics(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["specificity_index"] = out[SPECIFICITY_FLAGS].astype(float).mean(axis=1)
    out["promotional_rate"] = out["rhetoric_promotional"].astype(float)
    if "temporal" in out.columns:
        out["hypothetical_share"] = (out["temporal"] == "hypothetical").astype(float)
        out["risk_share"] = out["concepts"].apply(
            lambda cs: any(str(c).startswith("risk_") for c in (cs if cs is not None else []))
        ).astype(float)
    return out


def build_panel(con) -> tuple[pd.DataFrame, pd.Series]:
    quarterly = metrics(load_10q_frames(con))
    quarterly["quarter"] = pd.PeriodIndex(pd.to_datetime(quarterly["filing_date"]), freq="Q")
    outcomes = ["promotional_rate", "specificity_index", "hypothetical_share", "risk_share"]
    panel = (quarterly.groupby(["ticker", "quarter"])
             .agg(n_frames=("promotional_rate", "size"),
                  **{c: (c, "mean") for c in outcomes}).reset_index())
    panel = panel[panel["n_frames"] >= MIN_FRAMES_QUARTER]

    pre = metrics(load_10k_pre2024(con))
    pre_firm = pre.groupby("ticker").agg(n_pre=("promotional_rate", "size"),
                                         promo=("promotional_rate", "mean"),
                                         spec=("specificity_index", "mean"))
    pre_firm = pre_firm[pre_firm["n_pre"] >= MIN_FRAMES_PRE]
    # Vaguedad = promoción alta menos especificidad alta, ambas estandarizadas.
    # Se mide con 10-K PRE-2024 y el outcome con 10-Q: instrumentos distintos,
    # para que el mismo texto no defina el grupo y el resultado.
    vagueness = ((pre_firm["promo"] - pre_firm["promo"].mean()) / pre_firm["promo"].std()
                 - (pre_firm["spec"] - pre_firm["spec"].mean()) / pre_firm["spec"].std())
    return panel, vagueness.rename("vagueness")


def event_study(panel: pd.DataFrame, outcome: str, treatment: pd.Series,
                binary: bool) -> pd.DataFrame:
    data = panel.merge(treatment.reset_index(), on="ticker", how="inner").copy()
    if binary:
        data["treat"] = (data["vagueness"] > data["vagueness"].median()).astype(float)
    else:
        data["treat"] = ((data["vagueness"] - data["vagueness"].mean())
                         / data["vagueness"].std())
    data["event_time"] = (data["quarter"].astype("period[Q]")
                          - EVENT_QUARTER).apply(lambda x: x.n)
    data = data[data["event_time"].between(-6, 6)]
    # El trimestre anterior al evento es la referencia omitida: todos los
    # coeficientes se leen contra él, y los NEGATIVOS son el test de tendencias
    # paralelas — si el efecto ya estaba antes del evento, no es del evento.
    data["ev"] = pd.Categorical(data["event_time"],
                                categories=sorted(data["event_time"].unique()))
    formula = (f"{outcome} ~ C(ev, Treatment(reference={REFERENCE_OFFSET})):treat "
               f"+ C(ticker) + C(ev)")
    model = smf.ols(formula, data=data).fit(
        cov_type="cluster", cov_kwds={"groups": data["ticker"]})
    import re
    rows = []
    for name, value in model.params.items():
        if "treat" not in name:
            continue
        match = re.search(r"\[T?\.?(-?\d+)\]", name)
        if not match:
            continue
        offset = int(match.group(1))
        rows.append({"event_time": offset, "coef": float(value),
                     "se": float(model.bse[name]), "t": float(model.tvalues[name]),
                     "p": float(model.pvalues[name])})
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result = result[result["event_time"] != REFERENCE_OFFSET].sort_values("event_time")
    result.attrs["n_obs"] = int(model.nobs)
    result.attrs["n_firms"] = int(data["ticker"].nunique())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--database", type=Path, default=DB)
    parser.add_argument("--output", type=Path, default=OUT_DIR / "sec_event_study.json")
    args = parser.parse_args()

    con = duckdb.connect(str(args.database), read_only=True)
    try:
        panel, vagueness = build_panel(con)
    finally:
        con.close()
    print(f"panel: {len(panel):,} empresa-trimestre, {panel['ticker'].nunique():,} empresas "
          f"({panel['quarter'].min()} a {panel['quarter'].max()})")
    print(f"grupos: {len(vagueness):,} empresas clasificables con >= {MIN_FRAMES_PRE} "
          f"frames de 10-K pre-2024\n")

    report = {}
    for outcome in ("promotional_rate", "specificity_index", "risk_share",
                    "hypothetical_share"):
        for binary in (True, False):
            label = f"{outcome} ({'binario' if binary else 'continuo'})"
            result = event_study(panel, outcome, vagueness, binary)
            if result.empty:
                continue
            pre = result[result.event_time < REFERENCE_OFFSET]
            post = result[result.event_time >= 0]
            print(f"--- {label} | n={result.attrs['n_obs']:,} obs, "
                  f"{result.attrs['n_firms']} empresas ---")
            print("  " + "  ".join(f"t{r.event_time:+d}={r.coef:+.3f}"
                                   f"{'*' if r.p < 0.05 else ''}"
                                   for r in result.itertuples()))
            print(f"  pre-evento (test de tendencias paralelas): "
                  f"{'FALLA — hay efecto antes del evento' if (pre.p < 0.05).any() else 'pasa'}"
                  f" | post significativos: {int((post.p < 0.05).sum())} de {len(post)}")
            report[label] = {"coefficients": result.to_dict("records"),
                             "n_obs": result.attrs["n_obs"],
                             "n_firms": result.attrs["n_firms"],
                             "parallel_trends_violated": bool((pre.p < 0.05).any())}
    args.output.write_text(json.dumps(report, indent=2, default=float))
    print(f"\n-> {args.output}")


if __name__ == "__main__":
    main()
