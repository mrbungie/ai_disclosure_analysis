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
import re

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
# El grupo se define con 10-K de 2021-2022 SOLAMENTE, dejando 2023 como
# pre-período limpio. Si se define con todo el pre-2024, las empresas se
# clasifican con los mismos trimestres contra los que después se testea el
# pre-trend, y la reversión a la media aparece como violación de tendencias
# paralelas aunque no haya pasado nada.
GROUP_WINDOW_END = pd.Timestamp("2023-01-01")
MIN_QUARTERS_EACH_SIDE = 2
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


def load_all_forms(con) -> pd.DataFrame:
    """Frames de TODOS los formularios con su trimestre de presentación.

    La versión anterior usaba sólo 10-Q y se quedaba con 257 observaciones y 54
    empresas — sin potencia para detectar nada. Sumar 10-K, DEF 14A y 8-K
    multiplica el panel, pero mete un confusor: la mezcla de formularios de una
    empresa cambia trimestre a trimestre y los formularios difieren en retórica
    (la DEF 14A tiene 14,6% de frames promocionales contra 7,2% del 10-K). Por
    eso la regresión controla por la composición documental del trimestre — un
    control de la regresión, no una feature del instrumento de medición.

    Los frames se traen UNA vez y los dos manifiestos se unen en pandas. Escribir
    esto como un `UNION ALL` de dos consultas a `gold_ai_frames` parece
    equivalente y no lo es: obliga a DuckDB a instanciar la vista dos veces, y la
    vista recalcula por dentro la población vigente con una ventana sobre 37
    millones de filas de predicciones. Medido: así no termina en 115 segundos;
    de esta forma tarda ~2."""
    frames = con.execute("""
        SELECT accession_number, text_hash, frame_index, temporal, concepts,
               rhetoric_promotional, specificity_business_process,
               specificity_product_or_system, specificity_vendor_or_partner,
               specificity_quantified_metric, specificity_date_or_timeline
        FROM gold_ai_frames
        WHERE country_code = 'us' AND has_frame
    """).fetchdf()
    manifests = []
    for relation, form in (("filing_manifest", None), ("filing_manifest_10q", "10-Q")):
        table = con.execute(f"""
            SELECT accession_number, ticker, filing_date
                   {', form_type' if form is None else ''}
            FROM {relation} WHERE country_code = 'us' AND ticker IS NOT NULL
        """).fetchdf()
        if form is not None:
            table["form_type"] = form
        manifests.append(table)
    manifest = pd.concat(manifests, ignore_index=True)
    merged = frames.merge(manifest, on="accession_number", how="inner")
    return merged.drop_duplicates(["ticker", "filing_date", "form_type",
                                   "text_hash", "frame_index"])


def load_10k_pre2024(con) -> pd.DataFrame:
    return con.execute("""
        SELECT fm.ticker, f.text_hash, f.frame_index, f.rhetoric_promotional,
               f.specificity_business_process, f.specificity_product_or_system,
               f.specificity_vendor_or_partner, f.specificity_quantified_metric,
               f.specificity_date_or_timeline
        FROM gold_ai_frames f
        JOIN filing_manifest fm USING (country_code, accession_number)
        WHERE f.country_code = 'us' AND f.has_frame AND fm.form_type = '10-K'
          AND CAST(fm.filing_date AS DATE) < DATE '2023-01-01' AND fm.ticker IS NOT NULL
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
    quarterly = metrics(load_all_forms(con))
    quarterly["quarter"] = pd.PeriodIndex(pd.to_datetime(quarterly["filing_date"]), freq="Q")
    quarterly["es_proxy"] = (quarterly["form_type"] == "DEF 14A").astype(float)
    quarterly["es_10k"] = (quarterly["form_type"] == "10-K").astype(float)
    outcomes = ["promotional_rate", "specificity_index", "hypothetical_share", "risk_share"]
    panel = (quarterly.groupby(["ticker", "quarter"])
             .agg(n_frames=("promotional_rate", "size"),
                  mix_proxy=("es_proxy", "mean"), mix_10k=("es_10k", "mean"),
                  **{c: (c, "mean") for c in outcomes}).reset_index())
    panel = panel[panel["n_frames"] >= MIN_FRAMES_QUARTER]

    pre = metrics(load_10k_pre2024(con))  # sólo 2021-2022
    pre_firm = pre.groupby("ticker").agg(n_pre=("promotional_rate", "size"),
                                         promo=("promotional_rate", "mean"),
                                         spec=("specificity_index", "mean"))
    pre_firm = pre_firm[pre_firm["n_pre"] >= MIN_FRAMES_PRE]
    # Vaguedad = promoción alta menos especificidad alta, ambas estandarizadas.
    # Se mide con 10-K PRE-2024 y el outcome con 10-Q: instrumentos distintos,
    # para que el mismo texto no defina el grupo y el resultado.
    vagueness = ((pre_firm["promo"] - pre_firm["promo"].mean()) / pre_firm["promo"].std()
                 - (pre_firm["spec"] - pre_firm["spec"].mean()) / pre_firm["spec"].std())
    # Tratamiento alternativo: EXPOSICIÓN, no vaguedad. Cuánto hablaba la empresa
    # de IA antes de 2023, medido en volumen de frames, que NO es el outcome.
    #
    # Es la diferencia entre un diseño identificable y uno que no lo es. Definir
    # el grupo por el NIVEL pre-evento de `promotional_rate` y después medir
    # `promotional_rate` garantiza tendencias no paralelas: el grupo "vago" está
    # arriba por construcción y sólo puede bajar. La exposición evita eso —
    # separa a quién le apuntaba el escrutinio (las que ya hablaban de IA) sin
    # usar cómo hablaban.
    exposure = np.log(pre_firm["n_pre"])
    return panel, pd.DataFrame({"vagueness": vagueness, "exposure": exposure})


def event_study(panel: pd.DataFrame, outcome: str, treatment: pd.DataFrame,
                binary: bool, group_trend: bool = False,
                column: str = "vagueness") -> pd.DataFrame:
    data = panel.merge(treatment.reset_index(), on="ticker", how="inner").copy()
    if binary:
        data["treat"] = (data[column] > data[column].median()).astype(float)
    else:
        data["treat"] = (data[column] - data[column].mean()) / data[column].std()
    data["event_time"] = (data["quarter"].astype("period[Q]")
                          - EVENT_QUARTER).apply(lambda x: x.n)
    data = data[data["event_time"].between(-6, 6)]
    # El trimestre anterior al evento es la referencia omitida: todos los
    # coeficientes se leen contra él, y los NEGATIVOS son el test de tendencias
    # paralelas — si el efecto ya estaba antes del evento, no es del evento.
    # Panel utilizable: empresas observadas a los dos lados del corte. Sin esto,
    # una empresa que sólo aparece después del evento aporta a los coeficientes
    # post sin haber aportado nunca a los pre, y el "efecto" es composición.
    sides = data.groupby("ticker")["event_time"].agg(
        pre=lambda s: (s < 0).sum(), post=lambda s: (s >= 0).sum())
    balanced = sides[(sides["pre"] >= MIN_QUARTERS_EACH_SIDE)
                     & (sides["post"] >= MIN_QUARTERS_EACH_SIDE)].index
    data = data[data["ticker"].isin(balanced)]
    if data.empty:
        return pd.DataFrame()
    data["ev"] = pd.Categorical(data["event_time"],
                                categories=sorted(data["event_time"].unique()))
    controls = "+ mix_proxy + mix_10k + np.log(n_frames)"
    # Tendencias lineales propias de cada grupo: es el remedio estándar cuando
    # las tendencias paralelas fallan. Absorbe que los grupos vinieran
    # moviéndose a ritmos distintos ANTES del evento y deja como "efecto" sólo
    # el quiebre respecto de esa trayectoria.
    trend = "+ treat:event_time" if group_trend else ""
    formula = (f"{outcome} ~ C(ev, Treatment(reference={REFERENCE_OFFSET})):treat "
               f"+ C(ticker) + C(ev) {controls} {trend}")
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
    # Test formal de tendencias paralelas: los coeficientes PRE conjuntamente
    # iguales a cero. Mirar si alguno tiene p<0,05 uno por uno infla el falso
    # positivo con 5 coeficientes; el test conjunto es el que corresponde.
    pre_terms = [name for name in model.params.index
                 if "treat" in name and re.search(r"\[T?\.?(-?\d+)\]", name)
                 and int(re.search(r"\[T?\.?(-?\d+)\]", name).group(1)) < REFERENCE_OFFSET]
    if pre_terms:
        joint = model.f_test(" = 0, ".join(pre_terms) + " = 0")
        result.attrs["pretrend_F"] = float(np.squeeze(joint.fvalue))
        result.attrs["pretrend_p"] = float(np.squeeze(joint.pvalue))
    post_terms = [name for name in model.params.index
                  if "treat" in name and re.search(r"\[T?\.?(-?\d+)\]", name)
                  and int(re.search(r"\[T?\.?(-?\d+)\]", name).group(1)) >= 0]
    if post_terms:
        average = model.f_test(" + ".join(post_terms) + f" = 0")
        result.attrs["post_F"] = float(np.squeeze(average.fvalue))
        result.attrs["post_p"] = float(np.squeeze(average.pvalue))
        result.attrs["post_mean"] = float(np.mean([model.params[name] for name in post_terms]))
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
        for column in ("vagueness", "exposure"):
            label = f"{outcome} [tratamiento: {column}]"
            result = event_study(panel, outcome, vagueness, binary=True,
                                 group_trend=False, column=column)
            if result.empty:
                continue
            print(f"--- {label} | n={result.attrs['n_obs']:,} obs, "
                  f"{result.attrs['n_firms']} empresas ---")
            print("  " + "  ".join(f"t{r.event_time:+d}={r.coef:+.3f}"
                                   f"{'*' if r.p < 0.05 else ''}"
                                   for r in result.itertuples()))
            pretrend_p = result.attrs.get("pretrend_p", float("nan"))
            verdict = ("pasa" if pretrend_p > 0.10 else
                       "límite" if pretrend_p > 0.05 else "FALLA")
            print(f"  tendencias paralelas (test conjunto de los pre): "
                  f"F={result.attrs.get('pretrend_F', float('nan')):.2f}, "
                  f"p={pretrend_p:.3f} -> {verdict}")
            print(f"  efecto post promedio: {result.attrs.get('post_mean', float('nan')):+.4f} "
                  f"(p conjunto = {result.attrs.get('post_p', float('nan')):.3f})")
            report[label] = {"coefficients": result.to_dict("records"),
                             **{k: v for k, v in result.attrs.items()}}
    args.output.write_text(json.dumps(report, indent=2, default=float))
    print(f"\n-> {args.output}")


if __name__ == "__main__":
    main()
