"""DiD de dos grupos alrededor del escrutinio de la SEC, con figura.

    Y[i,t] = a[i] + lambda[t] + beta * (AltoRiesgo[i] x Post[t]) + e[i,t]

`AltoRiesgo` = empresas cuya divulgación de IA pre-2024 era más promocional que
la mediana. `Post` = a partir de 2024Q2 (el anuncio es el 18-mar-2024 y 2024Q1
mezcla filings de los dos lados). Efectos fijos de empresa y de trimestre,
errores estándar clusterizados por empresa.

**La decisión que hace que esto funcione**: el grupo se define con una dimensión
(retórica promocional) y el outcome se mide en OTRA — cuantificación, gobernanza,
especificidad. Definir el grupo por el nivel pre-evento de la misma variable que
después se mide produce convergencia mecánica: el grupo alto sólo puede bajar,
haya pasado algo o no. Con dimensiones distintas ese artefacto desaparece, y por
eso `promotional_rate` se reporta igual pero **como referencia negativa**: es
donde el artefacto debería verse, y sirve de control interno.

Lo que la afirmación puede sostener: *las empresas que ex ante se veían más
expuestas a AI-washing cambiaron su divulgación de forma distinta a las demás
después del escrutinio*. No sostiene que la SEC lo haya causado — el evento es
común en tiempo calendario y el boom de IA generativa corre en paralelo. La
propuesta de tesis pide lo primero ("quasi-causal techniques like DiD *may* be
used" como extensión), no lo segundo.

Salidas: tabla de coeficientes, event study por trimestre y una figura
`data/processed/clusters/shock_did_event_study.png` con los dos grupos
normalizados a cero antes del evento.

Uso:
    uv run python scripts/analytics/shock_did_simple.py
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

REPO_ROOT = Path(__file__).resolve().parents[2]
DB = REPO_ROOT / "duckdb" / "thesis.duckdb"
OUT_DIR = REPO_ROOT / "data" / "processed" / "clusters"
EVENT = pd.Period("2024Q1", freq="Q")   # aviso de Gensler, 5-dic-2023; ver shock_analysis.py
GROUP_END = "2023-12-01"
WINDOW = 5
REFERENCE = -1
MIN_QUARTERS_EACH_SIDE = 2
# El primero comparte dimensión con la definición del grupo: va como control
# interno, no como resultado.
OUTCOMES = ("promo_per_1k", "quant_per_1k", "gov_per_1k", "spec_per_1k")
GROUP_OUTCOME = "promo_per_1k"     # la dimensión que define el grupo: control interno
CONTROLS = "mix_proxy + np.log(n_words)"


def build(con) -> pd.DataFrame:
    """Todos los filings por empresa-trimestre con intensidades por 1.000
    PALABRAS y ceros (ver ai_intensity.py). Grupo = mitad más promocional
    ANTES del evento medida en promocionales por 1.000 palabras, sobre todas
    las empresas con filings pre."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from ai_intensity import document_table, aggregate, FILING_FORMS
    docs = document_table(con)
    docs = docs[docs["form"].isin(FILING_FORMS)]
    panel = aggregate(docs, ["ticker", "quarter"])
    mix = (docs.assign(is_proxy=(docs.form == "DEF 14A") * docs.n_words)
           .groupby(["ticker", "quarter"]).agg(p=("is_proxy", "sum"), n=("n_words", "sum")))
    panel = panel.merge((mix.p / mix.n).rename("mix_proxy").reset_index(), on=["ticker", "quarter"])
    pre = docs[docs["fecha"] < pd.Timestamp(GROUP_END)].groupby("ticker").agg(
        promo=("n_promo", "sum"), words=("n_words", "sum"))
    pre = pre[pre["words"] > 0]
    rate = 1000.0 * pre["promo"] / pre["words"]
    high_risk = (rate > rate.median()).astype(float).rename("alto_riesgo")
    panel = panel.merge(high_risk.reset_index(), on="ticker", how="inner")
    panel["event_time"] = (panel["quarter"].astype("period[Q]") - EVENT).apply(lambda x: x.n)
    panel = panel[panel["event_time"].abs() <= WINDOW]
    sides = panel.groupby("ticker")["event_time"].agg(
        pre=lambda s: (s < 0).sum(), post=lambda s: (s >= 0).sum())
    keep = sides[(sides["pre"] >= MIN_QUARTERS_EACH_SIDE)
                 & (sides["post"] >= MIN_QUARTERS_EACH_SIDE)].index
    panel = panel[panel["ticker"].isin(keep)].copy()
    panel["post"] = (panel["event_time"] >= 0).astype(float)
    panel["trimestre"] = panel["quarter"].astype(str)
    return panel


def fe_ols(data: pd.DataFrame, outcome: str, rhs: str):
    """Efectos fijos de empresa por demeaning + OLS con SE cluster por empresa
    (ver shock_analysis.fe_ols)."""
    import patsy
    X = patsy.dmatrix(rhs, data, return_type="dataframe")
    X = X.drop(columns=[c for c in X.columns if c == "Intercept"])
    groups = data["ticker"].to_numpy()
    y = data[outcome] - data.groupby("ticker")[outcome].transform("mean")
    Xd = X - X.groupby(groups).transform("mean")
    Xd = Xd.loc[:, Xd.abs().sum() > 1e-12]
    return sm.OLS(y.to_numpy(dtype=float), Xd).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(groups)[0]})


def did(panel: pd.DataFrame, outcome: str) -> dict:
    model = fe_ols(panel, outcome, f"alto_riesgo:post + C(trimestre) + {CONTROLS}")
    term = [n for n in model.params.index if "alto_riesgo:post" in n][0]
    return {"coef": float(model.params[term]), "se": float(model.bse[term]),
            "p": float(model.pvalues[term]), "n_obs": int(model.nobs),
            "n_firms": int(panel["ticker"].nunique())}


def event_study(panel: pd.DataFrame, outcome: str) -> pd.DataFrame:
    """Un coeficiente por trimestre relativo al evento, normalizado a t−1.

    Las dummies se construyen a mano en vez de usar `C(ev, Treatment(reference=-1))`:
    con un categórico de enteros, patsy ignora la referencia y estima TODOS los
    períodos, lo que deja el modelo sin normalizar — el coeficiente de t−1 salía
    +9,7 p.p. en vez de cero y el test de tendencias previas heredaba ese salto
    de nivel. Con dummies explícitas, el período omitido es el que uno dice."""
    data = panel.copy()
    periods = sorted(data["event_time"].unique())
    terms = []
    for period in periods:
        if period == REFERENCE:
            continue
        name = f"ev_{'m' if period < 0 else 'p'}{abs(period)}"
        data[name] = ((data["event_time"] == period) * data["alto_riesgo"]).astype(float)
        terms.append((period, name))
    model = fe_ols(data, outcome, " + ".join(name for _, name in terms) + f" + C(trimestre) + {CONTROLS}")
    rows = [{"event_time": period, "coef": float(model.params[name]),
             "se": float(model.bse[name]), "p": float(model.pvalues[name]), "term": name}
            for period, name in terms]
    rows.append({"event_time": REFERENCE, "coef": 0.0, "se": 0.0, "p": np.nan,
                 "term": "referencia"})
    table = pd.DataFrame(rows).sort_values("event_time").reset_index(drop=True)
    pre_terms = [name for period, name in terms if period < REFERENCE]
    if pre_terms:
        joint = model.f_test(" = 0, ".join(pre_terms) + " = 0")
        table.attrs["pretrend_p"] = float(np.squeeze(joint.pvalue))
    return table


def plot(table: pd.DataFrame, outcome: str, destination: Path, pretrend_p: float) -> None:
    """El gráfico estándar de event study: el coeficiente estimado por trimestre
    con su intervalo de 95%.

    La primera versión graficaba medias crudas por grupo, que es otra cosa: no
    lleva los efectos fijos ni los controles, y con 64 empresas el ruido tapa
    todo. Lo que hay que mostrar es lo que el modelo estima — la diferencia
    ajustada entre grupos, normalizada a t−1 — porque ahí se lee de una si las
    barras previas cruzan el cero (supuesto en pie) y si las posteriores se
    despegan (hubo cambio)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 4.5))
    x = table["event_time"].to_numpy(dtype=float)
    y = table["coef"].to_numpy(dtype=float)
    error = 1.96 * table["se"].to_numpy(dtype=float)
    pre = x < 0
    axis.errorbar(x[pre], y[pre], yerr=error[pre], fmt="o", color="#4c72b0",
                  capsize=3, label="before the event")
    axis.errorbar(x[~pre], y[~pre], yerr=error[~pre], fmt="s", color="#c44e52",
                  capsize=3, label="after")
    axis.axhline(0, color="grey", linewidth=1)
    axis.axvline(-0.5, color="black", linestyle=":", linewidth=1)
    axis.annotate("SEC, Dec-2023 / Mar-2024", xy=(-0.42, max(y + error) * 0.92), fontsize=9)
    axis.set_xlabel("quarters from the event (t = 0 is 2024Q1; t−1 is the reference)")
    axis.set_ylabel(f"{outcome}: high-risk − low-risk group")
    axis.set_title(f"{outcome} — pre-trend test: p = {pretrend_p:.3f}")
    axis.legend(fontsize=9, loc="lower left")
    figure.tight_layout()
    figure.savefig(destination, dpi=150)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--database", type=Path, default=DB)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    con = duckdb.connect(str(args.database), read_only=True)
    try:
        panel = build(con)
    finally:
        con.close()
    firms = panel.groupby("alto_riesgo")["ticker"].nunique()
    print(f"panel: {len(panel):,} empresa-trimestre | alto riesgo {int(firms.get(1.0, 0))} "
          f"empresas, bajo riesgo {int(firms.get(0.0, 0))} | ventana ±{WINDOW} trimestres\n")

    print(f"{'outcome':20s} {'DiD':>9s} {'SE':>7s} {'p':>7s} {'pre-tend p':>11s}")
    report = {}
    for outcome in OUTCOMES:
        estimate = did(panel, outcome)
        study = event_study(panel, outcome)
        pretrend = study.attrs.get("pretrend_p", float("nan"))
        marker = " <- misma dimensión que el grupo" if outcome == GROUP_OUTCOME else ""
        print(f"{outcome:20s} {estimate['coef']:+9.4f} {estimate['se']:7.4f} "
              f"{estimate['p']:7.3f} {pretrend:11.3f}{marker}")
        report[outcome] = {**estimate, "pretrend_p": pretrend,
                           "event_study": study.drop(columns="term").to_dict("records")}
        plot(study, outcome, args.output_dir / f"shock_did_{outcome}.png", pretrend)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "shock_did_simple.json").write_text(json.dumps(report, indent=2, default=float))
    print(f"\nfiguras -> {args.output_dir}/shock_did_<outcome>.png")
    print(f"-> {args.output_dir}/shock_did_simple.json")


if __name__ == "__main__":
    main()
