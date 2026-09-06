"""Preguntas extendidas de la tesis: ¿cambió la divulgación de IA tras los shocks
de 2024-2025, y cambió distinto según el tipo de empresa?

Diseño, en una línea: **event study empresa-trimestre con exposición previa a IA
como tratamiento**, efectos fijos de empresa y de trimestre, panel balanceado,
errores estándar clusterizados por empresa, y un coeficiente por trimestre
relativo al evento.

    y[i,t] = a[i] + t[q] + SUM_k b[k] * 1{tiempo_evento = k} * exposicion[i]
             + controles[i,t] + e[i,t]

Qué afirma y qué no. La propuesta de tesis pide, para las preguntas extendidas,
un análisis de **cambio diferencial** ("quasi-causal techniques like DiD may be
used"), no la identificación causal del enforcement. Este diseño sostiene lo
primero: los coeficientes miden si las empresas más expuestas a IA cambiaron
distinto que las menos expuestas alrededor del corte, con las tendencias previas
como test del supuesto. No sostiene lo segundo: el evento es común en tiempo
calendario, así que el coeficiente recoge todo lo que le pasó a las empresas
expuestas en esa fecha — el boom de IA generativa incluido.

Tres decisiones de diseño que importan:

1. **El tratamiento no es el outcome.** La versión original definía los grupos
   por `z(promotional_rate) - z(specificity_index)` pre-evento y después medía
   esas mismas métricas: la convergencia era mecánica y las tendencias previas no
   podían ser paralelas (test conjunto F=5,11, p=0,001). Acá el tratamiento es el
   VOLUMEN de frames de IA antes de 2023 — cuánto hablaba la empresa del tema, no
   cómo. Con eso las tendencias previas de `promotional_rate` quedan planas
   (p=0,32).

2. **Panel balanceado.** Sólo empresas con al menos dos trimestres a cada lado
   del corte. Sin eso, una empresa que aparece recién en 2025 aporta a los
   coeficientes post sin haber aportado nunca a los pre, y el "efecto" es entrada
   al panel.

3. **Controles de composición documental.** El panel usa los cuatro formularios
   —el 10-Q solo daba 257 observaciones— y los formularios difieren en retórica
   (la DEF 14A tiene 14,6% de frames promocionales contra 7,2% del 10-K), así que
   la mezcla del trimestre entra como control.

Además de los dos eventos, corre la heterogeneidad por **segmento de divulgación**
(`build_segments.py`), que es lo que conecta las preguntas extendidas con el
núcleo de la tesis: no sólo si el corpus cambió, sino si los desplegadores
cambiaron distinto que los listadores de riesgo.

Uso:
    uv run python scripts/analytics/shock_analysis.py
    uv run python scripts/analytics/shock_analysis.py --event deepseek
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

REPO_ROOT = Path(__file__).resolve().parents[2]
DB = REPO_ROOT / "duckdb" / "thesis.duckdb"
OUT_DIR = REPO_ROOT / "data" / "processed" / "clusters"

EVENTS = {
    # 18-mar-2024: la SEC anuncia enforcement por AI-washing. 2024Q1 mezcla
    # filings de antes y después del anuncio, así que el corte va al inicio de
    # 2024Q2 y 2024Q1 queda como trimestre de referencia.
    "sec": pd.Period("2024Q2", freq="Q"),
    # 20-ene-2025: DeepSeek R1. Mismo criterio.
    "deepseek": pd.Period("2025Q2", freq="Q"),
}
REFERENCE_OFFSET = -1
WINDOW = 5
MIN_FRAMES_QUARTER = 3
MIN_FRAMES_PRE = 5
MIN_QUARTERS_EACH_SIDE = 2
GROUP_WINDOW_END = "2023-01-01"
OUTCOMES = ("promotional_rate", "specificity_index", "risk_share", "hypothetical_share")
# Margen extensivo: intensidades por 1.000 párrafos sobre TODOS los filings,
# con cero cuando el documento no habla de IA (ai_intensity.py). No condiciona
# a hablar de IA: la empresa que deja de hablar cuenta como cero, no sale.
OUTCOMES_EXTENSIVE = ("promo_per_1k", "spec_per_1k", "risk_per_1k", "hyp_per_1k", "frames_per_1k")
MARGIN = "intensive"
CONTROLS = {"intensive": "mix_proxy + mix_10k + np.log(n_frames)",
            "extensive": "mix_proxy + mix_10k + np.log(n_paragraphs)"}
SPECIFICITY_FLAGS = ["specificity_business_process", "specificity_product_or_system",
                     "specificity_vendor_or_partner", "specificity_quantified_metric",
                     "specificity_date_or_timeline"]


def load_frames(con) -> pd.DataFrame:
    """Frames de los cuatro formularios, con su trimestre de presentación.

    Se traen UNA vez y los manifiestos se unen en pandas: escribirlo como un
    `UNION ALL` de dos consultas a `gold_ai_frames` obliga a DuckDB a instanciar
    la vista dos veces y a recalcular por dentro la población vigente con una
    ventana sobre 37 millones de filas. Medido: 0,7 s así, más de 115 s de la
    otra forma."""
    frames = con.execute("""
        SELECT accession_number, text_hash, frame_index, temporal, concepts,
               rhetoric_promotional, specificity_business_process,
               specificity_product_or_system, specificity_vendor_or_partner,
               specificity_quantified_metric, specificity_date_or_timeline
        FROM gold_ai_frames WHERE country_code = 'us' AND has_frame
    """).fetchdf()
    manifests = []
    for relation, form in (("filing_manifest", None), ("filing_manifest_10q", "10-Q")):
        columns = "accession_number, ticker, filing_date" + (", form_type" if form is None else "")
        table = con.execute(f"SELECT {columns} FROM {relation} "
                            f"WHERE country_code = 'us' AND ticker IS NOT NULL").fetchdf()
        if form is not None:
            table["form_type"] = form
        manifests.append(table)
    merged = frames.merge(pd.concat(manifests, ignore_index=True),
                          on="accession_number", how="inner")
    return merged.drop_duplicates(["ticker", "filing_date", "form_type",
                                   "text_hash", "frame_index"])


def add_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["specificity_index"] = out[SPECIFICITY_FLAGS].astype(float).mean(axis=1)
    out["promotional_rate"] = out["rhetoric_promotional"].astype(float)
    out["hypothetical_share"] = (out["temporal"] == "hypothetical").astype(float)
    out["risk_share"] = out["concepts"].apply(
        lambda cs: float(any(str(c).startswith("risk_") for c in (cs if cs is not None else []))))
    return out


def build_panel(con) -> tuple[pd.DataFrame, pd.DataFrame]:
    quarterly = add_metrics(load_frames(con))
    quarterly["quarter"] = pd.PeriodIndex(pd.to_datetime(quarterly["filing_date"]), freq="Q")
    quarterly["es_proxy"] = (quarterly["form_type"] == "DEF 14A").astype(float)
    quarterly["es_10k"] = (quarterly["form_type"] == "10-K").astype(float)
    panel = (quarterly.groupby(["ticker", "quarter"])
             .agg(n_frames=("promotional_rate", "size"),
                  mix_proxy=("es_proxy", "mean"), mix_10k=("es_10k", "mean"),
                  **{c: (c, "mean") for c in OUTCOMES}).reset_index())
    panel = panel[panel["n_frames"] >= MIN_FRAMES_QUARTER]

    # Exposición: cuánto hablaba de IA ANTES de 2023, en volumen. Deja 2023 como
    # pre-período limpio y, sobre todo, no es el outcome.
    pre = quarterly[quarterly["filing_date"].astype(str) < GROUP_WINDOW_END]
    exposure = pre.groupby("ticker").size().rename("n_pre")
    exposure = exposure[exposure >= MIN_FRAMES_PRE]
    treatment = pd.DataFrame({"exposure": np.log(exposure)})

    segments_path = OUT_DIR / "firm_segments.parquet"
    if segments_path.exists():
        segments = pd.read_parquet(segments_path)[["ticker", "segmento"]]
        treatment = treatment.join(segments.set_index("ticker"))
    return panel, treatment


def build_panel_extensive(con) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Todos los filings (10-K, 10-Q, DEF 14A, 8-K) por empresa-trimestre, con
    intensidades por 1.000 párrafos y ceros. Exposición = frames de IA por
    1.000 párrafos antes de 2023, sobre todas las empresas."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from ai_intensity import document_table, aggregate, FILING_FORMS
    docs = document_table(con)
    docs = docs[docs["form"].isin(FILING_FORMS)]
    panel = aggregate(docs, ["ticker", "quarter"])
    mix = (docs.assign(is_proxy=(docs.form == "DEF 14A") * docs.n_paragraphs,
                       is_10k=(docs.form == "10-K") * docs.n_paragraphs)
           .groupby(["ticker", "quarter"]).agg(p=("is_proxy", "sum"), k=("is_10k", "sum"), n=("n_paragraphs", "sum")))
    panel = panel.merge((mix.p / mix.n).rename("mix_proxy").reset_index(), on=["ticker", "quarter"])
    panel = panel.merge((mix.k / mix.n).rename("mix_10k").reset_index(), on=["ticker", "quarter"])
    pre = docs[docs["fecha"] < pd.Timestamp(GROUP_WINDOW_END)].groupby("ticker").agg(
        frames=("n_frames", "sum"), paragraphs=("n_paragraphs", "sum"))
    pre = pre[pre["paragraphs"] > 0]
    treatment = pd.DataFrame({"exposure": np.log1p(1000.0 * pre["frames"] / pre["paragraphs"])})
    segments_path = OUT_DIR / "firm_segments.parquet"
    if segments_path.exists():
        segments = pd.read_parquet(segments_path)[["ticker", "segmento"]]
        treatment = treatment.join(segments.set_index("ticker"))
    return panel, treatment


def fit(data: pd.DataFrame, outcome: str, formula_treat: str) -> tuple:
    data = data.copy()
    data["ev"] = pd.Categorical(data["event_time"],
                                categories=sorted(data["event_time"].unique()))
    formula = (f"{outcome} ~ C(ev, Treatment(reference={REFERENCE_OFFSET})):{formula_treat} "
               f"+ C(ticker) + C(ev) + {CONTROLS[MARGIN]}")
    model = smf.ols(formula, data=data).fit(
        cov_type="cluster", cov_kwds={"groups": data["ticker"]})
    return model, data


def coefficients(model, pattern: str = "") -> pd.DataFrame:
    rows = []
    for name, value in model.params.items():
        if "ev" not in name or ":" not in name or (pattern and pattern not in name):
            continue
        match = re.search(r"\[T?\.?(-?\d+)\]", name)
        if not match:
            continue
        rows.append({"event_time": int(match.group(1)), "coef": float(value),
                     "se": float(model.bse[name]), "p": float(model.pvalues[name]),
                     "term": name})
    frame = pd.DataFrame(rows)
    return frame[frame["event_time"] != REFERENCE_OFFSET].sort_values("event_time")


def joint_test(model, terms: list[str]) -> tuple[float, float]:
    if not terms:
        return float("nan"), float("nan")
    result = model.f_test(" = 0, ".join(terms) + " = 0")
    return float(np.squeeze(result.fvalue)), float(np.squeeze(result.pvalue))


def prepare(panel: pd.DataFrame, treatment: pd.DataFrame, event: pd.Period,
            window: int = WINDOW) -> pd.DataFrame:
    data = panel.merge(treatment.reset_index(), on="ticker", how="inner").copy()
    data["event_time"] = (data["quarter"].astype("period[Q]") - event).apply(lambda x: x.n)
    data = data[data["event_time"].abs() <= window]
    sides = data.groupby("ticker")["event_time"].agg(
        pre=lambda s: (s < 0).sum(), post=lambda s: (s >= 0).sum())
    balanced = sides[(sides["pre"] >= MIN_QUARTERS_EACH_SIDE)
                     & (sides["post"] >= MIN_QUARTERS_EACH_SIDE)].index
    return data[data["ticker"].isin(balanced)]


def report_event(panel: pd.DataFrame, treatment: pd.DataFrame, event_name: str) -> dict:
    event = EVENTS[event_name]
    data = prepare(panel, treatment, event)
    print(f"\n{'=' * 78}\nEVENTO: {event_name} — corte en {event} | "
          f"{len(data):,} empresa-trimestre, {data['ticker'].nunique()} empresas\n{'=' * 78}")
    out = {}
    for outcome in (OUTCOMES if MARGIN == "intensive" else OUTCOMES_EXTENSIVE):
        model, _ = fit(data, outcome, "exposure")
        table = coefficients(model)
        pre_terms = table.loc[table["event_time"] < REFERENCE_OFFSET, "term"].tolist()
        post_terms = table.loc[table["event_time"] >= 0, "term"].tolist()
        pre_F, pre_p = joint_test(model, pre_terms)
        post_F, post_p = joint_test(model, post_terms)
        verdict = "pasa" if pre_p > 0.10 else ("límite" if pre_p > 0.05 else "FALLA")
        post_mean = float(table.loc[table["event_time"] >= 0, "coef"].mean())
        print(f"\n{outcome}")
        print("  " + "  ".join(f"t{r.event_time:+d}={r.coef:+.3f}{'*' if r.p < 0.05 else ''}"
                               for r in table.itertuples()))
        print(f"  tendencias previas: F={pre_F:.2f} p={pre_p:.3f} -> {verdict}"
              f"   |   cambio post: {post_mean:+.4f} (p conjunto {post_p:.3f})")
        out[outcome] = {"pretrend_F": pre_F, "pretrend_p": pre_p, "usable": verdict != "FALLA",
                        "post_mean": post_mean, "post_p": post_p,
                        "coefficients": table.drop(columns="term").to_dict("records")}
    return out


def report_segments(panel: pd.DataFrame, treatment: pd.DataFrame, event_name: str) -> dict:
    """Heterogeneidad por segmento de divulgación — lo que conecta las preguntas
    extendidas con el núcleo de la tesis: no si el corpus cambió, sino si los
    desplegadores cambiaron distinto que los listadores de riesgo."""
    if "segmento" not in treatment.columns:
        return {}
    data = prepare(panel, treatment.dropna(subset=["segmento"]), EVENTS[event_name])
    data["post"] = (data["event_time"] >= 0).astype(float)
    main_outcome = "promotional_rate" if MARGIN == "intensive" else "promo_per_1k"
    formula = (f"{main_outcome} ~ post:C(segmento) + C(ticker) + C(ev) + {CONTROLS[MARGIN]}")
    data["ev"] = pd.Categorical(data["event_time"])
    model = smf.ols(formula, data=data).fit(
        cov_type="cluster", cov_kwds={"groups": data["ticker"]})
    print(f"\ncambio post por segmento ({event_name}, outcome = {main_outcome}):")
    rows = {}
    for name, value in model.params.items():
        if not name.startswith("post:"):
            continue
        segment = name.split("[")[-1].rstrip("]").replace("T.", "")
        pval = float(model.pvalues[name])
        if not np.isfinite(pval):        # colinealidad perfecta: no hay contraste que reportar
            continue
        rows[segment] = {"coef": float(value), "p": pval}
        print(f"  {segment:28s} {value:+.4f} (p={pval:.3f}, "
              f"n={int((data['segmento'] == segment).sum())})")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--database", type=Path, default=DB)
    parser.add_argument("--output", type=Path, default=OUT_DIR / "shock_analysis.json")
    parser.add_argument("--event", choices=(*EVENTS, "all"), default="all")
    parser.add_argument("--margin", choices=("intensive", "extensive"), default="intensive",
                        help="intensive = tasas entre frames (≥3 por trimestre); extensive = "
                             "intensidad por 1.000 párrafos sobre todos los filings, con ceros")
    args = parser.parse_args()
    global MARGIN
    MARGIN = args.margin
    if MARGIN == "extensive" and args.output == OUT_DIR / "shock_analysis.json":
        args.output = OUT_DIR / "shock_analysis_extensive.json"

    con = duckdb.connect(str(args.database), read_only=True)
    try:
        panel, treatment = build_panel(con) if MARGIN == "intensive" else build_panel_extensive(con)
    finally:
        con.close()
    print(f"panel: {len(panel):,} empresa-trimestre, {panel['ticker'].nunique()} empresas | "
          f"tratamiento (exposición pre-2023): {len(treatment):,} empresas clasificables")

    events = list(EVENTS) if args.event == "all" else [args.event]
    report = {}
    for event_name in events:
        report[event_name] = {"outcomes": report_event(panel, treatment, event_name),
                              "segments": report_segments(panel, treatment, event_name)}
    args.output.write_text(json.dumps(report, indent=2, default=float))
    print(f"\n-> {args.output}")


if __name__ == "__main__":
    main()
