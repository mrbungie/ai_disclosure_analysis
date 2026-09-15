"""AI-washing como brecha divulgación-sustancia, no como residuo de un modelo
de promoción esperada ni como ratio a nivel de claim.

    Substance_it = log(1 + n_activities_it) * Grounding_it
    W_it = pctrank(Disclosure_it) - pctrank(Substance_it)

donde `Disclosure_it` es la intensidad de divulgación de IA (`frames_per_1k`,
del pase semántico por frames: cuánto habla la empresa de IA) y `Substance_it`
combina cuánta actividad concreta de IA declara la empresa (`n_activities`,
del pase de extracción de actividades) con qué tan fundamentada está esa
actividad (`Grounding_it`: función nombrada, desplegada/escalada, producto o
proceso nombrado, resultado cuantificado, proveedor externo nombrado — media
con shrinkage empírico-Bayes por componente). Ambas partes se transforman a
rango percentil (no z-score) dentro de la muestra analítica de cada año.

## Por qué percentil y no z-score

Una primera versión estandarizaba ambos lados con z-score. Eso resultó en un
artefacto: `frames_per_1k` tiene una cola derecha mucho más pesada que
`Substance` (máximo a ~14 desviaciones estándar de la media frente a ~4 del
lado de sustancia), así que las empresas con MÁS actividad concreta absoluta
(Microsoft, NVIDIA — cientos de actividades fundamentadas) terminaban en la
cola de "washing" simplemente porque su divulgación era un outlier todavía
más extremo que su (también enorme) sustancia — no porque hablaran sin
respaldo. El rango percentil comprime ambos lados a [0, 1] sin importar la
forma de la distribución subyacente, y elimina ese artefacto: la cola de
washing pasa a estar compuesta por empresas que divulgan relativamente mucho
y no declaran ninguna actividad concreta (percentil de sustancia empatado en
el mínimo), que es la lectura pretendida.

## Por qué esta forma, y no un ratio a nivel de claim

Una versión anterior de este índice medía washing como la fracción de claims
PROMOCIONALES de una empresa-año que no traían un producto/sistema nombrado
ni una métrica cuantificada en la misma frase. Eso sólo era calculable para
empresas-año con al menos 5 claims promocionales — 95 de 2,475 (3.8% del
panel) — porque el ratio no tiene sentido con un denominador pequeño. La
brecha estandarizada de arriba usa las mismas dos extracciones LLM del resto
de la tesis (frames semánticos, actividades) pero como intensidades por
1.000 palabras y por inventario de actividad, no como conteos de claims
individuales: se puede calcular para cualquier empresa-año que mencione IA
en absoluto, sin importar cuánto promocione específicamente.

## Muestra analítica: sólo empresas-año que divulgan IA

El índice se define únicamente para empresas-año con `Disclosure_it > 0`
(al menos un frame de IA ese año). Las empresas-año sin ninguna divulgación
de IA no entran como observaciones de "washing cero": la pregunta "¿cuánto
se desacopla el discurso de IA de esta empresa de su actividad de IA?" no
tiene respuesta con sentido si la empresa no habla de IA en absoluto. Esto
deja alrededor del 64% del panel (frente al 3.8% del diseño anterior).

## Validación

No se reporta como una validación convergente independiente: `Grounding_it`
es, por construcción, la mitad del score. Se reporta como consistencia
mecánica esperada (¿el score se mueve como debería, dado cómo está
construido?), restringida a empresas-año con al menos una actividad
declarada (donde `Grounding_it` no es simplemente el prior de shrinkage).
Las dos validaciones que sí importan son:

  1. Persistencia año a año dentro de la misma empresa (¿mide un rasgo
     empresa-año con algo de estabilidad, o es ruido puro?).
  2. Coherencia económica: el signo esperado frente a intensidad de I+D
     (más I+D real, menos brecha) y la ausencia de relación mecánica con
     tamaño, beta, volatilidad idiosincrática o valoración (el score no
     debería ser simplemente un proxy de esas variables).

Las validaciones en sí (consistencia mecánica, persistencia, coherencia
económica, split-half) están en
`scripts/analytics/washing/validate_washing_score.py`, no acá: este script
sólo construye el índice.

Salida: covariates/firm_year/washing_score (n_activities y grounding_index
del último ejercicio con 10-K, substance, pct_disclosure, pct_substance, w,
washing, callada; nulos fuera de la muestra analítica) y
models/washing_grounding_shrinkage/model.pkl (los priors por componente y
año). Como en el índice firm_quarter, todo lo poblacional se estima dentro de
la sección cruzada de cada año calendario, con información de ese año
solamente: los priors de grounding, los rangos percentiles y las colas del 5%.
`_fit_shrink_prior`/`_apply_shrink` son también el shrinkage de los builders
firm_quarter y call.
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "common"))
import layers as L  # noqa: E402

BUILDER = "scripts/gold/firm_year/build_washing_score.py"
MODEL_PATH = REPO_ROOT / "models" / "washing_grounding_shrinkage" / "model.pkl"
YEARS = (2021, 2022, 2023, 2024, 2025, 2026)  # mismo panel que incremental_signal.py
GROUNDING_COMPONENTS = ["named_function", "deployed_or_scaled", "named_product_or_process",
                        "quantified_outcome", "third_party_named_provider"]
TAIL_Q = 0.05  # cola descriptiva: 5% superior/inferior de la distribución de W, no un test de hipótesis
SCORE_COLUMNS = ["n_activities", "grounding_index", "substance", "pct_disclosure", "pct_substance", "w", "washing",
                 "callada"]


def _fit_shrink_prior(numerator: pd.Series, denominator: pd.Series) -> tuple[float, float]:
    """Fits the Beta prior (method-of-moments) a rate's empirical-Bayes
    shrinkage uses -- the reusable, persistable half of `_shrink()`."""
    raw = np.where(denominator > 0, numerator / denominator, np.nan)
    s = pd.Series(raw, index=numerator.index)
    mean, var = float(s.mean(skipna=True)), float(s.var(skipna=True, ddof=1))
    strength = max(mean * (1 - mean) / var - 1, 1e-6) if 0 < mean < 1 and var > 0 else 1.0
    return mean * strength, (1 - mean) * strength


def _apply_shrink(numerator: pd.Series, denominator: pd.Series, alpha: float, beta_: float) -> pd.Series:
    """Applies an ALREADY-FITTED prior (see `_fit_shrink_prior`) to a rate.
    Denominator 0 returns the prior itself, not NaN."""
    raw = np.where(denominator > 0, numerator / denominator, np.nan)
    s = pd.Series(raw, index=numerator.index)
    return (s.fillna(0) * denominator + alpha) / (denominator + alpha + beta_)


def _shrink(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Fits AND applies the prior in one call -- see `_fit_shrink_prior`/
    `_apply_shrink` for the split version."""
    alpha, beta_ = _fit_shrink_prior(numerator, denominator)
    return _apply_shrink(numerator, denominator, alpha, beta_)


def load_activities_firm_year(persist_model: bool = False) -> pd.DataFrame:
    """`n_activities` y `Grounding_it` (media con shrinkage por componente de
    los cinco marcadores de fundamentación) por (empresa, año), con los
    priors ajustados sobre las empresas-año del mismo año.

    `persist_model=True` also saves the fitted per-year, per-component priors
    to MODEL_PATH -- the "model" for this index is the shrinkage prior, not
    a row-appliable regressor: percentile rank (see build()) is inherently
    population-level and gets recomputed over each year's analytical
    sample every run, not something you can apply to one new row."""
    a = L.read_gold("firm_year", ("covariates", "activities", ["n_activities"] + GROUNDING_COMPONENTS))
    priors, shrunk = {}, []
    for year, g in a.groupby("year", sort=True):
        priors[int(year)] = {c: _fit_shrink_prior(g[c], g["n_activities"]) for c in GROUNDING_COMPONENTS}
        shrunk.append(pd.concat([_apply_shrink(g[c], g["n_activities"], *priors[int(year)][c])
                                 for c in GROUNDING_COMPONENTS], axis=1).mean(axis=1))
    a["grounding_index"] = pd.concat(shrunk)
    if persist_model:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "grounding_priors": {year: {c: {"alpha": p[0], "beta": p[1]} for c, p in by_component.items()}
                                 for year, by_component in priors.items()},
            "grounding_components": GROUNDING_COMPONENTS,
        }, MODEL_PATH)
        print(f"-> {MODEL_PATH}")
    return a[["ticker", "year", "n_activities", "grounding_index"]]


def load_disclosure_firm_year() -> pd.DataFrame:
    """`Disclosure_it` (`frames_per_1k`) y `n_promo` (sólo para reportar,
    consumido por `channel_gap_analysis.py`) por (empresa, año)."""
    m = L.read_gold("firm_year", ("covariates", "disclosure_volume", ["frames_per_1k", "any_ai", "n_promo", "n_frames"]))
    m = m[m["year"].isin(YEARS)]
    return m[["ticker", "year", "frames_per_1k", "any_ai", "n_promo", "n_frames"]]


def load_10k_years() -> pd.DataFrame:
    """(empresa, año calendario de presentación) con al menos un 10-K filed.

    `Substance_it` se mide casi enteramente sobre 10-K/10-Q/DEF 14A/8-K (ver
    `activity_profiles.py`, canal 'filing'); ninguno de esos reemplaza al
    10-K como fuente principal de actividad fundamentada. Un año calendario
    sin 10-K filed (ejercicio fiscal aún en curso, p.ej. 2026 para una firma
    que cierra en septiembre) no es comparable a un año con 10-K: cualquier
    firma queda con `Substance_it = 0` por construcción hasta que su 10-K
    aparece, sin relación con cuánto vaya a divulgar. `build()` usa esta
    bandera para propagar (as-of) `n_activities`/`grounding_index` del
    último 10-K disponible en vez de tratar el año como sustancia cero."""
    return (L.scan("silver.filing_manifest")
            .filter((pl.col("country_code") == "us") & (pl.col("form_type") == "10-K")
                    & pl.col("filing_date").is_not_null())
            .select("ticker", pl.col("filing_date").dt.year().cast(pl.Int32).alias("year"))
            .unique(maintain_order=True)
            .collect().to_pandas())


def build(keys: tuple[str, ...] = ("ticker", "year"), persist_model: bool = False) -> pd.DataFrame:
    """Construye la muestra analítica (Disclosure_it > 0, ejercicio con 10-K
    filed) y el índice W_it por empresa-año."""
    if keys != ("ticker", "year"):
        raise ValueError("el índice W se define por empresa-año")
    disc = load_disclosure_firm_year()
    acts = load_activities_firm_year(persist_model=persist_model)
    d = disc.merge(acts, on=["ticker", "year"], how="left")
    has_10k = load_10k_years()
    has_10k["has_10k"] = True
    d = d.merge(has_10k, on=["ticker", "year"], how="left")
    d["has_10k"] = d["has_10k"].fillna(False)
    # Sin imputación: n_activities/grounding_index nulos quedan nulos (el as-of de
    # abajo sólo arrastra valores de ejercicios con 10-K propio).
    # as-of, no por exclusión: un ejercicio sin 10-K propio (fiscal year en curso)
    # no tiene Substance_it=0 porque no hay actividad, sino porque todavía no llegó
    # el 10-K que la documentaría. Se usa el último 10-K disponible hasta ese punto
    # (n_activities y grounding_index del ejercicio cerrado más reciente de la misma
    # empresa) en vez de descartar el firm-año o tratarlo como sustancia cero.
    d = d.sort_values(["ticker", "year"])
    for col in ("n_activities", "grounding_index"):
        asof = d[col].where(d["has_10k"])
        d[col] = asof.groupby(d["ticker"]).ffill()
    d = d.dropna(subset=["n_activities", "grounding_index"])
    d = d.drop(columns="has_10k")
    d = d[d["any_ai"] > 0].reset_index(drop=True)

    d["substance"] = np.log1p(d["n_activities"]) * d["grounding_index"]
    by_year = d.groupby("year")
    d["pct_disclosure"] = by_year["frames_per_1k"].rank(pct=True)
    d["pct_substance"] = by_year["substance"].rank(pct=True)
    d["w"] = d["pct_disclosure"] - d["pct_substance"]
    d["washing"] = d["w"] >= d.groupby("year")["w"].transform(lambda w: w.quantile(1 - TAIL_Q))
    d["callada"] = d["w"] <= d.groupby("year")["w"].transform(lambda w: w.quantile(TAIL_Q))
    return d


def main() -> None:
    """Fits and persists the grounding shrinkage priors (MODEL_PATH) and writes
    the firm-year index on the firm-year spine."""
    panel = build(("ticker", "year"), persist_model=True)
    print(f"n={len(panel):,} empresas-año con divulgación de IA, {panel['ticker'].nunique()} empresas "
          f"| mean(W)={panel['w'].mean():.4f}")
    out = L.read_gold("firm_year").merge(panel[["ticker", "year"] + SCORE_COLUMNS], on=["ticker", "year"],
                                         how="left", validate="one_to_one")
    L.write_gold("covariates", "firm_year", "washing_score", out, builder=BUILDER, inputs=[MODEL_PATH],
                 extra={"sample": "firm-years with AI frames and a 10-K as of the year",
                        "population_fit": "grounding priors, percentile ranks and tails within each calendar year"})


if __name__ == "__main__":
    main()
