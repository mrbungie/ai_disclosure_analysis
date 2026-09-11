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
rango percentil (no z-score) dentro de la muestra analítica, no sobre el
panel completo.

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

Uso:
    uv run python scripts/analytics/washing_score.py
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[2]
DB = REPO_ROOT / "duckdb" / "thesis.duckdb"
OUT_DIR = REPO_ROOT / "data" / "processed" / "clusters"
YEARS = (2021, 2022, 2023, 2024, 2025, 2026)  # mismo panel que incremental_signal.py
GROUNDING_COMPONENTS = ["named_function", "deployed_or_scaled", "named_product_or_process",
                        "quantified_outcome", "third_party_named_provider"]
TAIL_Q = 0.05  # cola descriptiva: 5% superior/inferior de la distribución de W, no un test de hipótesis


def _shrink(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Shrinkage empírico-Bayes (prior Beta ajustado por método de momentos)
    de una tasa numerador/denominador. Para denominador 0, devuelve el prior
    (la media del corpus) — no NaN — así que un componente de Grounding no
    definido no requiere imputación aparte."""
    raw = np.where(denominator > 0, numerator / denominator, np.nan)
    s = pd.Series(raw, index=numerator.index)
    mean, var = float(s.mean(skipna=True)), float(s.var(skipna=True, ddof=1))
    strength = max(mean * (1 - mean) / var - 1, 1e-6) if 0 < mean < 1 and var > 0 else 1.0
    alpha, beta_ = mean * strength, (1 - mean) * strength
    shrunk = (s.fillna(0) * denominator + alpha) / (denominator + alpha + beta_)
    return shrunk


def load_activities_firm_year() -> pd.DataFrame:
    """`n_activities` y `Grounding_it` (media con shrinkage por componente de
    los cinco marcadores de fundamentación) por (empresa, año)."""
    a = pd.read_parquet(OUT_DIR / "firm_year_activities.parquet")[
        ["ticker", "year", "n_activities"] + GROUNDING_COMPONENTS]
    shrunk = pd.concat([_shrink(a[c], a["n_activities"]) for c in GROUNDING_COMPONENTS], axis=1)
    a["grounding_index"] = shrunk.mean(axis=1)
    return a[["ticker", "year", "n_activities", "grounding_index"]]


def load_activities_pooled() -> pd.DataFrame:
    """Igual, pooled por empresa 2021-2025 (suma de actividades, shrinkage
    sobre los componentes sumados)."""
    a = pd.read_parquet(OUT_DIR / "firm_year_activities.parquet")
    a = a[a["year"].isin(YEARS)]
    g = a.groupby("ticker")[["n_activities"] + GROUNDING_COMPONENTS].sum().reset_index()
    shrunk = pd.concat([_shrink(g[c], g["n_activities"]) for c in GROUNDING_COMPONENTS], axis=1)
    g["grounding_index"] = shrunk.mean(axis=1)
    return g[["ticker", "n_activities", "grounding_index"]]


def load_disclosure_firm_year() -> pd.DataFrame:
    """`Disclosure_it` (`frames_per_1k`) y `n_promo` (sólo para reportar,
    consumido por `channel_gap_analysis.py`) por (empresa, año)."""
    m = pd.read_parquet(OUT_DIR / "firm_year_master_v2.parquet")
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
    con = duckdb.connect(str(DB), read_only=True)
    try:
        return con.execute("""
            SELECT DISTINCT ticker, EXTRACT(year FROM filing_date)::INT AS year
            FROM filing_manifest
            WHERE country_code = 'us' AND form_type = '10-K' AND filing_date IS NOT NULL
        """).df()
    finally:
        con.close()


def build(keys: tuple[str, ...]) -> pd.DataFrame:
    """Construye la muestra analítica (Disclosure_it > 0, ejercicio con 10-K
    filed) y el índice W_it, a nivel panel empresa-año
    (`keys=("ticker","year")`) o pooled por empresa (`keys=("ticker",)`)."""
    if keys == ("ticker", "year"):
        disc = load_disclosure_firm_year()
        acts = load_activities_firm_year()
        d = disc.merge(acts, on=["ticker", "year"], how="left")
        has_10k = load_10k_years()
        has_10k["has_10k"] = True
        d = d.merge(has_10k, on=["ticker", "year"], how="left")
        d["has_10k"] = d["has_10k"].fillna(False)
    else:
        m = pd.read_parquet(OUT_DIR / "firm_year_master_v2.parquet")
        m = m[m["year"].isin(YEARS)]
        disc = m.groupby("ticker").agg(
            frames_per_1k=("frames_per_1k", "mean"), n_promo=("n_promo", "sum"),
            n_frames=("n_frames", "sum"), any_ai=("any_ai", "max")).reset_index()
        acts = load_activities_pooled()
        d = disc.merge(acts, on="ticker", how="left")
    d["n_activities"] = d["n_activities"].fillna(0.0)
    d["grounding_index"] = d["grounding_index"].fillna(d["grounding_index"].median())
    if "has_10k" in d.columns:
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
    d["pct_disclosure"] = d["frames_per_1k"].rank(pct=True)
    d["pct_substance"] = d["substance"].rank(pct=True)
    d["w"] = d["pct_disclosure"] - d["pct_substance"]
    q_hi, q_lo = d["w"].quantile(1 - TAIL_Q), d["w"].quantile(TAIL_Q)
    d["washing"] = d["w"] >= q_hi
    d["callada"] = d["w"] <= q_lo
    return d


def validations(panel: pd.DataFrame) -> dict:
    out: dict = {}

    grounded = panel[panel["n_activities"] > 0]
    if grounded["grounding_index"].nunique() > 1:
        rho, p = stats.spearmanr(grounded["w"], grounded["grounding_index"])
        out["mechanical_consistency_vs_grounding"] = {"n": int(len(grounded)), "spearman": float(rho), "p": float(p)}

    p2 = panel[["ticker", "year", "w"]].copy()
    nxt = p2.assign(year=p2["year"] - 1).rename(columns={"w": "w_next"})
    pairs = p2.merge(nxt, on=["ticker", "year"])
    if len(pairs) > 10:
        rho, p = stats.spearmanr(pairs["w"], pairs["w_next"])
        out["persistence"] = {"n_pairs": int(len(pairs)), "spearman": float(rho), "p": float(p)}

    m = pd.read_parquet(OUT_DIR / "firm_year_master_v2.parquet")[
        ["ticker", "year", "rd_intensity", "market_cap", "beta", "idio_vol_252d", "ps_ratio"]]
    econ = panel.merge(m, on=["ticker", "year"], how="left")
    econ["log_market_cap"] = np.log(econ["market_cap"])
    coherence = {}
    for var in ("rd_intensity", "log_market_cap", "beta", "idio_vol_252d", "ps_ratio"):
        sub = econ.dropna(subset=["w", var])
        if sub[var].nunique() > 1:
            rho, p = stats.spearmanr(sub["w"], sub[var])
            coherence[var] = {"n": int(len(sub)), "spearman": float(rho), "p": float(p)}
    out["economic_coherence"] = coherence
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--database", type=Path, default=DB)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    print("=" * 74)
    print("PANEL EMPRESA-AÑO (primario) — usado como regresor en el Capítulo 5")
    print("=" * 74)
    panel = build(("ticker", "year"))
    print(f"n={len(panel):,} empresas-año con divulgación de IA (de "
          f"{len(load_disclosure_firm_year()):,} en el panel 2021-2025), "
          f"{panel['ticker'].nunique()} empresas | mean(W)={panel['w'].mean():.4f}")
    print(f"cola de washing (top {TAIL_Q:.0%} de W): {int(panel['washing'].sum())} | "
          f"cola callada (bottom {TAIL_Q:.0%}): {int(panel['callada'].sum())}")
    show = ["ticker", "year", "frames_per_1k", "n_activities", "grounding_index", "w"]
    if panel["washing"].any():
        print("\n--- cola de washing (por w) ---")
        print(panel[panel.washing].nlargest(15, "w")[show].round(3).to_string(index=False))

    val = validations(panel)
    if "mechanical_consistency_vs_grounding" in val:
        v = val["mechanical_consistency_vs_grounding"]
        print(f"\nconsistencia mecánica vs. grounding (n_activities>0, n={v['n']}): "
              f"Spearman={v['spearman']:+.3f} (p={v['p']:.3g})")
    if "persistence" in val:
        v = val["persistence"]
        print(f"persistencia año-a-año (n={v['n_pairs']} pares): Spearman={v['spearman']:+.3f} (p={v['p']:.3g})")
    print("coherencia económica:")
    for var, v in val["economic_coherence"].items():
        print(f"  {var:16s} n={v['n']:5d} Spearman={v['spearman']:+.3f} (p={v['p']:.3g})")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    panel_out = args.output_dir / "firm_year_washing_score.parquet"
    panel.to_parquet(panel_out, index=False)

    print("\n" + "=" * 74)
    print("POOLED POR EMPRESA (2021-2025) — usado en el Capítulo 4 y el Apéndice C")
    print("=" * 74)
    pooled = build(("ticker",))
    pooled["n_promo"] = pooled["n_promo"].fillna(0)
    print(f"n={len(pooled):,} empresas con divulgación de IA | mean(W)={pooled['w'].mean():.4f}")
    print(f"cola de washing: {int(pooled['washing'].sum())} | cola callada: {int(pooled['callada'].sum())}")
    show_p = ["ticker", "frames_per_1k", "n_activities", "grounding_index", "w"]
    if pooled["washing"].any():
        print("\n--- cola de washing (pooled, por w) ---")
        print(pooled[pooled.washing].nlargest(15, "w")[show_p].round(3).to_string(index=False))

    out = args.output_dir / "firm_washing_score.parquet"
    pooled.to_parquet(out, index=False)

    manifest = args.output_dir / "firm_year_washing_score_manifest.json"
    manifest.write_text(json.dumps(
        {"n_panel": int(len(panel)), "n_panel_firms": int(panel["ticker"].nunique()),
         "n_total_panel": int(len(load_disclosure_firm_year())),
         "n_pooled": int(len(pooled)), "tail_q": TAIL_Q,
         "validations": val,
         "built_at": datetime.now(timezone.utc).isoformat()},
        indent=2, default=float))
    print(f"\n-> {panel_out} ({len(panel):,} filas)\n-> {out} ({len(pooled):,} filas)\n-> {manifest}")


if __name__ == "__main__":
    main()
